"""COLMAP integration: locate the binary, build pipelines, run them with QProcess.

Option names changed between COLMAP releases (e.g. SiftExtraction.* ->
FeatureExtraction.*), so we read `colmap <command> -h` once and only pass
options the installed build understands.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
from .colmap_io import qvec_to_rot
from .engine import child_env
from PySide6.QtCore import QObject, QProcess, QTimer, Signal, QSettings, Qt, Slot
from .progress import (MappingMeter, MatchingMeter, command_progress, extraction_progress)

_HELP_CACHE: dict = {}


APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bundled_candidates():
    """COLMAP shipped next to the app: <app>/tools/colmap or <app>/../tools/colmap."""
    for root in (APP_ROOT, os.path.dirname(APP_ROOT)):
        d = os.path.join(root, "tools", "colmap")
        for name in ("COLMAP.bat", os.path.join("bin", "colmap.exe"), os.path.join("bin", "colmap"), "colmap"):
            yield os.path.join(d, name)


def _resolve(path: str) -> str:
    """Prefer the executable; keep DLL/Qt paths confined to its child process."""
    path = os.path.abspath(path)
    base = os.path.dirname(path)
    if path.lower().endswith(".bat"):
        exe = os.path.join(base, "bin", "colmap.exe")
        if os.path.isfile(exe):
            path, base = exe, os.path.join(base, "bin")
    return path


def _find_colmap(check_compatibility=True) -> str | None:
    """Project > Preferences path, then the COLMAP shipped next to the app
    (tools/colmap - the tested version, also in the portable package), then PATH."""
    custom = QSettings().value("colmap/path", "", str)
    if custom and os.path.isfile(custom):
        return _resolve(custom)
    for c in _bundled_candidates():
        if os.path.isfile(c) and (os.name == "nt" or os.access(c, os.X_OK)):
            from .engine import select_bundled
            selected = select_bundled(c) if check_compatibility else c
            if selected:
                return _resolve(selected)
    for name in ("colmap", "COLMAP", "colmap.bat", "COLMAP.bat", "colmap.exe"):
        p = shutil.which(name)
        if p:
            return _resolve(p)
    cands = []
    if sys.platform.startswith("win"):
        cands += [r"C:\COLMAP\COLMAP.bat", r"C:\Program Files\COLMAP\COLMAP.bat",
                  r"C:\Program Files\COLMAP\bin\colmap.exe"]
    for c in cands:
        if os.path.isfile(c):
            return _resolve(c)
    return None


def find_colmap() -> str | None:
    return _find_colmap(True)


def locate_colmap() -> str | None:
    """Path lookup only: safe for startup and interactive preferences."""
    return _find_colmap(False)


# ------------------------------------------------------------------ GLOMAP
GLOMAP = "@glomap"          # first element of a step's args -> run glomap.exe, not colmap


def find_glomap() -> str | None:
    """GLOMAP (global SfM, github.com/colmap/glomap) bundled in <app>/../tools/glomap."""
    custom = QSettings().value("glomap/path", "", str)
    if custom and os.path.isfile(custom):
        return os.path.abspath(custom)
    for root in (APP_ROOT, os.path.dirname(APP_ROOT)):
        for name in (os.path.join("bin", "glomap.exe"), os.path.join("bin", "glomap"), "glomap.exe"):
            c = os.path.join(root, "tools", "glomap", name)
            if os.path.isfile(c):
                return os.path.abspath(c)
    p = shutil.which("glomap")
    return os.path.abspath(p) if p else None


PYTHON = "@python"   # same sentinel as dense.PYTHON


def python_exe():
    """Interpreter for photomesh.tasks child processes (console python.exe,
    not pythonw.exe, so stdout can be piped)."""
    exe = sys.executable
    d, b = os.path.split(exe)
    if b.lower() == "pythonw.exe" and os.path.exists(os.path.join(d, "python.exe")):
        exe = os.path.join(d, "python.exe")
    return exe


def task_env():
    """Environment for child python processes: our package on PYTHONPATH."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    # PipelineRunner decodes task output as UTF-8, including project filenames.
    env['PYTHONIOENCODING'] = 'utf-8'
    return env


def resolve_cmd(exe, args):
    """(program, args) for a step: GLOMAP steps run glomap, PYTHON steps our
    photomesh.tasks module, the rest colmap."""
    if getattr(args, "executable", None):
        return args.executable, list(args)
    if args and args[0] == PYTHON:
        return python_exe(), ["-m", "photomesh.tasks", *[str(a) for a in args[1:]]]
    if args and args[0] == GLOMAP:
        g = find_glomap()
        if not g:
            raise RuntimeError("GLOMAP not found (expected in tools\\glomap\\bin)")
        return g, list(args[1:])
    return exe, list(args)


class Try(list):
    """Step args whose failure is not fatal: on a non-zero exit the runner
    inserts fallback() steps (or continues if fallback is None). Before
    fallback() runs, the runner sets .error (e.g. "exit code -1073740791"),
    .exit_code (int or None) and .output_tail (last console lines)."""
    def __init__(self, args, fallback=None, meter=None):
        super().__init__(args)
        self.fallback = fallback
        self.meter = meter
        self.error = ""
        self.exit_code = None
        self.output_tail = []


class Cmd(list):
    """Plain step args with a progress meter (see progress.py)."""
    def __init__(self, args, meter=None):
        super().__init__(args)
        self.meter = meter


def nvidia_driver_present() -> bool:
    """True if an NVIDIA display driver (the CUDA driver API) is installed.
    Without it the CUDA build of COLMAP can only run its CPU code paths."""
    if os.environ.get("PHOTOMESH_FORCE_NO_GPU"):
        return False
    if os.name == "nt":
        sysdir = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
        return os.path.exists(os.path.join(sysdir, "nvcuda.dll"))
    if os.path.exists("/proc/driver/nvidia/version"):
        return True
    import ctypes.util
    return ctypes.util.find_library("cuda") is not None


def gpu_status(exe=None) -> str:
    if not nvidia_driver_present():
        return "CPU only (no NVIDIA driver)"
    if exe and not has_cuda(exe):
        return "CPU only (COLMAP without CUDA)"
    return "CUDA"


def has_cuda(exe: str) -> bool:
    """CUDA usable: an NVIDIA driver is installed and COLMAP is a CUDA build
    (COLMAP prints '... without CUDA' in its banner for CPU-only builds)."""
    from .engine import bundle_manifest, gpu_device
    if bundle_manifest(exe):
        return bool(gpu_device(exe))
    if not nvidia_driver_present():
        return False
    key = (exe, "__banner__")
    if key not in _HELP_CACHE:
        try:
            out = subprocess.run([exe, "help"], capture_output=True, text=True, timeout=30,
                                 env=child_env(exe),
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            _HELP_CACHE[key] = out.stdout + out.stderr
        except Exception:
            _HELP_CACHE[key] = ""
    banner = _HELP_CACHE[key].strip()
    return bool(banner) and "without cuda" not in banner.lower()


def colmap_options(exe: str, command: str) -> set:
    key = (exe, command)
    if key not in _HELP_CACHE:
        try:
            out = subprocess.run([exe, command, "-h"], capture_output=True,
                                 text=True, timeout=30,
                                 env=child_env(exe),
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            text = (out.stdout + out.stderr) if getattr(out, "returncode", 0) == 0 else ""
        except Exception:
            text = ""
        _HELP_CACHE[key] = set(re.findall(r"--([A-Za-z0-9_.]+)", text))
    return _HELP_CACHE[key]


def opt(exe, command, candidates, value):
    """Return ['--name', value] for the first candidate option supported."""
    supported = colmap_options(exe, command)
    for c in candidates:
        if c in supported:
            return [f"--{c}", str(value)]
    if not supported:  # help unavailable: fall back to the legacy name
        return [f"--{candidates[-1]}", str(value)]
    return []


def alignment_photo_infos(cameras, infos):
    """Fill missing dimensions from source image headers without decoding pixels."""
    from PySide6.QtGui import QImageReader
    resolved = []
    for camera, info in zip(cameras, infos, strict=True):
        if not image_long_edge(info):
            size = QImageReader(camera.path).size()
            if size.isValid():
                info = dict(info, width=size.width(), height=size.height())
        resolved.append(info)
    return resolved


def image_long_edge(info):
    width, height = info.get("width") or 0, info.get("height") or 0
    return max(width, height) if width > 0 and height > 0 else 0


# --------------------------------------------------------------- pipelines
ACCURACY = ["Highest", "High", "Medium", "Low", "Lowest"]
ACCURACY_DOWNSCALE = {"Highest": 1, "High": 1, "Medium": 2, "Low": 4, "Lowest": 8}
QUALITY = ["Ultra high", "High", "Medium", "Low", "Lowest"]
DEPTH_FILTERING = ["Disabled", "Mild", "Moderate", "Aggressive"]


def alignment_extraction_sizes(longest_side, settings):
    """Return the selected and effective maximum sizes for the longest edge."""
    from .engine import extraction_size
    requested = max(256, (longest_side or 3200) // ACCURACY_DOWNSCALE[settings["accuracy"]])
    return requested, extraction_size(requested, settings)


def align_steps(exe, dataset, wd, s, names=None, infos=None):
    """Steps for Align Photos. s = settings dict from AlignPhotosDialog.

    s["mode"]: "Fast" (fastpipe: preselected pairs + coarse-to-fine, default),
               "Exhaustive" or "Sequential" (plain COLMAP matchers).
    """
    cameras = [c for c in dataset.cameras if c.enabled]
    if infos is None:
        infos = [c.info() for c in cameras]
    infos = alignment_photo_infos(cameras, infos)
    if s.get("mode", "Fast") == "Fast" and names is not None:
        from .fastpipe import fast_align_steps
        return fast_align_steps(exe, wd, names, infos, s)
    s = dict(s, sequential=s.get("mode") == "Sequential" or s.get("sequential", False))
    img_dir = image_dir(wd)
    db = os.path.join(wd, "database.db")
    sparse = os.path.join(wd, "sparse")
    maxdim = max(map(image_long_edge, infos), default=0)
    _, max_size = alignment_extraction_sizes(maxdim, s)
    fe = ["feature_extractor", "--database_path", db, "--image_path", img_dir,
          "--image_list_path", image_list(wd),
          "--ImageReader.single_camera", "0"]   # identical cameras merged below
    from .calibration import camera_model_arguments
    fe += camera_model_arguments(s)
    fe += opt(exe, "feature_extractor", ["FeatureExtraction.max_image_size",
                                          "SiftExtraction.max_image_size"], max_size)
    if s["key_point_limit"] > 0:
        fe += opt(exe, "feature_extractor", ["SiftExtraction.max_num_features"],
                  s["key_point_limit"])
    from .engine import matching_arguments
    fe += matching_arguments(exe, "feature_extractor", s, extraction=True)
    matcher = "sequential_matcher" if s.get("sequential") else "exhaustive_matcher"
    m = [matcher, "--database_path", db]
    m += matching_arguments(exe, matcher, s)
    from .fastpipe import mapper_steps, camera_groups_step
    n_img = len(cameras)
    if os.path.exists(db):
        os.remove(db)
    os.makedirs(sparse, exist_ok=True)
    os.makedirs(os.path.join(wd, "sparse_txt"), exist_ok=True)
    return [
        ("Detecting points", fe, 30),
        *([camera_groups_step(db, groups=s.get("calibration_groups"))] if s.get("single_camera") else []),
        ("Matching points", m, 30),
        *mapper_steps(exe, db, img_dir, sparse, s, n_img, image_list(wd)),
        ("Exporting tie points", ["model_converter", "--input_path",
         os.path.join(sparse, "0"), "--output_path",
         os.path.join(wd, "tie_points.ply"), "--output_type", "PLY"], 2),
        ("Exporting cameras", ["model_converter", "--input_path",
         os.path.join(sparse, "0"), "--output_path", os.path.join(wd, "sparse_txt"),
         "--output_type", "TXT"], 3),
    ]


def point_cloud_path(wd):
    """Fused dense cloud; must live at dense/fused.ply (+ .vis) for delaunay_mesher."""
    return os.path.join(wd, "dense", "fused.ply")


def image_dir(wd):
    """Folder COLMAP reads images from (see prepare_images)."""
    f = os.path.join(wd, "image_root.txt")
    if os.path.exists(f):
        with open(f, encoding="utf-8") as fh:
            r = fh.read().strip()
        if r:
            return r if os.path.isabs(r) else os.path.normpath(os.path.join(wd, r))
    return os.path.join(wd, "images")


def image_list(wd):
    return os.path.join(wd, "image_list.txt")


def prepare_images(dataset, wd):
    """Tell COLMAP which photos to use. Returns {photo path: COLMAP image name}.

    If all enabled photos live in one folder (the normal case) COLMAP reads
    them in place - nothing is copied or written next to the photos, and an
    image list restricts processing to the enabled ones. Otherwise photos are
    exposed in <wd>/images via hard links (same volume), symlinks, or copies.
    """
    os.makedirs(wd, exist_ok=True)
    cams = [c for c in dataset.cameras if c.enabled]
    dirs = {os.path.dirname(os.path.abspath(c.path)) for c in cams}
    # A re-alignment must not renumber aliases when photos are disabled,
    # removed, reordered or added: marker placements are keyed by those names.
    stored = [c.colmap_name for c in dataset.cameras if getattr(c, "colmap_name", "")]
    if len({name.lower() for name in stored}) != len(stored):
        raise OSError("Duplicate stored photo aliases; repair photo identities before alignment")
    used = {name.lower() for name in stored}
    mapping = {}
    for camera in cams:
        name = getattr(camera, "colmap_name", "")
        if not name:
            base, ext = os.path.splitext(os.path.basename(camera.path))
            name, suffix = base + ext, 1
            while name.lower() in used:
                name = f"{base}_{suffix}{ext}"; suffix += 1
            used.add(name.lower())
        if name != os.path.basename(name) or name in (".", ".."):
            raise OSError("Unsupported stored photo alias; repair photo identities before alignment")
        mapping[camera.path] = name
    if len(dirs) == 1 and all(mapping[c.path] == os.path.basename(c.path) for c in cams):
        root = dirs.pop()
    else:
        root = os.path.join(wd, "images")
        os.makedirs(root, exist_ok=True)
        manifest_path = os.path.join(wd, ".image_stage.json")
        try:
            with open(manifest_path, encoding="utf-8") as f:
                previous = json.load(f)
            if not isinstance(previous, dict):
                previous = {}
        except (OSError, ValueError):
            previous = {}

        def stamp(path):
            st = os.stat(path)
            return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]

        copied = {}
        for c in cams:
            name = mapping[c.path]
            dst = os.path.join(root, name)
            source = os.path.normcase(os.path.abspath(c.path))
            source_stamp = stamp(c.path)
            if os.path.lexists(dst):
                try:
                    if os.path.samefile(c.path, dst):
                        continue  # a link follows changes to the source automatically
                    cached = previous.get(name)
                    if cached == [source, source_stamp, stamp(dst)]:
                        copied[name] = cached
                        continue  # unchanged copy; avoid decoding/copying it again
                except OSError:
                    pass
                os.remove(dst)
            try:
                os.link(c.path, dst)          # hard link: no admin rights needed on NTFS
            except OSError:
                try:
                    os.symlink(c.path, dst)
                except OSError:
                    shutil.copy2(c.path, dst)
                    copied[name] = [source, source_stamp, stamp(dst)]
        tmp_manifest = manifest_path + ".part"
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(copied, f)
        os.replace(tmp_manifest, manifest_path)
    with open(os.path.join(wd, "image_root.txt"), "w", encoding="utf-8") as f:
        f.write("images" if root == os.path.join(wd, "images") else root)   # relative: portable data folder
    with open(image_list(wd), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(mapping.values())) + "\n")
    return mapping


def read_images_txt(path):
    """Return {image_name: (centre(3,), R(3x3))} from COLMAP images.txt."""
    res = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = [l for l in f if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].rstrip('\r\n').split(maxsplit=9)
        if len(parts) < 10:
            continue
        q = list(map(float, parts[1:5]))
        t = np.array(list(map(float, parts[5:8])))
        R = qvec_to_rot(q)
        res[parts[9]] = (-R.T @ t, R)
    return res


def count_points3d(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return sum(1 for l in f if l.strip() and not l.startswith("#"))
    except Exception:
        return 0


# ----------------------------------------------------------------- runner
class PipelineRunner(QObject):
    stepStarted = Signal(str)
    stageProgress = Signal(object)
    log = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, exe, steps, parent=None, *, photos=0):
        super().__init__(parent)
        self.exe = exe
        self.steps = steps
        self.photos = photos
        self.idx = -1
        self.proc = None
        self.cancelled = False
        self._ended = False
        self._tail = []
        self._output_buffer = b""
        self._callback_job = None
        self.failure = None          # details of the failed step, for error reports
        self._cmd = None
        self._command_cleanup = None
        self._meter = None
        self._parse_progress = None
        self._last_progress = None
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(1000)
        self._progress_timer.timeout.connect(self._poll_meter)
        self.finished.connect(self._mark_end)

    def _mark_end(self, *_):
        self._ended = True
        self._progress_timer.stop()

    def _fail(self, msg, exit_code=None, crashed=False, exc=None):
        import traceback as _tb
        try:
            self._cleanup_command(False)
        except Exception as error:
            msg += f'; restoring photo identities failed: {error}'
        label = self.steps[self.idx][0] if 0 <= self.idx < len(self.steps) else ""
        args = self.steps[self.idx][1] if 0 <= self.idx < len(self.steps) else None
        self.failure = {"step": label, "message": msg, "exit_code": exit_code, "crashed": crashed,
                        "command": self._cmd or (list(args) if isinstance(args, (list, tuple)) else None),
                        "traceback": "".join(_tb.format_exception(type(exc), exc, exc.__traceback__)) if exc else "",
                        "tail": list(self._tail)}
        self.finished.emit(False, msg)

    def _cleanup_command(self, success):
        callback, self._command_cleanup = self._command_cleanup, None
        if callback is not None:
            callback(success)

    def start(self):
        self._next()

    def cancel(self):
        self.cancelled = True
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.proc.kill()

    def _next(self):
        if self.cancelled:
            self.finished.emit(False, "Cancelled by user")
            return
        self.idx += 1
        if self.idx >= len(self.steps):
            self.finished.emit(True, "Finished")
            return
        label, args, _ = self.steps[self.idx]
        self._cmd = None
        if label == "log":
            self.log.emit(f"[fastpipe] {args}")
            return self._next()
        self._last_progress = None
        self.stepStarted.emit(label)
        if callable(args):
            from .bgjob import BackgroundJob
            job = BackgroundJob(lambda report, cancelled: args() or [], name=label, parent=self)
            self._callback_job = job
            job.done.connect(self._callable_done, Qt.QueuedConnection)
            job.start()
            return
        try:
            if hasattr(args, 'prepare'):
                self._command_cleanup = args.cleanup
                for message in args.prepare():
                    self.log.emit(message)
            prog, pargs = resolve_cmd(self.exe, args)
            if hasattr(args, 'on_start'):
                args.on_start()
        except Exception as e:
            if isinstance(args, Try):
                return self._fallback(args, str(e))
            self._fail(f"'{label}' failed: {e}", exc=e); return
        self._meter = getattr(args, "meter", None)
        command = args[0] if args else ""
        if self._meter is not None:
            self._parse_progress = self._meter.feed
            if hasattr(self._meter, "poll"):
                self._emit_progress(self._meter.current())
                self._progress_timer.start()
        elif command in (GLOMAP, "mapper", "global_mapper", "hierarchical_mapper",
                         "bundle_adjuster", "point_triangulator"):
            self._parse_progress = MappingMeter(self.photos, incremental=command == "mapper").feed
        elif command == "feature_extractor":
            self._parse_progress = extraction_progress
        elif command.endswith("_matcher") or command == "matches_importer":
            self._parse_progress = MatchingMeter().feed
        else:
            self._parse_progress = lambda line: command_progress(line, label)
        self._cmd = [prog] + list(pargs)
        self.log.emit(f"> {os.path.basename(prog)} {' '.join(pargs)}")
        self._tail = []
        self._output_buffer = b""
        self.proc = QProcess(self)
        if args and args[0] == PYTHON:
            from PySide6.QtCore import QProcessEnvironment
            env = QProcessEnvironment()
            for k, v in task_env().items():
                env.insert(k, v)
            if len(args) > 1 and args[1] == "georef":
                # The matrix-free solve makes many short BLAS reductions.
                env.insert("OPENBLAS_NUM_THREADS", "1")
            self.proc.setProcessEnvironment(env)
        else:
            from PySide6.QtCore import QProcessEnvironment
            from .engine import child_env
            env = QProcessEnvironment()
            for k, v in child_env(prog).items():
                env.insert(k, v)
            self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        self.proc.errorOccurred.connect(self._error)
        self.proc.start(prog, pargs)

    @Slot(object, object)
    def _callable_done(self, extra, error):
        job = self._callback_job
        if job is None:
            return
        job.wait()
        self._callback_job = None
        exception = job.exception
        job.deleteLater()
        if self.cancelled:
            self.finished.emit(False, "Cancelled by user")
        elif error:
            label = self.steps[self.idx][0]
            self._fail(f"'{label}' failed: {error}", exc=exception)
        else:
            self.steps[self.idx + 1:self.idx + 1] = extra or []
            self._next()

    def _emit_progress(self, update):
        if update is not None and update != self._last_progress and not self.cancelled:
            self._last_progress = update
            self.stageProgress.emit(update)

    def _poll_meter(self):
        if self._meter is not None and hasattr(self._meter, "poll") and not self.cancelled:
            self._emit_progress(self._meter.poll())

    def _read(self, final=False):
        data = bytes(self.proc.readAllStandardOutput()).replace(b"\r", b"\n")
        lines = (self._output_buffer + data).split(b"\n")
        self._output_buffer = b"" if final else lines.pop()
        for raw in lines:
            line = raw.decode("utf-8", "replace").rstrip()
            if not line:
                continue
            self.log.emit(line)
            self._tail = (self._tail + [line])[-40:]
            self._emit_progress(self._parse_progress(line))
        if final:
            self._poll_meter()

    def _fallback(self, args, why, code=None):
        label = self.steps[self.idx][0]
        args.error, args.exit_code, args.output_tail = why, code, list(self._tail)
        try:
            extra = args.fallback() if args.fallback else []
        except Exception as e:      # the fallback decided to give up: clear message
            self._fail(f"'{label}' failed: {e}", exit_code=code, exc=e)
            return
        self.log.emit(f"[fastpipe] '{label}' did not succeed ({why}) - "
                      + ("using fallback" if extra else "continuing"))
        self.steps[self.idx + 1:self.idx + 1] = extra
        self._next()

    def _error(self, err):
        if err == QProcess.FailedToStart and not self._ended:
            self._progress_timer.stop()
            args = self.steps[self.idx][1]
            if isinstance(args, Try):    # finished() is not emitted after FailedToStart
                return self._fallback(args, "failed to start")
            self._fail(f"Failed to start: {self.exe}")

    def _done(self, code, status):
        if self._ended:
            return
        self._progress_timer.stop()
        self._read(final=True)
        try:
            self._cleanup_command(code == 0 and status == QProcess.NormalExit and not self.cancelled)
        except Exception as error:
            self._fail(f'Pair matching could not finalize: {error}', exc=error)
            return
        if self.cancelled:
            self.finished.emit(False, "Cancelled by user")
            return
        if code != 0 or status != QProcess.NormalExit:
            label, args, _ = self.steps[self.idx]
            if isinstance(args, Try):
                return self._fallback(args, f"exit code {code}", code)
            from .dense import exit_code_text
            from .report import error_reason
            why = error_reason(self._tail) if status == QProcess.NormalExit else ""
            if why:                  # a Python task that stopped with a clear message (issue #3)
                self._fail(f"'{label}' failed: {why}", exit_code=code)
            else:
                self._fail(f"'{label}' failed (exit code {exit_code_text(code)}). See Console for details.",
                           exit_code=code, crashed=status != QProcess.NormalExit)
            return
        try:
            action=self.steps[self.idx][1]
            if hasattr(action,'on_success'):
                action.on_success()
        except Exception as error:
            self._fail(f'Could not save alignment progress: {error}',exc=error)
            return
        self._next()
