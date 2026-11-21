"""Status bar CPU / GPU load, sampled every ~1.5 s in a daemon thread (never
on the GUI thread) and delivered through a queued Qt signal.

* CPU: psutil.cpu_percent(None) (non-blocking, since the previous call) when
  psutil is installed, else the system CPU times (Windows GetSystemTimes via
  ctypes, Linux /proc/stat) - no extra package needed.
* GPU: NVIDIA NVML loaded from the display driver. On Windows, a failed
  sample retries periodically and can use the driver's nvidia-smi as a
  local fallback. Both report utilisation and memory of GPU 0.
"""
from __future__ import annotations

import ctypes
import csv
import os
import sys
import threading
import time

from PySide6.QtCore import QObject, Signal

INTERVAL = 1.5
GPU_RETRY_INTERVAL = 8.0


# ---------------------------------------------------------------- CPU
class CpuSampler:
    def __init__(self):
        self.ps = None
        try:
            import psutil
            psutil.cpu_percent(None)                  # prime: the next call measures since now
            self.ps = psutil
        except Exception:  # noqa: BLE001
            self.ps = None
        self.prev = self._times()

    @staticmethod
    def _times():
        """(idle, total) cumulative CPU times, or None."""
        try:
            if sys.platform == "win32":
                ft = [ctypes.c_ulonglong() for _ in range(3)]
                if not ctypes.windll.kernel32.GetSystemTimes(*(ctypes.byref(x) for x in ft)):
                    return None
                idle, kernel, user = (x.value for x in ft)       # kernel includes idle
                return idle, kernel + user
            with open("/proc/stat", encoding="ascii") as f:
                v = [int(x) for x in f.readline().split()[1:]]
            return v[3] + (v[4] if len(v) > 4 else 0), sum(v[:8])
        except Exception:  # noqa: BLE001
            return None

    def percent(self):
        if self.ps is not None:
            try:
                return float(self.ps.cpu_percent(None))
            except Exception:  # noqa: BLE001
                self.ps = None
        cur = self._times()
        prev, self.prev = self.prev, cur
        if cur is None or prev is None:
            return None
        di, dt = cur[0] - prev[0], cur[1] - prev[1]
        return max(0.0, min(100.0, 100.0 * (1 - di / dt))) if dt > 0 else None


# ---------------------------------------------------------------- GPU (NVML)
class _Util(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Mem(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def _nvml_candidates():
    if sys.platform == "win32":
        sysroot = os.environ.get("SystemRoot", r"C:\Windows")
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        return [os.path.join(sysroot, "System32", "nvml.dll"),
                os.path.join(pf, "NVIDIA Corporation", "NVSMI", "nvml.dll"), "nvml.dll"]
    return ["libnvidia-ml.so.1", "libnvidia-ml.so"]


def _windows_gpu_sample():
    """Use only the NVIDIA tool installed in a standard Windows driver location."""
    if sys.platform != "win32":
        return None
    system = os.environ.get("SystemRoot", r"C:\Windows")
    programs = os.environ.get("ProgramFiles", r"C:\Program Files")
    candidates = (os.path.join(system, "System32", "nvidia-smi.exe"),
                  os.path.join(programs, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"))
    executable = next((path for path in candidates if os.path.isfile(path)), None)
    if not executable:
        return None
    from .report import _run
    output = _run([executable, "-i", "0", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                   "--format=csv,noheader,nounits"], timeout=4)
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) != 4:
            continue
        name, usage, used, total = (value.strip() for value in row)
        try:
            load = float(usage)
        except ValueError:
            return name, None
        if 0 <= load <= 100:
            return name, (load, int(used) if used.isdecimal() else None,
                          int(total) if total.isdecimal() else None)
    return None


class GpuSampler:
    """Read GPU 0; recover a failed NVML handle without restarting the app."""

    def __init__(self, loader=None, fallback=None, retry_interval=GPU_RETRY_INTERVAL):
        self.lib = None
        self.handle = None
        self.name = ""
        self.source = ""
        self._loader = loader or self._load
        self._fallback = fallback if fallback is not None else (_windows_gpu_sample if loader is None else None)
        self._retry_interval = retry_interval
        self._retry_at = 0.0
        self._cached_sample = None
        self._connect()

    def _connect(self):
        try:
            lib = self._loader()
            if lib is None or lib.nvmlInit_v2() != 0:
                return
            self.lib = lib
            h = ctypes.c_void_p()
            if self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) != 0:
                self.close()
                return
            self.handle = h
            buf = ctypes.create_string_buffer(96)
            if self.lib.nvmlDeviceGetName(h, buf, 96) == 0:
                self.name = buf.value.decode(errors="replace")
        except Exception:  # noqa: BLE001
            self.close()

    @staticmethod
    def _load():
        for c in _nvml_candidates():
            if os.path.isabs(c) and not os.path.exists(c):
                continue
            try:
                return ctypes.CDLL(c)
            except OSError:
                continue
        return None

    @property
    def available(self):
        return self.lib is not None and self.handle is not None

    def sample(self):
        """Return a current NVML sample or a throttled driver-tool sample."""
        result = self._sample_nvml()
        if result is None:
            if time.monotonic() < self._retry_at:
                return self._cached_sample
            self.close()
            self._connect()
            result = self._sample_nvml()
        if result is not None:
            self.source = "NVML"
            self._retry_at = 0.0
            self._cached_sample = None
            return result
        self.close()
        reading = self._fallback() if self._fallback else None
        self._retry_at = time.monotonic() + self._retry_interval
        self._cached_sample = None
        self.source = ""
        if reading is not None:
            self.name, self._cached_sample = reading
            self.source = "NVIDIA driver tool"
        return self._cached_sample

    def _sample_nvml(self):
        """(gpu %, used MB, total MB) or None."""
        if not self.available:
            return None
        try:
            u = _Util(); m = _Mem()
            if self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(u)) != 0:
                return None
            mem = (None, None)
            if self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(m)) == 0:
                mem = (m.used // (1024 * 1024), m.total // (1024 * 1024))
            return float(u.gpu), mem[0], mem[1]
        except Exception:  # noqa: BLE001
            return None

    def close(self):
        try:
            if self.lib is not None:
                self.lib.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass
        self.lib = None
        self.handle = None


# ---------------------------------------------------------------- monitor
class ResourceMonitor(QObject):
    """Emit CPU load and GPU load, memory in MB, name, and sampling source."""
    sampled = Signal(object)

    def __init__(self, interval=INTERVAL, cpu=None, gpu=None, parent=None):
        super().__init__(parent)
        self.interval = interval
        self._cpu, self._gpu = cpu, gpu
        self._stop = threading.Event()
        self._thread = None
        self.last = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="resource-monitor", daemon=True)
            self._thread.start()

    def stop(self, timeout=2.0):
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout)
        self._thread = None

    def _loop(self):
        cpu = self._cpu or CpuSampler()
        gpu = self._gpu if self._gpu is not None else GpuSampler()   # NVML init off the GUI thread
        try:
            while not self._stop.wait(self.interval):
                g = gpu.sample()
                self.last = {"cpu": cpu.percent(), "gpu": g[0] if g else None,
                             "gpu_mem": (g[1], g[2]) if g else None, "gpu_name": gpu.name,
                             "gpu_source": getattr(gpu, "source", "")}
                if self._stop.is_set():
                    return
                try:
                    self.sampled.emit(self.last)
                except RuntimeError:                # receiver deleted at exit
                    return
        finally:
            if self._gpu is None:
                gpu.close()


def format_cpu(v):
    return f"CPU {v:3.0f}%" if v is not None else "CPU  --"


def format_gpu(v, mem=None):
    t = f"GPU {v:3.0f}%"
    if mem and mem[0] is not None and mem[1]:
        t += f"  {mem[0] / 1024:.1f}/{mem[1] / 1024:.0f} GB"
    return t
