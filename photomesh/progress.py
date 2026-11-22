"""Measured progress from native log events and validated intermediate files."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from statistics import median
import time


@dataclass(frozen=True)
class StageProgress:
    label: str
    done: int = 0
    total: int = 0
    unit: str = ""
    estimate: bool = True
    detail: str = ""

    @property
    def fraction(self):
        # Finishing the counted work doesn't finish the phase or publish the result.
        if self.estimate and 0 <= self.done < self.total:
            return self.done / self.total
        return None

    @property
    def count_text(self):
        if self.total:
            return f"{self.done:,}/{self.total:,} {self.unit}".strip()
        if self.done:
            return f"{self.done:,} {self.unit}".strip()
        return ""


def fmt_eta(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return ""
    s = int(round(seconds))
    if s < 60:
        return f"{s} s"
    m = s // 60
    if m < 60:
        return f"{m} min"
    return f"{m // 60} h {m % 60:02d} min"


_PHOTO = re.compile(r"Processed file\s+\[\s*(\d+)\s*/\s*(\d+)\s*\]")
_BLOCK = re.compile(r"(?:Processing block|Matching (?:block|image))\s+\[(\d+)/(\d+)(?:,\s*(\d+)/(\d+))?\]")
_MATCHED = re.compile(r"^(?:.*\bfeature_matching\.cc:\d+\]\s*)?\s*in \d+(?:\.\d+)?s\s*$")
_REGISTERING = re.compile(r"Registering image #\d+\s+\((?:(?:num_reg_frames|num_reg_images)=)?(\d+)\)")
_VIEW = re.compile(r"Processing view\s+\d+\s*/\s*\d+\s+for (.+?)(?:\s+===)?$")
_MAP_WRITE = re.compile(r"Writing (photometric|geometric) output for (.+)$")
_FUSE = re.compile(r"Fusing image \[(\d+)/(\d+)\]")
_FUSED = re.compile(r"\bin \d+(?:\.\d+)?s \(\d+ points\)\s*$")
_COUNT = re.compile(r"\[(\d+)/(\d+)\]")


def extraction_progress(line):
    match = _PHOTO.search(line)
    if match:
        done, total = map(int, match.groups())
        if 0 < done <= total:
            return StageProgress("Extracting photo features", done, total, "photos processed")
    return None


def command_progress(line, label):
    if line.startswith("Progress stage: "):
        return StageProgress(line.removeprefix("Progress stage: "))
    match = _COUNT.search(line)
    if match:
        done, total = map(int, match.groups())
        if 0 <= done <= total and total > 0:
            return StageProgress(label, done, total, "completed")
    return None


class MatchingMeter:
    """Keep starts and completion separate in the modern and legacy logs."""
    def __init__(self):
        self.pending = False
        self.done = 0
        self.total = 0
        self.unit = "blocks completed"

    def feed(self, line):
        match = _BLOCK.search(line)
        if match:
            _, first_total, col, second_total = match.groups()
            # A two-index block can be the start of a triangular traversal.
            self.total = int(first_total) if col is None or first_total == second_total == "1" else 0
            self.unit = "photos processed" if "Matching image" in line else "blocks completed"
            self.pending = True
        elif self.pending and _MATCHED.search(line):
            self.pending = False
            self.done += 1
        else:
            return None
        return StageProgress("Matching photos", self.done, self.total, self.unit)


class MappingMeter:
    """An incremental mapper can discard a cluster and then restart its count."""
    def __init__(self, photos=0, incremental=True):
        self.photos = photos
        self.incremental = incremental
        self.largest = 0

    def feed(self, line):
        match = _REGISTERING.search(line) if self.incremental else None
        if match:
            self.largest = max(self.largest, int(match.group(1)))
            label = "Registering camera positions"
        else:
            message = line.rsplit("] ", 1)[-1].strip(" =-").lower()
            if "global bundle adjustment" in message:
                label = "Global bundle adjustment"
            elif message.startswith("finding good initial image pair"):
                label = "Finding an initial photo pair"
            elif not self.incremental and "bundle adjustment" in message:
                label = "Global bundle adjustment"
            elif not self.incremental and "rotation averaging" in message:
                label = "Estimating camera rotations"
            elif not self.incremental and "global positioning" in message:
                label = "Estimating camera positions"
            else:
                return None
        total = self.photos if self.largest and self.largest <= self.photos else 0
        return StageProgress(label, self.largest, total, "photos in largest reported model", estimate=False)


class PatchMatchMeter:
    """Count only newly computed depth + normal pairs that pass file checks."""
    def __init__(self, photometric, geometric, valid_maps):
        self.targets = {"photometric": set(photometric), "geometric": set(geometric)}
        self.written = {kind: set() for kind in self.targets}
        self.pending = set()
        self.photometric_started = set()
        self.kind = "photometric" if self.targets["photometric"] else "geometric"
        self.valid_maps = valid_maps

    def current(self):
        return StageProgress(f"{self.kind.capitalize()} depth maps", len(self.written[self.kind]),
                             len(self.targets[self.kind]), "maps written")

    def feed(self, line):
        match = _MAP_WRITE.search(line)
        if match:
            kind, name = match.groups()
            if name not in self.targets[kind] or name in self.written[kind]:
                return None
            self.pending.add((kind, name))
            if kind != self.kind and kind == "geometric":
                self.kind = kind
                return self.current()
        match = _VIEW.search(line)
        if match and self.kind == "photometric":
            name = match.group(1)
            if name in self.targets["geometric"] and self.targets["photometric"] <= self.photometric_started:
                self.kind = "geometric"
                return self.current()
            if name in self.targets["photometric"]:
                self.photometric_started.add(name)
        return None

    def poll(self):
        completed = {(kind, name) for kind, name in self.pending if self.valid_maps(name, kind)}
        self.pending.difference_update(completed)
        for kind, name in completed:
            self.written[kind].add(name)
        if completed:
            return self.current()
        return None


class FusionMeter:
    def __init__(self, attempt=1):
        self.attempt = f" (attempt {attempt})" if attempt > 1 else ""
        self.pending = 0
        self.fused = set()
        self.total = 0

    def feed(self, line):
        if "Reading workspace" in line:
            return StageProgress(f"Loading depth maps for fusion{self.attempt}")
        if "Writing output" in line:
            return StageProgress(f"Writing fused point cloud{self.attempt}")
        match = _FUSE.search(line)
        if match:
            self.pending, self.total = map(int, match.groups())
        complete = self.pending and _FUSED.search(line)
        if complete:
            self.fused.add(self.pending)
            self.pending = 0
        if match or complete:
            return StageProgress(f"Fusing depth maps{self.attempt}", len(self.fused), self.total, "views fused")
        return None


class WorkProgress:
    """Observed work only. Unknown totals stay indeterminate; ETA needs a measured rate."""
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.reset()

    def reset(self):
        self.start = self.clock()
        self.done, self.total = 0, 0
        self.samples = []

    def update(self, done, total):
        if not math.isfinite(float(done)) or not math.isfinite(float(total)):
            return
        done, total = max(0, float(done)), max(0, float(total))
        if total != self.total or done < self.done:
            self.reset()
        now = self.clock()
        done = min(done, total) if total else done
        if done > self.done:
            self.samples.append((now, done))
            self.samples = self.samples[-30:]
        self.done, self.total = min(done, total) if total else done, total

    def remaining(self):
        """Recent observed throughput only; suppress short, stale or unstable rates."""
        if not self.total or len(self.samples) < 3:
            return None
        t0, d0 = self.samples[0]
        t1, d1 = self.samples[-1]
        intervals = [b[0] - a[0] for a, b in zip(self.samples, self.samples[1:])]
        if t1 - t0 < 5 or d1 <= d0 or any(dt <= 0 for dt in intervals):
            return None
        rates = [(b[1] - a[1]) / (b[0] - a[0])
                 for a, b in zip(self.samples, self.samples[1:])]
        if max(rates) > min(rates) * 3:
            return None
        if self.clock() - t1 > max(10, 3 * median(intervals)):
            return None
        return (t1 - t0) / (d1 - d0) * max(0, self.total - d1)

    def text(self, show_counts=True):
        now = self.clock()
        elapsed = fmt_eta(now - self.start)
        if not self.total:
            return f"Elapsed {elapsed} - working; remaining time unavailable"
        fraction = min(1., self.done / self.total)
        eta = self.remaining()
        left = f"about {fmt_eta(eta)} left" if eta is not None else "estimating remaining time"
        counts = f" ({self.done:g}/{self.total:g})" if show_counts else ""
        return f"{fraction:.1%}{counts} - elapsed {elapsed} - {left}"
