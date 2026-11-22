"""Independent background samples for system CPU and NVIDIA adapter usage."""
from __future__ import annotations

import ctypes
import csv
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal

INTERVAL = 1.5
GPU_RETRY_INTERVAL = 8.0
_log = logging.getLogger(__name__)


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
                self.prev = self._times()
                return None
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


@dataclass(frozen=True)
class GpuReading:
    index: int
    name: str
    load: float | None
    used: int | None
    total: int | None


def _windows_gpu_sample():
    """Read the driver tool with explicit errors and below-normal priority."""
    if sys.platform != "win32":
        raise OSError("The NVIDIA fallback is available on Windows only")
    locations = (
        os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"),
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
    )
    executable = next((path for path in locations if os.path.isfile(path)), None)
    if executable is None:
        raise OSError("nvidia-smi was not found in the Windows driver locations")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    try:
        result = subprocess.run(
            [executable, "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], stdin=subprocess.DEVNULL,
            capture_output=True, text=True, errors="replace", timeout=4, creationflags=flags)
    except subprocess.TimeoutExpired as error:
        raise OSError("nvidia-smi timed out after 4 s") from error
    if result.returncode:
        message = (result.stderr or result.stdout).strip().splitlines()
        raise OSError(f"nvidia-smi exit {result.returncode}: {' '.join(message[:1])}")
    readings = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 5:
            continue
        index, name, usage, used, total = (value.strip() for value in row)
        if not index.isdecimal():
            continue
        try:
            load = float(usage)
        except ValueError:
            load = None
        if load is not None and not 0 <= load <= 100:
            load = None
        readings.append(GpuReading(int(index), name, load, int(used) if used.isdecimal() else None,
                                   int(total) if total.isdecimal() else None))
    if not readings:
        raise OSError("nvidia-smi returned no adapter readings")
    return tuple(readings)


class GpuSampler:
    """Read every NVIDIA adapter. Cache only timestamped driver-tool samples."""

    def __init__(self, loader=None, fallback=None, retry_interval=GPU_RETRY_INTERVAL):
        self.lib = None
        self._handles = ()
        self._loader = loader or self._load
        self._fallback = fallback if fallback is not None else (_windows_gpu_sample if sys.platform == "win32" else None)
        self._retry_interval = retry_interval
        self._retry_at = 0.0
        self._connect_error = ""
        self._state = {"gpu_cards": (), "gpu_source": "", "gpu_time": None, "gpu_error": ""}
        self._connect()

    @property
    def available(self):
        return any(handle is not None for _, _, handle in self._handles)

    def _connect(self):
        self._connect_error = ""
        try:
            lib = self._loader()
            if lib is None:
                raise OSError("NVML library could not be loaded")
            result = lib.nvmlInit_v2()
            if result != 0:
                raise OSError(f"NVML initialization returned {result}")
            self.lib = lib
            count = ctypes.c_uint(1)
            count_fn = getattr(lib, "nvmlDeviceGetCount_v2", None)
            if count_fn is not None:
                result = count_fn(ctypes.byref(count))
                if result != 0:
                    raise OSError(f"NVML adapter enumeration returned {result}")
            handles = []
            errors = []
            for index in range(count.value):
                handle = ctypes.c_void_p()
                result = lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle))
                if result != 0:
                    errors.append(f"NVML GPU {index} handle returned {result}")
                name = ctypes.create_string_buffer(96)
                if result == 0 and lib.nvmlDeviceGetName(handle, name, len(name)) != 0:
                    name.value = b""
                handles.append((index, name.value.decode("utf-8", "replace"), handle if result == 0 else None))
            self._handles = tuple(handles)
            self._connect_error = "; ".join(errors) if handles else "NVML found no NVIDIA adapters"
        except (OSError, AttributeError) as error:
            self._connect_error = str(error)
            self.close()

    @staticmethod
    def _load():
        for candidate in _nvml_candidates():
            try:
                return ctypes.CDLL(candidate)
            except OSError:
                continue
        return None

    def _read_nvml(self):
        readings = []
        errors = [self._connect_error] if self._connect_error else []
        for index, name, handle in self._handles:
            load, used, total = None, None, None
            if handle is not None:
                utilization = _Util()
                try:
                    result = self.lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(utilization))
                    if result != 0:
                        raise OSError(f"returned {result}")
                    load = float(utilization.gpu)
                except (OSError, AttributeError) as error:
                    errors.append(f"NVML GPU {index} utilization: {error}")
                memory = _Mem()
                try:
                    result = self.lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory))
                    if result != 0:
                        raise OSError(f"returned {result}")
                    used, total = memory.used >> 20, memory.total >> 20
                except (OSError, AttributeError) as error:
                    errors.append(f"NVML GPU {index} memory: {error}")
            readings.append(GpuReading(index, name, load, used, total))
        return tuple(readings), "; ".join(errors)

    def sample(self):
        now = time.monotonic()
        if now < self._retry_at:
            return self._state
        cards, error = self._read_nvml()
        source = "NVML" if cards else ""
        if not cards or any(card.load is None for card in cards):
            self.close()
            self._connect()
            cards, error = self._read_nvml()
            source = "NVML" if cards else ""
            if not cards or any(card.load is None for card in cards):
                if self._fallback is not None:
                    try:
                        alternate = self._fallback()
                        if alternate:
                            # The two APIs use NVIDIA's physical adapter numbering.
                            by_index = {card.index: card for card in alternate}
                            for card in cards:
                                reading = by_index.get(card.index)
                                if reading is None:
                                    by_index[card.index] = card
                                else:
                                    by_index[card.index] = GpuReading(
                                        card.index, reading.name or card.name,
                                        card.load if card.load is not None else reading.load,
                                        card.used if card.used is not None else reading.used,
                                        card.total if card.total is not None else reading.total)
                            cards = tuple(by_index[index] for index in sorted(by_index))
                            source = "NVML + nvidia-smi" if source else "nvidia-smi"
                    except OSError as failure:
                        error = "; ".join(part for part in (error, str(failure)) if part)
                self._retry_at = time.monotonic() + self._retry_interval
        self._state = {"gpu_cards": cards, "gpu_source": source,
                       "gpu_time": time.monotonic() if cards else None, "gpu_error": error}
        return self._state

    def close(self):
        try:
            if self.lib is not None:
                self.lib.nvmlShutdown()
        except (OSError, AttributeError):
            pass
        self.lib = None
        self._handles = ()


class ResourceMonitor(QObject):
    """Publish CPU and GPU readings without making one sampler wait for the other."""
    sampled = Signal(object)

    def __init__(self, interval=INTERVAL, cpu=None, gpu=None, parent=None):
        super().__init__(parent)
        self.interval = interval
        self._cpu, self._gpu = cpu, gpu
        self._stop = threading.Event()
        self._threads = ()
        self._gpu_state = {}
        self.last = None

    def start(self):
        if any(thread.is_alive() for thread in self._threads):
            return
        self._stop.clear()
        self._gpu_state = {}
        self._threads = (
            threading.Thread(target=self._cpu_loop, name="cpu-monitor", daemon=True),
            threading.Thread(target=self._gpu_loop, name="gpu-monitor", daemon=True),
        )
        for thread in self._threads:
            thread.start()

    def stop(self, timeout=2.0):
        self._stop.set()
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            if thread.is_alive() and thread is not threading.current_thread():
                thread.join(max(0.0, deadline - time.monotonic()))

    def _cpu_loop(self):
        cpu = self._cpu or CpuSampler()
        while not self._stop.wait(self.interval):
            self.last = {"cpu": cpu.percent(), **self._gpu_state}
            try:
                self.sampled.emit(self.last)
            except RuntimeError:
                return

    def _gpu_loop(self):
        gpu = self._gpu if self._gpu is not None else GpuSampler()
        previous_error = ""
        try:
            while not self._stop.is_set():
                try:
                    state = gpu.sample()
                except (OSError, AttributeError) as error:
                    state = {"gpu_cards": (), "gpu_error": str(error), "gpu_time": None}
                error = state.get("gpu_error", "")
                if error and error != previous_error:
                    _log.warning("GPU usage monitor: %s", error)
                previous_error = error
                self._gpu_state = state
                self._stop.wait(self.interval)
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
