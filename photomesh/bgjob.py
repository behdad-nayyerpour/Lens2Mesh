"""Run work off the GUI thread.

BackgroundJob(fn): fn(report, cancelled) runs in a worker thread.
  report(done, total, text="", item=None) - throttled progress (Qt signal, queued
  to the GUI thread); items passed to report() are delivered in batches via
  `batch` so the GUI can apply partial results (e.g. thumbnails) as they come.
  cancelled() -> True once cancel() was called.
run_modal(...) shows a cancellable progress dialog and keeps the event loop
running until the job ends (the window never stops responding)."""
from __future__ import annotations

import logging
import threading
import time
import traceback

from PySide6.QtCore import QObject, Signal, QEventLoop, Qt

log = logging.getLogger("photomesh")


# most recent worker exception (name, error, traceback, time): the error dialog
# of the step that ran the job puts its traceback into the report
LAST_ERROR = {}


def recent_error(max_age=30.0):
    if LAST_ERROR and time.time() - LAST_ERROR.get("time", 0) <= max_age:
        return dict(LAST_ERROR)
    return {}


class BackgroundJob(QObject):
    progress = Signal(int, int, str)
    batch = Signal(list)
    done = Signal(object, object)          # result, error text (None if ok)

    def __init__(self, fn, name="job", interval=0.15, parent=None):
        super().__init__(parent)
        self.fn, self.name, self.interval = fn, name, interval
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._pending = []
        self._last = 0.0
        self.finished = False
        self.result = None
        self.error = None
        self.exception = None
        self.traceback = ""
        self.thread = None

    def cancel(self):
        self._cancel.set()

    def cancelled(self):
        return self._cancel.is_set()

    def _report(self, done, total, text="", item=None):
        flush = None
        with self._lock:
            if item is not None:
                self._pending.append(item)
            now = time.perf_counter()
            if now - self._last >= self.interval or done >= total:
                self._last = now
                flush, self._pending = self._pending, []
        if flush is not None:
            try:
                if flush:
                    self.batch.emit(flush)
                self.progress.emit(int(done), int(total), str(text))
            except RuntimeError:
                self.cancel()

    def _run(self):
        try:
            self.result = self.fn(self._report, self.cancelled)
        except Exception as e:  # noqa: BLE001
            self.exception = e
            self.error = f"{type(e).__name__}: {e}"
            notes = [str(n) for n in getattr(e, "__notes__", ())]
            if notes:
                self.error += "\n" + "\n".join(notes)
            self.traceback = traceback.format_exc()
            LAST_ERROR.update(name=self.name, error=self.error, traceback=self.traceback, time=time.time())
            log.error("%s failed:\n%s", self.name, self.traceback)
        with self._lock:
            rest, self._pending = self._pending, []
        self.finished = True
        try:
            if rest:
                self.batch.emit(rest)
            self.done.emit(self.result, self.error)
        except RuntimeError:
            pass  # the owning window was destroyed while the worker was stopping

    def start(self):
        self.thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self.thread.start()
        return self

    def wait(self, timeout=None):
        if self.thread is not None:
            self.thread.join(timeout)
        return self.finished


def run_modal(parent, title, label, fn, total, min_duration_ms=0):
    """Show the current stage of a single background operation."""
    from PySide6.QtWidgets import QApplication
    from .dialogs import ProgressDialog
    from .progress import StageProgress
    job = BackgroundJob(fn, name=title)
    dlg = ProgressDialog(title, parent)
    dlg.log.hide()
    dlg.begin_stage(label)
    dlg.setWindowModality(Qt.WindowModal)
    dlg.cancel_btn.clicked.connect(job.cancel)
    loop = QEventLoop()

    def prog(done, count, text):
        if not job.cancelled():
            dlg.set_stage_progress(StageProgress(label, done, count, detail=text if text != label else ""))
    job.progress.connect(prog)
    job.done.connect(loop.quit)
    busy = getattr(parent, "busy_indicator", None)
    if busy is not None:
        busy.begin(title)
    try:
        prog(0, total, label)
        dlg.show()
        QApplication.processEvents()
        job.start()
        if not job.finished:
            loop.exec()
        job.wait()
        QApplication.processEvents()
        if not job.error and not job.cancelled():
            dlg.finish_success()
    finally:
        dlg.running = False
        dlg._timer.stop()
        dlg.close()
        dlg.deleteLater()
        if busy is not None:
            busy.end()
    return job.result, job.cancelled(), job.error
