"""Run short in-process computations (loading a solution, a tip scan, ...) off the GUI thread."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)


class _Task(QRunnable):
    def __init__(self, fn, args, kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = _Signals()

    def run(self):
        try:
            res = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
            try:
                self.signals.failed.emit(msg)
            except RuntimeError:        # receiver deleted (app quitting)
                pass
            return
        try:
            self.signals.done.emit(res)
        except RuntimeError:
            pass


_keep = set()


def run_async(fn, *args, on_done=None, on_error=None, **kwargs):
    """Run fn(*args, **kwargs) in the global thread pool; callbacks run on the GUI thread.

    After the callback the connections are dropped, so the task, its arguments and the callbacks'
    closures can be freed (a signal connection that refers back to its own task is never collected).
    """
    task = _Task(fn, args, kwargs)
    sig = task.signals
    _keep.add(sig)

    def finish(cb, payload):
        try:
            if cb is not None:
                cb(payload)
        finally:
            for s in (sig.done, sig.failed):
                try:
                    s.disconnect()
                except (RuntimeError, TypeError):
                    pass
            _keep.discard(sig)

    sig.done.connect(lambda r: finish(on_done, r))
    sig.failed.connect(lambda m: finish(on_error, m))
    QThreadPool.globalInstance().start(task)
    return task
