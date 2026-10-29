"""'Cores' control for multi-run jobs and the matching runtime estimate."""
from __future__ import annotations

import math
import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSpinBox, QWidget

from ..parallel import default_workers

# Per-run slowdown when k runs share the machine (measured on an 18-core Apple M5 Pro, coarse steady solves:
# 1 → 30.4 s, 6 → 38.0 s, 12 → 42.1 s, 18 → 52.1 s per run).  Interpolated linearly in between.
_SLOWDOWN = ((1, 1.0), (6, 1.25), (12, 1.39), (18, 1.71))


def slowdown(k: int) -> float:
    pts = _SLOWDOWN
    if k <= pts[0][0]:
        return 1.0
    for (k0, s0), (k1, s1) in zip(pts, pts[1:]):
        if k <= k1:
            return s0 + (s1 - s0) * (k - k0) / (k1 - k0)
    return pts[-1][1] * k / pts[-1][0]          # beyond the core count: time-slicing


def estimate_s(per_run_s: float, n_runs: int, workers: int) -> float:
    """Wall time of n_runs independent runs of per_run_s each (single-core time) on `workers` processes."""
    if n_runs <= 0:
        return 0.0
    k = max(1, min(workers, n_runs))
    return math.ceil(n_runs / k) * per_run_s * slowdown(k)


class CoresSpin(QWidget):
    """'Cores [ 6 ] of 18' — how many runs of a job execute at the same time."""
    changed = Signal(int)

    def __init__(self, parent=None, label="Cores"):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.lab = QLabel(label)
        self.spin = QSpinBox()
        n = os.cpu_count() or 1
        self.spin.setRange(1, n)
        self.spin.setValue(default_workers(n))
        self.spin.setToolTip(f"How many of this job's independent runs execute at the same time (one core each). "
                             f"This Mac has {n} cores; the default leaves two for the desktop. "
                             f"A single steady-state solve always uses one core (its sparse LU solver is single-threaded).")
        self.of = QLabel(f"of {n}")
        self.of.setObjectName("muted")
        lay.addWidget(self.lab)
        lay.addWidget(self.spin)
        lay.addWidget(self.of)
        self._auto = True
        self.spin.valueChanged.connect(self._user_changed)

    def _user_changed(self, v):
        self._auto = False
        self.changed.emit(v)

    def set_runs(self, n_runs: int):
        """Follow the job size until the user picks a value themselves."""
        if self._auto:
            self.spin.blockSignals(True)
            self.spin.setValue(default_workers(max(1, n_runs)))
            self.spin.blockSignals(False)
            self.changed.emit(self.spin.value())

    def value(self) -> int:
        return self.spin.value()
