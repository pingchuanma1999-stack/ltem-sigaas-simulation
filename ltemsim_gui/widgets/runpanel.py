"""Status strip + live log for the job a page launched."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from .. import paths
from ..jobs import CANCELLED, FAILED, FINISHED, PREREQ_FAILED, QUEUED, RUNNING, STATUS_TEXT, Job, JobManager
from ..theme import repolish
from .common import reveal
from .logview import LogView


def fmt_elapsed(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"


class RunPanel(QWidget):
    """Shows one Job: status, progress, elapsed time, cancel / open-folder buttons and the log."""
    job_finished = Signal(object)

    def __init__(self, parent=None, show_log=True, show_actions=True):
        super().__init__(parent)
        self.job: Job | None = None
        self.show_actions = show_actions
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(10)
        self.badge = QLabel("Idle")
        self.badge.setObjectName("badge")
        top.addWidget(self.badge)
        self.title = QLabel("Not run yet")
        self.title.setObjectName("cardTitle")
        self.title.setMinimumWidth(60)
        self.title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)   # long titles never widen the page
        top.addWidget(self.title, 1)
        self.elapsed = QLabel("")
        self.elapsed.setObjectName("muted")
        top.addWidget(self.elapsed)
        self.btn_folder = QPushButton("Show in Finder")
        self.btn_folder.setObjectName("flat")
        self.btn_folder.clicked.connect(self._open_folder)
        top.addWidget(self.btn_folder)
        self.btn_cancel = QPushButton("Stop")
        self.btn_cancel.setObjectName("danger")
        self.btn_cancel.clicked.connect(self._cancel)
        top.addWidget(self.btn_cancel)
        lay.addLayout(top)
        prow = QHBoxLayout()
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        prow.addWidget(self.bar, 1)
        self.ptext = QLabel("")
        self.ptext.setObjectName("muted")
        self.ptext.setMinimumWidth(160)
        prow.addWidget(self.ptext)
        lay.addLayout(prow)
        self.log = LogView() if show_log else None
        if self.log is not None:
            lay.addWidget(self.log, 1)
        self._refresh()

    def attach(self, job: Job | None):
        if self.job is not None:
            try:
                self.job.changed.disconnect(self._refresh)
                self.job.output.disconnect(self._on_output)
            except (RuntimeError, TypeError):
                pass
        self.job = job
        if self.log is not None:
            self.log.set_text(job.full_log() if job else "")
        if job is not None:
            job.changed.connect(self._refresh)
            job.output.connect(self._on_output)
        self._last_status = job.status if job else None
        self._refresh()

    def _on_output(self, text):
        if self.log is not None:
            self.log.append_text(text)

    def _refresh(self):
        j = self.job
        if j is None:
            self.badge.setText("Idle")
            self.badge.setObjectName("badge")
            self.title.setText("Not run yet")
            self.elapsed.setText("")
            self.bar.setRange(0, 100)
            self.bar.setValue(0)
            self.ptext.setText("")
            self.btn_cancel.setVisible(False)
            self.btn_folder.setVisible(False)
            repolish(self.badge)
            return
        self.badge.setText(STATUS_TEXT.get(j.status, j.status))
        self.badge.setObjectName({FINISHED: "badgeOk", FAILED: "badgeErr", CANCELLED: "badgeWarn"}.get(j.status, "badge"))
        repolish(self.badge)
        self.title.setText(j.title)
        self.title.setToolTip(j.title)
        self.elapsed.setText(fmt_elapsed(j.elapsed()) if j.started else "")
        if j.status == RUNNING and j.progress is None:
            self.bar.setRange(0, 0)          # busy indicator
        elif j.status == CANCELLED:
            self.bar.setRange(0, 1000)
            self.bar.setValue(0)
        else:
            self.bar.setRange(0, 1000)
            self.bar.setValue(int(1000 * (j.progress or (1.0 if j.status == FINISHED else 0.0))))
        txt = j.progress_text
        if j.status == QUEUED:
            txt = "waiting for earlier jobs…"
        elif j.status == FAILED:
            txt = f"exit code {j.exit_code} — see the log"
        elif j.status == CANCELLED:
            txt = j.progress_text if j.progress_text == PREREQ_FAILED else "cancelled"
        self.ptext.setText(txt)
        self.btn_cancel.setVisible(self.show_actions and j.active)
        self.btn_folder.setVisible(self.show_actions and bool(j.out))
        if getattr(self, "_last_status", None) != j.status:
            self._last_status = j.status
            if not j.active:
                self.job_finished.emit(j)

    def _cancel(self):
        if self.job is not None:
            JobManager.instance().cancel(self.job)

    def _open_folder(self):
        if self.job is not None and self.job.out:
            reveal(paths.absolute(self.job.out))
