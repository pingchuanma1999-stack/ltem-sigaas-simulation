"""Job centre: every simulation job of this and earlier sessions, live progress, logs and actions."""
from __future__ import annotations

import math
import re
import shlex
import sys
import time
from pathlib import Path

from PySide6.QtCore import QEvent, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QFont, QGuiApplication, QIcon, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLineEdit,
                               QMenu, QMessageBox, QSplitter, QStackedWidget, QStyle, QStyledItemDelegate,
                               QStyleOptionViewItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout)

from .. import icons, nav, paths
from ..jobs import CANCELLED, FAILED, FINISHED, QUEUED, RUNNING, STATUS_TEXT, JobManager
from ..theme import ThemeManager
from ..widgets.common import Page, button, label, reveal
from ..widgets.runpanel import RunPanel, fmt_elapsed
from .home import KIND_ICON, KIND_TEXT, Elided, plural, status_color

COLS = ("Status", "Name", "Type", "Started", "Elapsed", "Progress", "Output")
C_STATUS, C_NAME, C_KIND, C_START, C_ELAPSED, C_PROG, C_OUT = range(7)
SORT_ROLE = Qt.ItemDataRole.UserRole + 1
PROG_ROLE = Qt.ItemDataRole.UserRole + 2
STATUS_ROLE = Qt.ItemDataRole.UserRole + 3
STATUS_ORDER = {RUNNING: 0, QUEUED: 1, FAILED: 2, CANCELLED: 3, FINISHED: 4}
FILTERS = (("All statuses", None), ("Running / queued", (RUNNING, QUEUED)), ("Finished", (FINISHED,)),
           ("Failed", (FAILED,)), ("Cancelled", (CANCELLED,)))
# job kind -> (page id, open() keyword) for "Open results"; finished "tool" jobs open in the tools page (by job id)
RESULT_PAGE = {"steady": ("steady", "path"), "sweep": ("sweep", "path"), "pulse": ("pulse", "run_dir"),
               "redemod": ("pulse", "run_dir"), "fdtd": ("fdtd", "path"), "fdtd_heights": ("fdtd", "path"),
               "tip_sim": ("tip", "path"), "tip_demod": ("tip", "path")}
PAGE_TEXT = {"steady": "Steady state", "sweep": "Parameter sweep", "pulse": "Single-pulse chain", "fdtd": "Pump FDTD (optics)",
             "tip": "Tip-position chain", "results": "Results browser", "tools": "Tools & figures"}     # sidebar names


# ----------------------------------------------------------------------------- helpers (pure)
def _spec_dirs(job) -> list[str]:
    """Folders a job writes into: `out`, then the script's own folder arguments (redemod run_dir, heights out, sweep root)."""
    spec = job.spec or {}
    kw = spec.get("kwargs") or {}
    ds = [job.out, kw.get("run_dir"), kw.get("out"), kw.get("root")]
    if job.kind in ("pulse", "redemod"):
        args = spec.get("args") or []
        ds.append(args[0] if args else None)
    return list(dict.fromkeys(d for d in ds if isinstance(d, str) and d))


def result_folder(job) -> Path | None:
    """Absolute folder that holds this job's results, or None when it has none."""
    ds = _spec_dirs(job)
    if not ds:
        return None
    p = paths.absolute(ds[0])
    if job.kind == "sweep":                    # custom_sweep writes <root>/<name>/
        kw = job.spec.get("kwargs") or {}
        name = kw.get("name") or kw.get("key")
        if isinstance(name, str) and name and (p / name).is_dir():
            p = p / name
    return p


def result_target(job) -> tuple[str, dict] | None:
    """(page id, open kwargs) that shows this job's results, or None when the job has no result folder."""
    if job.kind == "tool" and getattr(job, "status", None) == FINISHED:
        return "tools", {"job_id": job.id}
    d = result_folder(job)
    if d is None:
        return None
    page, key = RESULT_PAGE.get(job.kind, ("results", "path"))
    return page, {key: str(d)}


def _inside(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def rerun_warning(job) -> str | None:
    """Why re-running `job` from the job centre may overwrite repository results (None when it is safe)."""
    dirs = [paths.absolute(d) for d in _spec_dirs(job)]
    committed = [d for d in dirs if _inside(d, paths.RESULTS) and not _inside(d, paths.GUI_RUNS)]
    if committed:
        return ("This job writes into result folders committed to the repository:\n"
                + "\n".join(f"  {paths.rel(d)}" for d in committed)
                + "\n\nRe-running it overwrites files with the same names there (the confirmation and backup that "
                  "the original page does before a run are skipped).")
    spec = job.spec or {}
    if not dirs and job.kind == "tool" and spec.get("func", "main") is None and spec.get("script", "").startswith("scripts/"):
        return ("This script has no output folder of its own: it writes its figures to fixed locations (e.g. the "
                "committed figures in results/slides/).\n\nRe-running it overwrites those files directly (the "
                "confirmation and backup that Tools & figures does before a run are skipped).")
    return None


def _lit(v) -> str:
    """Python literal of a runner argument: JSON lists become tuples (as in the runner), inf / nan spelled out."""
    if isinstance(v, list):
        items = [_lit(x) for x in v]
        return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"
    if isinstance(v, float) and not math.isfinite(v):
        return f"float('{v}')"
    return repr(v)


def call_args(spec: dict) -> str:
    args = [_lit(a) for a in spec.get("args") or []]
    args += [f"{k}={_lit(v)}" for k, v in (spec.get("kwargs") or {}).items()]
    return ", ".join(args)


def _python_cmd() -> str:
    venv = paths.ROOT / ".venv" / "bin" / "python"
    if venv.exists():
        return ".venv/bin/python"
    exe = Path(sys.executable)
    try:
        return shlex.quote(str(exe.relative_to(paths.ROOT)))
    except ValueError:
        return shlex.quote(str(exe))


def _shell_str(code: str) -> str:
    return f'"{code}"' if not re.search(r'["$`\\!]', code) else shlex.quote(code)


def shell_command(job) -> str:
    """An equivalent command line for a terminal (cwd = repo root, scripts/ on sys.path, Agg backend, like the runner)."""
    spec = job.spec
    script = spec.get("script", "")
    head = f"cd {shlex.quote(str(paths.ROOT))} && MPLBACKEND=Agg {_python_cmd()}"
    if spec.get("func", "main") is None:
        argv = " ".join(shlex.quote(str(a)) for a in spec.get("argv", []))
        return f"{head} {shlex.quote(script)} {argv}".rstrip()
    sp = Path(script)
    func = spec.get("func", "main")
    if sp.parent.as_posix() == "scripts" and sp.stem.isidentifier():
        mod = sp.stem
    elif not sp.is_absolute() and all(p.isidentifier() for p in sp.with_suffix("").parts):
        mod = ".".join(sp.with_suffix("").parts)
    else:
        mod = None
    if mod:
        code = f"import sys; sys.path[:0]=['scripts','.']; import {mod}; {mod}.{func}({call_args(spec)})"
    else:
        code = (f"import sys, runpy; sys.path[:0]=['scripts','.']; "
                f"runpy.run_path({script!r})[{func!r}]({call_args(spec)})")
    return f"{head} -c {_shell_str(code)}"


def fmt_start(job) -> str:
    ts = job.started or job.created
    if not ts:
        return "—"
    lt = time.localtime(ts)
    today = time.strftime("%Y%m%d", lt) == time.strftime("%Y%m%d")
    if job.status == QUEUED:                # not started yet: when it was queued (seconds in the tooltip)
        return "queued " + time.strftime("%H:%M" if today else "%m-%d %H:%M", lt)
    return time.strftime("%H:%M:%S", lt) if today else time.strftime("%m-%d %H:%M", lt)


_dot_cache: dict[str, QIcon] = {}
_kind_cache: dict[tuple[str, str], QIcon] = {}


def kind_icon(kind: str, color: str) -> QIcon:
    """Cached: re-rendering the SVG for every row on every 1 s tick is wasted work."""
    key = (KIND_ICON.get(kind, "file"), color)
    ic = _kind_cache.get(key)
    if ic is None:
        ic = _kind_cache[key] = icons.icon(key[0], color)
    return ic


def dot_icon(color: str) -> QIcon:
    ic = _dot_cache.get(color)
    if ic is None:
        pm = QPixmap(24, 24)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor(color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QRectF(6, 6, 12, 12))
        p.end()
        pm.setDevicePixelRatio(2.0)
        ic = _dot_cache[color] = QIcon(pm)
    return ic


# ----------------------------------------------------------------------------- table pieces
class _Item(QTreeWidgetItem):
    def __lt__(self, other):
        col = self.treeWidget().sortColumn() if self.treeWidget() else 0
        a, b = self.data(col, SORT_ROLE), other.data(col, SORT_ROLE)
        if a is None or b is None or type(a) is not type(b):
            return str(self.text(col)) < str(other.text(col))
        return a < b


class ProgressDelegate(QStyledItemDelegate):
    """Rounded progress bar + percentage inside the Progress cell."""

    def paint(self, p: QPainter, opt, index):
        o = QStyleOptionViewItem(opt)
        self.initStyleOption(o, index)
        o.text = ""
        style = o.widget.style() if o.widget else None
        if style:
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, o, p, o.widget)
        t = ThemeManager.instance().tokens
        frac = index.data(PROG_ROLE)
        status = index.data(STATUS_ROLE)
        r = QRectF(opt.rect).adjusted(8, 0, -8, 0)
        text_w = 52
        bar = QRectF(r.x(), r.center().y() - 3.5, max(10.0, r.width() - text_w), 7)
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QPainterPath()
        track.addRoundedRect(bar, 3.5, 3.5)
        p.fillPath(track, QColor(t.surface2 if not t.dark else t.border))
        col = QColor(status_color(status, t))
        txt = "—"
        if status == RUNNING and frac is None:
            # unknown progress: a soft full-width band
            c = QColor(col)
            c.setAlpha(90)
            p.fillPath(track, c)
            txt = "running"
        elif frac is not None:
            f = max(0.0, min(1.0, float(frac)))
            if f > 0:
                chunk = QPainterPath()
                chunk.addRoundedRect(QRectF(bar.x(), bar.y(), max(7.0, bar.width() * f), bar.height()), 3.5, 3.5)
                p.fillPath(chunk, col)
            txt = f"{100 * f:.0f}%"
        p.setPen(QColor(t.muted))
        f = QFont(opt.font)
        f.setPointSizeF(max(8.0, f.pointSizeF() - 1))
        p.setFont(f)
        p.drawText(QRectF(bar.right() + 6, r.y(), text_w - 6, r.height()),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, txt)
        p.restore()


class _Splitter(QSplitter):
    """Prefers its minimum height: the page sits in the window's scroll area, which sizes it by its preferred height,
    so the natural heights of the table and the log would push the page past small windows (e.g. 1120 x 700)."""

    def sizeHint(self):
        return QSize(super().sizeHint().width(), self.minimumSizeHint().height())


# ----------------------------------------------------------------------------- page
class JobsPage(Page):
    title = "Job center"
    subtitle = "Live progress, logs and outputs of every job, including earlier sessions."

    def __init__(self):
        super().__init__()
        self.mgr = JobManager.instance()
        self.items: dict[str, _Item] = {}
        self.current = None

        self.counts = label("", "muted")
        self.header.add_action(self.counts)
        self.filter = QComboBox()
        for text, _ in FILTERS:
            self.filter.addItem(text)
        self.filter.currentIndexChanged.connect(self._apply_filter)
        self.header.add_action(self.filter)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search jobs…")
        self.search.setToolTip("Search by name, type, script, output folder or job ID (all words must match)")
        self.search.setClearButtonEnabled(True)
        self.search.setFixedWidth(200)
        self.search.textChanged.connect(self._apply_filter)
        self.header.add_action(self.search)
        self.btn_clear = button("Clear finished", self.clear_finished,
                                tooltip="Remove ended jobs from the list (no result files are deleted)")
        self.header.add_action(self.btn_clear)

        split = _Splitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)

        # ---------------- table
        top = QStackedWidget()
        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(COLS))
        self.tree.setHeaderLabels(COLS)
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(C_START, Qt.SortOrder.DescendingOrder)
        self.tree.setItemDelegateForColumn(C_PROG, ProgressDelegate(self.tree))
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.tree.currentItemChanged.connect(self._on_select)
        self.tree.itemDoubleClicked.connect(lambda *_: self.open_result())
        self.tree.setStyleSheet("QTreeView::item { padding: 5px 2px; }")
        self.tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        h = self.tree.header()
        h.setStretchLastSection(False)
        h.setSectionResizeMode(C_NAME, QHeaderView.ResizeMode.Stretch)
        for c, w in ((C_STATUS, 100), (C_KIND, 128), (C_START, 112), (C_ELAPSED, 88), (C_PROG, 124), (C_OUT, 250)):
            h.setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)
            self.tree.setColumnWidth(c, w)
        # the stretched Name column must not collapse on narrow windows: the Output column gives way (until the user sizes it)
        self._out_user_w = False
        self._fitting = False
        h.sectionResized.connect(self._on_section_resized)
        self.tree.viewport().installEventFilter(self)
        top.addWidget(self.tree)
        self.table_empty = label("", "muted", wrap=True)
        self.table_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(self.table_empty)
        self.top_stack = top
        split.addWidget(top)

        # ---------------- detail
        self.detail_stack = QStackedWidget()
        empty = QFrame()
        empty.setObjectName("card")
        el = QVBoxLayout(empty)
        self.detail_empty = label("Select a job above to see its log, parameters and output.", "muted", wrap=True)
        self.detail_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        el.addWidget(self.detail_empty)
        self.detail_stack.addWidget(empty)

        det = QFrame()
        det.setObjectName("card")
        dl = QVBoxLayout(det)
        dl.setContentsMargins(14, 12, 14, 12)
        dl.setSpacing(8)
        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar.addWidget(label("Job details", "cardTitle"))
        bar.addStretch(1)
        self.btn_stop = button("Stop", self.stop, "danger", "Stop this job (a queued job is simply cancelled)")
        self.btn_rerun = button("Re-run", lambda: self.rerun(), None, "Run it again with the same script, parameters and output folder")
        self.btn_open = button("Open results", self.open_result, "primary")
        self.btn_reveal = button("Show in Finder", self.reveal_output, None,
                                 "Show the output folder in Finder (the log file when the job has no output folder)")
        self.btn_copy = button("Copy command", self.copy_command, None, "Copy a terminal command that reproduces this job")
        for b in (self.btn_stop, self.btn_rerun, self.btn_reveal, self.btn_copy, self.btn_open):
            bar.addWidget(b)
        dl.addLayout(bar)
        self.panel = RunPanel(show_actions=False)          # the job centre has its own action bar
        dl.addWidget(self.panel, 1)
        meta = QGridLayout()
        meta.setHorizontalSpacing(12)
        meta.setVerticalSpacing(3)
        self.meta = {}
        for i, (k, text) in enumerate((("call", "Call"), ("out", "Output"), ("time", "Time"), ("log", "Log"))):
            meta.addWidget(label(text, "metricLabel"), i, 0, Qt.AlignmentFlag.AlignTop)
            if k in ("out", "log"):         # long paths: one elided line (full path in the tooltip) leaves room for the log
                v = Elided("", None, Qt.TextElideMode.ElideMiddle)
            else:
                v = label("", None, wrap=True, selectable=True)
                v.setTextFormat(Qt.TextFormat.PlainText)
            meta.addWidget(v, i, 1)
            self.meta[k] = v
        meta.setColumnStretch(1, 1)
        dl.addLayout(meta)
        self.detail_stack.addWidget(det)
        split.addWidget(self.detail_stack)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setSizes([330, 470])
        self.body.addWidget(split, 1)

        self._copy_timer = QTimer(self)
        self._copy_timer.setSingleShot(True)
        self._copy_timer.timeout.connect(lambda: self.btn_copy.setText("Copy command"))

        self._rebuild()                     # rows are filled with the current theme's colours
        self.mgr.job_added.connect(self._on_added)
        self.mgr.job_changed.connect(self._on_changed)
        self.mgr.jobs_reset.connect(self._rebuild)
        ThemeManager.instance().changed.connect(self._retheme)

    # ------------------------------------------------------------------ column fitting
    def eventFilter(self, obj, e):
        if obj is self.tree.viewport() and e.type() == QEvent.Type.Resize:
            self._fit_columns()
        return super().eventFilter(obj, e)

    def _on_section_resized(self, col, _old, _new):
        if col == C_OUT and not self._fitting:
            self._out_user_w = True

    def _fit_columns(self):
        if self._out_user_w:
            return
        rest = self.tree.viewport().width() - sum(self.tree.columnWidth(c) for c in
                                                   (C_STATUS, C_KIND, C_START, C_ELAPSED, C_PROG))
        self._fitting = True
        self.tree.setColumnWidth(C_OUT, int(max(110, min(300, 0.42 * rest))))
        self._fitting = False

    # ------------------------------------------------------------------ table maintenance
    def _job(self, jid):
        return next((j for j in self.mgr.jobs if j.id == jid), None)

    def _fill(self, it: _Item, job):
        t = ThemeManager.instance().tokens
        col = QColor(status_color(job.status, t))
        it.setText(C_STATUS, STATUS_TEXT.get(job.status, job.status))
        it.setIcon(C_STATUS, dot_icon(col.name()))
        it.setForeground(C_STATUS, col)
        f = QFont(self.tree.font())
        f.setWeight(QFont.Weight.DemiBold)
        it.setFont(C_STATUS, f)
        it.setData(C_STATUS, SORT_ROLE, (STATUS_ORDER.get(job.status, 9), -(job.started or job.created or 0)))
        it.setText(C_NAME, job.title)
        it.setToolTip(C_NAME, job.title)
        it.setData(C_NAME, SORT_ROLE, job.title.lower())
        it.setText(C_KIND, KIND_TEXT.get(job.kind, job.kind))
        it.setIcon(C_KIND, kind_icon(job.kind, t.muted))
        it.setData(C_KIND, SORT_ROLE, job.kind)
        it.setText(C_START, fmt_start(job))
        it.setToolTip(C_START, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job.started or job.created)))
        it.setData(C_START, SORT_ROLE, float(job.started or job.created or 0))
        it.setText(C_ELAPSED, fmt_elapsed(job.elapsed()) if job.started else "—")
        it.setData(C_ELAPSED, SORT_ROLE, float(job.elapsed()))
        it.setData(C_PROG, PROG_ROLE, job.progress if job.progress is not None or job.status == RUNNING else
                   (1.0 if job.status == FINISHED else None))
        it.setData(C_PROG, STATUS_ROLE, job.status)
        it.setData(C_PROG, SORT_ROLE, float(job.progress or (1.0 if job.status == FINISHED else 0.0)))
        it.setToolTip(C_PROG, job.progress_text or STATUS_TEXT.get(job.status, ""))
        d = result_folder(job)
        out = paths.rel(d) if d is not None else ""
        it.setText(C_OUT, out or "—")
        it.setToolTip(C_OUT, str(d) if d is not None else "This job has no output folder")
        it.setData(C_OUT, SORT_ROLE, out)
        it.setData(0, Qt.ItemDataRole.UserRole, job.id)

    def _add_item(self, job):
        it = _Item()
        self._fill(it, job)
        self.items[job.id] = it
        return it

    def _rebuild(self):
        cur = self.current.id if self.current is not None else None
        self.tree.setSortingEnabled(False)
        self.tree.clear()
        self.items = {}
        for j in self.mgr.jobs:
            self.tree.addTopLevelItem(self._add_item(j))
        self.tree.setSortingEnabled(True)
        self._apply_filter()
        if cur in self.items:
            self.tree.setCurrentItem(self.items[cur])
        else:
            self._select_first()
        self._update_counts()

    def _on_added(self, job):
        sorting = self.tree.isSortingEnabled()
        self.tree.setSortingEnabled(False)
        self.tree.addTopLevelItem(self._add_item(job))
        self.tree.setSortingEnabled(sorting)
        self._apply_filter()
        self._update_counts()

    def _on_changed(self, job):
        it = self.items.get(job.id)
        if it is None:
            return self._on_added(job)
        self._fill(it, job)
        if job is self.current:
            self._update_detail()
        self._update_counts()
        # a status change may move the row out of the current filter
        flt = FILTERS[self.filter.currentIndex()][1]
        if flt is not None:
            self._apply_filter()

    def _update_counts(self):
        js = self.mgr.jobs
        n = {s: sum(1 for j in js if j.status == s) for s in (RUNNING, QUEUED, FAILED)}
        parts = [plural(len(js), "job")]
        if n[RUNNING]:
            parts.append(f"{n[RUNNING]} running")
        if n[QUEUED]:
            parts.append(f"{n[QUEUED]} queued")
        if n[FAILED]:
            parts.append(f"{n[FAILED]} failed")
        self.counts.setText(" · ".join(parts))
        self.btn_clear.setEnabled(any(not j.active for j in js))

    def _matches(self, job) -> bool:
        flt = FILTERS[self.filter.currentIndex()][1]
        if flt is not None and job.status not in flt:
            return False
        q = self.search.text().strip().lower()
        if q:
            hay = " ".join(str(x) for x in (job.title, job.kind, KIND_TEXT.get(job.kind, ""), job.spec.get("script", ""),
                                            *_spec_dirs(job), job.id)).lower()
            return all(w in hay for w in q.split())
        return True

    def _apply_filter(self, *_):
        shown = 0
        jobs = {j.id: j for j in self.mgr.jobs}
        for jid, it in self.items.items():
            j = jobs.get(jid)
            ok = j is not None and self._matches(j)
            it.setHidden(not ok)
            shown += ok
        if not self.mgr.jobs:
            self.table_empty.setText("No jobs yet.\nStart a simulation on a page such as Steady state or Single-pulse chain, "
                                     "and it appears here with live progress and its log.")
            self.top_stack.setCurrentIndex(1)
        elif shown == 0:
            self.table_empty.setText("No jobs match the filter.")
            self.top_stack.setCurrentIndex(1)
        else:
            self.top_stack.setCurrentIndex(0)

    def _select_first(self):
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if not it.isHidden():
                self.tree.setCurrentItem(it)
                return
        self._show_job(None)

    # ------------------------------------------------------------------ detail
    def _on_select(self, cur, _prev):
        jid = cur.data(0, Qt.ItemDataRole.UserRole) if cur is not None else None
        self._show_job(self._job(jid) if jid else None)

    def _show_job(self, job):
        if job is self.current and job is not None:
            self._update_detail()
            return
        self.current = job
        self.panel.attach(job)
        self.detail_stack.setCurrentIndex(1 if job is not None else 0)
        self._update_detail()

    def _update_detail(self):
        j = self.current
        if j is None:
            return
        self.btn_stop.setVisible(j.active)
        self.btn_stop.setText("Cancel" if j.status == QUEUED else "Stop")
        self.btn_rerun.setEnabled(not j.active)
        self.btn_rerun.setToolTip("A job can be re-run once it has ended (so that two jobs never write to the same "
                                  "output folder)" if j.active else
                                  "Run it again with the same script, parameters and output folder")
        ok, tip = self._open_state(j)
        self.btn_open.setEnabled(ok)
        self.btn_open.setToolTip(tip)
        spec = j.spec
        if spec.get("func", "main") is None:
            call = f"python {spec.get('script', '')} {' '.join(map(str, spec.get('argv', [])))}".rstrip()
        else:
            call = f"{spec.get('script', '')} : {spec.get('func', 'main')}({call_args(spec)})"
        self.meta["call"].setText(call)
        d = result_folder(j)
        self.meta["out"].setText(str(d) if d is not None else "— (this job has no output folder)")
        self.meta["out"].setToolTip(str(d) if d is not None else "")
        fmt = lambda ts: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "—"
        tm = f"created {fmt(j.created)}   started {fmt(j.started)}   ended {fmt(j.finished)}"
        if j.exit_code is not None:
            tm += f"   exit code {j.exit_code}"
        if j.after is not None and j.status == QUEUED:
            tm += f"   waiting for “{j.after.title}” to finish"
        self.meta["time"].setText(tm)
        self.meta["log"].setText(f"job ID {j.id}   ·   {j.log_path}")
        self.meta["log"].setToolTip(str(j.log_path))

    # ------------------------------------------------------------------ actions
    @staticmethod
    def _open_state(j) -> tuple[bool, str]:
        """(can open, tooltip) for Open results."""
        if j.active:
            return False, "Results can be opened once the job has ended"
        tgt = result_target(j)
        if tgt is None:
            return False, "This job has no output folder"
        page = PAGE_TEXT.get(tgt[0], tgt[0])
        if tgt[0] == "tools":
            return True, f"Show this job's output in {page}"
        d = Path(next(iter(tgt[1].values())))
        if not d.exists():
            return False, "The output folder does not exist (the job wrote no output, or it was deleted)"
        return True, f"Open {paths.rel(d)} in {page}"

    def stop(self):
        if self.current is not None and self.current.active:
            self.mgr.cancel(self.current)

    def rerun(self, confirm=True):
        j = self.current
        if j is None or j.active:
            return None
        warn = rerun_warning(j)
        if warn and confirm:
            r = QMessageBox.question(self, "Re-running overwrites existing results", warn + "\n\nRe-run anyway?")
            if r != QMessageBox.StandardButton.Yes:
                return None
        job = self.mgr.rerun(j)
        self.filter.setCurrentIndex(0)
        it = self.items.get(job.id)
        if it is not None:
            self.tree.setCurrentItem(it)
            self.tree.scrollToItem(it)
        return job

    def open_result(self):
        j = self.current
        if j is None or not self._open_state(j)[0]:
            return
        page, kw = result_target(j)
        nav.navigate(page, **kw)

    def reveal_output(self):
        j = self.current
        if j is None:
            return
        d = result_folder(j)
        if d is not None and d.exists():
            reveal(d)
        elif Path(j.log_path).exists():
            reveal(j.log_path)

    def copy_command(self):
        if self.current is None:
            return
        QGuiApplication.clipboard().setText(shell_command(self.current))
        self.btn_copy.setText("Copied ✓")
        self._copy_timer.start(1600)

    def clear_finished(self, confirm=True):
        done = [j for j in self.mgr.jobs if not j.active]
        if not done:
            return
        if confirm:
            r = QMessageBox.question(self, "Clear finished jobs",
                                     f"Remove {plural(len(done), 'ended job')} from the job list?\n"
                                     "Result folders and log files are not deleted.")
            if r != QMessageBox.StandardButton.Yes:
                return
        self.mgr.clear_finished()

    def _context_menu(self, pos):
        it = self.tree.itemAt(pos)
        if it is None:
            return
        self.tree.setCurrentItem(it)
        j = self.current
        m = QMenu(self)
        for text, slot, enabled in (("Open results", self.open_result, self.btn_open.isEnabled()),
                                    ("Show in Finder", self.reveal_output, True),
                                    ("Copy command", self.copy_command, True),
                                    ("Re-run", lambda: self.rerun(), bool(j and not j.active)),
                                    ("Stop", self.stop, bool(j and j.active))):
            a = QAction(text, m)
            a.setEnabled(enabled)
            a.triggered.connect(slot)
            m.addAction(a)
        m.exec(self.tree.viewport().mapToGlobal(pos))

    # ------------------------------------------------------------------ misc
    def _refill(self):
        jobs = {j.id: j for j in self.mgr.jobs}
        for jid, it in self.items.items():
            if jid in jobs:
                self._fill(it, jobs[jid])

    def _retheme(self, *_):
        self._refill()
        self.tree.viewport().update()

    def on_show(self):
        self._refill()
        if self.current is None:
            self._select_first()

    def open(self, job_id=None, **_):
        if not job_id:
            return
        it = self.items.get(job_id)
        if it is None:
            return
        if it.isHidden():
            self.filter.setCurrentIndex(0)
            self.search.clear()
        self.tree.setCurrentItem(it)
        self.tree.scrollToItem(it)
