"""Main window: sidebar navigation, lazily created pages, status bar with running jobs."""
from __future__ import annotations

import importlib
import os
import sys
import traceback

from PySide6.QtCore import QEvent, QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence
from PySide6.QtWidgets import (QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QStackedWidget,
                               QVBoxLayout, QWidget)

from . import icons, nav, paths
from .jobs import RUNNING, JobManager
from .theme import ThemeManager
from .widgets.common import Page, label, reveal

# (id, section, title, icon, module, class)
PAGES = [
    ("home", "Overview", "Home", "home", "home", "HomePage"),
    ("steady", "Simulate", "Steady state", "grid", "steady", "SteadyPage"),
    ("sweep", "Simulate", "Parameter sweep", "sliders", "sweep", "SweepPage"),
    ("pulse", "Simulate", "Single-pulse chain", "zap", "pulse_chain", "PulseChainPage"),
    ("fdtd", "Simulate", "Pump FDTD (optics)", "waves", "fdtd", "FdtdPage"),
    ("tip", "Simulate", "Tip-position chain", "target", "tip_positions", "TipPositionsPage"),
    ("explorer", "Analyze", "Solution explorer", "layers", "explorer", "ExplorerPage"),
    ("results", "Analyze", "Results browser", "folder", "results", "ResultsPage"),
    ("measured", "Analyze", "Measured data", "table", "measured", "MeasuredPage"),
    ("tools", "Analyze", "Tools & figures", "chart", "tools", "ToolsPage"),
    ("jobs", "System", "Job center", "list", "jobs_page", "JobsPage"),
    ("docs", "System", "Docs", "book", "docs", "DocsPage"),
]


class _ErrorPage(Page):
    title = "Page failed to load"

    def __init__(self, err):
        super().__init__()
        l = label(err, "muted", wrap=True, selectable=True)
        l.setTextFormat(Qt.TextFormat.PlainText)
        self.body.addWidget(l)
        self.body.addStretch(1)


class _PageHolder(QScrollArea):
    """Shows a page at exactly the viewport size; scrolls only below the page's minimum size.

    (A plain widgetResizable scroll area sizes height-for-width pages to their preferred height, so pages
    with wrapped labels and large plots would scroll as a whole even when they could shrink.)
    """

    def __init__(self, page: QWidget):
        super().__init__()
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidgetResizable(False)
        self.setWidget(page)
        page.installEventFilter(self)

    def eventFilter(self, obj, e):
        if e.type() == QEvent.Type.LayoutRequest:
            QTimer.singleShot(0, self._fit)
        return False

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def _fit(self):
        w = self.widget()
        if w is None:
            return
        m = w.minimumSizeHint().expandedTo(w.minimumSize())
        avail = self.maximumViewportSize()
        need_w, need_h = m.width() > avail.width(), m.height() > avail.height()
        width = avail.width() - (self.verticalScrollBar().sizeHint().width() if need_h else 0)
        height = avail.height() - (self.horizontalScrollBar().sizeHint().height() if need_w else 0)
        w.resize(max(width, m.width()), max(height, m.height()))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(paths.APP_NAME + " — SI-GaAs near-field LTEM simulation")
        scr = QApplication.primaryScreen()
        avail = scr.availableGeometry() if scr is not None else None
        if avail is not None:
            self.resize(min(1440, int(avail.width() * 0.92)), min(900, int(avail.height() * 0.9)))
        else:
            self.resize(1440, 900)
        self.setMinimumSize(1120, 660)          # widest page (sweep) needs ~900 px next to the sidebar
        self.settings = QSettings(paths.APP_ID, paths.APP_ID)
        self.jobs = JobManager.instance()
        self.jobs.max_parallel = int(self.settings.value("max_parallel", 1))
        self.pages: dict[str, QWidget] = {}
        self._holders: dict[str, QScrollArea] = {}
        self._items: dict[str, QListWidgetItem] = {}

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_sidebar())
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self._build_menu()
        self._build_status()

        ThemeManager.instance().changed.connect(self._retheme)
        self.jobs.job_changed.connect(lambda *_: self._update_jobs_indicator())
        self.jobs.job_added.connect(lambda *_: self._update_jobs_indicator())
        self.jobs.job_finished.connect(self._on_job_finished)
        nav.set_window(self)
        self._retheme()
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        self.navigate(self.settings.value("last_page", "home") or "home")

    # ------------------------------------------------------------------ layout
    def _build_sidebar(self):
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(224)
        lay = QVBoxLayout(side)
        lay.setContentsMargins(0, 18, 0, 12)
        lay.setSpacing(0)
        head = QVBoxLayout()
        head.setContentsMargins(18, 0, 18, 10)
        head.setSpacing(2)
        head.addWidget(label("LTEM Sim", "appTitle"))
        head.addWidget(label("SI-GaAs near-field THz emission", "appSubtitle"))
        lay.addLayout(head)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setIconSize(QSize(18, 18))
        self.nav.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)   # short windows
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        section = None
        for pid, sec, title, ic, *_ in PAGES:
            if sec != section:
                section = sec
                it = QListWidgetItem()
                it.setFlags(Qt.ItemFlag.NoItemFlags)
                it.setSizeHint(QSize(200, 28))
                self.nav.addItem(it)
                self.nav.setItemWidget(it, label(sec.upper(), "navSection"))
            it = QListWidgetItem(title)
            it.setData(Qt.ItemDataRole.UserRole, pid)
            it.setData(Qt.ItemDataRole.UserRole + 1, ic)
            it.setSizeHint(QSize(200, 32))
            self.nav.addItem(it)
            self._items[pid] = it
        self.nav.currentItemChanged.connect(self._on_nav)
        lay.addWidget(self.nav, 1)

        # running-jobs indicator
        self.job_box = QFrame()
        self.job_box.setObjectName("card")
        self.job_box.setCursor(Qt.CursorShape.PointingHandCursor)
        jb = QVBoxLayout(self.job_box)
        jb.setContentsMargins(10, 8, 10, 8)
        jb.setSpacing(4)
        self.job_label = label("No running jobs", "muted", wrap=True)
        self.job_bar = QProgressBar()
        self.job_bar.setTextVisible(False)
        self.job_bar.setFixedHeight(6)
        jb.addWidget(self.job_label)
        jb.addWidget(self.job_bar)
        self.job_box.mousePressEvent = lambda e: self.navigate("jobs")
        wrap = QVBoxLayout()
        wrap.setContentsMargins(12, 6, 12, 6)
        wrap.addWidget(self.job_box)
        lay.addLayout(wrap)

        row = QHBoxLayout()
        row.setContentsMargins(12, 2, 12, 0)
        self.theme_btn = QPushButton("")
        self.theme_btn.setObjectName("flat")
        self.theme_btn.setToolTip("Switch theme: system → light → dark")
        self.theme_btn.clicked.connect(self._cycle_theme)
        row.addWidget(self.theme_btn)
        row.addStretch(1)
        ver = label("v1.0", "muted")
        row.addWidget(ver)
        lay.addLayout(row)
        return side

    def _build_menu(self):
        mb = self.menuBar()
        m = mb.addMenu("File")
        a = QAction("Open solution (solution.npz)…", self)
        a.setShortcut(QKeySequence.StandardKey.Open)
        a.triggered.connect(self._open_solution)
        m.addAction(a)
        a = QAction("Show results folder in Finder", self)
        a.triggered.connect(lambda: reveal(paths.RESULTS))
        m.addAction(a)
        a = QAction("Show project folder in Finder", self)
        a.triggered.connect(lambda: reveal(paths.ROOT))
        m.addAction(a)
        m.addSeparator()
        a = QAction("Quit", self)
        a.setShortcut(QKeySequence.StandardKey.Quit)
        a.triggered.connect(self.close)
        m.addAction(a)

        m = mb.addMenu("View")
        grp = QActionGroup(self)
        for mode, text in (("system", "System"), ("light", "Light"), ("dark", "Dark")):
            a = QAction(text, self, checkable=True)
            a.setChecked(ThemeManager.instance().mode == mode)
            a.triggered.connect(lambda _=False, md=mode: self._set_theme(md))
            grp.addAction(a)
            m.addAction(a)
        self._theme_actions = grp
        m.addSeparator()
        for i, (pid, sec, title, *_) in enumerate(PAGES):
            a = QAction(title, self)
            if i < 9:
                a.setShortcut(QKeySequence(f"Ctrl+{i + 1}"))
            a.triggered.connect(lambda _=False, p=pid: self.navigate(p))
            m.addAction(a)

        m = mb.addMenu("Jobs")
        grp = QActionGroup(self)
        for n in (1, 2, 3, 4):
            a = QAction(f"Run {n} job{'s' if n > 1 else ''} at a time", self, checkable=True)
            a.setChecked(self.jobs.max_parallel == n)
            a.triggered.connect(lambda _=False, k=n: self._set_parallel(k))
            grp.addAction(a)
            m.addAction(a)
        m.addSeparator()
        a = QAction("Job center", self)
        a.triggered.connect(lambda: self.navigate("jobs"))
        m.addAction(a)

        m = mb.addMenu("Help")
        a = QAction("Docs", self)
        a.triggered.connect(lambda: self.navigate("docs"))
        m.addAction(a)
        a = QAction("About LTEM Sim", self)
        a.triggered.connect(self._about)
        m.addAction(a)

    def _build_status(self):
        sb = self.statusBar()
        self.status_left = QLabel(f"Project: {paths.ROOT}")
        sb.addWidget(self.status_left, 1)
        py = f"Python {sys.version_info.major}.{sys.version_info.minor}"
        try:
            import numpy, scipy, matplotlib
            py += f" · numpy {numpy.__version__} · scipy {scipy.__version__} · matplotlib {matplotlib.__version__}"
        except Exception:
            pass
        sb.addPermanentWidget(QLabel(py))

    # ------------------------------------------------------------------ navigation
    def _on_nav(self, cur, _prev):
        if cur is None:
            return
        pid = cur.data(Qt.ItemDataRole.UserRole)
        if pid:
            self._show(pid)

    def _create(self, pid):
        for p, sec, title, ic, mod, cls in PAGES:
            if p == pid:
                try:
                    m = importlib.import_module(f".pages.{mod}", __package__)
                    w = getattr(m, cls)()
                except Exception:
                    w = _ErrorPage(traceback.format_exc())
                holder = _PageHolder(w)
                self.stack.addWidget(holder)
                self._holders[pid] = holder
                self.pages[pid] = w
                return w
        return None

    def _show(self, pid, **kwargs):
        w = self.pages.get(pid) or self._create(pid)
        if w is None:
            return
        self.stack.setCurrentWidget(self._holders[pid])
        self.settings.setValue("last_page", pid)
        if kwargs and hasattr(w, "open"):
            w.open(**kwargs)
        if hasattr(w, "on_show"):
            w.on_show()

    def navigate(self, pid, **kwargs):
        it = self._items.get(pid)
        if it is None:
            pid, it = "home", self._items["home"]
        if self.nav.currentItem() is not it:
            self.nav.blockSignals(True)
            self.nav.setCurrentItem(it)
            self.nav.blockSignals(False)
        self._show(pid, **kwargs)

    # ------------------------------------------------------------------ theme
    def _retheme(self, *_):
        t = ThemeManager.instance().tokens
        for pid, it in self._items.items():
            it.setIcon(icons.icon(it.data(Qt.ItemDataRole.UserRole + 1), t.muted, t.accent))
        mode = ThemeManager.instance().mode
        self.theme_btn.setIcon(icons.icon("moon" if t.dark else "sun", t.muted))
        self.theme_btn.setText({"system": "System", "light": "Light", "dark": "Dark"}[mode])

    def _set_theme(self, mode):
        ThemeManager.instance().apply(mode)
        self.settings.setValue("theme", mode)
        for a in self._theme_actions.actions():
            a.setChecked(a.text() == {"system": "System", "light": "Light", "dark": "Dark"}[mode])

    def _cycle_theme(self):
        order = ["system", "light", "dark"]
        cur = ThemeManager.instance().mode
        self._set_theme(order[(order.index(cur) + 1) % 3])

    # ------------------------------------------------------------------ jobs
    def _set_parallel(self, n):
        self.jobs.max_parallel = n
        self.settings.setValue("max_parallel", n)
        self.jobs._schedule()

    def _update_jobs_indicator(self):
        run = self.jobs.running()
        queued = [j for j in self.jobs.active() if j.status != RUNNING]
        if not run and not queued:
            self.job_label.setText("No running jobs")
            self.job_bar.setRange(0, 1)
            self.job_bar.setValue(0)
            return
        j = run[0] if run else queued[0]
        extra = f" (+{len(run) + len(queued) - 1} more)" if len(run) + len(queued) > 1 else ""
        txt = f"▶ {j.title}{extra}"
        if j.progress_text:
            txt += f"\n{j.progress_text}"
        self.job_label.setText(txt)
        if j.progress is None:
            self.job_bar.setRange(0, 0)
        else:
            self.job_bar.setRange(0, 1000)
            self.job_bar.setValue(int(1000 * j.progress))

    def _on_job_finished(self, job):
        self._update_jobs_indicator()
        msg = {"finished": "finished", "failed": "failed", "cancelled": "was cancelled"}.get(job.status, job.status)
        self.statusBar().showMessage(f"Job \"{job.title}\" {msg}", 8000)
        if job.status in ("finished", "failed") and not self.isActiveWindow():
            QApplication.alert(self)

    # ------------------------------------------------------------------ misc
    def _open_solution(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open steady-state solution", str(paths.RESULTS), "Solution (*.npz)")
        if path:
            self.navigate("explorer", path=path)

    def _about(self):
        QMessageBox.about(self, "About LTEM Sim",
                          "<h3>LTEM Sim</h3><p>Desktop interface for the SI-GaAs near-field LTEM simulation.</p>"
                          "<p>Steady-state drift-diffusion + EL2 solver, single-pulse emission chain (pump FDTD → carrier dynamics → "
                          "tip demodulation), parameter sweeps and result analysis.</p>"
                          f"<p>Project: {paths.ROOT}<br>Python: {sys.executable}</p>")

    def closeEvent(self, e):
        dirty = [title for pid, sec, title, *_ in PAGES
                 if hasattr(self.pages.get(pid), "is_dirty") and self.pages[pid].is_dirty()]
        if dirty:
            r = QMessageBox.question(self, "Unsaved changes", f"{', '.join(dirty)}: unsaved changes. Quit without saving?")
            if r != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
        active = self.jobs.active()
        if active:
            r = QMessageBox.question(self, "Jobs still running",
                                     f"{len(active)} job(s) are running or queued. Quitting stops them. Quit anyway?")
            if r != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
        self.jobs.shutdown()
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(e)


def main(argv=None):
    os.environ.setdefault("QT_API", "pyside6")
    from .macos import set_app_name
    set_app_name(paths.APP_NAME)
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName(paths.APP_NAME)
    app.setOrganizationName(paths.APP_ID)
    icon_path = paths.ASSETS / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    f = app.font()
    f.setPointSizeF(max(f.pointSizeF(), 13.0) if sys.platform == "darwin" else f.pointSizeF())
    app.setFont(f)
    from .fonts import install_cjk_fallback
    install_cjk_fallback()          # Chinese text in embedded plots
    settings = QSettings(paths.APP_ID, paths.APP_ID)
    ThemeManager.instance().apply(settings.value("theme", "system") or "system")
    w = MainWindow()
    w.show()
    return app.exec()
