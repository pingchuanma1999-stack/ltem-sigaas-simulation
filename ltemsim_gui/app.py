"""Main window: sidebar navigation, lazily created pages, status bar with running jobs."""
from __future__ import annotations

import importlib
import os
import sys
import traceback

from PySide6.QtCore import QSettings, QSize, Qt
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
    ("home", "概览", "首页", "home", "home", "HomePage"),
    ("steady", "模拟", "稳态求解", "grid", "steady", "SteadyPage"),
    ("sweep", "模拟", "参数扫描", "sliders", "sweep", "SweepPage"),
    ("pulse", "模拟", "单脉冲发射链", "zap", "pulse_chain", "PulseChainPage"),
    ("fdtd", "模拟", "泵浦 FDTD", "waves", "fdtd", "FdtdPage"),
    ("tip", "模拟", "针尖位置链", "target", "tip_positions", "TipPositionsPage"),
    ("explorer", "分析", "解查看器", "layers", "explorer", "ExplorerPage"),
    ("results", "分析", "结果浏览", "folder", "results", "ResultsPage"),
    ("measured", "分析", "实测数据", "table", "measured", "MeasuredPage"),
    ("tools", "分析", "分析与作图", "chart", "tools", "ToolsPage"),
    ("jobs", "系统", "任务中心", "list", "jobs_page", "JobsPage"),
    ("docs", "系统", "文档", "book", "docs", "DocsPage"),
]


class _ErrorPage(Page):
    title = "页面加载失败"

    def __init__(self, err):
        super().__init__()
        l = label(err, "muted", wrap=True, selectable=True)
        l.setTextFormat(Qt.TextFormat.PlainText)
        self.body.addWidget(l)
        self.body.addStretch(1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(paths.APP_NAME + " — SI-GaAs 近场 LTEM 模拟")
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
        head.addWidget(label("SI-GaAs 近场太赫兹发射模拟", "appSubtitle"))
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
                it.setSizeHint(QSize(200, 30))
                self.nav.addItem(it)
                self.nav.setItemWidget(it, label(sec.upper(), "navSection"))
            it = QListWidgetItem(title)
            it.setData(Qt.ItemDataRole.UserRole, pid)
            it.setData(Qt.ItemDataRole.UserRole + 1, ic)
            it.setSizeHint(QSize(200, 36))
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
        self.job_label = label("无运行中的任务", "muted", wrap=True)
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
        self.theme_btn.setToolTip("切换主题：跟随系统 → 浅色 → 深色")
        self.theme_btn.clicked.connect(self._cycle_theme)
        row.addWidget(self.theme_btn)
        row.addStretch(1)
        ver = label("v1.0", "muted")
        row.addWidget(ver)
        lay.addLayout(row)
        return side

    def _build_menu(self):
        mb = self.menuBar()
        m = mb.addMenu("文件")
        a = QAction("打开解文件 (solution.npz)…", self)
        a.setShortcut(QKeySequence.StandardKey.Open)
        a.triggered.connect(self._open_solution)
        m.addAction(a)
        a = QAction("在访达中打开 results 文件夹", self)
        a.triggered.connect(lambda: reveal(paths.RESULTS))
        m.addAction(a)
        a = QAction("在访达中打开项目文件夹", self)
        a.triggered.connect(lambda: reveal(paths.ROOT))
        m.addAction(a)
        m.addSeparator()
        a = QAction("退出", self)
        a.setShortcut(QKeySequence.StandardKey.Quit)
        a.triggered.connect(self.close)
        m.addAction(a)

        m = mb.addMenu("视图")
        grp = QActionGroup(self)
        for mode, text in (("system", "跟随系统"), ("light", "浅色"), ("dark", "深色")):
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

        m = mb.addMenu("任务")
        grp = QActionGroup(self)
        for n in (1, 2, 3, 4):
            a = QAction(f"同时运行 {n} 个任务", self, checkable=True)
            a.setChecked(self.jobs.max_parallel == n)
            a.triggered.connect(lambda _=False, k=n: self._set_parallel(k))
            grp.addAction(a)
            m.addAction(a)
        m.addSeparator()
        a = QAction("任务中心", self)
        a.triggered.connect(lambda: self.navigate("jobs"))
        m.addAction(a)

        m = mb.addMenu("帮助")
        a = QAction("文档", self)
        a.triggered.connect(lambda: self.navigate("docs"))
        m.addAction(a)
        a = QAction("关于 LTEM Sim", self)
        a.triggered.connect(self._about)
        m.addAction(a)

    def _build_status(self):
        sb = self.statusBar()
        self.status_left = QLabel(f"项目: {paths.ROOT}")
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
                holder = QScrollArea()          # only scrolls when the window is smaller than the page's minimum
                holder.setWidgetResizable(True)
                holder.setFrameShape(QFrame.Shape.NoFrame)
                holder.setWidget(w)
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
        self.theme_btn.setText({"system": "跟随系统", "light": "浅色", "dark": "深色"}[mode])

    def _set_theme(self, mode):
        ThemeManager.instance().apply(mode)
        self.settings.setValue("theme", mode)
        for a in self._theme_actions.actions():
            a.setChecked(a.text() == {"system": "跟随系统", "light": "浅色", "dark": "深色"}[mode])

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
            self.job_label.setText("无运行中的任务")
            self.job_bar.setRange(0, 1)
            self.job_bar.setValue(0)
            return
        j = run[0] if run else queued[0]
        extra = f"（另有 {len(run) + len(queued) - 1} 个）" if len(run) + len(queued) > 1 else ""
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
        msg = {"finished": "已完成", "failed": "失败", "cancelled": "已取消"}.get(job.status, job.status)
        self.statusBar().showMessage(f"任务「{job.title}」{msg}", 8000)
        if job.status in ("finished", "failed") and not self.isActiveWindow():
            QApplication.alert(self)

    # ------------------------------------------------------------------ misc
    def _open_solution(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开稳态解", str(paths.RESULTS), "Solution (*.npz)")
        if path:
            self.navigate("explorer", path=path)

    def _about(self):
        QMessageBox.about(self, "关于 LTEM Sim",
                          "<h3>LTEM Sim</h3><p>SI-GaAs 近场 LTEM 模拟的桌面界面。</p>"
                          "<p>稳态漂移-扩散 + EL2 求解器、单脉冲发射链（泵浦 FDTD → 载流子动力学 → 针尖解调）、"
                          "参数扫描与结果分析。</p>"
                          f"<p>项目路径：{paths.ROOT}<br>Python：{sys.executable}</p>")

    def closeEvent(self, e):
        dirty = [title for pid, sec, title, *_ in PAGES
                 if hasattr(self.pages.get(pid), "is_dirty") and self.pages[pid].is_dirty()]
        if dirty:
            r = QMessageBox.question(self, "有未保存的修改", f"「{'、'.join(dirty)}」有未保存的修改。不保存直接退出吗？")
            if r != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
        active = self.jobs.active()
        if active:
            r = QMessageBox.question(self, "仍有任务在运行",
                                     f"还有 {len(active)} 个任务在运行或排队。退出将终止它们，确定退出吗？")
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
