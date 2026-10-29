"""Results browser: tree / recently modified files of results/ (or any folder) with typed previews
(widgets/fileviews.py), run provenance, jumps to the simulation pages and safe deletion of GUI runs."""
from __future__ import annotations

import os
import re
import time
import zipfile
from fnmatch import translate
from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QEvent, QFile, QFileInfo, QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QFileIconProvider, QFileSystemModel, QFrame,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMenu, QMessageBox, QSizePolicy,
                               QSplitter, QStackedWidget, QTreeView, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)

from .. import icons, nav, paths
from ..jobs import JobManager
from ..theme import ThemeManager
from ..widgets.common import Page, button, error_box, label, open_external, reveal
from ..widgets.fileviews import (IMAGE_EXT, CompareView, ElideLabel, describe, fmt_time, human_size, kind_of,
                                 preview_for, segmented)
from ..workers import run_async

RECENT_N = 30
ROW_QSS = "QTreeView::item { height: 24px; }"
MATCH_CAP = 300
SCAN_CAP = 100_000
# page id -> (button text, icon, short text for narrow windows)
TARGETS = {"explorer": ("在解查看器中打开", "layers", "解查看器"), "steady": ("在稳态求解页面打开", "grid", "稳态求解"),
           "sweep": ("在参数扫描页面打开", "sliders", "参数扫描"), "pulse": ("在单脉冲发射链页面打开", "zap", "单脉冲发射链"),
           "fdtd": ("在泵浦 FDTD 页面打开", "waves", "泵浦 FDTD"), "tip": ("在针尖位置链页面打开", "target", "针尖位置链")}
SWEEP_MARKERS = ("compare.png", "sweep_summary.json", "sweep_spec.json")


# ---------------------------------------------------------------------------------------------- pure helpers
def _inside(p: Path, root: Path) -> bool:
    try:
        p, root = Path(p).resolve(), Path(root).resolve()
    except OSError:
        return False
    return p == root or root in p.parents


def can_delete(path) -> bool:
    """Only folders / files strictly inside results/gui_runs/ (created by the GUI) may be deleted."""
    try:
        p, g = Path(path).resolve(), paths.GUI_RUNS.resolve()
    except OSError:
        return False
    return p != g and g in p.parents and p.exists()


def _npz_has(p: Path, key: str) -> bool:
    try:
        with zipfile.ZipFile(p) as z:
            return f"{key}.npy" in z.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def _is_sweep_root(d: Path) -> bool:
    """Same rule as the sweep page: compare.png / sweep_summary.json / sweep_spec.json or <key>=<value>/summary.json."""
    if any((d / f).exists() for f in SWEEP_MARKERS):
        return True
    try:
        return any(c.is_dir() and "=" in c.name and (c / "summary.json").exists() for c in d.iterdir())
    except OSError:
        return False


def _folder_targets(d: Path) -> list[tuple[str, dict]]:
    """Pages that can open folder d, as (page id, open kwargs)."""
    out = []
    try:
        d = d.resolve()
    except OSError:
        return out
    R = paths.RESULTS.resolve()
    gui = d.parent == paths.GUI_RUNS.resolve()
    name = d.name
    if (d.parent == R / "pulse_chain" and ((d / "summary.json").exists() or any(d.glob("X*.npz")))) or \
            (gui and name.startswith("pulse_")):
        out.append(("pulse", {"run_dir": str(d)}))
    if d.parent == R / "tip_positions" or (gui and name.startswith("tip_")):
        out.append(("tip", {"path": str(d)}))
    if d in (R / "fdtd_pump", R / "fdtd_pump" / "heights") or (gui and name.startswith("fdtd")) or \
            (d / "fdtd_pump.png").exists():
        out.append(("fdtd", {"path": str(d)}))
    if (d / "sweep_summary.json").exists() or (gui and name.startswith("sweep_")):
        out.append(("sweep", {"path": str(d)}))
    if (d / "summary.json").exists() and (d / "maps.png").exists():
        out.append(("steady", {"path": str(d)}))
    # one value of a sweep (<key>=<value>/): the sweep page opens the whole sweep with this value selected
    if "=" in name and (d / "summary.json").exists() and _is_sweep_root(d.parent):
        out.append(("sweep", {"path": str(d)}))
    if (d / "solution.npz").exists() and _npz_has(d / "solution.npz", "params_json"):
        out.append(("explorer", {"path": str(d / "solution.npz")}))
    return out


def nav_targets(path) -> list[tuple[str, str, dict]]:
    """(button text, page id, kwargs) for every page that can open `path` (for a file: also its folder)."""
    p = Path(path)
    out: list[tuple[str, dict]] = []
    if p.is_file():
        if p.suffix.lower() == ".npz" and _npz_has(p, "params_json"):
            out.append(("explorer", {"path": str(p)}))
        out += [t for t in _folder_targets(p.parent) if t[0] != "explorer"]
    elif p.is_dir():
        out = _folder_targets(p)
    return [(TARGETS[pid][0], pid, kw) for pid, kw in out]


def _patterns(text: str) -> list[str]:
    """Filter text → shell patterns ('maps' → '*maps*'; several terms separated by spaces or commas)."""
    terms = [t for t in text.replace(",", " ").split() if t]
    return [t if any(c in t for c in "*?[") else f"*{t}*" for t in terms]


@lru_cache(maxsize=64)
def _pattern_rx(pats: tuple[str, ...]):
    """One compiled, case-insensitive regex for all shell patterns (fnmatch per file and pattern took
    ~150 ms per keystroke for 100 000 files)."""
    return re.compile("|".join(f"(?:{translate(p.lower())})" for p in pats))


def _matches(name: str, pats: list[str]) -> bool:
    return bool(pats) and _pattern_rx(tuple(pats)).match(name.lower()) is not None


def scan_files(root: Path, cap: int = SCAN_CAP) -> dict:
    """All non-hidden files under root as (path, mtime, size), newest first, plus their lower-case base names
    (worker thread)."""
    files, total, truncated = [], 0, False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for f in filenames:
            if f.startswith("."):
                continue
            fp = os.path.join(dirpath, f)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            files.append((fp, st.st_mtime, st.st_size))
            total += st.st_size
        if len(files) >= cap:
            truncated = True
            break
    files.sort(key=lambda r: -r[1])
    names = [os.path.basename(r[0]).lower() for r in files]
    return dict(root=str(root), files=files, names=names, total=total, truncated=truncated, t=time.time())


def _short(p: Path, maxlen: int = 44) -> str:
    """Compact folder label: repo-relative, ~/…, or …/last/two/ for long paths."""
    p = Path(p)
    if _inside(p, paths.ROOT):
        r = paths.rel(p)
        s = (r if r != "." else paths.ROOT.name) + "/"
    else:
        try:
            s = "~/" + str(p.relative_to(Path.home())) + "/"
        except ValueError:
            s = str(p) + "/"
    if len(s) > maxlen and len(p.parts) > 2:
        s = "…/" + "/".join(p.parts[-2:]) + "/"
    return s


class _FsModel(QFileSystemModel):
    HEAD = ("名称", "大小", "类型", "修改时间")

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole and 0 <= section < 4:
            return self.HEAD[section]
        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 3:
                return fmt_time(self.lastModified(index).toSecsSinceEpoch(), short=True)
            if index.column() == 1:
                return "" if self.isDir(index) else human_size(self.size(index))
        return super().data(index, role)


def _Elide(text="", obj=None, mode=Qt.TextElideMode.ElideMiddle) -> ElideLabel:
    """Single-line label eliding (by default in the middle) to the available width; full text in the tooltip."""
    return ElideLabel(text, obj, mode)


# ---------------------------------------------------------------------------------------------- page
class ResultsPage(Page):
    title = "结果浏览"
    subtitle = "预览图像、JSON、CSV、Markdown、文本与 npz；文件夹显示概况与运行记录，可一键跳到对应的模拟页面。"

    def __init__(self):
        super().__init__()
        self.settings = QSettings(paths.APP_ID, paths.APP_ID)
        self.root: Path = paths.RESULTS
        self.current: Path | None = None
        self.preview: QWidget | None = None
        self.comparing = False
        self._scan: dict | None = None
        self._scan_gen = 0
        self._first = True
        self._compact: int | None = None
        self._n_match: int | None = None
        self._icons = QFileIconProvider()

        # ---------------- header: root folder + refresh
        self.header.add_action(label("根目录", "muted"))
        self.root_combo = QComboBox()
        self.root_combo.setMinimumWidth(220)
        self.root_combo.setMaximumWidth(340)
        self.root_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.root_combo.setMinimumContentsLength(24)
        self.root_combo.activated.connect(self._on_root_combo)
        self.header.add_action(self.root_combo)
        self.refresh_btn = self.header.add_action(button("刷新", self.refresh, tooltip="重新扫描文件并重新载入预览"))

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        # ---------------- left: tree / recent
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)
        top = QHBoxLayout()
        self.mode = segmented(["目录树", "最近修改"])
        self.mode.currentChanged.connect(self._on_mode)
        top.addWidget(self.mode)
        top.addStretch(1)
        ll.addLayout(top)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("筛选文件名，如 maps、*.png、summary json")
        self.filter.setClearButtonEnabled(True)
        self._filter_timer = QTimer(self, singleShot=True, interval=220)
        self._filter_timer.timeout.connect(self._apply_filter)
        self.filter.textChanged.connect(lambda *_: self._filter_timer.start())
        ll.addWidget(self.filter)

        self.stack = QStackedWidget()
        self.model = _FsModel(self)
        self.model.setReadOnly(True)
        self.model.setNameFilterDisables(False)
        self.model.setIconProvider(self._icons)
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.tree.setUniformRowHeights(True)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree.setColumnHidden(2, True)
        self.tree.setIconSize(QSize(16, 16))
        self.tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)     # run folders differ in their suffixes
        self.tree.setStyleSheet(ROW_QSS)
        hd = self.tree.header()
        hd.setStretchLastSection(False)
        hd.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hd.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hd.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.selectionModel().currentChanged.connect(self._on_tree_current)
        self.tree.installEventFilter(self)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(
            lambda pos: self._show_menu(self._tree_path(self.tree.indexAt(pos)), self.tree.viewport().mapToGlobal(pos)))
        self.stack.addWidget(self.tree)

        rec = QWidget()
        rl = QVBoxLayout(rec)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        self.recent_title = label("", "muted")
        rl.addWidget(self.recent_title)
        self.recent = QTreeWidget()
        self.recent.setHeaderLabels(["名称", "所在文件夹", "修改时间"])
        self.recent.setRootIsDecorated(False)
        self.recent.setStyleSheet(ROW_QSS)
        self.recent.setUniformRowHeights(True)
        self.recent.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        hd = self.recent.header()
        hd.setStretchLastSection(False)
        hd.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hd.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        hd.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.recent.setColumnWidth(1, 110)
        self.recent.installEventFilter(self)
        self.recent.currentItemChanged.connect(
            lambda it, _p: it is not None and it.data(0, Qt.ItemDataRole.UserRole) and
            self._preview(Path(it.data(0, Qt.ItemDataRole.UserRole))))
        self.recent.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.recent.customContextMenuRequested.connect(self._recent_menu)
        rl.addWidget(self.recent, 1)
        self.stack.addWidget(rec)
        ll.addWidget(self.stack, 1)
        self.footer = _Elide("", "muted")
        ll.addWidget(self.footer)
        left.setMinimumWidth(300)
        split.addWidget(left)

        # ---------------- right: preview header + preview
        right = QWidget()
        rr = QVBoxLayout(right)
        rr.setContentsMargins(0, 0, 0, 0)
        rr.setSpacing(10)
        head = QFrame()
        head.setObjectName("card")
        hl = QVBoxLayout(head)
        hl.setContentsMargins(14, 10, 14, 10)
        hl.setSpacing(6)
        r1 = QHBoxLayout()
        r1.setSpacing(8)
        self.p_icon = QLabel()
        self.p_icon.setFixedSize(22, 22)
        r1.addWidget(self.p_icon)
        self.p_name = _Elide("", "cardTitle", Qt.TextElideMode.ElideRight)
        self.p_name.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        r1.addWidget(self.p_name)
        self.p_badge = label("", "badge")
        r1.addWidget(self.p_badge)
        r1.addStretch(1)
        hl.addLayout(r1)
        self.p_info = _Elide("", "muted")
        hl.addWidget(self.p_info)
        r3 = QHBoxLayout()
        r3.setContentsMargins(0, 2, 0, 0)
        r3.setSpacing(8)
        self.nav_box = QWidget()
        self.nav_row = QHBoxLayout(self.nav_box)
        self.nav_row.setContentsMargins(0, 0, 0, 0)
        self.nav_row.setSpacing(8)
        r3.addWidget(self.nav_box)
        r3.addStretch(1)
        self.b_reveal = button("在访达中显示", lambda: self.current and reveal(self.current))
        self.b_open = button("用默认程序打开", lambda: self.current and open_external(self.current))
        self.b_copy = button("复制路径", lambda: self.current and self._copy(self.current),
                             tooltip="复制绝对路径（右键菜单可复制相对路径）")
        self.b_delete = button("删除", lambda: self.current and self.delete(self.current), obj="danger")
        for b in (self.b_reveal, self.b_open, self.b_copy, self.b_delete):
            r3.addWidget(b)
        hl.addLayout(r3)
        rr.addWidget(head)
        self.head = head
        head.installEventFilter(self)
        self.holder = QVBoxLayout()
        self.holder.setContentsMargins(0, 0, 0, 0)
        rr.addLayout(self.holder, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([390, 900])
        self.body.addWidget(split, 1)
        # until the user drags the handle, the list takes ~1/3 (300 px in a 1100 px window, so the preview keeps
        # enough room; a fixed 390 px squeezed folder summaries and clipped them on the right)
        self.split = split
        self._split_user = False
        split.splitterMoved.connect(lambda *_: setattr(self, "_split_user", True))
        split.installEventFilter(self)

        sc = QShortcut(QKeySequence("Ctrl+Backspace"), self.tree)
        sc.setContext(Qt.ShortcutContext.WidgetShortcut)
        sc.activated.connect(lambda: self.current and can_delete(self.current) and self.delete(self.current))

        saved = Path(str(self.settings.value("results_page/root", "") or ""))
        self.set_root(saved if str(saved) not in ("", ".") and saved.is_dir() else paths.RESULTS, preview=False)
        self.mode.setCurrentIndex(1 if self.settings.value("results_page/mode", 0, int) == 1 else 0)
        JobManager.instance().job_finished.connect(self._on_job_finished)
        ThemeManager.instance().changed.connect(self._retheme)
        self._retheme()

    # ------------------------------------------------------------------ roots
    def _custom_roots(self) -> list[str]:
        v = self.settings.value("results_page/custom_roots", []) or []
        return [v] if isinstance(v, str) else list(v)

    def _rebuild_roots(self):
        items = [(paths.RESULTS, "results/  （默认）")]
        if paths.GUI_RUNS.is_dir():
            items.append((paths.GUI_RUNS, "results/gui_runs/  （界面运行）"))
        for c in self._custom_roots():
            p = Path(c)
            if p.is_dir() and all(p.resolve() != q.resolve() for q, _ in items):
                items.append((p, _short(p)))
        if all(self.root.resolve() != q.resolve() for q, _ in items):
            items.append((self.root, _short(self.root)))
        self.root_combo.blockSignals(True)
        self.root_combo.clear()
        for p, text in items:
            self.root_combo.addItem(icons.icon("folder", ThemeManager.instance().tokens.muted), text, str(p))
            self.root_combo.setItemData(self.root_combo.count() - 1, str(p), Qt.ItemDataRole.ToolTipRole)
            if p.resolve() == self.root.resolve():
                self.root_combo.setCurrentIndex(self.root_combo.count() - 1)
        self.root_combo.insertSeparator(self.root_combo.count())
        self.root_combo.addItem("选择其他文件夹…", None)
        self.root_combo.blockSignals(False)

    def _on_root_combo(self, i):
        d = self.root_combo.itemData(i)
        if d is None:
            self.choose_root()
        else:
            self.set_root(Path(d))

    def choose_root(self):
        d = QFileDialog.getExistingDirectory(self, "选择要浏览的文件夹", str(self.root))
        if d:
            roots = [r for r in self._custom_roots() if r != d][:5]
            self.settings.setValue("results_page/custom_roots", [d] + roots)
            self.set_root(Path(d))
        else:
            self._rebuild_roots()

    def set_root(self, root, preview=True):
        """Browse `root` (tree, recent list, footer); preview=True also shows the root folder summary."""
        self.root = Path(root)
        self.model.setRootPath(str(self.root))
        self.tree.setRootIndex(self.model.index(str(self.root)))
        val = "" if self.root == paths.RESULTS else str(self.root)
        if (self.settings.value("results_page/root", "") or "") != val:
            self.settings.setValue("results_page/root", val)
        self._rebuild_roots()
        self._scan = None
        self._fill_recent()
        self.rescan()
        if preview:
            self._preview(self.root, force=True)

    # ------------------------------------------------------------------ scanning & lists
    def rescan(self):
        self._scan_gen += 1
        gen = self._scan_gen
        self.footer.setText(f"{_short(self.root)}  ·  正在扫描…")
        run_async(scan_files, self.root, on_done=lambda d, g=gen: self._scan_done(g, d),
                  on_error=lambda e, g=gen: self._scan_failed(g, e))

    def _scan_done(self, gen, d):
        if gen != self._scan_gen:
            return
        self._scan = d
        self._fill_recent()

    def _scan_failed(self, gen, err):
        if gen == self._scan_gen:
            self.footer.setText(f"扫描失败：{err.splitlines()[0]}")

    def _update_footer(self):
        d = self._scan
        if d is None:
            return
        n = len(d["files"])
        txt = f"{_short(self.root)}  ·  {n}{'+' if d['truncated'] else ''} 个文件  ·  {human_size(d['total'])}"
        if self._n_match is not None and self.mode.currentIndex() == 0:
            # QFileSystemModel filters files only: folders stay visible, so say how many files match
            txt = f"{self._n_match} 个文件匹配（文件夹不筛选）；列表见「最近修改」"
        self.footer.setText(txt)

    def _on_mode(self, i):
        self.stack.setCurrentIndex(i)
        self.settings.setValue("results_page/mode", i)
        self._update_footer()
        if i == 0 and self.current is not None:
            self._select_in_tree(self.current)

    def _apply_filter(self):
        pats = _patterns(self.filter.text())
        self.model.setNameFilters(pats)
        self._fill_recent()

    def _fill_recent(self):
        self.recent.blockSignals(True)
        self.recent.clear()
        self.recent.blockSignals(False)
        if self._scan is None:
            self.recent_title.setText("正在扫描…")
            return
        pats = _patterns(self.filter.text())
        files = self._scan["files"]
        self._n_match = None
        if pats:
            rx = _pattern_rx(tuple(pats))
            rows = [r for r, n in zip(files, self._scan["names"]) if rx.match(n)]
            self._n_match = len(rows)
            shown = rows[:MATCH_CAP]
            self.recent_title.setText(f"匹配「{self.filter.text().strip()}」的文件：{len(rows)} 个"
                                      + (f"（显示最近的 {MATCH_CAP} 个）" if len(rows) > MATCH_CAP else "")
                                      + "，按修改时间排序")
        else:
            shown = files[:RECENT_N]
            self.recent_title.setText(f"最近修改的 {len(shown)} 个文件")
        root = Path(self._scan["root"])
        self.recent.blockSignals(True)
        for fp, mtime, size in shown:
            parent = os.path.relpath(os.path.dirname(fp), root)
            it = QTreeWidgetItem([os.path.basename(fp), "" if parent == "." else parent, fmt_time(mtime, short=True)])
            it.setIcon(0, self._icons.icon(QFileInfo(fp)))
            it.setData(0, Qt.ItemDataRole.UserRole, fp)
            it.setToolTip(0, f"{paths.rel(fp)}\n{human_size(size)}  ·  {fmt_time(mtime)}")
            it.setToolTip(1, parent)
            self.recent.addTopLevelItem(it)
            if self.current is not None and Path(fp) == self.current:
                self.recent.setCurrentItem(it)
        if not shown:
            it = QTreeWidgetItem(["没有匹配的文件" if pats else "（空）", "", ""])
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.recent.addTopLevelItem(it)
        self.recent.blockSignals(False)
        self._update_footer()

    # ------------------------------------------------------------------ tree
    def _tree_path(self, idx) -> Path | None:
        return Path(self.model.filePath(idx)) if idx.isValid() else None

    def _on_tree_current(self, cur, _prev):
        p = self._tree_path(cur)
        if p is not None:
            self._preview(p)

    def _select_in_tree(self, p: Path):
        if not _inside(p, self.root) or Path(p).resolve() == self.root.resolve():
            return
        idx = self.model.index(str(p))
        if not idx.isValid():
            return
        par = idx.parent()
        while par.isValid():
            self.tree.expand(par)
            par = par.parent()
        sm = self.tree.selectionModel()
        sm.blockSignals(True)
        self.tree.setCurrentIndex(idx)
        sm.blockSignals(False)
        self.tree.viewport().update()
        # rows move while the model is still loading / sorting: scroll again a little later
        QTimer.singleShot(250, lambda s=str(p): self._scroll_to(s))

    def _scroll_to(self, s):
        idx = self.model.index(s)
        if idx.isValid():
            self.tree.scrollTo(idx, QAbstractItemView.ScrollHint.PositionAtCenter)

    # ------------------------------------------------------------------ preview
    def _preview(self, path, force=False):
        p = Path(path)
        if not force and not self.comparing and self.current is not None and p == self.current:
            return
        self.current = p
        self.comparing = False
        self._update_head(p)
        w = preview_for(p, file_actions=False)
        w.open_requested.connect(self.show_path)
        self._set_preview(w)

    def _set_preview(self, w):
        if self.preview is not None:
            self.holder.removeWidget(self.preview)
            self.preview.hide()
            self.preview.deleteLater()
        self.preview = w
        self.holder.addWidget(w, 1)

    def show_path(self, path):
        """Select `path` in the tree (or the list) and preview it — used by thumbnails, links and open()."""
        p = Path(path)
        if not _inside(p, self.root):
            if _inside(p, paths.RESULTS):
                self.set_root(paths.RESULTS, preview=False)
            else:
                d = p if p.is_dir() else p.parent
                while not d.is_dir() and d.parent != d:         # a missing path: browse its nearest existing folder
                    d = d.parent
                roots = [r for r in self._custom_roots() if r != str(d)][:5]
                self.settings.setValue("results_page/custom_roots", [str(d)] + roots)
                self.set_root(d, preview=False)
        if self.mode.currentIndex() == 0:
            self._select_in_tree(p)
        self._preview(p, force=True)

    def _update_head(self, p: Path):
        t = ThemeManager.instance().tokens
        exists = p.exists()
        self.p_icon.setPixmap(self._icons.icon(QFileInfo(str(p))).pixmap(20, 20))
        self.p_name.setText(p.name or str(p))
        self.p_badge.setText(describe(p) if exists else "不存在")
        if exists:
            st = p.stat()
            size = "文件夹" if p.is_dir() else human_size(st.st_size)
            self.p_info.setText(f"{paths.rel(p)}   ·   {size}   ·   修改于 {fmt_time(st.st_mtime)}")
        else:
            self.p_info.setText(paths.rel(p))
        for b in (self.b_reveal, self.b_open, self.b_copy):
            b.setEnabled(exists)
        ok = can_delete(p)
        self.b_delete.setEnabled(ok)
        self.b_delete.setToolTip("把它移到废纸篓（可在废纸篓中恢复）" if ok else
                                 "只能删除 results/gui_runs/ 中由界面生成的结果；提交到仓库的结果受保护")
        # page jumps (+ image comparison)
        while self.nav_row.count():
            it = self.nav_row.takeAt(0)
            if it.widget() is not None:
                it.widget().deleteLater()
        targets = nav_targets(p) if exists else []
        self._nav_btns = []
        for i, (text, pid, kw) in enumerate(targets):
            b = button(text, lambda _=False, pid=pid, kw=kw: nav.navigate(pid, **kw), obj="primary" if i == 0 else None,
                       tooltip=f"{text}：{paths.rel(next(iter(kw.values())))}")
            b.setIcon(icons.icon(TARGETS[pid][1], t.on_accent if i == 0 else t.accent))
            self.nav_row.addWidget(b)
            self._nav_btns.append((b, text, TARGETS[pid][2]))
        has_img = exists and kind_of(p) == "image"
        if has_img:
            b = button("并排对比…", self.compare_dialog, tooltip="选择另一张图像与当前图像并排对比（同步缩放）")
            b.setIcon(icons.icon("image", t.accent))
            self.nav_row.addWidget(b)
            self._nav_btns.append((b, "并排对比…", "对比…"))
        self.nav_box.setVisible(bool(targets) or has_img)
        self._compact = None
        self._fit_actions()

    # ------------------------------------------------------------------ narrow windows
    FILE_ACTIONS = (("b_reveal", "在访达中显示"), ("b_open", "用默认程序打开"), ("b_copy", "复制路径"))

    def _fit_actions(self):
        """When the header card is too narrow for the full button texts (at 1100 px they were squeezed below
        their text width): level 1 = icon-only file actions, level 2 = also short page names."""
        nav = getattr(self, "_nav_btns", [])
        fm = self.b_reveal.fontMetrics()
        w = lambda t: fm.horizontalAdvance(t) + 48               # icon + padding
        fixed = 70 + 8 * (len(nav) + 4) + 28                      # delete button, spacing, card margins
        files_full = sum(w(t) for _, t in self.FILE_ACTIONS)
        nav_full = sum(w(t) for _, t, _ in nav)
        avail = self.head.width()
        level = 0 if avail >= nav_full + files_full + fixed else (1 if avail >= nav_full + 3 * 40 + fixed else 2)
        if level == self._compact:
            return
        self._compact = level
        for attr, text in self.FILE_ACTIONS:
            b = getattr(self, attr)
            b.setText("" if level else text)
            if level:
                b.setToolTip(text if attr != "b_copy" else "复制绝对路径（右键菜单可复制相对路径）")
        for b, text, short in nav:
            b.setText(short if level == 2 else text)

    def eventFilter(self, obj, e):
        if e.type() == QEvent.Type.Resize:
            if obj is getattr(self, "head", None):
                self._fit_actions()
            elif obj is self.tree:
                w = e.size().width()                             # (the viewport is resized after this filter)
                self.tree.setColumnHidden(3, w < 370)           # names first: dates / sizes go when narrow
                self.tree.setColumnHidden(1, w < 250)
            elif obj is getattr(self, "recent", None):
                self.recent.setColumnHidden(2, e.size().width() < 370)
            elif obj is getattr(self, "split", None) and not self._split_user:
                QTimer.singleShot(0, self._fit_split)
        return super().eventFilter(obj, e)

    def _fit_split(self):
        total = self.split.width() - self.split.handleWidth()
        if self._split_user or total <= 0:
            return
        left = max(300, min(420, int(total * 0.34)))
        if abs(self.split.sizes()[0] - left) > 2:
            self.split.setSizes([left, total - left])

    # ------------------------------------------------------------------ compare
    def compare_dialog(self):
        if self.current is None:
            return
        fn, _ = QFileDialog.getOpenFileName(self, "选择要对比的图像", str(self.current.parent),
                                            "图像 (" + " ".join(f"*{e}" for e in sorted(IMAGE_EXT)) + ")")
        if fn:
            self.compare(self.current, Path(fn))

    def compare(self, a, b):
        """Show images a and b side by side (the header keeps acting on a)."""
        a, b = Path(a), Path(b)
        self.current = a
        self._update_head(a)
        same = a.name == b.name
        self.p_name.setText(f"{a.parent.name + '/' if same else ''}{a.name}  ↔  {b.parent.name + '/' if same else ''}{b.name}")
        self.nav_box.setVisible(False)
        w = CompareView(a, b)
        w.closed.connect(lambda: self._preview(self.current, force=True))
        self._set_preview(w)
        self.comparing = True

    # ------------------------------------------------------------------ actions
    def _copy(self, p: Path, relative=False):
        QGuiApplication.clipboard().setText(paths.rel(p) if relative else str(p))

    def delete(self, path) -> bool:
        """Move a GUI run (or a file in it) to the trash after confirmation. Only inside results/gui_runs/."""
        p = Path(path)
        if not can_delete(p):
            error_box(self, "不能删除", "只能删除 results/gui_runs/ 中由界面生成的结果。\n"
                                        "提交到仓库的结果文件夹受保护，请在访达中自行处理。")
            return False
        rp = p.resolve()
        for j in JobManager.instance().active():
            if j.out:
                o = paths.absolute(j.out).resolve()
                if o == rp or rp in o.parents or o in rp.parents:
                    error_box(self, "正在使用", f"任务「{j.title}」正在写入此文件夹，请先等待它结束或取消它。")
                    return False
        if p.is_dir():
            n = size = 0
            for d, _, fs in os.walk(p):
                for f in fs:
                    n += 1
                    try:
                        size += os.lstat(os.path.join(d, f)).st_size       # broken links must not abort
                    except OSError:
                        pass
            what = f"文件夹（{n} 个文件，{human_size(size)}）"
        else:
            what = f"文件（{human_size(p.lstat().st_size)}）"
        r = QMessageBox.question(self, "移到废纸篓？", f"将{what}\n\n{paths.rel(p)}\n\n移到废纸篓？之后可在废纸篓中恢复。",
                                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                 QMessageBox.StandardButton.No)
        if r != QMessageBox.StandardButton.Yes:
            return False
        res = QFile.moveToTrash(str(p))
        ok = res[0] if isinstance(res, tuple) else bool(res)
        if not ok:
            error_box(self, "删除失败", f"无法把 {paths.rel(p)} 移到废纸篓。")
            return False
        if _inside(self.root, p):            # the browsed root itself went to the trash
            self.set_root(p.parent if p.parent.is_dir() else paths.RESULTS, preview=False)
        if self.current is not None and (self.current == p or p.resolve() in self.current.resolve().parents):
            nxt = p.parent if p.parent.exists() else self.root
            if self.mode.currentIndex() == 0:
                self._select_in_tree(nxt)
            self._preview(nxt, force=True)
        self._rebuild_roots()
        self.rescan()
        return True

    def build_menu(self, path: Path | None) -> QMenu:
        m = QMenu(self)
        m.setToolTipsVisible(True)
        if path is not None:
            for text, pid, kw in nav_targets(path):
                m.addAction(text, lambda pid=pid, kw=kw: nav.navigate(pid, **kw))
            if m.actions():
                m.addSeparator()
            m.addAction("在访达中显示", lambda: reveal(path))
            m.addAction("用默认程序打开", lambda: open_external(path))
            m.addAction("复制路径", lambda: self._copy(path))
            m.addAction("复制相对路径", lambda: self._copy(path, True))
            if (kind_of(path) == "image" and self.current is not None and self.current != path
                    and self.current.exists() and kind_of(self.current) == "image"):
                m.addAction(f"与 {self.current.name} 并排对比", lambda: self.compare(self.current, path))
            m.addSeparator()
            a = m.addAction("删除（移到废纸篓）…", lambda: self.delete(path))
            a.setEnabled(can_delete(path))
            if not a.isEnabled():
                a.setToolTip("只能删除 results/gui_runs/ 中由界面生成的结果")
            m.addSeparator()
        else:
            m.addAction("在访达中显示根目录", lambda: reveal(self.root))
            m.addAction("显示根目录概况", lambda: self._preview(self.root, force=True))
        m.addAction("刷新", self.refresh)
        return m

    def _show_menu(self, path, global_pos):
        self.build_menu(path).exec(global_pos)

    def _recent_menu(self, pos):
        it = self.recent.itemAt(pos)
        fp = it.data(0, Qt.ItemDataRole.UserRole) if it is not None else None
        self._show_menu(Path(fp) if fp else None, self.recent.viewport().mapToGlobal(pos))

    def refresh(self):
        self._rebuild_roots()
        self.rescan()
        if self.current is not None:
            p = self.current
            while not p.exists() and p != self.root and p.parent != p:
                p = p.parent
            self._preview(p, force=True)

    def _on_job_finished(self, job):
        if not self.isVisible():
            self._scan = None if self._scan is None else dict(self._scan, t=0)     # rescan on next show
            return
        self._rebuild_roots()
        self.rescan()
        if job.out and self.current is not None and self.current.is_dir() and not self.comparing:
            if _inside(paths.absolute(job.out), self.current):
                self._preview(self.current, force=True)

    def _retheme(self, *_):
        t = ThemeManager.instance().tokens
        self.refresh_btn.setIcon(icons.icon("refresh", t.muted))
        self.b_reveal.setIcon(icons.icon("folder", t.muted))
        self.b_open.setIcon(icons.icon("open", t.muted))
        self.b_copy.setIcon(icons.icon("file", t.muted))
        for i in range(self.root_combo.count()):
            if self.root_combo.itemData(i) is not None:
                self.root_combo.setItemIcon(i, icons.icon("folder", t.muted))
        if self.current is not None and not self.comparing:
            self._update_head(self.current)

    # ------------------------------------------------------------------ page protocol
    def on_show(self):
        if self._first:
            self._first = False
            if self.current is None:
                self._preview(self.root)
        if self._scan is None or time.time() - self._scan.get("t", 0) > 5:
            self.rescan()
        self._rebuild_roots()

    def open(self, path=None, **_):
        if path:
            self._first = False
            self.show_path(paths.absolute(path))
