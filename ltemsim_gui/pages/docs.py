"""Documentation: README, model notes, result summary tables, a parameter reference and the literature sources."""
from __future__ import annotations

import html
import re
from pathlib import Path

from PySide6.QtCore import QEvent, QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import (QColor, QDesktopServices, QGuiApplication, QImageReader, QKeySequence, QShortcut,
                           QTextBlockFormat, QTextCharFormat, QTextCursor, QTextDocument, QTextFormat, QTextFrameFormat,
                           QTextTable)
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLineEdit, QSizePolicy, QSplitter,
                               QStackedWidget, QTableWidget, QTableWidgetItem, QTextBrowser, QTextEdit, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import icons, nav, paths
from ..theme import ThemeManager
from ..widgets.common import Page, button, label, open_external, reveal
from ..workers import run_async
from .home import Elided

ROLE = Qt.ItemDataRole.UserRole
PARAMS, SOURCES = ":params", ":sources"
FONT_MIN, FONT_MAX, FONT_DEFAULT = 11, 22, 14
MAX_TEXT_W = 920                      # comfortable reading width of the markdown column

# repo-relative path -> Chinese title (others get "folder / stem")
TITLES = {
    "README.md": "项目说明 README",
    "LICENSE": "许可证 LICENSE",
    "docs/model.md": "模型与方程",
    "docs/pulse_chain.md": "单脉冲发射链",
    "docs/reference/pizzuto_fdm_demod_reference.m": "Pizzuto FDM 解调参考代码",
    "results/coarse_scenarios/contact_scenarios.md": "接触情景（粗网格）",
    "results/coarse_scenarios/lifetime_scenarios.md": "寿命情景（粗网格）",
    "results/coarse_scenarios/power_sweep.md": "功率扫描（粗网格）",
    "results/field_bending/summary.md": "场弯曲汇总",
    "results/pulse_chain/summary_table.md": "单脉冲链汇总表",
}
SOURCE_TERMS = {"McGregor1994": ("McGregor",), "StoutKushner1996": ("Stout",), "Pizzuto2023": ("Pizzuto",),
                "RalphGrischkowsky1991": ("Ralph", "Grischkowsky"), "Murakami2016": ("Murakami",),
                "memory": ("from memory", "Mitonneau", "Chantre")}
UNIT_PRETTY = (("um", "µm"), ("cm^-3", "cm⁻³"), ("cm^-1", "cm⁻¹"), ("cm^2/Vs", "cm²/Vs"), ("cm^3/s", "cm³/s"),
               ("cm^2", "cm²"), ("deg", "°"))
MONO = ["Menlo", "SF Mono", "Monaco", "Courier New"]
PULSE_CLI = {"P_avg_mW", "T", "V_bias", "w_um", "x_fine", "t_end_ps"}     # PulseConfig fields run_pulse_chain.py accepts


def pretty_unit(u: str) -> str:
    for a, b in UNIT_PRETTY:
        if u == a:
            return b
    return u.replace("^-3", "⁻³").replace("^-1", "⁻¹").replace("^2", "²").replace("^3", "³")


def doc_title(p: Path) -> str:
    rel = paths.rel(p)
    if rel in TITLES:
        return TITLES[rel]
    if p.parent.name and p.parent != paths.ROOT:
        return f"{p.parent.name} / {p.stem}"
    return p.name


def discover() -> list[tuple[str, list[Path]]]:
    """(group title, files) of the documents shipped with the repository."""
    R = paths.ROOT
    project = [p for p in (R / "README.md", R / "LICENSE") if p.exists()]
    docs = sorted(paths.DOCS.glob("*.md")) if paths.DOCS.exists() else []
    ref = sorted(p for p in (paths.DOCS / "reference").glob("*") if p.is_file()) if (paths.DOCS / "reference").exists() else []
    results = sorted(p for p in paths.RESULTS.rglob("*.md") if paths.GUI_RUNS not in p.parents) if paths.RESULTS.exists() else []
    return [("项目", project), ("模型文档", docs + ref), ("结果摘要", results)]


_specs_cache = None


def all_specs():
    """[(model label, FieldSpec)] for the steady Params and the single-pulse PulseConfig (parsed once)."""
    global _specs_cache
    if _specs_cache is None:
        from ..paramspec import params_specs, pulse_specs
        _specs_cache = [("稳态 Params", s) for s in params_specs()] + [("单脉冲 PulseConfig", s) for s in pulse_specs()]
    return _specs_cache


def sources_markdown() -> str:
    try:
        from ltemsim.params import SOURCES
    except Exception as e:         # keep the page usable if the package cannot be imported
        return f"# 文献来源\n\n无法读取 `ltemsim.params.SOURCES`：{e}"
    specs = all_specs()
    out = ["# 文献来源", "",
           "`ltemsim/params.py` 中 `SOURCES` 字典列出的参数来源。参数默认值的行内注释引用了这些文献；"
           "每条下面列出注释中提到它的参数（详见「参数速查」的说明列）。", ""]
    for key, text in SOURCES.items():
        terms = SOURCE_TERMS.get(key, (re.sub(r"\d+", "", key),))
        cited = [s.name for _, s in specs if any(t.lower() in s.help.lower() for t in terms)]
        cited = list(dict.fromkeys(cited))
        out += [f"## {key}", "", text, ""]
        if cited:
            out += ["**引用此来源的参数：** " + ", ".join(f"`{c}`" for c in cited), ""]
    return "\n".join(out)


_CJK_BREAK = re.compile(r"(?<=[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef])\n(?=[\u4e00-\u9fff\uff08\u201c])")


def join_cjk_lines(md: str) -> str:
    """Hard-wrapped Chinese prose: drop the soft line break between two CJK characters (Markdown makes it a space)."""
    return _CJK_BREAK.sub("", md)


def _frames(frame):
    for f in frame.childFrames():
        yield f
        yield from _frames(f)


def polish_document(doc: QTextDocument, t, px: int, folder: Path | None = None, max_img_w: int = MAX_TEXT_W):
    """Give the markdown that Qt imports a readable look: spacing, line height, code, table and image sizing."""
    cur = QTextCursor(doc)
    cur.beginEditBlock()
    for fr in _frames(doc.rootFrame()):
        if isinstance(fr, QTextTable):
            tf = fr.format()
            tf.setBorder(1)
            tf.setBorderBrush(QColor(t.border))
            tf.setBorderStyle(QTextFrameFormat.BorderStyle.BorderStyle_Solid)
            tf.setCellPadding(6)
            tf.setCellSpacing(0)
            tf.setBorderCollapse(True)
            tf.setTopMargin(6)
            tf.setBottomMargin(14)
            fr.setFormat(tf)
            for c in range(fr.columns()):
                cell = fr.cellAt(0, c)
                cf = cell.format().toTableCellFormat()
                cf.setBackground(QColor(t.surface2))
                cell.setFormat(cf)
    mono_px = max(9, px - 1)
    inline, images = [], []
    block = doc.begin()
    while block.isValid():
        bf = block.blockFormat()
        bc = QTextCursor(block)
        nf = QTextBlockFormat()
        is_code = bf.nonBreakableLines() or bf.hasProperty(QTextFormat.Property.BlockCodeFence)
        if is_code:
            prev, nxt = block.previous(), block.next()
            nf.setBackground(QColor(t.surface2))
            nf.setLeftMargin(0)
            nf.setTopMargin(0 if prev.isValid() and prev.blockFormat().nonBreakableLines() else 4)
            nf.setBottomMargin(0 if nxt.isValid() and nxt.blockFormat().nonBreakableLines() else 12)
            bc.mergeBlockFormat(nf)
            bc.movePosition(QTextCursor.MoveOperation.StartOfBlock)
            bc.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
            cf = QTextCharFormat()
            cf.setFontFamilies(MONO)
            cf.setProperty(QTextFormat.Property.FontPixelSize, mono_px)
            bc.mergeCharFormat(cf)
        else:
            level = bf.headingLevel()
            if level:
                nf.setTopMargin({1: 4, 2: 22, 3: 16}.get(level, 12))
                nf.setBottomMargin(8)
                bc.mergeBlockFormat(nf)
            elif bc.currentTable() is None and "\ufffc" not in block.text():     # U+FFFC: an inline image
                nf.setLineHeight(140, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
                nf.setBottomMargin(3 if block.textList() is not None else 10)
                bc.mergeBlockFormat(nf)
            it = block.begin()
            while not it.atEnd():
                f = it.fragment()
                if f.isValid() and f.charFormat().isImageFormat():
                    images.append((f.position(), f.length(), f.charFormat().toImageFormat()))
                elif f.isValid() and f.charFormat().fontFixedPitch():
                    inline.append((f.position(), f.position() + f.length()))
                it += 1
        block = block.next()
    icf = QTextCharFormat()
    icf.setFontFamilies(MONO)
    icf.setProperty(QTextFormat.Property.FontPixelSize, mono_px)
    icf.setBackground(QColor(t.surface2))
    icf.setForeground(QColor(t.accent))
    for a, b in inline:
        c = QTextCursor(doc)
        c.setPosition(a)
        c.setPosition(b, QTextCursor.MoveMode.KeepAnchor)
        c.mergeCharFormat(icf)
    _fit(doc, images, folder, max_img_w)
    cur.endEditBlock()


def image_file(name: str, folder: Path | None) -> Path | None:
    """Local file of a markdown image reference, resolved against the document folder (as QTextBrowser does)."""
    url = QUrl(name)
    if url.isLocalFile():
        return Path(url.toLocalFile())
    if url.scheme():
        return None                                      # http(s) images are not fetched
    return Path(QUrl.fromLocalFile(str(folder or paths.ROOT) + "/").resolved(url).toLocalFile())


def image_fragments(doc: QTextDocument) -> list:
    out = []
    block = doc.begin()
    while block.isValid():
        it = block.begin()
        while not it.atEnd():
            f = it.fragment()
            if f.isValid() and f.charFormat().isImageFormat():
                out.append((f.position(), f.length(), f.charFormat().toImageFormat()))
            it += 1
        block = block.next()
    return out


def _fit(doc, images, folder, max_w):
    """Scale images down to the reading width (keeping the aspect ratio); never enlarge them."""
    for pos, n, imf in images:
        src = image_file(imf.name(), folder)
        size = QImageReader(str(src)).size() if src is not None else None
        if size is None or not size.isValid() or size.width() <= 0:
            continue
        w = min(size.width(), max_w)
        if abs(imf.width() - w) < 0.5:
            continue
        imf.setWidth(w)
        imf.setHeight(size.height() * w / size.width())
        c = QTextCursor(doc)
        c.setPosition(pos)
        c.setPosition(pos + n, QTextCursor.MoveMode.KeepAnchor)
        c.setCharFormat(imf)


def fit_images(doc: QTextDocument, folder: Path | None, max_w: int) -> int:
    """Re-fit all images to a new reading width (window resized); returns the number of images."""
    images = image_fragments(doc)
    if images:
        cur = QTextCursor(doc)
        cur.beginEditBlock()
        _fit(doc, images, folder, max_w)
        cur.endEditBlock()
    return len(images)


# ----------------------------------------------------------------------------- reader
class Reader(QTextBrowser):
    """QTextBrowser with a centred reading column and its own font size."""

    def __init__(self):
        super().__init__()
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.document().setUndoRedoEnabled(False)
        self.font_px = FONT_DEFAULT
        self.folder: Path | None = None
        self.has_images = False
        self._img_w = 0
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.setInterval(120)
        self._fit_timer.timeout.connect(self.refit_images)

    def image_width(self) -> int:
        return max(240, min(MAX_TEXT_W, self.viewport().width() - 64))

    def refit_images(self):
        if self.has_images and abs(self.image_width() - self._img_w) > 4:
            self._img_w = self.image_width()
            fit_images(self.document(), self.folder, self._img_w)

    def set_font_px(self, px: int):
        self.font_px = max(FONT_MIN, min(FONT_MAX, int(px)))
        self.setStyleSheet(f"QTextBrowser {{ font-size: {self.font_px}px; }}")
        self._margins()

    def _margins(self):
        doc = self.document()
        side = max(24, (self.viewport().width() - MAX_TEXT_W) // 2)
        fmt = doc.rootFrame().frameFormat()
        if abs(fmt.leftMargin() - side) > 0.5 or fmt.topMargin() != 20:
            fmt.setLeftMargin(side)
            fmt.setRightMargin(side)
            fmt.setTopMargin(20)
            fmt.setBottomMargin(28)
            doc.rootFrame().setFrameFormat(fmt)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._margins()
        if self.has_images:
            self._fit_timer.start()


class ParamsView(QWidget):
    """Filterable table of every Params / PulseConfig field."""
    COLS = ("模型", "分组", "级别", "参数", "中文名", "默认值", "单位", "说明")
    WIDTHS = (52, 108, 46, 140, 112, 100, 60)
    OPTIONAL = (1, 2)                     # 分组 / 级别: hidden when the 说明 column would get too narrow
    C_ZH, C_HELP = 4, 7                   # narrower still: 说明 moves into a line under the table
    HINT = ("双击一行复制 name=默认值：稳态标量参数可直接用作 python scripts/run_baseline.py 的命令行覆盖（如 V_bias=-25），"
            "run_pulse_chain.py 接受的单脉冲参数同理；其余参数复制为 Python 关键字参数形式。")

    def __init__(self):
        super().__init__()
        self.loaded = False
        self.narrow = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        bar = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("筛选参数：名称、中文名、单位、分组或说明…  (⌘F)")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self.apply_filter)
        bar.addWidget(self.filter, 1)
        self.model = QComboBox()
        self.model.addItems(["全部模型", "稳态 Params", "单脉冲 PulseConfig"])
        self.model.currentIndexChanged.connect(self.apply_filter)
        bar.addWidget(self.model)
        self.count = label("", "muted")
        self.count.setMinimumWidth(96)
        self.count.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bar.addWidget(self.count)
        lay.addLayout(bar)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(True)
        self.table.setShowGrid(False)
        self.table.cellDoubleClicked.connect(self._copy_row)
        h = self.table.horizontalHeader()
        for c, w in enumerate(self.WIDTHS):
            self.table.setColumnWidth(c, w)
            h.setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)
        h.setSectionResizeMode(len(self.COLS) - 1, QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.table, 1)
        self.detail = label("", None, wrap=True, selectable=True)
        self.detail.setTextFormat(Qt.TextFormat.RichText)
        self.detail.hide()
        lay.addWidget(self.detail)
        self.table.currentCellChanged.connect(lambda *_: self._update_detail())
        self.hint = label(self.HINT, "muted", wrap=True)
        lay.addWidget(self.hint)
        self._hint_timer = QTimer(self)
        self._hint_timer.setSingleShot(True)
        self._hint_timer.setInterval(4000)
        self._hint_timer.timeout.connect(lambda: self.hint.setText(self.HINT))
        # word-wrapped rows must be re-measured when the 说明 column changes width
        self._rows_timer = QTimer(self)
        self._rows_timer.setSingleShot(True)
        self._rows_timer.setInterval(100)
        self._rows_timer.timeout.connect(self.table.resizeRowsToContents)
        h.sectionResized.connect(lambda c, *_: c in (self.C_ZH, self.C_HELP) and self.loaded and self._rows_timer.start())
        self.table.viewport().installEventFilter(self)

    def eventFilter(self, obj, e):
        if obj is self.table.viewport() and e.type() == QEvent.Type.Resize:
            self._fit_columns()
        return super().eventFilter(obj, e)

    def _fit_columns(self):
        vw = self.table.viewport().width()
        base = sum(self.WIDTHS)
        for c in self.OPTIONAL:
            self.table.setColumnHidden(c, vw < base + 220)
        narrow = vw < base - sum(self.WIDTHS[c] for c in self.OPTIONAL) + 200
        if narrow != self.narrow:
            self.narrow = narrow
            h = self.table.horizontalHeader()
            self.table.setColumnHidden(self.C_HELP, narrow)
            h.setSectionResizeMode(self.C_ZH, QHeaderView.ResizeMode.Stretch if narrow else QHeaderView.ResizeMode.Interactive)
            if not narrow:
                self.table.setColumnWidth(self.C_ZH, self.WIDTHS[self.C_ZH])
            self.detail.setVisible(narrow)
            self._update_detail()
            if self.loaded:
                self._rows_timer.start()

    def _update_detail(self):
        if not self.narrow:
            return
        r = self.table.currentRow()
        it = self.table.item(r, 0) if r >= 0 else None
        if it is None:
            self.detail.setText(f"<span style='color:{ThemeManager.instance().tokens.muted}'>选择一行查看参数说明</span>")
            return
        cell = lambda c: html.escape(self.table.item(r, c).text()) if self.table.item(r, c) else ""
        unit = f" {cell(6)}" if cell(6) else ""
        self.detail.setText(f"<b>{cell(3)}</b>（{cell(4) or '—'}，{cell(1)}）= {cell(5)}{unit}"
                            f"<br>{cell(7) or '（源码中没有说明）'}")

    def load(self):
        if self.loaded:
            return
        from ..paramspec import format_value
        specs = all_specs()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(specs))
        mono = self.table.font()
        mono.setFamilies(MONO)
        muted = ThemeManager.instance().tokens.muted
        for r, (model, s) in enumerate(specs):
            try:
                default = format_value(s, s.default)
            except Exception:
                default = str(s.default)
            label_zh = s.label if s.label != s.name else ""
            if label_zh.endswith(s.name):
                label_zh = label_zh[: -len(s.name)].strip()
            shown = default
            if s.type == "tuple" and isinstance(s.default, (tuple, list)) and len(s.default) > 6:
                parts = default.split(", ")
                shown = f"{', '.join(parts[:3])}, …, {parts[-1]}（{len(parts)} 个）"
            vals = (model.split()[0], s.section, "高级" if s.advanced else "基本", s.name, label_zh, shown,
                    pretty_unit(s.unit), s.help)
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c in (3, 5):
                    it.setFont(mono)
                if c == 2 and s.advanced:
                    it.setForeground(QColor(muted))
                it.setToolTip(f"{model} · {s.name} = {default} {pretty_unit(s.unit)}\n{s.help}".strip())
                self.table.setItem(r, c, it)
            self.table.item(r, 0).setData(ROLE, (s.name, copy_text(model, s, default)))
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
        self.loaded = True
        self._fit_columns()
        self.table.resizeRowsToContents()
        self.apply_filter()

    def apply_filter(self, *_):
        q = self.filter.text().strip().lower().split()
        m = self.model.currentIndex()
        want = {0: None, 1: "稳态", 2: "单脉冲"}[m]
        shown = 0
        for r in range(self.table.rowCount()):
            cells = [self.table.item(r, c).text() if self.table.item(r, c) else "" for c in range(len(self.COLS))]
            ok = (want is None or cells[0] == want) and all(w in " ".join(cells).lower() for w in q)
            self.table.setRowHidden(r, not ok)
            shown += ok
        self.count.setText(f"{shown} / {self.table.rowCount()} 个参数")

    def _copy_row(self, row, _col):
        d = self.table.item(row, 0).data(ROLE)
        if d:
            name, (text, usage) = d
            QGuiApplication.clipboard().setText(text)
            short = text if len(text) <= 60 else text[:57] + "…"
            self.hint.setText(f"已复制：{short}　{usage}")
            self._hint_timer.start()


def copy_text(model: str, s, formatted: str) -> tuple[str, str]:
    """(clipboard text, usage note) for a parameter row.

    Steady scalars and the PulseConfig fields run_pulse_chain.py parses use the command-line form (name=value); the
    rest (tuples, other PulseConfig fields) cannot be given on a command line and are copied as name=repr(value)."""
    if model.startswith("稳态") and s.type != "tuple":
        return f"{s.name}={formatted.replace(' ', '')}", "可直接用于 python scripts/run_baseline.py 的命令行覆盖"
    if s.name in PULSE_CLI:
        return f"{s.name}={formatted.replace(' ', '')}", "可直接用于 python scripts/run_pulse_chain.py 的命令行"
    v = tuple(s.default) if isinstance(s.default, list) else s.default
    where = "Params(...) / run_baseline.main(...)" if model.startswith("稳态") else "PulseConfig(...)"
    return f"{s.name}={v!r}", f"Python 关键字参数形式，用于 {where}（命令行不支持此参数）"


# ----------------------------------------------------------------------------- page
class DocsPage(Page):
    title = "文档"
    subtitle = "项目说明、模型与方程、结果摘要表，以及由源码生成的参数速查与文献来源。"

    def __init__(self):
        super().__init__()
        self.settings = QSettings(paths.APP_ID, paths.APP_ID)
        self.current: str | None = None
        self.matches: list[QTextCursor] = []
        self.match_i = -1
        self._items: dict[str, QTreeWidgetItem] = {}
        self._other: QTreeWidgetItem | None = None

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        # ---------------- left: document list
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setIndentation(10)
        self.tree.setIconSize(self.tree.iconSize())
        self.tree.setStyleSheet("QTreeView::item { padding: 5px 4px; }")
        self.tree.currentItemChanged.connect(self._on_select)
        ll.addWidget(self.tree, 1)
        rb = button("刷新列表", self.refresh_list, "flat", "重新扫描 docs/ 与 results/ 中的 Markdown 文件")
        ll.addWidget(rb, 0, Qt.AlignmentFlag.AlignLeft)
        left.setMinimumWidth(220)
        left.setMaximumWidth(380)
        split.addWidget(left)

        # ---------------- right: viewer
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)
        # row 1: document title + path; row 2: find bar (left) and view / file actions (right)
        trow = QHBoxLayout()
        trow.setSpacing(10)
        self.doc_title = Elided("", "cardTitle", Qt.TextElideMode.ElideRight)
        self.doc_title.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.doc_path = Elided("", "muted")
        trow.addWidget(self.doc_title)
        trow.addWidget(self.doc_path, 1)
        rl.addLayout(trow)
        top = QHBoxLayout()
        top.setSpacing(6)
        self.find_box = QWidget()
        fb = QHBoxLayout(self.find_box)
        fb.setContentsMargins(0, 0, 0, 0)
        fb.setSpacing(4)
        self.find = QLineEdit()
        self.find.setPlaceholderText("查找  ⌘F")
        self.find.setClearButtonEnabled(True)
        self.find.setFixedWidth(200)
        self.find.textChanged.connect(self._search)
        self.find.returnPressed.connect(self.find_next)
        self.find.installEventFilter(self)
        fb.addWidget(self.find)
        self.btn_prev = button("↑", self.find_prev, "flat", "上一个  ⇧⌘G / ⇧↩")
        self.btn_next = button("↓", self.find_next, "flat", "下一个  ⌘G / ↩")
        fb.addWidget(self.btn_prev)
        fb.addWidget(self.btn_next)
        self.find_count = label("", "muted")
        self.find_count.setMinimumWidth(56)
        fb.addWidget(self.find_count)
        top.addWidget(self.find_box)
        top.addStretch(1)
        self.font_box = QWidget()
        zb = QHBoxLayout(self.font_box)
        zb.setContentsMargins(8, 0, 0, 0)
        zb.setSpacing(0)
        zb.addWidget(button("A−", lambda: self.zoom(-1), "flat", "缩小字号  ⌘−"))
        self.font_lbl = label("", "muted")
        zb.addWidget(self.font_lbl)
        zb.addWidget(button("A+", lambda: self.zoom(+1), "flat", "放大字号  ⌘+"))
        top.addWidget(self.font_box)
        self.btn_reveal = button("在访达中显示", lambda: self.current and Path(self.current).exists() and reveal(self.current),
                                 "flat", "在访达中显示此文件")
        self.btn_ext = button("用默认程序打开", lambda: self.current and Path(self.current).exists() and open_external(self.current),
                              "flat", "用默认程序打开此文件")
        self._compact = None
        top.addWidget(self.btn_reveal)
        top.addWidget(self.btn_ext)
        rl.addLayout(top)

        self.stack = QStackedWidget()
        self.reader = Reader()
        self.reader.anchorClicked.connect(self._on_link)
        self.stack.addWidget(self.reader)
        self.params = ParamsView()
        self.stack.addWidget(self.params)
        rl.addWidget(self.stack, 1)
        split.addWidget(right)
        split.setSizes([260, 1000])
        split.setStretchFactor(1, 1)
        self.body.addWidget(split, 1)

        try:
            px = int(self.settings.value("docs/font_px", FONT_DEFAULT))
        except (TypeError, ValueError):
            px = FONT_DEFAULT
        self.reader.set_font_px(px)
        self.font_lbl.setText(f"{self.reader.font_px}px")

        ctx = Qt.ShortcutContext.WidgetWithChildrenShortcut
        for seq, slot in ((QKeySequence.StandardKey.Find, self.focus_find),
                          (QKeySequence.StandardKey.FindNext, self.find_next),
                          (QKeySequence.StandardKey.FindPrevious, self.find_prev),
                          (QKeySequence.StandardKey.ZoomIn, lambda: self.zoom(+1)),
                          (QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Equal), lambda: self.zoom(+1)),
                          (QKeySequence.StandardKey.ZoomOut, lambda: self.zoom(-1))):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(ctx)
            sc.activated.connect(slot)

        self.refresh_list()
        ThemeManager.instance().changed.connect(self._retheme)
        self._retheme()
        self._initial = True

    # ------------------------------------------------------------------ list
    def refresh_list(self):
        keep = self.current
        self.tree.blockSignals(True)
        self.tree.clear()
        self._items = {}
        self._other = None
        t = ThemeManager.instance().tokens
        for group, files in discover() + [("速查", [])]:
            g = QTreeWidgetItem([group])
            g.setFlags(Qt.ItemFlag.ItemIsEnabled)
            f = g.font(0)
            f.setBold(True)
            g.setFont(0, f)
            g.setForeground(0, QColor(t.muted))
            self.tree.addTopLevelItem(g)
            for p in files:
                self._add_doc(g, str(p), doc_title(p), "book" if p.suffix == ".md" else "file", paths.rel(p))
            if group == "速查":
                self._add_doc(g, PARAMS, "参数速查", "table", "由 ltemsim/params.py 与 pulse_dynamics.PulseConfig 生成")
                self._add_doc(g, SOURCES, "文献来源", "list", "ltemsim.params.SOURCES")
            g.setExpanded(True)
        self.tree.blockSignals(False)
        self._retheme_icons()
        if keep and keep in self._items:
            self.tree.setCurrentItem(self._items[keep])

    def _add_doc(self, group, key, title, icon, tip):
        it = QTreeWidgetItem([title])
        it.setData(0, ROLE, key)
        it.setData(0, ROLE + 1, icon)
        it.setToolTip(0, tip)
        group.addChild(it)
        self._items[key] = it
        return it

    def _retheme_icons(self):
        t = ThemeManager.instance().tokens
        for it in self._items.values():
            it.setIcon(0, icons.icon(it.data(0, ROLE + 1), t.muted, t.accent, 16))

    def _retheme(self, *_):
        t = ThemeManager.instance().tokens
        for i in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(i).setForeground(0, QColor(t.muted))
        self._retheme_icons()
        self.btn_reveal.setIcon(icons.icon("folder", t.muted))
        self.btn_ext.setIcon(icons.icon("open", t.muted))
        self._rerender()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        compact = self.width() < 1000           # ~1100 px windows: icon-only buttons
        if compact != self._compact:
            self._compact = compact
            self.btn_reveal.setText("" if compact else "在访达中显示")
            self.btn_ext.setText("" if compact else "用默认程序打开")

    # ------------------------------------------------------------------ showing documents
    def _on_select(self, cur, _prev):
        key = cur.data(0, ROLE) if cur is not None else None
        if key:
            self.show_doc(key)

    def show_doc(self, key: str):
        self.current = key
        self.settings.setValue("docs/last", key)
        if key == PARAMS:
            self.stack.setCurrentWidget(self.params)
            QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.params.load()
            finally:
                QGuiApplication.restoreOverrideCursor()
            self.doc_title.setText("参数速查")
            self.doc_title.setToolTip("")
            self.doc_path.setToolTip("")
            self.doc_path.setText("Params（ltemsim/params.py）＋ PulseConfig（ltemsim/pulse_dynamics.py）")
            self._mode(params=True, file=False)
            return
        self.stack.setCurrentWidget(self.reader)
        if key == SOURCES:
            QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                md = sources_markdown()
            finally:
                QGuiApplication.restoreOverrideCursor()
            self._set_markdown(md, paths.ROOT / "ltemsim")
            self.doc_title.setText("文献来源")
            self.doc_title.setToolTip("")
            self.doc_path.setToolTip("")
            self.doc_path.setText("ltemsim/params.py · SOURCES")
            self._mode(params=False, file=False)
        else:
            p = Path(key)
            self.doc_title.setText(doc_title(p))
            self.doc_title.setToolTip(doc_title(p))
            self.doc_path.setText(paths.rel(p))
            self.doc_path.setToolTip(str(p))
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                self._set_markdown(f"# 无法打开文档\n\n`{paths.rel(p)}`\n\n{e}", p.parent)
            else:
                if p.suffix.lower() in (".md", ".markdown"):
                    self._set_markdown(join_cjk_lines(text), p.parent)
                else:
                    lang = {".m": "matlab", ".py": "python", ".json": "json"}.get(p.suffix.lower(), "")
                    fence = "~~~~" if "```" in text else "```"
                    self._set_markdown(f"{fence}{lang}\n{text}\n{fence}\n", p.parent)
            self._mode(params=False, file=True)
        self._search(self.find.text())

    def _set_markdown(self, text: str, folder: Path, keep_scroll=False):
        r = self.reader
        pos = r.verticalScrollBar().value() if keep_scroll else 0
        self._md = (text, folder)
        r.setSearchPaths([str(folder), str(paths.ROOT)])
        r.document().setBaseUrl(QUrl.fromLocalFile(str(folder) + "/"))
        r.setMarkdown(text)
        r.document().setUndoRedoEnabled(False)
        r.folder, r._img_w = folder, r.image_width()
        polish_document(r.document(), ThemeManager.instance().tokens, r.font_px, folder, r._img_w)
        r.has_images = bool(image_fragments(r.document()))
        r._margins()
        r.verticalScrollBar().setValue(pos)

    def _rerender(self):
        """Re-apply theme colours / font size to the shown markdown, keeping the scroll position and search."""
        if self.stack.currentWidget() is self.reader and getattr(self, "_md", None):
            self._set_markdown(*self._md, keep_scroll=True)
            if self.find.text():
                self._search(self.find.text(), keep_index=True)

    def _mode(self, params: bool, file: bool):
        self.find_box.setVisible(not params)
        self.font_box.setVisible(not params)
        self.btn_reveal.setVisible(file)
        self.btn_ext.setVisible(file)

    def _on_link(self, url: QUrl):
        if url.scheme() in ("http", "https", "mailto"):
            QDesktopServices.openUrl(url)
            return
        if not url.path() and url.fragment():
            self.reader.scrollToAnchor(url.fragment())
            return
        base = Path(self.current).parent if self.current and not self.current.startswith(":") else paths.ROOT
        target = Path(url.toLocalFile()) if url.isLocalFile() else (base / url.path())
        try:
            target = target.resolve()
        except OSError:
            pass
        if not target.exists():
            target = (paths.ROOT / url.path()).resolve()
        if not target.exists():
            self.doc_path.setText(f"链接目标不存在：{url.toString()}")
            return
        if target.is_dir():
            nav.navigate("results", path=str(target))
        elif target.suffix.lower() in (".md", ".markdown") or target.name == "LICENSE":
            self.open(path=str(target))
        elif target.suffix.lower() in (".png", ".jpg", ".jpeg", ".svg", ".pdf", ".csv", ".json", ".npz"):
            nav.navigate("results", path=str(target))
        else:
            open_external(target)

    # ------------------------------------------------------------------ find
    def eventFilter(self, obj, e):
        if obj is self.find and e.type() == e.Type.KeyPress:
            if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.find_prev()
                return True
            if e.key() == Qt.Key.Key_Escape:
                self.find.clear()
                self.reader.setFocus()
                return True
        return super().eventFilter(obj, e)

    def focus_find(self):
        if self.stack.currentWidget() is self.params:
            self.params.filter.setFocus()
            self.params.filter.selectAll()
        else:
            self.find.setFocus()
            self.find.selectAll()

    def _search(self, text, keep_index=False):
        if self.stack.currentWidget() is self.params:
            return
        doc = self.reader.document()
        t = ThemeManager.instance().tokens
        self.matches = []
        sels = []
        q = text.strip() if isinstance(text, str) else ""
        if q:
            fmt = QTextCharFormat()
            hl = QColor(t.warning)
            hl.setAlpha(70 if t.dark else 55)
            fmt.setBackground(hl)
            c = QTextCursor(doc)
            while len(self.matches) < 5000:
                c = doc.find(q, c)
                if c.isNull():
                    break
                self.matches.append(QTextCursor(c))
                s = QTextEdit.ExtraSelection()
                s.cursor = QTextCursor(c)
                s.format = fmt
                sels.append(s)
        self.reader.setExtraSelections(sels)
        if not keep_index:
            self.match_i = -1
        if self.matches:
            if not keep_index:
                self._goto(0)
            else:
                self._update_count()
        else:
            self.find_count.setText("无结果" if q else "")
            if q:
                self.find.setStyleSheet(f"QLineEdit {{ border: 1px solid {t.danger}; }}")
            else:
                self.find.setStyleSheet("")
            return
        self.find.setStyleSheet("")

    def _goto(self, i):
        if not self.matches:
            return
        self.match_i = i % len(self.matches)
        self.reader.setTextCursor(self.matches[self.match_i])
        self.reader.ensureCursorVisible()
        self._update_count()

    def _update_count(self):
        self.find_count.setText(f"{self.match_i + 1}/{len(self.matches)}" if self.matches else "")

    def find_next(self):
        if self.matches:
            self._goto(self.match_i + 1)
        elif self.find.text():
            self._search(self.find.text())

    def find_prev(self):
        if self.matches:
            self._goto(self.match_i - 1)

    # ------------------------------------------------------------------ font
    def zoom(self, step):
        if self.stack.currentWidget() is self.params:
            return
        self.reader.set_font_px(self.reader.font_px + step)
        self.font_lbl.setText(f"{self.reader.font_px}px")
        self.settings.setValue("docs/font_px", self.reader.font_px)
        self._rerender()

    # ------------------------------------------------------------------ page API
    def on_show(self):
        if not getattr(self, "_prefetched", False):
            self._prefetched = True
            run_async(all_specs)            # parsing the dataclass sources takes ~0.2 s: not on the GUI thread
        if self._initial:
            self._initial = False
            if self.current is None:
                last = self.settings.value("docs/last", "") or ""
                key = last if last in self._items else str(paths.ROOT / "README.md")
                if key not in self._items and self._items:
                    key = next(iter(self._items))
                if key in self._items:
                    self.tree.setCurrentItem(self._items[key])

    def open(self, path=None, **_):
        if not path:
            return
        if path in (PARAMS, SOURCES, "params", "sources"):
            key = PARAMS if "param" in path else SOURCES
        else:
            p = paths.absolute(path)
            try:
                p = p.resolve()
            except OSError:
                pass
            key = str(p)
            if key not in self._items:
                if not p.is_file():
                    return
                if self._other is None:
                    self._other = QTreeWidgetItem(["其他"])
                    self._other.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    f = self._other.font(0)
                    f.setBold(True)
                    self._other.setFont(0, f)
                    self._other.setForeground(0, QColor(ThemeManager.instance().tokens.muted))
                    self.tree.addTopLevelItem(self._other)
                    self._other.setExpanded(True)
                self._add_doc(self._other, key, doc_title(p), "book" if p.suffix == ".md" else "file", paths.rel(p))
                self._retheme_icons()
        self._initial = False
        it = self._items[key]
        if self.tree.currentItem() is it:
            self.show_doc(key)
        else:
            self.tree.setCurrentItem(it)
        self.tree.scrollToItem(it)
