"""File previews shared by the pages: images, JSON, CSV, Markdown, text, npz/npy arrays and folders.

Typical use::

    from ltemsim_gui.widgets.fileviews import preview_for
    w = preview_for(path)                  # FileView subclass matching the file type, already loading `path`
    w.open_requested.connect(host.show)    # the view asks its host to show another path (thumbnail, link, log ...)

Every view is a `FileView`:
    `.load(path) -> bool`   False when the file is missing or could not be parsed (a Chinese message is shown in the
                            view); files above `FileView.ASYNC_BYTES` are parsed in a worker thread, then `.loading`
                            is True until the `loaded(bool ok)` signal fires
    `.path`                 the shown path (Path or None)
    `open_requested(str)`   emitted when the user clicks something that refers to another file or folder

Classes: `ImageFileView`, `JsonView` (`.set_data(obj)`), `CsvView`, `MarkdownView` (`.set_markdown(text, base_dir)`),
`TextView` (`.set_text(text)`), `NpzView` (`.select(name)`, `.arrays`), `FolderView`, `InfoView` (unknown types),
`CompareView(a, b)` (two images side by side, synchronised zoom).
Helpers: `preview_for`, `kind_of`, `describe`, `human_size`, `fmt_time`, `FlowLayout`, `FindBox`, `ElideLabel`,
`segmented(labels)` (tab-bar switch), `compact_plot(PlotWidget)` (minimum height only: PlotWidget styles its toolbar).
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
import weakref
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import shiboken6
from PySide6.QtCore import (QAbstractTableModel, QEvent, QFileInfo, QModelIndex, QPoint, QRect, QSize,
                            QSortFilterProxyModel, Qt, QTimer, QUrl, Signal)
from PySide6.QtGui import (QBrush, QColor, QDesktopServices, QGuiApplication, QImage, QImageReader, QKeySequence,
                           QPixmap, QShortcut, QTextCursor, QTextDocument, QTextLength, QTextTable, QTextTableFormat)
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFileIconProvider, QFrame,
                               QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLayout, QLineEdit, QMenu,
                               QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter,
                               QStackedWidget, QTableView, QTabBar, QTextBrowser, QToolButton, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import paths
from ..paramspec import format_float
from ..theme import ThemeManager
from ..workers import run_async
from .common import ClickableCard, MetricTile, label, open_external, reveal
from .imageview import ImageView
from .logview import LogView
from .mpl import PlotWidget

__all__ = ["FileView", "ImageFileView", "JsonView", "CsvView", "MarkdownView", "TextView", "NpzView", "FolderView",
           "InfoView", "CompareView", "FlowLayout", "FindBox", "ElideLabel", "preview_for", "kind_of", "describe",
           "human_size", "fmt_time", "segmented", "compact_plot", "IMAGE_EXT", "TEXT_EXT"]

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".tif", ".tiff", ".webp"}
TEXT_EXT = {".txt", ".log", ".m", ".py", ".sh", ".cfg", ".ini", ".toml", ".yaml", ".yml", ".tex", ".dat", ".out",
            ".done", ".rst", ".bib", ".gitignore"}
KIND_ZH = {"folder": "文件夹", "json": "JSON", "csv": "CSV 表格", "markdown": "Markdown", "npz": "NumPy 数组",
           "text": "文本", "other": "文件"}
# names of 1-D arrays that are probably coordinates (preferred as plot x axes / image axes)
COORD_NAMES = ("t", "time", "x", "z", "X", "Z", "xc", "zc", "xe", "ze", "x_um", "X_um", "z_um", "heights_nm",
               "heights", "h", "freq", "y", "r", "P", "V", "P_avg_mW", "V_bias")


# ---------------------------------------------------------------------------------------------- helpers
def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def fmt_time(ts: float, short: bool = False) -> str:
    lt = time.localtime(ts)
    if short and lt.tm_year == time.localtime().tm_year:
        return time.strftime("%m-%d %H:%M", lt)
    return time.strftime("%Y-%m-%d %H:%M", lt)


def _looks_text(p: Path) -> bool:
    try:
        with open(p, "rb") as fh:
            chunk = fh.read(4096)
    except OSError:
        return False
    if not chunk:
        return True
    if b"\0" in chunk:
        return False
    try:
        chunk.decode("utf-8")
        return True
    except UnicodeDecodeError as e:
        return e.start > len(chunk) - 4          # only a multi-byte character cut at the chunk end


def _has_long_line(text: str | None, limit: int) -> bool:
    if not text or len(text) <= limit:
        return False
    return max(map(len, text.split("\n"))) > limit


def kind_of(path) -> str:
    """'folder' | 'image' | 'json' | 'csv' | 'markdown' | 'npz' (also .npy) | 'text' | 'other'."""
    p = Path(path)
    if p.is_dir():
        return "folder"
    ext = p.suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext == ".json":
        return "json"
    if ext in (".csv", ".tsv"):
        return "csv"
    if ext in (".md", ".markdown"):
        return "markdown"
    if ext in (".npz", ".npy"):
        return "npz"
    if ext in TEXT_EXT or (p.is_file() and p.stat().st_size < 64 * 1024 * 1024 and _looks_text(p)):
        return "text"
    return "other"


def describe(path) -> str:
    """Short Chinese type label, e.g. 'PNG 图像', 'JSON', '文件夹'."""
    p = Path(path)
    k = kind_of(p)
    if k == "image":
        return f"{p.suffix[1:].upper()} 图像"
    if k == "npz" and p.suffix.lower() == ".npy":
        return "NumPy 数组 (.npy)"
    if k == "npz":
        return "NumPy 数组包 (.npz)"
    if k == "text" and p.suffix:
        return f"文本 ({p.suffix})"
    return KIND_ZH.get(k, "文件")


def _alive(obj) -> bool:
    try:
        return shiboken6.isValid(obj)
    except Exception:
        return False


def _weak_call(ref, method: str, *pre):
    """run_async callback calling ref().method(*pre, *args) while the view exists. Workers' task objects are
    never freed (their signal lambdas form a cycle through C++), so they must not hold the view strongly:
    a strong reference kept every replaced preview — with its arrays — alive (+200 MB per big npz)."""
    def cb(*a):
        s = ref()
        if s is not None and _alive(s):
            getattr(s, method)(*pre, *a)
    return cb


def _read_weak(ref, p):
    s = ref()
    return None if s is None else s._read(p)


def _is_log(p: Path) -> bool:
    return p.suffix.lower() == ".log" or "log" in p.stem.lower() or p.parent.name.lower() == "logs"


def _fnum(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return "∞" if f > 0 else "−∞"
    return format_float(f)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


_NUM_T = {int, float, type(None)}


def _num_list(v) -> bool:
    """A JSON list of numbers (None allowed as a gap) with at least two entries."""
    if not isinstance(v, list) or len(v) < 2:
        return False
    ts = set(map(type, v))          # C-speed type scan (big JSON arrays); bool is its own type
    return ts <= _NUM_T and (int in ts or float in ts)


def _num_matrix(v) -> bool:
    return (isinstance(v, list) and len(v) >= 2 and all(isinstance(r, list) for r in v)
            and len({len(r) for r in v}) == 1 and all(_num_list(r) for r in v))


def _plot_kind(v) -> str | None:
    return "list" if _num_list(v) else ("matrix" if _num_matrix(v) else None)


def _as_float(v) -> np.ndarray:
    return np.array(v, dtype=float)          # None -> NaN


def _coord_score(name: str) -> int:
    if name in COORD_NAMES:
        return COORD_NAMES.index(name)
    return 1000


def _uniform(c: np.ndarray) -> bool:
    if c.size < 2 or not np.all(np.isfinite(c)):
        return False
    d = np.diff(c)
    return bool(np.all(d > 0) and np.allclose(d, d[0], rtol=1e-3, atol=0))


def segmented(labels, parent=None) -> QTabBar:
    """A small tab bar used as a segmented switch (styled by the theme's QTabBar rules)."""
    tb = QTabBar(parent)
    tb.setDrawBase(False)
    tb.setExpanding(False)
    tb.setDocumentMode(True)
    tb.setUsesScrollButtons(False)          # never squeezed into scroll arrows by a crowded toolbar
    tb.setElideMode(Qt.TextElideMode.ElideNone)
    tb.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    for t in labels:
        tb.addTab(t)
    return tb


def _decimate(x: np.ndarray, y: np.ndarray, n_max: int = 5000):
    """Min/max envelope of a long series for display (Agg needs seconds for 10^5..10^6 noisy points).
    Returns (x, y, decimated?)."""
    n = len(y)
    if n <= n_max:
        return x, y, False
    step = int(math.ceil(n / (n_max // 2)))
    m = n // step * step
    yb = y[:m].reshape(-1, step)
    nan = np.isnan(yb)
    lo = np.argmin(np.where(nan, np.inf, yb), axis=1)
    hi = np.argmax(np.where(nan, -np.inf, yb), axis=1)
    base = np.arange(yb.shape[0]) * step
    idx = np.stack([base + np.minimum(lo, hi), base + np.maximum(lo, hi)], axis=1).ravel()
    idx = np.concatenate([idx, np.arange(m, n)])
    return x[idx], y[idx], True


def _decimated_note(ax, n: int):
    ax.text(0.995, 0.01, f"display: min/max envelope of {n:,} points", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=7, alpha=0.6)


def _explain(err: str) -> str:
    """Chinese explanation for the usual reasons a result file cannot be read (+ the original message)."""
    first = err.strip().splitlines()[0] if err.strip() else err
    rules = (
        (("BadZipFile", "not a zip file", "Bad magic number", "zlib.error", "CRC-32"),
         "文件已损坏或不完整（不是有效的 npz / zip 归档，可能是运行中断时只写了一半）"),
        (("pickled (object) data", "Failed to interpret file"),
         "不是有效的 NumPy 文件（文件头无法识别；若确为含 Python 对象的 npy，出于安全原因不加载）"),
        (("Object arrays cannot be loaded",), "包含 Python 对象数组（需要 pickle），出于安全原因不加载"),
        (("EOFError", "No data left"), "文件为空或被截断"),
        (("JSONDecodeError",), "不是有效的 JSON（文件可能仍在写入或已损坏）"),
        (("MemoryError",), "文件过大，内存不足"),
        (("PermissionError",), "没有读取权限"),
    )
    for keys, zh in rules:
        if any(k in err for k in keys):
            return f"{zh}\n\n{first}"
    return first


class ElideLabel(QLabel):
    """Single-line label that elides its text to the available width (full text in the tooltip), so long
    info texts never widen a toolbar / preview beyond the window."""

    def __init__(self, text="", obj=None, mode=Qt.TextElideMode.ElideRight):
        super().__init__()
        if obj:
            self.setObjectName(obj)
        self._full, self._mode = "", mode
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, t):
        self._full = t
        self.setToolTip(t)
        self._refresh()

    def text(self):
        return self._full

    def text_full(self):
        return self._full

    def sizeHint(self):
        h = super().sizeHint()
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + 6, h.height())

    def minimumSizeHint(self):
        return QSize(40, super().minimumSizeHint().height())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh()

    def _refresh(self):
        super().setText(self.fontMetrics().elidedText(self._full, self._mode, max(20, self.width())))


def _shortcut(widget, key, slot):
    sc = QShortcut(QKeySequence(key), widget)
    sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
    sc.activated.connect(slot)
    return sc


ROW_QSS = "QTreeView::item { height: 22px; }"


def compact_plot(w: PlotWidget, height: int = 220) -> PlotWidget:
    """Minimum height for plots embedded in previews (PlotWidget styles and re-themes its own toolbar)."""
    w.setMinimumHeight(height)
    return w


def _icon_for(path) -> Any:
    return QFileIconProvider().icon(QFileInfo(str(path)))


class FindBox(QLineEdit):
    """Search field: Return = next match, Shift+Return = previous. Signal `search(text, backwards)`."""
    search = Signal(str, bool)

    def __init__(self, placeholder="查找…", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setClearButtonEnabled(True)
        self.setMinimumWidth(96)            # shrinks in narrow previews instead of pushing the toolbar wider
        self.setMaximumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def sizeHint(self):
        return QSize(200, super().sizeHint().height())

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.search.emit(self.text(), bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier))
            return
        super().keyPressEvent(e)

    def set_found(self, ok: bool):
        t = ThemeManager.instance().tokens
        self.setStyleSheet("" if ok else f"QLineEdit {{ border: 1px solid {t.danger}; }}")


def _find_in(edit, text: str, back: bool) -> bool:
    """Find with wrap-around in a QPlainTextEdit / QTextBrowser."""
    if not text:
        return True
    flags = QTextDocument.FindFlag.FindBackward if back else QTextDocument.FindFlag(0)
    if edit.find(text, flags):
        return True
    cur = edit.textCursor()
    cur.movePosition(QTextCursor.MoveOperation.End if back else QTextCursor.MoveOperation.Start)
    edit.setTextCursor(cur)
    return edit.find(text, flags)


class FlowLayout(QLayout):
    """Lays out children left-to-right and wraps them into rows (for thumbnail grids)."""

    def __init__(self, parent=None, spacing=10):
        super().__init__(parent)
        self._items = []
        self._sp = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return self._arrange(QRect(0, 0, w, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        s = QSize()
        for it in self._items:
            if not it.isEmpty():
                s = s.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return s + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _arrange(self, rect, test):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        lines, cur, x = [], [], r.x()
        for it in self._items:
            if it.isEmpty():                # hidden widgets take no slot (and no spacing)
                continue
            hint = it.sizeHint()
            if cur and x + hint.width() > r.right() + 1:
                lines.append(cur)
                cur, x = [], r.x()
            cur.append((it, x, hint))
            x += hint.width() + self._sp
        if cur:
            lines.append(cur)
        y = r.y()
        for i, line in enumerate(lines):
            lh = max(h.height() for _, _, h in line)
            if not test:
                for it, x, h in line:           # vertically centred in their row
                    it.setGeometry(QRect(QPoint(x, y + (lh - h.height()) // 2), h))
            y += lh + (self._sp if i < len(lines) - 1 else 0)
        return y - rect.y() + m.bottom()


# ---------------------------------------------------------------------------------------------- base
class FileView(QWidget):
    """Base class of all previews (see the module docstring for the protocol).

    Subclasses implement `_read(path) -> data` (runs in a worker thread for large files: no widget access) and
    `_populate(data)` (GUI thread), and put their widget in place with `_set_content(widget)`.
    """
    open_requested = Signal(str)
    loaded = Signal(bool)
    ASYNC_BYTES = 256 * 1024

    def __init__(self, parent=None):
        super().__init__(parent)
        self.path: Path | None = None
        self.loading = False
        self._gen = 0
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._stack = QStackedWidget()
        self._note = QLabel("", alignment=Qt.AlignmentFlag.AlignCenter)
        self._note.setObjectName("muted")
        self._note.setWordWrap(True)
        self._note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._stack.addWidget(self._note)
        self._content: QWidget | None = None
        lay.addWidget(self._stack, 1)

    def _set_content(self, w: QWidget):
        self._content = w
        self._stack.addWidget(w)

    def message(self, text: str):
        """Replace the content by a centred message (errors, empty states, 'loading …')."""
        self._note.setText(text)
        self._stack.setCurrentWidget(self._note)

    def _show_content(self):
        if self._content is not None:
            self._stack.setCurrentWidget(self._content)

    def load(self, path) -> bool:
        p = Path(path)
        self.path = p
        self._gen += 1
        gen = self._gen
        if not p.exists():
            self.message(f"文件不存在：\n{paths.rel(p)}")
            self.loaded.emit(False)
            return False
        try:
            size = p.stat().st_size if p.is_file() else 0
        except OSError:
            size = 0
        if not p.is_dir() and not self._needs_async(p, size):
            try:
                data = self._read(p)
            except Exception as e:
                self._fail(f"{type(e).__name__}: {e}")
                return False
            return self._finish(gen, data)
        self.loading = True
        self.message(f"正在读取 {p.name}" + (f"（{human_size(size)}）" if size else "") + " …")
        ref = weakref.ref(self)
        run_async(_read_weak, ref, p, on_done=_weak_call(ref, "_finish", gen),
                  on_error=_weak_call(ref, "_finish_err", gen))
        return True

    def _needs_async(self, p: Path, size: int) -> bool:
        return size > self.ASYNC_BYTES

    def _finish(self, gen, data) -> bool:
        if gen != self._gen:
            return False
        self.loading = False
        self._show_content()                 # before _populate: an empty-state message() from it must win
        try:
            self._populate(data)
        except Exception as e:
            self._fail(f"{type(e).__name__}: {e}")
            return False
        self.loaded.emit(True)
        return True

    def _finish_err(self, gen, err: str):
        if gen == self._gen:
            self.loading = False
            self._fail(err.split("\n\n")[0])

    def _fail(self, msg: str):
        self.loading = False
        name = self.path.name if self.path else ""
        self.message(f"无法读取 {name}\n\n{_explain(msg)}")
        self.loaded.emit(False)

    def _read(self, p: Path):            # pragma: no cover - abstract
        return None

    def _populate(self, data):           # pragma: no cover - abstract
        pass


def _toolbar() -> tuple[QWidget, QHBoxLayout]:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 6)
    h.setSpacing(6)
    return w, h


def _flat(text, slot=None, tip=None) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName("flat")
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    if slot:
        b.clicked.connect(slot)
    if tip:
        b.setToolTip(tip)
    return b


# ---------------------------------------------------------------------------------------------- image
class ImageFileView(FileView):
    """Zoomable image (png/jpg/svg …). `file_actions=False` hides the reveal / open buttons (when the host has them)."""
    MAX_SIDE = 4096         # larger images are shown downscaled (Qt refuses to decode > 256 MB at full size)

    def __init__(self, parent=None, file_actions=True):
        super().__init__(parent)
        self.view = ImageView()
        if not file_actions:
            for b in self.view.findChildren(QPushButton):
                if b.text() in ("在访达中显示", "用默认程序打开"):
                    b.hide()
        self._set_content(self.view)

    def load(self, path) -> bool:
        p = Path(path)
        self.path = p
        self._gen += 1
        ok = self.view.load(p)
        if not ok and p.is_file():
            ok = self._load_scaled(p)
        self._show_content()
        self.loaded.emit(ok)
        return ok

    def _load_scaled(self, p: Path) -> bool:
        """Huge images: decode a downscaled copy (the full-size decode exceeds Qt's allocation limit)."""
        r = QImageReader(str(p))
        sz = r.size()
        if not sz.isValid() or max(sz.width(), sz.height()) <= self.MAX_SIDE:
            return False
        need = sz.width() * sz.height() * 4 // 2 ** 20 + 64         # MB: png / tiff decode at full size first
        if need > 2048:
            self.view.clear(f"图像过大（{sz.width()}×{sz.height()}），无法在应用内预览；请用默认程序打开")
            return False
        r.setScaledSize(sz.scaled(self.MAX_SIDE, self.MAX_SIDE, Qt.AspectRatioMode.KeepAspectRatio))
        old = QImageReader.allocationLimit()
        QImageReader.setAllocationLimit(max(old, need))
        try:
            img = r.read()
        finally:
            QImageReader.setAllocationLimit(old)
        if img.isNull():
            return False
        v = self.view
        pm = QPixmap.fromImage(img)
        v.path = p
        v.item.setPixmap(pm)
        v.scene.setSceneRect(QRect(0, 0, pm.width(), pm.height()).toRectF())
        v.empty.hide()
        v.view.show()
        if v.caption is not None:
            v.caption.setText(f"{p.name}  ·  {sz.width()}×{sz.height()}（缩小显示为 {pm.width()}×{pm.height()}）")
        v.view.resetTransform()
        v.view.fit()
        return True


# ---------------------------------------------------------------------------------------------- text
class TextView(FileView):
    """Monospace read-only text with search and line wrapping; `log=True` adds the run-log colouring and shows
    the END of long logs (where errors and the final result are)."""
    MAX_BYTES = 2 * 1024 * 1024         # more text blocks the GUI thread for > 150 ms when set into the editor
    LOG_TAIL = 1024 * 1024              # the log highlighter is slower: keep the shown tail smaller

    def __init__(self, parent=None, log=False):
        super().__init__(parent)
        self.log = log
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar, h = _toolbar()
        self.info = ElideLabel("", "muted")
        h.addWidget(self.info, 1)
        self.find = FindBox()
        self.find.search.connect(lambda t, back: self.find.set_found(_find_in(self.edit, t, back)))
        h.addWidget(self.find)
        self.wrap = QCheckBox("自动换行")
        self.wrap.toggled.connect(lambda on: self.edit.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.WidgetWidth if on else QPlainTextEdit.LineWrapMode.NoWrap))
        h.addWidget(self.wrap)
        lay.addWidget(bar)
        if log:
            self.edit = LogView(max_lines=0)
            self.edit.setPlaceholderText("（空文件）")
        else:
            self.edit = QPlainTextEdit()
            self.edit.setObjectName("log")
            self.edit.setReadOnly(True)
            self.edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            self.edit.setPlaceholderText("（空文件）")
        lay.addWidget(self.edit, 1)
        self._set_content(w)

    def _read(self, p):
        size = p.stat().st_size
        cap = self.LOG_TAIL if self.log else self.MAX_BYTES
        with open(p, "rb") as fh:
            if self.log and size > cap:
                fh.seek(size - cap)
                raw = fh.read(cap)
                raw = raw[raw.find(b"\n") + 1:]          # start at a line boundary
            else:
                raw = fh.read(cap)
        return raw.decode("utf-8", errors="replace"), size > cap, size

    def _populate(self, data):
        text, truncated, size = data
        self.set_text(text)
        n = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        cap = human_size(self.LOG_TAIL if self.log else self.MAX_BYTES)
        extra = (f"  ·  文件较大，仅显示{'最后' if self.log else '前'} {cap}（完整内容请用默认程序打开）"
                 if truncated else "")
        self.info.setText(f"{n} 行  ·  {human_size(size)}{extra}")
        self.info.setToolTip(self.info.text())

    def set_text(self, text: str):
        if isinstance(self.edit, LogView):
            self.edit.set_text(text)
        else:
            self.edit.setPlainText(text)
        self.info.setText(f"{text.count(chr(10)) + 1} 行" if text else "")
        self._show_content()


# ---------------------------------------------------------------------------------------------- json
def _json_search(data, needle: str, cap: int = 2000) -> list[tuple]:
    """Key paths whose key or scalar value contains `needle` (lower-case), at most `cap` (worker thread)."""
    out = []

    def rec(v, path):
        if len(out) >= cap:
            return
        if isinstance(v, dict):
            for k, x in v.items():
                if len(out) >= cap:
                    return
                if needle in str(k).lower() or (not isinstance(x, (dict, list)) and needle in str(x).lower()):
                    out.append(path + (k,))
                rec(x, path + (k,))
        elif isinstance(v, list):
            for i, x in enumerate(v):
                if len(out) >= cap:
                    return
                if not isinstance(x, (dict, list)) and needle in str(x).lower():
                    out.append(path + (i,))
                rec(x, path + (i,))

    if isinstance(data, (dict, list)):
        rec(data, ())
    elif needle in str(data).lower():
        out.append(())
    return out


class JsonView(FileView):
    """Collapsible JSON tree (lazy, for big files) with a raw-text toggle, search, copy and quick plots of
    numeric arrays (x axis: index or a same-length numeric array next to it / in an enclosing object)."""
    MAX_CHILDREN = 500
    RAW_MAX = 1_000_000         # characters shown in the raw view (setPlainText of 34 MB froze the GUI for 19 s)
    LONG_LINE = 20_000          # single-line (compact) JSON is pretty-printed for the raw view

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data: Any = None
        self._raw = ""
        self._long = False
        self._raw_shown = None
        self._token = object()          # identifies the shown data (raw view rendered for it?)
        self._search_gen = 0
        self._nodes: dict[int, tuple[Any, tuple, str | None]] = {}     # id -> (value, key path, plot kind)
        self._filled: set[int] = set()
        self._matches: list[tuple] = []
        self._needle = ""
        self._mi = -1

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar, h = _toolbar()
        self.mode = segmented(["树形", "原始文本"])
        self.mode.currentChanged.connect(self._on_mode)
        h.addWidget(self.mode)
        self.info = ElideLabel("", "muted")
        h.addWidget(self.info, 1)
        self.find = FindBox("查找键或值…")
        self.find.search.connect(self._on_find)
        h.addWidget(self.find)
        self.b_expand = _flat("展开", self._expand_all, "展开前三层")
        self.b_collapse = _flat("折叠", lambda: self.tree.collapseAll(), "全部折叠")
        self.b_copy = _flat("复制", self.copy_value, "复制所选节点的值（对象/数组为 JSON 文本）")
        self.b_plot = _flat("绘图", self.show_plot, "绘制所选的数值数组 / 矩阵")
        self.b_plot.setEnabled(False)
        for b in (self.b_expand, self.b_collapse, self.b_copy, self.b_plot):
            h.addWidget(b)
        lay.addWidget(bar)

        self.stack = QStackedWidget()
        self.split = QSplitter(Qt.Orientation.Vertical)
        self.split.setChildrenCollapsible(False)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["键", "值", "类型"])
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setStyleSheet(ROW_QSS)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setColumnWidth(0, 220)
        self.tree.itemExpanded.connect(self._fill)
        self.tree.currentItemChanged.connect(self._on_current)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        _shortcut(self.tree, QKeySequence.StandardKey.Copy, self.copy_value)
        self.split.addWidget(self.tree)

        self.plot_box = QWidget()
        pl = QVBoxLayout(self.plot_box)
        pl.setContentsMargins(0, 6, 0, 0)
        pl.setSpacing(4)
        ph = QHBoxLayout()
        ph.addWidget(QLabel("横轴"))
        self.x_combo = QComboBox()
        self.x_combo.setMinimumWidth(110)
        self.x_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.x_combo.setMinimumContentsLength(14)
        self.x_combo.currentIndexChanged.connect(lambda *_: self._draw())
        ph.addWidget(self.x_combo)
        self.logy = QCheckBox("对数 y")
        self.logy.toggled.connect(lambda *_: self._draw())
        ph.addWidget(self.logy)
        self.plot_title = ElideLabel("", "muted", Qt.TextElideMode.ElideLeft)
        self.plot_title.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        ph.addWidget(self.plot_title, 1)
        ph.addWidget(_flat("隐藏图表", lambda: self.plot_box.hide()))
        pl.addLayout(ph)
        self.plot = compact_plot(PlotWidget(figsize=(6, 3)))
        pl.addWidget(self.plot, 1)
        self.plot_box.hide()
        self.split.addWidget(self.plot_box)
        self.split.setSizes([400, 300])
        self.stack.addWidget(self.split)

        self.raw = QPlainTextEdit()
        self.raw.setObjectName("log")
        self.raw.setReadOnly(True)
        self.stack.addWidget(self.raw)
        lay.addWidget(self.stack, 1)
        self._set_content(w)
        self._plot_item = None

    # -- loading
    def _read(self, p):
        text = p.read_text(encoding="utf-8", errors="replace")
        obj = json.loads(text)                 # NaN / Infinity (written by the scripts) are accepted
        return obj, text, _has_long_line(text, self.LONG_LINE)

    def _populate(self, data):
        obj, text, long = data
        self.set_data(obj, text, long)

    def set_data(self, obj, raw: str | None = None, long_lines: bool | None = None):
        """Show a Python object (dict / list / scalar) as a tree; `raw` is the text for the raw view."""
        self.data = obj
        self._raw = raw
        self._long = _has_long_line(raw, self.LONG_LINE) if long_lines is None and raw is not None else bool(long_lines)
        self._token = object()
        self._nodes.clear()
        self._filled.clear()
        self._search_gen += 1
        self._matches, self._needle, self._mi = [], "", -1
        self.tree.clear()
        self.plot_box.hide()
        if isinstance(obj, dict):
            entries = list(obj.items())
        elif isinstance(obj, list):
            entries = list(enumerate(obj))
        else:
            entries = [("(值)", obj)]
        container = isinstance(obj, (dict, list))
        for i, (k, v) in enumerate(entries):
            if i >= self.MAX_CHILDREN:
                self._more(None, len(entries) - i)
                break
            self._add(None, self._key_text(k, isinstance(obj, list)), v, (k,) if container else ())
        # open the first level of small objects
        for i in range(min(self.tree.topLevelItemCount(), 40)):
            it = self.tree.topLevelItem(i)
            v = self._value(it)
            if isinstance(v, dict) and len(v) <= 12:
                it.setExpanded(True)
        n = len(obj) if isinstance(obj, (dict, list)) else 1
        kind = "对象" if isinstance(obj, dict) else ("数组" if isinstance(obj, list) else "值")
        self.info.setText(f"{kind} · {n} 项")
        if self.mode.currentIndex() == 1:
            self._on_mode(1)
        self._show_content()

    @staticmethod
    def _key_text(k, in_list):
        return f"[{k}]" if in_list else str(k)

    def _add(self, parent, key, value, path):
        kind = _plot_kind(value) if isinstance(value, list) else None
        val, typ = self._summary(value, kind)
        it = QTreeWidgetItem([key, val, typ])
        nid = len(self._nodes)
        self._nodes[nid] = (value, path, kind)
        it.setData(0, Qt.ItemDataRole.UserRole, nid)
        if isinstance(value, (dict, list)) and value:
            it.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
        if isinstance(value, (float, str)) and not isinstance(value, bool):
            it.setToolTip(1, repr(value) if isinstance(value, float) else value[:2000])
        it.setForeground(2, QBrush(QColor(ThemeManager.instance().tokens.muted)))
        if parent is None:
            self.tree.addTopLevelItem(it)
        else:
            parent.addChild(it)
        return it

    def _more(self, parent, n):
        it = QTreeWidgetItem([f"… 还有 {n} 项未显示", "", ""])
        it.setFlags(Qt.ItemFlag.NoItemFlags)
        if parent is None:
            self.tree.addTopLevelItem(it)
        else:
            parent.addChild(it)

    @staticmethod
    def _summary(v, kind=None) -> tuple[str, str]:
        if isinstance(v, dict):
            return f"{{{len(v)} 个键}}", "对象"
        if isinstance(v, list):
            if not v:
                return "[]", "数组"
            if kind == "list":
                if len(v) <= 6:
                    return "[" + ", ".join(_fnum(x) if x is not None else "null" for x in v) + "]", f"数值数组 [{len(v)}]"
                a = _as_float(v)
                fin = a[np.isfinite(a)]
                rng = f"{_fnum(fin.min())} … {_fnum(fin.max())}" if fin.size else "NaN"
                return rng, f"数值数组 [{len(v)}]"
            if kind == "matrix":
                return f"{len(v)} × {len(v[0])}", "数值矩阵"
            if len(v) <= 4 and all(isinstance(x, (str, int, float, bool)) or x is None for x in v):
                return json.dumps(v, ensure_ascii=False), f"数组 [{len(v)}]"
            return f"[{len(v)} 项]", "数组"
        if isinstance(v, bool):
            return ("true" if v else "false"), "布尔"
        if v is None:
            return "null", "null"
        if isinstance(v, int):
            return str(v), "整数"
        if isinstance(v, float):
            return _fnum(v), "数值"
        s = str(v)
        one = s.replace("\n", " ⏎ ")
        return (one[:300] + "…") if len(one) > 300 else one, "字符串"

    def _value(self, it):
        nid = it.data(0, Qt.ItemDataRole.UserRole)
        return None if nid is None else self._nodes[nid][0]

    def _kind(self, it):
        nid = None if it is None else it.data(0, Qt.ItemDataRole.UserRole)
        return None if nid is None else self._nodes[nid][2]

    def _fill(self, it):
        nid = it.data(0, Qt.ItemDataRole.UserRole)
        if nid is None or nid in self._filled:
            return
        self._filled.add(nid)
        value, path, _k = self._nodes[nid]
        in_list = isinstance(value, list)
        entries = enumerate(value) if in_list else value.items()
        for i, (k, v) in enumerate(entries):
            if i >= self.MAX_CHILDREN:
                self._more(it, len(value) - i)
                break
            self._add(it, self._key_text(k, in_list), v, path + (k,))

    def _expand_all(self, depth=3):
        def rec(it, d):
            if d >= depth:
                return
            v = self._value(it)
            if isinstance(v, (dict, list)) and v and self._kind(it) != "list":
                it.setExpanded(True)
                for i in range(min(it.childCount(), 60)):
                    rec(it.child(i), d + 1)
        for i in range(min(self.tree.topLevelItemCount(), 200)):
            rec(self.tree.topLevelItem(i), 0)

    # -- selection, copy, menu
    @staticmethod
    def path_text(path: tuple) -> str:
        s = ""
        for k in path:
            s += f"[{k}]" if isinstance(k, int) else (f".{k}" if s else str(k))
        return s

    def current_value(self):
        it = self.tree.currentItem()
        return None if it is None else self._value(it)

    def copy_value(self):
        it = self.tree.currentItem()
        if it is None or it.data(0, Qt.ItemDataRole.UserRole) is None:
            return
        v = self._value(it)
        txt = json.dumps(v, indent=1, ensure_ascii=False) if isinstance(v, (dict, list)) else (
            repr(v) if isinstance(v, float) else ("null" if v is None else str(v)))
        QGuiApplication.clipboard().setText(txt)

    def _copy_path(self):
        it = self.tree.currentItem()
        if it is not None and it.data(0, Qt.ItemDataRole.UserRole) is not None:
            QGuiApplication.clipboard().setText(self.path_text(self._nodes[it.data(0, Qt.ItemDataRole.UserRole)][1]))

    def _menu(self, pos):
        it = self.tree.itemAt(pos)
        if it is None or it.data(0, Qt.ItemDataRole.UserRole) is None:
            return
        self.tree.setCurrentItem(it)
        m = QMenu(self)
        m.addAction("复制值", self.copy_value)
        m.addAction("复制键路径", self._copy_path)
        v = self._value(it)
        if isinstance(v, (dict, list)) and v:
            m.addAction("展开全部子项", lambda: self._expand_item(it))
        if self._kind(it):
            m.addSeparator()
            m.addAction("绘制此数组", self.show_plot)
        m.exec(self.tree.viewport().mapToGlobal(pos))

    def _expand_item(self, it, depth=4):
        if depth <= 0:
            return
        it.setExpanded(True)
        for i in range(min(it.childCount(), 100)):
            c = it.child(i)
            v = self._value(c)
            if isinstance(v, (dict, list)) and v and self._kind(c) != "list":
                self._expand_item(c, depth - 1)

    def _on_current(self, it, _prev=None):
        plottable = self._kind(it) is not None
        self.b_plot.setEnabled(plottable)
        if plottable and not self.plot_box.isHidden():
            self.show_plot()

    def raw_text(self) -> str:
        """Text for the raw view: the file itself, or (huge / single-line files) the first RAW_MAX characters
        of a pretty-printed version (the encoder is lazy, so only that much is ever formatted)."""
        raw = self._raw
        if raw is not None and len(raw) <= self.RAW_MAX and not self._long:
            return raw
        out, n, cut = [], 0, False
        for chunk in json.JSONEncoder(indent=1, ensure_ascii=False).iterencode(self.data):
            out.append(chunk)
            n += len(chunk)
            if n >= self.RAW_MAX:
                cut = True
                break
        text = "".join(out)[:self.RAW_MAX]
        if cut:
            total = f"原始文本共 {human_size(len(raw.encode('utf-8', errors='replace')))}，" if raw is not None else ""
            text += f"\n\n… （已截断：{total}此处只显示格式化后的前 {self.RAW_MAX // 1000} 千字符；完整内容请用默认程序打开）"
        return text

    def _on_mode(self, i):
        if i == 1 and self._raw_shown is not self._token:
            self.raw.setPlainText(self.raw_text())
            self._raw_shown = self._token
        self.stack.setCurrentIndex(i)
        for b in (self.b_expand, self.b_collapse, self.b_copy, self.b_plot):
            b.setVisible(i == 0)

    # -- search
    def _on_find(self, text, back):
        if self.mode.currentIndex() == 1:
            self.find.set_found(_find_in(self.raw, text, back))
            return
        needle = text.strip().lower()
        if not needle:
            self.find.set_found(True)
            return
        if needle != self._needle:
            # walk the data in a worker thread (a 34 MB file took ~1 s per search on the GUI thread)
            self._needle, self._matches, self._mi = needle, [], -1
            self._search_gen += 1
            self.info.setText("正在查找…")
            run_async(_json_search, self.data, needle,
                      on_done=_weak_call(weakref.ref(self), "_search_done", self._search_gen, back))
            return
        self._goto_match(back)

    def _search_done(self, gen, back, matches):
        if gen != self._search_gen:
            return
        self._matches = matches
        self._goto_match(back)

    def _goto_match(self, back):
        if not self._matches:
            self.find.set_found(False)
            self.info.setText("无匹配")
            return
        self.find.set_found(True)
        self._mi = (self._mi + (-1 if back else 1)) % len(self._matches)
        if self.reveal(self._matches[self._mi]):
            self.info.setText(f"匹配 {self._mi + 1} / {len(self._matches)}" + ("+" if len(self._matches) >= 2000 else ""))

    def _search(self, needle, cap=2000):
        return _json_search(self.data, needle, cap)

    def reveal(self, path: tuple) -> bool:
        """Expand the tree down to the node at `path` (tuple of dict keys / list indices) and select it.
        Returns False when the node lies beyond the per-container display limit (MAX_CHILDREN)."""
        it, container = None, self.data
        if not path:
            it = self.tree.topLevelItem(0)
        for k in path:
            pos = list(container.keys()).index(k) if isinstance(container, dict) else int(k)
            if it is not None:
                self._fill(it)
                it.setExpanded(True)
            if pos >= self.MAX_CHILDREN:
                more = self.tree.topLevelItem(self.MAX_CHILDREN) if it is None else it.child(self.MAX_CHILDREN)
                if it is not None:
                    self.tree.setCurrentItem(it)
                if more is not None:
                    self.tree.scrollToItem(more)
                self.info.setText(f"{self.path_text(path)} 位于第 {pos + 1} 项，超出树形视图的显示上限"
                                  f"（每层前 {self.MAX_CHILDREN} 项），可在「原始文本」中查看")
                return False
            nxt = self.tree.topLevelItem(pos) if it is None else it.child(pos)
            if nxt is None:
                break
            it, container = nxt, container[k]
        if it is not None:
            self.tree.setCurrentItem(it)
            self.tree.scrollToItem(it)
        return it is not None

    # -- plot
    def _x_candidates(self, path, n):
        """Numeric lists of length n in the enclosing objects (closest first), as (label, path)."""
        out, seen = [], set()
        for depth in range(len(path) - 1, -1, -1):
            container = self.data
            for k in path[:depth]:
                container = container[k]
            if not isinstance(container, dict):
                continue
            for k, v in container.items():
                p = path[:depth] + (k,)
                if p != path and p not in seen and _num_list(v) and len(v) == n:
                    seen.add(p)
                    out.append((self.path_text(p), p))
        return out

    def show_plot(self):
        it = self.tree.currentItem()
        kind = self._kind(it)
        if kind is None:
            return
        v = self._value(it)
        nid = it.data(0, Qt.ItemDataRole.UserRole)
        path = self._nodes[nid][1]
        n = len(v[0]) if kind == "matrix" else len(v)
        cands = self._x_candidates(path, n)
        self.x_combo.blockSignals(True)
        self.x_combo.clear()
        self.x_combo.addItem("索引", None)
        for lab, p in cands:
            self.x_combo.addItem(lab, p)
        best = min(range(len(cands)), key=lambda i: _coord_score(str(cands[i][1][-1])), default=None)
        if best is not None and _coord_score(str(cands[best][1][-1])) < 1000:
            self.x_combo.setCurrentIndex(best + 1)
        self.x_combo.blockSignals(False)
        self._plot_item = (v, path)
        self.plot_box.show()
        self._draw()

    def _draw(self):
        if self._plot_item is None or self.plot_box.isHidden():
            return
        v, path = self._plot_item
        xp = self.x_combo.currentData()
        name = self.path_text(path) or "value"
        xname = self.x_combo.currentText() if xp is not None else "index"
        x = None
        if xp is not None:
            c = self.data
            for k in xp:
                c = c[k]
            x = _as_float(c)
        logy = self.logy.isChecked()
        self.plot_title.setText(name)

        def draw(fig):
            ax = fig.add_subplot()
            y = _as_float(v)
            rows = y if y.ndim == 2 else y[None, :]
            xx = x if x is not None else np.arange(rows.shape[1])
            if y.ndim == 2 and rows.shape[0] > 12:
                im = ax.imshow(np.abs(rows) if logy else rows, aspect="auto", origin="lower", interpolation="nearest",
                               norm="log" if logy else None)
                fig.colorbar(im, ax=ax)
                ax.set_xlabel(xname)
                ax.set_ylabel("row")
            else:
                for i, r in enumerate(rows):
                    r = np.abs(r) if logy else r
                    xs, ys, dec = _decimate(np.asarray(xx, float), r)
                    ax.plot(xs, ys, lw=1.4 if not dec else 0.8, marker="o" if len(r) <= 60 else None, ms=3.5,
                            label=f"[{i}]" if rows.shape[0] > 1 else None)
                    if dec and i == 0:
                        _decimated_note(ax, len(r))
                if rows.shape[0] > 1:
                    ax.legend(fontsize=8)
                if logy:
                    ax.set_yscale("log")
                ax.set_xlabel(xname)
                ax.set_ylabel(f"|{name}|" if logy else name)
                ax.grid(True, alpha=0.3)
        self.plot.plot(draw)


# ---------------------------------------------------------------------------------------------- csv
def _isfloat(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


class _CsvModel(QAbstractTableModel):
    def __init__(self, header, rows, numeric, cols):
        super().__init__()
        self.header, self.rows, self.numeric, self.cols = header, rows, numeric, cols

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.header)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        r, c = index.row(), index.column()
        row = self.rows[r]
        s = row[c] if c < len(row) else ""
        if role == Qt.ItemDataRole.DisplayRole:
            if self.numeric[c] and len(s) > 12:
                v = self.cols[c][r]
                return f"{v:.6g}" if np.isfinite(v) else s
            return s
        if role == Qt.ItemDataRole.ToolTipRole:
            return s if len(s) > 12 else None
        if role == Qt.ItemDataRole.UserRole:
            if self.numeric[c]:
                v = float(self.cols[c][r])
                return v if np.isfinite(v) else -math.inf
            return s.lower()
        if role == Qt.ItemDataRole.TextAlignmentRole and self.numeric[c]:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self.header[section] if section < len(self.header) else None
        return str(section + 1)


class CsvView(FileView):
    """Sortable table (numeric columns sort numerically) + quick plot of chosen x / y columns."""
    MAX_ROWS = 200_000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.header: list[str] = []
        self.numeric: list[bool] = []
        self.cols: list[np.ndarray] = []
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar, h = _toolbar()
        self.info = ElideLabel("", "muted")
        h.addWidget(self.info, 1)
        h.addWidget(QLabel("x"))
        self.x_combo = QComboBox()
        self.x_combo.setMinimumWidth(90)
        self.x_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.x_combo.setMinimumContentsLength(10)
        self.x_combo.currentIndexChanged.connect(lambda *_: self._draw())
        h.addWidget(self.x_combo)
        h.addWidget(QLabel("y"))
        self.y_btn = QToolButton()
        self.y_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.y_btn.setMinimumWidth(90)
        self.y_btn.setMaximumWidth(220)
        self.y_menu = QMenu(self.y_btn)
        self.y_btn.setMenu(self.y_menu)
        h.addWidget(self.y_btn)
        self.logy = QCheckBox("对数 y")
        self.logy.toggled.connect(lambda *_: self._draw())
        h.addWidget(self.logy)
        lay.addWidget(bar)
        self.split = QSplitter(Qt.Orientation.Vertical)
        self.split.setChildrenCollapsible(False)
        self.table = QTableView()
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setResizeContentsPrecision(50)    # sample rows: 83 ms -> ~15 ms for 222 rows
        _shortcut(self.table, QKeySequence.StandardKey.Copy, self.copy_selection)
        self.split.addWidget(self.table)
        self.plot = compact_plot(PlotWidget(figsize=(6, 3)), 200)
        self.split.addWidget(self.plot)
        self.split.setSizes([380, 320])
        lay.addWidget(self.split, 1)
        self._set_content(w)

    def _read(self, p):
        with open(p, "rb") as fh:
            raw = fh.read(64 * 1024 * 1024)
        text = raw.decode("utf-8-sig", errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        if p.suffix.lower() == ".tsv":
            delim = "\t"
        else:
            try:
                delim = csv.Sniffer().sniff("\n".join(lines[:40]), delimiters=",;\t").delimiter
            except csv.Error:
                delim = ","
        rows = list(csv.reader(lines[:self.MAX_ROWS + 1], delimiter=delim))
        rows = [[c.strip() for c in r] for r in rows]
        if not rows:
            return [], [], [], [], False
        has_header = any(c and not _isfloat(c) for c in rows[0])
        ncol = max(len(r) for r in rows)
        header = (rows[0] + [f"col{i + 1}" for i in range(len(rows[0]), ncol)]) if has_header else \
            [f"col{i + 1}" for i in range(ncol)]
        body = rows[1:] if has_header else rows
        numeric, cols = [], []
        for c in range(ncol):
            cells = [r[c] if c < len(r) else "" for r in body]
            ok = any(cells) and all((not s) or _isfloat(s) for s in cells)
            numeric.append(ok)
            cols.append(np.array([float(s) if s else np.nan for s in cells]) if ok else np.full(len(body), np.nan))
        return header, body, numeric, cols, len(lines) > self.MAX_ROWS + 1

    def _populate(self, data):
        header, body, numeric, cols, truncated = data
        self.header, self.numeric, self.cols = header, numeric, cols
        if not header:
            self.message("（空表格）")
            return
        model = _CsvModel(header, body, numeric, cols)
        proxy = QSortFilterProxyModel(self)
        proxy.setSourceModel(model)
        proxy.setSortRole(Qt.ItemDataRole.UserRole)
        self.table.setModel(proxy)
        self.table.sortByColumn(-1, Qt.SortOrder.AscendingOrder)
        self.table.resizeColumnsToContents()
        for c in range(len(header)):
            self.table.setColumnWidth(c, min(max(self.table.columnWidth(c), 70), 220))
        self.info.setText(f"{len(body)} 行 × {len(header)} 列" + ("  ·  已截断" if truncated else ""))
        num = [i for i, ok in enumerate(numeric) if ok]
        self.x_combo.blockSignals(True)
        self.x_combo.clear()
        self.x_combo.addItem("行号", -1)
        for i in num:
            self.x_combo.addItem(header[i], i)
        self.y_menu.clear()
        default_x = num[0] if len(num) >= 2 else -1
        default_y = [c for c in num if c != default_x][:1]
        for i in num:
            a = self.y_menu.addAction(header[i])
            a.setCheckable(True)
            a.setChecked(i in default_y)
            a.setData(i)
            a.toggled.connect(lambda *_: self._draw())
        self.x_combo.setCurrentIndex(self.x_combo.findData(default_x))
        self.x_combo.blockSignals(False)
        for wdg in (self.x_combo, self.y_btn, self.logy):
            wdg.setEnabled(bool(num))
        self.plot.setVisible(bool(num))
        self._draw()

    def selected_y(self) -> list[int]:
        return [a.data() for a in self.y_menu.actions() if a.isChecked()]

    def set_plot(self, x: str | None, ys: list[str]):
        """Choose plot columns by header name (x=None: row number)."""
        self.x_combo.setCurrentIndex(max(0, self.x_combo.findText(x)) if x else 0)
        for a in self.y_menu.actions():
            a.blockSignals(True)
            a.setChecked(a.text() in ys)
            a.blockSignals(False)
        self._draw()

    def _draw(self):
        if not self.header:
            return
        ys = self.selected_y()
        self.y_btn.setText((", ".join(self.header[i] for i in ys) if ys else "选择列") + "  ▾")
        xi = self.x_combo.currentData()
        logy = self.logy.isChecked()
        header, cols = self.header, self.cols

        def draw(fig):
            ax = fig.add_subplot()
            if not ys:
                ax.text(0.5, 0.5, "choose y columns", ha="center", va="center", transform=ax.transAxes, alpha=0.6)
                ax.set_axis_off()
                return
            n = len(cols[ys[0]])
            x = cols[xi] if xi is not None and xi >= 0 else np.arange(1, n + 1, dtype=float)
            dec = False
            for i in ys:
                y = np.abs(cols[i]) if logy else cols[i]
                xs, ys_, dec = _decimate(x, y)
                ax.plot(xs, ys_, lw=1.4 if not dec else 0.8, marker="o" if n <= 60 else None, ms=3.5, label=header[i])
            if dec:
                _decimated_note(ax, n)
            if len(ys) > 1:
                ax.legend(fontsize=8)
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel(header[xi] if xi is not None and xi >= 0 else "row")
            if len(ys) == 1:
                ax.set_ylabel(("|%s|" if logy else "%s") % header[ys[0]])
            ax.grid(True, alpha=0.3)
        self.plot.plot(draw)

    def copy_selection(self):
        idx = sorted(self.table.selectionModel().selectedIndexes(), key=lambda i: (i.row(), i.column()))
        if not idx:
            return
        rows: dict[int, list[str]] = {}
        for i in idx:
            rows.setdefault(i.row(), []).append(str(i.data(Qt.ItemDataRole.ToolTipRole) or i.data()))
        QGuiApplication.clipboard().setText("\n".join("\t".join(r) for r in rows.values()))


# ---------------------------------------------------------------------------------------------- markdown
class _MdBrowser(QTextBrowser):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def loadResource(self, rtype, url):
        if rtype in (QTextDocument.ResourceType.ImageResource, QTextDocument.ResourceType.ImageResource.value) \
                and self.owner.base_dir is not None:
            u = QUrl(url)
            p = Path(u.toLocalFile()) if u.isLocalFile() else self.owner.base_dir / u.path()
            if p.exists():
                img = QImage(str(p))
                maxw = max(320, self.viewport().width() - 40)
                if not img.isNull() and img.width() > maxw:
                    img = img.scaledToWidth(maxw, Qt.TransformationMode.SmoothTransformation)
                return img
        return super().loadResource(rtype, url)


class MarkdownView(FileView):
    """Rendered Markdown (tables, code, images relative to the file's folder). Links to local files are
    forwarded through `open_requested`; web links open in the browser."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.base_dir: Path | None = None
        self._text = ""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar, h = _toolbar()
        self.mode = segmented(["渲染", "源文本"])
        self.mode.currentChanged.connect(lambda i: self.stack.setCurrentIndex(i))
        h.addWidget(self.mode)
        h.addStretch(1)
        self.find = FindBox()
        self.find.search.connect(self._on_find)
        h.addWidget(self.find)
        lay.addWidget(bar)
        self.stack = QStackedWidget()
        self.browser = _MdBrowser(self)
        self.browser.setOpenLinks(False)
        self.browser.anchorClicked.connect(self._on_link)
        self.browser.document().setDocumentMargin(14)
        self.stack.addWidget(self.browser)
        self.source = QPlainTextEdit()
        self.source.setObjectName("log")
        self.source.setReadOnly(True)
        self.stack.addWidget(self.source)
        lay.addWidget(self.stack, 1)
        self._set_content(w)
        ThemeManager.instance().changed.connect(self._retheme)

    def _read(self, p):
        return p.read_text(encoding="utf-8", errors="replace")

    def _populate(self, text):
        self.set_markdown(text, self.path.parent if self.path else None)

    def set_markdown(self, text: str, base_dir=None):
        self._text = text
        self.base_dir = Path(base_dir) if base_dir else None
        if self.base_dir is not None:
            self.browser.setSearchPaths([str(self.base_dir)])
            self.browser.document().setBaseUrl(QUrl.fromLocalFile(str(self.base_dir) + "/"))
        self.browser.setMarkdown(text)
        self.source.setPlainText(text)
        self._style_tables()
        self._show_content()

    def _style_tables(self):
        t = ThemeManager.instance().tokens

        def walk(frame):
            for f in frame.childFrames():
                if isinstance(f, QTextTable):
                    fmt = f.format()
                    fmt.setBorder(1)
                    fmt.setBorderStyle(QTextTableFormat.BorderStyle.BorderStyle_Solid)
                    fmt.setBorderBrush(QBrush(QColor(t.border)))
                    fmt.setCellSpacing(0)
                    fmt.setCellPadding(5)
                    fmt.setBorderCollapse(True)
                    fmt.setWidth(QTextLength(QTextLength.Type.VariableLength, 0))
                    f.setFormat(fmt)
                    for c in range(f.columns()):
                        cell = f.cellAt(0, c)
                        cf = cell.format()
                        cf.setBackground(QBrush(QColor(t.surface2)))
                        cell.setFormat(cf)
                walk(f)
        walk(self.browser.document().rootFrame())

    def _retheme(self, *_):
        if self._text:
            self.set_markdown(self._text, self.base_dir)

    def _on_find(self, text, back):
        ed = self.browser if self.stack.currentIndex() == 0 else self.source
        self.find.set_found(_find_in(ed, text, back))

    def _on_link(self, url: QUrl):
        if url.scheme() in ("http", "https", "mailto"):
            QDesktopServices.openUrl(url)
            return
        if not url.path() and url.fragment():
            self.browser.scrollToAnchor(url.fragment())
            return
        p = Path(url.toLocalFile()) if url.isLocalFile() else (self.base_dir or paths.ROOT) / url.path()
        if p.exists():
            self.open_requested.emit(str(p.resolve()))


# ---------------------------------------------------------------------------------------------- npz
def _array_info(a: np.ndarray) -> dict:
    """Kind and min / max strings of an array (runs in the worker thread)."""
    info = {"kind": "other", "min": "", "max": ""}
    k = a.dtype.kind
    if k in "US":
        if a.ndim == 0:
            s = str(a.item())
            info["kind"] = "text"
            st = s.lstrip()[:1]
            if st in ("{", "["):
                try:
                    info["json"] = json.loads(s)
                    info["kind"] = "json"
                except ValueError:
                    pass
            one = s.replace("\n", " ")
            info["min"] = ("JSON：" if info["kind"] == "json" else "") + (one[:60] + ("…" if len(one) > 60 else ""))
        else:
            info["kind"] = "strings"
        return info
    if k not in "biufc" or a.size == 0:
        return info
    info["kind"] = "num"
    if a.size > 50_000_000:
        return info
    v = np.abs(a) if k == "c" else a
    if k == "b":
        info["min"], info["max"] = str(bool(v.min())), str(bool(v.max()))
        return info
    if k == "f" or k == "c":
        fin = v[np.isfinite(v)]
        if fin.size == 0:
            info["min"] = info["max"] = "NaN"
            return info
        lo, hi = fin.min(), fin.max()
    else:
        lo, hi = v.min(), v.max()
    info["min"], info["max"] = _fnum(lo), _fnum(hi)
    if k == "c":
        info["min"], info["max"] = "|·| " + info["min"], "|·| " + info["max"]
    return info


class NpzView(FileView):
    """Arrays of a .npz (or a single .npy): table of name / shape / dtype / min / max and a viewer —
    1-D: line plot vs index or a same-length 1-D array; 2-D: image (colormap, log, transpose, coordinate axes
    picked from matching 1-D arrays); 0-d: value; strings: text (JSON strings as a tree).
    Files with cell-edge arrays `xe` / `ze` (ltemsim Solution.save: fields stored flattened, index k*Nx + i)
    show those flat fields as (z, x) maps on the cell centres."""
    CMAPS = ("自动", "viridis", "magma", "inferno", "plasma", "cividis", "turbo", "RdBu_r", "coolwarm", "gray")
    MAX_IMG = 1600              # pixels per axis drawn; bigger 2-D arrays are strided for display only
    SYNC_UNPACKED = 2 * 1024 * 1024

    def __init__(self, parent=None):
        super().__init__(parent)
        self.arrays: dict[str, np.ndarray] = {}
        self.info: dict[str, dict] = {}
        self.current: str | None = None
        self.grid: tuple[np.ndarray, np.ndarray] | None = None      # (x centres, z centres) from xe / ze
        self._grid_um = False
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar, h = _toolbar()
        self.summary = ElideLabel("", "muted")
        h.addWidget(self.summary, 1)
        lay.addWidget(bar)
        self.split = QSplitter(Qt.Orientation.Vertical)
        self.split.setChildrenCollapsible(False)
        self.table = QTreeWidget()
        self.table.setHeaderLabels(["名称", "形状", "dtype", "最小值", "最大值"])
        self.table.setRootIsDecorated(False)
        self.table.setUniformRowHeights(True)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(ROW_QSS)
        self.table.setSortingEnabled(False)
        hd = self.table.header()
        hd.setStretchLastSection(False)
        for c, mode in enumerate((QHeaderView.ResizeMode.Interactive, QHeaderView.ResizeMode.ResizeToContents,
                                  QHeaderView.ResizeMode.ResizeToContents, QHeaderView.ResizeMode.Stretch,
                                  QHeaderView.ResizeMode.Stretch)):
            hd.setSectionResizeMode(c, mode)
        self.table.setColumnWidth(0, 150)
        self.table.currentItemChanged.connect(lambda it, _p: it is not None and self.select(it.text(0)))
        self.table.setMinimumHeight(110)
        self.split.addWidget(self.table)

        viewer = QWidget()
        vl = QVBoxLayout(viewer)
        vl.setContentsMargins(0, 8, 0, 0)
        vl.setSpacing(4)
        self.controls = QWidget()
        cl = FlowLayout(self.controls, spacing=8)        # wraps in narrow windows instead of clipping
        self.x_combo, self.row_combo, self.col_combo = QComboBox(), QComboBox(), QComboBox()
        self.cmap = QComboBox()
        self.cmap.addItems(self.CMAPS)
        self.slice = QSpinBox()
        self.slice.setMinimumWidth(70)

        def pair(text, wdg):
            box = QWidget()
            hl = QHBoxLayout(box)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(4)
            hl.addWidget(QLabel(text))
            hl.addWidget(wdg)
            cl.addWidget(box)
            return box
        self.g_x = pair("横轴", self.x_combo)
        self.g_row = pair("行坐标", self.row_combo)
        self.g_col = pair("列坐标", self.col_combo)
        self.g_cmap = pair("色图", self.cmap)
        self.g_slice = pair("切片", self.slice)
        self.as_grid = QCheckBox("按网格 (z, x)")
        self.as_grid.setChecked(True)
        self.as_grid.setToolTip("Solution 文件把场按单元展平保存（索引 k·Nx + i）；勾选后用 xe / ze 的单元中心重排为二维图")
        self.logc = QCheckBox("对数")
        self.transpose = QCheckBox("转置")
        self.export_btn = _flat("导出 CSV…", self.export_csv, "把当前显示的数组（全分辨率）保存为 CSV")
        for wdg in (self.as_grid, self.logc, self.transpose, self.export_btn):
            cl.addWidget(wdg)
        for c in (self.x_combo, self.row_combo, self.col_combo):
            c.setMinimumWidth(100)
            c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(12)
            c.currentIndexChanged.connect(lambda *_: self._draw())
        self.cmap.currentIndexChanged.connect(lambda *_: self._draw())
        self.logc.toggled.connect(lambda *_: self._draw())
        self.transpose.toggled.connect(lambda *_: self._draw())
        self.slice.valueChanged.connect(lambda *_: self._draw())
        self.as_grid.toggled.connect(lambda *_: self.current and self.select(self.current))
        vl.addWidget(self.controls)
        self.vstack = QStackedWidget()
        self.plot = compact_plot(PlotWidget(figsize=(6, 3.6)))
        self.vstack.addWidget(self.plot)
        self.text = QPlainTextEdit()
        self.text.setObjectName("log")
        self.text.setReadOnly(True)
        self.vstack.addWidget(self.text)
        self.json: JsonView | None = None          # created on first JSON string (saves ~20 ms per preview)
        vl.addWidget(self.vstack, 1)
        self.split.addWidget(viewer)
        self.split.setStretchFactor(0, 0)
        self.split.setStretchFactor(1, 1)
        self.split.setSizes([200, 500])
        lay.addWidget(self.split, 1)
        self._set_content(w)

    def _needs_async(self, p, size):
        if size > self.ASYNC_BYTES:
            return True
        if p.suffix.lower() == ".npz":          # a small compressed file can unpack to hundreds of MB
            try:
                with zipfile.ZipFile(p) as z:
                    return sum(i.file_size for i in z.infolist()) > self.SYNC_UNPACKED
            except (OSError, zipfile.BadZipFile):
                return False
        return False

    def _read(self, p):
        arrays, errors = {}, {}
        if p.suffix.lower() == ".npy":
            arrays[p.stem] = np.load(p, allow_pickle=False)
        else:
            with np.load(p, allow_pickle=False) as z:
                for k in z.files:
                    try:
                        arrays[k] = z[k]
                    except Exception as e:          # object arrays need pickle: never unpickle untrusted data
                        errors[k] = f"{type(e).__name__}: {e}"
        return arrays, {k: _array_info(a) for k, a in arrays.items()}, errors

    def _populate(self, data):
        self.arrays, self.info, errors = data
        self.current = None
        self.grid = None
        xe, ze = self.arrays.get("xe"), self.arrays.get("ze")
        if (xe is not None and ze is not None and xe.ndim == ze.ndim == 1 and xe.size > 2 and ze.size > 2
                and xe.dtype.kind in "iuf" and ze.dtype.kind in "iuf"):
            xe, ze = xe.astype(float), ze.astype(float)
            self.grid = (0.5 * (xe[1:] + xe[:-1]), 0.5 * (ze[1:] + ze[:-1]))
            self._grid_um = "params_json" in self.arrays           # ltemsim grids are in µm
        self.table.blockSignals(True)
        self.table.clear()
        for k, a in self.arrays.items():
            inf = self.info[k]
            shape = "标量" if a.ndim == 0 else ("(" + ", ".join(map(str, a.shape)) + ")")
            g = self._grid_rows(k)
            if g:
                shape += f"  → {g} × {a.size // g}"
            it = QTreeWidgetItem([k, shape, str(a.dtype), inf["min"], inf["max"]])
            it.setToolTip(1, f"按网格重排为 (z, x) = ({g}, {a.size // g})" if g else "")
            it.setToolTip(3, inf["min"])
            self.table.addTopLevelItem(it)
        for k, e in errors.items():
            it = QTreeWidgetItem([k, "", "object", "（对象数组，出于安全原因未加载）", ""])
            it.setToolTip(3, e)
            it.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.addTopLevelItem(it)
        self.table.blockSignals(False)
        total = sum(a.nbytes for a in self.arrays.values())
        self.summary.setText(f"{len(self.arrays)} 个数组  ·  解压后 {human_size(total)}"
                             + (f"  ·  {len(errors)} 个对象数组未加载" if errors else ""))
        # default: the first numeric non-coordinate array, else the first entry
        order = sorted(self.arrays, key=lambda k: (self.info[k]["kind"] != "num" or self.arrays[k].ndim == 0,
                                                   _coord_score(k) < 1000 or k in ("xe", "ze"),
                                                   list(self.arrays).index(k)))
        if order:
            self.select(order[0])
        else:
            self.message("（npz 中没有可读取的数组）" if not errors else
                         f"npz 中只有对象数组（{', '.join(errors)}），出于安全原因不加载")

    def _same_len(self, n, exclude=None):
        """1-D numeric arrays of length n, coordinate-like names first."""
        c = [k for k, a in self.arrays.items() if k != exclude and a.ndim == 1 and a.shape[0] == n
             and a.dtype.kind in "iuf"]
        return sorted(c, key=lambda k: (_coord_score(k), list(self.arrays).index(k)))

    def _grid_rows(self, name) -> int | None:
        """Rows (z cells) when the flat 1-D array `name` is a field on the xe / ze grid, else None."""
        a = self.arrays.get(name)
        if self.grid is None or a is None or a.ndim != 1 or self.info[name]["kind"] != "num" or name in ("xe", "ze"):
            return None
        nx, nz = len(self.grid[0]), len(self.grid[1])
        k = a.size // nx
        return k if a.size % nx == 0 and 2 <= k <= nz else None

    def _gridded(self) -> int | None:
        k = self._grid_rows(self.current) if self.current else None
        return k if k and self.as_grid.isChecked() else None

    def _coord(self, key, n):
        if key == "@x":
            return self.grid[0][:n]
        if key == "@z":
            return self.grid[1][:n]
        return np.asarray(self.arrays[key], float)

    def _axis_name(self, key):
        if key in ("@x", "@z"):
            return f"{key[1]} (µm)" if self._grid_um else f"{key[1]} (cell centres of {key[1]}e)"
        return key

    def select(self, name: str):
        """Show array `name` (also selects its row)."""
        if name not in self.arrays:
            return
        self.current = name
        for i in range(self.table.topLevelItemCount()):
            it = self.table.topLevelItem(i)
            if it.text(0) == name and self.table.currentItem() is not it:
                self.table.blockSignals(True)
                self.table.setCurrentItem(it)
                self.table.blockSignals(False)
        a, inf = self.arrays[name], self.info[name]
        is_num = inf["kind"] == "num"
        g = self._grid_rows(name)
        k = self._gridded()
        nd = 2 if k else a.ndim
        one, two = is_num and nd == 1, is_num and nd >= 2
        self.g_x.setVisible(one)
        for wdg in (self.g_row, self.g_col, self.g_cmap, self.transpose):
            wdg.setVisible(two)
        self.as_grid.setVisible(bool(g))
        self.logc.setVisible(one or two)
        self.logc.setText("对数 y" if one else "对数")
        self.g_slice.setVisible(is_num and nd >= 3)
        self.export_btn.setVisible(is_num and nd >= 1)
        self.controls.setVisible(one or two)
        if inf["kind"] == "json":
            if self.json is None:
                self.json = JsonView()
                self.vstack.addWidget(self.json)
            self.json.set_data(inf["json"], str(a.item()))
            self.vstack.setCurrentWidget(self.json)
            return
        if not is_num or nd == 0:
            if nd == 0:
                v = a.item()
                txt = f"{name} = {v!r}\n\ndtype: {a.dtype}" if is_num else str(v)
            elif a.size == 0:
                txt = f"（空数组，形状 {a.shape}）"
            else:
                flat = a.ravel()
                txt = "\n".join(map(str, flat[:20000])) + ("\n…" if flat.size > 20000 else "")
            self.text.setPlainText(txt)
            self.vstack.setCurrentWidget(self.text)
            return
        for c in (self.x_combo, self.row_combo, self.col_combo):
            c.blockSignals(True)
            c.clear()
        if one:
            self.x_combo.addItem("索引", None)
            for key in self._same_len(a.shape[0], name):
                self.x_combo.addItem(key, key)
            if self.x_combo.count() > 1 and _coord_score(self.x_combo.itemData(1)) < 1000:
                self.x_combo.setCurrentIndex(1)
        elif k:
            for combo, key in ((self.row_combo, "@z"), (self.col_combo, "@x")):
                combo.addItem("索引", None)
                combo.addItem(self._axis_name(key), key)
                combo.setCurrentIndex(1)
            self.transpose.blockSignals(True)
            self.transpose.setChecked(False)            # z vertical, x horizontal
            self.transpose.blockSignals(False)
            self.cmap.blockSignals(True)
            self.cmap.setCurrentIndex(0)
            self.cmap.blockSignals(False)
        else:
            if nd >= 3:
                self.slice.blockSignals(True)
                self.slice.setRange(0, a.shape[0] - 1)
                self.slice.setValue(0)
                self.slice.blockSignals(False)
            sh = a.shape[-2:]
            for combo, n in ((self.row_combo, sh[0]), (self.col_combo, sh[1])):
                combo.addItem("索引", None)
                for key in self._same_len(n, name):
                    combo.addItem(key, key)
                if combo.count() > 1 and _coord_score(combo.itemData(1)) < 1000:
                    combo.setCurrentIndex(1)
            if self.row_combo.currentData() == self.col_combo.currentData() and self.row_combo.currentData():
                self.col_combo.setCurrentIndex(min(2, self.col_combo.count() - 1) if self.col_combo.count() > 2 else 0)
            rc = str(self.row_combo.currentData() or "")
            self.transpose.blockSignals(True)
            self.transpose.setChecked(rc.lower().startswith("x"))       # x as the horizontal axis
            self.transpose.blockSignals(False)
            self.cmap.blockSignals(True)
            self.cmap.setCurrentIndex(0)
            self.cmap.blockSignals(False)
        for c in (self.x_combo, self.row_combo, self.col_combo):
            c.blockSignals(False)
        self.vstack.setCurrentWidget(self.plot)
        self._draw()

    def _shown_2d(self, display=True):
        """(A, row coord, col coord, row name, col name, stride) of the current 2-D view; `display` strides
        arrays larger than MAX_IMG per axis (export uses the full array)."""
        a = self.arrays[self.current]
        k = self._gridded()
        A = a.reshape(k, -1) if k else a
        if A.ndim >= 3:
            A = A[(self.slice.value(),) + (0,) * (A.ndim - 3)]
        rn, cn = self.row_combo.currentData(), self.col_combo.currentData()
        rc = None if rn is None else self._coord(rn, A.shape[0])
        cc = None if cn is None else self._coord(cn, A.shape[1])
        stride = 1
        if display and max(A.shape) > self.MAX_IMG:
            stride = int(math.ceil(max(A.shape) / self.MAX_IMG))
            A = A[::stride, ::stride]
            rc = None if rc is None else rc[::stride]
            cc = None if cc is None else cc[::stride]
        A = np.abs(A) if A.dtype.kind == "c" else np.asarray(A, dtype=float)
        rn, cn = (None if rn is None else self._axis_name(rn)), (None if cn is None else self._axis_name(cn))
        if self.transpose.isChecked():
            A, rc, cc, rn, cn = A.T, cc, rc, cn, rn
        return A, rc, cc, rn, cn, stride

    def _draw(self):
        name = self.current
        if name is None or self.info[name]["kind"] != "num" or self.arrays[name].ndim == 0:
            return
        a = self.arrays[name]
        log = self.logc.isChecked()
        if a.ndim == 1 and not self._gridded():
            xn = self.x_combo.currentData()
            x = np.arange(a.shape[0], dtype=float) if xn is None else np.asarray(self.arrays[xn], float)

            def draw(fig):
                ax = fig.add_subplot()
                y = np.abs(a) if (a.dtype.kind == "c" or log) else a.astype(float)
                xs, ys, dec = _decimate(x, y)
                ax.plot(xs, ys, lw=1.3 if not dec else 0.8, marker="o" if a.shape[0] <= 60 else None, ms=3.5)
                if dec:
                    _decimated_note(ax, a.shape[0])
                if log:
                    ax.set_yscale("log")
                ax.set_xlabel(xn or "index")
                ax.set_ylabel(f"|{name}|" if (log or a.dtype.kind == "c") else name)
                ax.set_title(f"{name}  {a.shape}", loc="left")
                ax.grid(True, alpha=0.3)
            self.plot.plot(draw)
            return
        A, rc, cc, rn, cn, stride = self._shown_2d()
        cmap_name = self.cmap.currentText()
        k = self._gridded()
        title = (f"{name}  {a.shape}" + (f" → ({k}, {a.size // k})" if k else "")
                 + (f"  [{self.slice.value()}]" if a.ndim >= 3 else "")
                 + ("  (transposed)" if self.transpose.isChecked() else "")
                 + (f"  (every {stride}th pixel shown)" if stride > 1 else ""))

        def draw(fig):
            from matplotlib.colors import LogNorm, Normalize
            ax = fig.add_subplot()
            Z, r, c = A, rc, cc
            # ascending coordinates
            if r is not None and r.size > 1 and r[0] > r[-1]:
                Z, r = Z[::-1, :], r[::-1]
            if c is not None and c.size > 1 and c[0] > c[-1]:
                Z, c = Z[:, ::-1], c[::-1]
            fin = Z[np.isfinite(Z)]
            cmap = None if cmap_name == "自动" else cmap_name
            if log:
                Z = np.abs(Z)
                pos = Z[np.isfinite(Z) & (Z > 0)]
                Z = np.where(Z > 0, Z, np.nan)
                norm = LogNorm(vmin=pos.min(), vmax=pos.max()) if pos.size else None
                cmap = cmap or "viridis"
                clabel = f"|{name}|"
            else:
                clabel = name
                if fin.size and fin.min() < 0 < fin.max() and cmap in (None, "RdBu_r", "coolwarm"):
                    m = float(np.abs(fin).max())
                    norm, cmap = Normalize(-m, m), cmap or "RdBu_r"
                else:
                    norm, cmap = None, cmap or "viridis"
            big = Z.size > 4_000_000
            if (r is None or _uniform(r) or big) and (c is None or _uniform(c) or big):
                def ext(v, n):
                    if v is None:
                        return -0.5, n - 0.5
                    d = (v[-1] - v[0]) / max(n - 1, 1)
                    return v[0] - d / 2, v[-1] + d / 2
                x0, x1 = ext(c, Z.shape[1])
                y0, y1 = ext(r, Z.shape[0])
                im = ax.imshow(Z, origin="lower", aspect="auto", extent=(x0, x1, y0, y1), interpolation="nearest",
                               cmap=cmap, norm=norm)
            else:
                X = c if c is not None else np.arange(Z.shape[1])
                Y = r if r is not None else np.arange(Z.shape[0])
                im = ax.pcolormesh(X, Y, Z, shading="nearest", cmap=cmap, norm=norm)
            fig.colorbar(im, ax=ax, label=clabel)
            ax.set_xlabel(cn or "column index")
            ax.set_ylabel(rn or "row index")
            ax.set_title(title, loc="left", fontsize=9)
        self.plot.plot(draw)

    def export_csv(self):
        global _EXPORT_DIR
        name = self.current
        if name is None or self.path is None:
            return
        fn, _ = QFileDialog.getSaveFileName(self, "导出数组为 CSV",
                                            str(_export_dir(self.path) / f"{self.path.stem}_{name}.csv"), "CSV (*.csv)")
        if not fn:
            return
        _EXPORT_DIR = Path(fn).parent
        self.save_csv(fn)

    def save_csv(self, fn):
        """Write the array that is currently shown at full resolution (1-D: with its x column; otherwise the
        shown 2-D slice / grid map)."""
        a = self.arrays[self.current]
        if a.ndim == 1 and not self._gridded():
            xn = self.x_combo.currentData()
            x = np.arange(a.shape[0]) if xn is None else self.arrays[xn]
            np.savetxt(fn, np.column_stack([x, np.abs(a) if a.dtype.kind == "c" else a]), delimiter=",",
                       header=f"{xn or 'index'},{self.current}", comments="")
        else:
            np.savetxt(fn, self._shown_2d(display=False)[0], delimiter=",")


_EXPORT_DIR: Path | None = None


def _export_dir(src: Path) -> Path:
    """Default folder for exports: the last one used, else next to the source unless that is a committed
    result folder (results/ outside gui_runs/), else ~/Downloads."""
    if _EXPORT_DIR is not None and _EXPORT_DIR.is_dir():
        return _EXPORT_DIR
    try:
        r, g, s = paths.RESULTS.resolve(), paths.GUI_RUNS.resolve(), src.resolve().parent
        committed = (s == r or r in s.parents) and not (s == g or g in s.parents)
    except OSError:
        committed = True
    if not committed:
        return src.parent
    dl = Path.home() / "Downloads"
    return dl if dl.is_dir() else Path.home()


# ---------------------------------------------------------------------------------------------- folder
SUMMARY_FILES = ("summary.json", "sweep_summary.json", "meta.json", "params.json")


def _wall(v) -> str:
    if not isinstance(v, (int, float)):
        return "—"
    from .runpanel import fmt_elapsed
    return f"{v:.1f} s" if v < 10 else fmt_elapsed(v)


def _mtime(f) -> float:
    try:
        return os.path.getmtime(f)
    except OSError:
        return 0.0


def _log_status(log: Path) -> str | None:
    """'failed' | 'cancelled' | 'finished' from the runner markers at the end of a gui_log*.txt."""
    try:
        with open(log, "rb") as fh:
            fh.seek(max(0, log.stat().st_size - 4096))
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    return "failed" if "@@FAILED" in tail else ("cancelled" if "@@CANCELLED" in tail else
                                                ("finished" if "@@DONE" in tail else None))


def _active_job(folder: Path, run: dict):
    """The queued / running job that writes this run record into `folder` (same script and function), if any."""
    from ..jobs import JobManager
    try:
        target = folder.resolve()
    except OSError:
        return None
    for j in JobManager.instance().active():
        spec = getattr(j, "spec", None) or {}
        if not j.out or spec.get("script") != run.get("script") or spec.get("func", "main") != run.get("func", "main"):
            continue
        try:
            if paths.absolute(j.out).resolve() == target:
                return j
        except OSError:
            continue
    return None


def _headline(obj, cap=14) -> list[tuple[str, str]]:
    """Key / value strings for the most informative top-level entries of a summary json."""
    out = []

    def val(v):
        if isinstance(v, dict):
            return f"{{{len(v)} 个键}}"
        if isinstance(v, list):
            if _num_list(v):
                if len(v) <= 4:
                    return "[" + ", ".join(_fnum(x) for x in v) + "]"
                a = _as_float(v)
                return f"[{len(v)}]  {_fnum(np.nanmin(a))} … {_fnum(np.nanmax(a))}"
            return f"[{len(v)} 项]"
        if isinstance(v, str):
            return v if len(v) < 80 else v[:77] + "…"
        return _fnum(v) if v is not None else "null"
    if not isinstance(obj, dict):
        return [("(内容)", val(obj))]
    scal = [(k, v) for k, v in obj.items() if not isinstance(v, (dict, list))]
    nested = [(k, v) for k, v in obj.items() if isinstance(v, dict)]
    lists = [(k, v) for k, v in obj.items() if isinstance(v, list)]
    for k, v in scal:
        out.append((k, val(v)))
    for k, d in nested:
        sub = [(k2, v2) for k2, v2 in d.items() if not isinstance(v2, (dict, list))]
        if sub and len(sub) <= 6:
            out += [(f"{k}.{k2}", val(v2)) for k2, v2 in sub]
        else:
            keys = list(d)
            out.append((k, f"{{{len(d)} 个键}}  " + ", ".join(map(str, keys[:5])) + ("…" if len(keys) > 5 else "")))
    for k, v in lists:
        out.append((k, val(v)))
    return out[:cap] + ([("…", f"另有 {len(out) - cap} 项")] if len(out) > cap else [])


def _thumb(path: str, w: int, h: int) -> QImage | None:
    r = QImageReader(path)
    r.setAutoTransform(True)
    sz = r.size()
    if sz.isValid() and path.lower().endswith((".svg", ".jpg", ".jpeg")):
        r.setScaledSize(sz.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio))
    img = r.read()
    if img.isNull():
        return None
    return img.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


class _Thumb(ClickableCard):
    W, H = 216, 146

    def __init__(self, path: str, caption: str, parent=None):
        super().__init__(parent)
        self.path = path
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 6)
        lay.setSpacing(4)
        self.pic = QLabel("载入中…", alignment=Qt.AlignmentFlag.AlignCenter)
        self.pic.setObjectName("muted")
        self.pic.setFixedSize(self.W, self.H)
        lay.addWidget(self.pic)
        cap = QLabel()
        cap.setObjectName("muted")
        cap.setText(cap.fontMetrics().elidedText(caption, Qt.TextElideMode.ElideMiddle, self.W - 14))
        cap.setFixedWidth(self.W)
        lay.addWidget(cap)
        self.setToolTip(paths.rel(path))

    def set_image(self, img: QImage | None):
        if img is None:
            self.pic.setText("无法读取")
            return
        pm = QPixmap.fromImage(img)
        pm.setDevicePixelRatio(max(1.0, img.width() / self.W, img.height() / self.H))
        self.pic.setPixmap(pm)


class FolderView(FileView):
    """Folder summary: counts / size / latest change, run provenance from gui_run*.json (one card per job that
    wrote into the folder; live status from the job manager), headline values of
    summary.json (or sweep_summary / meta / params.json), a clickable thumbnail grid of its images and its
    contents. Clicking a thumbnail or an entry emits `open_requested(path)`."""
    MAX_THUMBS = 48

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data: dict | None = None
        self.thumbs: list[_Thumb] = []
        self._tiles: list[QWidget] = []
        self._tile_grid: QGridLayout | None = None
        self._kv: list[tuple[QLabel, QLabel]] = []
        self._kv_grid: QGridLayout | None = None
        self._cols: tuple | None = None
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        inner.setObjectName("page")
        self.col = QVBoxLayout(inner)
        self.col.setContentsMargins(0, 0, 6, 0)
        self.col.setSpacing(12)
        area.setWidget(inner)
        self._set_content(area)

    def _read(self, p: Path) -> dict:
        n_files = n_dirs = total = 0
        latest = 0.0
        truncated = False
        imgs: list[tuple[str, int]] = []
        child_size: dict[str, int] = {}
        child_count: dict[str, int] = {}
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            rel = Path(dirpath).relative_to(p).parts
            n_dirs += len(dirnames)
            for f in sorted(filenames):
                if f.startswith("."):
                    continue
                fp = os.path.join(dirpath, f)
                try:
                    st = os.stat(fp)
                except OSError:
                    continue
                n_files += 1
                total += st.st_size
                latest = max(latest, st.st_mtime)
                if rel:
                    child_size[rel[0]] = child_size.get(rel[0], 0) + st.st_size
                    child_count[rel[0]] = child_count.get(rel[0], 0) + 1
                if len(rel) <= 2 and os.path.splitext(f)[1].lower() in IMAGE_EXT and len(imgs) < 400:
                    imgs.append((fp, len(rel)))
            if n_files > 200_000:
                truncated = True
                break
        direct = [f for f, d in imgs if d == 0]
        images = direct or [f for f, _ in imgs]
        children = []
        try:
            entries = sorted(os.scandir(p), key=lambda e: (not e.is_dir(), e.name.lower()))
        except OSError:
            entries = []
        for e in entries:
            if e.name.startswith("."):
                continue
            try:
                st = e.stat()
            except OSError:
                continue
            isdir = e.is_dir()
            children.append((e.path, isdir, child_size.get(e.name, 0) if isdir else st.st_size, st.st_mtime,
                             child_count.get(e.name, 0) if isdir else 0))
        # run records: gui_run.json (first job that wrote here) + gui_run_<func>.json of follow-up jobs
        runs = []
        for f in sorted(p.glob("gui_run*.json"), key=lambda f: (f.name != "gui_run.json", _mtime(f))):
            tag = f.stem[len("gui_run"):].lstrip("_")
            log = p / (f"gui_log_{tag}.txt" if tag else "gui_log.txt")
            try:
                run = json.loads(f.read_text())
                if not isinstance(run, dict):
                    raise ValueError
            except (OSError, ValueError):
                run = {"_error": f"{f.name} 无法解析"}
            runs.append(dict(json=f.name, log=log.name if log.is_file() else None, run=run, status=_log_status(log)))
        head = None
        for name in SUMMARY_FILES:
            f = p / name
            if f.is_file() and f.stat().st_size < 8 * 1024 * 1024:
                try:
                    head = (name, _headline(json.loads(f.read_text())))
                    break
                except (OSError, ValueError):
                    continue
        return dict(n_files=n_files, n_dirs=n_dirs, total=total, latest=latest, truncated=truncated,
                    images=images[:self.MAX_THUMBS], n_images=len(images), images_direct=bool(direct),
                    children=children, runs=runs, head=head)

    def _clear(self):
        while self.col.count():
            it = self.col.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        self.thumbs = []

    def _section(self, text):
        self.col.addWidget(label(text, "sectionHeader"))

    def _populate(self, d: dict):
        self.data = d
        self._clear()
        p = self.path
        # ---- overview tiles (4 in a row, 2 x 2 in narrow windows: see _reflow)
        tw = QWidget()
        self._tile_grid = QGridLayout(tw)
        self._tile_grid.setSpacing(8)
        self._tile_grid.setContentsMargins(0, 0, 0, 0)
        self._tiles, self._kv, self._kv_grid, self._cols = [], [], None, None
        lt = time.localtime(d["latest"]) if d["latest"] else None
        for lab, val, hint in (
                ("文件", f"{d['n_files']}" + ("+" if d["truncated"] else ""), f"{d['n_dirs']} 个子文件夹"),
                ("总大小", human_size(d["total"]), "含子文件夹"),
                ("图像", str(d["n_images"]), "在此文件夹" if d["images_direct"] else
                 ("在子文件夹中" if d["n_images"] else "")),
                ("最近修改", time.strftime("%Y-%m-%d", lt) if lt else "—", time.strftime("%H:%M:%S", lt) if lt else "")):
            tile = MetricTile(lab, val, hint)
            tile.hint.setWordWrap(False)        # short one-line hints: wrapped ones were clipped in narrow windows
            self._tiles.append(tile)
        self.col.addWidget(tw)
        # ---- provenance (one card per job that wrote into this folder)
        for r in d.get("runs", []):
            self.col.addWidget(self._run_card(r["run"], r["status"], p, r["json"], r["log"]))
        # ---- summary headline
        if d.get("head"):
            name, items = d["head"]
            c = QFrame()
            c.setObjectName("card")
            cl = QVBoxLayout(c)
            cl.setContentsMargins(14, 10, 14, 12)
            cl.setSpacing(6)
            top = QHBoxLayout()
            top.addWidget(label(f"{name} 摘要", "cardTitle"), 1)
            top.addWidget(_flat(f"打开 {name}", lambda _=False, f=str(p / name): self.open_requested.emit(f)))
            cl.addLayout(top)
            g = QGridLayout()
            g.setHorizontalSpacing(14)
            g.setVerticalSpacing(4)
            for k, v in items:
                kl = label(k, "muted")
                vl = label(v, selectable=True)
                vl.setWordWrap(True)
                self._kv.append((kl, vl))
            self._kv_grid = g
            cl.addLayout(g)
            self.col.addWidget(c)
        # ---- thumbnails
        if d["images"]:
            more = f"（显示前 {len(d['images'])} 个）" if d["n_images"] > len(d["images"]) else ""
            self._section(f"图像  {d['n_images']}{more}" + ("" if d["images_direct"] else "  ·  来自子文件夹"))
            grid = QWidget()
            flow = FlowLayout(grid, spacing=10)
            gen = self._gen
            for f in d["images"]:
                cap = os.path.relpath(f, p)
                t = _Thumb(f, cap)
                t.clicked.connect(lambda f=f: self.open_requested.emit(f))
                flow.addWidget(t)
                self.thumbs.append(t)
                run_async(_thumb, f, _Thumb.W * 2, _Thumb.H * 2, on_done=_weak_call(weakref.ref(self), "_thumb_done",
                                                                            weakref.ref(t), gen))
            self.col.addWidget(grid)
        # ---- contents
        ch = d["children"]
        self._section(f"内容  {len(ch)} 项")
        tree = QTreeWidget()
        tree.setHeaderLabels(["名称", "大小", "修改时间"])
        tree.setRootIsDecorated(False)
        tree.setUniformRowHeights(True)
        tree.setStyleSheet(ROW_QSS)
        tree.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        hd = tree.header()
        hd.setStretchLastSection(False)
        hd.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hd.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hd.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        shown = ch[:300]
        for path, isdir, size, mtime, cnt in shown:
            it = QTreeWidgetItem([os.path.basename(path) + ("/" if isdir else ""),
                                  (f"{cnt} 个文件 · " if isdir else "") + human_size(size), fmt_time(mtime)])
            it.setIcon(0, _icon_for(path))
            it.setData(0, Qt.ItemDataRole.UserRole, path)
            it.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            tree.addTopLevelItem(it)
        if len(ch) > len(shown):
            it = QTreeWidgetItem([f"… 还有 {len(ch) - len(shown)} 项", "", ""])
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            tree.addTopLevelItem(it)
        tree.itemClicked.connect(lambda it, _c: it.data(0, Qt.ItemDataRole.UserRole) and
                                 self.open_requested.emit(it.data(0, Qt.ItemDataRole.UserRole)))
        tree.setCursor(Qt.CursorShape.PointingHandCursor)
        rows = max(1, tree.topLevelItemCount())
        row_h = max(tree.sizeHintForRow(0), 22) if tree.topLevelItemCount() else 22
        tree.setFixedHeight(rows * row_h + tree.header().sizeHint().height() + 6)
        if not ch:
            self.col.addWidget(label("（空文件夹）", "muted"))
        else:
            self.col.addWidget(tree)
        self.col.addStretch(1)
        self._reflow()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._reflow()

    def _reflow(self):
        """Tiles 4 → 2 per row and summary key / value pairs 2 → 1 per row when the preview is narrow."""
        w = self.width()
        cols = (4 if w >= 560 else 2, 2 if w >= 640 and len(self._kv) > 6 else 1)
        if cols == self._cols or self._tile_grid is None:
            return
        self._cols = cols
        tc, kc = cols
        for i, t in enumerate(self._tiles):
            self._tile_grid.removeWidget(t)
            self._tile_grid.addWidget(t, i // tc, i % tc)
        for c in range(4):
            self._tile_grid.setColumnStretch(c, 1 if c < tc else 0)
        g = self._kv_grid
        if g is not None:
            rows = (len(self._kv) + kc - 1) // kc
            for i, (kl, vl) in enumerate(self._kv):
                g.removeWidget(kl)
                g.removeWidget(vl)
                r, c = (i % rows, 2 * (i // rows))
                g.addWidget(kl, r, c, Qt.AlignmentFlag.AlignTop)
                g.addWidget(vl, r, c + 1, Qt.AlignmentFlag.AlignTop)
            for c in range(4):
                g.setColumnStretch(c, 1 if c in (1, 3) and c < 2 * kc else 0)

    def _thumb_done(self, tref, gen, img):
        t = tref()
        if t is not None and _alive(t) and gen == self._gen:
            t.set_image(img)

    def _run_card(self, run: dict, status, p: Path, json_name="gui_run.json", log_name="gui_log.txt") -> QFrame:
        c = QFrame()
        c.setObjectName("card")
        cl = QVBoxLayout(c)
        cl.setContentsMargins(14, 10, 14, 12)
        cl.setSpacing(6)
        top = QHBoxLayout()
        tag = Path(json_name).stem[len("gui_run"):].lstrip("_")
        top.addWidget(label(f"运行记录 · {tag}" if tag else "运行记录", "cardTitle"))
        if "_error" in run:
            top.addWidget(label(run["_error"], "badgeErr"))
            top.addStretch(1)
            cl.addLayout(top)
            return c
        job = _active_job(p, run)
        if job is not None:            # a re-run rewrote gui_run.json; the old log may still say FAILED
            from ..jobs import STATUS_ZH
            st, obj = STATUS_ZH.get(job.status, "运行中"), "badgeWarn"
        elif run.get("finished"):
            st, obj = "已完成", "badgeOk"
        elif status == "failed":
            st, obj = "失败", "badgeErr"
        elif status == "cancelled":
            st, obj = "已取消", "badgeWarn"
        else:
            st, obj = "未完成（中断或仍在运行）", "badgeWarn"
        top.addWidget(label(st, obj))
        top.addStretch(1)
        if log_name and (p / log_name).is_file():
            top.addWidget(_flat("查看日志", lambda _=False, f=str(p / log_name): self.open_requested.emit(f)))
        top.addWidget(_flat(json_name, lambda _=False, f=str(p / json_name): self.open_requested.emit(f)))
        cl.addLayout(top)
        g = QGridLayout()
        g.setHorizontalSpacing(14)
        g.setVerticalSpacing(4)
        script = run.get("script", "")
        if run.get("func") is None and "argv" in run:
            call = f"python {script} " + " ".join(map(str, run.get("argv", [])))
        else:
            call = f"{script} : {run.get('func', 'main')}()"
        params = []
        if run.get("args"):
            params.append("args = " + ", ".join(json.dumps(a, ensure_ascii=False) for a in run["args"]))
        for k, v in (run.get("kwargs") or {}).items():
            if k != "out":
                params.append(f"{k} = {json.dumps(v, ensure_ascii=False)}")
        rows = [("任务", run.get("title", "")), ("调用", call), ("参数", "\n".join(params) or "（默认）"),
                ("输出", run.get("out", "")), ("开始", run.get("started", "—")),
                ("结束", run.get("finished", "—")),
                ("用时", _wall(run.get("wall_s")))]
        for i, (k, v) in enumerate(rows):
            g.addWidget(label(k, "muted"), i, 0, Qt.AlignmentFlag.AlignTop)
            vl = label(str(v), selectable=True)
            vl.setWordWrap(True)
            g.addWidget(vl, i, 1)
        g.setColumnStretch(1, 1)
        cl.addLayout(g)
        return c


# ---------------------------------------------------------------------------------------------- other files
class InfoView(FileView):
    """Fallback for files that cannot be previewed: type, size, date and an 'open with default app' button."""

    def __init__(self, parent=None, file_actions=True):
        super().__init__(parent)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addStretch(1)
        self.icon = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.icon)
        self.title = label("", "cardTitle")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.title)
        self.desc = label("", "muted", wrap=True)
        self.desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.desc)
        row = QHBoxLayout()
        row.addStretch(1)
        b = QPushButton("用默认程序打开")
        b.clicked.connect(lambda: self.path and open_external(self.path))
        row.addWidget(b)
        if file_actions:
            b2 = QPushButton("在访达中显示")
            b2.clicked.connect(lambda: self.path and reveal(self.path))
            row.addWidget(b2)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addStretch(2)
        self._set_content(w)

    def _read(self, p):
        st = p.stat()
        return st.st_size, st.st_mtime

    def _populate(self, data):
        size, mtime = data
        self.icon.setPixmap(_icon_for(self.path).pixmap(64, 64))
        self.title.setText(self.path.name)
        self.desc.setText(f"{describe(self.path)}  ·  {human_size(size)}  ·  修改于 {fmt_time(mtime)}\n\n"
                          "此类型的文件无法在应用内预览。")

    def load(self, path) -> bool:
        p = Path(path)
        if p.is_dir():
            return super().load(p)
        self.path = p
        if not p.exists():
            self.message(f"文件不存在：\n{paths.rel(p)}")
            return False
        return self._finish(self._gen, self._read(p))


# ---------------------------------------------------------------------------------------------- compare
class CompareView(QWidget):
    """Two images side by side with synchronised zoom / pan. Signal `closed` when the user leaves the comparison."""
    closed = Signal()

    def __init__(self, a=None, b=None, parent=None):
        super().__init__(parent)
        self._syncing = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        bar = QHBoxLayout()
        bar.addWidget(label("并排对比", "cardTitle"))
        self.sync = QCheckBox("同步缩放与平移")
        self.sync.setChecked(True)
        bar.addWidget(self.sync)
        bar.addStretch(1)
        bar.addWidget(_flat("适应窗口", self.fit))
        bar.addWidget(_flat("交换左右", self.swap))
        bar.addWidget(_flat("退出对比", self.closed.emit))
        lay.addLayout(bar)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        self.sides = []
        for side in range(2):
            w = QWidget()
            wl = QVBoxLayout(w)
            wl.setContentsMargins(0, 0, 0, 0)
            wl.setSpacing(4)
            row = QHBoxLayout()
            cap = label("", "muted")
            cap.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            row.addWidget(cap, 1)
            row.addWidget(_flat("更换…", lambda _=False, s=side: self._pick(s)))
            wl.addLayout(row)
            iv = ImageView(toolbar=False)
            wl.addWidget(iv, 1)
            split.addWidget(w)
            self.sides.append((iv, cap))
            iv.view.viewport().installEventFilter(self)
            for sb in (iv.view.horizontalScrollBar(), iv.view.verticalScrollBar()):
                sb.valueChanged.connect(lambda *_, s=side: self._sync_scroll(s))
        lay.addWidget(split, 1)
        self.paths = [None, None]
        self.set_images(a, b)

    def set_images(self, a, b):
        for i, p in enumerate((a, b)):
            self.set_image(i, p)

    def set_image(self, side: int, path):
        iv, cap = self.sides[side]
        self.paths[side] = Path(path) if path else None
        if path:
            iv.load(path)
            cap.setText(paths.rel(path))
            cap.setToolTip(paths.rel(path))
        else:
            iv.clear("点击「更换…」选择图像")
            cap.setText("")

    def _pick(self, side):
        start = self.paths[side] or self.paths[1 - side]
        fn, _ = QFileDialog.getOpenFileName(self, "选择图像", str(start.parent if start else paths.RESULTS),
                                            "图像 (" + " ".join(f"*{e}" for e in sorted(IMAGE_EXT)) + ")")
        if fn:
            self.set_image(side, fn)

    def swap(self):
        a, b = self.paths
        self.set_images(b, a)

    def fit(self):
        for iv, _ in self.sides:
            iv.view.fit()

    def eventFilter(self, obj, e):
        if e.type() in (QEvent.Type.Wheel, QEvent.Type.NativeGesture, QEvent.Type.MouseButtonDblClick):
            for i, (iv, _) in enumerate(self.sides):
                if obj is iv.view.viewport():
                    QTimer.singleShot(0, lambda s=i: self._sync_transform(s))
        return False

    def _sync_transform(self, src):
        if not self.sync.isChecked() or self._syncing:
            return
        a, b = self.sides[src][0].view, self.sides[1 - src][0].view
        self._syncing = True
        b.setTransform(a.transform())
        b._fit = a._fit
        self._syncing = False
        self._sync_scroll(src)

    def _sync_scroll(self, src):
        if not self.sync.isChecked() or self._syncing:
            return
        a, b = self.sides[src][0].view, self.sides[1 - src][0].view
        self._syncing = True
        for sa, sb in ((a.horizontalScrollBar(), b.horizontalScrollBar()),
                       (a.verticalScrollBar(), b.verticalScrollBar())):
            sb.setValue(round(sa.value() / sa.maximum() * sb.maximum()) if sa.maximum() > 0 else 0)
        self._syncing = False


# ---------------------------------------------------------------------------------------------- dispatcher
def preview_for(path, parent=None, file_actions=True) -> FileView:
    """The right preview widget for `path` (already loading it).

    png/jpg/svg… → ImageFileView, .json → JsonView, .csv/.tsv → CsvView, .md → MarkdownView, .npz/.npy → NpzView,
    txt/log/m/py and other text → TextView, folder → FolderView, anything else → InfoView.
    `file_actions=False` hides the views' own reveal / open buttons when the host provides them.
    """
    p = Path(path)
    k = kind_of(p) if p.exists() else ("folder" if not p.suffix else "other")
    if k == "image":
        w = ImageFileView(parent, file_actions=file_actions)
    elif k == "json":
        w = JsonView(parent)
    elif k == "csv":
        w = CsvView(parent)
    elif k == "markdown":
        w = MarkdownView(parent)
    elif k == "npz":
        w = NpzView(parent)
    elif k == "text":
        w = TextView(parent, log=_is_log(p))
    elif k == "folder":
        w = FolderView(parent)
    else:
        w = InfoView(parent, file_actions=file_actions)
    w.load(p)
    return w
