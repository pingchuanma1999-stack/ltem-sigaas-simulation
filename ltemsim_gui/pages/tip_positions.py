"""Per-tip-position chain, route 2 (scripts/run_tip_positions.py: simulate → demod).

For every beam position X and every apex height h one PulseSim with the tip as an equipotential body in Poisson
(optionally with the per-height FDTD pump map) records Q_tip(t); demod projects Q_tip(h(θ), t) on cos nθ.
"""
from __future__ import annotations

import csv
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QBoxLayout, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                               QScrollArea, QSizePolicy, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)

from .. import icons, paths
from ..jobs import CANCELLED, FAILED, FINISHED, RUNNING, STATUS_ZH, JobManager
from ..paramspec import FieldSpec
from ..theme import ThemeManager, repolish
from ..widgets.common import MetricTile, Page, button, card, error_box, label, reveal
from ..widgets.fileviews import FlowLayout
from ..widgets.imageview import ImageView
from ..widgets.mpl import PlotWidget
from ..widgets.paramform import ParamForm
from ..widgets.runpanel import RunPanel, fmt_elapsed
from ..workers import run_async

SCRIPT = "scripts/run_tip_positions.py"
# the script's defaults (not imported: it calls matplotlib.use("Agg") and imports pyplot at import time)
HEIGHTS_NM = (10, 15, 25, 45, 80, 130, 200, 270, 330, 370)
X_LIST = (-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5)
COMMITTED = paths.RESULTS / "tip_positions"
FDTD_HEIGHTS = paths.RESULTS / "fdtd_pump" / "heights"
FDTD_REF = paths.RESULTS / "fdtd_pump" / "notip.npz"
RUN_RE = re.compile(r"^X=([+-]\d+\.\d) h=\s*(\d+) nm:")
HARMONICS = (1, 2, 3, 4)
FDTD_SECONDS = (75.0, 90.0)          # one run_fdtd_heights height on this machine
DEFAULT_SERIES = {"det1", "det2", "det3", "det4", "det3_d1", "far", "meas+", "meas-"}
NARROW_PX = 620                      # plots narrower than this get a compact layout (legend below, short titles)


def run_seconds(t_end_ps: float):
    """(low, high) wall time of one PulseSim incl. setup: measured 2 s per 0.5 ps run and 17–20 s per 4 ps run here,
    32–40 s per 4 ps run in the original logs."""
    return 0.5 + 4.2 * t_end_ps, 2.0 + 12.0 * t_end_ps


def _dur(s: float) -> str:
    if s < 90:
        return f"{max(1, round(s))} s"
    if s < 5400:
        return f"{round(s / 60)} min"
    return f"{s / 3600:.1f} h"


def fmt_range(lo: float, hi: float) -> str:
    a, b = _dur(lo), _dur(hi)
    if a.split()[1] == b.split()[1]:
        return a if a == b else f"{a.split()[0]}–{b}"
    return f"{a} – {b}"


def npz_name(X, h) -> str:
    return f"X{float(X):+.1f}_h{int(h)}.npz"


def fx(X) -> str:
    """Signed beam position for display ('0' rather than '+0')."""
    return "0" if float(X) == 0 else f"{float(X):+g}"


def xkey(X) -> str:
    return f"{float(X):+.1f}"


def read_json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def demod_name(A, h_min) -> str:
    return f"demod_A{float(A):g}_h{float(h_min):g}.json"


def demod_files(d: Path):
    def key(p):
        m = re.match(r"demod_A([\d.]+)_h([\d.]+)\.json", p.name)
        return (float(m.group(1)), float(m.group(2))) if m else (1e9, 1e9)
    return sorted(Path(d).glob("demod_A*_h*.json"), key=key)


def npz_count(d: Path, xs, hs) -> int:
    return sum((Path(d) / npz_name(X, h)).exists() for X in xs for h in hs)


def _inside(d, root: Path) -> bool:
    try:
        Path(d).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def is_committed(d: Path) -> bool:
    """A hand-curated (committed) result folder: anywhere under results/ except results/gui_runs/."""
    return _inside(d, paths.RESULTS) and not _inside(d, paths.GUI_RUNS)


def fdtd_ref_for(folder: Path) -> Path:
    """The no-tip reference the ratio maps in `folder` were computed with (its gui_run.json), else notip.npz."""
    for name in ("gui_run.json", "gui_run_fdtd_heights_summary.json"):
        ref = ((read_json(Path(folder) / name) or {}).get("kwargs") or {}).get("ref")
        if ref:
            return paths.absolute(ref)
    return FDTD_REF


def quarantine_ratio(folder: Path):
    """A run_fdtd_heights height killed while being written leaves a broken ratio_h*.npz that both scripts would
    treat as done: rename the newest one to *.npz.partial when its zip is damaged. Returns the bad name or None."""
    files = sorted(Path(folder).glob("ratio_h*.npz"), key=lambda p: p.stat().st_mtime) if Path(folder).is_dir() else []
    if not files:
        return None
    f = files[-1]
    try:
        with zipfile.ZipFile(f) as z:
            if z.testzip() is None:
                return None
    except (zipfile.BadZipFile, OSError):
        pass
    f.rename(f.with_suffix(".npz.partial"))
    return f.name


class _HomeOnBlur(QObject):
    """Show the start of long tuple fields (X list, heights) whenever they are not being edited."""

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.FocusOut and isinstance(obj, QLineEdit):
            obj.setCursorPosition(0)
        return False


_MEASURED = None


def measured():
    """data/measured_tip_scan.csv → {bias: (X, rms, err)} (parsed here: importing run_baseline pulls in pyplot)."""
    global _MEASURED
    if _MEASURED is None:
        rows = {}
        try:
            with open(paths.MEASURED_CSV, newline="") as fh:
                for r in csv.DictReader(fh):
                    rows.setdefault(round(float(r["bias_V"])), []).append(
                        (float(r["X_um"]), float(r["rms"]), float(r["rms_err"])))
        except (OSError, ValueError, KeyError):
            rows = {}
        _MEASURED = {b: tuple(np.array(c) for c in zip(*v)) for b, v in rows.items()}
    return _MEASURED


def sim_specs():
    S, G = "激发与偏压 (simulate)", "扫描网格 (simulate)"
    return [
        FieldSpec("P_avg_mW", 2.0, "float", "平均泵浦功率", "mW", "80 MHz 重复频率、3 µm 光斑、820 nm。仓库里有 2 mW 和 30 mW 的结果",
                  S, minimum=0.0),
        FieldSpec("V_bias", 25.0, "float", "间隙偏压 V_bias", "V", "阳极固定在 −x 一侧", S),
        FieldSpec("V_tip", 0.0, "float", "针尖电位 V_tip", "V",
                  "相对间隙中点；0 = 接地针尖；P2mW_sym_Vtip12 / P2mW_sym_Vtipm12 分别用 +12.5 / −12.5 V", S),
        FieldSpec("T", 300.0, "float", "温度 T", "K", "载流子/晶格温度（决定 kT/q 扩散和 N_c）", S, minimum=1.0),
        FieldSpec("degenerate", True, "bool", "简并扩散 (Fermi-Dirac)", "", "广义 Einstein 关系；关闭 = 经典 Boltzmann 扩散", S),
        FieldSpec("x_list", X_LIST, "tuple", "光斑位置 X", "µm", "光束相对针尖的位置，逗号分隔；文件名用 X{X:+.1f}", G),
        FieldSpec("heights", HEIGHTS_NM, "tuple", "针尖高度 h", "nm",
                  "敲击周期内的针尖-样品间距，每个高度一次完整运行；文件名用 int(h)", G),
        FieldSpec("t_end_ps", 4.0, "float", "模拟时长 t_end", "ps", "dt = 0.5 fs，每 5 fs 记录一次；运行时间与 t_end 成正比", G,
                  minimum=0.05),
    ]


def demod_specs():
    D = "解调 (demod)"
    return [
        FieldSpec("A_nm", 180.0, "float", "敲击振幅 A", "nm", "h(θ) = h_min + A(1 + cos θ)；h_min + 2A 不应超过最大模拟高度", D,
                  minimum=0.0),
        FieldSpec("h_min_nm", 10.0, "float", "最小高度 h_min", "nm", "敲击周期的最低点", D, minimum=0.0),
        FieldSpec("f_cut_thz", 2.2, "float", "探测器截止频率", "THz", "8 阶 Butterworth 陡峭低通", D, minimum=0.01),
        FieldSpec("deriv", "2", "choice", "检测波形导数阶", "", "2 = 辐射场 ∝ d²S/dt²（当前约定）；1 = 旧的一阶约定", D,
                  choices=("2", "1")),
        FieldSpec("band", (0.3, 2.0), "tuple", "幅度评估频带", "THz", "band_metric 的频带 (f_lo, f_hi)", D, advanced=True),
    ]


# ----------------------------------------------------------------------------- background work
def _load_wave_data(d, xs, hs):
    """Q_tip(t) of every X*_h*.npz present → {"t": {xk: t}, "Q": {xk: {int h: Q}}}."""
    t, Q = {}, {}
    for X in xs:
        row = {}
        for h in hs:
            f = Path(d) / npz_name(X, h)
            if not f.exists():
                continue
            try:
                with np.load(f) as z:
                    row[int(h)] = np.asarray(z["Q_tip"], float)
                    t.setdefault(xkey(X), np.asarray(z["t"], float))
            except Exception:        # a file truncated by a cancelled run
                continue
        if row:
            Q[xkey(X)] = row
    return {"t": t, "Q": Q}


def _wave_work(d, xs, hs, dem, data=None):
    from ltemsim.tip_demod import demodulate_positions, detected_waveform
    if data is None:
        data = _load_wave_data(d, xs, hs)
    det = {}
    for xk, row in data["Q"].items():
        if not all(int(h) in row for h in hs):
            continue
        t = data["t"][xk]
        Q = np.array([row[int(h)] for h in hs])
        S = demodulate_positions(list(hs), Q, A_nm=dem["A_nm"], h_min_nm=dem["h_min_nm"], harmonics=HARMONICS)
        det[xk] = {n: detected_waveform(t, S[n], order=int(dem["deriv"]), f_cut_thz=dem["f_cut_thz"]) for n in HARMONICS}
    return data, det


# ----------------------------------------------------------------------------- small widgets
class _TileRow(QWidget):
    """Metric tiles in one row when there is room, otherwise in rows of three."""

    def __init__(self, tiles, parent=None):
        super().__init__(parent)
        self._tiles = list(tiles)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(8)
        self._cols = 0
        self._arrange(len(self._tiles))

    def _arrange(self, cols):
        if cols == self._cols:
            return
        for c in range(max(self._cols, cols)):
            self._grid.setColumnStretch(c, 1 if c < cols else 0)
        self._cols = cols
        for i, t in enumerate(self._tiles):
            self._grid.addWidget(t, i // cols, i % cols)      # re-adding moves the widget to its new cell

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._arrange(len(self._tiles) if e.size().width() >= 118 * len(self._tiles) else 3)


class _FigTabs(QTabWidget):
    """Plot tabs that take whatever height is left: neither the canvas' 7x4.5 in size hint nor the wrapping series
    checkboxes (height-for-width) may make the results scroll area taller than the window when everything fits."""

    def hasHeightForWidth(self):
        return False

    def sizeHint(self):
        return QSize(super().sizeHint().width(), self.minimumHeight())


class RunGrid(QWidget):
    """X × h matrix of the simulate runs: finished / running / pending."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.xs, self.hs, self.done, self.running = [], [], set(), None
        self.setMinimumSize(250, 150)

    def set_state(self, xs, hs, done, running=None):
        self.xs, self.hs, self.done, self.running = list(xs), list(hs), set(done), running
        self.updateGeometry()
        self.update()

    def sizeHint(self):
        return QSize(58 + 26 * max(1, len(self.hs)), 44 + 20 * max(1, len(self.xs)))

    def paintEvent(self, _e):
        t = ThemeManager.instance().tokens
        nx, nh = len(self.xs), len(self.hs)
        if not nx or not nh:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        f = self.font()
        f.setPixelSize(10)
        p.setFont(f)
        lw, top, bottom = 46.0, 16.0, 22.0
        cw = min(26.0, (self.width() - lw) / nh)
        ch = min(20.0, (self.height() - top - bottom) / nx)
        muted, text = QColor(t.muted), QColor(t.text)
        p.setPen(muted)
        p.drawText(QRectF(0, 0, lw - 4, top), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "X \\ h")
        for j, h in enumerate(self.hs):
            p.drawText(QRectF(lw + j * cw, 0, cw, top), Qt.AlignmentFlag.AlignCenter, f"{float(h):g}")
        for i, X in enumerate(self.xs):
            p.setPen(text)
            p.drawText(QRectF(0, top + i * ch, lw - 6, ch), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, fx(X))
            for j in range(nh):
                r = QRectF(lw + j * cw + 2, top + i * ch + 2, cw - 4, ch - 4)
                if (i, j) in self.done:
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QColor(t.accent))
                elif (i, j) == self.running:
                    p.setPen(QPen(QColor(t.warning), 1.5))
                    p.setBrush(QColor(t.accent_soft))
                else:
                    edge = QColor(t.muted)
                    edge.setAlpha(110)
                    p.setPen(QPen(edge, 1))
                    p.setBrush(QColor(t.surface2))
                p.drawRoundedRect(r, 3, 3)
        y = top + nx * ch + 6
        x = lw
        for txt, fill, pen in ((f"已完成 {len(self.done)}/{nx * nh}", t.accent, None), ("计算中", t.accent_soft, t.warning),
                               ("待算", t.surface2, t.muted)):
            p.setPen(QPen(QColor(pen), 1.5) if pen else Qt.PenStyle.NoPen)
            p.setBrush(QColor(fill))
            p.drawRoundedRect(QRectF(x, y + 3, 10, 10), 2, 2)
            p.setPen(muted)
            w = p.fontMetrics().horizontalAdvance(txt)
            p.drawText(QRectF(x + 14, y, w + 4, 16), Qt.AlignmentFlag.AlignVCenter, txt)
            x += w + 30
        p.end()


class _Step(QFrame):
    """One step of the job chain: status badge, name, elapsed / progress; click → show its log."""
    clicked = Signal()

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("点击查看此步骤的日志")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 3, 2, 3)
        lay.setSpacing(8)
        self.badge = label("—", "badge")
        self.badge.setMinimumWidth(52)
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name = label(title)
        self.info = label("", "muted")
        lay.addWidget(self.badge)
        lay.addWidget(self.name, 1)
        lay.addWidget(self.info)

    def set_job(self, job, focused):
        self.badge.setText(STATUS_ZH.get(job.status, job.status))
        self.badge.setObjectName({FINISHED: "badgeOk", FAILED: "badgeErr", CANCELLED: "badgeWarn"}.get(job.status, "badge"))
        repolish(self.badge)
        self.name.setText(f"<b>{self.title}</b>" if focused else self.title)
        info = fmt_elapsed(job.elapsed()) if job.started else ""
        if job.status == RUNNING and job.progress_text:
            info = f"{job.progress_text} · {info}"
        elif job.status == CANCELLED and job.progress_text == "前置任务未完成":
            info = "前置任务未完成"
        self.info.setText(info)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(e)


# ----------------------------------------------------------------------------- page
class TipPositionsPage(Page):
    title = "针尖位置链"
    subtitle = ("路线 2：对每个光斑位置 X 和敲击高度 h 各跑一次载流子动力学（针尖是 Poisson 中的等势体，可选每个高度的 FDTD 泵浦阴影图），"
                "记录针尖感应电荷 Q_tip(t)，再沿 h(θ) = h_min + A(1 + cos θ) 解调得 S_n(t)，与实测 ±25 V 扫描比较。"
                "默认 70 次运行约 25–60 分钟，解调约 1 秒。")

    def __init__(self):
        super().__init__()
        self.chain: dict = {}
        self.result_dir: Path | None = None
        self.meta: dict = {}
        self.demod: dict | None = None
        self.n_expected = self.n_present = 0
        self._wave = None              # (key, data, det)
        self._wave_gen = 0
        self._fdtd_extra: list[str] = []
        self._narrow = {"amp": None, "wave": None}
        jm = JobManager.instance()

        self.demod_btn = button("仅解调", self.redemod)
        self.header.add_action(self.demod_btn)
        self.run_btn = button("  模拟 + 解调", self.run, "primary")
        self.header.add_action(self.run_btn)

        split = self.split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._build_left())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_run_tab(), "运行")
        self.tabs.addTab(self._build_results_tab(), "结果")
        self.tabs.addTab(self._build_history_tab(), "历史结果")
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([430, 900])
        self._user_split = False             # until the user drags the handle, the form column follows the window width
        split.splitterMoved.connect(lambda *_: setattr(self, "_user_split", True))
        split.installEventFilter(self)
        self.body.addWidget(split, 1)

        jm.job_changed.connect(self._on_job_changed)
        jm.job_finished.connect(self._on_job_finished)
        self.run_btn.setIcon(icons.icon("play", ThemeManager.instance().tokens.on_accent))
        self._refresh_fdtd_choices()
        self.refresh_history()
        default = COMMITTED / "P2mW_sym"
        if default.is_dir():
            self.show_result(default, switch=False)
        else:
            self._update_demod_btn()
        ThemeManager.instance().changed.connect(self._retheme)

    # ------------------------------------------------------------------ layout
    def _build_left(self):
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)
        self.form = ParamForm(sim_specs() + demod_specs())
        self.form.changed.connect(self._update_plan)
        self.form.changed.connect(self._home_tuples)
        self._blur = _HomeOnBlur(self)
        for k in ("x_list", "heights"):
            self.form.rows[k].editor.installEventFilter(self._blur)
        self._home_tuples()
        ll.addWidget(self.form, 1)

        fd = card()
        fl = QVBoxLayout(fd)
        fl.setContentsMargins(12, 10, 12, 10)
        fl.setSpacing(6)
        fl.addWidget(label("泵浦 FDTD 针尖图 (fdtd)", "cardTitle"))
        row = QHBoxLayout()
        self.fdtd_combo = QComboBox()
        self.fdtd_combo.setToolTip("none = 对称高斯泵浦；或含 ratio_h<h>.npz 的文件夹（scripts/run_fdtd_heights.py 的输出）")
        self.fdtd_combo.currentIndexChanged.connect(self._update_plan)
        row.addWidget(self.fdtd_combo, 1)
        row.addWidget(button("选择…", self._pick_fdtd, tooltip="选择含 ratio_h*.npz 的文件夹"))
        fl.addLayout(row)
        self.fdtd_status = label("", wrap=True)
        self.fdtd_status.setTextFormat(Qt.TextFormat.RichText)
        fl.addWidget(self.fdtd_status)
        ll.addWidget(fd)

        orow = QHBoxLayout()
        orow.addWidget(QLabel("输出文件夹"))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("自动：results/gui_runs/tip_<时间>")
        self.out_edit.setToolTip("留空 = 每次运行新建文件夹。选已有文件夹可续算：已存在的 X*_h*.npz 会被脚本跳过")
        self.out_edit.textChanged.connect(self._update_plan)
        orow.addWidget(self.out_edit, 1)
        b = QPushButton("选择…")
        b.clicked.connect(self._pick_out)
        orow.addWidget(b)
        ll.addLayout(orow)
        left.setMinimumWidth(370)
        return left

    def _build_run_tab(self):
        w = QWidget()
        rl = QVBoxLayout(w)
        rl.setContentsMargins(12, 12, 12, 12)
        rl.setSpacing(10)
        plan = card()
        pg = QGridLayout(plan)
        pg.setContentsMargins(16, 12, 16, 12)
        pg.setHorizontalSpacing(14)
        pg.setVerticalSpacing(6)
        pg.addWidget(label("运行计划（「模拟 + 解调」将执行）", "cardTitle"), 0, 0, 1, 2)
        self.plan = {}
        for i, (k, text) in enumerate((("size", "规模"), ("time", "预计耗时"), ("pump", "泵浦"), ("out", "输出"))):
            pg.addWidget(label(text, "muted"), i + 1, 0, Qt.AlignmentFlag.AlignTop)
            v = label("", wrap=True, selectable=True)
            v.setTextFormat(Qt.TextFormat.RichText)
            pg.addWidget(v, i + 1, 1)
            self.plan[k] = v
        self.plan_warn = label("", wrap=True)
        self.plan_warn.setTextFormat(Qt.TextFormat.RichText)
        pg.addWidget(self.plan_warn, 5, 0, 1, 2)
        pg.setColumnStretch(1, 1)
        rl.addWidget(plan)

        self.chain_card = card()
        cl = self._chain_lay = QHBoxLayout(self.chain_card)     # steps above the run grid when narrow (eventFilter)
        cl.setContentsMargins(16, 12, 16, 12)
        cl.setSpacing(18)
        col = QVBoxLayout()
        col.setSpacing(4)
        col.addWidget(label("当前任务链", "cardTitle"))
        self.steps = {}
        for k, title in (("fdtd", "① FDTD 高度图 · run_fdtd_heights"), ("sim", "② 模拟 · simulate"),
                         ("demod", "③ 解调 · demod")):
            st = _Step(title)
            st.clicked.connect(lambda k=k: self._focus_step(k))
            col.addWidget(st)
            self.steps[k] = st
        self.last_line = label("", "muted", wrap=True, selectable=True)
        col.addWidget(self.last_line)
        col.addStretch(1)
        cl.addLayout(col, 1)
        self.grid_w = RunGrid()
        cl.addWidget(self.grid_w)
        self.chain_card.setVisible(False)
        self.chain_card.installEventFilter(self)
        rl.addWidget(self.chain_card)

        self.panel = RunPanel()
        self.panel.title.setWordWrap(True)       # long job titles must not widen the whole right-hand pane
        self.panel.setMinimumHeight(240)
        rl.addWidget(self.panel, 1)
        sc = QScrollArea()                       # small windows: scroll instead of clipping the plan / chain cards
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        return sc

    def _build_results_tab(self):
        w = QWidget()
        xl = QVBoxLayout(w)
        xl.setContentsMargins(12, 12, 12, 12)
        xl.setSpacing(8)
        sc = QScrollArea()                 # small windows: scroll instead of squeezing the plots to nothing
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        top = QHBoxLayout()
        self.res_title = label("尚未载入结果", "cardTitle", wrap=True)
        self.res_title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        top.addWidget(self.res_title, 1)
        self.btn_params = button("载入参数到表单", self._load_result_params, "flat",
                                 "把此结果的 meta.json 与所选解调参数填入左侧表单（输出文件夹不变）")
        self.btn_resume = button("续算此文件夹", self._resume, "flat", "载入参数并把输出文件夹设为此文件夹；已存在的 npz 会被跳过")
        self.btn_redemod = button("以左侧参数重新解调", self.redemod, "flat")
        self.btn_folder = button("打开文件夹", lambda: self.result_dir and reveal(self.result_dir), "flat")
        for b in (self.btn_params, self.btn_folder):
            top.addWidget(b)
        xl.addLayout(top)
        self.res_note = label("", "muted", wrap=True)
        self.res_note.setVisible(False)
        xl.addWidget(self.res_note)

        self.tiles = {k: MetricTile(text, "—") for k, text in (
            ("P", "泵浦功率 P_avg"), ("Vb", "间隙偏压 V_bias"), ("Vt", "针尖电位 V_tip"), ("pump", "泵浦 (fdtd)"),
            ("h", "针尖高度"), ("X", "光斑位置 X"))}
        xl.addWidget(_TileRow(self.tiles.values()))

        drow = QWidget()                     # wraps the buttons below the selector in narrow windows
        dflow = FlowLayout(drow, spacing=8)
        sel = QWidget()
        sl = QHBoxLayout(sel)
        sl.setContentsMargins(0, 0, 8, 0)
        sl.addWidget(label("解调结果", "muted"))
        self.demod_combo = QComboBox()
        self.demod_combo.setMinimumWidth(210)
        self.demod_combo.currentIndexChanged.connect(self._on_demod_selected)
        sl.addWidget(self.demod_combo)
        dflow.addWidget(sel)
        dflow.addWidget(self.btn_resume)
        dflow.addWidget(self.btn_redemod)
        xl.addWidget(drow)
        self.demod_desc = label("", "muted", wrap=True)
        self.demod_desc.setTextFormat(Qt.TextFormat.RichText)
        xl.addWidget(self.demod_desc)

        self.fig_tabs = _FigTabs()
        self.fig_tabs.addTab(self._build_amp_tab(), "幅度 vs X")
        png = QWidget()
        pl = QVBoxLayout(png)
        pl.setContentsMargins(6, 6, 6, 6)
        self.png_view = ImageView()
        if self.png_view.caption is not None:      # a long file-name caption must not widen the whole results pane
            self.png_view.caption.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        pl.addWidget(self.png_view, 1)
        note = label("tip_positions_A{A}.png 的文件名不含 h_min：同一 A、不同 h_min 的解调会互相覆盖此图（以最近一次为准）。",
                     "muted", wrap=True)
        note.setTextFormat(Qt.TextFormat.PlainText)
        pl.addWidget(note)
        self.fig_tabs.addTab(png, "脚本图")
        self.fig_tabs.addTab(self._build_wave_tab(), "波形 Q_tip(t)")
        self.fig_tabs.addTab(self._build_table_tab(), "数值表")
        self.fig_tabs.setMinimumHeight(450)
        self.fig_tabs.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        xl.addWidget(self.fig_tabs, 1)
        return sc

    def _build_amp_tab(self):
        w = QWidget()
        al = QVBoxLayout(w)
        al.setContentsMargins(8, 8, 8, 4)
        al.setSpacing(4)
        self.series = {}
        groups = (
            ("检测 d^kS_n/dt^k  n =", [(f"det{n}", str(n), f"det{n}：检测波形 d^kS_{n}/dt^k 在评估频带内的幅度（k = 解调导数阶）")
                                       for n in HARMONICS]),
            ("一阶约定 dS_n/dt  n =", [(f"det{n}_d1", str(n), f"det{n}_d1：一阶导数约定 dS_{n}/dt 的频带幅度") for n in HARMONICS]),
            ("", [("far", "远场 dJ/dt", "far：远场参考 dJ/dt 的频带幅度")]),
            ("峰值 |S_n|  n =", [(f"S{n}", str(n), f"S{n}：解调信号 S_{n}(t) 的峰值") for n in HARMONICS]),
            ("", [("meas+", "实测 +25 V", "data/measured_tip_scan.csv，bias = +25 V"),
                  ("meas-", "实测 −25 V (X→−X)", "bias = −25 V，按 X → −X 镜像")]),
        )
        self.series_box = QWidget()
        flow = FlowLayout(self.series_box, spacing=6)        # groups wrap as a whole in narrow windows
        flow.setContentsMargins(0, 0, 0, 0)

        def group():
            g = QWidget()
            gl = QHBoxLayout(g)
            gl.setContentsMargins(0, 0, 18, 0)
            gl.setSpacing(6)
            flow.addWidget(g)
            return gl
        for gname, items in groups:
            gl = group()
            if gname:
                gl.addWidget(label(gname, "muted"))
            for key, text, tip in items:
                c = QCheckBox(text)
                c.setToolTip(tip)
                c.setChecked(key in DEFAULT_SERIES)
                c.toggled.connect(self._redraw_amp)
                gl.addWidget(c)
                self.series[key] = c
        self.abs_box = QCheckBox("原始 rms · 对数轴")
        self.abs_box.setToolTip("画原始 rms（各量单位不同，只在同一组内可比）；实测点只在归一化模式下显示")
        self.abs_box.toggled.connect(self._redraw_amp)
        group().addWidget(self.abs_box)
        al.addWidget(self.series_box)
        self.amp_stack = QStackedWidget()
        self.amp_empty = label("", "muted", wrap=True)
        self.amp_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.amp_stack.addWidget(self.amp_empty)
        self.amp_plot = PlotWidget()
        self.amp_plot.installEventFilter(self)
        self.amp_stack.addWidget(self.amp_plot)
        al.addWidget(self.amp_stack, 1)
        return w

    def _build_wave_tab(self):
        w = QWidget()
        wl = QVBoxLayout(w)
        wl.setContentsMargins(8, 8, 8, 4)
        wl.setSpacing(4)
        ctl = QWidget()
        flow = FlowLayout(ctl, spacing=16)
        self.wave_x = QComboBox()
        self.wave_x.currentIndexChanged.connect(lambda *_: self._redraw_wave())
        self.wave_n = QComboBox()
        self.wave_n.addItems([str(n) for n in HARMONICS])
        self.wave_n.setCurrentText("3")
        self.wave_n.currentIndexChanged.connect(lambda *_: self._redraw_wave())
        for text, combo in (("(a) 光斑位置 X", self.wave_x), ("(b) 谐波 n", self.wave_n)):
            g = QWidget()
            gl = QHBoxLayout(g)
            gl.setContentsMargins(0, 0, 0, 0)
            gl.setSpacing(6)
            gl.addWidget(label(text, "muted"))
            gl.addWidget(combo)
            flow.addWidget(g)
        self.wave_status = label("", "muted")
        self.wave_status.setMinimumHeight(self.wave_x.sizeHint().height())
        flow.addWidget(self.wave_status)
        wl.addWidget(ctl)
        self.wave_stack = QStackedWidget()
        self.wave_empty = label("", "muted", wrap=True)
        self.wave_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wave_stack.addWidget(self.wave_empty)
        self.wave_plot = PlotWidget()
        self.wave_plot.installEventFilter(self)
        self.wave_stack.addWidget(self.wave_plot)
        wl.addWidget(self.wave_stack, 1)
        return w

    def _build_table_tab(self):
        w = QWidget()
        tl = QVBoxLayout(w)
        tl.setContentsMargins(8, 8, 8, 8)
        row = QHBoxLayout()
        self.table_note = label("每列按其最大值归一化 (rms_norm)", "muted", wrap=True)
        row.addWidget(self.table_note, 1)
        row.addWidget(button("复制为 TSV", self._copy_table, "flat", "复制到剪贴板，可直接粘贴到 Excel / Origin"))
        tl.addLayout(row)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        tl.addWidget(self.table, 1)
        return w

    def _build_history_tab(self):
        w = QWidget()
        hl = QVBoxLayout(w)
        hl.setContentsMargins(12, 12, 12, 12)
        hrow = QHBoxLayout()
        hrow.addWidget(label("已有的针尖位置链结果（双击载入）", "cardTitle", wrap=True), 1)
        hrow.addWidget(button("打开其他文件夹…", self._open_other, "flat"))
        hrow.addWidget(button("刷新", self.refresh_history, "flat"))
        hl.addLayout(hrow)
        hl.addWidget(label("仓库中的 results/tip_positions/* 只含 meta.json、解调 JSON 和 PNG（原始 X*_h*.npz 被 git 忽略）；"
                           "GUI 运行的结果在 results/gui_runs/tip_*。", "muted", wrap=True))
        self.hist = QListWidget()
        self.hist.setSpacing(2)
        self.hist.itemDoubleClicked.connect(self._hist_open)
        hl.addWidget(self.hist, 1)
        return w

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.Resize:
            if obj is self.chain_card:
                d = QBoxLayout.Direction.TopToBottom if obj.width() < 560 else QBoxLayout.Direction.LeftToRight
                if self._chain_lay.direction() != d:
                    self._chain_lay.setDirection(d)
            elif obj is self.split and not self._user_split:
                QTimer.singleShot(0, self._fit_split)
            elif obj is self.amp_plot and self._narrow["amp"] not in (None, obj.width() < NARROW_PX):
                self._narrow["amp"] = None
                QTimer.singleShot(0, self._redraw_amp)
            elif obj is self.wave_plot and self._narrow["wave"] not in (None, obj.width() < NARROW_PX):
                self._narrow["wave"] = None
                QTimer.singleShot(0, self._redraw_wave)
        return super().eventFilter(obj, ev)

    def _fit_split(self):
        """Form column: 390 px in wide windows, 380 px (the narrowest without a horizontal scroll bar) in small ones."""
        tot = sum(self.split.sizes())
        if tot > 0 and not self._user_split:
            left = int(min(390, max(380, tot - 520)))
            if self.split.sizes()[0] != left:
                self.split.setSizes([left, tot - left])

    def _retheme(self, *_):
        t = ThemeManager.instance().tokens
        self.run_btn.setIcon(icons.icon("play", t.on_accent))
        self._update_plan()
        self._update_note_colors()
        self.grid_w.update()

    # ------------------------------------------------------------------ form values
    def _values(self):
        """(simulate kwargs without out/fdtd, demod kwargs without out); raises ValueError with a Chinese message."""
        if not self.form.is_valid():
            raise ValueError("以下参数的值无法解析：" + ", ".join(self.form.invalid_fields()))
        v = self.form.values()
        sim = {k: v[k] for k in ("P_avg_mW", "degenerate", "T", "V_bias", "V_tip", "t_end_ps")}
        sim["degenerate"] = bool(sim["degenerate"])
        sim["heights"] = list(v["heights"])
        sim["x_list"] = [float(x) for x in v["x_list"]]
        dem = dict(A_nm=float(v["A_nm"]), h_min_nm=float(v["h_min_nm"]), f_cut_thz=float(v["f_cut_thz"]),
                   deriv=int(v["deriv"]), band=[float(b) for b in v["band"]])
        return sim, dem

    def _problems(self, sim, dem):
        """(blocking errors, warnings) for the grid / tapping range."""
        errs, warns = [], []
        xs, hs = sim["x_list"], sim["heights"]
        if not xs:
            errs.append("光斑位置 X 列表为空")
        if not hs:
            errs.append("针尖高度列表为空")
        if any(h <= 0 for h in hs):
            errs.append("针尖高度必须为正")
        if len({int(h) for h in hs}) != len(hs):
            errs.append("有高度的 int(h) 相同（文件名 X…_h<int(h)>.npz 会冲突），如 12.5 与 12")
        if len({xkey(X) for X in xs}) != len(xs):
            errs.append("有 X 在保留一位小数后相同（文件名 X{X:+.1f} 会冲突）")
        errs += self._demod_errors(dem)
        if hs and dem["h_min_nm"] + 2 * dem["A_nm"] > max(hs) + 1e-9:
            warns.append(f"敲击轨迹最高到 h_min + 2A = {dem['h_min_nm'] + 2 * dem['A_nm']:g} nm，超过最大模拟高度 {max(hs):g} nm："
                         "超出部分按最高高度取值（插值截断）")
        if hs and dem["h_min_nm"] < min(hs) - 1e-9:
            warns.append(f"h_min = {dem['h_min_nm']:g} nm 低于最小模拟高度 {min(hs):g} nm：低于部分按最低高度取值")
        return errs, warns

    @staticmethod
    def _demod_errors(dem):
        errs = []
        if dem["A_nm"] <= 0:
            errs.append("敲击振幅 A 必须大于 0（A = 0 时 S_n 只剩数值噪声，归一化后的曲线没有意义）")
        if len(dem["band"]) != 2 or dem["band"][0] >= dem["band"][1]:
            errs.append("幅度评估频带应为两个递增的数 (f_lo, f_hi)")
        return errs

    def _home_tuples(self):
        for k in ("x_list", "heights"):
            e = self.form.rows[k].editor
            if not e.hasFocus():
                e.setCursorPosition(0)

    def _out_dir(self):
        txt = self.out_edit.text().strip()
        return paths.absolute(txt) if txt else None

    def _new_out(self) -> Path:
        """A fresh results/gui_runs/tip_<time> (a chain queued in the same second has not created its folder yet)."""
        base = out = paths.new_run_dir("tip")
        i = 2
        while self._busy(out) or out.exists():
            out = base.with_name(f"{base.name}_{i}")
            i += 1
        return out

    def _fdtd_value(self) -> str:
        return self.fdtd_combo.currentData() or "none"

    @staticmethod
    def _missing_ratio(folder: Path, hs):
        return [h for h in hs if not (folder / f"ratio_h{int(h)}.npz").exists()]

    @staticmethod
    def _active_fdtd_jobs(folder: Path):
        out = []
        for j in JobManager.instance().active():
            if j.kind == "fdtd_heights" and j.spec.get("script", "").endswith("run_fdtd_heights.py"):
                o = j.spec.get("kwargs", {}).get("out", "results/fdtd_pump/heights")
                if paths.absolute(o).resolve() == folder.resolve():
                    out.append(j)
        return out

    def _covering_fdtd_job(self, folder: Path, missing):
        """The queued/running run_fdtd_heights job into `folder` that will produce every missing map, else None."""
        running = self._active_fdtd_jobs(folder)
        covered = {int(h) for j in running for h in j.spec.get("kwargs", {}).get("heights", HEIGHTS_NM)}
        return running[-1] if running and all(int(h) in covered for h in missing) else None

    @staticmethod
    def _busy(folder: Path):
        """Active tip jobs writing to this folder."""
        r = folder.resolve()
        return [j for j in JobManager.instance().active()
                if j.kind in ("tip_sim", "tip_demod") and j.out and paths.absolute(j.out).resolve() == r]

    # ------------------------------------------------------------------ plan
    def _update_plan(self, *_):
        t = ThemeManager.instance().tokens
        warn = lambda s: f"<span style='color:{t.warning}'>{s}</span>"
        ok = lambda s: f"<span style='color:{t.success}'>{s}</span>"
        try:
            sim, dem = self._values()
        except ValueError as e:
            for k in self.plan:
                self.plan[k].setText("—")
            self.plan_warn.setText(f"<span style='color:{t.danger}'>{e}</span>")
            self.fdtd_status.setText("")
            return
        xs, hs = sim["x_list"], sim["heights"]
        total = len(xs) * len(hs)
        out = self._out_dir()
        present = npz_count(out, xs, hs) if out and out.is_dir() else 0
        remaining = total - present
        xr = f"{fx(min(xs))} … {fx(max(xs))} µm" if xs else "—"
        hr = f"{min(hs):g}–{max(hs):g} nm" if hs else "—"
        size = (f"{len(xs)} 个 X（{xr}）× {len(hs)} 个高度（{hr}）= <b>{total}</b> 次 PulseSim，"
                f"每次 t_end = {sim['t_end_ps']:g} ps")
        if present:
            size += f"；输出文件夹中已有 {present} 个，将跳过，<b>剩余 {remaining}</b> 个"
        self.plan["size"].setText(size)
        lo, hi = run_seconds(sim["t_end_ps"])
        tl, th = remaining * lo + 3, remaining * hi + 6

        fdtd = self._fdtd_value()
        extra = ""
        if fdtd == "none":
            self.plan["pump"].setText("none — 对称高斯泵浦（不含针尖对泵浦光的遮挡）")
            self.fdtd_status.setText("<span>对称高斯泵浦，不使用 FDTD 针尖图。</span>")
        else:
            folder = paths.absolute(fdtd)
            missing = self._missing_ratio(folder, hs)
            running = self._covering_fdtd_job(folder, missing) is not None if missing else False
            if not missing:
                s = short = ok(f"✓ 所选 {len(hs)} 个高度的 ratio_h*.npz 都已就绪")
            else:
                hl = ", ".join(f"{h:g}" for h in missing[:6]) + (" …" if len(missing) > 6 else "")
                where = ("（这是仓库中的文件夹：FDTD 图会算到新的 results/gui_runs/fdtd_heights_&lt;时间&gt;，不改动它）"
                         if is_committed(folder) else "")
                s = warn(f"⚠ 缺少 {len(missing)}/{len(hs)} 个高度的 ratio_h*.npz（h = {hl}）。"
                         + ("已有 FDTD 高度任务在运行/排队，模拟会排在其后。" if running else
                            "脚本会无限等待缺失文件；运行时会提示先排队计算 FDTD 图，或改用 none。" + where))
                short = warn(f"⚠ 缺 {len(missing)}/{len(hs)} 个 ratio 图，" + ("排在正在运行的 FDTD 任务之后" if running else
                                                                               "运行前会先询问：排队计算 FDTD 图，或改用 none"))
                if not running:
                    n_fdtd = len(hs) if is_committed(folder) else len(missing)
                    fl, fh = n_fdtd * FDTD_SECONDS[0], n_fdtd * FDTD_SECONDS[1]
                    tl, th = tl + fl, th + fh
                    extra = f"，其中 FDTD 高度图约 {fmt_range(fl, fh)}"
            self.fdtd_status.setText(s)
            self.plan["pump"].setText(f"FDTD 针尖图 {paths.rel(folder)} — " + short)
        per = f"每次约 {lo:.0f}–{hi:.0f} s" if remaining else "无需模拟"
        self.plan["time"].setText(f"<b>约 {fmt_range(tl, th)}</b>（{per}{extra}）+ 解调约 1–5 s")
        if out is None:
            self.plan["out"].setText("results/gui_runs/tip_&lt;时间&gt;（每次运行自动新建）")
        else:
            o = paths.rel(out)
            if is_committed(out):
                o += " — " + warn("仓库中的结果文件夹：meta.json 会被覆盖")
            elif present:
                o += f" — 续算（已有 {present} 个 npz）"
            self.plan["out"].setText(o)
        errs, warns = self._problems(sim, dem)
        lines = [f"<span style='color:{t.danger}'>✕ {e}</span>" for e in errs] + [warn("⚠ " + w) for w in warns]
        self.plan_warn.setText("<br>".join(lines))
        self.plan_warn.setVisible(bool(lines))

    def _refresh_fdtd_choices(self):
        cur = self.fdtd_combo.currentData() or "none"
        self.fdtd_combo.blockSignals(True)
        self.fdtd_combo.clear()
        self.fdtd_combo.addItem("none — 对称高斯泵浦", "none")
        cands = [FDTD_HEIGHTS] + sorted((p for p in paths.GUI_RUNS.glob("fdtd_heights_*") if p.is_dir()), reverse=True)
        cands += [paths.absolute(p) for p in self._fdtd_extra]
        seen = set()
        for p in cands:
            r = paths.rel(p)
            if r in seen:
                continue
            seen.add(r)
            n = len(list(p.glob("ratio_h*.npz"))) if p.is_dir() else 0
            self.fdtd_combo.addItem(f"{r} · {n} 图", r)
            self.fdtd_combo.setItemData(self.fdtd_combo.count() - 1, f"{r}\n含 {n} 个 ratio_h*.npz", Qt.ItemDataRole.ToolTipRole)
        i = self.fdtd_combo.findData(cur)
        self.fdtd_combo.setCurrentIndex(max(0, i))
        self.fdtd_combo.blockSignals(False)
        self._update_plan()

    def _set_fdtd(self, value):
        value = value or "none"
        if value != "none" and self.fdtd_combo.findData(value) < 0:
            self._fdtd_extra.append(value)
            self._refresh_fdtd_choices()
        self.fdtd_combo.setCurrentIndex(max(0, self.fdtd_combo.findData(value)))

    def _pick_fdtd(self):
        d = QFileDialog.getExistingDirectory(self, "选择含 ratio_h*.npz 的文件夹", str(FDTD_HEIGHTS.parent))
        if d:
            self._set_fdtd(paths.rel(d))

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出文件夹", str(paths.GUI_RUNS if paths.GUI_RUNS.exists() else paths.RESULTS))
        if d:
            self.out_edit.setText(paths.rel(d))

    # ------------------------------------------------------------------ dialogs (overridable in tests)
    def _confirm(self, title, text) -> bool:
        return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes

    def _error(self, title, text):
        error_box(self, title, text)

    def _ask_existing(self, out: Path, diffs, present, remaining) -> str:
        """'go' | 'new' | 'cancel' for an output folder that already holds npz files."""
        m = QMessageBox(self)
        m.setWindowTitle("输出文件夹已有结果")
        if diffs:
            m.setIcon(QMessageBox.Icon.Warning)
            m.setText(f"{paths.rel(out)} 中已有 {present} 个 npz，是用不同参数算的：")
            m.setInformativeText("\n".join(diffs) + "\n\n脚本会直接跳过已存在的 X…_h….npz，新旧结果会混在一起。建议使用新的输出文件夹。")
            b_new = m.addButton("使用新文件夹", QMessageBox.ButtonRole.AcceptRole)
            b_go = m.addButton("仍然续算", QMessageBox.ButtonRole.DestructiveRole)
        else:
            m.setIcon(QMessageBox.Icon.Question)
            m.setText(f"{paths.rel(out)} 中已有 {present} 个 npz（参数一致）。")
            m.setInformativeText(f"脚本会跳过它们，只计算剩余 {remaining} 个（续算）。")
            b_go = m.addButton("续算", QMessageBox.ButtonRole.AcceptRole)
            b_new = m.addButton("使用新文件夹", QMessageBox.ButtonRole.ActionRole)
        m.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        m.exec()
        c = m.clickedButton()
        return "go" if c is b_go else "new" if c is b_new else "cancel"

    def _ask_fdtd(self, folder: Path, missing, sim_time: str, target: Path, todo, ref: Path) -> str:
        """'fdtd' | 'none' | 'cancel' when ratio maps are missing (`todo` heights would be computed into `target`)."""
        hl = ", ".join(f"{h:g}" for h in missing)
        m = QMessageBox(self)
        m.setIcon(QMessageBox.Icon.Warning)
        m.setWindowTitle("缺少 FDTD 针尖图")
        m.setText(f"{paths.rel(folder)} 中缺少 {len(missing)} 个高度的 ratio_h*.npz（h = {hl} nm）。")
        info = ("run_tip_positions.py 在 fdtd 模式下会一直等待这些文件，而且不输出任何信息，所以必须先把它们算好。\n\n")
        b_fdtd = None
        n = len(todo)
        if ref.exists():
            where = paths.rel(target)
            if target != folder:
                where = (f"新文件夹 {paths.rel(target)}（{paths.rel(folder)} 是仓库中的结果文件夹，run_fdtd_heights 会覆盖其中"
                         f"已提交的 summary.json，所以不写入它；新文件夹中要计算全部 {n} 个高度）")
            info += (f"• 先计算 FDTD 图：排队运行 scripts/run_fdtd_heights.py（{n} 个高度 × 75–90 s ≈ "
                     f"{fmt_range(n * FDTD_SECONDS[0], n * FDTD_SECONDS[1])}），写入 {where}，完成后自动用全部 "
                     f"ratio 图重建 summary.json，再开始模拟 + 解调（模拟约 {sim_time}）。\n")
        else:
            info += f"• 无法先计算 FDTD 图：缺少参考场 {paths.rel(ref)}，请先在「泵浦 FDTD」页运行无针尖的 FDTD。\n"
        info += "• 改用 none：对称高斯泵浦（不含针尖对泵浦光的遮挡）。"
        m.setInformativeText(info)
        if ref.exists():
            b_fdtd = m.addButton("先计算 FDTD 图", QMessageBox.ButtonRole.AcceptRole)
        b_none = m.addButton("改用 none", QMessageBox.ButtonRole.ActionRole)
        m.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        m.exec()
        c = m.clickedButton()
        return "fdtd" if (b_fdtd is not None and c is b_fdtd) else "none" if c is b_none else "cancel"

    # ------------------------------------------------------------------ run
    @staticmethod
    def _meta_diff(meta, sim, fdtd):
        if not meta:
            return []
        out = []
        for k in ("P_avg_mW", "V_bias", "V_tip", "T", "t_end_ps"):
            a, b = meta.get(k), sim[k]
            if a is not None and abs(float(a) - float(b)) > 1e-9:
                out.append(f"{k}: {float(a):g} → {float(b):g}")
        if "degenerate" in meta and bool(meta["degenerate"]) != sim["degenerate"]:
            out.append(f"degenerate: {bool(meta['degenerate'])} → {sim['degenerate']}")
        if (meta.get("fdtd") or "none") != fdtd:
            out.append(f"fdtd: {meta.get('fdtd') or 'none'} → {fdtd}")
        return out

    def run(self):
        try:
            sim, dem = self._values()
        except ValueError as e:
            self._error("参数无效", str(e))
            return
        errs, _ = self._problems(sim, dem)
        if errs:
            self._error("无法运行", "\n".join(errs))
            return
        xs, hs = sim["x_list"], sim["heights"]
        total = len(xs) * len(hs)
        fdtd = self._fdtd_value()
        out = self._out_dir()
        if out is None:
            out = self._new_out()
        if self._busy(out):
            self._error("文件夹正在使用", f"{paths.rel(out)} 已有正在运行或排队的针尖位置链任务，请等待其结束或换一个输出文件夹。")
            return
        present = npz_count(out, xs, hs) if out.is_dir() else 0
        if present:
            diffs = self._meta_diff(read_json(out / "meta.json"), sim, fdtd)
            c = self._ask_existing(out, diffs, present, total - present)
            if c == "cancel":
                return
            if c == "new":
                out = self._new_out()
                self.out_edit.clear()
                present = 0
        if is_committed(out) and not self._confirm(
                "写入仓库中的结果？", f"{paths.rel(out)} 是仓库中提交的结果文件夹，模拟会覆盖其中的 meta.json，解调会覆盖 JSON 与 PNG。继续吗？"):
            return
        remaining = total - present
        lo, hi = run_seconds(sim["t_end_ps"])
        sim_time = fmt_range(remaining * lo + 3, remaining * hi + 6)

        jm = JobManager.instance()
        fdtd_job = None
        asked = False
        if fdtd != "none":
            folder = paths.absolute(fdtd)
            missing = self._missing_ratio(folder, hs)
            running = self._active_fdtd_jobs(folder)
            covering = self._covering_fdtd_job(folder, missing) if missing else None
            if covering is not None:
                fdtd_job = covering              # wait for the FDTD job that is already producing these maps
            elif missing:
                asked = True
                # never let run_fdtd_heights rewrite a committed summary.json: compute into a fresh GUI folder instead
                target = paths.new_run_dir("fdtd_heights") if is_committed(folder) else folder
                todo = list(hs) if target != folder else list(missing)
                ref = fdtd_ref_for(folder)
                c = self._ask_fdtd(folder, missing, sim_time, target, todo, ref)
                if c == "cancel":
                    return
                if c == "none":
                    fdtd = "none"
                    self._set_fdtd("none")
                else:
                    fdtd_job = jm.submit(
                        f"FDTD 高度图 · {len(todo)} 个高度 · {target.name}", "fdtd_heights", "scripts/run_fdtd_heights.py",
                        "main", kwargs=dict(out=paths.rel(target), heights=todo, ref=paths.rel(ref)),
                        out=target, progress={"type": "count", "pattern": r"^h = \S+ nm: ", "total": len(todo)},
                        after=running[-1] if (running and target == folder) else None)
                    jm.submit(f"FDTD 高度图 · 重建 summary.json · {target.name}", "fdtd_heights", "ltemsim_gui/gui_tasks.py",
                              "fdtd_heights_summary", kwargs=dict(out=paths.rel(target), ref=paths.rel(ref)),
                              out=target, after=fdtd_job)
                    if target != folder:
                        fdtd = paths.rel(target)
                        self._set_fdtd(fdtd)
        if not asked and hi * remaining > 15 * 60 and not self._confirm(
                "开始长时间运行？", f"将运行 {remaining} 次 PulseSim，预计约 {sim_time}。\n结果写入 {paths.rel(out)}，"
                                  "可随时在任务中心停止；已完成的 npz 会保留，之后可续算。\n\n继续吗？"):
            return

        rel = paths.rel(out)
        pump = "FDTD 图" if fdtd != "none" else "对称泵浦"
        sim_job = jm.submit(
            f"针尖位置链 · 模拟 · {sim['P_avg_mW']:g} mW · V_tip {sim['V_tip']:g} V · {pump} · {len(xs)}×{len(hs)}",
            "tip_sim", SCRIPT, "simulate", kwargs=dict(out=rel, fdtd=fdtd, **sim), out=out,
            progress={"type": "count", "pattern": r"^X=[+-]\d+\.\d h=", "total": max(1, remaining)}, after=fdtd_job)
        demod_job = jm.submit(
            f"针尖位置链 · 解调 · A {dem['A_nm']:g} / h_min {dem['h_min_nm']:g} nm · {out.name}", "tip_demod", SCRIPT, "demod",
            kwargs=dict(out=rel, **dem), out=out, after=sim_job)
        self._set_chain(dict(fdtd=fdtd_job, sim=sim_job, demod=demod_job))
        self.tabs.setCurrentIndex(0)

    def redemod(self):
        d = self.result_dir
        if d is None or not self._can_redemod():
            self._update_demod_btn()
            self._error("无法重新解调", self.demod_btn.toolTip())
            return
        try:
            _, dem = self._values()
        except ValueError as e:
            self._error("参数无效", str(e))
            return
        errs = self._demod_errors(dem)
        if errs:
            self._error("参数无效", "\n".join(errs))
            return
        name = demod_name(dem["A_nm"], dem["h_min_nm"])
        if is_committed(d) and not self._confirm(
                "写入仓库中的结果？", f"{paths.rel(d)} 是仓库中提交的结果文件夹，解调会写入/覆盖其中的 {name} 与 PNG。继续吗？"):
            return
        if (d / name).exists() and not is_committed(d) and not self._confirm(
                "覆盖已有解调结果？", f"{name} 与 tip_positions_A{dem['A_nm']:g}.png 已存在，将被新的解调结果覆盖。继续吗？"):
            return
        job = JobManager.instance().submit(
            f"针尖位置链 · 解调 · A {dem['A_nm']:g} / h_min {dem['h_min_nm']:g} nm · {d.name}", "tip_demod", SCRIPT, "demod",
            kwargs=dict(out=paths.rel(d), **dem), out=d)
        self._set_chain(dict(demod=job))
        self.tabs.setCurrentIndex(0)

    # ------------------------------------------------------------------ chain tracking
    def _set_chain(self, chain):
        old = self.chain.get("sim")
        if old is not None:
            try:
                old.output.disconnect(self._on_sim_output)
            except (RuntimeError, TypeError):
                pass
        self.chain = {k: j for k, j in chain.items() if j is not None}
        sim = self.chain.get("sim")
        if sim is not None:
            sim.output.connect(self._on_sim_output)
        first = next((j for j in self.chain.values() if j.active), None) or list(self.chain.values())[-1]
        self.panel.attach(first)
        self.last_line.setText("")
        self._update_chain()

    def _chain_jobs(self):
        return [j for j in self.chain.values()]

    def _focus_step(self, k):
        j = self.chain.get(k)
        if j is not None:
            self.panel.attach(j)
            self._update_chain()

    def _update_chain(self):
        self._update_demod_btn()             # the shown folder may just have become busy / free
        self.chain_card.setVisible(bool(self.chain))
        for k, st in self.steps.items():
            j = self.chain.get(k)
            st.setVisible(j is not None)
            if j is not None:
                st.set_job(j, j is self.panel.job)
        self._update_grid()

    def _update_grid(self):
        j = self.chain.get("sim")
        self.grid_w.setVisible(j is not None)
        if j is None:
            return
        kw = j.spec.get("kwargs", {})
        out = paths.absolute(kw.get("out", ""))
        xs, hs = kw.get("x_list", X_LIST), kw.get("heights", HEIGHTS_NM)
        done = {(i, k) for i, X in enumerate(xs) for k, h in enumerate(hs) if (out / npz_name(X, h)).exists()}
        running = None
        if j.status == RUNNING:
            running = next(((i, k) for i in range(len(xs)) for k in range(len(hs)) if (i, k) not in done), None)
        self.grid_w.set_state(xs, hs, done, running)

    def _on_sim_output(self, text):
        hits = [l for l in text.split("\n") if RUN_RE.match(l)]
        if hits:
            self.last_line.setText("最近完成：" + hits[-1])
            self._update_grid()

    def _on_job_changed(self, job):
        if job not in self._chain_jobs():
            return
        p = self.panel.job
        if job.status == RUNNING and p is not job and (p is None or p.status != RUNNING):
            self.panel.attach(job)
        self._update_chain()

    def _on_job_finished(self, job):
        if job.kind in ("tip_sim", "tip_demod"):
            self.refresh_history()
        if job.kind == "fdtd_heights":
            self._refresh_fdtd_choices()
        shown = (job.kind in ("tip_sim", "tip_demod") and job.out and self.result_dir is not None
                 and paths.absolute(job.out).resolve() == self.result_dir.resolve())
        if job not in self._chain_jobs():
            if shown:                       # e.g. re-run from the job center into the folder on display
                self.show_result(self.result_dir, switch=False, prefer=self._selected_demod_name())
            return
        if job is self.chain.get("sim") and job.status in (CANCELLED, FAILED) and job.started:
            self._quarantine_partial(job)
        if job is self.chain.get("fdtd") and job.status in (CANCELLED, FAILED) and job.started:
            folder = paths.absolute(job.spec.get("kwargs", {}).get("out", ""))
            run_async(quarantine_ratio, folder, on_done=self._ratio_quarantined)
        self._update_chain()
        if job is self.chain.get("demod") and job.status == FINISHED and job.out:
            kw = job.spec.get("kwargs", {})
            self.show_result(paths.absolute(job.out), prefer=demod_name(kw.get("A_nm", 180), kw.get("h_min_nm", 10)))
        elif shown:
            self.show_result(self.result_dir, switch=False, prefer=self._selected_demod_name())
        self._update_plan()

    def _ratio_quarantined(self, bad):
        if bad:
            self.last_line.setText(f"⚠ FDTD 图 {bad} 在停止时没有写完整，已改名为 *.npz.partial；下次会重新计算这一高度。")
        self._refresh_fdtd_choices()

    def _quarantine_partial(self, job):
        """A run stopped inside np.savez_compressed leaves a truncated npz that simulate would skip forever on resume:
        rename the newest one to *.broken if it cannot be read."""
        out = paths.absolute(job.spec.get("kwargs", {}).get("out", ""))
        files = sorted(out.glob("X*_h*.npz"), key=lambda p: p.stat().st_mtime) if out.is_dir() else []
        if not files:
            return None
        f = files[-1]
        try:
            with np.load(f) as z:
                z["Q_tip"], z["t"]
            return None
        except Exception:
            bad = f.with_name(f.name + ".broken")
            f.rename(bad)
            self.last_line.setText(f"⚠ {f.name} 在停止时没有写完整，已改名为 {bad.name}；续算时会重新计算这一项。")
            return bad

    def _adopt_active_chain(self):
        """Re-attach to a tip chain started earlier (e.g. before this page existed, or re-run from the job center)."""
        act = [j for j in JobManager.instance().active() if j.kind in ("tip_sim", "tip_demod")]
        if not act:
            return
        j = act[-1]
        if j.kind == "tip_demod" and j.after is not None and j.after.kind == "tip_sim":
            sim, demod = j.after, j
        elif j.kind == "tip_sim":
            sim = j
            demod = next((d for d in JobManager.instance().jobs if d.kind == "tip_demod" and d.after is sim), None)
        else:
            sim, demod = None, j
        fdtd = sim.after if sim is not None and sim.after is not None and sim.after.kind == "fdtd_heights" else None
        self._set_chain(dict(fdtd=fdtd, sim=sim, demod=demod))

    # ------------------------------------------------------------------ results
    def show_result(self, d, switch=True, prefer=None):
        d = Path(d)
        if d.is_file():
            prefer = d.name if d.name.startswith("demod_") else prefer
            d = d.parent
        self.result_dir = d
        demods = demod_files(d)
        meta = read_json(d / "meta.json") or (read_json(demods[0]) if demods else None) or {}
        self.meta = meta
        xs, hs = meta.get("X") or [], meta.get("heights_nm") or []
        self.n_expected = len(xs) * len(hs)
        self.n_present = npz_count(d, xs, hs) if self.n_expected else len(list(d.glob("X*_h*.npz")))
        complete = self.n_expected > 0 and self.n_present == self.n_expected

        busy = bool(self._busy(d))
        state = (f"原始 npz {self.n_present}/{self.n_expected}" if self.n_expected else "无 meta.json")
        tag = "仓库结果" if is_committed(d) else "GUI 运行" if d.parent == paths.GUI_RUNS else ""
        self.res_title.setText("  ·  ".join(s for s in (paths.rel(d), tag, state, "运行中" if busy else "") if s))

        P = meta.get("P_avg_mW")
        self.tiles["P"].set(f"{P:g} mW" if P is not None else "—", "3 µm 光斑")
        Vb, T = meta.get("V_bias"), meta.get("T")
        hint = f"{T:g} K · " + ("简并" if meta.get("degenerate") else "经典") if T is not None else ""
        self.tiles["Vb"].set(f"{Vb:g} V" if Vb is not None else "—", hint)
        self.tiles["Vb"].setToolTip("阳极在 −x；T = 载流子温度；简并 = Fermi-Dirac 扩散")
        Vt = meta.get("V_tip")
        self.tiles["Vt"].set((f"{Vt:+g} V" if Vt else "0 V") if Vt is not None else "—", "相对间隙中点")
        fd = meta.get("fdtd")
        sym = fd in (None, "none")
        self.tiles["pump"].set("—" if not meta else ("对称" if sym else "FDTD 图"),
                               "高斯光斑" if sym else Path(str(fd)).name)
        self.tiles["pump"].setToolTip("对称高斯泵浦（无针尖阴影）" if sym else f"每个高度的 FDTD 泵浦比值图：{fd}")
        self.tiles["h"].set(f"{len(hs)} 个" if hs else "—", f"{min(hs):g}–{max(hs):g} nm" if hs else "")
        self.tiles["h"].setToolTip(", ".join(f"{h:g}" for h in hs) + " nm" if hs else "")
        te = meta.get("t_end_ps")
        self.tiles["X"].set(f"{len(xs)} 个" if xs else "—", f"{fx(min(xs))}…{fx(max(xs))} µm" if xs else "")
        self.tiles["X"].setToolTip((", ".join(fx(x) for x in xs) + " µm" + (f"；t_end = {te:g} ps" if te else ""))
                                   if xs else "")

        self.demod_combo.blockSignals(True)
        self.demod_combo.clear()
        for p in demods:
            js = read_json(p) or {}
            self.demod_combo.addItem(f"A = {js.get('A_nm', '?'):g} nm · h_min = {js.get('h_min_nm', '?'):g} nm" if js else p.name,
                                     p.name)
        idx = 0
        for want in (prefer, "demod_A180_h10.json"):
            if want and self.demod_combo.findData(want) >= 0:
                idx = self.demod_combo.findData(want)
                break
        if demods:
            self.demod_combo.setCurrentIndex(idx)
        self.demod_combo.setEnabled(bool(demods))
        self.demod_combo.blockSignals(False)

        if not meta and not demods:
            note = "该文件夹中没有 meta.json 或解调结果 (demod_A*_h*.json)。"
        elif not demods and not self.n_present:
            note = ("该文件夹只有 meta.json：没有解调结果，原始 X*_h*.npz 也不在仓库中（被 git 忽略）。"
                    "要查看结果，请「载入参数到表单」后重新「模拟 + 解调」。")
        elif self.n_present == 0:
            note = ("原始 X*_h*.npz 不在此文件夹中（仓库中被 git 忽略），所以无法用新的 A / h_min 重新解调，也看不到 Q_tip(t) 波形。"
                    "需要时请「载入参数到表单」→「模拟 + 解调」重新计算（默认 70 次运行约 25–60 min）。")
        elif not complete:
            note = (f"原始数据不完整（{self.n_present}/{self.n_expected}）：模拟仍在运行或被中断。"
                    + ("" if busy else "可点「续算此文件夹」用相同参数补算剩余部分（已有的 npz 会被跳过）。"))
        elif not demods:
            note = "模拟已完成，但还没有解调结果：点「以左侧参数重新解调」生成。"
        else:
            note = ""
        self.res_note.setText(note)
        self.res_note.setVisible(bool(note))
        self.btn_params.setEnabled(bool(meta))
        self.btn_resume.setVisible(bool(meta) and not complete and self.n_expected > 0 and not is_committed(d) and not busy
                                   and self.n_present > 0)
        self.btn_folder.setEnabled(d.exists())
        self._wave = None if (self._wave and self._wave[0][0] != str(d)) else self._wave
        self._on_demod_selected()
        self._update_demod_btn()
        if switch:
            self.tabs.setCurrentIndex(1)

    def _selected_demod_name(self):
        return self.demod_combo.currentData() if self.demod_combo.count() else None

    def _on_demod_selected(self, *_):
        d = self.result_dir
        name = self._selected_demod_name()
        self.demod = read_json(d / name) if (d is not None and name) else None
        js = self.demod
        if js:
            A, hm = float(js.get("A_nm", 180)), float(js.get("h_min_nm", 10))
            k = int(js.get("deriv", 2))
            b = js.get("band_thz", [0.3, 2.0])
            der = "d²S/dt²" if k == 2 else "dS/dt" if k == 1 else f"d^{k}S/dt^{k}"
            hs = self.meta.get("heights_nm") or []
            rng = f"敲击范围 {hm:g}–{hm + 2 * A:g} nm"
            if hs and hm + 2 * A > max(hs) + 1e-9:
                t = ThemeManager.instance().tokens
                rng = f"<span style='color:{t.warning}'>{rng}（超过最大高度 {max(hs):g} nm，已截断）</span>"
            self.demod_desc.setText(f"f_cut = {float(js.get('f_cut_thz', 2.2)):g} THz · 检测 ∝ {der} · 频带 {b[0]:g}–{b[1]:g} THz · {rng}")
            png = d / f"tip_positions_A{A:g}.png"
        else:
            self.demod_desc.setText("没有解调结果" if d is not None else "")
            pngs = sorted(d.glob("tip_positions_A*.png")) if d is not None else []
            png = pngs[0] if pngs else None
        if png is not None and png.exists():
            self.png_view.load(png)
        else:
            self.png_view.clear("此结果没有脚本生成的 tip_positions_A*.png" if d is not None else "暂无图像")
        self._redraw_amp()
        self._fill_table()
        self._start_wave_load()

    def _redraw_amp(self, *_):
        js = self.demod
        if not js or not js.get("X"):
            self.amp_empty.setText("此文件夹没有解调结果 (demod_A*_h*.json)，没有可画的幅度 vs X。"
                                   if self.result_dir is not None else "尚未载入结果")
            self.amp_stack.setCurrentIndex(0)
            self.series_box.setVisible(False)
            return
        self.amp_stack.setCurrentIndex(1)
        self.series_box.setVisible(True)
        sel = [k for k, c in self.series.items() if c.isChecked()]
        absolute = self.abs_box.isChecked()
        meta = self.meta or js
        narrow = self._narrow["amp"] = self.amp_plot.width() < NARROW_PX

        def draw(fig):
            t = ThemeManager.instance().tokens
            ax = fig.add_subplot(111)
            X = np.asarray(js["X"], float)
            o = np.argsort(X)
            src = js.get("rms" if absolute else "rms_norm", {})
            cyc = [c["color"] for c in __import__("matplotlib").rcParams["axes.prop_cycle"]]
            for key in sel:
                if key in src:
                    y = np.asarray(src[key], float)[o]
                    if key == "far":
                        ax.plot(X[o], y, "x:", color=t.muted, lw=1.4, ms=7, label="far field dJ/dt")
                        continue
                    n = int(key[-1] if key[0] == "S" else key[3])
                    col = cyc[(n - 1) % len(cyc)]
                    if key.startswith("S"):
                        ax.plot(X[o], y, "s:", color=col, alpha=0.7, lw=1.2, ms=5, label=f"peak |S{n}|")
                    elif key.endswith("_d1"):
                        ax.plot(X[o], y, "^--", color=col, alpha=0.8, lw=1.2, ms=5, label=f"n = {n}, dS/dt")
                    else:
                        ax.plot(X[o], y, "o-", color=col, lw=1.8, ms=5.5, label=f"n = {n}")
            if not absolute:
                m = measured()
                for b, key, mk, col in ((25, "meas+", "o", t.text), (-25, "meas-", "s", t.muted)):
                    if key in sel and b in m:
                        xm, r, e = m[b]
                        ax.errorbar(-xm if b < 0 else xm, r, yerr=e, fmt=mk, color=col, mfc="none" if b < 0 else col,
                                    capsize=3, ms=6, lw=1.0, label=f"measured {b:+d} V" + (" (X→−X)" if b < 0 else ""))
            ax.axvline(0, color=t.border, lw=0.8, zorder=0)
            ax.set_xlabel("X (µm)" if narrow else "X (µm), beam position relative to the tip")
            b = js.get("band_thz", [0.3, 2.0])
            if absolute:
                ax.set_yscale("log")
                ax.set_ylabel("band amplitude (arb. u.)" if narrow else f"amplitude in {b[0]:g}–{b[1]:g} THz (arb. u.)")
            else:
                ax.set_ylabel("band amplitude (norm.)" if narrow else f"amplitude in {b[0]:g}–{b[1]:g} THz, normalised")
                ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.35)
            pump = "FDTD tip map" if meta.get("fdtd") not in (None, "none") else "symmetric pump"
            fig.suptitle(f"{meta.get('P_avg_mW', 0):g} mW, V_bias = {meta.get('V_bias', 0):g} V, "
                         f"V_tip = {meta.get('V_tip', 0):g} V, {pump}" + ("\n" if narrow else ";  ")
                         + f"A = {js.get('A_nm', 0):g} nm, h_min = {js.get('h_min_nm', 0):g} nm", fontsize=8.5 if narrow else 10)
            h, l = ax.get_legend_handles_labels()
            if h and narrow:                 # a legend at the side would leave no room for the axes
                fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=7)
            elif h:
                fig.legend(h, l, loc="outside right center", fontsize=8)
        self.amp_plot.plot(draw)

    def _fill_table(self):
        js = self.demod
        self.table.clear()
        if not js or not js.get("X"):
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        src = js.get("rms_norm", {})
        keys = [k for k in [*(f"det{n}" for n in HARMONICS), *(f"det{n}_d1" for n in HARMONICS),
                            *(f"S{n}" for n in HARMONICS), "far"] if k in src]
        raw = js.get("rms", {})
        X = js["X"]
        cols = ["X (µm)"] + keys + [f"{k} (rms)" for k in keys if k in raw]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(X))
        for i, x in enumerate(X):
            vals = [fx(x)] + [f"{src[k][i]:.3f}" for k in keys] + [f"{raw[k][i]:.4g}" for k in keys if k in raw]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, c, it)
        self.table.resizeColumnsToContents()
        self.table_note.setText(f"{self.demod_combo.currentData()} · det/S/far 列按各自最大值归一化 (rms_norm)；"
                                "标 (rms) 的列为原始值")

    def _copy_table(self):
        t = self.table
        rows = ["\t".join(t.horizontalHeaderItem(c).text() for c in range(t.columnCount()))]
        for r in range(t.rowCount()):
            rows.append("\t".join(t.item(r, c).text() if t.item(r, c) else "" for c in range(t.columnCount())))
        QGuiApplication.clipboard().setText("\n".join(rows))

    # -- waveforms (only when the npz are present)
    def _wave_params(self):
        js = self.demod or {}
        return dict(A_nm=float(js.get("A_nm", 180.0)), h_min_nm=float(js.get("h_min_nm", 10.0)),
                    f_cut_thz=float(js.get("f_cut_thz", 2.2)), deriv=int(js.get("deriv", 2)))

    def _start_wave_load(self):
        d = self.result_dir
        self._wave_gen += 1
        if d is None or not self.n_present:
            self.wave_empty.setText("此结果文件夹没有原始 X*_h*.npz（仓库中被 git 忽略），无法显示 Q_tip(t) 波形。\n"
                                    "重新运行「模拟 + 解调」后即可在这里查看每个高度的 Q_tip(t) 和各 X 的检测波形。"
                                    if d is not None else "尚未载入结果")
            self.wave_stack.setCurrentIndex(0)
            self.wave_status.setText("")
            self.wave_x.clear()
            return
        xs, hs = self.meta.get("X") or [], self.meta.get("heights_nm") or []
        dem = self._wave_params()
        key = (str(d), self.n_present)
        cached = self._wave[1] if (self._wave and self._wave[0] == key) else None
        gen = self._wave_gen
        self.wave_status.setText("正在读取 npz 并解调…")
        if cached is None:                 # another folder: do not leave its plot (or an old message) on display
            self.wave_empty.setText("正在读取 npz 并解调…")
            self.wave_stack.setCurrentIndex(0)
        run_async(_wave_work, str(d), xs, hs, dem, cached,
                  on_done=lambda r, g=gen, k=key: self._wave_done(g, k, r),
                  on_error=lambda e, g=gen: g == self._wave_gen and self.wave_status.setText("读取失败：" + e.splitlines()[0]))

    def _wave_done(self, gen, key, res):
        if gen != self._wave_gen:
            return
        data, det = res
        self._wave = (key, data, det)
        cur = self.wave_x.currentData()
        xs = [xkey(X) for X in (self.meta.get("X") or []) if xkey(X) in data["Q"]]
        self.wave_x.blockSignals(True)
        self.wave_x.clear()
        for k in xs:
            self.wave_x.addItem(f"{fx(k)} µm", k)
        i = self.wave_x.findData(cur if cur in xs else ("+5.0" if "+5.0" in xs else (xs[0] if xs else None)))
        self.wave_x.setCurrentIndex(max(0, i))
        self.wave_x.blockSignals(False)
        nx = len(self.meta.get("X") or [])
        self.wave_status.setText(f"{self.n_present} 个 npz · {len(det)}/{nx} 个 X 高度齐全")
        self.wave_stack.setCurrentIndex(1)
        self._redraw_wave()

    def _redraw_wave(self):
        if not self._wave or self.wave_stack.currentIndex() != 1:
            return
        _, data, det = self._wave
        xk = self.wave_x.currentData()
        n = int(self.wave_n.currentText())
        dem = self._wave_params()
        narrow = self._narrow["wave"] = self.wave_plot.width() < NARROW_PX

        def draw(fig):
            import matplotlib
            t = ThemeManager.instance().tokens
            ax1, ax2 = fig.subplots(1, 2)
            fs = 8.5 if narrow else 9.5
            row = data["Q"].get(xk, {})
            tt = data["t"].get(xk)
            hs = sorted(row)
            cmap = matplotlib.colormaps["viridis"]
            c0, c1 = (0.3, 1.0) if t.dark else (0.0, 0.85)      # keep both ends of viridis readable on the background
            for k, h in enumerate(hs):
                ax1.plot(tt * 1e12, row[h] * 1e12, color=cmap(c0 + (c1 - c0) * k / max(1, len(hs) - 1)), lw=1.3,
                         label=f"h = {h} nm")
            ax1.set_xlabel("t (ps)")
            ax1.set_ylabel("Q_tip (pC/cm)" if narrow else "induced tip charge Q_tip (pC/cm)")
            ax1.set_title(f"(a) Q_tip(t) per tip height\nX = {fx(xk)} µm" if xk else "(a) Q_tip(t)", fontsize=fs)
            if hs:
                ax1.legend(fontsize=7, ncol=2)
            ax1.grid(True, alpha=0.3)
            cyc = [c["color"] for c in matplotlib.rcParams["axes.prop_cycle"]]
            keys = [k for k in sorted(det, key=float)]
            peak = max((np.abs(det[k][n]).max() for k in keys), default=1.0) or 1.0
            for i, k in enumerate(keys):
                ax2.plot(data["t"][k] * 1e12, det[k][n] / peak, color=cyc[i % len(cyc)], lw=2.2 if k == xk else 1.2,
                         label=f"X = {fx(k)}")
            if keys:
                ax2.legend(fontsize=8)
            else:
                ax2.text(0.5, 0.5, "no X with all heights yet", transform=ax2.transAxes, ha="center", color=t.muted)
            ax2.axhline(0, color=t.border, lw=0.8)
            der = "d²" if dem["deriv"] == 2 else "d" if dem["deriv"] == 1 else f"d^{dem['deriv']}"
            sq = '²' if dem['deriv'] == 2 else ''
            ax2.set_title(f"(b) {der}S{n}/dt{sq}, all X\nA = {dem['A_nm']:g}, h_min = {dem['h_min_nm']:g} nm" if narrow else
                          f"(b) detected ∝ {der}S{n}/dt{sq}, all X\n"
                          f"{dem['f_cut_thz']:g} THz LP, A = {dem['A_nm']:g} nm, h_min = {dem['h_min_nm']:g} nm", fontsize=fs)
            ax2.set_xlabel("t (ps)")
            ax2.set_ylabel("normalised" if narrow else "normalised (largest |trace| = 1)")
            ax2.grid(True, alpha=0.3)
            tmax = max((float(v[-1]) for v in data["t"].values()), default=3e-12) * 1e12
            ax1.set_xlim(0, min(3.0, tmax))
            ax2.set_xlim(0, min(2.5, tmax))
        self.wave_plot.plot(draw)

    def _can_redemod(self):
        d = self.result_dir
        return (d is not None and self.n_expected > 0 and self.n_present == self.n_expected and not self._busy(d))

    def _update_demod_btn(self):
        d = self.result_dir
        ok = self._can_redemod()
        if d is None:
            tip = "先在「历史结果」中载入一个含原始 X*_h*.npz 的结果文件夹"
        elif ok:
            tip = f"用左侧「解调」参数重新解调 {paths.rel(d)}（约 1–5 s），写入 demod_A<A>_h<h_min>.json 与 PNG"
        elif self._busy(d):
            tip = "此文件夹的模拟/解调任务仍在运行"
        elif self.n_present == 0:
            tip = (f"{paths.rel(d)} 没有原始 X*_h*.npz（仓库中被 git 忽略），无法重新解调。"
                   "请先重新模拟（载入参数到表单 → 模拟 + 解调）。")
        else:
            tip = f"{paths.rel(d)} 的原始数据不完整（{self.n_present}/{self.n_expected}），需先补算剩余的模拟"
        for b in (self.demod_btn, self.btn_redemod):
            b.setEnabled(ok)
            b.setToolTip(tip)

    def _update_note_colors(self):
        """Rich-text colours follow the theme (the plots re-render themselves)."""
        if self.result_dir is not None and self.demod:
            js, hs = self.demod, self.meta.get("heights_nm") or []
            if hs and float(js.get("h_min_nm", 10)) + 2 * float(js.get("A_nm", 180)) > max(hs) + 1e-9:
                self._on_demod_selected()

    def _load_result_params(self):
        m = self.meta
        if not m:
            return
        vals = {k: m[k] for k in ("P_avg_mW", "V_bias", "V_tip", "T", "degenerate", "t_end_ps") if k in m}
        if "heights_nm" in m:
            vals["heights"] = tuple(m["heights_nm"])
        if "X" in m:
            vals["x_list"] = tuple(float(x) for x in m["X"])
        js = self.demod
        if js:
            vals.update({k: js[k] for k in ("A_nm", "h_min_nm", "f_cut_thz") if k in js})
            if "deriv" in js:
                vals["deriv"] = str(int(js["deriv"]))
            if "band_thz" in js:
                vals["band"] = tuple(js["band_thz"])
        self.form.set_values(vals, reset_others=True)
        self._home_tuples()
        self._set_fdtd(m.get("fdtd") or "none")
        self._update_plan()

    def _resume(self):
        if self.result_dir is None:
            return
        self._load_result_params()
        self.out_edit.setText(paths.rel(self.result_dir))
        self.tabs.setCurrentIndex(0)

    # ------------------------------------------------------------------ history
    def refresh_history(self):
        self.hist.clear()
        dirs = sorted(p for p in COMMITTED.iterdir() if p.is_dir()) if COMMITTED.is_dir() else []
        dirs += sorted((p for p in paths.GUI_RUNS.glob("tip_*") if p.is_dir()), reverse=True)
        for d in dirs:
            demods = demod_files(d)
            meta = read_json(d / "meta.json") or (read_json(demods[0]) if demods else None) or {}
            xs, hs = meta.get("X") or [], meta.get("heights_nm") or []
            n_exp = len(xs) * len(hs)
            n = npz_count(d, xs, hs) if n_exp else 0
            bits = []
            if meta:
                bits.append(f"{meta.get('P_avg_mW', 0):g} mW")
                bits.append(f"V_tip {meta.get('V_tip', 0):g} V")
                bits.append("对称泵浦" if meta.get("fdtd") in (None, "none") else "FDTD 图")
                bits.append(f"{len(xs)} X × {len(hs)} h")
            bits.append("解调: " + (", ".join(p.stem.replace("demod_", "") for p in demods) if demods else "无"))
            bits.append(f"原始 npz {n}/{n_exp}" if n_exp else "无 meta.json")
            if self._busy(d):
                bits.append("运行中")
            if not is_committed(d):         # checkout time of committed folders says nothing
                stamp = (d / "meta.json") if (d / "meta.json").exists() else d
                bits.append(time.strftime("%Y-%m-%d %H:%M", time.localtime(stamp.stat().st_mtime)))
            tag = "仓库" if is_committed(d) else "GUI"
            it = QListWidgetItem(f"{paths.rel(d)}   [{tag}]\n    " + " · ".join(bits))
            it.setData(Qt.ItemDataRole.UserRole, str(d))
            it.setToolTip("双击载入到「结果」" + ("" if n == n_exp and n_exp else "；原始 npz 不全，无法重新解调"))
            self.hist.addItem(it)
        if not dirs:
            self.hist.addItem("还没有针尖位置链结果")

    def _hist_open(self, it):
        p = it.data(Qt.ItemDataRole.UserRole)
        if p:
            self.show_result(p)

    def _open_other(self):
        d = QFileDialog.getExistingDirectory(self, "选择针尖位置链结果文件夹", str(COMMITTED))
        if d:
            self.show_result(d)

    # ------------------------------------------------------------------ page API
    def on_show(self):
        if not self.chain:
            self._adopt_active_chain()
        self._refresh_fdtd_choices()
        self.refresh_history()
        self._update_chain()

    def open(self, path=None, **_):
        if path:
            self.show_result(paths.absolute(path))
