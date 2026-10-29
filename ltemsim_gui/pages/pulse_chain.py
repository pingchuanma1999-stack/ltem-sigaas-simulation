"""Single-pulse emission chain page (scripts/run_pulse_chain.py, scripts/redemodulate.py).

Generation (Beer–Lambert Gaussian × optional FDTD tip map) -> hydrodynamic carriers + Poisson -> quasi-static
near field above the tip -> FDM tip scattering + tapping demodulation, scanned over the beam position X.
"""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

import matplotlib
import numpy as np
from PySide6.QtCore import QRect, QSize, Qt, QTimer
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLayout, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox, QPushButton, QScrollArea, QSplitter,
                               QTableWidget, QTableWidgetItem, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)

from .. import icons, nav, paths
from ..jobs import FINISHED, JobManager
from ..paramspec import FieldSpec
from ..theme import ThemeManager
from ..widgets.common import MetricTile, Page, button, card, error_box, label, reveal
from ..widgets.imageview import ImageView
from ..widgets.mpl import PlotWidget
from ..widgets.paramform import ParamForm
from ..widgets.runpanel import RunPanel
from ..workers import run_async

X_DEFAULT = (-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5)
REPO_RATIO = "results/fdtd_pump/ratio.npz"
SEC_PER_X = 30.0                 # typical wall time per X for Nx = 242 and t_end = 4 ps
EST_RANGE = (0.45, 1.8)          # this machine: ~13 s per X (1.7 ms/step); repo logs: 31–54 s per X
# lowpass() convolves with an 83-sample kernel (mode="same" returns max(len) samples); traces shorter than that
# (t_end < 0.415 ps at 5 fs sampling) make the script's overview figure fail with a shape mismatch
T_END_MIN = 0.5
DEFAULT_RUN = "P2mW_notipmap_degen"
CHAIN_SCRIPT = "scripts/run_pulse_chain.py"
X_RE = re.compile(r"^X([+-]?\d+(?:\.\d+)?)\.npz$")
# values the script used before they were written to summary.json (older runs)
LEGACY_DEFAULTS = {"w_um": 3.0, "V_bias": 25.0, "T": 300.0, "degenerate": False}


def chain_specs() -> list[FieldSpec]:
    S = FieldSpec
    return [
        S("P_avg_mW", 2.0, "float", "平均功率 P_avg", "mW",
          "80 MHz 下的平均泵浦功率。每脉冲载流子密度在 n_sat = 2×10¹⁹ cm⁻³ 处软饱和（吸收饱和）。",
          "激发与偏压", minimum=1e-6),
        S("V_bias", 25.0, "float", "偏压 V_bias", "V",
          "间隙偏压，阳极在 −x。0 V 给出与偏压无关的偶项；负值为反向极性。", "激发与偏压"),
        S("w_um", 3.0, "float", "光斑半径 w (1/e²)", "µm",
          "沿 x 的 1/e² 强度半径。实测形状要求有效半径 3–4 µm；w = 7 µm 时每个 X 约慢一倍。", "激发与偏压", minimum=0.2),
        S("T", 300.0, "float", "载流子温度 T", "K",
          "设定扩散 kT/q 与 N_c。物理上预期 T_e ≈ 600–800 K；1500 K 时已不简并。", "输运", minimum=1.0),
        S("degenerate", False, "bool", "简并扩散 (Fermi–Dirac)", "",
          "广义 Einstein 关系 D = μ(kT/q)·F₁/₂(η)/F₋₁/₂(η)，只在高密度核心起作用。脚本默认关闭（经典扩散）。", "输运"),
        S("x_list", X_DEFAULT, "tuple", "光束位置 X 列表", "µm",
          "光斑中心相对针尖的位置，逗号分隔，精度 0.1 µm（结果文件名为 X{X:+.1f}.npz）。"
          "每个 X 是一次完整的单脉冲模拟（t_end = 4 ps 时约 15–55 s，视机器而定）。只有 X = 0 那一次会输出时间进度。", "扫描"),
        S("t_end_ps", 4.0, "float", "模拟时长 t_end", "ps",
          f"显式时间步 0.5 fs；耗时与 t_end 成正比。至少 {T_END_MIN:g} ps：脚本对每条曲线做 2.2 THz 高斯低通"
          "（卷积核 83 个采样 ≈ 0.41 ps），更短的时长会在作总览图时出错。", "扫描", minimum=T_END_MIN),
        S("x_fine", "", "str", "细网格半宽 x_fine", "µm",
          "0.1 µm 细网格覆盖 |x| ≤ x_fine。留空 = 自动 max(8, 2w + max|X| − 4)，保证每个 X 的光斑都在细网格内。",
          "扫描", advanced=True),
        S("A_nm", 180.0, "float", "轻敲振幅 A", "nm",
          "h(θ) = h_min + A(1 + cos θ)，实验约 180 nm。只影响解调，可事后用「重新解调」改。", "针尖解调", minimum=1.0),
        S("h_min_nm", 0.0, "float", "最小高度 h_min", "nm", "轻敲周期中的最小针尖高度。", "针尖解调",
          advanced=True, minimum=0.0),
        S("harmonics", (1, 2, 3, 4), "tuple", "解调谐波 n", "",
          "Pizzuto FDM 约定下的锁相谐波。必须包含 2 和 3（脚本的进度输出和总览图要用）。", "针尖解调", advanced=True),
    ]


# ----------------------------------------------------------------------------- data helpers
def read_json(p: Path) -> dict:
    try:
        d = json.loads(Path(p).read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def run_files(d) -> dict[float, Path]:
    """{X: path} of the per-position traces X{X:+.1f}.npz in a chain run."""
    out = {}
    d = Path(d)
    if not d.is_dir():
        return out
    for p in d.glob("X*.npz"):
        m = X_RE.match(p.name)
        if m:
            out[float(m.group(1)) + 0.0] = p
    return dict(sorted(out.items()))


def chain_spec(d: Path) -> dict:
    """The GUI spec of the chain job that wrote into `d`: gui_run.json, or gui_run_main.json when another job
    owned the folder first. Specs of other scripts are ignored."""
    for name in ("gui_run.json", "gui_run_main.json"):
        spec = read_json(Path(d) / name)
        if spec.get("script") == CHAIN_SCRIPT:
            return spec
    return {}


def run_settings(d: Path) -> dict:
    """Run settings: summary.json, completed by the GUI spec (gui_run.json) when present.
    Keys taken from LEGACY_DEFAULTS are listed in "_inferred"."""
    s = read_json(d / "summary.json")
    kw = chain_spec(d).get("kwargs") or {}
    out = {k: s[k] for k in ("P_avg_mW", "A_nm", "h_min_nm", "fdtd_ratio", "degenerate", "T", "V_bias", "w_um")
           if k in s}
    if s.get("X"):
        out["x_list"] = tuple(s["X"])
    for k in ("P_avg_mW", "A_nm", "h_min_nm", "T", "V_bias", "w_um", "t_end_ps", "x_fine", "harmonics", "fdtd"):
        if k in kw and kw[k] is not None:
            out.setdefault(k, kw[k])
    if "x_list" in kw and kw["x_list"]:
        out.setdefault("x_list", tuple(kw["x_list"]))
    if "degenerate" in kw:
        out.setdefault("degenerate", bool(kw["degenerate"]))
    if "fdtd" in kw:
        out.setdefault("fdtd_ratio", kw["fdtd"] not in (None, "", "none"))
    if "harmonics" not in out and s.get("rms"):
        hs = sorted(int(m.group(1)) for k in s["rms"] for m in [re.match(r"S(\d)$", k)] if m)
        if hs:
            out["harmonics"] = tuple(hs)
    inferred = []
    if s or kw:
        for k, v in LEGACY_DEFAULTS.items():
            if k not in out:
                out[k] = v
                inferred.append(k)
    out["_inferred"] = inferred
    return out


def describe_run(st: dict) -> str:
    if not st or set(st) <= {"_inferred"}:
        return ""
    inf = set(st.get("_inferred", ()))
    star = lambda k: "*" if k in inf else ""
    parts = []
    if "P_avg_mW" in st:
        parts.append(f"{st['P_avg_mW']:g} mW")
    if "w_um" in st:
        parts.append(f"w = {st['w_um']:g} µm{star('w_um')}")
    if "V_bias" in st:
        parts.append(f"V_bias = {st['V_bias']:+g} V{star('V_bias')}".replace("+0 V", "0 V"))
    if "T" in st:
        parts.append(f"T = {st['T']:g} K{star('T')}")
    if "degenerate" in st:
        parts.append(("简并扩散" if st["degenerate"] else "经典扩散") + star("degenerate"))
    if "fdtd_ratio" in st:
        parts.append("FDTD 针尖图" if st["fdtd_ratio"] else "无针尖图")
    if "A_nm" in st:
        parts.append(f"A = {st['A_nm']:g} nm" + (f", h_min = {st['h_min_nm']:g} nm" if st.get("h_min_nm") else ""))
    if "t_end_ps" in st:
        parts.append(f"t_end = {st['t_end_ps']:g} ps")
    if st.get("x_list"):
        xs = st["x_list"]
        parts.append(f"X = {xs[0]:+g} µm" if len(xs) == 1 else f"{len(xs)} 个 X ({min(xs):+g} … {max(xs):+g} µm)")
    return " · ".join(parts)


def load_measured() -> list[dict]:
    rows = []
    try:
        with open(paths.MEASURED_CSV, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    rows.append(dict(bias=float(r["bias_V"]), X=float(r["X_um"]), rms=float(r["rms"]),
                                     err=float(r.get("rms_err") or 0.0)))
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        pass
    return rows


def xscan_metrics(X, y) -> dict:
    """Peak position, (+5)/(−5) ratio, normalised value at X = 0 and RMS deviation from the measured +25 V scan."""
    X = np.asarray(X, float)
    y = np.abs(np.asarray(y, float))
    ok = np.isfinite(X) & np.isfinite(y)
    X, y = X[ok], y[ok]
    out = {}
    if X.size == 0 or not np.nanmax(y) > 0:
        return out
    o = np.argsort(X)
    X, y = X[o], y[o]
    yn = y / y.max()
    out["peak_X"] = float(X[np.argmax(y)])
    lo, hi = X.min(), X.max()
    if X.size > 1 and lo <= -5 and hi >= 5:
        am5 = np.interp(-5.0, X, y)
        out["ratio5"] = float(np.interp(5.0, X, y) / am5) if am5 > 0 else float("inf")
    if X.size > 1 and lo <= 0 <= hi:
        out["zero"] = float(np.interp(0.0, X, yn))
    meas = [r for r in load_measured() if r["bias"] > 0 and lo <= r["X"] <= hi]
    if X.size > 1 and meas:
        d = [np.interp(r["X"], X, yn) - r["rms"] for r in meas]
        out["dev"] = float(np.sqrt(np.mean(np.square(d))))
        out["n_meas"] = len(meas)
    return out


def load_traces(files: dict) -> dict:
    """{X: {key: 1-D array}} with every time trace of the run (worker thread)."""
    out = {}
    for X, p in files.items():
        with np.load(p) as d:
            if "t" not in d.files:
                continue
            nt = d["t"].shape[0]
            tr = {}
            for k in d.files:
                if k in ("xc", "zc", "heights_nm", "N", "Ez_tip"):
                    continue
                a = d[k]
                if a.ndim == 1 and a.shape[0] == nt:
                    tr[k] = np.asarray(a, float)
            out[X] = tr
    return out


def band_table(files: dict, A_nm=180.0, h_min_nm=0.0, harmonics=(1, 2, 3, 4), deriv=2, band=(0.3, 2.0)) -> dict:
    """Band-limited spectral amplitude of d^deriv S_n/dt^deriv for every X (docs/pulse_chain.md: preferred metric)."""
    from ltemsim.tip_demod import band_metric, demodulate_pizzuto
    Xs, tab, far = [], {n: [] for n in harmonics}, []
    for X, p in sorted(files.items()):
        with np.load(p) as d:
            t = d["t"]
            dem = demodulate_pizzuto(t, d["heights_nm"], d["Ez_tip"], d["n_center"], A_nm=A_nm, h_min_nm=h_min_nm,
                                     harmonics=tuple(harmonics))
            for n in harmonics:
                tab[n].append(band_metric(t, dem[n], deriv=deriv, band_thz=band))
            # far-field reference dJx/dt = d²P/dt² with P = ∫J dt
            far.append(band_metric(t, np.cumsum(d["Jx"]) * (t[1] - t[0]), deriv=2, band_thz=band)
                       if "Jx" in d.files else np.nan)
        Xs.append(X)
    return dict(X=np.array(Xs), n={n: np.array(v) for n, v in tab.items()}, far=np.array(far))


def redemod_name(A_nm, h_min_nm, ltem) -> str:
    return f"redemod_A{A_nm:g}_h{h_min_nm:g}{'_ltemfactor' if ltem else ''}.json"


def redemod_label(p: Path) -> str:
    m = re.match(r"redemod_A([\d.]+)_h([\d.]+)(_ltemfactor)?\.json$", p.name)
    if not m:
        return p.name
    return f"A = {m.group(1)} nm, h_min = {m.group(2)} nm" + ("，LTEM (1−β) 因子" if m.group(3) else "")


def load_redemod(p: Path):
    """(X array, {metric: raw array}) from a redemodulate.py JSON."""
    d = read_json(p)
    Xs = sorted(d, key=float)
    keys = []
    for X in Xs:
        for k in d[X]:
            if k not in keys:
                keys.append(k)
    keys.sort(key=_redemod_order)
    tab = {k: np.array([d[X].get(k, np.nan) for X in Xs], float) for k in keys}
    return np.array([float(x) for x in Xs]), tab


def _redemod_order(k):
    m = re.match(r"(d?)S(\d)(?:dt)?_(rms|peak)$", k)
    if not m:
        return (9, 0, k)
    fam = {("d", "rms"): 0, ("", "rms"): 1, ("d", "peak"): 2}.get((m.group(1), m.group(3)), 3)
    return (fam, int(m.group(2)), k)


def metric_order(k):
    m = re.match(r"(d?)S(\d)(_dt)?$", k)
    if m:
        return (0 if m.group(1) else 1, int(m.group(2)), k)
    rest = ["dJx_dt", "dJz_dt", "E_loc_mid", "Jx", "Jz", "Ex_center", "n_center"]
    return (2, rest.index(k) if k in rest else 99, k)


def metric_desc(k) -> str:
    m = re.match(r"(d?)S(\d)(_dt)?$", k)
    if m:
        return f"dS{m.group(2)}/dt，第 {m.group(2)} 次谐波（探测波形）" if m.group(1) else f"S{m.group(2)}，第 {m.group(2)} 次谐波解调近场"
    return {"dJx_dt": "远场参考 dJx/dt", "dJz_dt": "dJz/dt", "E_loc_mid": "裸近场 E_loc（h_min + A 处）",
            "Jx": "横向电流 Jx", "Jz": "纵向电流 Jz", "Ex_center": "光斑中心表面 E_x", "n_center": "针尖下方表面密度"}.get(k, k)


def trace_unit(key: str):
    """(base unit, intrinsic time-derivative order) of an npz trace."""
    if re.match(r"S\d$", key) or key in ("E_loc_mid", "Ex_center"):
        return "V/cm", 0
    if re.match(r"dS\d_dt$", key):
        return "V/cm", 1
    if key in ("Jx", "Jz"):
        return "A/cm", 0
    if key in ("dJx_dt", "dJz_dt"):
        return "A/cm", 1
    if key == "n_center":
        return "cm⁻³", 0
    return "", 0


def estimate(n_x: int, w_um: float, max_abs_x: float, x_fine, t_end_ps: float):
    """(typical, low, high, typical per X) seconds for a chain run; wall time ∝ Nx · t_end. Nx ≈ 2·x_fine/0.1 + 82
    (242 for the default x_fine = 8 µm, 470 for w = 7 µm); +2 s per X for the PulseSim setup."""
    xf = x_fine if x_fine else max(8.0, 2.0 * w_um + max_abs_x - 4.0)
    nx = 2.0 * xf / 0.1 + 82.0
    per = SEC_PER_X * (nx / 242.0) * (t_end_ps / 4.0) + 2.0
    return n_x * per, n_x * per * EST_RANGE[0], n_x * per * EST_RANGE[1], per


def fmt_minutes(s: float) -> str:
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.1f} min"


def is_committed(d: Path) -> bool:
    """Inside results/ but not one of the GUI's own run folders."""
    try:
        p = Path(d).resolve()
        p.relative_to(paths.RESULTS.resolve())
    except ValueError:
        return False
    try:
        p.relative_to(paths.GUI_RUNS.resolve())
        return False
    except ValueError:
        return True


class _SortItem(QTreeWidgetItem):
    """Tree row sorted numerically where a sort key is stored (UserRole + 1)."""

    def __lt__(self, other):
        tw = self.treeWidget()
        c = tw.sortColumn() if tw else 0
        a, b = self.data(c, Qt.ItemDataRole.UserRole + 1), other.data(c, Qt.ItemDataRole.UserRole + 1)
        if a is not None and b is not None:
            try:
                return a < b
            except TypeError:
                pass
        return self.text(c) < other.text(c)


def _cycle() -> list:
    """Colour cycle of the current (themed) rc context."""
    return list(matplotlib.rcParams["axes.prop_cycle"].by_key()["color"])


def _legend_right(ax):
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, frameon=False)


def _fit_table(tw: QTableWidget, rows_visible: int):
    """Stretch the X columns, size the last (raw max) column to its contents; keep at least `rows_visible`
    (at most 2) rows visible so the plot above it in the splitter is not squeezed on small windows."""
    h = tw.horizontalHeader()
    h.setMinimumSectionSize(46)
    h.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    tw.setToolTip("列：光束位置 X (µm)，每行按自身最大值归一化；悬停单元格看原始值，最后一列为原始最大值")
    if tw.columnCount():
        h.setSectionResizeMode(tw.columnCount() - 1, QHeaderView.ResizeMode.ResizeToContents)
    tw.setMinimumHeight(h.sizeHint().height() + min(rows_visible, 2) * tw.verticalHeader().defaultSectionSize() + 6)


def _spin(value, lo, hi, step, decimals=1, suffix="", width=92):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    if suffix:
        s.setSuffix(" " + suffix)
    s.setMinimumWidth(width)
    return s


class _Flow(QLayout):
    """Left-to-right row of controls that wraps onto further lines when the width runs out
    (Qt's flow-layout example; items are vertically centred within each line)."""

    def __init__(self, parent=None, hspacing=10, vspacing=6):
        super().__init__(parent)
        self._items = []
        self._h, self._v = hspacing, vspacing
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
        return self._place(QRect(0, 0, w, 0), False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._place(rect, True)

    def _shown(self):
        return [it for it in self._items if not it.isEmpty()]

    def sizeHint(self):
        items = self._shown()
        if not items:
            return QSize(0, 0)
        return QSize(sum(it.sizeHint().width() for it in items) + self._h * (len(items) - 1),
                     max(it.sizeHint().height() for it in items))

    def minimumSize(self):
        s = QSize(0, 0)
        for it in self._shown():
            s = s.expandedTo(it.minimumSize())
        return s

    def _place(self, rect, apply):
        lines, line, x, W = [], [], 0, max(1, rect.width())
        for it in self._shown():
            s = it.sizeHint()
            w = min(s.width(), W)
            if line and x + w > W:
                lines.append(line)
                line, x = [], 0
            line.append((it, w, s.height()))
            x += w + self._h
        if line:
            lines.append(line)
        y = rect.y()
        for ln in lines:
            lh = max(h for _, _, h in ln)
            if apply:
                x = rect.x()
                for it, w, h in ln:
                    it.setGeometry(QRect(x, y + (lh - h) // 2, w, h))
                    x += w + self._h
            y += lh + self._v
        return max(0, y - self._v - rect.y())


def _group(*widgets, spacing=4) -> QWidget:
    """Keep a caption and its control(s) together inside a _Flow."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for x in widgets:
        lay.addWidget(QLabel(x) if isinstance(x, str) else x)
    return w


class _Split(QSplitter):
    """Vertical plot/table splitter whose preferred height is its minimum, so a _scrollable tab only scrolls when
    the window is really too low (QScrollArea sizes resizable content by its preferred height for the width)."""

    def __init__(self):
        super().__init__(Qt.Orientation.Vertical)
        self.setChildrenCollapsible(False)

    def sizeHint(self):
        return self.minimumSizeHint()


def _scrollable(w: QWidget) -> QScrollArea:
    """Tab content that scrolls vertically when the window is too low for its controls plus a usable plot."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(w)
    return area


def _flow(*items, hspacing=10) -> _Flow:
    f = _Flow(hspacing=hspacing)
    for x in items:
        f.addWidget(x)
    return f


class _TileRow(QWidget):
    """Metric tiles side by side; when the row is too narrow for all of them, one wrapped line of text with the
    same numbers instead (the plot below needs the height on small windows). Call sync() after updating tiles."""

    def __init__(self, tiles, min_tile=140, spacing=8):
        super().__init__()
        self._tiles = list(tiles)
        self._need = len(self._tiles) * min_tile + (len(self._tiles) - 1) * spacing
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._row = QWidget()
        rl = QHBoxLayout(self._row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(spacing)
        for t in self._tiles:
            rl.addWidget(t, 1)
        self._text = label("", wrap=True)
        self._text.setTextFormat(Qt.TextFormat.RichText)
        self._text.hide()
        lay.addWidget(self._row)
        lay.addWidget(self._text)
        self.setMinimumWidth(200)
        self.sync()

    def sync(self):
        self._text.setText("　·　".join(f"{t.lab.text()} <b>{t.val.text()}</b>" for t in self._tiles))
        self._text.setToolTip("\n".join(f"{t.lab.text()}：{t.val.text()}（{t.hint.text()}）" for t in self._tiles))

    def compact(self) -> bool:
        return self._text.isVisibleTo(self)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        narrow = e.size().width() < self._need
        if narrow != self.compact():
            self._row.setVisible(not narrow)
            self._text.setVisible(narrow)


# ----------------------------------------------------------------------------- page
class PulseChainPage(Page):
    title = "单脉冲发射链"
    subtitle = ("单个泵浦脉冲：高斯光斑（可乘 FDTD 针尖图）→ 电子/空穴流体动力学 + Poisson → 针尖上方准静态近场 → "
                "有限偶极子散射与轻敲解调（n = 1–4），扫描光斑-针尖位置 X 并与实测对比。")

    def __init__(self):
        super().__init__()
        self.job = None
        self.redemod_job = None
        self._redemod_target = None
        self.run_dir: Path | None = None
        self.summary: dict = {}
        self.settings: dict = {}
        self.files: dict = {}
        self.traces: dict = {}
        self.band: dict | None = None
        self._band_key = None
        self._band_want = None
        self._gen = 0
        self._shown_once = False
        self._trace_pref = "dS3_dt"

        self.run_btn = button("  运行发射链", self.run, "primary")
        self.header.add_action(self.run_btn)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._build_left())
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)
        rl.addWidget(self._build_run_bar())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_run_tab(), "运行")
        self.res_tab = self._build_results_tab()
        self.tabs.addTab(self.res_tab, "结果")
        self.wave_tab = self._build_wave_tab()
        self.tabs.addTab(self.wave_tab, "波形")
        self.band_tab = self._build_band_tab()
        self.tabs.addTab(self.band_tab, "频带指标")
        self.redemod_tab = self._build_redemod_tab()
        self.tabs.addTab(self.redemod_tab, "重新解调")
        self.hist_tab = self._build_history_tab()
        self.tabs.addTab(self.hist_tab, "历史结果")
        self.tabs.currentChanged.connect(self._on_tab)
        rl.addWidget(self.tabs, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([400, 900])
        self.body.addWidget(split, 1)

        self._band_timer = QTimer(self)
        self._band_timer.setSingleShot(True)
        self._band_timer.setInterval(350)
        self._band_timer.timeout.connect(lambda: self.tabs.currentWidget() is self.band_tab and self.compute_band())

        self._update_estimate()
        self._retheme()
        ThemeManager.instance().changed.connect(self._retheme)
        ThemeManager.instance().changed.connect(lambda *_: self.refresh_history())
        JobManager.instance().job_finished.connect(self._on_any_job_finished)
        self._clear_result()

    # ================================================================== layout
    def _build_left(self):
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)
        # the form, the tip-map choice and the estimate scroll together; the output row stays visible
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        inner.setObjectName("page")
        il = QVBoxLayout(inner)
        il.setContentsMargins(0, 0, 6, 0)
        il.setSpacing(6)
        self.form = ParamForm(chain_specs(), scroll=False)
        self.form.rows["x_fine"].editor.setPlaceholderText("自动")
        self.form.rows["x_list"].editor.setCursorPosition(0)
        self.form.changed.connect(self._update_estimate)
        il.addWidget(self.form)

        g = QGroupBox("泵浦针尖图 (FDTD)")
        gl = QVBoxLayout(g)
        gl.setContentsMargins(8, 10, 8, 6)
        gl.setSpacing(5)
        row = QHBoxLayout()
        self.fdtd_combo = QComboBox()
        self.fdtd_combo.setMinimumWidth(160)
        self.fdtd_combo.setToolTip("生成率乘以 Q_tip/Q_notip（裁剪到 0–50，FDTD 区域外为 1）。始终显式传给脚本。")
        row.addWidget(self.fdtd_combo, 1)
        b = button("浏览…", self._browse_fdtd)
        row.addWidget(b)
        gl.addLayout(row)
        hint = label("2-D 针尖是无限长的楔形，会挡住 +x 一侧的整束光，因此针尖图是遮挡效应的上限。"
                     "文档中的主要结果（路线 1）不使用针尖图。", "muted", wrap=True)
        gl.addWidget(hint)
        gen = button("去「泵浦 FDTD」生成新的针尖图 →", lambda: nav.navigate("fdtd"), "flat")
        gen.setStyleSheet("text-align: left; padding-left: 0;")
        gl.addWidget(gen)
        il.addWidget(g)
        il.addStretch(1)
        area.setWidget(inner)
        ll.addWidget(area, 1)

        est = card()
        el = QVBoxLayout(est)
        el.setContentsMargins(12, 8, 12, 8)
        el.setSpacing(2)
        self.est_title = label("预计耗时 —", "cardTitle")
        self.est_text = label("", "muted", wrap=True)
        el.addWidget(self.est_title)
        el.addWidget(self.est_text)
        ll.addWidget(est)

        orow = QHBoxLayout()
        orow.addWidget(QLabel("输出文件夹"))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("自动：results/gui_runs/pulse_<时间>")
        orow.addWidget(self.out_edit, 1)
        b = QPushButton("选择…")
        b.clicked.connect(self._pick_out)
        orow.addWidget(b)
        ll.addLayout(orow)
        left.setMinimumWidth(380)          # ParamForm needs ~362 px + the scroll bar
        self.refresh_fdtd_choices()
        return left

    def _build_run_bar(self):
        bar = card()
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 8, 8, 8)
        bl.setSpacing(6)
        col = QVBoxLayout()
        col.setSpacing(1)
        self.res_title = label("尚未载入结果", "cardTitle", selectable=True)
        self.res_info = label("", "muted", wrap=True, selectable=True)
        col.addWidget(self.res_title)
        col.addWidget(self.res_info)
        bl.addLayout(col, 1)
        self.btn_params = button("载入此结果的参数", self._load_result_params, "flat",
                                 "把该结果的设置填入左侧参数表（GUI 运行可完整恢复，仓库结果按 summary.json）")
        self.btn_folder = button("打开文件夹", lambda: self.run_dir and reveal(self.run_dir), "flat")
        bcol = QVBoxLayout()
        bcol.setSpacing(0)
        for b in (self.btn_params, self.btn_folder):
            bcol.addWidget(b, 0, Qt.AlignmentFlag.AlignRight)
        bcol.addStretch(1)
        bl.addLayout(bcol)
        return bar

    def _build_run_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12)
        self.panel = RunPanel()
        self.panel.job_finished.connect(self._on_finished)
        lay.addWidget(self.panel)
        return w

    def _build_results_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)
        self.metric_combo = QComboBox()
        self.metric_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.metric_combo.setMinimumContentsLength(24)
        self.metric_combo.currentIndexChanged.connect(self._metric_changed)
        self.stat_combo = QComboBox()
        self.stat_combo.addItem("RMS", "rms")
        self.stat_combo.addItem("峰值", "peak")
        self.stat_combo.setToolTip("summary.json 中 2.2 THz 高斯低通后的 RMS 或峰值（旧指标；推荐对比见「频带指标」）")
        self.stat_combo.currentIndexChanged.connect(self._metric_changed)
        self.cb_family = QCheckBox("叠加全部谐波")
        self.cb_family.toggled.connect(lambda *_: self.xscan_plot.plot(self._draw_xscan))
        self.cb_neg = QCheckBox("−25 V 实测 (X → −X)")
        self.cb_neg.setChecked(True)
        self.cb_neg.toggled.connect(lambda *_: self.xscan_plot.plot(self._draw_xscan))
        lay.addLayout(_flow(_group("指标", self.metric_combo, self.stat_combo, spacing=8), self.cb_family, self.cb_neg))

        self.m_peak = MetricTile("峰值位置 X", "—", "实测：+5 µm")
        self.m_ratio = MetricTile("X = +5 / X = −5", "—", "实测：≫ 1")
        self.m_zero = MetricTile("X = 0（归一化）", "—", "实测：0.24")
        self.m_dev = MetricTile("与 +25 V 实测的偏差", "—", "实测点上的 RMS 差")
        for m in (self.m_peak, self.m_ratio, self.m_zero, self.m_dev):
            m.setToolTip("按左上「指标」所选的量计算；实测指 data/measured_tip_scan.csv 的 +25 V 扫描")
        self.m_ratio.setToolTip(self.m_ratio.toolTip() + "（+25 V 实测在 X = −5 µm 处 ≈ 0）")
        self.tiles = _TileRow((self.m_peak, self.m_ratio, self.m_zero, self.m_dev))
        lay.addWidget(self.tiles)

        self.res_sub = QTabWidget()
        self.xscan_plot = PlotWidget()
        self.res_sub.addTab(self.xscan_plot, "幅度 vs X")
        self.overview = ImageView()
        self.res_sub.addTab(self.overview, "总览图 pulse_chain.png")
        lay.addWidget(self.res_sub, 1)
        return w

    def _build_wave_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)
        self.trace_combo = QComboBox()
        self.trace_combo.setMinimumWidth(130)
        self.trace_combo.currentIndexChanged.connect(self._trace_changed)
        self.deriv_combo = QComboBox()
        for txt, k in (("无", 0), ("d/dt", 1), ("d²/dt²", 2)):
            self.deriv_combo.addItem(txt, k)
        self.deriv_combo.setToolTip("对所选曲线再求时间导数（np.gradient）。探测到的辐射场 ∝ d²S_n/dt²。")
        self.deriv_combo.currentIndexChanged.connect(self._redraw_waves)
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("高斯低通 2.2 THz", "gauss")
        self.filter_combo.addItem("Butterworth 8 阶 2.2 THz", "steep")
        self.filter_combo.addItem("不滤波", "none")
        self.filter_combo.setToolTip("高斯：ltemsim.tip_demod.lowpass（脚本作图所用）；Butterworth：lowpass_steep（探测器响应，"
                                     "适合二阶导数）")
        self.filter_combo.currentIndexChanged.connect(self._redraw_waves)
        self.cb_wnorm = QCheckBox("归一化")
        self.cb_wnorm.toggled.connect(self._redraw_waves)
        lay.addLayout(_flow(_group("曲线", self.trace_combo), _group("导数", self.deriv_combo),
                            _group("滤波", self.filter_combo), self.cb_wnorm))

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(4)
        sl.addWidget(label("光束位置 X", "cardTitle"))
        self.x_list = QListWidget()
        self.x_list.itemChanged.connect(self._redraw_waves)
        sl.addWidget(self.x_list, 1)
        brow = QHBoxLayout()
        for txt, on in (("全选", True), ("清空", False)):
            b = button(txt, lambda _=False, v=on: self._check_all_x(v), "flat")
            brow.addWidget(b)
        sl.addLayout(brow)
        side.setMinimumWidth(130)
        side.setMaximumWidth(200)
        split.addWidget(side)
        self.wave_plot = PlotWidget()
        split.addWidget(self.wave_plot)
        split.setSizes([140, 700])
        lay.addWidget(split, 1)
        return w

    def _build_band_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(6)
        intro = label("推荐的对比指标：重新解调后 d²S_n/dt² 在频带内的谱幅度（band_metric），参数改动后自动重算。",
                      "muted", wrap=True)
        intro.setToolTip("docs/pulse_chain.md：等价于探测器带限波形的 RMS，且与滤波器滚降无关（旧的 2.2 THz 低通 RMS "
                         "被 5–15 THz 的电流突变泄漏主导）。远场参考为 dJx/dt = d²P/dt²，P = ∫J dt。")
        lay.addWidget(intro)
        self.b_A = _spin(180.0, 1.0, 2000.0, 10.0, 1, "nm")
        self.b_h = _spin(0.0, 0.0, 1000.0, 5.0, 1, "nm")
        self.b_deriv = QComboBox()
        self.b_deriv.addItem("d²S/dt²", 2)
        self.b_deriv.addItem("dS/dt", 1)
        self.b_lo = _spin(0.3, 0.0, 50.0, 0.1, 2, "THz", 84)
        self.b_hi = _spin(2.0, 0.05, 100.0, 0.1, 2, "THz", 84)
        lay.addLayout(_flow(_group("A", self.b_A), _group("h_min", self.b_h), _group("波形", self.b_deriv),
                            _group("频带", self.b_lo, "–", self.b_hi)))
        self.b_harm = {}
        for n in (1, 2, 3, 4):
            cb = QCheckBox(f"n = {n}")
            cb.setChecked(True)
            cb.toggled.connect(self._band_params_changed)
            self.b_harm[n] = cb
        self.b_far = QCheckBox("远场参考 dJx/dt")
        self.b_far.setChecked(True)
        self.b_far.toggled.connect(lambda *_: self.band_plot.plot(self._draw_band))
        self.b_neg = QCheckBox("−25 V 实测 (X → −X)")
        self.b_neg.setChecked(True)
        self.b_neg.toggled.connect(lambda *_: self.band_plot.plot(self._draw_band))
        self.b_run = button("重新计算", self.compute_band)
        lay.addLayout(_flow(_group("谐波", *self.b_harm.values(), spacing=10), self.b_far, self.b_neg, self.b_run))
        for wdg in (self.b_A, self.b_h, self.b_lo, self.b_hi):
            wdg.valueChanged.connect(self._band_params_changed)
        self.b_deriv.currentIndexChanged.connect(self._band_params_changed)
        self.b_status = label("", "muted", wrap=True)
        lay.addWidget(self.b_status)

        split = _Split()
        self.band_plot = PlotWidget()
        self.band_plot.setMinimumHeight(240)
        split.addWidget(self.band_plot)
        self.band_table_w = QTableWidget()
        self.band_table_w.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.band_table_w.setAlternatingRowColors(True)
        self.band_table_w.verticalHeader().setDefaultSectionSize(24)
        split.addWidget(self.band_table_w)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([520, 160])
        lay.addWidget(split, 1)
        return _scrollable(w)

    def _build_redemod_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(6)
        self.r_A = _spin(180.0, 1.0, 2000.0, 10.0, 1, "nm")
        self.r_h = _spin(0.0, 0.0, 1000.0, 5.0, 1, "nm")
        self.r_ltem = QCheckBox("LTEM (1−β) 因子")
        self.r_ltem.setToolTip("按 Pizzuto 的 LTEM FDM 再乘 (1 − β)。默认关闭：载流子模拟已给出针尖下方的发射场。")
        self.r_btn = button("运行重新解调", self._run_redemod)
        lay.addLayout(_flow(_group("A", self.r_A), _group("h_min", self.r_h), self.r_ltem, self.r_btn))
        self.r_hint = label("", "muted", wrap=True)
        lay.addWidget(self.r_hint)
        self.r_panel = RunPanel(show_log=False)
        self.r_panel.job_finished.connect(self._on_redemod_finished)
        lay.addWidget(self.r_panel)

        self.r_list = QComboBox()
        self.r_list.setMinimumWidth(200)
        self.r_list.setToolTip("本结果文件夹中的 redemod_*.json")
        self.r_list.currentIndexChanged.connect(lambda *_: self._show_redemod())
        self.r_family = QComboBox()
        self.r_family.addItem("dS_n/dt RMS（2.2 THz 低通）", "dS{n}dt_rms")
        self.r_family.addItem("S_n RMS（2.2 THz 低通）", "S{n}_rms")
        self.r_family.addItem("dS_n/dt 峰值", "dS{n}dt_peak")
        self.r_family.currentIndexChanged.connect(lambda *_: self._show_redemod())
        lay.addLayout(_flow(_group("已有结果", self.r_list), _group("指标", self.r_family)))
        vs = _Split()
        self.r_plot = PlotWidget()
        self.r_plot.setMinimumHeight(240)
        vs.addWidget(self.r_plot)
        self.r_table = QTableWidget()
        self.r_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.r_table.setAlternatingRowColors(True)
        self.r_table.verticalHeader().setDefaultSectionSize(24)
        vs.addWidget(self.r_table)
        vs.setStretchFactor(0, 1)
        vs.setStretchFactor(1, 0)
        vs.setSizes([460, 130])
        lay.addWidget(vs, 1)
        return _scrollable(w)

    def _build_history_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12)
        row = QHBoxLayout()
        row.addWidget(label("已有的发射链结果（双击载入，点表头排序）", "cardTitle", wrap=True), 1)
        for txt, slot in (("载入所选", self._load_selected_history), ("在访达中显示", self._reveal_selected_history),
                          ("刷新", self.refresh_history)):
            row.addWidget(button(txt, slot, "flat"))
        lay.addLayout(row)
        self.hist = QTreeWidget()
        self.hist.setHeaderLabels(["结果", "P (mW)", "w (µm)", "偏压 (V)", "T (K)", "扩散", "针尖图", "A (nm)"])
        self.hist.setIndentation(14)
        for c, tip in enumerate(("", "平均功率 P_avg_mW", "光斑半径 w_um", "V_bias", "载流子温度 T", "简并 / 经典扩散",
                                 "是否乘了 FDTD 针尖图", "运行时的轻敲振幅 A_nm（旧版运行为 60 nm）")):
            if tip:
                self.hist.headerItem().setToolTip(c, tip)
        self.hist.setAlternatingRowColors(True)
        self.hist.setUniformRowHeights(True)
        self.hist.setSortingEnabled(True)
        self.hist.sortByColumn(-1, Qt.SortOrder.AscendingOrder)
        self.hist.itemDoubleClicked.connect(lambda it, _c: self._load_history_item(it))
        h = self.hist.header()
        h.setStretchLastSection(False)
        h.setMinimumSectionSize(36)
        lay.addWidget(self.hist, 1)
        self.hist_note = label("* summary.json 未记录该项（旧版运行），显示的是当时脚本的默认值。", "muted", wrap=True)
        lay.addWidget(self.hist_note)
        self.refresh_history()
        return w

    def _retheme(self, *_):
        t = ThemeManager.instance().tokens
        self.run_btn.setIcon(icons.icon("play", t.on_accent))
        self.r_btn.setIcon(icons.icon("refresh", t.text))

    # ================================================================== run
    def refresh_fdtd_choices(self):
        cur = self.fdtd_combo.currentData()
        extra = [self.fdtd_combo.itemData(i) for i in range(self.fdtd_combo.count())
                 if self.fdtd_combo.itemData(i) not in (None, "none")]
        self.fdtd_combo.blockSignals(True)
        self.fdtd_combo.clear()
        self.fdtd_combo.addItem("无针尖图 (none)", "none")
        seen = set()
        cands = []
        if (paths.ROOT / REPO_RATIO).exists():
            cands.append((f"仓库 · {REPO_RATIO}", REPO_RATIO))
        for p in sorted(paths.GUI_RUNS.glob("fdtd_*/ratio.npz"), reverse=True):
            cands.append((f"GUI 运行 · {p.parent.name}", paths.rel(p)))
        for v in extra:
            if v not in [c[1] for c in cands] and paths.absolute(v).exists():
                cands.append((f"自选 · {Path(v).parent.name}/{Path(v).name}", v))
        for txt, v in cands:
            if v not in seen:
                seen.add(v)
                self.fdtd_combo.addItem(txt, v)
                self.fdtd_combo.setItemData(self.fdtd_combo.count() - 1, str(paths.absolute(v)), Qt.ItemDataRole.ToolTipRole)
        i = self.fdtd_combo.findData(cur) if cur else 0
        self.fdtd_combo.setCurrentIndex(max(0, i))
        self.fdtd_combo.blockSignals(False)

    def _select_fdtd(self, value):
        if not value or value == "none":
            self.fdtd_combo.setCurrentIndex(0)
            return
        i = self.fdtd_combo.findData(value)
        if i < 0 and paths.absolute(value).exists():
            self.fdtd_combo.addItem(f"自选 · {Path(value).parent.name}/{Path(value).name}", value)
            i = self.fdtd_combo.count() - 1
        self.fdtd_combo.setCurrentIndex(max(0, i))

    def _browse_fdtd(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择 FDTD 针尖比值图 ratio.npz", str(paths.RESULTS), "NumPy (*.npz)")
        if not p:
            return
        try:
            with np.load(p) as d:
                missing = [k for k in ("x", "z", "ratio") if k not in d.files]
        except Exception as e:      # unreadable / not an npz
            error_box(self, "无法读取", f"无法读取 {p}", str(e))
            return
        if missing:
            error_box(self, "不是针尖比值图", f"{Path(p).name} 缺少键 {', '.join(missing)}。\n"
                      "需要 scripts/run_fdtd_pump.py 输出的 ratio.npz（键 x, z, ratio）。")
            return
        self._select_fdtd(paths.rel(p))

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出文件夹", str(paths.RESULTS))
        if d:
            self.out_edit.setText(paths.rel(d))

    def _parse_form(self) -> tuple[dict | None, str]:
        """(kwargs for run_pulse_chain.main without out/fdtd, error message)."""
        if not self.form.is_valid():
            bad = []
            for n in self.form.invalid_fields():
                sp = self.form.rows[n].spec
                bad.append(n + (f"（≥ {sp.minimum:g}）" if sp.minimum is not None else ""))
            return None, "以下参数的值无法解析或超出范围：\n" + ", ".join(bad)
        v = self.form.values()
        raw = [float(x) for x in v["x_list"]]
        if not raw:
            return None, "光束位置 X 列表为空。"
        off = [x for x in raw if abs(x * 10 - round(x * 10)) > 1e-6]
        if off:
            return None, (f"X = {', '.join(f'{x:g}' for x in off)} µm 不是 0.1 µm 的整数倍：脚本按 X{{X:+.1f}}.npz "
                          "命名结果文件，只保留一位小数，会与其他 X 混淆或标错位置。")
        xs = sorted({round(x, 1) + 0.0 for x in raw})
        if any(n != int(n) for n in v["harmonics"]):
            return None, "谐波 n 必须是正整数。"
        harm = tuple(sorted({int(n) for n in v["harmonics"]}))
        if not harm or min(harm) < 1:
            return None, "谐波 n 必须是正整数。"
        if 2 not in harm or 3 not in harm:
            return None, "谐波列表必须包含 2 和 3：脚本的逐 X 进度输出用 S2，总览图用 dS3/dt。"
        xf = str(v["x_fine"]).strip()
        x_fine = None
        if xf:
            try:
                x_fine = float(xf)
            except ValueError:
                return None, f"x_fine = “{xf}” 不是数字（留空表示自动）。"
            if x_fine <= 0:
                return None, "x_fine 必须为正数。"
        kw = dict(P_avg_mW=float(v["P_avg_mW"]), A_nm=float(v["A_nm"]), h_min_nm=float(v["h_min_nm"]),
                  t_end_ps=float(v["t_end_ps"]), harmonics=harm, degenerate=1.0 if v["degenerate"] else 0.0,
                  T=float(v["T"]), V_bias=float(v["V_bias"]), x_list=tuple(xs), w_um=float(v["w_um"]), x_fine=x_fine)
        return kw, ""

    def _update_estimate(self, *_):
        kw, err = self._parse_form()
        if kw is None:
            self.est_title.setText("预计耗时 —")
            self.est_text.setText(err.replace("\n", " "))
            return
        xs = kw["x_list"]
        tot, lo, hi, per = estimate(len(xs), kw["w_um"], max(abs(x) for x in xs), kw["x_fine"], kw["t_end_ps"])
        self.est_title.setText(f"预计耗时 ≈ {fmt_minutes(tot)}")
        txt = (f"{len(xs)} 个 X × ≈ {per:.0f} s（每个 X 约 {per * EST_RANGE[0]:.0f}–{per * EST_RANGE[1]:.0f} s，视机器而定）。"
               f"范围 {fmt_minutes(lo)} – {fmt_minutes(hi)}。")
        if 0.0 not in xs:
            txt += " X 列表不含 0：运行中只有按 X 计数的进度。"
        self.est_text.setText(txt)

    def run(self):
        kw, err = self._parse_form()
        if kw is None:
            error_box(self, "参数无效", err)
            return
        fdtd = self.fdtd_combo.currentData() or "none"
        if fdtd != "none" and not paths.absolute(fdtd).exists():
            error_box(self, "针尖图不存在", f"{fdtd} 不存在。\n请选择其他针尖图、选「无针尖图」，或先在「泵浦 FDTD」页生成。")
            return
        # folders of chain jobs that are queued or running (a job's folder is created only when it starts)
        busy = {paths.absolute(j.out).resolve() for j in JobManager.instance().active() if j.out}
        out = self.out_edit.text().strip()
        if out:
            out = paths.absolute(out)
            if out.resolve() in busy:
                error_box(self, "输出文件夹正被占用", f"已有一个排队中或运行中的任务写入 {paths.rel(out)}。\n"
                          "请等它结束，或换一个输出文件夹（留空则自动新建）。")
                return
        else:
            base = out = paths.new_run_dir("pulse")
            i = 2
            while out.resolve() in busy or out.exists():        # e.g. two clicks within the same second
                out = base.with_name(f"{base.name}_{i}")
                i += 1
        if out.exists() and any(out.iterdir()):
            committed = is_committed(out)
            r = QMessageBox.question(
                self, "覆盖已有结果？",
                f"{paths.rel(out)} 已有文件，运行会覆盖其中的 X*.npz、summary.json 和 pulse_chain.png。"
                + ("\n\n注意：这是仓库中已提交的结果文件夹。" if committed else "") + "\n\n继续吗？")
            if r != QMessageBox.StandardButton.Yes:
                return
        xs = kw["x_list"]
        progress = {"type": "count", "pattern": r"^X=[+-]\d", "total": len(xs), "sub": r"t=([0-9.]+) ps",
                    "sub_total": kw["t_end_ps"]}
        title = (f"发射链 · {kw['P_avg_mW']:g} mW, w = {kw['w_um']:g} µm, {kw['V_bias']:+g} V, T = {kw['T']:g} K"
                 + (", 简并" if kw["degenerate"] else "") + (", 针尖图" if fdtd != "none" else "") + f" · {len(xs)} 个 X")
        self.job = JobManager.instance().submit(
            title, "pulse", CHAIN_SCRIPT, "main", kwargs=dict(out=paths.rel(out), fdtd=fdtd, **kw),
            out=out, progress=progress)
        self.panel.attach(self.job)
        self.tabs.setCurrentIndex(0)

    def _on_finished(self, job):
        if job is not self.job:
            return
        self.refresh_history()          # failed / cancelled runs leave a partial folder that the history lists
        if job.status == FINISHED and job.out:
            self.show_result(paths.absolute(job.out))

    def _on_any_job_finished(self, job):
        if job.kind == "pulse" and job is not self.job:
            self.refresh_history()
        elif job.kind == "redemod" and job is not self.redemod_job and job.status == FINISHED and self.run_dir:
            rd = (job.spec.get("kwargs") or {}).get("run_dir")
            if rd and paths.absolute(rd).resolve() == Path(self.run_dir).resolve():
                self.refresh_redemod_list()

    # ================================================================== results
    def _clear_result(self):
        self.res_title.setText("尚未载入结果")
        self.res_info.setText("在「历史结果」中双击一个结果，或运行新的发射链。")
        self.btn_params.setEnabled(False)
        self.btn_folder.setEnabled(False)
        self.overview.clear("尚未载入结果")
        self.xscan_plot.plot(self._draw_xscan)
        self.wave_plot.plot(self._draw_waves)
        self.band_plot.plot(self._draw_band)
        self.r_plot.plot(self._draw_redemod)
        self._update_redemod_hint()

    def show_result(self, d, switch=True):
        d = Path(d)
        if d.is_file():
            d = d.parent
        self._gen += 1
        self.run_dir = d
        self.summary = read_json(d / "summary.json")
        self.settings = run_settings(d)
        self.files = run_files(d)
        self.traces = {}
        self.band = None
        self._band_key = None
        state = ""
        if not d.is_dir():
            state = "  ·  ⚠ 文件夹不存在"
        elif not self.summary:
            state = "  ·  未完成（没有 summary.json）" if self.files else "  ·  ⚠ 没有 summary.json 和 X*.npz"
        self.res_title.setText(d.name + state)
        self.res_title.setToolTip(paths.rel(d))
        info = describe_run(self.settings)
        if not self.files and d.is_dir():
            info += ("　·　" if info else "") + "没有 X*.npz（波形、频带指标、重新解调不可用）"
        elif self.files and not self.summary:
            info += ("　·　" if info else "") + (f"只有 {len(self.files)} 个 X*.npz（X = "
                                                 + ", ".join(f"{x:+g}" for x in self.files)
                                                 + " µm）：幅度表和总览图不可用，波形、频带指标、重新解调可用")
        elif self.files:
            # a re-run into this folder that was interrupted rewrites X*.npz but not summary.json / pulse_chain.png
            try:
                t_sum = (d / "summary.json").stat().st_mtime
                newer = [x for x, p in self.files.items() if p.stat().st_mtime > t_sum + 120.0]   # margin: plain copies reset mtimes
            except OSError:
                newer = []
            if newer:
                info += ("　·　" if info else "") + ("⚠ X = " + ", ".join(f"{x:+g}" for x in newer)
                                                     + " 的 X*.npz 比 summary.json 新（重新运行被中断？）："
                                                     "幅度表和总览图可能与波形、频带指标不一致")
        self.res_info.setText(info)
        self.res_info.setToolTip("* 旧版 summary.json 未记录该项，按当时脚本的默认值" if self.settings.get("_inferred") else "")
        self.btn_params.setEnabled(bool(self.settings) and set(self.settings) != {"_inferred"})
        self.btn_folder.setEnabled(d.exists())

        # metric selector
        prev = self.metric_combo.currentData()
        keys = sorted((self.summary.get("rms") or {}).keys(), key=metric_order)
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        for k in keys:
            self.metric_combo.addItem(f"{k}  ·  {metric_desc(k)}", k)
        want = prev if prev in keys else ("dS3_dt" if "dS3_dt" in keys else (keys[0] if keys else None))
        if want:
            self.metric_combo.setCurrentIndex(keys.index(want))
        self.metric_combo.blockSignals(False)
        self._metric_changed()
        if not self.overview.load(d / "pulse_chain.png"):
            self.overview.clear("没有 pulse_chain.png（运行未完成或被中断）")

        # waveforms: load the time traces off the GUI thread
        self.x_list.clear()
        self.trace_combo.clear()
        self.wave_plot.plot(self._draw_waves)
        if self.files:
            gen = self._gen
            run_async(load_traces, dict(self.files), on_done=lambda r, g=gen: self._traces_loaded(g, r),
                      on_error=lambda e, g=gen: g == self._gen and self._wave_error(e))
        # band metric: computed lazily when its tab is visible
        self.band_plot.plot(self._draw_band)
        self._fill_band_table()
        self.b_status.setText("" if self.files else "该结果没有 X*.npz，无法计算频带指标。")
        if self.tabs.currentWidget() is self.band_tab:
            self.compute_band()
        # redemodulation
        self.refresh_redemod_list()
        self._update_redemod_hint()
        if switch and self.tabs.currentWidget() not in (self.res_tab, self.wave_tab, self.band_tab, self.redemod_tab):
            self.tabs.setCurrentWidget(self.res_tab)

    def _current_metric(self):
        return self.metric_combo.currentData(), self.stat_combo.currentData() or "rms"

    def _metric_changed(self, *_):
        key, stat = self._current_metric()
        s = self.summary
        m = {}
        if key and s.get(stat, {}).get(key) is not None and s.get("X"):
            m = xscan_metrics(s["X"], s[stat][key])
        self.m_peak.set(f"{m['peak_X']:+.1f} µm" if "peak_X" in m else "—", "实测：+5 µm")
        r5 = m.get("ratio5")
        self.m_ratio.set("—" if r5 is None else ("∞" if not np.isfinite(r5) else f"{r5:.2f}"),
                         "实测：≫ 1" if r5 is not None else "需要 X 覆盖 ±5 µm")
        self.m_zero.set(f"{m['zero']:.2f}" if "zero" in m else "—", "实测：0.24" if "zero" in m else "需要 X 覆盖 0")
        self.m_dev.set(f"{m['dev']:.2f}" if "dev" in m else "—",
                       f"{m['n_meas']} 个实测点上的 RMS 差" if "dev" in m else "需要 ≥ 2 个 X 覆盖实测点")
        self.tiles.sync()
        self.xscan_plot.plot(self._draw_xscan)

    def _plot_measured(self, ax, neg=True):
        t = ThemeManager.instance().tokens
        M = load_measured()
        pos = [r for r in M if r["bias"] > 0]
        if pos:
            ax.errorbar([r["X"] for r in pos], [r["rms"] for r in pos], yerr=[r["err"] for r in pos], fmt="o",
                        color=t.text, ms=6, capsize=3, lw=1.3, zorder=6, label=f"measured {pos[0]['bias']:+g} V")
        ng = [r for r in M if r["bias"] < 0]
        if neg and ng:
            ax.errorbar([-r["X"] for r in ng], [r["rms"] for r in ng], yerr=[r["err"] for r in ng], fmt="s",
                        mfc=t.surface, mec=t.text, ecolor=t.text, ms=6, capsize=3, lw=1.1, zorder=6,
                        label=f"measured {ng[0]['bias']:+g} V (X → −X)")

    @staticmethod
    def _decorate_x(ax, t):
        ax.axvline(0, color=t.muted, lw=0.8, ls=":", zorder=0)
        ax.set_xlabel("X = beam position relative to tip (µm)      [anode at −x]", wrap=True)
        ax.grid(True, alpha=0.5)

    @staticmethod
    def _empty(fig, msg):
        t = ThemeManager.instance().tokens
        fig.text(0.5, 0.5, msg, ha="center", va="center", color=t.muted, fontsize=11, wrap=True)

    def _draw_xscan(self, fig):
        key, stat = self._current_metric()
        s = self.summary
        if not key or not s.get("X") or key not in s.get(stat, {}):
            self._empty(fig, "No summary.json loaded" if self.run_dir is None else "summary.json missing for this run")
            return
        t = ThemeManager.instance().tokens
        ax = fig.add_subplot()
        X = np.asarray(s["X"], float)
        o = np.argsort(X)
        keys = [key]
        m = re.match(r"(d?S)\d(_dt)?$", key)
        if self.cb_family.isChecked() and m:
            fam = sorted((k for k in s[stat] if re.match(rf"{m.group(1)}\d{m.group(2) or ''}$", k)), key=metric_order)
            keys = fam
        cyc = _cycle()
        for i, k in enumerate(keys):
            y = np.abs(np.asarray(s[stat][k], float))[o]
            mx = np.nanmax(y) if y.size else 0
            if not mx > 0:
                continue
            main = k == key
            ax.plot(X[o], y / mx, "o-", color=cyc[0] if main else cyc[(i + 1) % len(cyc)], lw=2.2 if main else 1.2,
                    ms=6 if main else 4, alpha=1.0 if main else 0.8, zorder=5 if main else 3, label=f"model {k}")
        self._plot_measured(ax, self.cb_neg.isChecked())
        self._decorate_x(ax, t)
        ax.set_ylabel("amplitude (norm. to max)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{'RMS' if stat == 'rms' else 'peak'} of the 2.2 THz low-passed trace (summary.json)", wrap=True)
        _legend_right(ax)

    # ---------------------------------------------------------------- waveforms
    def _wave_error(self, e):
        self.wave_plot.plot(lambda fig: self._empty(fig, "Could not read X*.npz:\n" + e.splitlines()[0]))

    def _traces_loaded(self, gen, tr):
        if gen != self._gen:
            return
        self.traces = tr
        keys = sorted({k for d in tr.values() for k in d if k != "t"}, key=metric_order)
        self.trace_combo.blockSignals(True)
        self.trace_combo.clear()
        for k in keys:
            self.trace_combo.addItem(k, k)
            self.trace_combo.setItemData(self.trace_combo.count() - 1, metric_desc(k), Qt.ItemDataRole.ToolTipRole)
        i = self.trace_combo.findData(self._trace_pref)
        if i < 0:
            i = self.trace_combo.findData("dS3_dt")
        self.trace_combo.setCurrentIndex(max(i, 0))
        self.trace_combo.blockSignals(False)
        self.x_list.blockSignals(True)
        self.x_list.clear()
        default = {0.0, 2.5, -2.5, 5.0, -5.0}
        pick = default & set(tr) or set(tr)
        for X in sorted(tr):
            it = QListWidgetItem(f"X = {X:+.1f} µm")
            it.setData(Qt.ItemDataRole.UserRole, X)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if X in pick else Qt.CheckState.Unchecked)
            self.x_list.addItem(it)
        self.x_list.blockSignals(False)
        self._redraw_waves()

    def _trace_changed(self, *_):
        if self.trace_combo.currentData():
            self._trace_pref = self.trace_combo.currentData()
        self._redraw_waves()

    def _check_all_x(self, on):
        self.x_list.blockSignals(True)
        for i in range(self.x_list.count()):
            self.x_list.item(i).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self.x_list.blockSignals(False)
        self._redraw_waves()

    def _redraw_waves(self, *_):
        self.wave_plot.plot(self._draw_waves)

    def selected_x(self):
        return [self.x_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.x_list.count())
                if self.x_list.item(i).checkState() == Qt.CheckState.Checked]

    def _draw_waves(self, fig):
        if self.run_dir is None:
            self._empty(fig, "No run loaded")
            return
        if not self.files:
            self._empty(fig, "This run has no X*.npz trace files\n(they are not in the repository or the run was interrupted)")
            return
        if not self.traces:
            self._empty(fig, "Loading traces …")
            return
        key = self.trace_combo.currentData()
        sel = self.selected_x()
        if not key or not sel:
            self._empty(fig, "Select a trace and at least one X")
            return
        from ltemsim.tip_demod import lowpass, lowpass_steep
        t_tok = ThemeManager.instance().tokens
        deriv = int(self.deriv_combo.currentData() or 0)
        filt = self.filter_combo.currentData()
        base, k0 = trace_unit(key)
        order = k0 + deriv
        curves = []
        for X in sorted(sel):
            d = self.traces.get(X, {})
            if key not in d:
                continue
            t, y = d["t"], d[key]
            for _ in range(deriv):
                y = np.gradient(y, t)
            if filt == "gauss":
                y = lowpass(t, y)
            elif filt == "steep":
                y = lowpass_steep(t, y)
            curves.append((X, t, y * (1e-12) ** order))
        if not curves:
            self._empty(fig, f"'{key}' is not stored in the selected X files")
            return
        ax = fig.add_subplot()
        norm = self.cb_wnorm.isChecked()
        scale = max(float(np.max(np.abs(c[2]))) for c in curves) if norm else 1.0
        scale = scale if scale > 0 else 1.0
        cyc = _cycle()
        for i, (X, t, y) in enumerate(curves):
            ax.plot(t * 1e12, y / scale, color=t_tok.text if X == 0 else cyc[i % len(cyc)], lw=1.4,
                    ls="-" if i < len(cyc) else "--", label=f"X = {X:+.1f} µm")
        ax.axhline(0, color=t_tok.muted, lw=0.6, zorder=0)
        unit = base + ("" if order == 0 else ("/ps" if order == 1 else f"/ps$^{order}$"))
        dname = {0: "", 1: "d/dt ", 2: "d²/dt² "}[deriv]
        ax.set_xlabel("t (ps)")
        ax.set_ylabel(f"{dname}{key} (norm.)" if norm else f"{dname}{key} ({unit})" if unit else f"{dname}{key}")
        fname = {"gauss": ", 2.2 THz Gaussian low-pass", "steep": ", 2.2 THz Butterworth (order 8)", "none": ""}[filt]
        ax.set_title(f"{dname}{key}{fname}", wrap=True)
        ax.grid(True, alpha=0.5)
        ax.legend(fontsize=8, loc="best", ncol=2 if len(curves) > 5 else 1)

    # ---------------------------------------------------------------- band metric
    def _band_params(self):
        harm = tuple(n for n, cb in self.b_harm.items() if cb.isChecked())
        lo, hi = self.b_lo.value(), self.b_hi.value()
        return (round(self.b_A.value(), 3), round(self.b_h.value(), 3), harm, int(self.b_deriv.currentData()),
                (round(lo, 3), round(hi, 3)))

    def _band_params_changed(self, *_):
        self._band_timer.start()

    def compute_band(self):
        if self.run_dir is None or not self.files:
            self.b_status.setText("该结果没有 X*.npz，无法计算频带指标。" if self.run_dir else "尚未载入结果。")
            return
        A, h, harm, deriv, band = self._band_params()
        if not harm:
            self.b_status.setText("请至少选择一个谐波。")
            return
        if band[1] <= band[0]:
            self.b_status.setText("频带上限必须大于下限。")
            return
        key = (self._gen, A, h, harm, deriv, band)
        if key == self._band_key or key == self._band_want:
            return
        self._band_want = key
        self.b_status.setText(f"正在计算：{len(self.files)} 个 X 的重新解调与频谱 …")
        self.b_run.setEnabled(False)
        run_async(band_table, dict(self.files), A, h, harm, deriv, band,
                  on_done=lambda r, k=key: self._band_done(k, r), on_error=lambda e, k=key: self._band_failed(k, e))

    def _band_done(self, key, res):
        if key != self._band_want:
            return
        self._band_want = None
        self.b_run.setEnabled(True)
        if key[0] != self._gen:
            return
        self._band_key = key
        res["params"] = dict(A=key[1], h=key[2], harm=key[3], deriv=key[4], band=key[5])
        self.band = res
        n_ref = 3 if 3 in res["n"] else key[3][0]
        m = xscan_metrics(res["X"], res["n"][n_ref])
        parts = [f"n = {n_ref}：峰值 X = {m['peak_X']:+.1f} µm" if "peak_X" in m else f"n = {n_ref}"]
        if "ratio5" in m:
            parts.append(f"(+5)/(−5) = {m['ratio5']:.2f}")
        if "zero" in m:
            parts.append(f"X = 0 处 {m['zero']:.2f}")
        if "dev" in m:
            parts.append(f"与 +25 V 实测 RMS 偏差 {m['dev']:.2f}")
        self.b_status.setText("　·　".join(parts))
        self.band_plot.plot(self._draw_band)
        self._fill_band_table()

    def _band_failed(self, key, err):
        if key == self._band_want:
            self._band_want = None
            self.b_run.setEnabled(True)
        if key[0] == self._gen:
            self.b_status.setText("计算失败：" + err.splitlines()[0])

    def _draw_band(self, fig):
        if self.run_dir is None:
            self._empty(fig, "No run loaded")
            return
        if not self.files:
            self._empty(fig, "This run has no X*.npz trace files — the band metric needs them")
            return
        b = self.band
        if not b:
            self._empty(fig, "Computing …")
            return
        t = ThemeManager.instance().tokens
        ax = fig.add_subplot()
        X = b["X"]
        cyc = _cycle()
        for i, (n, y) in enumerate(b["n"].items()):
            mx = np.nanmax(y) if len(y) else 0
            if mx > 0:
                ax.plot(X, y / mx, "o-", color=cyc[(n - 1) % len(cyc)], lw=2.4 if n == 3 else 1.3, ms=6 if n == 3 else 4,
                        zorder=5 if n == 3 else 3, label=f"model n = {n}")
        if self.b_far.isChecked() and np.nanmax(b["far"]) > 0:
            ax.plot(X, b["far"] / np.nanmax(b["far"]), "^--", color=t.muted, lw=1.2, ms=4, label="far field dJx/dt")
        self._plot_measured(ax, self.b_neg.isChecked())
        self._decorate_x(ax, t)
        p = b["params"]
        ax.set_ylim(bottom=0)
        ax.set_ylabel("amplitude (norm. to max)")
        ax.set_title(f"{p['band'][0]:g}–{p['band'][1]:g} THz amplitude of "
                     f"{'d²S_n/dt²' if p['deriv'] == 2 else 'dS_n/dt'}  (A = {p['A']:g} nm, h_min = {p['h']:g} nm)", wrap=True)
        _legend_right(ax)

    def _fill_band_table(self):
        tw = self.band_table_w
        tw.clear()
        b = self.band
        if not b:
            tw.setRowCount(0)
            tw.setColumnCount(0)
            return
        X = b["X"]
        rows = [(f"n = {n}", y) for n, y in b["n"].items()] + [("远场 dJx/dt", b["far"])]
        tw.setColumnCount(len(X) + 1)
        tw.setRowCount(len(rows))
        tw.setHorizontalHeaderLabels([f"{x:+g}" for x in X] + ["最大值"])
        tw.setVerticalHeaderLabels([r[0] for r in rows])
        for i, (_, y) in enumerate(rows):
            mx = np.nanmax(y) if len(y) else np.nan
            for j, v in enumerate(y):
                it = QTableWidgetItem(f"{v / mx:.2f}" if mx > 0 else "—")
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                it.setToolTip(f"{v:.4g}")
                tw.setItem(i, j, it)
            it = QTableWidgetItem(f"{mx:.3g}")
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            tw.setItem(i, len(X), it)
        _fit_table(tw, len(rows))

    # ---------------------------------------------------------------- redemodulation
    def _update_redemod_hint(self):
        d = self.run_dir
        if d is None:
            self.r_hint.setText("先载入一个结果。")
            self.r_btn.setEnabled(False)
            return
        ok = bool(self.files)
        self.r_btn.setEnabled(ok)
        if not ok:
            self.r_hint.setText("该结果没有 X*.npz，无法重新解调。")
            return
        n = len(self.files)
        txt = (f"scripts/redemodulate.py（{n} 个 X，约 {max(1, round(0.3 * n))} s）在结果文件夹中写入 "
               "redemod_A…_h….json。旧指标（2.2 THz 低通后的 RMS），推荐对比见「频带指标」。")
        if is_committed(d):
            txt += "\n⚠ 仓库中已提交的结果文件夹：只新增或覆盖这一个 JSON，运行前会再次确认。"
        self.r_hint.setText(txt)

    def _confirm(self, title, text) -> bool:
        return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes

    def _run_redemod(self):
        d = self.run_dir
        if d is None or not self.files:
            error_box(self, "无法重新解调", "当前结果没有 X*.npz。")
            return
        A, h, ltem = self.r_A.value(), self.r_h.value(), self.r_ltem.isChecked()
        target = d / redemod_name(A, h, ltem)
        warn = []
        if is_committed(d):
            warn.append(f"{paths.rel(d)} 是仓库中已提交的结果文件夹，脚本会在其中写入 {target.name}（只写这一个文件）。")
        if target.exists():
            warn.append(f"{target.name} 已存在，将被覆盖。")
        if warn and not self._confirm("写入结果文件夹？", "\n\n".join(warn) + "\n\n继续吗？"):
            return
        title = f"重新解调 · {d.name} · A = {A:g} nm, h_min = {h:g} nm" + ("，LTEM 因子" if ltem else "")
        # no out=: the runner would add gui_run_main.json / gui_log_main.txt next to the JSON, and the confirmation
        # above promises that only the one redemod_*.json is written into the run folder
        self.redemod_job = JobManager.instance().submit(
            title, "redemod", "scripts/redemodulate.py", "main",
            kwargs=dict(run_dir=paths.rel(d), A_nm=float(A), h_min_nm=float(h), ltem_factor=bool(ltem)))
        self._redemod_target = (d, target)
        self.r_panel.attach(self.redemod_job)

    def _on_redemod_finished(self, job):
        if job is not self.redemod_job or self._redemod_target is None:
            return
        d, target = self._redemod_target
        if job.status == FINISHED and self.run_dir is not None and Path(self.run_dir) == Path(d):
            self.refresh_redemod_list(select=target)

    def refresh_redemod_list(self, select: Path | None = None):
        cur = self.r_list.currentData()
        if select is None and self.run_dir is not None and cur and Path(cur).parent != Path(self.run_dir):
            cur = None
        self.r_list.blockSignals(True)
        self.r_list.clear()
        files = sorted(self.run_dir.glob("redemod_*.json")) if self.run_dir and self.run_dir.is_dir() else []
        for p in files:
            self.r_list.addItem(redemod_label(p), str(p))
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
            self.r_list.setItemData(self.r_list.count() - 1, f"{p.name}\n修改于 {when}", Qt.ItemDataRole.ToolTipRole)
        if files:
            i = self.r_list.findData(str(select) if select else cur)
            self.r_list.setCurrentIndex(max(i, 0))
        else:
            self.r_list.addItem("（还没有）", None)
        self.r_list.setEnabled(bool(files))
        self.r_list.blockSignals(False)
        self._show_redemod()

    def _redemod_data(self):
        f = self.r_list.currentData()
        if not f:
            return None
        p = Path(f)
        X, tab = load_redemod(p)
        return p, X, tab

    def _show_redemod(self):
        self.r_plot.plot(self._draw_redemod)
        tw = self.r_table
        tw.clear()
        data = self._redemod_data()
        fam = self.r_family.currentData()
        keys = [fam.format(n=n) for n in (1, 2, 3, 4)]
        keys = [k for k in keys if data and k in data[2]]
        ok = bool(data and len(data[1]) and keys)
        tw.setVisible(ok)
        if not ok:
            tw.setRowCount(0)
            tw.setColumnCount(0)
            return
        _, X, tab = data
        o = np.argsort(X)
        tw.setColumnCount(len(X) + 1)
        tw.setRowCount(len(keys))
        tw.setHorizontalHeaderLabels([f"{x:+g}" for x in X[o]] + ["最大值"])
        tw.setVerticalHeaderLabels(keys)
        for i, k in enumerate(keys):
            y = np.abs(tab[k])[o]
            mx = np.nanmax(y) if len(y) else np.nan
            for j, v in enumerate(y):
                it = QTableWidgetItem(f"{v / mx:.2f}" if mx > 0 else "—")
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                it.setToolTip(f"{v:.4g}")
                tw.setItem(i, j, it)
            it = QTableWidgetItem(f"{mx:.3g}")
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            tw.setItem(i, len(X), it)
        _fit_table(tw, len(keys))

    def _draw_redemod(self, fig):
        if self.run_dir is None:
            self._empty(fig, "No run loaded")
            return
        data = self._redemod_data()
        if not data:
            self._empty(fig, "No redemod_*.json in this run yet —\nset A / h_min above and run the redemodulation")
            return
        p, X, tab = data
        if not len(X):
            self._empty(fig, f"{p.name} is empty (no X*.npz when it was written)")
            return
        t = ThemeManager.instance().tokens
        fam = self.r_family.currentData()
        ax = fig.add_subplot()
        o = np.argsort(X)
        cyc = _cycle()
        for n in (1, 2, 3, 4):
            k = fam.format(n=n)
            if k not in tab:
                continue
            y = np.abs(tab[k])[o]
            mx = np.nanmax(y)
            if mx > 0:
                ax.plot(X[o], y / mx, "o-", color=cyc[(n - 1) % len(cyc)], lw=2.4 if n == 3 else 1.3,
                        ms=6 if n == 3 else 4, label=f"model {k}")
        self._plot_measured(ax, True)
        self._decorate_x(ax, t)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("amplitude (norm. to max)")
        fam_en = {"dS{n}dt_rms": "RMS of low-passed dS_n/dt", "S{n}_rms": "RMS of low-passed S_n",
                  "dS{n}dt_peak": "peak of low-passed dS_n/dt"}.get(fam, fam)
        ax.set_title(f"{fam_en}  ({p.name})", wrap=True)
        _legend_right(ax)

    # ================================================================== history
    def _history_dirs(self):
        repo, gui = [], []
        for d in sorted((paths.RESULTS / "pulse_chain").glob("*")):
            if d.is_dir() and ((d / "summary.json").exists() or run_files(d)):
                repo.append(d)
        for d in sorted(paths.GUI_RUNS.glob("pulse_*"), reverse=True):
            if d.is_dir() and ((d / "summary.json").exists() or run_files(d) or (d / "gui_run.json").exists()):
                gui.append(d)
        return repo, gui

    def refresh_history(self):
        self.hist.setSortingEnabled(False)
        self.hist.clear()
        repo, gui = self._history_dirs()
        R = Qt.ItemDataRole.UserRole
        for title, dirs in (("仓库 · results/pulse_chain", repo), ("GUI 运行 · gui_runs/pulse_*", gui)):
            top = QTreeWidgetItem([f"{title}（{len(dirs)}）"])
            top.setFlags(Qt.ItemFlag.ItemIsEnabled)
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            self.hist.addTopLevelItem(top)
            for d in dirs:
                st = run_settings(d)
                inf = set(st.get("_inferred", ()))
                star = lambda k: "*" if k in inf else ""
                n_x = len(st.get("x_list") or run_files(d))
                done = (d / "summary.json").exists()
                vals = [
                    (d.name + ("" if done else "（未完成）"), d.name),
                    (f"{st['P_avg_mW']:g}" if "P_avg_mW" in st else "—", st.get("P_avg_mW")),
                    (f"{st['w_um']:g}{star('w_um')}" if "w_um" in st else "—", st.get("w_um")),
                    ((f"{st['V_bias']:+g}" if st["V_bias"] else "0") + star("V_bias") if "V_bias" in st else "—",
                     st.get("V_bias")),
                    (f"{st['T']:g}{star('T')}" if "T" in st else "—", st.get("T")),
                    ((("简并" if st["degenerate"] else "经典") + star("degenerate")) if "degenerate" in st else "—",
                     int(bool(st.get("degenerate")))),
                    (("有" if st["fdtd_ratio"] else "无") if "fdtd_ratio" in st else "—", int(bool(st.get("fdtd_ratio")))),
                    (f"{st['A_nm']:g}" if "A_nm" in st else "—", st.get("A_nm")),
                ]
                it = _SortItem([v[0] for v in vals])
                for c, (_, key) in enumerate(vals):
                    if key is not None:
                        it.setData(c, R + 1, key)
                it.setData(0, R, str(d))
                mtime = (d / "summary.json" if done else d).stat().st_mtime
                xs = st.get("x_list") or tuple(run_files(d))
                xtxt = f"{n_x} 个 X：" + ", ".join(f"{x:+g}" for x in xs) if xs else "没有 X 数据"
                it.setToolTip(0, f"{paths.rel(d)}\n{xtxt}\n修改于 {time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))}")
                for c in range(1, len(vals)):
                    it.setTextAlignment(c, Qt.AlignmentFlag.AlignCenter)
                    if "*" in vals[c][0]:
                        it.setToolTip(c, "summary.json 未记录（旧版运行），按当时脚本的默认值")
                if not done:
                    it.setForeground(0, self.hist.palette().placeholderText())
                top.addChild(it)
            top.setExpanded(True)
            if not dirs:
                e = QTreeWidgetItem(["还没有结果" if dirs is repo else "还没有 GUI 运行的发射链结果"])
                e.setFlags(Qt.ItemFlag.ItemIsEnabled)
                e.setForeground(0, self.hist.palette().placeholderText())
                top.addChild(e)
        self.hist.setSortingEnabled(True)
        self._fit_history_columns()

    def _fit_history_columns(self):
        """Name column to contents, the narrow value columns to header/contents (+ room for the sort arrow)."""
        tw, h = self.hist, self.hist.header()
        fm = tw.fontMetrics()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        rows = [tw.topLevelItem(i).child(j) for i in range(tw.topLevelItemCount())
                for j in range(tw.topLevelItem(i).childCount())]
        for c in range(1, tw.columnCount()):
            w = max([fm.horizontalAdvance(tw.headerItem().text(c)) + 26] + [fm.horizontalAdvance(r.text(c)) + 14 for r in rows])
            h.setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)
            tw.setColumnWidth(c, w)

    def _load_history_item(self, it):
        d = it.data(0, Qt.ItemDataRole.UserRole) if it else None
        if d:
            self.show_result(d)

    def _load_selected_history(self):
        self._load_history_item(self.hist.currentItem())

    def _reveal_selected_history(self):
        it = self.hist.currentItem()
        d = it.data(0, Qt.ItemDataRole.UserRole) if it else None
        if d:
            reveal(d)

    # ================================================================== misc
    def _load_result_params(self):
        st = self.settings
        if not st:
            return
        vals = {}
        for k in ("P_avg_mW", "V_bias", "w_um", "T", "t_end_ps", "A_nm", "h_min_nm"):
            if k in st:
                vals[k] = float(st[k])
        if "degenerate" in st:
            vals["degenerate"] = bool(st["degenerate"])
        if st.get("x_list"):
            vals["x_list"] = tuple(float(x) for x in st["x_list"])
        if st.get("harmonics"):
            vals["harmonics"] = tuple(int(n) for n in st["harmonics"])
        vals["x_fine"] = "" if st.get("x_fine") in (None, "") else str(st["x_fine"])
        self.form.set_values(vals, reset_others=True)
        self.form.rows["x_list"].editor.setCursorPosition(0)
        fd = st["fdtd"] if "fdtd" in st else (REPO_RATIO if st["fdtd_ratio"] else "none") if "fdtd_ratio" in st else None
        if fd is not None:
            self._select_fdtd(fd)
        self._update_estimate()
        if fd not in (None, "", "none") and not paths.absolute(fd).exists():
            error_box(self, "针尖图不存在", f"该结果使用的针尖图 {fd} 已不存在，左侧已改为「无针尖图」。\n"
                      "如需复现，请先在「泵浦 FDTD」页重新生成，或用「浏览…」选择其他 ratio.npz。")

    def _on_tab(self, i):
        w = self.tabs.widget(i)
        if w is self.band_tab and self.files and self._band_key is None:
            self.compute_band()
        elif w is self.hist_tab:
            self.refresh_history()

    def on_show(self):
        self.refresh_fdtd_choices()
        if not self._shown_once:
            self._shown_once = True
            if self.run_dir is None:
                d = paths.RESULTS / "pulse_chain" / DEFAULT_RUN
                if d.is_dir():
                    self.show_result(d, switch=self.job is None)
        else:
            self.refresh_history()
            if self.run_dir is not None:
                self.refresh_redemod_list()

    def open(self, run_dir=None, path=None, **_):
        target = run_dir or path
        if target:
            self._shown_once = True
            self.show_result(paths.absolute(target))
            self.tabs.setCurrentWidget(self.res_tab)
