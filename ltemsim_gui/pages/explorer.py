"""Interactive explorer for steady-state solutions (solution.npz written by Solution.save).

Everything here is computed in-process from the saved solution (Simulation(sol.params) is ~10 ms to build,
a tip scan a few ms): 2-D maps with line cuts, depth profiles, tip scans vs the measured RMS, and overlays of
several solutions.  No simulation is ever started from this page.
"""
from __future__ import annotations

import csv
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import patheffects
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QCursor, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFileDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLayout, QLineEdit,
                               QListWidget, QListWidgetItem, QMenu, QPlainTextEdit, QRadioButton, QScrollArea,
                               QSizePolicy, QSlider, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)

from .. import icons, paths
from ..theme import LIGHT, ThemeManager, Tokens, mpl_rc, repolish
from ..widgets.common import MetricTile, Page, button, card, label, reveal
from ..widgets.device import FieldView, device_from_params, draw_device, load_steady_fields
from ..widgets.mpl import PlotWidget
from ..workers import run_async

COARSE_DIR = paths.RESULTS / "coarse_scenarios" / "solutions"
DEFAULT_SOLUTION = COARSE_DIR / "coarse_P2.00.npz"
GOLD = "#c9a227"                      # Au electrodes
MAX_COMPARE = 8                       # one categorical hue each, never cycled
DIVERGING = ["RdBu_r", "coolwarm", "PuOr_r", "BrBG"]
SEQUENTIAL = ["viridis", "magma", "inferno", "plasma", "cividis", "Greys"]
LIGHT_MID = {"RdBu_r", "coolwarm", "PuOr_r", "BrBG", "Greys"}   # colormaps whose low/mid end is pale -> dark overlays


@dataclass(frozen=True)
class Quantity:
    key: str
    name: str          # combo text (UI)
    label: str         # axis / colour-bar label (plot)
    semi: bool         # defined on GaAs cells only
    kind: str          # "signed" | "pos" | "log"
    cmap: str


QUANTITIES = [
    Quantity("V", "Potential V", "potential V (V)", False, "signed", "RdBu_r"),
    Quantity("Ex", "Field E_x", "E_x (kV/cm)", False, "signed", "RdBu_r"),
    Quantity("Ez", "Field E_z (+ up)", "E_z (kV/cm, + up)", False, "signed", "RdBu_r"),
    Quantity("E", "Field magnitude |E|", "|E| (kV/cm)", False, "pos", "magma"),
    Quantity("logn", "Electrons n (log10)", "log10 n (n in cm⁻³)", True, "log", "viridis"),
    Quantity("logp", "Holes p (log10)", "log10 p (p in cm⁻³)", True, "log", "viridis"),
    Quantity("logT", "Ionized EL2⁺ (log10)", "log10 EL2⁺ (EL2⁺ in cm⁻³)", True, "log", "magma"),
    Quantity("NTp", "Ionized EL2⁺ (linear)", "EL2⁺ (cm⁻³)", True, "pos", "magma"),
    Quantity("f", "EL2 neutral fraction f", "EL2 neutral fraction f", True, "pos", "cividis"),
    Quantity("G", "Generation G / G_max", "generation G / G_max", True, "pos", "inferno"),
    Quantity("w", "Emission n_pulse·|E|", "emission weight n_pulse·|E| (kV/cm)", True, "pos", "viridis"),
    Quantity("Px", "P_x = n_pulse·E_x", "P_x ∝ n_pulse·E_x (arb.)", True, "signed", "RdBu_r"),
    Quantity("Pz", "P_z = n_pulse·E_z", "P_z ∝ n_pulse·E_z (arb.)", True, "signed", "RdBu_r"),
]
QBY = {q.key: q for q in QUANTITIES}
# compact axis labels for the (short) line-cut panels
CUT_LABEL = {"V": "V (V)", "Ez": "E_z (kV/cm)", "logn": "log10 n (cm⁻³)", "logp": "log10 p (cm⁻³)",
             "logT": "log10 EL2⁺ (cm⁻³)", "f": "f (EL2 neutral)", "G": "G / G_max", "w": "n_pulse·|E| (kV/cm)",
             "Px": "P_x (arb.)", "Pz": "P_z (arb.)"}

# parameters shown elsewhere in the info card or irrelevant for telling solutions apart
_NOT_LISTED = {"V_bias", "anode_side", "P_avg_mW", "spot_w_um", "tip_x_scan_um", "generation_ramp", "gummel_max_iter",
               "gummel_tol", "poisson_max_iter", "poisson_tol", "bias_ramp_steps", "rep_rate"}


# ---------------------------------------------------------------------------------------------- data
@dataclass
class Loaded:
    path: str
    name: str
    mtime: float
    sol: object
    sim: object
    data: dict                   # quantity key -> 2-D array ((Nz, Nx) or (Nzs, Nx))
    raw: dict                    # physical n, p (cm^-3) for the read-out
    electrodes: list = field(default_factory=list)                 # [(x_a, x_b)] of the Au films as discretised
    lock: threading.Lock = field(default_factory=threading.Lock)   # serialises Poisson solves on this sim

    @property
    def g(self):
        return self.sim.grid

    @property
    def P(self):
        return self.sol.params


def display_name(path) -> str:
    p = Path(path)
    if p.name == "solution.npz":
        try:
            r = p.parent.resolve().relative_to(paths.RESULTS.resolve())
            parts = r.parts[1:] if r.parts and r.parts[0] == "gui_runs" else r.parts
            return "/".join(parts) or p.parent.name
        except ValueError:
            return p.parent.name
    return p.stem


def load_solution(path: str) -> Loaded:
    """Solution + Simulation + every mapped quantity (runs in a worker thread)."""
    from ltemsim import Simulation, Solution
    try:
        sol = Solution.load(path)
    except FileNotFoundError:
        raise FileNotFoundError("File not found (it may have been moved or deleted)") from None
    except Exception as e:          # zip errors, missing members, wrong dtypes ...
        raise ValueError(f"Not a solution file written by Solution.save ({type(e).__name__}: {e})") from None
    sim = Simulation(sol.params, verbose=False)
    g = sim.grid
    sg = sol.grid
    if (g.xe.shape != sg.xe.shape or g.ze.shape != sg.ze.shape or g.ks != sg.ks
            or not (np.allclose(g.xe, sg.xe) and np.allclose(g.ze, sg.ze))):
        raise ValueError("The grid rebuilt from the saved parameters does not match the saved grid, so the fields cannot be "
                         "computed (the parameter definitions may have changed).")
    ks = g.ks
    Ex, Ez = sol.fields_physical(sim)
    n, p, NTp = sol.densities_physical(sim)
    V = sol.potential_V(sim)
    Gs = np.asarray(sol.G, float).reshape(g.Nzs, g.Nx)
    gmax = float(Gs.max()) if Gs.size else 0.0
    npulse = Gs / gmax if gmax > 0 else np.zeros_like(Gs)
    E = np.hypot(Ex, Ez)
    f = np.asarray(sol.f, float).reshape(g.Nzs, g.Nx) if sol.f is not None else np.full((g.Nzs, g.Nx), np.nan)
    data = dict(V=V, Ex=Ex, Ez=Ez, E=E, logn=np.log10(np.maximum(n, 1.0)), logp=np.log10(np.maximum(p, 1.0)),
                logT=np.log10(np.maximum(NTp, 1.0)), NTp=NTp, f=f, G=npulse, w=npulse * E[:ks], Px=npulse * Ex[:ks],
                Pz=npulse * Ez[:ks])
    return Loaded(path=str(path), name=display_name(path), mtime=os.path.getmtime(path), sol=sol, sim=sim,
                  data=data, raw=dict(n=n, p=p), electrodes=electrode_spans(sim))


def electrode_spans(sim) -> list:
    """x extent of the two electrodes exactly as the model has them: whole surface columns with |x_c| >= gap/2
    (on coarse grids the film starts up to a cell width outside ±gap/2)."""
    g, m = sim.grid, sim.top.electrode_mask
    out = []
    for side in (m & (g.xc < 0), m & (g.xc > 0)):
        idx = np.flatnonzero(side)
        if idx.size:
            out.append((float(g.xe[idx[0]]), float(g.xe[idx[-1] + 1])))
    return out


def cell_index(edges, v) -> int:
    """Index of the cell whose [edge_j, edge_j+1) contains v (clipped to the grid) -- what pcolormesh draws there."""
    return int(np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2))


class _Cache:
    """Loaded solutions shared by all tabs (and by the comparison worker)."""

    def __init__(self, size=24):
        self.d: dict[str, Loaded] = {}
        self.lock = threading.Lock()
        self.size = size

    def get(self, path):
        with self.lock:
            L = self.d.get(path)
        if L is not None:
            try:
                if os.path.getmtime(path) != L.mtime:
                    return None
            except OSError:
                return None
        return L

    def load(self, path):
        L = self.get(path)
        if L is None:
            L = load_solution(path)
            with self.lock:
                self.d.pop(path, None)
                self.d[path] = L
                while len(self.d) > self.size:
                    self.d.pop(next(iter(self.d)))
        return L


CACHE = _Cache()


def read_meta(path) -> dict | None:
    """Cheap per-file summary for the list (reads only a few npz members)."""
    try:
        with np.load(path, allow_pickle=False) as d:
            pj = json.loads(str(d["params_json"]))
            gen = float(d["gen_factor"])
            I = float(d["I_left_mA"])
        return dict(V=float(pj.get("V_bias", np.nan)), P=float(pj.get("P_avg_mW", np.nan)) * gen, I=I)
    except Exception:
        return None


def discover_solutions(extra) -> list[dict]:
    """All known solution files: committed coarse scenarios, GUI runs, other results, files opened by hand."""
    out, seen = [], set()

    def add(p, group):
        p = Path(p)
        try:
            rp = str(p.resolve())
            mtime = p.stat().st_mtime
        except OSError:
            return
        if rp in seen:
            return
        seen.add(rp)
        out.append(dict(path=rp, group=group, name=display_name(rp), mtime=mtime, meta=read_meta(rp)))

    for p in sorted(COARSE_DIR.glob("*.npz")):
        add(p, "coarse")
    gui = sorted(paths.GUI_RUNS.glob("**/solution.npz"), key=lambda q: q.stat().st_mtime, reverse=True) \
        if paths.GUI_RUNS.exists() else []
    for p in gui:
        add(p, "gui")
    for p in sorted(paths.RESULTS.glob("**/solution.npz")):
        add(p, "results")
    for p in extra:
        if Path(p).exists():
            add(p, "recent")
    return out


GROUPS = (("coarse", "Coarse-grid scenarios · results/coarse_scenarios"), ("gui", "GUI runs · results/gui_runs"),
          ("results", "Other results · results/**/solution.npz"), ("recent", "Opened manually"))

_MEASURED = None


def measured():
    """data/measured_tip_scan.csv as a structured array (None if missing)."""
    global _MEASURED
    if _MEASURED is None:
        try:
            rows = []
            with open(paths.MEASURED_CSV) as fh:
                for r in csv.DictReader(fh):
                    rows.append((float(r["bias_V"]), float(r["X_um"]), float(r["rms"]), float(r["rms_err"])))
            _MEASURED = np.array(rows, dtype=[("bias_V", float), ("X_um", float), ("rms", float), ("rms_err", float)])
        except (OSError, KeyError, ValueError):
            _MEASURED = np.zeros(0, dtype=[("bias_V", float), ("X_um", float), ("rms", float), ("rms_err", float)])
    return _MEASURED


def scan_metrics(X, S) -> dict:
    o = np.argsort(X)
    X, A = np.asarray(X, float)[o], np.abs(np.asarray(S, float)[o])
    m = A.max() if A.size else 0.0
    out = dict(peak=np.nan, s0=np.nan, ratio=np.nan)
    if not m > 0:
        return out
    out["peak"] = float(X[np.argmax(A)]) + 0.0       # no "-0.0"
    if X[0] <= 0 <= X[-1]:
        out["s0"] = float(np.interp(0.0, X, A) / m)
    if X[0] <= -5 and X[-1] >= 5:
        am5 = np.interp(-5.0, X, A)
        out["ratio"] = float(np.interp(5.0, X, A) / am5) if am5 > 0 else np.nan
    return out


def measured_deviation(X, S, V):
    """RMS of (model |S|/max - measured RMS) over the points at +-V (the -V data against the mirrored model)."""
    meas = measured()
    o = np.argsort(X)
    X, A = np.asarray(X, float)[o], np.abs(np.asarray(S, float)[o])
    if not A.size or not A.max() > 0:
        return np.nan, 0
    A = A / A.max()
    d = []
    for r in meas:
        if r["bias_V"] == V:
            xm = r["X_um"]
        elif r["bias_V"] == -V:
            xm = -r["X_um"]
        else:
            continue
        if X[0] <= xm <= X[-1]:
            d.append(np.interp(xm, X, A) - r["rms"])
    return (float(np.sqrt(np.mean(np.square(d)))) if d else np.nan), len(d)


# ---------------------------------------------------------------------------------------------- helpers
def colors(t: Tokens) -> dict:
    if t.dark:
        return dict(n="#74b9ff", p="#ff7675", T="#c39bff", w="#55d58e", anode="#ff8a7a", cath="#74b9ff")
    return dict(n="#1f77b4", p="#d62728", T="#7d3c98", w="#2e7d32", anode="#c0392b", cath="#2471a3")


def series_colors(t: Tokens) -> list:
    return [c["color"] for c in mpl_rc(t)["axes.prop_cycle"]]


def electrode_titles(sol):
    """(left text, right text, left is anode) for the ±x electrodes; left-is-anode is None without bias."""
    la = None if sol.V_left == sol.V_right else sol.V_left > sol.V_right
    name = {True: ("anode (+)", "cathode (−)"), False: ("cathode (−)", "anode (+)"), None: ("electrode", "electrode")}[la]
    lt = f"← {name[0]} {sol.V_left + 0.0:+.1f} V".replace("-", "−")
    rt = f"{name[1]} {sol.V_right + 0.0:+.1f} V →".replace("-", "−")
    return lt, rt, la


def electrode_title_axes(ax, sol, t):
    """Left / right electrode labels as the axes' side titles (anode in the 'anode' colour)."""
    lt, rt, la = electrode_titles(sol)
    c = colors(t)
    cl, cr = (t.muted, t.muted) if la is None else ((c["anode"], c["cath"]) if la else (c["cath"], c["anode"]))
    ax.set_title(lt, loc="left", fontsize=8.5, color=cl)
    ax.set_title(rt, loc="right", fontsize=8.5, color=cr)


def shade_electrodes(ax, L, x0, x1):
    for a, b in L.electrodes:
        lo, hi = max(a, x0), min(b, x1)
        if hi > lo:
            ax.axvspan(lo, hi, color=GOLD, alpha=0.13, lw=0, zorder=0)


def volts(v) -> str:
    return f"{round(v) + 0:+d} V".replace("-", "−") if abs(v) >= 0.5 else "0 V"


def plural(n, word) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def fnum(v, unit="") -> str:
    if v is None or not np.isfinite(v):
        return "—"
    a = abs(v)
    s = f"{v:.3e}" if (a >= 1e5 or 0 < a < 1e-3) else f"{v:.4g}"
    return f"{s} {unit}".rstrip()


def empty(fig, t, text):
    fig.text(0.5, 0.5, text, ha="center", va="center", color=t.muted, fontsize=11, wrap=True)


def _ask_save(parent, title, name, filt):
    s = QSettings(paths.APP_ID, paths.APP_ID)
    d = s.value("explorer/export_dir", str(paths.RESULTS)) or str(paths.RESULTS)
    path, _ = QFileDialog.getSaveFileName(parent, title, str(Path(d) / name), filt)
    if path:
        s.setValue("explorer/export_dir", str(Path(path).parent))
    return path or None


def export_figure(parent, draw, size, name, path=None):
    """Re-draw `draw(fig, tokens, interactive)` in the light theme on white and save it (readable in any app theme)."""
    path = path or _ask_save(parent, "Save figure", name, "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
    if not path:
        return None
    w, h = size
    with matplotlib.rc_context(mpl_rc(LIGHT)):
        fig = Figure(figsize=(max(w, 5.0), max(h, 3.0)), layout="constrained")
        fig.set_facecolor("white")
        draw(fig, LIGHT, False)
        fig.savefig(path, dpi=200, facecolor="white")
    return path


def write_csv(path, header: list[str], columns: list[tuple[str, np.ndarray]]):
    n = max((len(a) for _, a in columns), default=0)
    with open(path, "w", newline="") as fh:
        for h in header:
            fh.write(f"# {h}\n")
        w = csv.writer(fh)
        w.writerow([c for c, _ in columns])
        for i in range(n):
            row = []
            for _, a in columns:
                if i >= len(a):
                    row.append("")
                else:
                    v = a[i]
                    row.append(f"{v:.8g}" if isinstance(v, (float, np.floating)) else str(v))
            w.writerow(row)


def spin(lo, hi, val, step, decimals=2, suffix="", width=84):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(val)
    s.setKeyboardTracking(False)
    if suffix:
        s.setSuffix(suffix)
    s.setFixedWidth(width)
    return s


class FlowLayout(QLayout):
    """Left-to-right layout that wraps to a new line when the width runs out (control rows of narrow windows)."""

    def __init__(self, parent=None, hspacing=12, vspacing=6):
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
        return self._arrange(QRect(0, 0, w, 0), False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, True)

    def sizeHint(self):
        w = sum(it.sizeHint().width() for it in self._items) + self._h * max(len(self._items) - 1, 0)
        return QSize(w, max((it.sizeHint().height() for it in self._items), default=0))

    def minimumSize(self):
        s = QSize()
        for it in self._items:
            s = s.expandedTo(it.minimumSize())
        return s

    def _arrange(self, rect, apply):
        x0, right = rect.x(), rect.right() + 1
        lines, line, x = [], [], x0
        for it in self._items:
            if it.isEmpty():
                continue
            w = it.sizeHint().width()
            if line and x + w > right:
                lines.append(line)
                line, x = [], x0
            line.append(it)
            x += w + self._h
        if line:
            lines.append(line)
        y = rect.y()
        for line in lines:
            hh = max(it.sizeHint().height() for it in line)
            x = x0
            for it in line:
                s = it.sizeHint()
                if apply:
                    it.setGeometry(QRect(QPoint(x, y + (hh - s.height()) // 2), QSize(min(s.width(), right - x), s.height())))
                x += s.width() + self._h
            y += hh + self._v
        return max(y - self._v - rect.y(), 0)


def flow(*groups, hspacing=12) -> QWidget:
    """A wrapping row of control groups (each group is kept on one line)."""
    w = QWidget()
    fl = FlowLayout(w, hspacing=hspacing)
    for g in groups:
        fl.addWidget(g)
    return w


def hgroup(*items, spacing=5) -> QWidget:
    """Widgets (or label strings) side by side that must not be split across lines."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for it in items:
        lay.addWidget(label(it) if isinstance(it, str) else it)
    return w


class _FitHeight(QObject):
    """Keeps a word-wrapped label at least as tall as its wrapped text (splitter panes and grid rows otherwise
    squeeze it and the text is clipped or drawn under the neighbouring widget in narrow windows)."""

    def eventFilter(self, w, e):
        if e.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self.fit(w)
        return False

    @staticmethod
    def fit(w):
        h = w.heightForWidth(max(w.width(), 1))
        if h > 0 and h != w.minimumHeight():
            w.setMinimumHeight(h)


_FIT = None


def fit_height(*labels):
    global _FIT
    if _FIT is None:
        _FIT = _FitHeight()
    for l in labels:
        l.installEventFilter(_FIT)


def debounce(ms, slot, parent):
    t = QTimer(parent)
    t.setSingleShot(True)
    t.setInterval(ms)
    t.timeout.connect(slot)
    return t


# ---------------------------------------------------------------------------------------------- tabs
class _Tab(QWidget):
    def __init__(self, page):
        super().__init__()
        self.page = page
        self.L: Loaded | None = None
        self.dirty = True

    def set_loaded(self, L):
        self.L = L
        self.on_loaded()
        self.dirty = True
        if self.page.tabs.currentWidget() is self:
            self.refresh()

    def ensure(self):
        if self.dirty:
            self.refresh()

    def on_loaded(self):
        pass

    def refresh(self):
        self.dirty = False

    def empty_text(self):
        return self.page.empty_message()

    def save_figure(self, path=None):
        pass

    def export_csv(self, path=None):
        pass


class MapTab(_Tab):
    """2-D map of one quantity with a click-to-place crosshair and the two line cuts through it."""

    def __init__(self, page):
        super().__init__(page)
        self.cross = (0.0, -0.5)
        self._ax = None
        self._art = None
        self._drag = False
        self._streams: dict = {}           # streamline geometry per (solution, window): the integration is slow
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 8)
        lay.setSpacing(6)

        self.q = QComboBox()
        for q in QUANTITIES:
            self.q.addItem(q.name, q.key)
        self.q.setMinimumWidth(165)
        self.cmap = QComboBox()
        self.cmap.addItem("Auto", "")
        for c in DIVERGING + SEQUENTIAL:
            self.cmap.addItem(c, c)
        self.cmap.setToolTip("Auto: diverging RdBu_r for signed quantities, a sequential map for positive ones")
        self.sym = QCheckBox("Symmetric")
        self.sym.setChecked(True)
        self.sym.setToolTip("Center the color scale on 0 for signed quantities (V, E_x, E_z, P_x, P_z)")
        self.vmin = QLineEdit()
        self.vmax = QLineEdit()
        for e, tip in ((self.vmin, "Color-scale minimum (empty = auto: data range inside the view)"),
                       (self.vmax, "Color-scale maximum (empty = auto)")):
            e.setPlaceholderText("auto")
            e.setFixedWidth(72)
            e.setToolTip(tip)
        self.x0, self.x1 = spin(-1000, 1000, -20, 1, width=76), spin(-1000, 1000, 20, 1, width=76)
        self.z0, self.z1 = spin(-1000, 1000, -4, 0.5, width=76), spin(-1000, 1000, 1, 0.5, width=76)
        self.fieldmode = QComboBox()
        self.fieldmode.addItems(["None", "Streamlines", "Arrows"])
        self.fieldmode.setToolTip("Overlay electric-field streamlines (line width ∝ |E|) or direction arrows")
        self.aspect = QCheckBox("1:1 aspect")
        self.aspect.setToolTip("Same scale for x and z (by default z is stretched to show the surface layer)")
        reset = button("Reset view", self._reset_view, "flat")
        # ordered so that the groups pack into three rows even in the narrowest window
        lay.addWidget(flow(hgroup("Quantity", self.q), hgroup("Colors", self.cmap), self.sym,
                           hgroup("Range", self.vmin, "–", self.vmax), hgroup("Field lines", self.fieldmode),
                           self.aspect, hgroup("x", self.x0, "…", self.x1, "µm"),
                           hgroup("z", self.z0, "…", self.z1, "µm"), reset))

        self.map = PlotWidget(figsize=(7, 4))
        self.map.setMinimumHeight(150)
        self.map.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.map.canvas.setToolTip("Click or drag to place the crosshair; arrow keys move it by one cell (Shift+arrow: 5 cells)")
        c = self.map.canvas
        c.mpl_connect("button_press_event", self._on_press)
        c.mpl_connect("motion_notify_event", self._on_motion)
        c.mpl_connect("button_release_event", lambda *_: setattr(self, "_drag", False))
        c.mpl_connect("key_press_event", self._on_key)

        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 4, 0, 0)
        bl.setSpacing(4)
        self.readout = label("", wrap=True, selectable=True)
        self.readout.setTextFormat(Qt.TextFormat.RichText)
        fit_height(self.readout)
        bl.addWidget(self.readout)
        self.cuts = PlotWidget(toolbar=False, figsize=(7, 2.4))
        self.cuts.setMinimumHeight(110)
        bl.addWidget(self.cuts, 1)
        vs = QSplitter(Qt.Orientation.Vertical)
        vs.setChildrenCollapsible(False)
        vs.addWidget(self.map)
        vs.addWidget(bottom)
        vs.setSizes([520, 330])
        lay.addWidget(vs, 1)

        self._redraw = debounce(120, self.refresh, self)
        self._cross_timer = debounce(35, self._move_cross, self)
        self.q.currentIndexChanged.connect(self._on_quantity)
        for s in (self.cmap.currentIndexChanged, self.fieldmode.currentIndexChanged, self.sym.toggled,
                  self.aspect.toggled, self.x0.valueChanged, self.x1.valueChanged, self.z0.valueChanged,
                  self.z1.valueChanged, self.vmin.editingFinished, self.vmax.editingFinished):
            s.connect(lambda *_: self._redraw.start())
        self._on_quantity()

    # ---------------------------------------------------------------- state
    def quantity(self) -> Quantity:
        return QBY[self.q.currentData()]

    def set_quantity(self, key):
        self.q.setCurrentIndex(max(0, self.q.findData(key)))

    def window(self):
        x0, x1 = sorted((self.x0.value(), self.x1.value()))
        z0, z1 = sorted((self.z0.value(), self.z1.value()))
        if x1 - x0 < 1e-6:
            x1 = x0 + 1
        if z1 - z0 < 1e-6:
            z1 = z0 + 0.5
        return x0, x1, z0, z1

    def cell(self):
        """Grid indices (i, k) of the cell under the crosshair (snapped to GaAs for GaAs-only quantities).
        The containing cell, not the nearest centre: on graded grids those differ near cell edges."""
        g = self.L.g
        x, z = self.cross
        i, k = cell_index(g.xe, x), cell_index(g.ze, z)
        if self.quantity().semi:
            k = min(k, g.ks - 1)
        return i, k

    def _on_quantity(self, *_):
        q = self.quantity()
        self.sym.setEnabled(q.kind == "signed")
        self.vmin.clear()
        self.vmax.clear()
        self._redraw.start()

    def _reset_view(self):
        for s, v in ((self.x0, -20), (self.x1, 20), (self.z0, -4), (self.z1, 1)):
            s.blockSignals(True)
            s.setValue(v)
            s.blockSignals(False)
        self.vmin.clear()
        self.vmax.clear()
        self.refresh()

    def clim(self, q, x0, x1, z0, z1):
        g = self.L.g
        data = self.L.data[q.key]
        zc = g.zc[: data.shape[0]]
        sel = data[np.ix_((zc >= z0) & (zc <= z1), (g.xc >= x0) & (g.xc <= x1))]
        sel = sel[np.isfinite(sel)]
        if sel.size == 0:
            sel = data[np.isfinite(data)]
        lo, hi = (float(sel.min()), float(sel.max())) if sel.size else (0.0, 1.0)
        if q.kind == "signed" and self.sym.isChecked():
            m = max(abs(lo), abs(hi)) or 1.0
            lo, hi = -m, m
        for e, attr in ((self.vmin, "lo"), (self.vmax, "hi")):
            try:
                v = float(e.text().replace("−", "-"))
            except ValueError:
                continue
            if attr == "lo":
                lo = v
            else:
                hi = v
        if hi <= lo:
            hi = lo + (abs(lo) * 1e-6 or 1e-12)
        return lo, hi

    # ---------------------------------------------------------------- drawing
    def refresh(self):
        self.dirty = False
        self._ax = None
        self.map.plot(lambda fig: self.draw_map(fig, ThemeManager.instance().tokens, True))
        self._draw_cuts()

    def draw_map(self, fig, t, interactive):
        L = self.L
        if L is None:
            empty(fig, t, self.empty_text())
            return
        q, g, sol = self.quantity(), L.g, L.sol
        data = L.data[q.key]
        nz = data.shape[0]
        x0, x1, z0, z1 = self.window()
        ax = fig.add_subplot(111)
        lo, hi = self.clim(q, x0, x1, z0, z1)
        cmap = self.cmap.currentData() or q.cmap
        pc = ax.pcolormesh(g.xe, g.ze[: nz + 1], data, cmap=cmap, vmin=lo, vmax=hi, shading="flat", rasterized=True)
        cb = fig.colorbar(pc, ax=ax, pad=0.015, fraction=0.045)
        cb.set_label(q.label)
        cb.outline.set_edgecolor(t.border)
        over = "#1b1f2a" if cmap in LIGHT_MID else "white"
        if q.semi and z1 > 0.15 * (z1 - z0):
            ax.text(x1 - 0.01 * (x1 - x0), z1 - 0.05 * (z1 - z0), "air: quantity defined in GaAs only", ha="right",
                    va="top", fontsize=8, color=t.muted)
        mode = self.fieldmode.currentIndex()
        if mode:
            self._field_overlay(ax, L, (x0, x1, z0, min(z1, 0.0) if q.semi else z1), over, mode)
        ax.axhline(0, color=over if not q.semi else t.muted, lw=0.7, alpha=0.6, zorder=5)
        for a, b in L.electrodes:
            ax.plot([a, b], [0, 0], color=GOLD, lw=5, solid_capstyle="butt", zorder=6)
        i, k = self.cell()
        xc, zc = g.xc[i], g.zc[k]
        st = [patheffects.withStroke(linewidth=2.6, foreground="#111111", alpha=0.55)]
        vl = ax.axvline(xc, color="white", lw=1.0, ls=(0, (4, 3)), path_effects=st, zorder=8)
        hl = ax.axhline(zc, color="white", lw=1.0, ls=(0, (4, 3)), path_effects=st, zorder=8)
        pt, = ax.plot([xc], [zc], "o", ms=7, mfc="none", mec="white", mew=1.5, path_effects=st, zorder=9)
        ax.set_xlim(x0, x1)
        ax.set_ylim(z0, z1)
        if self.aspect.isChecked():
            ax.set_aspect("equal")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("z (µm)")
        electrode_title_axes(ax, sol, t)
        fig.suptitle(f"{q.label}   ·   {L.name}", fontsize=10)      # own line: no clash with the electrode labels
        if interactive:
            self._ax = ax
            self._art = (vl, hl, pt)
            ax.format_coord = self._format_coord

    def _field_overlay(self, ax, L, win, color, mode):
        from scipy.interpolate import RegularGridInterpolator
        g = L.g
        x0, x1, z0, z1 = win
        x0, x1 = max(x0, g.xc[0]), min(x1, g.xc[-1])
        z0, z1 = max(z0, g.zc[0]), min(z1, g.zc[-1])
        if x1 <= x0 or z1 <= z0:
            return
        nx, nz = (150, 90) if mode == 1 else (34, 14)
        xu, zu = np.linspace(x0, x1, nx), np.linspace(z0, z1, nz)
        Zg, Xg = np.meshgrid(zu, xu, indexing="ij")
        pts = np.column_stack([Zg.ravel(), Xg.ravel()])
        U = RegularGridInterpolator((g.zc, g.xc), L.data["Ex"], bounds_error=False, fill_value=None)(pts).reshape(Zg.shape)
        W = RegularGridInterpolator((g.zc, g.xc), L.data["Ez"], bounds_error=False, fill_value=None)(pts).reshape(Zg.shape)
        sp = np.hypot(U, W)
        smax = float(sp.max()) or 1.0
        if mode == 1:
            self._streamlines(ax, (L.path, L.mtime, x0, x1, z0, z1), color,
                              lambda: ax.streamplot(xu, zu, U, W, density=(1.8, 1.1), color=color,
                                                    linewidth=0.35 + 1.5 * sp / smax, arrowsize=0.7, zorder=4))
            return
        # arrows: equal on-screen length (direction as drawn), length grows with sqrt(|E|)
        bb = ax.get_position()
        fw, fh = ax.figure.get_size_inches()
        sx = bb.width * fw * 0.85 / (x1 - x0)
        sz = sx if self.aspect.isChecked() else bb.height * fh * 0.85 / (z1 - z0)
        ell = 0.8 * min((x1 - x0) / nx * sx, (z1 - z0) / nz * sz) * (0.3 + 0.7 * np.sqrt(sp / smax))
        Ld = np.hypot(U * sx, W * sz)
        Ld[Ld == 0] = np.inf
        ax.quiver(Xg, Zg, U * ell / Ld, W * ell / Ld, angles="xy", scale_units="xy", scale=1, pivot="mid",
                  color=color, width=0.0022, headwidth=4, headlength=4.5, headaxislength=4, alpha=0.9, zorder=4)

    def _streamlines(self, ax, key, color, compute):
        """Streamplot integration costs 0.2-0.4 s on the GUI thread: integrate once per (solution, window) and
        re-create the artists from the cached geometry on every other redraw (quantity, colours, theme, export)."""
        from matplotlib.collections import LineCollection
        from matplotlib.patches import FancyArrowPatch
        geo = self._streams.get(key)
        if geo is None:
            before = set(map(id, ax.patches))
            ss = compute()
            arrows = [p for p in ax.patches if id(p) not in before and isinstance(p, FancyArrowPatch)]
            for a in [ss.lines, *arrows]:
                a.set_in_layout(False)
            try:
                geo = (ss.lines.get_segments(), ss.lines.get_linewidths(),
                       [(tuple(p._posA_posB[0]), tuple(p._posA_posB[1]), p.get_linewidth()) for p in arrows])
            except (AttributeError, TypeError, IndexError):      # private matplotlib detail changed: no cache
                return
            if len(self._streams) >= 8:
                self._streams.pop(next(iter(self._streams)))
            self._streams[key] = geo
            return
        segs, lws, arrows = geo
        lc = LineCollection(segs, linewidths=lws, colors=color, zorder=4)
        lc.set_in_layout(False)
        ax.add_collection(lc, autolim=False)
        for a, b, lw in arrows:
            p = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7.0, color=color, linewidth=lw, zorder=4)
            p.set_in_layout(False)
            ax.add_artist(p)            # not add_patch: its data-limit update costs ~0.6 ms per arrow

    def _format_coord(self, x, z):
        L = self.L
        if L is None:
            return ""
        g = L.g
        q = self.quantity()
        data = L.data[q.key]
        i, k = cell_index(g.xe, x), cell_index(g.ze, z)
        inside = g.xe[0] <= x <= g.xe[-1] and g.ze[0] <= z <= g.ze[-1]
        v = data[k, i] if inside and k < data.shape[0] else np.nan
        return f"x = {x:+.3f} µm, z = {z:+.3f} µm   {q.label.split(' (')[0]} = {fnum(v)}"

    def _draw_cuts(self):
        self.cuts.plot(lambda fig: self.draw_cuts(fig, ThemeManager.instance().tokens, True))
        self._update_readout()

    def draw_cuts(self, fig, t, interactive):
        L = self.L
        if L is None:
            empty(fig, t, "Click the map above to place the crosshair; the x and z line cuts through it appear here")
            return
        q, g = self.quantity(), L.g
        data = L.data[q.key]
        i, k = self.cell()
        x0, x1, z0, z1 = self.window()
        ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.7, 1]})
        col = t.accent
        sx = (g.xc >= x0) & (g.xc <= x1)
        shade_electrodes(ax1, L, x0, x1)
        ax1.plot(g.xc[sx], data[k, sx], color=col, lw=1.4)
        ax1.axvline(g.xc[i], color=t.muted, lw=0.8, ls="--")
        ax1.plot([g.xc[i]], [data[k, i]], "o", color=col, ms=5)
        ax1.set_xlim(x0, x1)
        ax1.set_xlabel("x (µm)")
        ax1.set_ylabel(CUT_LABEL.get(q.key, q.label), fontsize=8.5)
        ax1.set_title(f"along x at z = {g.zc[k]:+.3f} µm", fontsize=9.5)
        zq = g.zc[: data.shape[0]]
        sz = (zq >= z0) & (zq <= z1)
        if z1 > 0:
            ax2.axvspan(max(0, z0), z1, color=t.muted, alpha=0.08, lw=0)
        ax2.axvline(0, color=t.muted, lw=0.6)
        if sz.any():
            ax2.plot(zq[sz], data[sz, i], color=col, lw=1.4)
        ax2.axvline(g.zc[k], color=t.muted, lw=0.8, ls="--")
        ax2.plot([g.zc[k]], [data[k, i]], "o", color=col, ms=5)
        ax2.set_xlim(z0, z1)
        ax2.set_xlabel("z (µm)   [GaAs < 0 < air]")
        ax2.set_title(f"along z at x = {g.xc[i]:+.3f} µm", fontsize=9.5)
        if q.kind == "signed":
            ax1.axhline(0, color=t.muted, lw=0.5)
            ax2.axhline(0, color=t.muted, lw=0.5)

    def _update_readout(self):
        L = self.L
        if L is None:
            self.readout.setText("")
            return
        t = ThemeManager.instance().tokens
        g, q = L.g, self.quantity()
        i, k = self.cell()
        d = L.data
        v = d[q.key][k, i]
        mu = f"color:{t.muted}"
        head = (f"<b>Crosshair</b> <span style='{mu}'>x =</span> {g.xc[i]:+.3f} µm, <span style='{mu}'>z =</span> "
                f"{g.zc[k]:+.3f} µm <span style='{mu}'>(cell i = {i}, k = {k}, {'GaAs' if k < g.ks else 'air'})</span>"
                f" &nbsp;→&nbsp; <b style='color:{t.accent}'>{q.label.split(' (')[0]} = {fnum(v)}</b>")
        if q.semi and self.cross[1] > 0:
            head += f" <span style='{mu}'>(defined in GaAs only: snapped to the top GaAs cell)</span>"
        vals = [("V", d["V"][k, i], "V"), ("E_x", d["Ex"][k, i], "kV/cm"), ("E_z", d["Ez"][k, i], "kV/cm"),
                ("|E|", d["E"][k, i], "kV/cm")]
        if k < g.ks:
            vals += [("n", L.raw["n"][k, i], "cm⁻³"), ("p", L.raw["p"][k, i], "cm⁻³"), ("EL2⁺", d["NTp"][k, i], "cm⁻³"),
                     ("f", d["f"][k, i], ""), ("G/G_max", d["G"][k, i], ""), ("n_pulse·|E|", d["w"][k, i], "kV/cm")]
        tail = " &nbsp;·&nbsp; ".join(f"<span style='{mu}'>{n} =</span> {fnum(x, u)}" for n, x, u in vals)
        self.readout.setText(f"<span style='font-size:12px'>{head}<br>{tail}</span>")
        _FitHeight.fit(self.readout)

    # ---------------------------------------------------------------- interaction
    def _toolbar_busy(self):
        tb = self.map.toolbar
        return bool(tb is not None and tb.mode)

    def set_cross(self, x, z):
        self.cross = (float(x), float(z))
        self._move_cross()

    def _move_cross(self):
        if self.L is None:
            return
        g = self.L.g
        i, k = self.cell()
        if self._ax is not None and self._art is not None:
            vl, hl, pt = self._art
            vl.set_xdata([g.xc[i], g.xc[i]])
            hl.set_ydata([g.zc[k], g.zc[k]])
            pt.set_data([g.xc[i]], [g.zc[k]])
            self.map.canvas.draw_idle()
        self._draw_cuts()

    def _on_press(self, ev):
        if ev.button != 1 or self._ax is None or ev.inaxes is not self._ax or self._toolbar_busy():
            return
        self._drag = True
        self.set_cross(ev.xdata, ev.ydata)

    def _on_motion(self, ev):
        if not self._drag or ev.inaxes is not self._ax or ev.xdata is None:
            return
        self.cross = (float(ev.xdata), float(ev.ydata))
        if not self._cross_timer.isActive():
            self._cross_timer.start()

    def _on_key(self, ev):
        if self.L is None or not ev.key:
            return
        key = ev.key.replace("shift+", "")
        step = 5 if ev.key.startswith("shift+") else 1
        if key not in ("left", "right", "up", "down"):
            return
        g = self.L.g
        i, k = self.cell()
        i += {"left": -step, "right": step}.get(key, 0)
        k += {"down": -step, "up": step}.get(key, 0)
        i, k = min(max(i, 0), g.Nx - 1), min(max(k, 0), g.Nz - 1)
        self.set_cross(g.xc[i], g.zc[k])

    # ---------------------------------------------------------------- export
    def save_figure(self, path=None, which=None):
        if self.L is None:
            return None
        if which is None and path is None:
            m = QMenu(self)
            a1 = m.addAction("2-D map…")
            a2 = m.addAction("Line cuts…")
            a = m.exec(QCursor.pos())
            which = "map" if a is a1 else "cuts" if a is a2 else None
            if which is None:
                return None
        which = which or "map"
        stem = Path(self.L.name).name.replace("/", "_")
        if which == "map":
            return export_figure(self, self.draw_map, self.map.figure.get_size_inches(),
                                 f"{stem}_{self.quantity().key}_map.png", path)
        return export_figure(self, self.draw_cuts, self.cuts.figure.get_size_inches(),
                             f"{stem}_{self.quantity().key}_cuts.png", path)

    def cut_data(self):
        L, q = self.L, self.quantity()
        g = L.g
        data = L.data[q.key]
        i, k = self.cell()
        x0, x1, z0, z1 = self.window()
        sx = (g.xc >= x0) & (g.xc <= x1)
        zq = g.zc[: data.shape[0]]
        sz = (zq >= z0) & (zq <= z1)
        return dict(i=i, k=k, x=g.xc[sx], vx=data[k, sx], z=zq[sz], vz=data[sz, i], xc=g.xc[i], zc=g.zc[k])

    def export_csv(self, path=None):
        if self.L is None:
            return None
        q = self.quantity()
        stem = Path(self.L.name).name.replace("/", "_")
        path = path or _ask_save(self, "Export line cuts as CSV", f"{stem}_{q.key}_cuts.csv", "CSV (*.csv)")
        if not path:
            return None
        c = self.cut_data()
        write_csv(path, [f"solution: {self.L.path}", f"quantity: {q.label}",
                         f"cut along x at z = {c['zc']:+.4f} um (cell k = {c['k']}); "
                         f"cut along z at x = {c['xc']:+.4f} um (cell i = {c['i']})",
                         "columns 1-2: along x; columns 3-4: along z (different lengths)"],
                  [("x_um", c["x"]), (f"{q.key}_along_x", c["vx"]), ("z_um", c["z"]), (f"{q.key}_along_z", c["vz"])])
        return path


def sci(v) -> str:
    """1.2e4 -> '1.2 × 10⁴' (plain notation between 0.01 and 1e4)."""
    if v == 0 or 1e-2 <= abs(v) < 1e4:
        return f"{v:.3g}"
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10 ** e
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"10{sup}" if abs(m - 1) < 5e-4 else f"{m:.3g} × 10{sup}"


class _DevicePage(QWidget):
    """The device structure (draw_device: two panels side by side) at a fitting aspect, its key dimensions below.
    draw_device sizes its annotations by the figure width at draw time: re-drawn when the width changes noticeably."""

    FACTS = (("box", "Simulated box"), ("pump", "Pump"), ("gap", "Electrode gap"), ("alpha", "Absorption"),
             ("bias", "Electrodes"), ("tip", "Tip"))

    def __init__(self):
        super().__init__()
        self.dev = None
        self._drawn_w = 0                 # canvas width the device was drawn at
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.plot = PlotWidget(figsize=(8, 3.6))
        lay.addWidget(self.plot)
        self.facts = QWidget()
        grid = QGridLayout(self.facts)
        grid.setContentsMargins(4, 0, 4, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        self.val = {}
        top = Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        for n, (k, txt) in enumerate(self.FACTS):
            r, c = divmod(n, 2)
            kl = label(txt, "muted")
            kl.setAlignment(top)
            vl = label("—", wrap=True, selectable=True)
            vl.setAlignment(top)
            vl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            grid.addWidget(kl, r, 2 * c, top)
            grid.addWidget(vl, r, 2 * c + 1)
            self.val[k] = vl
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        lay.addWidget(self.facts)
        lay.addStretch(1)
        self._rescale = debounce(150, lambda: self._render(only_if_resized=True), self)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()
        if self.dev is not None:
            self._rescale.start()

    def _fit(self):
        # panel (a) is drawn to scale (120 x 60 µm): a canvas much taller than ~0.45 x its width only adds empty space
        fh = self.facts.heightForWidth(self.width())          # wrapped labels: the height at this width
        free = self.height() - (fh if fh > 0 else self.facts.sizeHint().height()) - 8
        h = max(220, min(int(0.45 * self.width()) + 30, free))
        if h != self.plot.height():
            self.plot.setFixedHeight(h)

    def _render(self, only_if_resized=False):
        w = self.plot.canvas.width()
        if self.dev is None or (only_if_resized and abs(w - self._drawn_w) <= 0.06 * max(w, 1)):
            return
        self._drawn_w = w
        dev = self.dev
        self.plot.plot(lambda fig: draw_device(fig, dev))

    def show_message(self, msg):
        self.dev = None
        self.plot.plot(lambda fig: empty(fig, ThemeManager.instance().tokens, msg))
        for v in self.val.values():
            v.setText("—")

    def show_device(self, L, dev):
        """Draw Device `dev` of Loaded `L` and list its key dimensions."""
        self.dev = dev
        self._render()
        P, g = L.P, L.g
        a, b = dev.inner_edges
        mm = lambda v: f"{v + 0.0:+.4g}".replace("-", "−")
        xs = [-x for x in P.tip_x_scan_um]
        back = ("floating back surface" if P.back_boundary == "neumann"
                else f"back surface held at {mm(P.V_back)} V")
        self.val["box"].setText(f"{2 * P.x_half:g} × {P.depth + P.air_height:g} µm: GaAs {P.depth:g} µm + air "
                                f"{P.air_height:g} µm, {back}; grid {g.Nx} × {g.Nz}")
        self.val["gap"].setText(f"{b - a:.4g} µm between the electrode edges as discretized (nominal {P.gap:g} µm)")
        self.val["bias"].setText(f"Au films at {mm(L.sol.V_left)} V (left) and {mm(L.sol.V_right)} V (right); "
                                 f"anode at {P.anode_side.replace('-', '−')}")
        where = "centered on the gap" if P.spot_x0_um == 0 else f"centered at x = {mm(P.spot_x0_um)} µm"
        self.val["pump"].setText(f"{dev.P_mW:.3g} mW at {P.wavelength_nm:g} nm, normal incidence, w = "
                                 f"{P.spot_w_um:g} µm (1/e²) {where}; R = {P.reflectivity:g}")
        self.val["alpha"].setText(f"α = {sci(P.alpha_cm)} cm⁻¹, 1/α = {1e4 / P.alpha_cm:.3g} µm")
        h = f"E_z read {P.tip_height_um * 1000:.0f} nm above the surface"
        self.val["tip"].setText(f"{h}; scan X = {mm(min(xs))}…{mm(max(xs))} µm ({len(xs)} positions)" if xs else h)
        self._fit()


class FieldTab(_Tab):
    """COMSOL-style surface plot of the solved electric field (widgets.device.FieldView) and the device
    cross-section (draw_device) for the selected solution. The fields load lazily: when the tab is shown, or when
    the selection changes while it is visible."""

    def __init__(self, page):
        super().__init__(page)
        self._seq = 0
        self._key = None                  # (path, mtime) of the fields on display
        self._pending = None              # key of the load in flight
        self._cache: dict = {}            # (path, mtime) -> load_steady_fields() result
        self._data = None                 # the fields on display
        self._dev_key = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 8)
        lay.setSpacing(6)
        self.r_field = QRadioButton("Electric field")
        self.r_field.setToolTip("Surface plot of the solved field with arrows or streamlines, in COMSOL's style")
        self.r_dev = QRadioButton("Device structure")
        self.r_dev.setToolTip("Simulated cross-section to scale, with the gap region enlarged (from this solution's "
                              "parameters; electrode edges as discretized on its grid)")
        self.r_field.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.r_field)
        grp.addButton(self.r_dev)
        self.fv = FieldView()                 # its own controls (quantity, view, overlay, colors, log, aspect)
        self.dev_page = _DevicePage()
        self.dev = self.dev_page.plot
        self.status = label("", "muted", wrap=True, selectable=True)   # caption under the field map
        fit_height(self.status)
        lay.addWidget(flow(hgroup(self.r_field, self.r_dev, spacing=12)))
        self.stack = QStackedWidget()
        self.stack.addWidget(self.fv)
        self.stack.addWidget(self.dev_page)
        lay.addWidget(self.stack, 1)
        lay.addWidget(self.status)
        self.r_dev.toggled.connect(self._on_mode)
        self.fv.qty.currentIndexChanged.connect(self._on_quantity)

    # ---------------------------------------------------------------- state
    def _on_mode(self, *_):
        dev = self.r_dev.isChecked()
        for r in (self.r_field, self.r_dev):      # the checked indicator is wider (QSS border): buttons cache
            repolish(r)                           # their size hint until a style change, so force one
            r.updateGeometry()
        self.stack.setCurrentWidget(self.dev_page if dev else self.fv)
        self.refresh()

    def _on_quantity(self, *_):
        """Log scale applies to |E| only: FieldView ignores it for signed quantities, so grey it out there."""
        q = getattr(self.fv, "qtys", {}).get(self.fv.qty.currentText())
        self.fv.log.setEnabled(bool(q) and not q[2])

    def refresh(self):
        self.dirty = False
        if self.r_dev.isChecked():
            self._draw_device()
        else:
            self._load_fields()
        self._update_status()

    # ---------------------------------------------------------------- field map
    def _load_fields(self):
        L = self.L
        if L is None:
            self._seq += 1
            self._key = self._pending = self._data = None
            self.fv.set_fields(None, self.empty_text())
            return
        key = (L.path, L.mtime)
        if key == self._key or key == self._pending:
            return
        if key in self._cache:
            self._show(key, self._cache[key])
            return
        self._seq += 1
        seq = self._seq
        self._pending = key
        self._key = self._data = None
        self.fv.set_fields(None, f"Computing the electric field of {L.name}…")
        run_async(load_steady_fields, L.path, on_done=lambda d: self._done(seq, key, d),
                  on_error=lambda e: self._fail(seq, e))

    def _done(self, seq, key, d):
        if len(self._cache) >= 8:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = d
        if seq != self._seq:                  # a newer request superseded this one
            return
        self._pending = None
        L = self.L
        if L is not None and (L.path, L.mtime) == key and not self.r_dev.isChecked():
            self._show(key, d)
        else:                                 # selection or mode changed meanwhile: load again when shown
            self.dirty = True

    def _fail(self, seq, err):
        if seq != self._seq:
            return
        self._pending = None
        self.fv.set_fields(None, "Could not compute the field map:\n" + err.splitlines()[0])
        self._update_status()

    def _show(self, key, d):
        """Show new fields, keeping the quantity and view the user picked for the previous solution."""
        fv = self.fv
        q, v = fv.qty.currentText(), fv.view.currentText()
        fv.set_fields(d)
        changed = False
        for combo, txt in ((fv.qty, q), (fv.view, v)):
            i = combo.findText(txt) if txt else -1
            if i >= 0 and i != combo.currentIndex():
                combo.blockSignals(True)
                combo.setCurrentIndex(i)
                combo.blockSignals(False)
                changed = True
        if changed:                           # one redraw through FieldView's own slot
            fv.view.currentIndexChanged.emit(fv.view.currentIndex())
        self._key, self._data = key, d
        self._on_quantity()
        self._update_status()

    def _update_status(self):
        L, d = self.L, self._data
        self.status.setToolTip("")
        show = L is not None and d is not None and not self.r_dev.isChecked()
        self.status.setVisible(show)
        if not show:
            self.status.setText("")
        else:
            xe, ze = np.asarray(d["xe"]), np.asarray(d["ze"])
            E = np.hypot(d["Ex"], d["Ez"])
            k, i = np.unravel_index(int(np.nanargmax(E)), E.shape)
            x, z = 0.5 * (xe[i] + xe[i + 1]), 0.5 * (ze[k] + ze[k + 1])
            g0, g1 = d.get("gap", self._edges(L))
            edge = abs(z) < 1.0 and min(abs(x - g0), abs(x - g1)) < 2.5
            V = L.P.V_bias
            avg = abs(V) / (g1 - g0) * 10.0 if g1 > g0 else np.nan      # V/µm -> kV/cm
            s = lambda v: f"{v + 0.0:+.2f}".replace("-", "−")
            self.status.setText(f"max |E| = {E[k, i]:.3g} kV/cm at x = {s(x)} µm, z = {s(z)} µm"
                                f"{' (electrode edge)' if edge else ''} · bare-gap average {avg:.3g} kV/cm")
            self.status.setToolTip(f"Bare-gap average = |V_bias| / gap = {abs(V):g} V over {g1 - g0:.4g} µm (the gap "
                                   "between the electrode edges as discretized)")
        _FitHeight.fit(self.status)

    # ---------------------------------------------------------------- device
    @staticmethod
    def _edges(L):
        """Inner electrode edges as discretized (load_steady_fields uses the same columns)."""
        lefts = [b for a, b in L.electrodes if b <= 0]
        rights = [a for a, b in L.electrodes if a >= 0]
        return (max(lefts) if lefts else -L.P.gap / 2, min(rights) if rights else L.P.gap / 2)

    def device(self):
        L = self.L
        P = L.P
        # the pump power that was solved for is P_avg × gen_factor
        return device_from_params(P, electrode_edges=self._edges(L), P_mW=P.P_avg_mW * L.sol.gen_factor)

    def _draw_device(self):
        L = self.L
        if L is None:
            self._dev_key = None
            self.dev_page.show_message(self.empty_text())
            return
        key = (L.path, L.mtime)
        if key == self._dev_key:
            return
        self._dev_key = key
        self.dev_page.show_device(L, self.device())

    # ---------------------------------------------------------------- export
    def save_figure(self, path=None):
        if self.L is None:
            return None
        stem = Path(self.L.name).name.replace("/", "_")
        if self.r_dev.isChecked():
            w, name = self.dev, f"{stem}_device.png"
        elif self._data is not None:
            w, name = self.fv.plot, f"{stem}_field.png"
        else:
            return None
        path = path or _ask_save(self, "Save figure", name, "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not path:
            return None
        w.export(path)            # dark theme: re-drawn light on white
        return path

    def export_csv(self, path=None):
        d, L = self._data, self.L
        if d is None or L is None:
            return None
        stem = Path(L.name).name.replace("/", "_")
        path = path or _ask_save(self, "Export field as CSV", f"{stem}_field.csv", "CSV (*.csv)")
        if not path:
            return None
        xe, ze = np.asarray(d["xe"]), np.asarray(d["ze"])
        xc, zc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ze[1:] + ze[:-1])
        Z, X = np.meshgrid(zc, xc, indexing="ij")
        cols = [("x_um", X.ravel()), ("z_um", Z.ravel()), ("Ex_kVcm", np.ravel(d["Ex"])), ("Ez_kVcm", np.ravel(d["Ez"])),
                ("E_kVcm", np.hypot(d["Ex"], d["Ez"]).ravel())]
        if d.get("V") is not None:
            cols.append(("V_V", np.ravel(d["V"])))
        write_csv(path, [f"solution: {L.path}", f"grid cell centres, Nx = {len(xc)}, Nz = {len(zc)} (x varies fastest); "
                         "z < 0: GaAs, z > 0: air", "E in kV/cm, V in volts"], cols)
        return path


class ProfileTab(_Tab):
    """Interactive fig_profiles: E_x/E_z, n/p/EL2+ and the emission weight vs x at a chosen depth."""

    def __init__(self, page):
        super().__init__(page)
        self.req = 0.5
        self.depths = np.array([0.5])
        self.pr = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 8)
        lay.setSpacing(6)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setFixedWidth(220)
        self.slider.setToolTip("Step through the grid rows (cell centers) from the surface down")
        self.depth = spin(0, 1000, 0.5, 0.1, 3, " µm", 100)
        self.depth.setToolTip("Requested depth; the nearest GaAs cell center is used")
        self.actual = label("", "muted")
        self.xr = spin(1, 1000, 20, 1, 1, " µm", 90)
        lay.addWidget(flow(hgroup("Depth", self.slider, self.depth), self.actual, hgroup("x range ±", self.xr)))
        self.plot = PlotWidget(figsize=(7, 7))
        lay.addWidget(self.plot, 1)
        self.slider.valueChanged.connect(self._on_slider)
        self.depth.valueChanged.connect(self._on_spin)
        self.xr.valueChanged.connect(lambda *_: self.refresh())

    def on_loaded(self):
        if self.L is None:
            return
        g = self.L.g
        self.depths = -g.zc[: g.ks][::-1]          # row 0 = top GaAs cell
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(self.depths) - 1)
        self.slider.setValue(int(np.argmin(np.abs(self.depths - self.req))))
        self.slider.blockSignals(False)

    def _on_slider(self, r):
        self.req = float(self.depths[r])
        self.depth.blockSignals(True)
        self.depth.setValue(self.req)
        self.depth.blockSignals(False)
        self.refresh()

    def _on_spin(self, v):
        self.req = float(v)
        self.slider.blockSignals(True)
        self.slider.setValue(int(np.argmin(np.abs(self.depths - self.req))))
        self.slider.blockSignals(False)
        self.refresh()

    def set_depth(self, d):
        self.depth.setValue(d)

    def refresh(self):
        self.dirty = False
        if self.L is not None:
            from ltemsim.tip import surface_profiles
            self.pr = surface_profiles(self.L.sim, self.L.sol, self.req)
            # the row surface_profiles really used (a request half-way between two centres picks the deeper one)
            r = int(np.argmin(np.abs(self.depths - self.pr["depth"])))
            self.slider.blockSignals(True)
            self.slider.setValue(r)
            self.slider.blockSignals(False)
            self.actual.setText(f"→ used {self.pr['depth']:.3f} µm (cell row {r + 1} from the surface)")
        else:
            self.pr = None
            self.actual.setText("")
        self.plot.plot(lambda fig: self.draw(fig, ThemeManager.instance().tokens, True))

    def draw(self, fig, t, interactive):
        L, pr = self.L, self.pr
        if L is None or pr is None:
            empty(fig, t, self.empty_text())
            return
        c = colors(t)
        x = pr["x"]
        X = self.xr.value()
        s = (x >= -X) & (x <= X)
        axs = fig.subplots(3, 1, sharex=True)
        for ax in axs:
            shade_electrodes(ax, L, -X, X)
            ax.axvline(0, color=t.border, lw=0.8, zorder=0)
        ax = axs[0]
        ax.plot(x[s], pr["Ex"][s], color=t.text, lw=1.4, label="E_x")
        ax.plot(x[s], pr["Ez"][s], color=t.muted, lw=1.2, ls="--", label="E_z (+ up)")
        ax.axhline(0, color=t.muted, lw=0.5)
        ax.set_ylabel("field (kV/cm)")
        ax.legend(loc="best")
        electrode_title_axes(ax, L.sol, t)
        fig.suptitle(f"depth {pr['depth']:.3f} µm  ·  {L.name}", fontsize=10)
        ax = axs[1]
        ax.semilogy(x[s], np.maximum(pr["n"][s], 1), color=c["n"], lw=1.4, label="n")
        ax.semilogy(x[s], np.maximum(pr["p"][s], 1), color=c["p"], lw=1.4, label="p")
        ax.semilogy(x[s], np.maximum(pr["NTp"][s], 1), color=c["T"], lw=1.4, label="EL2⁺")
        ax.set_ylabel("density (cm⁻³)")
        ax.legend(loc="best")
        ax = axs[2]
        w = pr["w_int"]
        wm = np.abs(w).max() or 1.0
        wl = pr["w"]
        ax.plot(x[s], w[s] / wm, color=c["w"], lw=1.6, label="n_pulse·|E| (depth-integrated)")
        ax.plot(x[s], pr["wx_int"][s] / wm, color=t.text, lw=0.9, label="n_pulse·E_x (integrated)")
        ax.plot(x[s], pr["wz_int"][s] / wm, color=t.muted, lw=0.9, ls="--", label="n_pulse·E_z (integrated)")
        ax.plot(x[s], wl[s] / (np.abs(wl).max() or 1.0), color=c["w"], lw=1.0, ls=":",
                label=f"n_pulse·|E| at {pr['depth']:.2f} µm")
        ax.axhline(0, color=t.muted, lw=0.5)
        ax.set_ylabel("emission weight (norm.)")
        ax.set_xlabel("x (µm)   [spot centred at x = 0]")
        ax.set_xlim(-X, X)
        ax.legend(loc="best", fontsize=7.5)

    def save_figure(self, path=None):
        if self.L is None:
            return None
        stem = Path(self.L.name).name.replace("/", "_")
        return export_figure(self, self.draw, self.plot.figure.get_size_inches(),
                             f"{stem}_profiles_{self.pr['depth']:.2f}um.png", path)

    def export_csv(self, path=None):
        if self.L is None or self.pr is None:
            return None
        pr = self.pr
        stem = Path(self.L.name).name.replace("/", "_")
        path = path or _ask_save(self, "Export profiles as CSV", f"{stem}_profiles_{pr['depth']:.2f}um.csv", "CSV (*.csv)")
        if not path:
            return None
        write_csv(path, [f"solution: {self.L.path}", f"depth: {pr['depth']:.4f} um (nearest GaAs cell centre)",
                         "w_*_int: depth-integrated emission weights (arb.)"],
                  [("x_um", pr["x"]), ("Ex_kVcm", pr["Ex"]), ("Ez_kVcm", pr["Ez"]), ("n_cm3", pr["n"]),
                   ("p_cm3", pr["p"]), ("EL2plus_cm3", pr["NTp"]), ("f_neutral", pr["f"]),
                   ("w_local", pr["w"]), ("w_int", pr["w_int"]), ("wx_int", pr["wx_int"]), ("wz_int", pr["wz_int"])])
        return path


class TipTab(_Tab):
    """ltemsim.tip.tip_scan with adjustable height and range, compared with the measured RMS."""

    def __init__(self, page):
        super().__init__(page)
        self.ts = None
        self.ts_L = None
        self._seq = 0
        self._busy = False
        self._pending = False
        self._h_user = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 8)
        lay.setSpacing(8)
        self.h = spin(1, 20000, 50, 10, 0, " nm", 96)
        self.h.setToolTip("Height of the tip apex above the GaAs surface; E_z is read at the nearest air-cell center")
        self.h_actual = label("", "muted")
        self.sx0, self.sx1 = spin(-55, 55, -10, 0.5, 2, width=78), spin(-55, 55, 10, 0.5, 2, width=78)
        self.sdx = spin(0.01, 10, 0.5, 0.1, 2, width=70)
        self.meas = QCheckBox("Show measured RMS")
        self.meas.setChecked(True)
        self.meas.setToolTip(f"{paths.rel(paths.MEASURED_CSV)}: measured points at the model bias (and at the opposite bias, "
                             "against the mirrored curve)")
        self.status = label("", "muted", wrap=True)
        lay.addWidget(flow(hgroup("Tip height", self.h, self.h_actual), hgroup("X range", self.sx0, "…", self.sx1, "µm"),
                           hgroup("Step", self.sdx, "µm"), self.meas))
        lay.addWidget(self.status)
        grid = QGridLayout()
        grid.setSpacing(8)
        self.m_peak = MetricTile("Peak of |S| at X", "—", "measured at +25 V: +5 µm")
        self.m_s0 = MetricTile("S(0) / max", "—", "measured at +25 V: 0.24")
        self.m_ratio = MetricTile("|S(+5)| / |S(−5)|", "—", ">1: stronger on the X > 0 side")
        self.m_dev = MetricTile("RMS deviation from measured", "—", "")
        for i, m in enumerate((self.m_peak, self.m_s0, self.m_ratio, self.m_dev)):
            grid.addWidget(m, 0, i)
            fit_height(m.hint)
        fit_height(self.status)
        lay.addLayout(grid)
        self.plot = PlotWidget(figsize=(7, 4.2))
        lay.addWidget(self.plot, 1)
        self._timer = debounce(200, self._start, self)
        self.h.valueChanged.connect(self._on_h)
        for s in (self.sx0.valueChanged, self.sx1.valueChanged, self.sdx.valueChanged):
            s.connect(lambda *_: self._timer.start())
        self.meas.toggled.connect(lambda *_: self._replot())

    def _on_h(self, *_):
        self._h_user = True
        self._timer.start()

    def on_loaded(self):
        if self.L is not None and not self._h_user:
            self.h.blockSignals(True)
            self.h.setValue(self.L.P.tip_height_um * 1000)
            self.h.blockSignals(False)

    def x_scan(self):
        a, b = sorted((self.sx0.value(), self.sx1.value()))
        d = max(self.sdx.value(), 0.01)
        n = min(int(np.floor((b - a) / d + 1e-9)) + 1, 4001)
        return a + d * np.arange(max(n, 2)) if b > a else np.array([a - 1, a + 1])

    @staticmethod
    def snapped_height(L, h_um) -> float:
        """The height tip_scan really samples: the nearest air-cell centre (ltemsim.tip.apex_field)."""
        z = L.g.zc[L.g.ks:]
        return float(z[int(np.argmin(np.abs(z - h_um)))])

    def refresh(self):
        self.dirty = False
        self._start()

    def _start(self):
        if self.L is None:
            self.ts = None
            self._replot()
            return
        self.h_actual.setText(f"→ {self.snapped_height(self.L, self.h.value() / 1000) * 1000:.0f} nm")
        self.h_actual.setToolTip("E_z is read at the nearest air-cell center (the actual height after snapping to the grid)")
        if self._busy:
            self._pending = True
            return
        L = self.L
        X = self.x_scan()
        h_um = self.h.value() / 1000
        self._seq += 1
        seq = self._seq
        self._busy = True
        self.status.setText("Computing…")

        def work():
            from ltemsim.tip import tip_scan
            with L.lock:
                return tip_scan(L.sim, L.sol, x_tip=-X, h_um=h_um)

        run_async(work, on_done=lambda ts: self._done(seq, L, ts), on_error=lambda e: self._fail(seq, e))

    def _after(self):
        self._busy = False
        if self._pending:
            self._pending = False
            self._start()

    def _done(self, seq, L, ts):
        if seq == self._seq and L is self.L:
            self.ts, self.ts_L = ts, L
            self.status.setText(f"{len(ts.X_exp)} tip positions · height {ts.h_um * 1000:.0f} nm, snapped to the "
                                f"nearest air-cell center at {self.snapped_height(L, ts.h_um) * 1000:.0f} nm · "
                                "X = −x_tip (spot position relative to the tip)")
            self._update_metrics()
            for w in (self.status, self.m_peak.hint, self.m_s0.hint, self.m_ratio.hint, self.m_dev.hint):
                _FitHeight.fit(w)
            self._replot()
        self._after()

    def _fail(self, seq, err):
        if seq == self._seq:
            self.ts = None
            self.status.setText("Tip scan failed")
            self.plot.plot(lambda fig: empty(fig, ThemeManager.instance().tokens, "Tip scan failed:\n" + err.splitlines()[0]))
        self._after()

    def _update_metrics(self):
        ts, L = self.ts, self.ts_L
        m = scan_metrics(ts.X_exp, ts.S)
        self.m_peak.set(fnum(m["peak"]) + " µm" if np.isfinite(m["peak"]) else "—")
        self.m_s0.set(f"{m['s0']:.3f}" if np.isfinite(m["s0"]) else "—")
        la = electrode_titles(L.sol)[2]                       # X > 0 <=> tip at x < 0 (the -x electrode side)
        side = "the −x electrode side" if la is None else "the anode side" if la else "the cathode side"
        self.m_ratio.set(f"{m['ratio']:.2f}" if np.isfinite(m["ratio"]) else "—",
                         f">1: stronger with the tip on {side}" if np.isfinite(m["ratio"])
                         else "the scan range must cover X = ±5 µm")
        V = L.P.V_bias
        dev, n = measured_deviation(ts.X_exp, ts.S, V)
        self.m_dev.set(f"{dev:.3f}" if np.isfinite(dev) else "—",
                       (f"{n} measured points ({volts(-V)} vs. the mirrored model)" if V else f"{n} measured points at 0 V") if n
                       else f"no measured data at {'±' + volts(abs(V))[1:] if V else '0 V'}")
        self.metrics = m | dict(dev=dev, n_dev=n)

    def _replot(self):
        self.plot.plot(lambda fig: self.draw(fig, ThemeManager.instance().tokens, True))

    def draw(self, fig, t, interactive):
        ts, L = self.ts, self.ts_L
        if ts is None or L is None:
            empty(fig, t, self.empty_text() if self.L is None else "Computing the tip scan…")
            return
        c = colors(t)
        o = np.argsort(ts.X_exp)
        X, S, Sx, Sz = ts.X_exp[o], ts.S[o], ts.Sx[o], ts.Sz[o]
        m = np.abs(S).max() or 1.0
        V = L.P.V_bias
        h = self.snapped_height(L, ts.h_um) * 1000          # the height this scan was computed at
        ax1, ax2 = fig.subplots(1, 2)
        xl = "X = spot − tip (µm)" if fig.get_figwidth() < 7.5 else "X = spot position relative to tip (µm)"
        ax1.plot(X, S / m, color=t.text, lw=1.6, label="total E_z at apex")
        ax1.plot(X, Sx / m, color=c["n"], lw=1.0, label="from in-plane P_x")
        ax1.plot(X, Sz / m, color=c["p"], lw=1.0, ls="--", label="from vertical P_z")
        ax1.axhline(0, color=t.muted, lw=0.5)
        ax1.axvline(0, color=t.muted, lw=0.5)
        ax1.set_xlabel(xl)
        ax1.set_ylabel("signed tip signal (norm.)")
        ax1.set_title(f"model {volts(V)}  ·  h = {h:.0f} nm")
        ax1.legend(loc="best", fontsize=8)
        A = np.abs(S) / m
        ax2.plot(X, A, color=c["anode"], lw=1.6, label=f"model {volts(V)}")
        biases = [(V, c["anode"], "o")]
        if V:                                                  # 0 V: no opposite bias to mirror
            ax2.plot(-X, A, color=c["cath"], lw=1.4, ls="--", label=f"model {volts(-V)} (mirror)")
            biases.append((-V, c["cath"], "s"))
        meas = measured()
        if self.meas.isChecked() and len(meas):
            for Vb, col, mk in biases:
                mm = meas[meas["bias_V"] == Vb]
                if len(mm):
                    ax2.errorbar(mm["X_um"], mm["rms"], yerr=mm["rms_err"], fmt=mk, color=col, ms=6, capsize=3,
                                 mfc=t.surface, mew=1.4, label=f"measured {volts(Vb)} (RMS)")
        ax2.axvline(0, color=t.muted, lw=0.5)
        ax2.set_xlim(X.min(), X.max())
        ncols = 2 if fig.get_figwidth() >= 7.5 else 1
        rows = -(-len(ax2.get_legend_handles_labels()[1]) // ncols)
        # headroom for the legend above the data: ~12 pt per legend row of an axes ~70 % of the figure high
        free = 1 - min(0.6, rows * 12 / 72 / (0.7 * fig.get_figheight()) + 0.04)
        ax2.set_ylim(0, max(1.0, ax2.get_ylim()[1] / 1.05) * 1.02 / free)
        ax2.set_xlabel(xl)
        ax2.set_ylabel("|signal| (norm. to model max)")
        ax2.set_title("model vs measured")
        ax2.legend(loc="upper left", fontsize=7.5, ncols=ncols)

    def save_figure(self, path=None):
        if self.ts is None:
            return None
        stem = Path(self.ts_L.name).name.replace("/", "_")
        return export_figure(self, self.draw, self.plot.figure.get_size_inches(), f"{stem}_tip_scan.png", path)

    def export_csv(self, path=None):
        ts, L = self.ts, self.ts_L
        if ts is None:
            return None
        stem = Path(L.name).name.replace("/", "_")
        path = path or _ask_save(self, "Export tip scan as CSV", f"{stem}_tip_scan.csv", "CSV (*.csv)")
        if not path:
            return None
        o = np.argsort(ts.X_exp)
        m = np.abs(ts.S).max() or 1.0
        write_csv(path, [f"solution: {L.path}", f"V_bias = {L.P.V_bias:+g} V, tip height requested "
                         f"{ts.h_um * 1000:.0f} nm, used {self.snapped_height(L, ts.h_um) * 1000:.1f} nm (nearest air cell)",
                         "X = spot position relative to tip = -x_tip; S in arbitrary units"],
                  [("X_um", ts.X_exp[o]), ("x_tip_um", ts.x_tip[o]), ("S", ts.S[o]), ("S_from_Px", ts.Sx[o]),
                   ("S_from_Pz", ts.Sz[o]), ("abs_S_norm", np.abs(ts.S[o]) / m)])
        return path


class CompareTab(_Tab):
    """Overlay several solutions (interactive scripts/compare_scenarios.py)."""

    COLS = ["Scenario", "V_bias (V)", "P (mW)", "τ_srh (ns)", "S (cm/s)", "Contacts (cm⁻³)", "|I| (mA)", "Gain",
            "n(0) (cm⁻³)", "E_x(−5) (kV/cm)", "E_x(+5) (kV/cm)", "Peak X (µm)", "S(0)/max"]

    def __init__(self, page):
        super().__init__(page)
        self.slots: dict[str, int] = {}
        self.results: list[dict] = []
        self.rows: list[list] = []
        self._seq = 0
        self._busy = False
        self._pending = False
        self._fill = False
        self.used = (0.5, 0.05)          # (profile depth um, tip height um) the shown results were computed with
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 8)
        hs = QSplitter(Qt.Orientation.Horizontal)
        hs.setChildrenCollapsible(False)
        lay.addWidget(hs)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)
        ll.addWidget(label(f"Solutions to compare (≤ {MAX_COMPARE})", "cardTitle"))
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._apply_filter)
        ll.addWidget(self.filter)
        self.list = QListWidget()
        self.list.itemChanged.connect(self._on_item)
        ll.addWidget(self.list, 1)
        ll.addWidget(flow(button("Clear", self.clear_checks, "flat"),
                          button("Power series", self._preset_power, "flat", "Check coarse_P0.02 … coarse_P2.00"),
                          button("+ Current", self._add_current, "flat", "Check the solution that is open now"),
                          hspacing=0))
        left.setMinimumWidth(165)
        hs.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        self.depth = spin(0, 100, 0.5, 0.1, 3, " µm", 100)
        self.depth.setToolTip("Depth of the E_x and density profiles (each solution snaps to its nearest GaAs cell center)")
        self.h = spin(1, 20000, 50, 10, 0, " nm", 96)
        self.h.setToolTip("Tip height of the scans (snapped to the nearest air-cell center)")
        self.status = label("", "muted")
        rl.addWidget(flow(hgroup("Profile depth", self.depth), hgroup("Tip height", self.h), self.status))
        vs = QSplitter(Qt.Orientation.Vertical)
        vs.setChildrenCollapsible(False)
        self.plot = PlotWidget(figsize=(7, 5.5))
        vs.addWidget(self.plot)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.doubleClicked.connect(self._open_row)
        self.table.setToolTip("Double-click a row to open that solution in the 2-D map tab")
        vs.addWidget(self.table)
        vs.setSizes([560, 190])
        rl.addWidget(vs, 1)
        right.setMinimumWidth(300)
        hs.addWidget(right)
        hs.setStretchFactor(1, 1)
        hs.setSizes([185, 900])
        self._timer = debounce(250, self._start, self)
        self.depth.valueChanged.connect(lambda *_: self._timer.start())
        self.h.valueChanged.connect(lambda *_: self._timer.start())

    # ---------------------------------------------------------------- list
    def populate(self, entries):
        checked = set(self.checked())
        self._fill = True
        self.list.clear()
        t = ThemeManager.instance().tokens
        for e in entries:
            if not e["meta"]:               # not a Solution file: nothing to compare
                continue
            it = QListWidgetItem(e["name"])
            it.setData(Qt.ItemDataRole.UserRole, e["path"])
            it.setToolTip(paths.rel(e["path"]))
            it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
            it.setCheckState(Qt.CheckState.Checked if e["path"] in checked else Qt.CheckState.Unchecked)
            self.list.addItem(it)
        self._fill = False
        self._apply_filter()
        self._recolor(t)
        if not self.checked() and self.page.L is not None:
            self.set_checked([self.page.L.path])

    def _apply_filter(self, *_):
        s = self.filter.text().strip().lower()
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setHidden(bool(s) and s not in (it.text() + " " + it.toolTip()).lower())

    def _items(self):
        return [self.list.item(i) for i in range(self.list.count())]

    def checked(self) -> list[str]:
        return [it.data(Qt.ItemDataRole.UserRole) for it in self._items() if it.checkState() == Qt.CheckState.Checked]

    def set_checked(self, paths_):
        want = set(paths_)
        self._fill = True
        for it in self._items():
            p = it.data(Qt.ItemDataRole.UserRole)
            it.setCheckState(Qt.CheckState.Checked if p in want else Qt.CheckState.Unchecked)
        self._fill = False
        self._sync_slots()
        self._timer.start()

    def clear_checks(self):
        self.set_checked([])

    def _preset_power(self):
        ps = [it.data(Qt.ItemDataRole.UserRole) for it in self._items()
              if Path(it.data(Qt.ItemDataRole.UserRole)).stem.startswith("coarse_P")]
        self.set_checked(ps[:MAX_COMPARE])

    def _add_current(self):
        if self.page.L is not None:
            self.set_checked(self.checked() + [self.page.L.path])

    def _on_item(self, it):
        if self._fill:
            return
        if it.checkState() == Qt.CheckState.Checked and len(self.checked()) > MAX_COMPARE:
            self._fill = True
            it.setCheckState(Qt.CheckState.Unchecked)
            self._fill = False
            self.status.setText(f"At most {MAX_COMPARE} solutions can be compared at once")
            return
        self._sync_slots()
        self._timer.start()

    def _sync_slots(self):
        """Colour follows the solution: a slot is assigned when it is checked and kept while it stays checked."""
        ch = self.checked()
        for p in list(self.slots):
            if p not in ch:
                del self.slots[p]
        for p in ch:
            if p not in self.slots:
                used = set(self.slots.values())
                self.slots[p] = next(i for i in range(MAX_COMPARE + 1) if i not in used)
        self._recolor(ThemeManager.instance().tokens)

    def _recolor(self, t):
        cols = series_colors(t)
        self.list.blockSignals(True)        # setIcon emits itemChanged
        for it in self._items():
            p = it.data(Qt.ItemDataRole.UserRole)
            if p in self.slots:
                pm = QPixmap(12, 12)
                pm.fill(QColor(cols[self.slots[p] % len(cols)]))
                it.setIcon(QIcon(pm))
            else:
                it.setIcon(QIcon())
        self.list.blockSignals(False)

    def on_loaded(self):
        if self.L is not None and not self.checked():
            self.set_checked([self.L.path])

    # ---------------------------------------------------------------- compute
    def refresh(self):
        self.dirty = False
        self._start()

    def _start(self):
        sel = self.checked()
        if self._busy:
            self._pending = True
            return
        if not sel:
            self.results, self.rows = [], []
            self._show()
            return
        depth, h_um = self.depth.value(), self.h.value() / 1000
        self._seq += 1
        seq = self._seq
        self._busy = True
        self.status.setText(f"Computing {plural(len(sel), 'solution')}…")

        def work():
            from ltemsim.tip import surface_profiles, tip_scan
            out = [dict(used=(depth, h_um))]
            for p in sel:
                try:
                    L = CACHE.load(p)
                    with L.lock:
                        pr = surface_profiles(L.sim, L.sol, depth)
                        ts = tip_scan(L.sim, L.sol, h_um=h_um)
                    out.append(dict(path=p, L=L, pr=pr, ts=ts))
                except Exception as e:     # one unreadable file must not kill the comparison
                    out.append(dict(path=p, error=f"{type(e).__name__}: {e}"))
            return out

        run_async(work, on_done=lambda res: self._done(seq, res), on_error=lambda e: self._fail(seq, e))

    def _after(self):
        self._busy = False
        if self._pending:
            self._pending = False
            self._start()

    def _done(self, seq, res):
        if seq == self._seq:
            self.used = res.pop(0)["used"]
            self.results = [r for r in res if "error" not in r and r["path"] in self.slots]
            errs = [r for r in res if "error" in r]
            self.rows = [self._row(r) for r in self.results]
            self.status.setText(plural(len(self.results), "solution") + (f" · {len(errs)} unreadable" if errs else ""))
            if errs:
                self.status.setToolTip("\n".join(f"{paths.rel(r['path'])}: {r['error']}" for r in errs))
            self._show()
        self._after()

    def _fail(self, seq, err):
        if seq == self._seq:
            self.status.setText("Computation failed: " + err.splitlines()[0])
        self._after()

    def _row(self, r):
        L, pr, ts = r["L"], r["pr"], r["ts"]
        P = L.P
        x = pr["x"]
        ug = P.unity_gain_current_mA * L.sol.gen_factor
        I = abs(L.sol.I_left_mA)
        m = scan_metrics(ts.X_exp, ts.S)
        return [L.name, P.V_bias, P.P_avg_mW * L.sol.gen_factor, P.tau_srh_ns, P.S_surface, P.contact_density, I,
                I / ug if ug else np.nan, pr["n"][np.argmin(np.abs(x))], pr["Ex"][np.argmin(np.abs(x + 5))],
                pr["Ex"][np.argmin(np.abs(x - 5))], m["peak"], m["s0"]]

    def _show(self):
        self.plot.plot(lambda fig: self.draw(fig, ThemeManager.instance().tokens, True))
        self._fill_table()

    def _fill_table(self):
        fmts = [None, "{:+.0f}", "{:.3g}", "{:.3g}", "{:.3g}", "{:.3g}", "{:.4f}", "{:.3f}", "{:.3e}", "{:.3f}",
                "{:.3f}", "{:+.2f}", "{:.3f}"]
        cols = series_colors(ThemeManager.instance().tokens)
        self.table.setRowCount(len(self.rows))
        for i, (row, r) in enumerate(zip(self.rows, self.results)):
            for j, v in enumerate(row):
                txt = v if fmts[j] is None else (fmts[j].format(v) if np.isfinite(v) else "—")
                it = QTableWidgetItem(txt)
                if j == 0:
                    pm = QPixmap(12, 12)
                    pm.fill(QColor(cols[self.slots.get(r["path"], 0) % len(cols)]))
                    it.setIcon(QIcon(pm))
                    it.setData(Qt.ItemDataRole.UserRole, r["path"])
                    it.setToolTip(paths.rel(r["path"]))
                else:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, j, it)

    def _open_row(self, idx):
        it = self.table.item(idx.row(), 0)
        if it is not None:
            self.page.load(it.data(Qt.ItemDataRole.UserRole))
            self.page.tabs.setCurrentIndex(0)

    def draw(self, fig, t, interactive):
        if not self.results:
            empty(fig, t, f"Check the solutions to compare on the left (up to {MAX_COMPARE})")
            return
        cols = series_colors(t)
        axs = fig.subplots(2, 2)
        handles = []
        Vs = set()
        for r in self.results:
            L, pr, ts = r["L"], r["pr"], r["ts"]
            col = cols[self.slots.get(r["path"], 0) % len(cols)]
            x = pr["x"]
            ln, = axs[0, 0].plot(x, pr["Ex"], color=col, lw=1.4)
            axs[0, 1].semilogy(x, np.maximum(pr["n"], 1), color=col, lw=1.4)
            axs[0, 1].semilogy(x, np.maximum(pr["NTp"], 1), color=col, lw=1.2, ls="--")
            w = pr["w_int"]
            axs[1, 0].plot(x, w / (np.abs(w).max() or 1.0), color=col, lw=1.4)
            o = np.argsort(ts.X_exp)
            S = np.abs(ts.S[o])
            axs[1, 1].plot(ts.X_exp[o], S / (S.max() or 1.0), color=col, lw=1.4)
            ln.set_label(L.name)
            handles.append(ln)
            Vs.add(L.P.V_bias)
        meas = measured()
        mk = iter("osD^v")
        shown = []
        for V in sorted(Vs, reverse=True):
            mm = meas[meas["bias_V"] == V] if len(meas) else meas
            if len(mm):
                axs[1, 1].errorbar(mm["X_um"], mm["rms"], yerr=mm["rms_err"], fmt=next(mk, "o"), color=t.text, ms=5,
                                   capsize=3, mfc=t.surface, zorder=5)
                shown.append(volts(V))
        d = self.used[0]
        short = fig.get_figwidth() < 5.0
        a = axs[0, 0]
        a.axhline(0, color=t.muted, lw=0.5)
        a.set_xlim(-20, 20)
        a.set_ylabel("E_x (kV/cm)")
        a.set_title(f"E_x, ≈{d:.2f} µm deep" if short else f"E_x at ≈{d:.2f} µm depth", fontsize=9.5)
        a = axs[0, 1]
        a.set_xlim(-20, 20)
        a.set_ylabel("density (cm⁻³)")
        a.set_title("n (—), EL2⁺ (--)" if short else "n (solid), EL2⁺ (dashed)", fontsize=9.5)
        a = axs[1, 0]
        a.set_xlim(-12, 12)
        a.axhline(0, color=t.muted, lw=0.5)
        a.set_ylabel("norm.")
        a.set_title("emission weight" if short else "emission weight (depth-integrated)", fontsize=9.5)
        a = axs[1, 1]
        a.set_xlim(-10, 10)
        a.set_ylim(bottom=0)
        a.set_ylabel("norm.")
        a.set_xlabel("X = spot − tip (µm)")
        a.set_title(("|S|" if short else "|tip signal|") + ((" (○ meas.)" if short else f" (○ measured {', '.join(shown)})") if shown else ""),
                    fontsize=9.5)
        for a in axs[0]:
            a.set_xlabel("x (µm)")
        axs[1, 0].set_xlabel("x (µm)")
        # on screen the list and table swatches are the key; the legend only where it leaves room for the plots
        if not interactive or fig.get_figheight() * fig.dpi >= 420:
            ncols = max(1, min(len(handles), int(fig.get_figwidth() * 72 / 112)))
            fig.legend(handles=handles, loc="outside lower center", ncols=ncols, fontsize=8, frameon=False)

    def save_figure(self, path=None):
        if not self.results:
            return None
        return export_figure(self, self.draw, self.plot.figure.get_size_inches(), "compare_solutions.png", path)

    def export_csv(self, path=None):
        if not self.rows:
            return None
        path = path or _ask_save(self, "Export comparison table as CSV", "compare_solutions.csv", "CSV (*.csv)")
        if not path:
            return None
        keys = ["scenario", "V_bias_V", "P_mW", "tau_srh_ns", "S_surface_cms", "contact_density_cm3", "I_mA", "gain",
                "n0_cm3", "Ex_m5_kVcm", "Ex_p5_kVcm", "peak_X_um", "S0_over_max"]
        with open(path, "w", newline="") as fh:
            fh.write(f"# profiles at ~{self.used[0]:.3f} um depth (nearest cell per grid), tip height "
                     f"{self.used[1] * 1000:.0f} nm requested (nearest air cell per grid)\n")
            w = csv.writer(fh)
            w.writerow(keys + ["path"])
            for row, r in zip(self.rows, self.results):
                w.writerow([row[0]] + [f"{v:.6g}" for v in row[1:]] + [paths.rel(r["path"])])
        return path


# ---------------------------------------------------------------------------------------------- page
class ExplorerPage(Page):
    title = "Solution explorer"
    subtitle = "Interactive views of saved steady-state solutions (solution.npz)"

    def __init__(self):
        super().__init__()
        self.settings = QSettings(paths.APP_ID, paths.APP_ID)
        self.L: Loaded | None = None
        self.entries: list[dict] = []
        self.error: str | None = None
        self._load_seq = 0
        self._loading: str | None = None
        self._disc_seq = 0
        self._discovered = False

        self.btn_save = button("Save figure", self.save_figure,
                               tooltip="Save the current tab's figure (PNG / PDF / SVG, light theme on white)")
        self.btn_csv = button("Export CSV", self.export_csv,
                              tooltip="2-D map: both line cuts · Field: E and V on the grid · Profiles: every profile "
                                      "at that depth · Tip scan: S, Sx, Sz · Compare: summary table")
        self.btn_side = button("File list", self._toggle_side,
                               tooltip="Show or hide the solution list and info on the left (more room for plots)")
        self.btn_side.setCheckable(True)
        self.btn_side.setChecked(self.settings.value("explorer/side_visible", True, type=bool))
        self.header.add_action(self.btn_side)
        self.header.add_action(self.btn_save)
        self.header.add_action(self.btn_csv)

        self.split = split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        self.side = self._build_left()
        self.side.setVisible(self.btn_side.isChecked())
        self._sized = False
        split.addWidget(self.side)
        self.tabs = QTabWidget()
        self.map_tab = MapTab(self)
        self.prof_tab = ProfileTab(self)
        self.tip_tab = TipTab(self)
        self.field_tab = FieldTab(self)
        self.cmp_tab = CompareTab(self)
        for w, name in ((self.map_tab, "2-D map"), (self.field_tab, "Field (COMSOL style)"), (self.prof_tab, "Profiles"),
                        (self.tip_tab, "Tip scan"), (self.cmp_tab, "Compare")):
            self.tabs.addTab(w, name)
        self.tabs.currentChanged.connect(self._on_tab)
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([345, 1000])
        self.body.addWidget(split, 1)
        self._fill_info()
        self._retheme()
        ThemeManager.instance().changed.connect(self._retheme)

    # ------------------------------------------------------------------ left column
    def _build_left(self):
        vs = QSplitter(Qt.Orientation.Vertical)
        vs.setChildrenCollapsible(False)
        c1 = card()
        l1 = QVBoxLayout(c1)
        l1.setContentsMargins(12, 10, 12, 10)
        l1.setSpacing(6)
        row = QHBoxLayout()
        row.addWidget(label("Solutions", "cardTitle"), 1)
        self.count = label("", "muted")
        row.addWidget(self.count)
        self.btn_refresh = button("", self.discover, "flat", "Rescan results/ for solution files")
        row.addWidget(self.btn_refresh)
        l1.addLayout(row)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter by name or path")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._apply_filter)
        l1.addWidget(self.filter)
        self.list = QListWidget()
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list.currentItemChanged.connect(self._on_select)
        l1.addWidget(self.list, 1)
        row = QHBoxLayout()
        self.btn_open = button("Open…", self._open_file, tooltip="Open any .npz file written by Solution.save")
        row.addWidget(self.btn_open)
        row.addStretch(1)
        self.btn_reveal = button("Show in Finder", lambda: self.L and reveal(self.L.path), "flat")
        row.addWidget(self.btn_reveal)
        l1.addLayout(row)
        vs.addWidget(c1)

        c2 = card()
        l2 = QVBoxLayout(c2)
        l2.setContentsMargins(12, 10, 12, 10)
        l2.setSpacing(4)
        self.info_title = label("Nothing loaded", "cardTitle", wrap=True)
        self.info_path = label("", "muted", wrap=True, selectable=True)
        l2.addWidget(self.info_title)
        l2.addWidget(self.info_path)
        self.info_details = QWidget()
        dl = QVBoxLayout(self.info_details)
        dl.setContentsMargins(0, 4, 0, 0)
        dl.setSpacing(4)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        self.info = {}
        for r, (k, txt) in enumerate((("bias", "Bias"), ("anode", "Anode"), ("power", "Pump power"), ("spot", "Spot w"),
                                      ("grid", "Grid Nx×Nz"), ("cell", "Finest cell"), ("I", "Current"), ("gain", "Gain"),
                                      ("other", "Non-default"))):
            kl = label(txt, "muted")
            kl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            vl = label("—", wrap=True, selectable=True)
            vl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            grid.addWidget(kl, r, 0)
            grid.addWidget(vl, r, 1)
            self.info[k] = vl
        grid.setColumnStretch(1, 1)
        dl.addLayout(grid)
        dl.addWidget(label("Solver stages", "sectionHeader"))
        self.info_log = label("", wrap=True, selectable=True)
        self.info_log.setTextFormat(Qt.TextFormat.RichText)
        dl.addWidget(self.info_log)
        row = QHBoxLayout()
        self.btn_params = button("All parameters…", self._show_params, "flat", "Show the complete Params saved with this solution")
        row.addWidget(self.btn_params)
        row.addStretch(1)
        dl.addLayout(row)
        l2.addWidget(self.info_details)
        l2.addStretch(1)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.Shape.NoFrame)
        sa.setWidget(c2)
        vs.addWidget(sa)
        vs.setSizes([450, 380])
        vs.setMinimumWidth(260)
        return vs

    def _toggle_side(self, on):
        self.side.setVisible(on)
        self.settings.setValue("explorer/side_visible", bool(on))

    def heightForWidth(self, w):
        """The app's QScrollArea holder sizes a height-for-width page to its *preferred* height (header + the
        tallest tab's figure size hint, ~960 px), so the page overflowed the window and scrolled. Report the minimum
        instead: the page fills the window and only scrolls when the window is smaller than that."""
        lay = self.layout()
        return lay.minimumHeightForWidth(w) if lay is not None and lay.hasHeightForWidth() else -1

    def showEvent(self, e):
        super().showEvent(e)
        if not self._sized and self.split.width() > 100:     # left column ~30 % of the page, 260..345 px
            self._sized = True
            w = self.split.width()
            left = int(min(345, max(260, 0.3 * w)))
            self.split.setSizes([left, w - left])

    def _retheme(self, *_):
        t = ThemeManager.instance().tokens
        self.btn_refresh.setIcon(icons.icon("refresh", t.muted))
        self.btn_open.setIcon(icons.icon("open", t.text))
        self.btn_save.setIcon(icons.icon("image", t.text))
        self.btn_csv.setIcon(icons.icon("file", t.text))
        self.btn_side.setIcon(icons.icon("list", t.text))
        self._populate(keep=True)
        self._fill_info()
        self.cmp_tab._recolor(t)
        self.cmp_tab._fill_table()          # swatch colours; the plots re-render themselves (PlotWidget)
        self.map_tab._update_readout()

    # ------------------------------------------------------------------ discovery / list
    def _recent(self) -> list[str]:
        v = self.settings.value("explorer/recent", []) or []
        return [v] if isinstance(v, str) else list(v)

    def _remember(self, path):
        rec = [p for p in self._recent() if p != path]
        self.settings.setValue("explorer/recent", [path] + rec[:9])

    def discover(self):
        self._disc_seq += 1
        seq = self._disc_seq
        run_async(discover_solutions, self._recent(), on_done=lambda e: self._on_discovered(seq, e),
                  on_error=lambda err: self.count.setText("scan failed"))

    def _on_discovered(self, seq, entries):
        if seq != self._disc_seq:
            return
        changed = (not self._discovered or
                   [(e["path"], e["mtime"]) for e in entries] != [(e["path"], e["mtime"]) for e in self.entries])
        self.entries = entries
        self._discovered = True
        if changed:
            self._populate()
            self.cmp_tab.populate(entries)
        if self.L is None and self._loading is None and self.error is None:
            first = str(DEFAULT_SOLUTION.resolve()) if DEFAULT_SOLUTION.exists() else (
                entries[0]["path"] if entries else None)
            if first:
                self.load(first)
            else:
                self._fill_info()
                for tab in (self.map_tab, self.field_tab, self.prof_tab, self.tip_tab):
                    tab.set_loaded(None)

    def _populate(self, keep=False):
        t = ThemeManager.instance().tokens
        cur = self.L.path if self.L else None
        self.list.blockSignals(True)
        self.list.clear()
        bold = QFont()
        bold.setBold(True)
        bold.setPointSizeF(max(bold.pointSizeF() - 1, 9))
        for gid, gtitle in GROUPS:
            es = [e for e in self.entries if e["group"] == gid]
            if not es:
                continue
            h = QListWidgetItem(f"{gtitle}  ({len(es)})")
            h.setFlags(Qt.ItemFlag.NoItemFlags)
            h.setFont(bold)
            h.setForeground(QBrush(QColor(t.muted)))
            h.setData(Qt.ItemDataRole.UserRole + 1, gid)
            self.list.addItem(h)
            for e in es:
                m = e["meta"]
                sub = (f"{m['V']:+.0f} V · {m['P']:.3g} mW · |I| = {abs(m['I']):.3f} mA" if m
                       else "unreadable (not a Solution file?)")
                it = QListWidgetItem(f"{e['name']}\n    {sub}")
                it.setData(Qt.ItemDataRole.UserRole, e["path"])
                it.setData(Qt.ItemDataRole.UserRole + 1, gid)
                it.setToolTip(paths.rel(e["path"]))
                self.list.addItem(it)
                if e["path"] == cur:
                    self.list.setCurrentItem(it)
        if not self.entries and self._discovered:
            it = QListWidgetItem("No solution files found.\nRun a solve on the Steady state page\n"
                                 "(the coarse-grid mode takes ~30 s),\nor click Open… to choose an .npz file.")
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            it.setForeground(QBrush(QColor(t.muted)))
            self.list.addItem(it)
        self.list.blockSignals(False)
        self.count.setText(f"{len(self.entries)} found" if self._discovered else "")
        self._apply_filter()

    def _apply_filter(self, *_):
        s = self.filter.text().strip().lower()
        visible = {}
        for i in range(self.list.count()):
            it = self.list.item(i)
            p = it.data(Qt.ItemDataRole.UserRole)
            if p:
                hide = bool(s) and s not in (it.text() + " " + paths.rel(p)).lower()
                it.setHidden(hide)
                gid = it.data(Qt.ItemDataRole.UserRole + 1)
                visible[gid] = visible.get(gid, False) or not hide
        for i in range(self.list.count()):
            it = self.list.item(i)
            if not it.data(Qt.ItemDataRole.UserRole) and it.data(Qt.ItemDataRole.UserRole + 1):
                it.setHidden(not visible.get(it.data(Qt.ItemDataRole.UserRole + 1), False))

    def _select_in_list(self, path):
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == path:
                self.list.blockSignals(True)
                self.list.setCurrentItem(it)
                self.list.blockSignals(False)
                self.list.scrollToItem(it)
                return True
        return False

    def _on_select(self, cur, _prev):
        if cur is not None and cur.data(Qt.ItemDataRole.UserRole):
            self.load(cur.data(Qt.ItemDataRole.UserRole))

    def _open_file(self):
        start = str(Path(self.L.path).parent) if self.L else str(paths.RESULTS)
        p, _ = QFileDialog.getOpenFileName(self, "Open solution file", start, "Solution (*.npz);;All files (*)")
        if p:
            self.open(path=p)

    # ------------------------------------------------------------------ loading
    def load(self, path):
        p = str(paths.absolute(path).resolve())
        if self.L is not None and self.L.path == p and CACHE.get(p) is self.L:
            self._select_in_list(p)
            return
        self._load_seq += 1
        seq = self._load_seq
        self._loading = p
        self.error = None
        L = CACHE.get(p)
        if L is not None:
            self._on_loaded(seq, L)
            return
        self.info_title.setText("Loading…")
        self.info_path.setText(paths.rel(p))
        run_async(CACHE.load, p, on_done=lambda L, s=seq: self._on_loaded(s, L),
                  on_error=lambda e, s=seq, q=p: self._on_load_error(s, q, e))

    def _on_loaded(self, seq, L):
        if seq != self._load_seq:
            return
        self._loading = None
        self.L = L
        self.error = None
        self._fill_info()
        for tab in (self.map_tab, self.field_tab, self.prof_tab, self.tip_tab, self.cmp_tab):
            tab.set_loaded(L)
        if not self._select_in_list(L.path) and self._discovered:
            self.discover()

    def _on_load_error(self, seq, path, err):
        if seq != self._load_seq:
            return
        self._loading = None
        self.L = None
        self.error = f"Could not load {paths.rel(path)}:\n{err.splitlines()[0]}"
        self._fill_info()
        for tab in (self.map_tab, self.field_tab, self.prof_tab, self.tip_tab):
            tab.set_loaded(None)

    def empty_message(self):
        if self.error:
            return self.error
        if self._loading:
            return "Loading…"
        if self._discovered and not self.entries:
            return ("No solution files found. Run a solve on the Steady state page (the coarse-grid mode takes ~30 s), "
                    "or click Open… on the left to choose an .npz file.")
        return "Select a solution file in the list on the left"

    def _fill_info(self):
        t = ThemeManager.instance().tokens
        L = self.L
        has = L is not None
        for b in (self.btn_reveal, self.btn_params, self.btn_save, self.btn_csv):
            b.setEnabled(has)
        self.info_details.setVisible(has)
        if not has:
            self.info_title.setText("Load failed" if self.error else "Nothing loaded")
            self.info_path.setText(self.error or "Select a solution in the list above, or click Open…")
            for v in self.info.values():
                v.setText("—")
            self.info_log.setText("")
            return
        sol, P, g = L.sol, L.P, L.g
        self.info_title.setText(L.name)
        self.info_path.setText(paths.rel(L.path))
        gen = sol.gen_factor
        ug = P.unity_gain_current_mA * gen
        I = abs(sol.I_left_mA)
        m = lambda v, f: format(v, f).replace("-", "−")
        la = electrode_titles(sol)[2]
        side = ("none (both electrodes at the same potential)" if la is None else
                "−x (left) electrode" if la else "+x (right) electrode")
        self.info["bias"].setText(f"{m(P.V_bias, '+.4g')} V (left {m(sol.V_left, '+.2f')} V / right {m(sol.V_right, '+.2f')} V)")
        self.info["anode"].setText(f"{side} · anode_side = {P.anode_side}")
        self.info["power"].setText(f"{P.P_avg_mW * gen:.3g} mW = P_avg {P.P_avg_mW:.3g} × gen_factor {gen:.3g}")
        self.info["spot"].setText(f"{P.spot_w_um:.3g} µm (1/e² radius)")
        self.info["grid"].setText(f"{g.Nx} × {g.Nz} ({g.ks} GaAs rows)")
        self.info["cell"].setText(f"dx {g.dx.min():.3g} µm · dz {g.dz.min():.3g} µm")
        self.info["I"].setText(f"I_left {m(sol.I_left_mA, '+.4f')} · I_right {m(sol.I_right_mA, '+.4f')} mA")
        self.info["gain"].setText(f"{I / ug:.3f} (|I| / I_unity, I_unity = {ug:.3f} mA)" if ug else "—")
        from ltemsim import Params
        d0 = Params()
        diff = [f"{k} = {getattr(P, k)}" for k in P.__dataclass_fields__
                if k not in _NOT_LISTED and getattr(P, k) != getattr(d0, k)]
        self.info["other"].setText(" · ".join(diff) if diff else "none (all other parameters at their defaults)")
        rows = []
        for st in sol.log or []:
            c = st.get("converged")
            mark = (f"<span style='color:{t.success}'>✓ converged</span>" if c else
                    f"<span style='color:{t.danger}'>✗ not converged</span>" if c is not None else
                    f"<span style='color:{t.muted}'>—</span>")
            bits = [f"{st.get('steps')} steps" if st.get("steps") is not None else "",
                    f"t_sim = {st['t_sim']:.3g}" if isinstance(st.get("t_sim"), (int, float)) else "",
                    f"wall time {st['t_wall']:.1f} s" if isinstance(st.get("t_wall"), (int, float)) else ""]
            rows.append(f"{mark} &nbsp;<b>{st.get('stage', '?')}</b><br><span style='color:{t.muted}'>"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;{' · '.join(b for b in bits if b)}</span>")
        self.info_log.setText("<br>".join(rows) if rows else f"<span style='color:{t.muted}'>no log</span>")

    def _show_params(self):
        if self.L is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Parameters — {self.L.name}")
        dlg.resize(560, 640)
        lay = QVBoxLayout(dlg)
        ed = QPlainTextEdit(self.L.P.to_json())
        ed.setObjectName("log")
        ed.setReadOnly(True)
        lay.addWidget(ed)
        lay.addWidget(button("Close", dlg.accept))
        dlg.exec()

    # ------------------------------------------------------------------ tabs / export
    def _on_tab(self, i):
        w = self.tabs.widget(i)
        if isinstance(w, _Tab):
            w.ensure()

    def save_figure(self):
        return self.tabs.currentWidget().save_figure()

    def export_csv(self):
        return self.tabs.currentWidget().export_csv()

    # ------------------------------------------------------------------ page API
    def on_show(self):
        self.discover()

    def open(self, path=None, **_):
        if not path:
            return
        p = str(paths.absolute(path).resolve())
        if not any(e["path"] == p for e in self.entries) and not p.startswith(str(paths.RESULTS.resolve())):
            self._remember(p)
        self.load(p)
