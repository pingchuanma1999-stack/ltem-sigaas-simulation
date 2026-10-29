"""One-parameter sweeps of the steady-state model (the scripts/sweep.py workflow, run through parallel.sweep:
the values are solved at the same time on separate cores)."""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                               QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from .. import icons, nav, paths
from ..jobs import FINISHED, JobManager
from ..paramspec import FieldSpec, format_float, params_specs
from ..theme import ThemeManager
from ..widgets.common import MetricTile, Page, button, card, error_box, label, reveal
from ..widgets.cores import CoresSpin, estimate_s, slowdown
from ..widgets.imageview import ImageView
from ..widgets.mpl import PlotWidget
from ..widgets.paramform import ParamForm
from ..widgets.runpanel import RunPanel, fmt_elapsed
from .measured import read_measured, series_colors
from .steady import steady_metrics

# legacy Params fields that no code reads, and rep_rate (pulse model only)
HIDDEN = {"gummel_max_iter", "gummel_tol", "poisson_max_iter", "poisson_tol", "generation_ramp", "bias_ramp_steps",
          "rep_rate"}
QUICK = {"dx_fine": 0.5, "x_fine": 8.0, "dz_fine": 0.25}      # coarse grid of results/coarse_scenarios
MIN_PER_RUN = {False: 6.5, True: 0.55}                       # minutes per solve on one core: production / coarse
# copy of scripts/sweep.py SWEEPS (importing it would pull pyplot into the GUI process)
PRESETS = (
    ("power", "P_avg_mW", (0.5, 1.0, 2.0, 4.0, 8.0), "Pump power"),
    ("el2", "N_T", (5e15, 1.5e16, 3e16), "EL2 density"),
    ("sigma_p", "sigma_p", (1e-18, 2e-18, 1e-17, 1e-16), "Hole capture cross-section"),
    ("sigma_n", "sigma_n", (1e-15, 1e-14, 1e-13), "Electron capture cross-section"),
    ("spot", "spot_w_um", (2.0, 3.0, 4.5), "Spot radius"),
    ("anode", "anode_side", ("-x", "+x"), "Anode side"),
)
PROGRESS_PATTERN = r"^@@PAR \d+/\d+ done"          # parallel.sweep prints one per finished value
_DONE_RE = re.compile(PROGRESS_PATTERN)
COLS = ("Value", "|I| (mA)", "Gain", "Peak X (µm)", "S(+5)/S(−5)", "S(0)/max", "Converged", "Time")
COL_TIPS = ("Value of the swept parameter", "Photocurrent |I_left|", "I / I_unity (unity-gain current)",
            "X where |S|(X) is largest (X = spot − tip)", "|S| at X = +5 µm divided by |S| at X = −5 µm",
            "|S|(0) / max |S|", "Whether both the dark and the illuminated solve converged",
            "Solve time for this value")
# value separators: ASCII comma / semicolon / whitespace, plus the full-width comma and semicolon and the
# ideographic comma (U+FF0C, U+FF1B, U+3001) that a Chinese input method types when it is left switched on
_SEP_RE = re.compile(r"[,;\s\uff0c\uff1b\u3001]+")
# ordinal single-hue ramps (blue), low → high value; the dark one runs toward the light end
RAMP = {False: ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"),
        True: ("#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb")}


# ----------------------------------------------------------------------------- helpers
def pretty_unit(u: str) -> str:
    for a, b in (("cm^2/Vs", "cm²/Vs"), ("cm^3/s", "cm³/s"), ("cm^-3", "cm⁻³"), ("cm^-1", "cm⁻¹"), ("cm^2", "cm²"),
                 ("um", "µm")):
        u = u.replace(a, b)
    return u


def fmt_x(v: float) -> str:
    """Signed position with a real minus sign and no '−0.0'."""
    r = round(v, 1) + 0.0
    return "0.0" if r == 0 else f"{r:+.1f}".replace("-", "−")


def fmt_value(v) -> str:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    return format_float(float(v))


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def fmt_dur(s: float) -> str:
    """Short runtime estimate: '45 s', '7.8 min', '33 min', '1.6 h'."""
    if s < 90:
        return f"{s:.0f} s"
    m = s / 60
    if m < 10:
        return f"{m:.1f} min"
    return f"{m:.0f} min" if m < 90 else f"{m / 60:.1f} h"


def parse_values(spec: FieldSpec, text: str) -> list:
    """Comma / space separated values typed for `spec`; raises ValueError with a short message."""
    toks = [t for t in _SEP_RE.split(text.strip()) if t]
    if not toks:
        raise ValueError("Enter at least one value")
    out = []
    for t in toks:
        try:
            if spec.type == "float":
                v = float(t)
                if not math.isfinite(v):
                    raise ValueError
            elif spec.type == "int":
                f = float(t)
                if not f.is_integer():
                    raise ValueError
                v = int(f)
            elif spec.type == "bool":
                if t.lower() in ("1", "true", "yes", "on"):
                    v = True
                elif t.lower() in ("0", "false", "no", "off"):
                    v = False
                else:
                    raise ValueError
            elif spec.type == "choice":
                if t not in spec.choices:
                    raise ValueError
                v = t
            else:
                v = t
        except ValueError:
            why = {"float": "is not a number", "int": "is not an integer", "bool": "is not True or False",
                   "choice": f"is not one of {', '.join(map(str, spec.choices or ()))}"}.get(spec.type,
                                                                                         "is not a valid value")
            raise ValueError(f"“{t}” {why}") from None
        lo, hi = spec.minimum, spec.maximum
        if _is_num(v) and (lo is not None and v < lo or hi is not None and v > hi):
            rng = (f"{format_float(lo)} … {format_float(hi)}" if lo is not None and hi is not None
                   else f"≥ {format_float(lo)}" if lo is not None else f"≤ {format_float(hi)}")
            raise ValueError(f"{t} is outside the allowed range ({rng})")
        out.append(v)
    labels = [f"{spec.name}={v}" for v in out]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate values (they would be written to the same folder)")
    return out


def generate(a: float, b: float, n: int, log: bool) -> list[float]:
    if log:
        if a <= 0 or b <= 0:
            raise ValueError("log spacing needs both ends > 0")
        vals = np.geomspace(a, b, n)
    else:
        vals = np.linspace(a, b, n)
    return [float(f"{v:.6g}") for v in vals]


def run_label(key: str, v) -> str:
    """Folder name scripts/sweep.py uses for one value."""
    return f"{key}={v}".replace("/", "_")


def ramp_colors(n: int, dark: bool) -> list[str]:
    stops = [QColor(c) for c in RAMP[bool(dark)]]
    if n == 1:
        return [RAMP[bool(dark)][len(stops) // 2]]
    out = []
    for i in range(n):
        f = i / (n - 1) * (len(stops) - 1)
        k = min(int(f), len(stops) - 2)
        u = f - k
        a, b = stops[k], stops[k + 1]
        out.append(QColor(round(a.red() + (b.red() - a.red()) * u), round(a.green() + (b.green() - a.green()) * u),
                          round(a.blue() + (b.blue() - a.blue()) * u)).name())
    return out


def is_sweep_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    if any((p / f).exists() for f in ("compare.png", "sweep_summary.json", "sweep_spec.json")):
        return True
    return any(c.is_dir() and "=" in c.name and (c / "summary.json").exists() for c in p.iterdir())


def find_sweep_dir(path) -> tuple[Path | None, str | None]:
    """(sweep folder, run label to select) for a sweep folder, its parent (a run root) or one of its run folders."""
    p = Path(path)
    if p.is_file():
        p = p.parent
    if not p.exists():
        return None, None
    if is_sweep_dir(p):
        return p, None
    if "=" in p.name and is_sweep_dir(p.parent):            # a run folder (also an interrupted one)
        return p.parent, p.name
    kids = [c for c in sorted(p.iterdir()) if is_sweep_dir(c)] if p.is_dir() else []
    return (kids[0], None) if kids else (None, None)


def _same_base(params: dict, key: str, base: dict) -> bool:
    """True when a run's params (base overrides + swept key) were solved on top of `base`."""
    rest = {k: v for k, v in params.items() if k != key}
    if rest.keys() != base.keys():
        return False
    for k, v in rest.items():
        b = base[k]
        if _is_num(v) and _is_num(b):
            if not math.isclose(v, b, rel_tol=1e-12, abs_tol=0.0):
                return False
        elif v != b:
            return False
    return True


def load_sweep(d: Path) -> dict:
    """Everything the results tab shows for one sweep folder (<root>/<name>/).

    Run folders left over from an earlier sweep into the same folder with other base parameters
    (sweep_spec.json is rewritten by every GUI sweep) are listed in `foreign`, not mixed into `runs`.
    """
    d = Path(d)
    try:
        spec = json.loads((d / "sweep_spec.json").read_text())
    except (OSError, ValueError):
        spec = {}
    runs, foreign = [], []
    for s in d.glob("*/summary.json"):
        lab = s.parent.name
        if "=" not in lab:
            continue
        try:
            summ = json.loads(s.read_text())
        except (OSError, ValueError):
            continue
        key, raw = lab.split("=", 1)
        val = summ.get("params", {}).get(key, raw)
        if isinstance(val, str) and key != "anode_side":
            try:
                val = float(val)
            except ValueError:
                pass
        runs.append(dict(label=lab, dir=s.parent, key=key, value=val, summary=summ, metrics=steady_metrics(summ),
                         done=True))
    key = spec.get("key") or (runs[0]["key"] if runs else None)
    runs = [r for r in runs if r["key"] == key] if key else runs
    if key and isinstance(spec.get("base"), dict):
        def belongs(r):
            prm = r["summary"].get("params")
            return not isinstance(prm, dict) or key not in prm or _same_base(prm, key, spec["base"])
        foreign = sorted(r["label"] for r in runs if not belongs(r))
        runs = [r for r in runs if belongs(r)]
    expected = list(spec.get("values") or [])
    done = {r["label"] for r in runs}
    for v in expected:
        lab = run_label(key, v)
        if lab not in done:
            runs.append(dict(label=lab, dir=d / lab, key=key, value=v, summary={}, metrics={}, done=False,
                             started=(d / lab).exists()))
    if runs and all(_is_num(r["value"]) for r in runs):
        runs.sort(key=lambda r: r["value"])
    else:
        # the sweep's own order, else the field's choice order (anode_side: -x, +x), else by name
        choices = next((s.choices for s in params_specs() if s.name == key and s.choices), ()) if key else ()
        order = {run_label(key, v): i for i, v in enumerate(expected or choices)}
        runs.sort(key=lambda r: (order.get(r["label"], 1e9), r["label"]))
    return dict(dir=d, name=d.name, key=key, spec=spec, runs=runs, base=dict(spec.get("base") or {}), foreign=foreign,
                compare=d / "compare.png", n_done=sum(r["done"] for r in runs), n_total=len(runs))


def sweep_info(d: Path) -> dict:
    """Cheap description for the history list (no summary.json parsing)."""
    try:
        spec = json.loads((d / "sweep_spec.json").read_text())
    except (OSError, ValueError):
        spec = {}
    labels = [c.name for c in d.iterdir() if c.is_dir() and "=" in c.name]
    done = [l for l in labels if (d / l / "summary.json").exists()]
    key = spec.get("key") or (done[0].split("=", 1)[0] if done else (labels[0].split("=", 1)[0] if labels else "?"))
    if spec.get("values"):           # leftovers of an earlier sweep into the same folder are not counted
        done = [l for l in done if l in {run_label(key, v) for v in spec["values"]}]
    vals = spec.get("values") or [l.split("=", 1)[1] for l in done]
    if vals and all(_is_num(v) or _floatable(v) for v in vals):
        vals = sorted(float(v) for v in vals)
    base = spec.get("base") or {}
    return dict(key=key, values=vals, n_done=len(done), n_total=max(len(vals), len(done)),
                complete=(d / "compare.png").exists(), coarse=all(base.get(k) == v for k, v in QUICK.items()))


def _floatable(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def list_sweeps() -> list[Path]:
    """GUI runs (newest first), then the committed results/sweeps/*."""
    out = []
    for root in sorted(paths.GUI_RUNS.glob("sweep_*"), reverse=True):
        if root.is_dir():
            out += [c for c in sorted(root.iterdir()) if is_sweep_dir(c)]
    sw = paths.RESULTS / "sweeps"
    if sw.is_dir():
        out += [c for c in sorted(sw.iterdir()) if is_sweep_dir(c)]
    return out


def _swatch(color: str) -> QIcon:
    pm = QPixmap(12, 12)
    pm.fill(QColor(color))
    return QIcon(pm)


class _Reflow(QWidget):
    """Container that reports its width (when narrow the metric tiles collapse into one text line)."""

    def __init__(self, on_width):
        super().__init__()
        self._on_width = on_width

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._on_width(e.size().width())


# ----------------------------------------------------------------------------- page
class SweepPage(Page):
    title = "Parameter sweep"
    subtitle = ("One-parameter sweeps of the steady-state model (scripts/sweep.py): one full steady solve per value, "
                "several values at once on separate cores.")
    confirm_minutes = 20          # ask before queueing sweeps estimated to take longer than this

    def __init__(self):
        super().__init__()
        self.job = None
        self._follow = None           # sweep folder of the job launched here: results follow it until the user picks another
        self.sweep: dict | None = None
        self.highlight: int | None = None
        self._first_show = True
        all_specs = [s for s in params_specs() if s.name not in HIDDEN]
        self.specs = {s.name: s for s in all_specs}
        self._swept = None

        self._n = 0                   # number of valid values in the list
        self.est_head = label("", "muted")
        self.header.add_action(self.est_head)
        self.cores = CoresSpin()
        self.cores.changed.connect(self._update_estimate)
        self.header.add_action(self.cores)
        self.run_btn = button("  Run sweep", lambda: self.run(), "primary")     # not clicked(bool) → confirm
        self.header.add_action(self.run_btn)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        # ---------------- left: sweep definition + base parameters
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)
        box = card()
        g = QGridLayout(box)
        g.setContentsMargins(14, 12, 14, 12)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(7)
        g.setColumnStretch(1, 1)
        g.addWidget(label("Swept parameter", "cardTitle"), 0, 0, 1, 3)
        self.preset = QComboBox()
        for name, key, vals, desc in PRESETS:
            self.preset.addItem(f"{desc}  ·  {key}", name)
            unit = pretty_unit(self.specs[key].unit)
            self.preset.setItemData(self.preset.count() - 1,
                                    f"scripts/sweep.py preset “{name}”: {', '.join(fmt_value(v) for v in vals)}"
                                    + (f" {unit}" if unit else ""), Qt.ItemDataRole.ToolTipRole)
        self.preset.addItem("Custom (any parameter)", "custom")
        self.preset.currentIndexChanged.connect(self._on_preset)
        g.addWidget(QLabel("Preset"), 1, 0)
        g.addWidget(self.preset, 1, 1, 1, 2)
        self.key = QComboBox()
        self.key.setMaxVisibleItems(18)
        for s in all_specs:
            if s.type == "tuple":
                continue
            txt = s.label if s.name in s.label else f"{s.label}  ·  {s.name}"
            self.key.addItem(txt, s.name)
        self.key.currentIndexChanged.connect(self._on_key)
        for cb in (self.preset, self.key):       # long item texts must not set the column's minimum width
            cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(14)
        g.addWidget(QLabel("Parameter"), 2, 0)
        g.addWidget(self.key, 2, 1, 1, 2)
        self.values_edit = QLineEdit()
        self.values_edit.setPlaceholderText("Comma-separated, e.g. 0.5, 1, 2, 4")
        self.values_edit.textChanged.connect(self._update_preview)
        self.unit_lab = label("", "muted")
        g.addWidget(QLabel("Values"), 3, 0)
        g.addWidget(self.values_edit, 3, 1)
        g.addWidget(self.unit_lab, 3, 2)
        self.gen_box = QWidget()
        gl = QHBoxLayout(self.gen_box)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(5)
        self.gen_mode = QComboBox()
        self.gen_mode.addItems(["Linear", "Log"])
        self.gen_mode.setToolTip("Spacing of the generated values")
        self.gen_from = QLineEdit()
        self.gen_to = QLineEdit()
        for e, ph in ((self.gen_from, "from"), (self.gen_to, "to")):
            e.setPlaceholderText(ph)
            e.setMinimumWidth(48)
            e.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.gen_n = QSpinBox()
        self.gen_n.setRange(2, 40)
        self.gen_n.setValue(5)
        self.gen_n.setSuffix(" pts")
        self.gen_n.setToolTip("Number of values")
        gb = QPushButton("Fill")
        gb.setToolTip("Replace the value list above with evenly spaced values (linear or logarithmic)")
        gb.clicked.connect(self._generate)
        gl.addWidget(self.gen_mode)
        gl.addWidget(self.gen_from, 1)
        gl.addWidget(label("→", "muted"))
        gl.addWidget(self.gen_to, 1)
        gl.addWidget(self.gen_n)
        gl.addWidget(gb)
        g.addWidget(self.gen_box, 4, 0, 1, 3)        # no row label: keeps the card narrow enough for 1100 px windows
        self.preview = label("", "muted", wrap=True)
        self.preview.setTextFormat(Qt.TextFormat.RichText)
        g.addWidget(self.preview, 5, 0, 1, 3)
        self.quick = QCheckBox("Coarse grid (fast)")
        self.quick.setToolTip("dx_fine = 0.5 µm, x_fine = 8 µm, dz_fine = 0.25 µm (as in results/coarse_scenarios): "
                              "about 0.55 min per solve, good for a first look at trends")
        self.quick.toggled.connect(self._toggle_quick)
        self.per_run = label("", "muted")
        self.per_run.setToolTip("Time of one solve on one core")
        qrow = QHBoxLayout()
        qrow.addWidget(self.quick)
        qrow.addStretch(1)
        qrow.addWidget(self.per_run)
        g.addLayout(qrow, 6, 0, 1, 3)
        self.est = label("", "muted", wrap=True)
        g.addWidget(self.est, 7, 0, 1, 3)
        ll.addWidget(box)
        base_head = label("Base parameters · used for every solve", "sectionHeader")
        base_head.setToolTip("Every solve uses these values; the swept parameter is replaced by each value in turn")
        ll.addWidget(base_head)
        self.form = ParamForm(all_specs)
        self.form.changed.connect(self._on_form_changed)
        # the form scrolls by itself: its tall size hint must not make the whole page scroll in a small window
        self.form.setSizePolicy(self.form.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Ignored)
        self.form.setMinimumHeight(140)
        ll.addWidget(self.form, 1)
        orow = QHBoxLayout()
        orow.addWidget(QLabel("Output folder"))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Auto: results/gui_runs/sweep_<timestamp>")
        orow.addWidget(self.out_edit, 1)
        b = QPushButton("Choose…")
        b.clicked.connect(self._pick_out)
        orow.addWidget(b)
        ll.addLayout(orow)
        left.setMinimumWidth(380)          # ParamForm rows need ~362 px + scrollbar
        split.addWidget(left)

        # ---------------- right: run / results / history
        self.tabs = QTabWidget()
        run_tab = QWidget()
        rl = QVBoxLayout(run_tab)
        rl.setContentsMargins(12, 12, 12, 12)
        self.panel = RunPanel()
        self.panel.job_finished.connect(self._on_finished)
        rl.addWidget(self.panel)
        self.tabs.addTab(run_tab, "Run")
        self.tabs.addTab(self._build_results(), "Results")
        self.tabs.addTab(self._build_history(), "History")
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 900])
        self.body.addWidget(split, 1)

        self._on_preset(0)
        self._retheme()
        ThemeManager.instance().changed.connect(self._retheme)

    def _build_results(self):
        # scrolls instead of squashing the 2×2 figure when the window is small
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        w = _Reflow(self._flow_tiles)
        w.setMinimumHeight(700)          # fits the 1440×900 window; smaller windows scroll
        area.setWidget(w)
        xl = QVBoxLayout(w)
        xl.setContentsMargins(12, 12, 12, 12)
        xl.setSpacing(8)
        top = QHBoxLayout()
        self.res_title = label("No sweep loaded", "cardTitle", wrap=True)
        self.res_sub = label("", "muted", wrap=True, selectable=True)
        top.addWidget(self.res_title, 1)
        self.btn_settings = button("Load settings", self.load_settings,
                                   tooltip="Copy this sweep's parameter, values and base parameters into the form")
        self.btn_folder = button("Show in Finder", lambda: self.sweep and reveal(self.sweep["dir"]))
        for b in (self.btn_settings, self.btn_folder):
            b.setObjectName("flat")
            top.addWidget(b, 0, Qt.AlignmentFlag.AlignTop)
        xl.addLayout(top)
        xl.addWidget(self.res_sub)
        self.tiles = QWidget()
        tl = QHBoxLayout(self.tiles)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(8)
        self.m_runs = MetricTile("Completed solves", "—")
        self.m_I = MetricTile("Photocurrent |I|", "—")
        self.m_peak = MetricTile("Peak of |S| at X", "—", "measured: +5 µm")
        self.m_peak.setToolTip("The measured +25 V tip scan peaks at X = +5 µm")
        self.m_wall = MetricTile("Total compute time", "—")
        for m in (self.m_runs, self.m_I, self.m_peak, self.m_wall):
            tl.addWidget(m, 1)
        xl.addWidget(self.tiles)
        self.metrics_line = label("", wrap=True, selectable=True)     # the same numbers in one line when narrow
        self.metrics_line.setVisible(False)
        xl.addWidget(self.metrics_line)

        self.vsplit = vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.setChildrenCollapsible(False)
        # its (large) size hint must not make the scroll area taller than the window: it takes what is left
        vsplit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.fig_tabs = QTabWidget()
        corner = QWidget()
        cl = QHBoxLayout(corner)
        cl.setContentsMargins(0, 0, 6, 2)
        self.opt_measured = QCheckBox("Overlay measured +25 V")
        self.opt_measured.setChecked(True)
        self.opt_measured.toggled.connect(self.replot)
        cl.addWidget(self.opt_measured)
        self.fig_tabs.setCornerWidget(corner, Qt.Corner.TopRightCorner)
        self.plot = PlotWidget(figsize=(8, 5.5))
        self.fig_tabs.addTab(self.plot, "Interactive plot")
        self.compare = ImageView()
        self.fig_tabs.addTab(self.compare, "compare.png")
        vsplit.addWidget(self.fig_tabs)
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(26)
        for c, tip in enumerate(COL_TIPS):
            self.table.horizontalHeaderItem(c).setToolTip(tip)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)      # scrolls sideways when narrow
        hh.setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        self.table.cellDoubleClicked.connect(self._open_run)
        self.table.setToolTip("Click a row to highlight that solve in the plots; double-click to open its full result "
                              "on the Steady state page")
        self.table.setMinimumHeight(84)
        vsplit.addWidget(self.table)
        vsplit.setStretchFactor(0, 4)
        vsplit.setStretchFactor(1, 1)
        vsplit.setSizes([520, 170])
        xl.addWidget(vsplit, 1)
        return area

    def _flow_tiles(self, width: int):
        narrow = width < 700
        self.tiles.setVisible(not narrow)
        self.metrics_line.setVisible(narrow)

    def _build_history(self):
        w = QWidget()
        hl = QVBoxLayout(w)
        hl.setContentsMargins(12, 12, 12, 12)
        hrow = QHBoxLayout()
        hrow.addWidget(label("Saved sweeps", "cardTitle", wrap=True), 1)   # wraps: must not widen the page
        for text, slot in (("Open folder…", self._open_folder), ("Show in Finder", self._reveal_history),
                           ("Refresh", self.refresh_history)):
            b = button(text, slot)
            b.setObjectName("flat")
            hrow.addWidget(b)
        hl.addLayout(hrow)
        self.hist = QListWidget()
        self.hist.setSpacing(3)
        self.hist.itemDoubleClicked.connect(self._on_history)
        hl.addWidget(self.hist, 1)
        hl.addWidget(label("Double-click a sweep to load it. results/sweeps/ holds the sweeps committed to the "
                           "repository, results/gui_runs/sweep_* the sweeps run from this app; use “Open folder…” "
                           "for sweeps written elsewhere.", "muted", wrap=True))
        self.hist.setToolTip("Double-click to load")
        return w

    def _retheme(self, *_):
        t = ThemeManager.instance().tokens
        self.run_btn.setIcon(icons.icon("play", t.on_accent))
        self._update_preview()
        if self.sweep:
            self._fill_table()

    # ------------------------------------------------------------------ sweep definition
    def current_key(self) -> str:
        return self.key.currentData()

    def _on_preset(self, _i):
        name = self.preset.currentData()
        custom = name == "custom"
        self.key.setEnabled(custom)
        if not custom:
            p = next(p for p in PRESETS if p[0] == name)
            self._set_key(p[1])
            self._set_values(p[2])
        self._update_preview()

    def _set_key(self, key):
        self.key.blockSignals(True)
        i = self.key.findData(key)
        if i >= 0:
            self.key.setCurrentIndex(i)
        self.key.blockSignals(False)
        self._key_changed()

    def _on_key(self, _i):
        self._key_changed()
        s = self.specs[self.current_key()]
        d = s.default
        if s.type == "choice":
            self._set_values(list(s.choices))
        elif s.type == "bool":
            self._set_values([True, False])
        elif _is_num(d) and d != 0:
            self._set_values(generate(d / 2, d * 2, 3, log=True) if d > 0 else [d * 2, d, d / 2])
        else:
            self._set_values([d])

    def _key_changed(self):
        """Disable the swept field in the base form and adapt the editors to its type."""
        key = self.current_key()
        if self._swept and self._swept != key:
            self.form.set_enabled_fields([self._swept], True)
            self.form.rows[self._swept].label.setToolTip("")
        self.form.set_enabled_fields([key], False)
        self.form.rows[key].label.setToolTip("This parameter is being swept; the value here is not used")
        self._swept = key
        s = self.specs[key]
        numeric = s.type in ("float", "int")
        self.gen_box.setVisible(numeric)
        self.unit_lab.setText(pretty_unit(s.unit))

    def _set_values(self, vals):
        self.values_edit.setText(", ".join(fmt_value(v) for v in vals))
        nums = [float(v) for v in vals if _is_num(v)]
        if len(nums) >= 2:
            a, b = min(nums), max(nums)
            self.gen_from.setText(format_float(a))
            self.gen_to.setText(format_float(b))
            self.gen_n.setValue(max(2, min(40, len(nums))))
            self.gen_mode.setCurrentIndex(1 if a > 0 and b / a >= 20 else 0)

    def _generate(self):
        try:
            a, b = float(self.gen_from.text()), float(self.gen_to.text())
            vals = generate(a, b, self.gen_n.value(), self.gen_mode.currentIndex() == 1)
        except ValueError as e:
            why = str(e) if str(e).startswith("log spacing") else "enter numeric start and end values"
            self.preview.setText(f"<span style='color:{ThemeManager.instance().tokens.danger}'>"
                                 f"Cannot fill values: {why}</span>")
            return
        if self.specs[self.current_key()].type == "int":
            vals = sorted({int(round(v)) for v in vals})
        self._set_values(vals)

    def values(self) -> list:
        return parse_values(self.specs[self.current_key()], self.values_edit.text())

    def _update_preview(self, *_):
        if not hasattr(self, "est"):
            return
        t = ThemeManager.instance().tokens
        s = self.specs[self.current_key()]
        try:
            vals = self.values()
            unit = pretty_unit(s.unit)
            txt = f"{plural(len(vals), 'solve')}: " + ", ".join(map(fmt_value, vals)) + (f" {unit}" if unit else "")
            if s.type == "choice":
                txt += f"<br>Choices: {', '.join(s.choices)}"
            if self.preset.currentData() == "anode":
                txt += "<br>Note: the measured points in compare.png are always the +25 V data"
            self.preview.setText(txt)
            n = len(vals)
        except ValueError as e:
            self.preview.setText(f"<span style='color:{t.danger}'>{e}</span>")
            n = 0
        self._n = n
        self.cores.set_runs(max(1, n))          # follows the number of values until the user sets it
        self._update_estimate()
        self.run_btn.setEnabled(n > 0)

    def estimate(self, n: int | None = None) -> tuple[int, float, float]:
        """(cores used, wall-time estimate in s, time on one core in s) for n values."""
        n = self._n if n is None else n
        per_s = MIN_PER_RUN[self.quick.isChecked()] * 60
        k = max(1, min(self.cores.value(), n))
        return k, estimate_s(per_s, n, k), n * per_s

    def _update_estimate(self, *_):
        if not hasattr(self, "est"):
            return
        n = self._n
        per = MIN_PER_RUN[self.quick.isChecked()]
        self.per_run.setText(f"{per:g} min per solve")
        if not n:
            self.est.setText("")
            self.est_head.setText("")
            return
        k, wall, one = self.estimate(n)
        if n == 1:
            txt = f"1 solve ≈ {fmt_dur(one)} (a single solve runs on one core)"
        elif k == 1:
            txt = f"{n} solves one after another ≈ {fmt_dur(one)}"
        else:
            txt = f"{n} solves on {k} cores ≈ {fmt_dur(wall)} (≈ {fmt_dur(one)} on one core)"
        self.est.setText(txt)
        if k == 1:
            tip = f"{per:g} min per solve on one core, solved one after another."
        else:
            rounds = math.ceil(n / k)
            tip = (f"{per:g} min per solve on one core. Solves that share the machine run a little slower "
                   f"(≈ {slowdown(k):.2f}× with {k} at once)"
                   + (f"; with {k} cores for {n} values they run in {rounds} rounds." if rounds > 1 else "."))
        self.est.setToolTip(tip + " A single solve cannot use more than one core: its sparse LU solver is "
                                  "single-threaded.")
        self.est_head.setText(f"{plural(n, 'solve')} · ≈ {fmt_dur(wall)}"
                              + (" · coarse grid" if self.quick.isChecked() else ""))

    def _toggle_quick(self, on):
        if on:
            self.form.set_values({k: v for k, v in QUICK.items() if k != self.current_key()})
        else:
            self.form.set_values({k: self.form.rows[k].spec.default for k in QUICK if k != self.current_key()})
        self._update_preview()

    def _on_form_changed(self):
        on = all(self.form.values().get(k) == v for k, v in QUICK.items() if k != self.current_key())
        if on != self.quick.isChecked():
            self.quick.blockSignals(True)
            self.quick.setChecked(on)
            self.quick.blockSignals(False)
            self._update_preview()

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder",
                                             str(paths.GUI_RUNS if paths.GUI_RUNS.exists() else paths.RESULTS))
        if d:
            self.out_edit.setText(paths.rel(d))

    # ------------------------------------------------------------------ run
    def run(self, confirm=True):
        key = self.current_key()
        try:
            vals = self.values()
        except ValueError as e:
            error_box(self, "Invalid values", str(e))
            return None
        if not self.form.is_valid():
            error_box(self, "Invalid parameters", "These base parameters have values that cannot be read:\n"
                      + ", ".join(self.form.invalid_fields()))
            return None
        base = {k: v for k, v in self.form.overrides().items() if k != key}
        n = len(vals)
        quick = self.quick.isChecked()
        workers, wall, one = self.estimate(n)
        out = self.out_edit.text().strip()
        root = paths.absolute(Path(out).expanduser()) if out else paths.new_run_dir("sweep")
        if confirm:
            if (root / key).exists() and any((root / key).iterdir()):
                r = QMessageBox.question(self, "Overwrite existing results?",
                                         f"{paths.rel(root / key)} already contains files. Results for values that "
                                         "are swept again will be overwritten. Continue?")
                if r != QMessageBox.StandardButton.Yes:
                    return None
            if wall / 60 > self.confirm_minutes:
                how = ("one after another" if workers == 1
                       else f"{workers} at a time" if workers < n else f"all at once on {workers} cores")
                r = QMessageBox.question(self, "Start sweep?",
                                         f"Run {plural(n, 'steady solve')} on the "
                                         f"{'coarse' if quick else 'production'} grid, {how}? "
                                         f"This takes ≈ {fmt_dur(wall)}"
                                         + (f" (≈ {fmt_dur(one)} on one core)" if workers > 1 else "") + ".\n\n"
                                         "The job runs in the background; you can keep using the other pages.")
                if r != QMessageBox.StandardButton.Yes:
                    return None
        title = (f"Sweep {key} · {plural(n, 'value')} · {plural(workers, 'core')}"
                 + (" · coarse grid" if quick else ""))
        if self.job is not None:              # only the newest sweep drives the live refresh
            try:
                self.job.output.disconnect(self._on_job_output)
            except (RuntimeError, TypeError):
                pass
        self.job = JobManager.instance().submit(
            title, "sweep", "ltemsim_gui/parallel.py", "sweep",
            kwargs=dict(root=paths.rel(root), key=key, values=vals, base=base, name=key, workers=workers), out=root,
            progress={"type": "count", "pattern": PROGRESS_PATTERN, "total": n})
        self.job.output.connect(self._on_job_output)
        self._follow = (root / key).resolve()
        self.panel.attach(self.job)
        self.tabs.setCurrentIndex(0)
        return self.job

    def _job_sweep_dir(self, job):
        if job is None or not job.out:
            return None
        return paths.absolute(job.out) / job.spec.get("kwargs", {}).get("name", job.spec.get("kwargs", {}).get("key", ""))

    def _running_here(self, d) -> bool:
        """Is the sweep launched from this page still queued / running into folder d?"""
        jd = self._job_sweep_dir(self.job)
        return jd is not None and self.job.active and jd.resolve() == Path(d).resolve()

    def _on_job_output(self, text):
        # a value finished: refresh the results view live (unless the user has opened another sweep meanwhile)
        if not any(_DONE_RE.search(l) for l in text.splitlines()):
            return
        d = self._job_sweep_dir(self.job)
        if d is not None and d.is_dir() and d.resolve() == self._follow:
            self.show_result(d, switch=False)
            self.refresh_history()

    def _on_finished(self, job):
        if job is not self.job:
            return
        d = self._job_sweep_dir(job)
        if d is not None and d.is_dir():
            self.show_result(d, switch=job.status == FINISHED)
        self.refresh_history()

    # ------------------------------------------------------------------ results
    def show_result(self, path, switch=True, select=None):
        d, lab = find_sweep_dir(path)
        if d is None:            # clear the view: a stale table under a "not found" title would be misleading
            self.sweep, self.highlight = None, None
            self.res_title.setText("No sweep found")
            self.res_sub.setText(f"{paths.rel(path)} is not a sweep folder "
                                 "(expected compare.png or <parameter>=<value>/summary.json)")
            for m in (self.m_runs, self.m_I, self.m_wall):
                m.set("—", "")
            self.m_peak.set("—")                # keeps its fixed hint (measured peak position)
            self.metrics_line.setText("")
            self.table.setRowCount(0)
            self.compare.clear()
            self.btn_settings.setEnabled(False)
            self.replot()
            return False
        old = self.sweep
        if select is None and lab is None and old and Path(old["dir"]) == d and self.highlight is not None \
                and self.highlight < len(old["runs"]):
            select = old["runs"][self.highlight]["label"]          # same sweep reloaded: keep the highlighted run
        self.sweep = load_sweep(d)
        sw = self.sweep
        self.highlight = None
        key = sw["key"] or "?"
        spec = self.specs.get(key)
        done = [r for r in sw["runs"] if r["done"]]
        ms = [r["metrics"] for r in done]
        conv = [m.get("converged") for m in ms if m.get("converged") is not None]
        state = "" if not conv else ("  ·  all converged" if all(conv)
                                     else f"  ·  ⚠ {conv.count(False)} not fully converged")
        count = (plural(sw["n_done"], "value") if sw["n_done"] >= sw["n_total"]
                 else f"{sw['n_done']} of {plural(sw['n_total'], 'value')}")
        self.res_title.setText(f"{spec.label if spec else key} sweep  ·  {count}{state}")
        self.res_title.setToolTip(key)
        base = sw["base"]
        coarse = all(base.get(k) == v for k, v in QUICK.items())
        btxt = ", ".join(f"{k}={fmt_value(v)}" for k, v in base.items() if not (coarse and k in QUICK))
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime))
        foreign = sw["foreign"]
        self.res_sub.setText(f"{key}  ·  {paths.rel(d)}  ·  {when}  ·  {'coarse grid' if coarse else 'production grid'}"
                             + (f"  ·  base changes: {btxt}" if btxt
                                else ("" if sw["spec"] else "  ·  other parameters at their defaults"))
                             + (f"\n{plural(len(foreign), 'more value folder')} from an earlier sweep into this folder "
                                "with different base parameters, not listed: "
                                + ", ".join(foreign[:4]) + (" …" if len(foreign) > 4 else "") if foreign else ""))
        fmt = lambda v, f: f.format(v) if v is not None and np.isfinite(v) else "—"
        missing = sw["n_total"] - sw["n_done"]
        self.m_runs.set(f"{sw['n_done']} / {sw['n_total']}", f"{missing} not finished" if missing else "all finished")
        Is = [m["I"] for m in ms if m.get("I") is not None and np.isfinite(m["I"])]
        self.m_I.set(f"{min(Is):.3f} – {max(Is):.3f} mA" if len(Is) > 1 else fmt(Is[0] if Is else None, "{:.3f} mA"),
                     "measured ≈ 0.6 mA at +25 V")
        Xs = sorted({round(m["peak_X"], 1) + 0.0 for m in ms if m.get("peak_X") is not None})
        self.m_peak.set(f"{fmt_x(Xs[0])} … {fmt_x(Xs[-1])} µm" if len(Xs) > 1 else f"{fmt_x(Xs[0])} µm" if Xs else "—")
        walls = [m["wall"] for m in ms if m.get("wall")]
        workers = sw["spec"].get("workers")
        self.m_wall.set(fmt_elapsed(sum(walls)) if walls else "—",
                        (f"≈ {fmt_elapsed(sum(walls) / len(walls))} per value" if walls else "")
                        + (f" · {workers} cores" if walls and isinstance(workers, int) and workers > 1 else ""))
        self.metrics_line.setText("  ·  ".join(f"{m.lab.text()}: {m.val.text()}" for m in
                                               (self.m_runs, self.m_I, self.m_peak, self.m_wall)))
        self.metrics_line.setToolTip("\n".join(f"{m.lab.text()}: {m.val.text()} ({m.hint.text()})" for m in
                                                (self.m_runs, self.m_I, self.m_peak, self.m_wall) if m.hint.text()))
        cmp_ok = sw["compare"].exists()
        if cmp_ok and self._running_here(d):       # a compare.png older than the running re-sweep is not its figure
            cmp_ok = sw["compare"].stat().st_mtime >= (self.job.started or time.time())
        if cmp_ok:
            self.compare.load(sw["compare"])
        else:
            # short lines: the empty-state label does not wrap and would widen the whole results column
            self.compare.clear("No compare.png yet\nscripts/sweep.py draws it once every value is done\n"
                               "(missing when the sweep is unfinished or failed)")
        self._fill_table()
        h = 34 + 26 * max(2, min(6, len(sw["runs"])))
        tot = sum(self.vsplit.sizes()) or 690
        self.vsplit.setSizes([max(200, tot - h), h])
        want = lab or select
        self.highlight = next((i for i, r in enumerate(sw["runs"]) if r["label"] == want), None) if want else None
        self.table.blockSignals(True)          # selection follows self.highlight; one replot below
        self.table.clearSelection()
        if self.highlight is not None:
            self.table.selectRow(self.highlight)
        self.table.blockSignals(False)
        self.replot()
        self.btn_settings.setEnabled(bool(sw["runs"]))
        if switch:
            self.tabs.setCurrentIndex(1)
        return True

    def _colors(self) -> list[str]:
        runs = self.sweep["runs"] if self.sweep else []
        dark = ThemeManager.instance().tokens.dark
        if runs and all(_is_num(r["value"]) for r in runs):
            return ramp_colors(len(runs), dark)
        cat = series_colors(dark)
        return [cat[i % len(cat)] for i in range(len(runs))]

    def _fill_table(self):
        sw = self.sweep
        t = ThemeManager.instance().tokens
        colors = self._colors()
        spec = self.specs.get(sw["key"] or "")
        unit = pretty_unit(spec.unit) if spec else ""
        self.table.blockSignals(True)
        self.table.setRowCount(len(sw["runs"]))
        running = self._running_here(sw["dir"])
        for i, r in enumerate(sw["runs"]):
            m = r["metrics"]
            f = lambda k, fs: fs.format(m[k]) if m.get(k) is not None and np.isfinite(m[k]) else "—"
            if r["done"]:
                conv = m.get("converged")
                cells = [f"{fmt_value(r['value'])} {unit}".strip(), f("I", "{:.4f}"), f("gain", "{:.3f}"),
                         fmt_x(m["peak_X"]) if m.get("peak_X") is not None else "—", f("ratio5", "{:.2f}"),
                         f("S0", "{:.2f}"),
                         "—" if conv is None else ("✓" if conv else "⚠ no"),
                         fmt_elapsed(m["wall"]) if m.get("wall") else "—"]
            else:
                st = ("running" if r.get("started") else "queued") if running else \
                    ("interrupted" if r.get("started") else "not run")
                cells = [f"{fmt_value(r['value'])} {unit}".strip(), "—", "—", "—", "—", "—", st, "—"]
            for c, txt in enumerate(cells):
                it = QTableWidgetItem(txt)
                it.setTextAlignment((Qt.AlignmentFlag.AlignLeft if c == 0 else Qt.AlignmentFlag.AlignHCenter if c == 6
                                     else Qt.AlignmentFlag.AlignRight) | Qt.AlignmentFlag.AlignVCenter)
                if c == 0:
                    it.setIcon(_swatch(colors[i]) if r["done"] else QIcon())
                    it.setToolTip(str(r["dir"]))
                if not r["done"]:
                    it.setForeground(QColor(t.muted))
                elif c == 6 and m.get("converged") is False:
                    it.setForeground(QColor(t.warning))
                self.table.setItem(i, c, it)
        self.table.blockSignals(False)

    def _on_row_selected(self):
        rows = {i.row() for i in self.table.selectionModel().selectedRows()}
        self.highlight = next(iter(rows)) if rows else None
        self.replot()

    def _open_run(self, row, _col):
        if not self.sweep or row >= len(self.sweep["runs"]):
            return
        r = self.sweep["runs"][row]
        if r["done"]:
            nav.navigate("steady", path=str(r["dir"]))

    def replot(self, *_):
        self.plot.plot(self._draw)

    def _draw(self, fig):
        t = ThemeManager.instance().tokens
        sw = self.sweep
        runs = [(i, r) for i, r in enumerate(sw["runs"]) if r["done"]] if sw else []
        if not runs:
            ax = fig.add_subplot(111)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No completed runs yet" if sw else "No sweep loaded", ha="center", va="center",
                    color=t.muted, transform=ax.transAxes)
            return
        colors = self._colors()
        key = sw["key"]
        spec = self.specs.get(key)
        unit = pretty_unit(spec.unit) if spec else ""
        axs = fig.subplots(2, 2)
        a_s, a_w, a_n, a_i = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
        hl = self.highlight
        for i, r in runs:
            s = r["summary"]
            on = hl is None or hl == i
            kw = dict(color=colors[i], lw=2.6 if hl == i else 1.8, alpha=1.0 if on else 0.28, zorder=4 if hl == i else 3)
            lab = f"{fmt_value(r['value'])} {unit}".strip()
            ts = s.get("tip_scan") or {}
            if ts.get("X"):
                X = np.asarray(ts["X"], float)
                S = np.abs(np.asarray(ts["S"], float))
                o = np.argsort(X)
                a_s.plot(X[o], S[o] / S.max() if S.max() > 0 else S[o], label=lab, **kw)
            pr = s.get("profile") or {}
            if pr.get("x"):
                x = np.asarray(pr["x"], float)
                w = np.asarray(pr["w_int"], float)
                a_w.plot(x, w / w.max() if w.max() > 0 else w, label=lab, **kw)
                a_n.plot(x, np.asarray(pr["NTp"], float), label=lab, **kw)
        if self.opt_measured.isChecked():
            meas = [m for m in read_measured() if m[0] == 25]
            if meas:
                mx, my, me = (np.array([m[k] for m in meas]) for k in (1, 2, 3))
                a_s.errorbar(mx, my, yerr=me, fmt="o", color=t.text, mfc=t.surface, mew=1.5, ms=6, capsize=3,
                             label="measured +25 V", zorder=6)
        a_s.set(xlim=(-10, 10), xlabel="X = spot − tip (µm)", ylabel="|S| / max", title="Tip signal |S|(X)")
        a_w.set(xlim=(-12, 12), xlabel="x (µm)", ylabel="weight (norm.)", title="Next-pulse weight")
        a_n.set(xlim=(-25, 25), yscale="log", xlabel="x (µm)", ylabel=r"EL2$^+$ (cm$^{-3}$)",
                title=r"EL2$^+$ at 0.5 µm depth")
        for ax in (a_s, a_w, a_n):
            ax.axvline(0, color=t.border, lw=1, zorder=0)
        ncol = 1 if len(runs) <= 5 else 2
        a_s.legend(loc="upper left", fontsize=7.5, ncol=ncol, handlelength=1.4)

        # photocurrent vs swept value
        vals = [r["value"] for _, r in runs]
        I = np.array([r["metrics"].get("I", np.nan) for _, r in runs], float)
        numeric = all(_is_num(v) for v in vals)
        xs = np.array(vals, float) if numeric else np.arange(len(vals), dtype=float)
        if numeric:
            a_i.plot(xs, I, "-", color=t.muted, lw=1.2, zorder=2)
        for (i, r), x, y in zip(runs, xs, I):
            on = hl is None or hl == i
            a_i.plot([x], [y], "o", color=colors[i], ms=9 if hl == i else 7, mec=t.surface, mew=1.2,
                     alpha=1.0 if on else 0.35, zorder=3)
        if numeric and len(xs) > 1 and xs.min() > 0 and xs.max() / xs.min() >= 20:
            a_i.set_xscale("log")
        if not numeric:
            a_i.set_xticks(xs, [str(v) for v in vals])
            a_i.set_xlim(-0.6, len(vals) - 0.4)
        top = np.nanmax(I) if np.isfinite(I).any() else 1.0
        a_i.set_ylim(0, top * 1.18 if top > 0 else 1.0)
        a_i.set(xlabel=f"{key}" + (f" ({unit})" if unit else ""), ylabel="|I| (mA)", title="Photocurrent")
        a_i.grid(True, axis="y", lw=0.6)

    # ------------------------------------------------------------------ settings / history
    def load_settings(self):
        """Fill the sweep definition and base parameters from the loaded sweep."""
        sw = self.sweep
        if not sw or not sw["key"] or sw["key"] not in self.specs:
            return
        key = sw["key"]
        vals = sw["spec"].get("values") or [r["value"] for r in sw["runs"]]
        name = sw["name"]
        idx = next((i for i, p in enumerate(PRESETS) if p[0] == name and p[1] == key), None)
        if idx is None:
            idx = next((i for i, p in enumerate(PRESETS) if p[1] == key), None)
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(idx if idx is not None else len(PRESETS))
        self.preset.blockSignals(False)
        self.key.setEnabled(idx is None)
        self._set_key(key)
        self._set_values(vals)
        base = {k: v for k, v in sw["base"].items() if k in self.form.rows}
        self.form.set_values(base, reset_others=True)
        self._on_form_changed()
        self._update_preview()

    def refresh_history(self):
        self.hist.clear()
        found = list_sweeps()
        for d in found:
            try:
                info = sweep_info(d)
            except OSError:
                continue
            spec = self.specs.get(info["key"])
            unit = pretty_unit(spec.unit) if spec else ""
            vals = ", ".join(fmt_value(v) for v in info["values"][:8]) + (" …" if len(info["values"]) > 8 else "")
            state = "complete" if info["complete"] else f"incomplete {info['n_done']}/{info['n_total']}"
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime))
            txt = (f"{paths.rel(d)}\n    {info['key']}: {vals}{f' {unit}' if unit else ''} · "
                   f"{plural(info['n_done'], 'solve')} · {state}"
                   f"{' · coarse grid' if info['coarse'] else ''} · {when}")
            it = QListWidgetItem(txt)
            it.setData(Qt.ItemDataRole.UserRole, str(d))
            self.hist.addItem(it)
        if not found:
            it = QListWidgetItem("No sweeps yet")
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.hist.addItem(it)
        return found

    def _on_history(self, it):
        p = it.data(Qt.ItemDataRole.UserRole)
        if p:
            self._follow = None
            self.show_result(p)

    def _open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Choose a sweep folder",
                                             str(paths.GUI_RUNS if paths.GUI_RUNS.exists() else paths.RESULTS))
        if d:
            self._follow = None
            if not self.show_result(d):
                error_box(self, "Not a sweep folder", f"No sweep results found in {paths.rel(d)}.\n"
                          "Expected sweep_spec.json, compare.png or summary.json files in <parameter>=<value> "
                          "subfolders.")

    def _reveal_history(self):
        it = self.hist.currentItem()
        p = it.data(Qt.ItemDataRole.UserRole) if it else None
        reveal(p or (paths.RESULTS / "sweeps"))

    def on_show(self):
        found = self.refresh_history()
        if self._first_show:
            self._first_show = False
            if self.sweep is None and found:
                self.show_result(found[0], switch=self.job is None)

    def open(self, path=None, **_):
        if path:
            self._follow = None
            self.show_result(path)
