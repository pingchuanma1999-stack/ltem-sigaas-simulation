"""Steady-state drift-diffusion + EL2 solver page (scripts/run_baseline.py)."""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QSize, Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSizePolicy, QSplitter,
                               QTabWidget, QVBoxLayout, QWidget)

from .. import icons, nav, paths
from ..jobs import FINISHED, STATUS_TEXT, JobManager
from ..paramspec import params_specs
from ..theme import ThemeManager, repolish
from ..widgets.common import MetricTile, Page, button, error_box, label, reveal
from ..widgets.device import FieldView, device_from_params, draw_device, load_steady_fields
from ..widgets.imageview import ImageView
from ..widgets.mpl import PlotWidget
from ..widgets.paramform import ParamForm
from ..widgets.runpanel import RunPanel
from ..workers import run_async

# Params fields that no code reads (verified with grep) or that only the pulse model uses
HIDDEN = {"gummel_max_iter", "gummel_tol", "poisson_max_iter", "poisson_tol", "generation_ramp", "bias_ramp_steps",
          "rep_rate"}
QUICK = {"dx_fine": 0.5, "x_fine": 8.0, "dz_fine": 0.25}      # the coarse grid of results/coarse_scenarios (30-60 s)
FIGURES = (("maps.png", "Maps"), ("profiles.png", "Profiles"), ("tip_scan.png", "Tip scan"),
           ("polarisation.png", "Polarisation"))
FIGURE_TIPS = {"maps.png": "Maps of the potential, E_x, E_z, n, p and EL2⁺ (maps.png)",
               "profiles.png": "Lateral profiles 0.5 µm below the surface (profiles.png)",
               "tip_scan.png": "Tip E_z versus spot–tip offset X, with the measured scan (tip_scan.png)",
               "polarisation.png": "Polarisation P = n·E (polarisation.png)"}
RUNTIME = "≈ 6–7 min on the production grid, 30–60 s on the coarse grid"
# Params fields the device diagram depends on (redraw only when one of them changes)
DEVICE_KEYS = ("gap", "x_half", "depth", "air_height", "V_bias", "anode_side", "spot_x0_um", "spot_w_um", "alpha_cm",
               "P_avg_mW", "tip_height_um", "tip_x_scan_um")
# Allowed ranges (inclusive). Out-of-range values make the solver fail late or hang: a negative dx_fine never
# finishes building the grid, T = 0 or spot_w_um = 0 divide by zero, P_avg_mW < 0 or reflectivity > 1 give G < 0.
_POS = 1e-30
BOUNDS = {
    "T": (1.0, 2000.0), "gap": (0.1, None), "x_half": (1.0, None), "depth": (0.5, None), "air_height": (0.5, None),
    "dx_fine": (0.005, None), "x_fine": (0.05, None), "dx_coarse": (0.01, None), "dz_fine": (0.005, None),
    "z_fine": (0.05, None), "dz_coarse_semi": (0.01, None), "dz_coarse_air": (0.01, None),
    "eps_r": (1.0, None), "Eg": (0.01, None), "Nc": (1.0, None), "Nv": (1.0, None), "mu_n": (_POS, None),
    "mu_p": (_POS, None), "v_sat_n": (1.0, None), "v_sat_p": (1.0, None), "B_rad": (0.0, None), "S_surface": (0.0, None),
    "contact_density": (0.0, None), "tau_srh_ns": (0.0, None), "N_A": (0.0, None), "N_D": (0.0, None),
    "N_T": (0.0, None), "E_T_below_Ec": (0.0, None), "sigma_n": (0.0, None), "sigma_p": (0.0, None),
    "sigma_n_opt": (0.0, None), "sigma_p_opt": (0.0, None), "m_eff_n": (1e-3, None), "m_eff_p": (1e-3, None),
    "wavelength_nm": (1.0, None), "P_avg_mW": (0.0, None), "spot_w_um": (0.01, None), "reflectivity": (0.0, 1.0),
    "alpha_cm": (0.0, None), "tip_height_um": (1e-4, None), "n_floor": (_POS, None),
}


def steady_specs():
    out = []
    for s in params_specs():
        if s.name in HIDDEN:
            continue
        lo, hi = BOUNDS.get(s.name, (None, None))
        out.append(dataclasses.replace(s, minimum=lo, maximum=hi) if s.type in ("float", "int") else s)
    return out


def check_params(v: dict) -> list[str]:
    """Cross-field checks the per-field ranges cannot express (empty = ok)."""
    bad = []
    if v["x_fine"] >= v["x_half"]:
        bad.append(f"The fine-grid extent x_fine = {v['x_fine']:g} µm must be smaller than the box half-width "
                   f"x_half = {v['x_half']:g} µm")
    if v["gap"] / 2 >= v["x_half"]:
        bad.append(f"Half the electrode gap (gap = {v['gap']:g} µm) must be smaller than x_half = {v['x_half']:g} µm, "
                   "so that both electrodes lie inside the box")
    if v["z_fine"] >= min(v["depth"], v["air_height"]):
        bad.append(f"The fine-grid z extent z_fine = {v['z_fine']:g} µm must be smaller than both the GaAs depth "
                   "and the air height")
    if v["tip_height_um"] >= v["air_height"]:
        bad.append(f"The tip sampling height {v['tip_height_um']:g} µm must be smaller than the air height "
                   f"{v['air_height']:g} µm")
    if v["E_T_below_Ec"] >= v["Eg"]:
        bad.append("The EL2 level depth E_c − E_T must be smaller than the band gap E_g")
    xs = v.get("tip_x_scan_um") or ()
    if len(xs) < 2:
        bad.append("The tip scan needs at least 2 positions (an empty list only fails after the solve)")
    elif max(abs(x) for x in xs) >= v["x_half"]:
        bad.append(f"All tip scan positions must lie inside the box (|X| < x_half = {v['x_half']:g} µm)")
    return bad


def steady_metrics(summary: dict) -> dict:
    """Headline numbers from a run_baseline / sweep summary.json."""
    out = {}
    I = summary.get("I_mA", {})
    if I:
        out["I"] = abs(I.get("left", float("nan")))
        ug = I.get("unity_gain")
        out["gain"] = out["I"] / ug if ug else float("nan")
        out["unity"] = ug
    ts = summary.get("tip_scan")
    if ts and ts.get("X"):
        X = np.asarray(ts["X"], float)
        S = np.asarray(ts["S"], float)
        o = np.argsort(X)
        X, S = X[o], S[o]
        A = np.abs(S)
        out["peak_X"] = float(X[np.argmax(A)])
        out["S0"] = float(np.interp(0.0, X, A) / A.max()) if A.max() > 0 else float("nan")
        a5, am5 = np.interp(5.0, X, A), np.interp(-5.0, X, A)
        out["ratio5"] = float(a5 / am5) if am5 > 0 else float("nan")
    log = summary.get("log")
    if log:
        out["converged"] = all(bool(l.get("converged")) for l in log if l.get("converged") is not None)
        out["wall"] = max((l.get("t_wall", 0) for l in log), default=0)      # t_wall is cumulative per stage
    elif isinstance(summary.get("converged"), list):          # sweep run summaries
        out["converged"] = all(bool(c) for c in summary["converged"] if c is not None)
        out["wall"] = summary.get("wall_s")
    return out


def _fmt(v, f):
    return f.format(v + 0.0) if v is not None and np.isfinite(v) else "—"      # + 0.0: no "-0.0"


def result_params(d: Path, summary: dict | None = None) -> dict:
    """Params overrides stored with a result: params.json (run_baseline) or summary["params"] (sweep runs)."""
    try:
        p = json.loads((d / "params.json").read_text())
        if isinstance(p, dict):
            return p
    except (OSError, ValueError):
        pass
    if summary is None:
        try:
            summary = json.loads((d / "summary.json").read_text())
        except (OSError, ValueError):
            summary = {}
    p = summary.get("params") if isinstance(summary, dict) else None
    return p if isinstance(p, dict) else {}


def is_quick(pr: dict) -> bool:
    try:
        return all(abs(float(pr.get(k, -1)) - v) < 1e-12 for k, v in QUICK.items())
    except (TypeError, ValueError):
        return False


def no_field_message(d: Path, has_summary: bool) -> str:
    """Empty-state text of the Electric field tab for a result folder without solution.npz."""
    if not d.is_dir():
        return f"The folder {paths.rel(d)} does not exist."
    if not has_summary:
        return ("This run did not finish (no summary.json), so there is no solution.npz to show.\n\n"
                "Run the solve again — the coarse-grid quick mode takes about 30–60 s.")
    return ("No field data: this result has no solution.npz.\n\n"
            "Solution files are not committed to the repository, so results/baseline and the parameter-sweep "
            "folders only keep their figures.\n\n"
            "Run a solve on this page to see the electric field — the coarse-grid quick mode takes about 30–60 s.")


class _Elided(QLabel):
    """One-line label that elides in the middle instead of forcing the layout to its full text width."""

    def __init__(self, text="", obj=None):
        super().__init__()
        if obj:
            self.setObjectName(obj)
        self._full = ""
        self.setText(text)

    def setText(self, s):
        self._full = s
        self.setToolTip(s)
        self._elide()
        self.updateGeometry()

    def text(self):
        return self._full

    def sizeHint(self):
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + 4, super().sizeHint().height())

    def minimumSizeHint(self):
        return QSize(80, super().minimumSizeHint().height())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._elide()

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._elide()
            self.updateGeometry()

    def _elide(self):
        super().setText(self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideMiddle, max(20, self.width())))


class SteadyPage(Page):
    title = "Steady state"
    subtitle = ("Self-consistent 2-D drift-diffusion + EL2 steady state of the pumped, biased SI-GaAs gap, and the tip "
                f"E_z versus the spot–tip offset X. {RUNTIME}.")
    WIDE = 640          # results tab width above which the metric tiles fit in one row
    SHORT = 620         # tab height below which a one-line summary replaces the tiles (room for the field map)
    SIDE_BY_SIDE = 1.45  # device canvas width / height above which the two diagram panels sit side by side

    def __init__(self):
        super().__init__()
        self.job = None
        self.result_dir = None
        self._field_token = 0
        self._dev_key = None
        self.quick = QCheckBox("Coarse-grid quick mode")
        self.quick.setToolTip("dx_fine = 0.5 µm, x_fine = 8 µm, dz_fine = 0.25 µm — the grid of "
                              "results/coarse_scenarios (≈ 30–60 s). Coarser numbers than the production grid; "
                              "use it to try parameters quickly.")
        self.quick.toggled.connect(self._toggle_quick)
        self.header.add_action(self.quick)
        self.run_btn = button("  Run solve", self.run, "primary")
        self.header.add_action(self.run_btn)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        # ---------------- left: parameters
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)
        self.form = ParamForm(steady_specs())
        self.form.changed.connect(self._sync_quick)
        ll.addWidget(self.form, 1)
        orow = QHBoxLayout()
        orow.addWidget(QLabel("Output folder"))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Automatic: results/gui_runs/steady_<time>")
        self.out_edit.setToolTip("Leave empty to create a new results/gui_runs/steady_<time> for every run "
                                 "(existing results are never overwritten)")
        orow.addWidget(self.out_edit, 1)
        b = QPushButton("Choose…")
        b.clicked.connect(self._pick_out)
        orow.addWidget(b)
        ll.addLayout(orow)
        left.setMinimumWidth(330)     # the form needs ~376 px without a horizontal scroll bar; 330 keeps 1000 px usable
        split.addWidget(left)

        # ---------------- right: device / run / results / history
        self.tabs = QTabWidget()
        self.dev_tab = dev_tab = QWidget()
        dl = QVBoxLayout(dev_tab)
        dl.setContentsMargins(10, 10, 10, 6)
        dl.setSpacing(6)
        drow = QHBoxLayout()
        drow.addWidget(label("Device cross-section", "cardTitle"))
        dhint = _Elided("Drawn from the parameters in the form — redraws as you edit them", "muted")
        drow.addWidget(dhint, 1)
        dl.addLayout(drow)
        self.dev_warn = label("", wrap=True)
        self.dev_warn.hide()
        dl.addWidget(self.dev_warn)
        self.dev_plot = PlotWidget(figsize=(8, 4.6))
        # the canvas' 800 × 460 px size hint must not set the page height (it would scroll in small windows)
        self.dev_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.dev_plot.setMinimumHeight(220)
        self.dev_plot.canvas.installEventFilter(self)
        self._dev_side = None
        dl.addWidget(self.dev_plot, 1)
        self.tabs.addTab(dev_tab, "Device")
        self._dev_timer = QTimer(self)
        self._dev_timer.setSingleShot(True)
        self._dev_timer.setInterval(300)
        self._dev_timer.timeout.connect(self._update_device)
        self.form.changed.connect(self._dev_timer.start)

        self.run_tab = run_tab = QWidget()
        rl = QVBoxLayout(run_tab)
        rl.setContentsMargins(12, 12, 12, 12)
        self.panel = RunPanel()
        # long job titles must not widen the page (RunPanel's title label does not elide)
        self.panel.title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.panel.job_finished.connect(self._on_finished)
        cores_note = label(
            "One solve runs on one CPU core: about 96 % of its time is the sparse LU factorisation (SciPy SuperLU), "
            "which is single-threaded, and multithreaded solvers do not help at this problem size. To use all cores, "
            "run several solves at once — Parameter sweep does that, or allow more jobs at a time in the Jobs menu.",
            "muted", wrap=True)
        rl.addWidget(cores_note)
        rl.addWidget(self.panel)
        self.tabs.addTab(run_tab, "Run")

        self.res_tab = res_tab = QWidget()
        xl = QVBoxLayout(res_tab)
        xl.setContentsMargins(10, 10, 10, 10)
        xl.setSpacing(8)
        self.res_head = QGridLayout()
        self.res_head.setHorizontalSpacing(6)
        self.res_head.setVerticalSpacing(4)
        self.res_bar = QWidget()            # folder name (elided) + status badges
        ml = QHBoxLayout(self.res_bar)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(6)
        self.res_title = _Elided("No result loaded", "cardTitle")
        self.res_state = label("", "badge")
        self.res_grid = label("Coarse grid", "badge")
        self.res_grid.setToolTip("dx_fine = 0.5 µm, x_fine = 8 µm, dz_fine = 0.25 µm: coarser than the production "
                                 "grid — use the numbers for quick comparisons only")
        ml.addWidget(self.res_title)
        ml.addWidget(self.res_state)
        ml.addWidget(self.res_grid)
        ml.addStretch(1)
        self.btn_explore = button("Open in explorer", self._open_explorer)
        self.btn_folder = button("Show in Finder", lambda: self.result_dir and reveal(self.result_dir))
        self.btn_params = button("Load parameters", self._load_result_params,
                                 tooltip="Copy this result's parameters into the form")
        for b in (self.btn_params, self.btn_explore, self.btn_folder):
            b.setObjectName("flat")
        xl.addLayout(self.res_head)
        self.tile_grid = QGridLayout()
        self.tile_grid.setSpacing(6)
        self.m_I = MetricTile("Photocurrent |I|", "—")
        self.m_gain = MetricTile("Gain I / I_unity", "—")
        self.m_peak = MetricTile("|S| peak at X", "—")
        self.m_ratio = MetricTile("S(+5) / S(−5)", "—")
        self.m_s0 = MetricTile("S(0) / max", "—")
        self.tiles = (self.m_I, self.m_gain, self.m_peak, self.m_ratio, self.m_s0)
        for m in self.tiles:        # MetricTile's Maximum height policy clips a hint that wraps to two lines
            m.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        xl.addLayout(self.tile_grid)
        self.summary = label("", wrap=True, selectable=True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.summary.hide()
        xl.addWidget(self.summary)
        self._layout_state = None
        self._wide = True
        self._relayout(True, False)
        self.fig_tabs = QTabWidget()
        self.fig_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)   # as dev_plot
        self.fig_tabs.setMinimumHeight(250)
        self.field = FieldView()
        self.fig_tabs.addTab(self.field, "Electric field")
        self.fig_tabs.setTabToolTip(0, "COMSOL-style map of |E|, E_x, E_z and the potential from solution.npz")
        self.views = {}
        for fn, title in FIGURES:
            v = ImageView()
            # the "name · W×H" caption must not add its width to the toolbar's minimum (keeps 1000 px windows usable)
            v.caption.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            self.views[fn] = v
            self.fig_tabs.addTab(v, title)
            self.fig_tabs.setTabToolTip(self.fig_tabs.count() - 1, FIGURE_TIPS[fn])
        xl.addWidget(self.fig_tabs, 1)
        self.tabs.addTab(res_tab, "Results")

        hist_tab = QWidget()
        hl = QVBoxLayout(hist_tab)
        hl.setContentsMargins(12, 12, 12, 12)
        hrow = QHBoxLayout()
        hrow.addWidget(label("Existing steady-state results (double-click to load)", "cardTitle"), 1)
        rb = button("Refresh", self.refresh_history)
        rb.setObjectName("flat")
        hrow.addWidget(rb)
        hl.addLayout(hrow)
        self.hist = QListWidget()
        self.hist.itemDoubleClicked.connect(self._on_hist)
        hl.addWidget(self.hist, 1)
        self.tabs.addTab(hist_tab, "History")
        split.addWidget(self.tabs)
        # follow the tab widget's width: a hidden tab page is not resized, and a stale wide layout there
        # would keep the whole page wider than the window
        self.tabs.installEventFilter(self)
        self.tabs.currentChanged.connect(self._on_tab)
        split.setStretchFactor(0, 0)          # extra width goes to the results; the form keeps its size
        split.setStretchFactor(1, 1)
        split.setSizes([430, 900])
        self.body.addWidget(split, 1)

        self.refresh_history()
        default = paths.RESULTS / "baseline"
        if (default / "summary.json").exists():
            self.show_result(default, switch=False)
        else:
            self._field_message("No result loaded yet. Run a solve — the coarse-grid quick mode takes about 30–60 s.")
        self._sync_quick()
        self._retheme()
        ThemeManager.instance().changed.connect(self._retheme)
        JobManager.instance().job_finished.connect(self._on_any_finished)
        QTimer.singleShot(0, self._update_device)       # first diagram after the page is shown

    def _retheme(self, *_):
        t = ThemeManager.instance().tokens
        self.run_btn.setIcon(icons.icon("play", t.on_accent))
        self.dev_warn.setStyleSheet(f"color: {t.warning};")
        self._update_summary()

    # ------------------------------------------------------------------ layout
    def eventFilter(self, obj, e):
        if obj is self.tabs and e.type() == QEvent.Type.Resize:
            self._relayout(obj.width() - 4 >= self.WIDE, obj.height() < self.SHORT)
        elif obj is self.dev_plot.canvas and e.type() == QEvent.Type.Resize and self._dev_side is not None:
            if self._side_by_side() != self._dev_side:
                self._dev_key = None            # panel arrangement changed: redraw
                self._dev_timer.start()
        return super().eventFilter(obj, e)

    def _relayout(self, wide: bool, short: bool):
        """One row of tiles / buttons beside the title when wide; 3 + 2 tiles and buttons below the title when narrow;
        a one-line summary instead of the tiles when the window is short."""
        if (wide, short) == self._layout_state:
            return
        self._layout_state = (wide, short)
        self._wide = wide
        for m in self.tiles:
            m.setVisible(not short)
        self.summary.setVisible(short)
        h, g = self.res_head, self.tile_grid
        head = (self.res_bar, self.btn_params, self.btn_explore, self.btn_folder)
        for w in (*head, *self.tiles):
            (h if w in head else g).removeWidget(w)
        for c in range(6):
            h.setColumnStretch(c, 0)
            g.setColumnStretch(c, 0)
        if wide:
            h.addWidget(self.res_bar, 0, 0)
            for i, b in enumerate((self.btn_params, self.btn_explore, self.btn_folder)):
                h.addWidget(b, 0, i + 1)
            h.setColumnStretch(0, 1)
        else:
            h.addWidget(self.res_bar, 0, 0, 1, 4)
            for i, b in enumerate((self.btn_params, self.btn_explore, self.btn_folder)):
                h.addWidget(b, 1, i)
            h.setColumnStretch(3, 1)
        cols = 5 if wide else 3
        for i, m in enumerate(self.tiles):
            g.addWidget(m, i // cols, i % cols)
            m.layout().setContentsMargins(*((12, 10, 12, 10) if wide else (12, 5, 12, 6)))    # compact: 2 rows
        for c in range(cols):
            g.setColumnStretch(c, 1)
        self._tile_hints()
        self.res_tab.layout().invalidate()       # the tab may be hidden: drop its cached minimum size now

    def _tile_hints(self):
        """Narrow layout: two rows of tiles already take the height, so the hints live only in the tooltips."""
        for m in self.tiles:
            m.setToolTip(m.property("tip") or m.hint.text())
            m.hint.setVisible(bool(self._wide) and bool(m.hint.text()))
        self.summary.setToolTip("\n".join(f"{m.lab.text()}: {m.property('tip') or m.hint.text() or '—'}"
                                          for m in self.tiles))

    def _update_summary(self):
        """The tiles' values as one wrapped line (short windows)."""
        if not hasattr(self, "summary"):
            return
        mu = ThemeManager.instance().tokens.muted
        nb = lambda t: t.replace(" ", "&nbsp;").replace("|", "|\u2060")     # wrap only between the items
        self.summary.setText("&nbsp;&nbsp;·&nbsp; ".join(
            f"<span style='color:{mu}'>{nb(m.lab.text())}</span>&nbsp;<b>{nb(m.val.text())}</b>" for m in self.tiles))

    # ------------------------------------------------------------------ device diagram
    def _on_tab(self, _i):
        if self.tabs.currentWidget() is self.dev_tab:
            self._update_device()

    def _update_device(self):
        """Redraw the cross-section from the form (only while the tab is visible, and only if it changed)."""
        v = self.form.values()
        notes = []
        bad = [self.form.rows[n].spec.label for n in self.form.invalid_fields() if n in DEVICE_KEYS]
        if bad:
            notes.append("Invalid values, drawn with their defaults: " + ", ".join(bad))
        notes += check_params(v)
        self.dev_warn.setText("\n".join(f"⚠ {n}" for n in notes))
        self.dev_warn.setVisible(bool(notes))
        if self.tabs.currentWidget() is not self.dev_tab:
            return
        key = tuple(repr(v.get(k)) for k in DEVICE_KEYS)
        if key == self._dev_key:
            return
        self._dev_key = key
        dev = device_from_params(v, alpha_cm=max(float(v.get("alpha_cm") or 0.0), 1e-3))   # α = 0: no 1/α division
        side = self._dev_side = self._side_by_side()

        def draw(fig):
            ax, az = draw_device(fig, dev)
            if not side:                        # narrow canvas: whole device on top, gap zoom below
                gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.1])
                ax.set_subplotspec(gs[0])
                az.set_subplotspec(gs[1])
        self.dev_plot.plot(draw)

    def _side_by_side(self) -> bool:
        c = self.dev_plot.canvas
        return c.width() >= self.SIDE_BY_SIDE * max(c.height(), 1)

    # ------------------------------------------------------------------ run
    def _toggle_quick(self, on):
        if on:
            self.form.set_values(QUICK)
        else:
            self.form.set_values({k: self.form.rows[k].spec.default for k in QUICK})

    def _sync_quick(self):
        """The checkbox mirrors the grid fields (editing dx_fine etc. by hand un-ticks it)."""
        q = is_quick(self.form.values())
        if self.quick.isChecked() != q:
            self.quick.blockSignals(True)
            self.quick.setChecked(q)
            self.quick.blockSignals(False)
        self.run_btn.setToolTip(f"Submit to the job queue ({'coarse grid: ≈ 30–60 s' if q else 'production grid: ≈ 6–7 min'})")

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder", str(paths.GUI_RUNS if paths.GUI_RUNS.exists()
                                                                              else paths.RESULTS))
        if d:
            self.out_edit.setText(paths.rel(d))

    def _output_dir(self, quick: bool) -> Path | None:
        """Validated output folder (asks before overwriting); None = cancelled."""
        txt = self.out_edit.text().strip()
        if not txt:
            return paths.new_run_dir("steady_quick" if quick else "steady")
        out = paths.absolute(Path(txt).expanduser())
        r = out.resolve()
        if r in (paths.ROOT.resolve(), paths.RESULTS.resolve(), paths.GUI_RUNS.resolve()) or r in paths.ROOT.resolve().parents:
            error_box(self, "Unsuitable output folder", f"{txt} is the project or results root. Choose (or type) a "
                      "dedicated subfolder, or leave the field empty to create results/gui_runs/steady_<time> "
                      "automatically.")
            return None
        if out.exists() and not out.is_dir():
            error_box(self, "Output path is not a folder", f"{txt} is a file. Enter a folder path, or leave the field "
                      "empty to create a new folder automatically.")
            return None
        busy = [j for j in JobManager.instance().active() if j.out and paths.absolute(j.out).resolve() == r]
        if busy:
            error_box(self, "Output folder in use", f"The job “{busy[0].title}” ({STATUS_TEXT.get(busy[0].status, '')}) "
                      f"is writing to {paths.rel(out)}. Choose another folder, or leave the field empty to create a "
                      "new one.")
            return None
        if out.is_dir() and any(out.iterdir()):
            committed = r.is_relative_to(paths.RESULTS.resolve()) and not r.is_relative_to(paths.GUI_RUNS.resolve())
            msg = (f"{paths.rel(out)} already contains files. The run will overwrite its figures, CSV files, "
                   "summary.json, params.json and solution.npz.")
            if committed:
                msg += "\n\nNote: this result folder is committed to the repository; once overwritten it can only be " \
                       "restored with git."
            if (out / "summary.json").exists():
                msg += "\n\nThe folder already holds a complete steady-state result."
            r2 = QMessageBox.question(self, "Overwrite existing result?", msg + "\n\nContinue?",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
            if r2 != QMessageBox.StandardButton.Yes:
                return None
        return out

    def run(self):
        if not self.form.is_valid():
            rows = self.form.rows
            lines = []
            for n in self.form.invalid_fields():
                s = rows[n].spec
                lims = ([] if s.minimum is None else [f"≥ {s.minimum:g}"]) + ([] if s.maximum is None else [f"≤ {s.maximum:g}"])
                rng = f" (allowed: {', '.join(lims)})" if lims else ""
                lines.append(f"• {s.label} ({n}) = {rows[n].editor.text() if hasattr(rows[n].editor, 'text') else '?'}{rng}")
            error_box(self, "Invalid parameters", "These parameters cannot be parsed or are out of range:\n" + "\n".join(lines))
            return
        v = self.form.values()
        bad = check_params(v)
        if bad:
            error_box(self, "Invalid parameter combination", "Please fix first:\n" + "\n".join(f"• {b}" for b in bad))
            return
        if self.job is not None and self.job.active:
            r = QMessageBox.question(self, "Steady-state job already active",
                                     f"“{self.job.title}” is still {STATUS_TEXT.get(self.job.status, '').lower()}.\n"
                                     "Submit another job? It will wait in the queue until the earlier jobs finish.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes:
                return
        quick = is_quick(v)
        out = self._output_dir(quick)
        if out is None:
            return
        ov = self.form.overrides()
        desc = ", ".join(f"{k}={x}" for k, x in ov.items() if not (quick and k in QUICK)) or "default parameters"
        if len(desc) > 60:
            desc = desc[:58].rstrip(", ") + " …"
        title = ("Steady state (coarse) · " if quick else "Steady state · ") + desc
        self.job = JobManager.instance().submit(
            title, "steady", "scripts/run_baseline.py", "main", kwargs=dict(out=paths.rel(out), **ov), out=out,
            progress={"type": "steady"})
        self.panel.attach(self.job)
        self.panel.title.setToolTip(", ".join(f"{k}={x}" for k, x in ov.items()) or "default parameters")
        self.tabs.setCurrentWidget(self.run_tab)

    def _on_finished(self, job):
        if job is self.job and job.status == FINISHED and job.out:
            self.show_result(paths.absolute(job.out))
            self.refresh_history()

    def _on_any_finished(self, job):
        # runs not attached to the panel (queued behind another one, re-run from the job center) still show up
        if job.kind == "steady" and job is not self.job and job.status == FINISHED:
            self.refresh_history()

    # ------------------------------------------------------------------ results
    def show_result(self, d, switch=True):
        d = Path(d)
        if d.is_file():
            d = d.parent
        self.result_dir = d
        try:
            summ = json.loads((d / "summary.json").read_text())
            if not isinstance(summ, dict):
                summ = {}
        except (OSError, ValueError):
            summ = {}
        pr = result_params(d, summ)
        m = steady_metrics(summ)
        try:
            V = float(pr.get("V_bias", 25.0))
        except (TypeError, ValueError):
            V = float("nan")
        side = pr.get("anode_side", "-x")
        # X = -x_tip, so X > 0 puts the tip on the -x side; that is the anode side when the -x electrode is positive
        anode_at_pos_X = None if not np.isfinite(V) or V == 0 else ((side == "-x") == (V > 0))
        ug = m.get("unity")
        same = anode_at_pos_X in (None, True)          # same polarity as the measured +25 V scan
        sd = "anode" if anode_at_pos_X else "cathode"
        hints = {   # tile: (short hint under the value, tooltip)
            self.m_I: ("measured 0.6 mA", "Measured photocurrent ≈ 0.6 mA (+25 V)"),
            self.m_gain: (f"I_unity {ug:.2f} mA" if ug else "",
                          "Gain = |I| / I_unity, where I_unity is the current for one charge per absorbed photon"
                          if ug else ""),
            self.m_peak: ("measured +5 µm" if same else "mirrored: −5 µm",
                          "The measured +25 V scan peaks at X = +5 µm (tip on the anode side)" +
                          ("" if same else "; this result has the opposite polarity, so compare with X = −5 µm")),
            self.m_ratio: ({None: "no bias", True: "> 1: anode side", False: "> 1: cathode side"}
                           [anode_at_pos_X],
                           "No bias, so there is no anode or cathode side" if anode_at_pos_X is None else
                           f"S(+5)/S(−5) > 1 means a stronger signal with the tip on the {sd} side "
                           f"(X = spot − tip > 0 is the {sd} side)"),
            self.m_s0: ("measured 0.24", "Measured at +25 V: S(0) / max = 0.24"),
        }
        for tile, key, f in ((self.m_I, "I", "{:.3f} mA"), (self.m_gain, "gain", "{:.2f}"), (self.m_peak, "peak_X", "{:+.1f} µm"),
                             (self.m_ratio, "ratio5", "{:.2f}"), (self.m_s0, "S0", "{:.2f}")):
            tile.set(_fmt(m.get(key), f), hints[tile][0])
            tile.setProperty("tip", hints[tile][1])
        self._tile_hints()
        self._update_summary()
        conv = m.get("converged")
        if not d.is_dir():
            state, obj, tip = "Folder not found", "badgeErr", str(d)
        elif not summ:
            state, obj, tip = ("Incomplete: no summary.json", "badgeWarn",
                               "The run did not finish, was cancelled or failed (no summary.json in the folder)")
        elif conv is None:
            state, obj, tip = "", "badge", ""
        else:
            state, obj, tip = (("Converged", "badgeOk", "The steady-state Newton converged in both the dark and the "
                                "illuminated stage") if conv else
                               ("Not fully converged", "badgeWarn", "In at least one stage the steady-state Newton had "
                                "not converged when t_end or the step limit was reached"))
        self.res_title.setText(paths.rel(d))
        self.res_state.setText(state)
        self.res_state.setToolTip(tip)
        self.res_state.setObjectName(obj)
        self.res_state.setVisible(bool(state))
        repolish(self.res_state)
        self.res_grid.setVisible(bool(summ) and is_quick(pr))
        for fn, _ in FIGURES:
            if (d / fn).exists():
                self.views[fn].load(d / fn)
            else:
                self.views[fn].clear(f"This result has no {fn}")
        sol = d / "solution.npz"
        has_sol = sol.exists()
        self.btn_explore.setEnabled(has_sol)
        self.btn_explore.setToolTip("Inspect ψ, E, n, p and EL2⁺ interactively in the Solution explorer" if has_sol else
                                    "This result has no solution.npz (solution files are not committed to the repository)")
        self.btn_params.setEnabled(bool(pr))
        self._field_token += 1
        if has_sol:
            self._field_message("Loading the electric field from solution.npz …")
            tok = self._field_token
            run_async(load_steady_fields, sol, on_done=lambda data: self._on_fields(tok, data),
                      on_error=lambda msg: self._on_fields_error(tok, msg))
        else:
            self._field_message(no_field_message(d, bool(summ)))
        if switch:
            self.tabs.setCurrentWidget(self.res_tab)

    def _field_message(self, msg: str):
        self.field.set_fields(None, msg)

    def _on_fields(self, tok, data):
        if tok != self._field_token:          # a newer result was loaded meanwhile
            return
        self.field.set_fields(data)

    def _on_fields_error(self, tok, msg):
        if tok != self._field_token:
            return
        first = (msg or "").strip().splitlines()[0] if (msg or "").strip() else "unknown error"
        self._field_message(f"Could not read solution.npz:\n{first}")

    def _load_result_params(self):
        if not self.result_dir:
            return
        d = result_params(self.result_dir)
        if not d:
            error_box(self, "Cannot load parameters", f"{paths.rel(self.result_dir)} has no readable params.json "
                      "or summary.json.")
            return
        self.form.set_values({k: v for k, v in d.items() if k in self.form.rows}, reset_others=True)

    def _open_explorer(self):
        if self.result_dir and (self.result_dir / "solution.npz").exists():
            nav.navigate("explorer", path=str(self.result_dir / "solution.npz"))

    def _on_hist(self, it):
        d = it.data(Qt.ItemDataRole.UserRole)
        if d:
            self.show_result(d)

    def refresh_history(self):
        cur = self.hist.currentItem()
        keep = cur.data(Qt.ItemDataRole.UserRole) if cur is not None else None
        self.hist.clear()

        def mtime(p):
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0

        gui = sorted((d for d in paths.GUI_RUNS.glob("steady*") if (d / "summary.json").exists()),
                     key=lambda d: mtime(d / "summary.json"), reverse=True)
        cands = [d for d in [paths.RESULTS / "baseline"] if (d / "summary.json").exists()] + gui
        cands += [s.parent for s in sorted(paths.RESULTS.glob("sweeps/*/*/summary.json"))]
        for d in cands:
            try:
                summ = json.loads((d / "summary.json").read_text())
                m = steady_metrics(summ)
            except (OSError, ValueError, AttributeError, TypeError):
                summ, m = {}, {}
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime(d / "summary.json")))
            tags = ("  ·  coarse grid" if is_quick(result_params(d, summ)) else "") + \
                   ("  ·  field map" if (d / "solution.npz").exists() else "") + \
                   ("" if m.get("converged", True) else "  ·  ⚠ not fully converged")
            txt = (f"{paths.rel(d)}{tags}\n    I = {_fmt(m.get('I'), '{:.3f} mA')} · peak X = "
                   f"{_fmt(m.get('peak_X'), '{:+.1f} µm')} · {when}")
            it = QListWidgetItem(txt)
            it.setData(Qt.ItemDataRole.UserRole, str(d))
            it.setToolTip(str(d) + ("\nHas solution.npz: the Electric field tab shows its field map"
                                    if (d / "solution.npz").exists() else ""))
            self.hist.addItem(it)
            if keep == str(d):
                self.hist.setCurrentItem(it)
        if not cands:
            it = QListWidgetItem("No steady-state results yet. Results appear here after you run a solve.")
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.hist.addItem(it)

    def on_show(self):
        self.refresh_history()          # cheap; picks up runs finished from the job center or the command line

    def open(self, path=None, **_):
        if path:
            self.show_result(path)
