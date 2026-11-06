"""Pump-steering page: where the pump spot goes when the KS1-Z8 steering mirror outside the AFM moves.

Front end of ltemsim/steering.py, the adapter of the vendored ltem-steering-module (exact chief-ray trace + scalar
diffraction; the module's README holds the physics and the conventions).  The chief-ray trace takes ~1 ms and runs on
the GUI thread at every actuator change.  The diffraction spots, the peak-intensity curves, the shape gallery and the
link table run in one background thread (the core keeps its caches in plain dicts) with "latest wins" coalescing.
The 3-D beam-path model is the user's reference page, embedded by widgets/steering3d.py.
"""
from __future__ import annotations

import csv
import json
import math
import re

from PySide6.QtCore import QEvent, QObject, QSettings, QThreadPool, Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QBoxLayout, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLineEdit, QMenu, QScrollArea,
                               QSizePolicy, QSlider, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                               QToolButton, QVBoxLayout, QWidget)
from shiboken6 import isValid

from ltemsim import steering
from ltemsim.steering import SteeringParams

from .. import nav, paths
from ..paramspec import steering_specs
from ..theme import ThemeManager
from ..widgets import steering_views as views
from ..widgets.common import Page, button, card, error_box, hline, label, open_external
from ..widgets.mpl import PlotWidget
from ..widgets.paramform import ParamForm
from ..widgets.steering3d import Steering3DView
from ..workers import run_async

SPIN_UM = 500.0                 # spin-box range ±500 µm (as the reference page)
SLIDER_UM = 20.0                # slider range ±20 µm in 0.1 µm steps; values beyond it pin the slider at its end
STEPS = (-1.0, -0.2, 0.2, 1.0)
CURVE_TRAVELS = tuple(float(t) for t in range(-20, 21))        # peak-intensity curves, 'trend' quality
GALLERY_TRAVELS = (-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0)
TABLE_TRAVELS = (-20, -10, -5, -2, 0, 2, 5, 10, 20)             # data table under the curves (as the reference page)
X_DEFAULT = "-7.5, -5, -2.5, 0, 2.5, 5, 7.5"                    # the beam positions of the tip-position chain (X_LIST)
MODES = (("lower", "Lower actuator only (recommended)"), ("both", "Both actuators (Y = 0)"),
         ("upper", "Upper actuator only"))
_MODE_NAMES = {k: v.split(" (")[0].lower() for k, v in MODES}    # link status: "lower actuator only", ...
VIEWS = (("top", "Overview (top)"), ("inside", "Inside"), ("side", "Side"), ("mirror", "Mirror"))
GAINS = (10, 100, 400)
ACTUATORS = (("lower", "Lower · horizontal"), ("upper", "Upper · vertical"))
TRAIL_MAX = 200                 # trail points kept
TRAIL_RETRACE = 60              # at most this many are re-traced (evenly thinned) when the parameters change
FINE_MS = 250                   # the 'fine' spot follows this long after the last slider movement
REDRAW_MS = 30                  # plot redraws are throttled to this interval
RESIZE_MS = 150                 # a resized plot is redrawn this long after its last resize (layouts depend on the size)
SAVE_MS = 600

EXPLAIN = ("Per the Thorlabs KS1-Z8 drawing, seen from the actuator side the manual knob (the pivot) is at the lower "
           "left. The <b>lower</b> actuator, level with it, only tilts the mirror horizontally; the <b>upper</b> one, "
           "directly above it, only tilts it vertically. Values are the push-rod extension, i.e. the change of the "
           "KDC101 reading (0.001 mm = 1 µm).")
LINK_TEXT = ("The tip-position chain moves the pump spot to each offset X along the bias axis and assumes the same "
             "spot at every X. In the experiment the steering mirror moves it: this table gives the actuator travels "
             "for each target X (exact chief-ray trace) and the spot that really lands there — off the bias axis "
             "(landed Y), defocused and dimmer.")
LAB_ROWS = (("x", "Left/right (lab X)"), ("y", "Front/back (lab Y)"), ("tot", "Total"),
            ("plane", "Along · across plane of incidence"), ("port", "Beam shift at the port"),
            ("defocus", "Focus off the sample"), ("time", "Pump arrival-time change"))
SIM_ROWS = (("X", "X along the bias axis"), ("Y", "Y across it"))
SHAPE_ROWS = (("fwhm", "FWHM (long × short)"), ("ang", "Long axis vs plane of incidence"),
              ("area", "Half-max area (original = 1)"), ("peak", "Peak intensity (original = 100 %)"),
              ("off", "Brightest point vs chief-ray hit"))
LINK_COLS = ("X (µm)", "Lower (µm)", "Lower (counts)", "Upper (µm)", "Upper (counts)", "Landed Y (µm)",
             "Defocus (µm)", "Δ arrival (fs)", "Peak (%)", "FWHM (µm)")


def settings() -> QSettings:
    """Where the page keeps its state (tests point this at a scratch file)."""
    return QSettings(paths.APP_ID, paths.APP_ID)


def _t():
    return ThemeManager.instance().tokens


# ----------------------------------------------------------------------------- text helpers
def _m(s: str) -> str:
    """Typographic minus signs."""
    return s.replace("-", "−")


def lr(v: float) -> str:
    return "left" if v < 0 else "right"


def fb(v: float) -> str:
    return "front" if v < 0 else "back"


def fmt_abs(v: float) -> str:
    """|v| with 2 decimals below 10, 1 below 100, none above (as the reference page)."""
    a = abs(v)
    return f"{a:.{2 if a < 10 else 1 if a < 100 else 0}f}"


def signed(v: float, d: int = 2) -> str:
    return f"{0:.{d}f}" if abs(v) < 0.5 * 10 ** -d else _m(f"{v:+.{d}f}")


def lab_readouts(st) -> dict:
    """Readout texts of one Steering (lab frame in words, simulation frame signed)."""
    dx, dy = st.spot_um
    py, pz = st.port_um
    port = []
    if abs(py) >= 0.05:
        port.append(f"{fb(py)} {fmt_abs(py)}")
    if abs(pz) >= 0.05:
        port.append(f"{'down' if pz < 0 else 'up'} {fmt_abs(pz)}")
    return {
        "x": "0.00 µm" if abs(dx) < 0.005 else f"{lr(dx)} {fmt_abs(dx)} µm",
        "y": "0.00 µm" if abs(dy) < 0.005 else f"{fb(dy)} {fmt_abs(dy)} µm",
        "tot": f"{fmt_abs(st.shift_um)} µm",
        "plane": f"{fmt_abs(st.along_um)} · {fmt_abs(st.across_um)} µm",
        "port": " · ".join(port) + " µm" if port else "0 µm",
        "defocus": f"≈ {fmt_abs(st.defocus_um)} µm",
        "time": f"≈ {abs(st.arrival_fs):.1f} fs",
        "X": signed(st.X_um) + " µm",
        "Y": signed(st.Y_um) + " µm",
    }


def shape_readouts(s: dict, ref: dict, plane) -> dict:
    """Readout texts of a spot against the aligned one; plane = horizontal unit vector of the plane of incidence."""
    inc = math.atan2(plane[1], plane[0]) if plane else -math.pi / 4
    d = math.degrees(s["ang"] - inc)
    d = ((d + 90) % 180 + 180) % 180 - 90                   # an axis: fold to (-90, 90]
    ox, oy = s["peakAt"]
    rel = s.get("rel", s["peak"])
    return {
        "fwhm": f"{s['fwhmL']:.2f} × {s['fwhmS']:.2f} µm",
        "ang": f"{abs(d):.0f}°",
        "area": f"× {s['area'] / ref['area']:.2f}" if ref.get("area") else "—",
        "peak": f"{rel * 100:.0f} %",
        "off": ("< 0.05 µm" if math.hypot(ox, oy) < 0.05
                else f"{lr(ox)} {abs(ox):.2f} · {fb(oy)} {abs(oy):.2f} µm"),
    }


def per_um_text(per, curve_rows=None) -> str:
    """'Per µm → spot …' line of one actuator (rich text), with the peak at ±5 µm when the curve is known."""
    px, py = per
    mag = math.hypot(px, py)
    txt = (f"Per µm → spot <b>{mag:.2f} µm</b> ({lr(px)} {abs(px):.2f} · {fb(py)} {abs(py):.2f})"
           f" · 0.2 µm step → {mag * 200:.0f} nm")
    if curve_rows:
        p5 = [r["rel"] for r in curve_rows if abs(abs(r["travel_um"]) - 5.0) < 1e-9 and r.get("rel") is not None]
        if len(p5) == 2:
            txt += f" · at ±5 µm, peak ≈ <b>{sum(p5) / 2 * 100:.0f} %</b>"
    return txt


def counts_text(travel_um: float) -> str:
    """KDC101 encoder counts of a push-rod travel, signed, with thousands separators."""
    c = steering.kdc101_counts(travel_um)
    return "0" if abs(c) < 0.5 else _m(f"{c:+,.0f}")


def kdc_text(travel_um: float) -> str:
    return f"KDC101 change: {signed(travel_um / 1000, 5)} mm · {counts_text(travel_um)} counts"


def parse_xs(text: str) -> tuple:
    """'-7.5, -5, 0 2.5' -> (-7.5, -5.0, 0.0, 2.5); raises ValueError."""
    parts = [p for p in re.split(r"[,;\s]+", text.replace("−", "-").strip()) if p]
    if not parts:
        raise ValueError("enter at least one X value")
    xs = tuple(float(p) for p in parts)
    if len(xs) > 41:
        raise ValueError("at most 41 X values")
    if any(not math.isfinite(x) or abs(x) > 200 for x in xs):
        raise ValueError("X values must lie within ±200 µm")
    return xs


def read_measured(path=None) -> list:
    """[(bias_V, X_um, rms normalised to 1 per bias)] from data/measured_tip_scan.csv (empty if unreadable)."""
    data: dict[float, list] = {}
    try:
        with open(path or paths.MEASURED_CSV, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    data.setdefault(float(r["bias_V"]), []).append((float(r["X_um"]), float(r["rms"])))
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return []
    out = []
    for v, pts in sorted(data.items()):
        top = max((abs(r) for _, r in pts), default=0.0)
        if top > 0:
            out += [(v, x, r / top) for x, r in sorted(pts)]
    return out


def _key(sp: SteeringParams) -> str:
    return json.dumps(sp.to_dict(), sort_keys=True)


def _form_values(sp: SteeringParams) -> dict:
    """SteeringParams -> ParamForm values (the ±1 actuator sides are string choices)."""
    d = sp.to_dict()
    for k in ("yawSide", "pitchSide"):
        d[k] = "1" if d[k] >= 0 else "-1"
    return d


def beyond_range(row: dict) -> bool:
    """A link-table row whose travels the actuator controls cannot set (the spin boxes stop at ±SPIN_UM)."""
    return "error" not in row and max(abs(row["lower_um"]), abs(row["upper_um"])) > SPIN_UM


def _float(v, default=0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


# ----------------------------------------------------------------------------- background calls (worker thread)
def compute_spot(sp, lower, upper, quality, need_ref):
    return {"D": steering.beam_diameter_mm(sp), "ref": steering.reference_spot(sp) if need_ref else None,
            "cur": steering.spot(sp, lower, upper, quality)}


def compute_curves(sp):
    return {a: steering.peak_curve(sp, a, CURVE_TRAVELS, "trend") for a, _ in ACTUATORS}


def compute_gallery(sp):
    out = {"lower": [], "upper": []}
    for t in GALLERY_TRAVELS:
        for a, (lo, up) in (("lower", (t, 0.0)), ("upper", (0.0, t))):
            try:
                s = steering.spot(sp, lo, up, "trend")
            except (ValueError, ArithmeticError):
                s = None
            out[a].append((t, s))
    return out


def compute_link(sp, xs, mode):
    """plan_offsets row by row, so one X the actuators cannot reach does not lose the others."""
    rows = []
    for X in xs:
        try:
            rows += steering.plan_offsets(sp, [X], mode, "trend")
        except (ValueError, ArithmeticError, FloatingPointError) as e:
            rows.append({"X_um": float(X), "error": str(e)})
    return rows


class _Failed:
    """Stands in for the result of a background call that raised."""

    def __init__(self, msg: str):
        self.msg = (msg or "").split("\n\n", 1)[0]       # run_async appends the traceback after a blank line


class _CoreQueue(QObject):
    """One background thread for the optics core, with "latest wins" coalescing.

    Every call runs in a private QThreadPool with a single thread, so the core's caches (auto D, aligned spot) are
    used by one call at a time.  Each kind of request has at most one pending entry: a newer request replaces the
    one still waiting.  Pending kinds start in ORDER (the current spot first).
    """
    ORDER = ("spot", "curves", "gallery", "link")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.pending: dict = {}
        self.busy = False
        self.closed = False

    def submit(self, kind, fn, args, on_done):
        if self.closed:
            return
        self.pending[kind] = (fn, args, on_done)
        self._next()

    def idle(self) -> bool:
        return not self.busy and not self.pending

    def close(self):
        self.closed = True
        self.pending.clear()

    def _next(self):
        if self.busy or self.closed:
            return
        kind = next((k for k in self.ORDER if k in self.pending), None)
        if kind is None:
            return
        fn, args, cb = self.pending.pop(kind)
        self.busy = True
        run_async(fn, *args, pool=self.pool, on_done=lambda r: self._finish(cb, r),
                  on_error=lambda m: self._finish(cb, _Failed(m)))

    def _finish(self, cb, result):
        self.busy = False
        if self.closed or not isValid(self):        # page gone: never touch its widgets
            return
        try:
            cb(result)
        finally:
            self._next()


# ----------------------------------------------------------------------------- actuator control
class _Actuator(QWidget):
    """Spin box, slider and step buttons of one Z812, kept in sync.

    changed(value_um, dragging) fires on every change; released() when the slider is let go.
    """
    changed = Signal(float, bool)
    released = Signal()

    def __init__(self, key: str, title: str, parent=None):
        super().__init__(parent)
        self.key = key
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(5)
        head = QVBoxLayout()
        head.setSpacing(0)
        self.name = label("● " + title)
        self.hint = label("", "muted", wrap=True)
        head.addWidget(self.name)
        head.addWidget(self.hint)
        lay.addLayout(head)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(-SPIN_UM, SPIN_UM)
        self.spin.setDecimals(2)
        self.spin.setSingleStep(0.1)
        self.spin.setSuffix(" µm")
        self.spin.setAccelerated(True)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.spin.setFixedWidth(116)
        self.spin.setToolTip(f"{title}: push-rod extension = change of the KDC101 reading (0.001 mm = 1 µm), "
                             f"±{SPIN_UM:g} µm. + = the rod extends.")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        n = round(SLIDER_UM * 10)
        self.slider.setRange(-n, n)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        self.slider.setToolTip(f"±{SLIDER_UM:g} µm in 0.1 µm steps; larger travels (spin box) pin it at the end")
        row.addWidget(self.spin)
        row.addWidget(self.slider, 1)
        lay.addLayout(row)

        steps = QHBoxLayout()
        steps.setSpacing(4)
        for d in STEPS:
            b = button(_m(f"{d:+g}"), lambda _=False, d=d: self.step(d),
                       tooltip=f"{'Extend' if d > 0 else 'Retract'} the push rod by {abs(d):g} µm")
            b.setStyleSheet("padding: 3px 4px;")
            steps.addWidget(b, 1)
        z = button("Zero", lambda: self.spin.setValue(0.0), tooltip="Back to the aligned position (0 µm)")
        z.setStyleSheet("padding: 3px 8px;")
        steps.addWidget(z, 1)
        lay.addLayout(steps)
        self.per = label("", "muted", wrap=True)
        self.per.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.per)
        self.kdc = label(kdc_text(0.0), "muted", selectable=True)
        self.kdc.setToolTip("Change of the KDC101 reading for this travel (Z812 encoder: 1 mm = 34,555 counts, "
                            "≈ 29 nm per count)")
        lay.addWidget(self.kdc)

        self.spin.valueChanged.connect(self._on_spin)
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.sliderReleased.connect(self.released)

    def value(self) -> float:
        return float(self.spin.value())

    def set_value(self, v: float):
        """Set without emitting `changed` (the page updates once for several actuators)."""
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self._sync(self.spin.value())

    def step(self, d: float):
        self.spin.setValue(round(self.spin.value() + d, 2))

    def set_color(self, c: str, t):
        self.name.setStyleSheet(f"color: {c}; font-weight: 650;")
        self.slider.setStyleSheet(f"""
QSlider::groove:horizontal {{ height: 4px; background: {t.border}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {c}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}""")

    def _sync(self, v: float):
        n = self.slider.maximum()
        self.slider.blockSignals(True)
        self.slider.setValue(max(-n, min(n, round(v * 10))))
        self.slider.blockSignals(False)
        self.kdc.setText(kdc_text(v))

    def _on_spin(self, v):
        self._sync(v)
        self.changed.emit(float(v), self.slider.isSliderDown())

    def _on_slider(self, i):
        self.spin.setValue(i / 10)


# ----------------------------------------------------------------------------- page
class SteeringPage(Page):
    title = "Pump steering"
    subtitle = ("Where the pump spot goes when the KS1-Z8 steering mirror outside the AFM moves "
                "(exact chief-ray trace + scalar diffraction, module ltem-steering-module).")
    LEFT_W = 360            # control column (the parameter form needs ~330 px without a horizontal scroll bar)
    SPOT_SIDE_PX = 640      # tab width from which the simulation-frame readouts sit beside the lab-frame ones
    SHAPE_SIDE_PX = 820     # tab width from which the spot-shape tab shows the spot beside the curves
    SHAPE_STACKED_H = 640   # height of the stacked (narrow) spot-shape tab; it scrolls in shorter windows
    SHAPE_TABLE_H = 580     # least height of the side-by-side spot-shape tab with the data table open
    TABLE_H = 146           # data table under the curves
    LINK_MIN_H = 640        # least height of the link tab (table + plot); it scrolls in shorter windows

    def __init__(self):
        super().__init__()
        self.sp = SteeringParams()
        self._pkey = None
        self.lower = self.upper = 0.0
        self.st = None              # Steering at the current travels (None: the chief ray misses)
        self.st0 = None             # Steering at (0, 0): per-µm coefficients, plane of incidence, hints
        self.st_error = ""
        self.trail = []             # [(lower, upper, spot_xy, port_yz)], oldest first
        self.D = None               # resolved beam diameter (mm) of the current parameters
        self.ref = None             # aligned spot of the current parameters
        self.cur = None             # newest current spot (lags behind the actuators while they move)
        self.cur_at = None          # (lower, upper, quality) of self.cur
        self.spot_error = ""
        self.curves = None          # {'lower': rows, 'upper': rows} | _Failed | None (computing)
        self.gallery = None         # {'lower': [(travel, spot)], 'upper': [...]} | _Failed | None
        self.link_rows = None       # plan_offsets rows | _Failed | None
        self.link_error = ""
        self._link_key = None
        self._link_for = None       # (SteeringParams, mode) the link rows were computed for
        self._seq = 0               # spot requests are numbered; a result older than the one shown is dropped
        self._shown = 0
        self._dirty = set()
        self._closed = False
        self._measured = None
        self._shape_wide = self._spot_wide = None
        self.queue = _CoreQueue(self)

        self.header.add_action(button("Reference page ↗", lambda: open_external(steering.REFERENCE_PAGE), "flat",
                                      "Open the module's interactive bilingual page (ltem-steering.html) in the "
                                      "default browser"))
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(self._build_left())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_spot_tab(), "Spot")
        self.tabs.addTab(self._build_shape_tab(), "Spot shape")
        self.tabs.addTab(self._build_setup_tab(), "Setup")
        self.tab3d = QWidget()
        l3 = QVBoxLayout(self.tab3d)
        l3.setContentsMargins(8, 8, 8, 8)
        self.view3d = Steering3DView()
        l3.addWidget(self.view3d)
        self.tabs.addTab(self.tab3d, "3D view")
        self.tabs.addTab(self._build_link_tab(), "Link to simulation")
        self.tabs.installEventFilter(self)
        row.addWidget(self.tabs, 1)
        self.body.addLayout(row, 1)
        self.plots = {"port": self.plot_port, "sample": self.plot_sample, "map": self.plot_map,
                      "curves": self.plot_curves, "gallery": self.plot_gallery, "schematic": self.plot_schem,
                      "link": self.plot_link}

        self._redraw_timer = self._timer(REDRAW_MS, self._redraw)
        self._fine_timer = self._timer(FINE_MS, lambda: self._request_spot("fine"))
        self._param_timer = self._timer(200, self._apply_params)
        self._link_timer = self._timer(400, self._request_link)
        self._save_timer = self._timer(SAVE_MS, self._save)
        self._resize_timer = self._timer(RESIZE_MS, self._redraw_resized)
        self._resized = set()
        self._drawn_size = {}       # plot name -> widget size of its last drawing
        self._plot_name = {p: n for n, p in self.plots.items()}
        for p in self.plots.values():
            p.installEventFilter(self)

        self._restore()
        self.form.changed.connect(self._on_form)
        for a in self.act.values():
            a.changed.connect(lambda v, dragging, k=a.key: self._on_actuator(k, v, dragging))
            a.released.connect(self._on_released)
        self.tabs.currentChanged.connect(self._on_tab)
        self.view_combo.currentIndexChanged.connect(lambda *_: self._changed_view())
        self.gain_combo.currentIndexChanged.connect(lambda *_: self._changed_view())
        self.x_edit.textChanged.connect(lambda *_: self._link_timer.start())
        self.mode_combo.currentIndexChanged.connect(lambda *_: self._request_link())
        self.measured_cb.toggled.connect(lambda *_: (self._mark("link"), self._save_timer.start()))
        ThemeManager.instance().changed.connect(self._retheme)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown)
        self._retheme()
        self._sync_d_field()
        try:
            self._apply_params(force=True)
        except Exception:                       # a saved state the page cannot use must not break it for good
            self.form.reset_all()
            self._sync_d_field()
            self._apply_params(force=True)
            self.params_warn.setText("⚠ The saved parameters could not be used; back to the defaults.")
            self.params_warn.show()

    def _timer(self, ms, slot) -> QTimer:
        tm = QTimer(self)
        tm.setSingleShot(True)
        tm.setInterval(ms)
        tm.timeout.connect(slot)
        return tm

    # ------------------------------------------------------------------ layout
    def _build_left(self) -> QWidget:
        inner = QWidget()
        inner.setObjectName("page")
        ll = QVBoxLayout(inner)
        ll.setContentsMargins(0, 0, 8, 0)
        ll.setSpacing(10)
        c = card()
        cl = QVBoxLayout(c)
        cl.setContentsMargins(14, 12, 14, 12)
        cl.setSpacing(8)
        cl.addWidget(label("Two actuators", "cardTitle"))
        ex = label(EXPLAIN, "muted", wrap=True)
        ex.setTextFormat(Qt.TextFormat.RichText)
        cl.addWidget(ex)
        self.act = {}
        for key, title in ACTUATORS:
            cl.addWidget(hline())
            a = self.act[key] = _Actuator(key, title)
            cl.addWidget(a)
        cl.addWidget(hline())
        br = QHBoxLayout()
        br.addWidget(button("Zero both", lambda: self._set_travels(0.0, 0.0),
                            tooltip="Both actuators back to the aligned position"))
        br.addWidget(button("Clear trail", self._clear_trail,
                            tooltip="Forget the path drawn in the two scopes and in the 3D view"))
        br.addStretch(1)
        cl.addLayout(br)
        ll.addWidget(c)

        ph = QHBoxLayout()
        ph.addWidget(label("Instrument and beam parameters", "cardTitle"), 1)
        ll.addLayout(ph)
        self.params_warn = label("", wrap=True)
        self.params_warn.hide()
        ll.addWidget(self.params_warn)
        self.form = ParamForm(steering_specs(), scroll=False)
        ll.addWidget(self.form)
        ll.addStretch(1)

        area = self._scrolled(inner)
        area.setFixedWidth(self.LEFT_W)
        return area

    @staticmethod
    def _scrolled(w: QWidget) -> QScrollArea:
        """w in a frameless scroll area that scrolls vertically only."""
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setWidget(w)
        return area

    def _plot(self, toolbar=False, min_h=160, min_w=170, name="figure.png") -> PlotWidget:
        p = PlotWidget(toolbar=toolbar, figsize=(5, 4))
        # the canvas' size hint must not set the page size (it would scroll in small windows)
        p.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        p.setMinimumSize(min_w, min_h)
        if not toolbar:                     # no toolbar: save from the context menu
            c = p.canvas
            c.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

            def menu(pos, p=p, c=c):
                m = QMenu(c)
                m.addAction("Save figure…", lambda: p.save_dialog(name))
                m.exec(c.mapToGlobal(pos))
            c.customContextMenuRequested.connect(menu)
        return p

    def _build_spot_tab(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(10, 10, 10, 8)
        vl.setSpacing(6)
        self.warn = label("", wrap=True)
        self.warn.hide()
        vl.addWidget(self.warn)
        row = QHBoxLayout()
        row.setSpacing(10)
        for attr, title, sub, fn in (("plot_port", "Beam at the port", "outside the box, facing the port", "port.png"),
                                     ("plot_sample", "Spot on the sample", "top view", "spot_on_sample.png")):
            col = QVBoxLayout()
            col.setSpacing(2)
            cap = label(f"<b>{title}</b> · {sub}", "muted", wrap=True)
            cap.setTextFormat(Qt.TextFormat.RichText)
            cap.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            col.addWidget(cap)
            p = self._plot(min_h=170, name=fn)
            setattr(self, attr, p)
            col.addWidget(p, 1)
            row.addLayout(col, 1)
        vl.addLayout(row, 1)

        grid = self._ro_grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(3)
        self.ro = {}
        for i, (k, text) in enumerate(LAB_ROWS):
            grid.addWidget(label(text, "muted", wrap=True), i, 0)
            v = self.ro[k] = label("—", selectable=True)
            grid.addWidget(v, i, 1)
        sim = []
        for k, text in SIM_ROWS:
            self.ro[k] = label("—", selectable=True)
            sim.append((label(text, "muted", wrap=True), self.ro[k]))
        self.sim_note = label("", "muted", wrap=True)
        self._sim_parts = (label("Simulation frame", "cardTitle"), sim, self.sim_note)
        self._spot_layout(True)
        vl.addLayout(grid)
        return w

    def _spot_layout(self, wide: bool):
        """Simulation-frame readouts beside the lab-frame ones (wide) or below them (narrow)."""
        if wide == self._spot_wide:
            return
        self._spot_wide = wide
        g = self._ro_grid
        head, rows, note = self._sim_parts
        for w in (head, note, *(x for r in rows for x in r)):
            g.removeWidget(w)
        n = len(LAB_ROWS)
        r0, c0 = (0, 2) if wide else (n, 0)
        g.addWidget(head, r0, c0, 1, 2)
        for i, (lab, val) in enumerate(rows):
            g.addWidget(lab, r0 + 1 + i, c0)
            g.addWidget(val, r0 + 1 + i, c0 + 1)
        r = r0 + 1 + len(rows)
        g.addWidget(note, r, c0, (n - r) if wide else 1, 2, Qt.AlignmentFlag.AlignTop)
        for c, st in enumerate((0, 2, 0, 2) if wide else (0, 1, 0, 0)):
            g.setColumnStretch(c, st)

    def _build_shape_tab(self) -> QWidget:
        w = self._shape_inner = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(10, 10, 10, 8)
        self.shape_split = QSplitter(Qt.Orientation.Horizontal)
        self.shape_split.setChildrenCollapsible(False)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(4)
        cap = label("<b>Current spot</b> · top view, magnified, centred on the chief-ray hit · solid = half-max "
                    "contour, dashed = original spot", "muted", wrap=True)
        cap.setTextFormat(Qt.TextFormat.RichText)
        lv.addWidget(cap)
        self._shape_box = QBoxLayout(QBoxLayout.Direction.TopToBottom)
        self._shape_box.setSpacing(8)
        self.plot_map = self._plot(min_h=140, min_w=160, name="spot_shape.png")
        self._shape_box.addWidget(self.plot_map, 1)
        mw = QWidget()
        mg = QGridLayout(mw)
        mg.setContentsMargins(0, 0, 0, 0)
        mg.setHorizontalSpacing(10)
        mg.setVerticalSpacing(3)
        self.shape_ro = {}
        for i, (k, text) in enumerate(SHAPE_ROWS):
            mg.addWidget(label(text, "muted", wrap=True), i, 0)
            v = self.shape_ro[k] = label("—", wrap=True, selectable=True)
            mg.addWidget(v, i, 1)
        self.shape_status = label("", "muted", wrap=True)
        mg.addWidget(self.shape_status, len(SHAPE_ROWS), 0, 1, 2)
        mg.setColumnStretch(1, 1)
        mg.setRowStretch(len(SHAPE_ROWS) + 1, 1)
        mw.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self._shape_metrics = mw
        self._shape_box.addWidget(mw)
        lv.addLayout(self._shape_box, 1)
        self.shape_split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(4)
        rv.addWidget(label("Peak intensity vs push-rod travel", "cardTitle", wrap=True))
        self.plot_curves = self._plot(min_h=110, name="peak_vs_travel.png")
        self.plot_curves.setToolTip("Moving one actuator at a time ('trend' quality); dots mark both actuators' "
                                    "current positions")
        rv.addWidget(self.plot_curves, 3)
        rv.addWidget(label("Shape vs travel (same scale, each box ±10 µm)", "cardTitle", wrap=True))
        self.plot_gallery = self._plot(min_h=90, name="shape_vs_travel.png")
        rv.addWidget(self.plot_gallery, 2)
        self.table_btn = QToolButton()
        self.table_btn.setObjectName("flat")
        self.table_btn.setCheckable(True)
        self.table_btn.setText("▸ Data table")
        self.table_btn.toggled.connect(self._toggle_table)
        rv.addWidget(self.table_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self.data_table = QTableWidget()
        self.data_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setFixedHeight(self.TABLE_H)
        self.data_table.hide()
        rv.addWidget(self.data_table)
        self.shape_split.addWidget(right)
        self.shape_split.setStretchFactor(0, 4)
        self.shape_split.setStretchFactor(1, 5)
        vl.addWidget(self.shape_split, 1)
        return self._scrolled(w)

    def _build_setup_tab(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(10, 10, 10, 8)
        vl.setSpacing(6)
        ctl = QHBoxLayout()
        ctl.setSpacing(8)
        ctl.addWidget(label("View", "muted"))
        self.view_combo = QComboBox()
        for key, text in VIEWS:
            self.view_combo.addItem(text, key)
        ctl.addWidget(self.view_combo)
        ctl.addSpacing(8)
        ctl.addWidget(label("Gain", "muted"))
        self.gain_combo = QComboBox()
        for g in GAINS:
            self.gain_combo.addItem(f"×{g}", g)
        self.gain_combo.setCurrentIndex(GAINS.index(100))
        self.gain_combo.setToolTip("Mirror tilts and the resulting beam deflections are drawn this many times larger")
        ctl.addWidget(self.gain_combo)
        hint = label("Exact chief-ray trace; mirror tilts and deflections drawn × gain", "muted")
        hint.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        hint.setToolTip(hint.text())                # cut off in narrow windows
        ctl.addWidget(hint, 1)
        vl.addLayout(ctl)
        self.plot_schem = self._plot(toolbar=True, min_h=220)
        vl.addWidget(self.plot_schem, 1)
        return w

    def _build_link_tab(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(10, 10, 10, 8)
        vl.setSpacing(6)
        self.link_note = label(LINK_TEXT, "muted", wrap=True)
        vl.addWidget(self.link_note)
        ctl = QGridLayout()
        ctl.setHorizontalSpacing(8)
        ctl.setVerticalSpacing(6)
        ctl.addWidget(label("Target X (µm)", "muted"), 0, 0)
        self.x_edit = QLineEdit(X_DEFAULT)
        self.x_edit.setToolTip("Spot–tip offsets X along the bias axis, separated by commas or spaces "
                               "(default = the beam positions of the tip-position chain)")
        ctl.addWidget(self.x_edit, 0, 1, 1, 2)
        ctl.addWidget(label("Actuators", "muted"), 1, 0)
        self.mode_combo = QComboBox()
        for key, text in MODES:
            self.mode_combo.addItem(text, key)
        self.mode_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.mode_combo.setMinimumContentsLength(16)        # elides in narrow windows instead of widening the page
        self.mode_combo.setToolTip("Which actuators move the spot. The lower one moves it mostly across the plane of "
                                   "incidence and defocuses least; with both, the spot stays on the bias axis (Y = 0).")
        ctl.addWidget(self.mode_combo, 1, 1)
        self.measured_cb = QCheckBox("Measured scan")
        self.measured_cb.setToolTip("Overlay the measured near-field RMS of data/measured_tip_scan.csv, normalised to "
                                    "1 per bias, on the intensity axes")
        ctl.addWidget(self.measured_cb, 1, 2, Qt.AlignmentFlag.AlignRight)
        ctl.setColumnStretch(1, 1)
        vl.addLayout(ctl)
        srow = QHBoxLayout()
        srow.setSpacing(8)
        self.link_status = label("", "muted", wrap=True)
        srow.addWidget(self.link_status, 1)
        self.move_btn = button("Move actuators to selected row", self._move_to_row,
                               tooltip="Set both actuators to the travels of the selected row (or double-click a row)")
        self.move_btn.setEnabled(False)
        srow.addWidget(self.move_btn)
        vl.addLayout(srow)

        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        self.link_table = QTableWidget()
        self.link_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.link_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.link_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.link_table.setAlternatingRowColors(True)
        self.link_table.verticalHeader().setVisible(False)
        self.link_table.setMinimumHeight(90)
        self.link_table.itemSelectionChanged.connect(self._sync_link_buttons)
        self.link_table.itemDoubleClicked.connect(lambda *_: self._move_to_row())
        split.addWidget(self.link_table)
        self.plot_link = self._plot(toolbar=True, min_h=130)
        split.addWidget(self.plot_link)
        split.setSizes([200, 300])
        vl.addWidget(split, 1)

        bl = QHBoxLayout()
        bl.setSpacing(8)
        bl.addStretch(1)
        bl.addWidget(button("Export CSV…", self._export_csv, tooltip="Save the table with all its columns"))
        bl.addWidget(button("Open tip-position chain", lambda: nav.navigate("tip"),
                            tooltip="Simulate → Tip-position chain: the X scan this table feeds"))
        vl.addLayout(bl)
        w.setMinimumHeight(self.LINK_MIN_H)         # the table and the plot keep a readable height; short windows scroll
        return self._scrolled(w)

    def eventFilter(self, obj, e):
        # hidden tab pages are not resized, so the spot-shape arrangement follows the tab widget's width
        if obj is self.tabs and e.type() == QEvent.Type.Resize:
            self._spot_layout(obj.width() >= self.SPOT_SIDE_PX)
            self._shape_layout(obj.width() >= self.SHAPE_SIDE_PX)
        elif e.type() == QEvent.Type.Resize and obj in self._plot_name and not self._closed:
            # the views pick labels, panel arrangement and ranges from the figure size: redraw once resizing rests
            self._resized.add(self._plot_name[obj])
            self._resize_timer.start()
        return super().eventFilter(obj, e)

    def _redraw_resized(self):
        names, self._resized = self._resized, set()
        self._mark(*(n for n in names if self.plots[n].size() != self._drawn_size.get(n)))

    def _shape_layout(self, wide: bool):
        """Wide: spot and metrics on the left, curves and gallery on the right.  Narrow: spot beside its metrics on
        top, curves and gallery below."""
        if wide == self._shape_wide:
            return
        self._shape_wide = wide
        self.shape_split.setOrientation(Qt.Orientation.Horizontal if wide else Qt.Orientation.Vertical)
        self._shape_box.setDirection(QBoxLayout.Direction.TopToBottom if wide else QBoxLayout.Direction.LeftToRight)
        self._shape_metrics.setMaximumWidth(16777215 if wide else 340)
        self._shape_min_height()
        tot = max(sum(self.shape_split.sizes()), 2)
        f = 0.44 if wide else 0.4
        self.shape_split.setSizes([int(tot * f), tot - int(tot * f)])
        self._mark("map", "curves", "gallery")

    def _shape_min_height(self):
        """Stacked, or with the data table open, the spot, the curves and the gallery keep a readable height and the
        tab scrolls instead of squeezing them."""
        h = 0 if self._shape_wide else self.SHAPE_STACKED_H
        if self.table_btn.isChecked():
            h = (h or self.SHAPE_TABLE_H - self.TABLE_H) + self.TABLE_H
        self._shape_inner.setMinimumHeight(h)

    def _toggle_table(self, on):
        self.table_btn.setText(("▾" if on else "▸") + " Data table")
        self.data_table.setVisible(on)
        if self._shape_wide is not None:
            self._shape_min_height()

    def _retheme(self, *_):
        t = _t()
        pal = views.palette(t)
        for key, a in self.act.items():
            a.set_color(pal[key], t)
        for w in (self.warn, self.params_warn):
            w.setStyleSheet(f"color: {t.warning};")
        # the plots re-run their own drawing routines on theme changes (PlotWidget)

    # ------------------------------------------------------------------ state persistence
    def _restore(self):
        s = settings()
        try:
            d = json.loads(s.value("steering/params", "") or "{}")
        except (TypeError, ValueError):
            d = {}
        if isinstance(d, dict) and d:
            try:
                sp = SteeringParams.from_dict(d)
                good = {k: v for k, v in _form_values(sp).items()      # json.loads accepts NaN and Infinity
                        if not (isinstance(v, float) and not math.isfinite(v))}
                self.form.set_values(good)
            except (TypeError, ValueError, OverflowError):
                pass
            for n in self.form.invalid_fields():    # out of range now (e.g. bounds changed): back to the default
                self.form.reset_field(n)
        lo = max(-SPIN_UM, min(SPIN_UM, _float(s.value("steering/lower_um"), 0.0)))
        up = max(-SPIN_UM, min(SPIN_UM, _float(s.value("steering/upper_um"), 0.0)))
        self.act["lower"].set_value(lo)
        self.act["upper"].set_value(up)
        self.lower, self.upper = self.act["lower"].value(), self.act["upper"].value()
        i = self.view_combo.findData(s.value("steering/view", "top"))
        self.view_combo.setCurrentIndex(max(i, 0))
        i = self.gain_combo.findData(int(_float(s.value("steering/gain"), 100)))
        if i >= 0:
            self.gain_combo.setCurrentIndex(i)
        xs = s.value("steering/link_x")
        if isinstance(xs, str) and xs.strip():
            self.x_edit.setText(xs)
        i = self.mode_combo.findData(s.value("steering/link_mode", "lower"))
        self.mode_combo.setCurrentIndex(max(i, 0))
        self.measured_cb.setChecked(str(s.value("steering/measured", "false")).lower() in ("1", "true"))
        tab = int(_float(s.value("steering/tab"), 0))
        if 0 <= tab < self.tabs.count():
            self.tabs.setCurrentIndex(tab)

    def _save(self):
        if not isValid(self):
            return
        s = settings()
        s.setValue("steering/lower_um", self.lower)
        s.setValue("steering/upper_um", self.upper)
        s.setValue("steering/params", json.dumps(_form_values(self.sp)))
        s.setValue("steering/view", self.view_combo.currentData())
        s.setValue("steering/gain", self.gain_combo.currentData())
        s.setValue("steering/link_x", self.x_edit.text())
        s.setValue("steering/link_mode", self.mode_combo.currentData())
        s.setValue("steering/measured", self.measured_cb.isChecked())
        s.setValue("steering/tab", self.tabs.currentIndex())

    def _shutdown(self):
        """Application quitting: stop scheduling work and keep the state."""
        if self._closed:
            return
        self._closed = True
        self.queue.close()
        for tm in (self._redraw_timer, self._fine_timer, self._param_timer, self._link_timer, self._save_timer,
                   self._resize_timer):
            tm.stop()
        self._save()

    # ------------------------------------------------------------------ parameters
    def _sync_d_field(self):
        """D is edited only with D auto off; the auto value is shown on the checkbox."""
        auto = self.form.rows["D_auto"].editor
        self.form.set_enabled_fields(["D"], not auto.isChecked())
        if auto.isChecked():
            auto.setText(f"{self.D:.1f} mm" if self.D is not None and self.sp.D_auto else "…")
            auto.setToolTip("Resolved diameter for the smallest spot with the current aperture")
        else:
            auto.setText("")

    def _on_form(self):
        self._sync_d_field()
        self._param_timer.start()

    def _apply_params(self, force=False):
        invalid = self.form.invalid_fields()
        # D is unused (and greyed out) while D auto is on: an invalid D must not hold back the other fields
        d_unused = self.form.rows["D_auto"].editor.isChecked() and "D" in invalid
        bad = [n for n in invalid if not (d_unused and n == "D")]
        if not bad:
            vals = self.form.values()
            if d_unused:
                vals["D"] = self.sp.D               # keep the last valid D for when auto is switched off
            try:
                sp = SteeringParams.from_dict(vals)
            except (TypeError, ValueError) as e:
                self.params_warn.setText(f"⚠ {e}")
                self.params_warn.show()
                return
            # NaN / inf pass every range check: never let them reach the optics, the 3D page or the saved state
            bad = [k for k, v in sp.to_dict().items() if isinstance(v, float) and not math.isfinite(v)]
        if bad:
            names = ", ".join(self.form.rows[n].spec.label if n in self.form.rows else n for n in bad)
            self.params_warn.setText(f"⚠ Invalid value: {names}. The plots keep the last valid parameters.")
            self.params_warn.show()
            return
        self.params_warn.hide()
        key = _key(sp)
        if key == self._pkey and not force:
            return
        self.sp, self._pkey = sp, key
        self.D = None if sp.D_auto else float(sp.D)
        self.ref = self.cur = self.cur_at = None
        self.spot_error = ""
        self._shown = self._seq                 # results still on their way belong to the old parameters
        self.curves = self.gallery = None
        try:
            self.st0 = steering.steer(sp)
        except (ValueError, ArithmeticError):
            self.st0 = None
        self._retrace_trail()
        self._sync_d_field()
        self._update_steering(record=True)
        self._update_shape_readouts()
        self._fill_data_table()
        self._request_spot("fine")
        self.queue.submit("curves", compute_curves, (sp,), lambda r, k=key: self._on_curves(k, r))
        self.queue.submit("gallery", compute_gallery, (sp,), lambda r, k=key: self._on_gallery(k, r))
        self._request_link()
        self._mark(*self.plots)
        self._save_timer.start()

    # ------------------------------------------------------------------ actuators
    def _on_actuator(self, key, value, dragging):
        setattr(self, key, float(value))
        self._update_steering(record=True)
        if dragging:                            # cheap spot now, the fine one when the slider rests
            self._request_spot("fast")
            self._fine_timer.start()
        else:
            self._fine_timer.stop()
            self._request_spot("fine")
        self._save_timer.start()

    def _on_released(self):
        self._fine_timer.stop()
        self._request_spot("fine")

    def _set_travels(self, lower: float, upper: float):
        """Set both actuators with a single update."""
        self.act["lower"].set_value(lower)
        self.act["upper"].set_value(upper)
        self.lower, self.upper = self.act["lower"].value(), self.act["upper"].value()
        self._fine_timer.stop()
        self._update_steering(record=True)
        self._request_spot("fine")
        self._save_timer.start()

    def _clear_trail(self):
        self.trail = []
        if self.st is not None:
            self.trail.append((self.lower, self.upper, self.st.spot_um, self.st.port_um))
        self.view3d.clear_trail()
        self._mark("port", "sample")

    def _retrace_trail(self):
        """The trail stores travels; new parameters move every point (evenly thinned to TRAIL_RETRACE points)."""
        pts = self.trail
        if len(pts) > TRAIL_RETRACE:
            n = len(pts) - 1
            pts = [pts[round(i * n / (TRAIL_RETRACE - 1))] for i in range(TRAIL_RETRACE)]
        out = []
        for lo, up, *_ in pts:
            try:
                st = steering.steer(self.sp, lo, up)
            except (ValueError, ArithmeticError):
                continue
            out.append((lo, up, st.spot_um, st.port_um))
        self.trail = out

    def _update_steering(self, record=False):
        """Chief-ray trace at the current travels (~1 ms): readouts, actuator texts, scopes, 3D view."""
        try:
            self.st = steering.steer(self.sp, self.lower, self.upper)
            self.st_error = ""
        except (ValueError, ArithmeticError):
            self.st = None
            self.st_error = (f"The chief ray misses the parabolic mirror at lower {self.lower:.2f} µm, upper "
                             f"{self.upper:.2f} µm: no spot on the sample.")
        if record and self.st is not None:
            last = self.trail[-1] if self.trail else None
            if last is None or abs(last[0] - self.lower) > 1e-6 or abs(last[1] - self.upper) > 1e-6:
                self.trail.append((self.lower, self.upper, self.st.spot_um, self.st.port_um))
                del self.trail[:-TRAIL_MAX]
        self._update_readouts()
        self._update_actuator_texts()
        self.view3d.set_state(self.lower, self.upper, self.sp)
        self._mark("port", "sample", "map", "curves", "schematic")

    def _update_readouts(self):
        st = self.st
        if st is None:
            for v in self.ro.values():
                v.setText("—")
            self.warn.setText("⚠ " + self.st_error)
            self.warn.show()
        else:
            self.warn.hide()
            for k, txt in lab_readouts(st).items():
                self.ro[k].setText(txt)
        phi = self.sp.bias_axis_deg
        self.sim_note.setText(
            f"X is the spot–tip offset X of the tip-position chain. Bias axis at φ = {phi:g}° from lab +X"
            + (" (assumed; set φ under Sample orientation)." if phi == 0 else "."))

    def _update_actuator_texts(self):
        st0 = self.st0
        rows = self.curves if isinstance(self.curves, dict) else {}
        for key, a in self.act.items():
            if st0 is None:
                a.per.setText("Per µm: — (the chief ray misses the parabolic mirror at the aligned position)")
                a.hint.setText("")
                continue
            a.per.setText(per_um_text(st0.per_um[key], rows.get(key)))
            if key == "lower":
                a.hint.setText(f"Extend: mirror turns {'CCW' if st0.lower_turns_beam_to_front else 'CW'} (top view)")
            else:
                a.hint.setText(f"Extend: mirror tilts {'down' if st0.port_per_um['upper'][1] < 0 else 'up'}")

    # ------------------------------------------------------------------ background results
    def _request_spot(self, quality):
        if self._closed:
            return
        self._seq += 1
        req = (self._pkey, self.lower, self.upper, quality, self._seq)
        self.queue.submit("spot", compute_spot, (self.sp, self.lower, self.upper, quality, self.ref is None),
                          lambda r, req=req: self._on_spot(req, r))
        self._update_shape_status()

    def _on_spot(self, req, r):
        pkey, lo, up, quality, seq = req
        if self._closed or pkey != self._pkey or seq <= self._shown:
            return                              # other parameters, or an older request than the one shown
        self._shown = seq
        if isinstance(r, _Failed):
            self.cur, self.cur_at, self.spot_error = None, (lo, up, quality), r.msg
        else:
            self.D = r["D"]
            if r["ref"] is not None:
                self.ref = r["ref"]
            self.cur, self.cur_at, self.spot_error = r["cur"], (lo, up, quality), ""
        self._sync_d_field()
        self._update_shape_readouts()
        self._mark("sample", "map", "schematic")

    def _on_curves(self, key, r):
        if self._closed or key != self._pkey:
            return
        self.curves = r
        self._update_actuator_texts()
        self._fill_data_table()
        self._mark("curves")

    def _on_gallery(self, key, r):
        if self._closed or key != self._pkey:
            return
        self.gallery = r
        self._mark("gallery")

    def _request_link(self):
        if self._closed or self._pkey is None:
            return
        try:
            xs = parse_xs(self.x_edit.text())
        except ValueError as e:
            # the rows on screen no longer belong to the X list, mode and parameters: drop them (no Move or Export of
            # stale rows) and any result still on its way; the next valid list is computed afresh
            self.link_error = f"⚠ Target X: {e}"
            self.link_rows = self._link_key = self._link_for = None
            self._fill_link_table()
            self._mark("link")
            return
        self.link_error = ""
        mode = self.mode_combo.currentData()
        key = (self._pkey, xs, mode)
        if key == self._link_key and self.link_rows is not None:
            self._update_link_status()
            return
        self._link_key = key
        self.link_rows = self._link_for = None
        self._fill_link_table()
        self.queue.submit("link", compute_link, (self.sp, xs, mode),
                          lambda r, k=key, sp=self.sp, m=mode: self._on_link(k, r, sp, m))
        self._mark("link")
        self._save_timer.start()

    def _on_link(self, key, r, sp, mode):
        if self._closed or key != self._link_key:
            return
        self.link_rows = r
        self._link_for = (sp, mode)
        self._fill_link_table()
        self._mark("link")

    # ------------------------------------------------------------------ spot shape
    def _update_shape_status(self):
        if self.cur_at is None:
            txt = "Computing the spot…" if not self.spot_error else ""
        else:
            lo, up, q = self.cur_at
            txt = f"{q} quality · lower {lo:.2f} µm, upper {up:.2f} µm"
            if (lo, up) != (self.lower, self.upper) or q != "fine":
                txt += " · updating…"
        if self.spot_error:
            txt = f"⚠ Spot failed: {self.spot_error}"
        elif self.cur_at is not None and self.cur is None:
            txt = "⚠ No spot forms here: fewer than 20 rays reach the sample."
        self.shape_status.setText(txt)

    def _update_shape_readouts(self):
        s, ref = self.cur, self.ref
        if s is None or ref is None:
            for v in self.shape_ro.values():
                v.setText("—")
        else:
            plane = self.st0.incidence_plane if self.st0 is not None else None
            for k, txt in shape_readouts(s, ref, plane).items():
                self.shape_ro[k].setText(txt)
        self._update_shape_status()

    def _fill_data_table(self):
        tb = self.data_table
        tb.clear()
        tb.setColumnCount(len(TABLE_TRAVELS))
        tb.setHorizontalHeaderLabels([_m(f"{t:+d}") if t else "0" for t in TABLE_TRAVELS])
        heads = ("Horiz. · peak", "Horiz. · FWHM µm", "Vert. · peak", "Vert. · FWHM µm")
        tb.setRowCount(len(heads))
        tb.setVerticalHeaderLabels(heads)
        curves = self.curves if isinstance(self.curves, dict) else {}
        for r, (a, kind) in enumerate((("lower", "peak"), ("lower", "fw"), ("upper", "peak"), ("upper", "fw"))):
            by_t = {round(row["travel_um"]): row for row in curves.get(a, [])}
            for c, t in enumerate(TABLE_TRAVELS):
                row = by_t.get(t)
                if row is None:
                    txt = "…" if self.curves is None else "—"
                elif row.get("rel") is None:
                    txt = "—"
                else:
                    txt = f"{row['rel'] * 100:.0f} %" if kind == "peak" else f"{row['fwhmL']:.1f}×{row['fwhmS']:.1f}"
                it = QTableWidgetItem(txt)
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                tb.setItem(r, c, it)
        tb.horizontalHeader().setToolTip("Push-rod travel (µm) of the one actuator that moves")
        tb.resizeColumnsToContents()

    # ------------------------------------------------------------------ link to the simulation
    def _fill_link_table(self):
        tb = self.link_table
        tb.clearSpans()
        tb.clearContents()
        tb.setColumnCount(len(LINK_COLS))
        tb.setHorizontalHeaderLabels(LINK_COLS)
        rows = self.link_rows if isinstance(self.link_rows, list) else []
        tb.setRowCount(len(rows))
        for i, row in enumerate(rows):
            if "error" in row:
                vals = [signed(row["X_um"], 2)] + [""] * (len(LINK_COLS) - 1)
                vals[1] = "cannot reach: " + row["error"]
                tip = row["error"]
            else:
                fw = ("—" if row.get("fwhmL") is None else f"{row['fwhmL']:.2f} × {row['fwhmS']:.2f}")
                vals = [signed(row["X_um"], 2), signed(row["lower_um"], 2), counts_text(row["lower_um"]),
                        signed(row["upper_um"], 2), counts_text(row["upper_um"]), signed(row["Y_got_um"], 2),
                        f"{row['defocus_um']:.2f}", f"{row['arrival_fs']:.1f}",
                        "—" if row.get("rel") is None else f"{row['rel'] * 100:.0f}", fw]
                tip = " · ".join(t for t, on in (
                    ("Did not converge to the target X", not row.get("converged", True)),
                    (f"Beyond the ±{SPIN_UM:g} µm control range of the actuators: Move is disabled",
                     beyond_range(row))) if on)
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if tip:
                    it.setForeground(QColor(_t().warning))
                    it.setToolTip(tip)
                tb.setItem(i, c, it)
            if "error" in row:
                tb.setSpan(i, 1, 1, len(LINK_COLS) - 1)
        tb.resizeColumnsToContents()
        self._update_link_status()
        self._sync_link_buttons()

    def _update_link_status(self):
        rows = self.link_rows
        if self.link_error:
            txt = self.link_error
        elif rows is None:
            txt = "Computing the actuator travels…"
        elif isinstance(rows, _Failed):
            txt = f"⚠ Failed: {rows.msg}"
        else:
            mode = self._link_for[1] if self._link_for else self.mode_combo.currentData()
            bad = sum("error" in r or not r.get("converged", True) for r in rows)
            far = sum(beyond_range(r) for r in rows)
            txt = f"{len(rows)} offsets · {_MODE_NAMES[mode]} · peak and FWHM at 'trend' quality"
            if bad:
                txt += f" · ⚠ {bad} not reached"
            if far:
                txt += f" · ⚠ {far} beyond the ±{SPIN_UM:g} µm actuator range"
        self.link_status.setText(txt)

    def _selected_link_row(self):
        """The selected link-table row if the actuators can be moved to it, else None."""
        rows = self.link_rows if isinstance(self.link_rows, list) else []
        sel = self.link_table.selectionModel().selectedRows() if self.link_table.selectionModel() else []
        if not sel:
            return None
        i = sel[0].row()
        if 0 <= i < len(rows) and "error" not in rows[i] and not beyond_range(rows[i]):
            return rows[i]
        return None

    def _sync_link_buttons(self):
        self.move_btn.setEnabled(self._selected_link_row() is not None)

    def _move_to_row(self):
        row = self._selected_link_row()
        if row is None:
            return
        self._set_travels(row["lower_um"], row["upper_um"])

    def _export_csv(self):
        rows = self.link_rows if isinstance(self.link_rows, list) else []
        if not rows:
            error_box(self, "Export CSV", "The table is still empty.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export link table", str(paths.RESULTS / "steering_offsets.csv"),
                                              "CSV (*.csv)")
        if not path:
            return
        cols = ("X_um", "lower_um", "lower_counts", "upper_um", "upper_counts", "X_got_um", "Y_got_um", "defocus_um",
                "arrival_fs", "rel", "fwhmL", "fwhmS", "converged", "error")
        sp, mode = self._link_for or (self.sp, self.mode_combo.currentData())     # what the rows were computed for
        D = self.D if sp == self.sp else (None if sp.D_auto else sp.D)
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow([*cols, "actuators", "bias_axis_deg", "R_mm", "L_mm", "D_mm"])
                for r in rows:
                    w.writerow([*("" if r.get(c) is None else r.get(c) for c in cols), mode, sp.bias_axis_deg,
                                sp.R, sp.L, "" if D is None else D])
        except OSError as e:
            error_box(self, "Export CSV", f"Could not write {path}", str(e))

    def _measured_pts(self):
        if self._measured is None:
            self._measured = read_measured()
        return self._measured

    # ------------------------------------------------------------------ drawing
    def _changed_view(self):
        self._mark("schematic")
        self._save_timer.start()

    def _on_tab(self, _i):
        self._redraw_timer.start()
        self._save_timer.start()

    def _mark(self, *names):
        """Schedule a redraw of these plots (throttled; plots in hidden tabs wait until they are shown)."""
        self._dirty.update(names)
        if not self._redraw_timer.isActive() and not self._closed:
            self._redraw_timer.start()

    def _redraw(self):
        if self._closed or not isValid(self):
            return
        for name in list(self._dirty):
            plot = self.plots[name]
            if not plot.isVisible():
                continue
            if name == "schematic" and self.sp.D_auto and self.D is None and not self.spot_error:
                continue                        # resolve() would compute the auto D on the GUI thread
            self._dirty.discard(name)
            self._draw(name, plot)

    def _draw(self, name, plot):
        self._drawn_size[name] = plot.size()
        st, sp, s, ref = self.st, self.sp, self.cur, self.ref
        if name == "port":
            trail = [p for *_, p in self.trail]
            plot.plot(lambda fig: views.draw_port_scope(fig.add_subplot(), st, trail, _t()))
        elif name == "sample":
            trail = [q for _, _, q, _ in self.trail]
            plot.plot(lambda fig: views.draw_sample_scope(fig.add_subplot(), st, s, ref, trail, sp, _t()))
        elif name == "map":
            plot.plot(lambda fig: views.draw_spot_map(fig.add_subplot(), s, ref, st, _t()))
        elif name == "curves":
            curves, cur = self.curves, {"lower": self.lower, "upper": self.upper}
            plot.plot(lambda fig: self._pending_note(fig, curves, "Computing the peak-intensity curves…")
                      or views.draw_peak_curves(fig.add_subplot(), curves, cur, _t()))
        elif name == "gallery":
            gal = self.gallery
            plot.plot(lambda fig: self._pending_note(fig, gal, "Computing the spot shapes…")
                      or views.draw_shape_gallery(fig, gal, _t(), box_um=10.0))
        elif name == "schematic":
            view, gain, lo, up = self.view_combo.currentData(), self.gain_combo.currentData(), self.lower, self.upper
            plot.plot(lambda fig: views.draw_schematic(fig, sp, lo, up, view, gain, _t()))
        elif name == "link":
            rows = self.link_rows
            good = [r for r in rows if "error" not in r] if isinstance(rows, list) else rows
            lsp, mode = self._link_for or (sp, self.mode_combo.currentData())     # what the rows were computed for
            meas = self._measured_pts() if self.measured_cb.isChecked() else None
            note = self.link_error or "Computing the actuator travels…"
            plot.plot(lambda fig: self._pending_note(fig, good, note)
                      or views.draw_link(fig, good, lsp, mode, _t(), measured=meas))

    @staticmethod
    def _pending_note(fig, value, text) -> bool:
        """Draw a muted note instead of a plot while `value` is being computed (None) or failed; True if drawn."""
        if value is not None and not isinstance(value, _Failed):
            return False
        t = _t()
        ax = fig.add_subplot()
        ax.set_axis_off()
        msg, color = (f"Failed: {value.msg}", t.danger) if isinstance(value, _Failed) else (text, t.muted)
        ax.text(0.5, 0.5, msg, ha="center", va="center", color=color, wrap=True, transform=ax.transAxes)
        return True

    # ------------------------------------------------------------------ page API
    def on_show(self):
        self._redraw_timer.start()

    def showEvent(self, e):
        super().showEvent(e)
        self._redraw_timer.start()

    def hideEvent(self, e):
        super().hideEvent(e)
        if not self._closed:
            self._save()
