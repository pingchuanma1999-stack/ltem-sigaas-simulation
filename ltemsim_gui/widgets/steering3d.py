"""3D beam-path model of the setup: the user's reference page in a QWebEngineView, driven by the native controls.

The page (ltem-steering-module/reference/ltem-steering.html) is loaded unchanged.  Once it has loaded we press its
English button, hide everything but the 3D section with an injected style sheet and push the actuator travels and
instrument parameters through the page's own inputs (#numYaw, #numPitch, #pR ... #pD, #pDauto), dispatching the
events its handlers listen to.  The page's sliders clamp and round to their range and step; what the model cannot
show is reported in a banner above the view.  three.js 0.128.0 comes from cdn.jsdelivr.net: offline, the page's own
fallback text stays in the stage.

QtWebEngine wants Qt.ApplicationAttribute.AA_ShareOpenGLContexts set before the QApplication is created (app.py
main()).  Without it, or without QtWebEngine, the widget shows why and offers the page in the default browser.
"""
from __future__ import annotations

import importlib
import json
import math
from dataclasses import replace

from PySide6.QtCore import QCoreApplication, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget

from ltemsim.steering import REFERENCE_PAGE, SteeringParams

from ..theme import ThemeManager
from .common import button, label

INFO = "Interactive 3D model from the reference page (three.js from the internet) · drag to rotate, scroll to zoom"
# SteeringParams field -> (input id, min, max, step, label, unit): the page's range inputs (#params in the HTML)
RANGES = {
    "R": ("pR", 6.0, 30.0, 0.5, "R", "mm"),
    "theta": ("pTheta", 40.0, 72.0, 1.0, "θ", "°"),
    "beta": ("pBeta", 0.0, 65.0, 1.0, "β", "°"),
    "L": ("pL", 15.0, 60.0, 0.1, "L", "mm"),
    "alpha": ("pAlpha", 15.0, 70.0, 1.0, "α", "°"),
    "dist": ("pDist", 80.0, 600.0, 10.0, "mirror–port distance", "mm"),
    "lam": ("pLam", 700.0, 1600.0, 10.0, "λ", "nm"),
    "NA": ("pNA", 0.25, 0.6, 0.01, "NA", ""),
    "D": ("pD", 3.0, 24.0, 0.5, "D", "mm"),
}
MOTOR_LIMIT_UM = 500.0               # setMotor() in the page clamps the travel to ±500 um (and rounds to 0.01 um)
NOT_ON_PAGE = ("zLow", "portX", "foldY", "yawSide", "pitchSide")
UNSYNCED_HELP = (
    "The 3D model follows the actuator travels and R, θ, β, L, α, the mirror–port distance, λ, NA and D (or D auto). "
    "The reference page has no inputs for zLow, portX, foldY, yawSide, pitchSide or bias_axis_deg: it always draws "
    "zLow = 1 mm, the entrance port at 260 mm, the fold mirror 6 mm behind the tip and an upright mount "
    "(yawSide = pitchSide = +1), in the lab frame (+X port wall, +Y back panel), so the bias axis is not drawn. "
    "Its trail on the sample records each actuator separately: a jump of both travels at once (Zero both, Move to "
    "a link-table row) shows as an L-shaped corner.")
LOAD_TEXT = "Loading the 3D model (three.js from cdn.jsdelivr.net)…"

# hide everything but the section holding #stage and let the stage fill the view
_CSS = """
html, body { margin: 0; height: 100%; overflow: hidden; background: var(--scene); }
.wrap { max-width: none; padding: 0; gap: 0; }
.wrap > :not(.ltemsim-3d) { display: none !important; }
.ltemsim-3d .stage { position: fixed; inset: 0; height: auto; border: 0; border-radius: 0; }
"""
_PREPARE_JS = """
(function (theme, css) {
  var root = document.documentElement;
  root.setAttribute('data-theme', theme);
  var en = document.querySelector('.langseg button[data-lang="en"]');
  if (en && root.getAttribute('data-lang') !== 'en') en.click();     // setLang() lives inside the page's IIFE
  var stage = document.getElementById('stage');
  if (!stage) return JSON.stringify({ok: false});         // (PySide hands JS objects back as '': use JSON)
  stage.closest('section').classList.add('ltemsim-3d');
  if (!document.getElementById('ltemsim-3d-style')) {
    var st = document.createElement('style');
    st.id = 'ltemsim-3d-style';
    st.textContent = css;
    (document.head || root).appendChild(st);
  }
  window.dispatchEvent(new Event('resize'));
  var fb = document.getElementById('fallback');
  return JSON.stringify({ok: true, three: !!window.THREE, webgl: !!(fb && fb.hidden)});
})(%s, %s)
"""
_CLEAR_TRAIL_JS = "var b = document.getElementById('clearTrail'); if (b) b.click(); true;"
_PUSH_JS = """
(function (vals, dauto) {
  Object.keys(vals).forEach(function (id) {
    var el = document.getElementById(id);
    if (!el) return;
    el.value = vals[id];
    el.dispatchEvent(new Event('input', {bubbles: true}));
  });
  if (dauto !== null) {                                    // after #pD: moving the D slider switches auto off
    var c = document.getElementById('pDauto');
    if (c) { c.checked = dauto; c.dispatchEvent(new Event('change', {bubbles: true})); }
  }
  return true;
})(%s, %s)
"""


def _webengine_problem() -> str | None:
    """Why QtWebEngine cannot be used in this process, or None."""
    app = QCoreApplication.instance()
    if app is not None and not QCoreApplication.testAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts):
        return ("The embedded 3D view needs Qt WebEngine, which must be prepared before the application starts "
                "(Qt.AA_ShareOpenGLContexts was not set before the application object was created).")
    try:
        importlib.import_module("PySide6.QtWebEngineWidgets")
    except Exception as e:                                     # ImportError, or a missing Qt library
        return f"The embedded 3D view needs Qt WebEngine, which is not available here ({e})."
    return None


def _fmt(x: float) -> str:
    return f"{x:.10g}"


def page_values(lower_um: float, upper_um: float, sp: SteeringParams) -> tuple[dict, bool, list[str]]:
    """What the page will show: ({input id: value}, D auto, notes on everything it shows differently)."""
    notes = []
    vals = {}
    for key, actuator in (("numYaw", lower_um), ("numPitch", upper_um)):
        v = float(actuator)
        if not math.isfinite(v):
            notes.append(f"{'lower' if key == 'numYaw' else 'upper'} actuator {v:g} µm not shown")
            continue
        if abs(v) > MOTOR_LIMIT_UM:
            name = "lower" if key == "numYaw" else "upper"
            notes.append(f"{name} actuator {v:g} µm clipped to ±{MOTOR_LIMIT_UM:g} µm")
            v = math.copysign(MOTOR_LIMIT_UM, v)
        vals[key] = _fmt(round(v, 2))
    for field, (el, lo, hi, step, name, unit) in RANGES.items():
        if field == "D" and sp.D_auto:
            continue                                           # the page finds the auto diameter itself
        v = float(getattr(sp, field))
        u = f" {unit}" if unit else ""
        if not math.isfinite(v):                               # min/max keep NaN, math.floor(NaN) raises
            notes.append(f"{name} {v:g}{u} not shown")
            continue
        c = min(max(v, lo), hi)
        shown = min(lo + math.floor((c - lo) / step + 0.5) * step, hi)   # a range input snaps to min + k*step
        shown = float(f"{shown:.10g}")
        if c != v:
            notes.append(f"{name} {v:g}{u} clipped to {lo:g}–{hi:g}{u}")
        elif abs(shown - v) > 1e-9 * max(1.0, abs(v)):
            notes.append(f"{name} {v:g}{u} shown as {shown:g}{u} (slider step {step:g})")
        vals[el] = _fmt(shown)
    default = SteeringParams()
    off = [f for f in NOT_ON_PAGE if getattr(sp, f) != getattr(default, f)]
    if off:
        notes.append(f"{', '.join(off)} not shown (the page draws the defaults)")
    return vals, bool(sp.D_auto), notes


class Steering3DView(QWidget):
    """The user's reference page (steering.REFERENCE_PAGE) in a QWebEngineView, reduced to its 3D beam-path model
    and driven by the native controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = None                  # (lower_um, upper_um, SteeringParams) last set
        self._pushed = {}                   # input id -> value the page has now (cleared on every load)
        self._view = None
        self._interceptor = None
        self._started = False
        self._ready = False                 # page loaded and reduced to the 3D section
        self._timer = QTimer(self)          # coalesces bursts of set_state (slider drags)
        self._timer.setSingleShot(True)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._push)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._bar = QWidget()
        bar = QHBoxLayout(self._bar)
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(8)
        self.info = label(INFO, "muted", wrap=True)
        self.info.setToolTip(UNSYNCED_HELP)
        bar.addWidget(self.info, 1)
        self.reload_btn = button("Reload", self.reload, tooltip="Load the reference page again")
        bar.addWidget(self.reload_btn)
        bar.addWidget(button("Open full page in browser", self.open_in_browser,
                             tooltip=f"{REFERENCE_PAGE.name}: the full bilingual page with its own controls"))
        lay.addWidget(self._bar)
        self.banner = label("", "muted", wrap=True)
        self.banner.setToolTip(UNSYNCED_HELP)
        self.banner.hide()
        lay.addWidget(self.banner)

        self._stack = QStackedWidget()
        self._placeholder = QWidget()
        pl = QVBoxLayout(self._placeholder)
        pl.addStretch(1)
        self._message = label(LOAD_TEXT, "muted", wrap=True)
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl.addWidget(self._message)
        self._fallback_btn = button("Open full page in browser", self.open_in_browser)
        self._fallback_btn.hide()
        pl.addWidget(self._fallback_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        pl.addStretch(1)
        self._stack.addWidget(self._placeholder)
        lay.addWidget(self._stack, 1)
        ThemeManager.instance().changed.connect(self._on_theme)

    # ------------------------------------------------------------------ public
    def set_state(self, lower_um: float, upper_um: float, sp: SteeringParams) -> None:
        """Remember the actuator travels (um) and parameters; pushed to the page once it has loaded."""
        self._state = (float(lower_um), float(upper_um), replace(sp))
        notes = page_values(*self._state)[2]
        self.banner.setText("3D view: " + " · ".join(notes) if notes else "")
        self.banner.setVisible(bool(notes) and self._view is not None)
        if self._ready:
            self._timer.start()

    def clear_trail(self) -> None:
        """Restart the page's trail on the sample at the current travels (its hidden 'Clear trail' button)."""
        if self._ready and self._view is not None:
            self._view.page().runJavaScript(_CLEAR_TRAIL_JS)

    def open_in_browser(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(REFERENCE_PAGE)))

    def reload(self) -> None:
        if self._view is None:
            return
        self._ready = False
        self._pushed = {}
        self._show_message(LOAD_TEXT)
        self._view.setUrl(QUrl.fromLocalFile(str(REFERENCE_PAGE)))

    # ------------------------------------------------------------------ web view
    def showEvent(self, e):
        super().showEvent(e)
        if not self._started:
            self._started = True
            self._start()

    def _start(self):
        problem = _webengine_problem()
        if problem is None and not REFERENCE_PAGE.is_file():
            problem = f"The reference page is missing: {REFERENCE_PAGE}"
        if problem:
            self._bar.hide()
            self._show_message(problem)
            self._fallback_btn.setVisible(REFERENCE_PAGE.is_file())
            return
        from PySide6.QtWebEngineCore import QWebEngineSettings, QWebEngineUrlRequestInterceptor
        from PySide6.QtWebEngineWidgets import QWebEngineView

        class FontBlocker(QWebEngineUrlRequestInterceptor):
            # The page's Google Fonts style sheet is render- and script-blocking: where it is slow or unreachable it
            # would hold back three.js and the model for a TCP timeout.  The labels fall back to system fonts.
            def interceptRequest(self, info):
                if info.requestUrl().host() in ("fonts.googleapis.com", "fonts.gstatic.com"):
                    info.block(True)

        view = QWebEngineView()
        view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        page = view.page()
        s = page.settings()
        s.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        self._interceptor = FontBlocker(page)
        page.setUrlRequestInterceptor(self._interceptor)
        page.setBackgroundColor(QColor(ThemeManager.instance().tokens.bg))
        page.loadFinished.connect(self._on_loaded)
        page.renderProcessTerminated.connect(self._on_crashed)
        self._view = view
        self._stack.addWidget(view)
        self.banner.setVisible(bool(self.banner.text()))
        self.reload()

    def _show_message(self, text):
        self._message.setText(text)
        self._stack.setCurrentWidget(self._placeholder)

    def _on_loaded(self, ok):
        self._ready = False
        self._pushed = {}
        if not ok:
            self._show_message(f"Could not load {REFERENCE_PAGE}. Try Reload, or open the page in the browser.")
            return
        dark = "dark" if ThemeManager.instance().tokens.dark else "light"
        js = _PREPARE_JS % (json.dumps(dark), json.dumps(_CSS))
        self._view.page().runJavaScript(js, 0, self._on_prepared)

    def _on_prepared(self, res):
        try:
            res = json.loads(res)
        except (TypeError, ValueError):
            res = {}
        if not isinstance(res, dict) or not res.get("ok"):
            self._show_message("The reference page has no 3D stage (#stage). Open it in the browser instead.")
            return
        self._ready = True
        self._stack.setCurrentWidget(self._view)
        if not res.get("three"):
            self.info.setText("three.js could not be loaded from cdn.jsdelivr.net (offline?), so the page shows its "
                              "fallback text. Press Reload once online.")
        elif not res.get("webgl"):
            self.info.setText("WebGL is not available in this window, so the page shows its fallback text. "
                              "Open the full page in the browser.")
        else:
            self.info.setText(INFO)
        self._push()
        if self._state is not None:
            # the page seeds its trail with a lower-actuator path 0 -> 4 um that never happened; scripts run in order,
            # so this starts the trail at the travels just pushed
            self.clear_trail()

    def _on_crashed(self, status, code):
        self._ready = False
        self._show_message(f"The 3D view's web process stopped (exit code {code}). Press Reload to start it again.")

    def _push(self):
        """Send the inputs that changed since the last push (the page's sliders are hidden: only we change them)."""
        if not self._ready or self._state is None:
            return
        vals, dauto, _ = page_values(*self._state)
        changed = {k: v for k, v in vals.items() if self._pushed.get(k) != v}
        send_dauto = self._pushed.get("pDauto") != dauto or "pD" in changed
        if not changed and not send_dauto:
            return
        self._view.page().runJavaScript(_PUSH_JS % (json.dumps(changed), json.dumps(dauto if send_dauto else None)))
        self._pushed.update(changed)
        self._pushed["pDauto"] = dauto
        if dauto:
            self._pushed.pop("pD", None)    # the page overwrites #pD with its own optimum while auto is on

    def _on_theme(self, t):
        if self._view is None:
            return
        self._view.page().setBackgroundColor(QColor(t.bg))
        if self._ready:
            self._view.page().runJavaScript(
                f"document.documentElement.setAttribute('data-theme', {json.dumps('dark' if t.dark else 'light')})")
