"""Offscreen smoke test of the pump-steering page (ltemsim_gui/pages/steering.py).

Run with:  python -m pytest tests/test_gui_steering.py   (skipped when PySide6 is not installed)
"""
import os
import sys
import time
from dataclasses import fields

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("PySide6.QtWidgets")

from ltemsim.steering import SteeringParams  # noqa: E402


@pytest.fixture(autouse=True)
def _scratch_app_id(monkeypatch):
    """No test may reach the user's real preferences: QSettings(APP_ID, APP_ID) is the native plist of the app
    (setDefaultFormat/setPath do not redirect it).  The page's own settings() is pointed at a scratch ini by each
    test; this keeps anything else that builds QSettings from paths.APP_ID off the real file too."""
    from ltemsim_gui import paths
    monkeypatch.setattr(paths, "APP_ID", "LTEMSimTestScratch")


def _app():
    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtWidgets import QApplication
    if QApplication.instance() is None:
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=20.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    app.processEvents()
    return cond()


def _pump(app, seconds):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        app.processEvents()
        time.sleep(0.01)


def _plot_errors(page):
    """Texts of PlotWidget's 'Plot failed' fallback in any of the page's figures."""
    out = []
    for name, p in page.plots.items():
        for t in p.figure.texts:
            if t.get_text().startswith("Plot failed"):
                out.append(f"{name}: {t.get_text()}")
    return out


def test_steering_specs_cover_every_field():
    from ltemsim_gui.paramspec import steering_specs
    specs = steering_specs()
    assert [s.name for s in specs] == [f.name for f in fields(SteeringParams)]
    assert all(s.label != s.name for s in specs)                     # every field has an English label
    assert SteeringParams.from_dict({s.name: s.default for s in specs}) == SteeringParams()
    by = {s.name: s for s in specs}
    assert by["yawSide"].choices == ("1", "-1") and by["pitchSide"].choices == ("1", "-1")
    assert (by["R"].unit, by["theta"].unit, by["lam"].unit) == ("mm", "deg", "nm")
    assert not by["R"].help.startswith("mm")
    assert {s.name for s in specs if not s.advanced} == {"R", "theta", "beta", "NA", "L", "tilt_scale", "lam",
                                                         "D_auto", "D", "bias_axis_deg"}
    # the tilt calibration: dimensionless, positive, beside the lever arm it scales
    k = by["tilt_scale"]
    assert (k.label, k.unit, k.minimum, k.maximum) == ("Tilt calibration κ", "", 0.001, 10.0)
    assert k.section == by["L"].section == "Steering mirror" and k.default == SteeringParams().tilt_scale
    assert "nominal 1/L" in k.help
    # beta < 0 would put the pump on the THz collection half of the parabolic mirror
    assert by["beta"].minimum == 0.0
    # the side help describes the geometry; an upside-down mount needs both sides flipped
    assert "yawSide = pitchSide = -1" in by["yawSide"].help and "yawSide = -1" in by["pitchSide"].help
    assert "upside down" not in by["pitchSide"].help.split("(")[0]


def test_top_view_puts_the_lower_actuator_on_its_side():
    """Setup 'Overview (top)': the lower Z812 sits on the yawSide of the mount (as in the mirror view), the upper one
    on the other side, level with the pivot."""
    import numpy as np
    from matplotlib.figure import Figure
    from ltemsim import steering
    from ltemsim_gui.theme import LIGHT
    from ltemsim_gui.widgets import steering_views as views

    def lateral(sp):
        fig = Figure(figsize=(9, 4))
        views.draw_schematic(fig, sp, 4.0, 2.0, "top", 100, LIGHT)
        g = steering.optics.geometry(steering.resolve(sp))
        out = {}
        for ax in fig.axes:
            for txt in ax.texts:
                if txt.get_text().endswith("Z812"):
                    out[txt.get_text().split()[0]] = float(np.dot(np.asarray(txt.xy) - g["S"][:2], g["uBack"][:2]))
        return out

    up, mirrored = lateral(SteeringParams()), lateral(SteeringParams(yawSide=-1))
    assert up["lower"] > 10 and up["upper"] < -10
    assert mirrored["lower"] == pytest.approx(-up["lower"], abs=1e-6)
    assert mirrored["upper"] == pytest.approx(-up["upper"], abs=1e-6)
    assert lateral(SteeringParams(pitchSide=-1)) == pytest.approx(up, abs=1e-6)   # pitchSide: above/below only


def test_3d_page_values_skip_non_finite():
    from ltemsim_gui.widgets.steering3d import page_values
    vals, dauto, notes = page_values(0.0, 0.0, SteeringParams(R=float("nan"), D_auto=False, D=float("inf")))
    assert "pR" not in vals and "pD" not in vals and "pTheta" in vals
    assert any(n.startswith("R nan") for n in notes) and any(n.startswith("D inf") for n in notes)


def test_3d_page_values_show_the_defaults():
    """The reference page's R slider (2-30 mm in 0.01 mm) holds the default R: no banner with the defaults."""
    import re
    from ltemsim import steering
    from ltemsim_gui.widgets.steering3d import RANGES, page_values
    html = steering.REFERENCE_PAGE.read_text(encoding="utf-8")
    for field, (el, lo, hi, step, *_) in RANGES.items():          # RANGES mirrors the page's range inputs
        tag = re.search(rf'<input type="range" id="{el}"[^>]*>', html).group(0)
        attr = {k: float(v) for k, v in re.findall(r'(min|max|step)="([^"]+)"', tag)}
        assert (attr["min"], attr["max"], attr["step"]) == (lo, hi, step), field
    vals, dauto, notes = page_values(0.0, 0.0, SteeringParams())
    assert notes == [] and dauto is True and "pD" not in vals
    assert (vals["pR"], vals["pTheta"]) == ("16", "45")
    assert page_values(0.0, 0.0, steering.module_defaults())[2] == []
    # clipping and visible roundings are still reported
    assert page_values(0.0, 0.0, SteeringParams(R=1.5))[2] == ["R 1.5 mm clipped to 2–30 mm"]
    assert page_values(0.0, 0.0, SteeringParams(R=2.004))[2] == ["R 2.004 mm shown as 2 mm (slider step 0.01)"]
    assert page_values(0.0, 0.0, SteeringParams(theta=38.0))[2] == ["θ 38° clipped to 40–72°"]
    assert page_values(0.0, 0.0, SteeringParams(D_auto=False))[2] == ["D 13.1 mm shown as 13 mm (slider step 0.5)"]


def test_3d_page_values_push_the_effective_travel():
    """The page tilts the mirror by travel / L: it is handed kappa * travel (steering.core_travel), so the model shows
    the calibrated tilt; the nominal lever (module defaults, kappa = 1) passes the travels through."""
    from ltemsim import steering
    from ltemsim_gui.widgets.steering3d import UNSYNCED_HELP, page_values
    sp = SteeringParams()
    vals, _, notes = page_values(4.0, -2.0, sp)
    assert (vals["numYaw"], vals["numPitch"]) == ("1.33", "-0.67") and notes == []
    assert vals["numYaw"] == f"{round(steering.core_travel(sp, 4.0), 2):.10g}"
    vals, _, notes = page_values(4.0, -2.0, steering.module_defaults())
    assert (vals["numYaw"], vals["numPitch"]) == ("4", "-2") and notes == []
    # the page's ±500 µm limit applies to what it is handed
    vals, _, notes = page_values(400.0, 0.0, SteeringParams(tilt_scale=2.0))
    assert vals["numYaw"] == "500" and notes == ["lower actuator 400 µm (κ × travel = 800 µm) clipped to ±500 µm"]
    assert page_values(400.0, 0.0, sp)[2] == []                   # 133 µm on the page
    vals, _, notes = page_values(4.0, 0.0, SteeringParams(tilt_scale=float("nan")))
    assert "numYaw" not in vals and "numPitch" not in vals and notes[0].startswith("lower actuator not shown")
    assert "κ × travel" in UNSYNCED_HELP


class _FakePage:
    """Records the scripts Steering3DView sends to its QWebEnginePage."""
    def __init__(self):
        self.calls = []

    def runJavaScript(self, js, *args):
        self.calls.append(js)


def _fake_view():
    from PySide6.QtWidgets import QWidget

    class FakeView(QWidget):                # stands in for the QWebEngineView (no web process in the tests)
        def __init__(self):
            super().__init__()
            self._page = _FakePage()

        def page(self):
            return self._page
    return FakeView()


def _pushed(js):
    """(values, dauto) a _PUSH_JS call sent to the page."""
    import json
    import re
    m = re.search(r"\}\)\((.*), (true|false|null)\)\s*$", js.strip())
    return json.loads(m.group(1)), json.loads(m.group(2))


def test_3d_view_resends_D_after_auto_and_clears_the_seeded_trail():
    app = _app()
    from ltemsim_gui.widgets import steering3d as m3d
    w = m3d.Steering3DView()
    try:
        w._view = _fake_view()
        w._stack.addWidget(w._view)
        calls = w._view.page().calls
        w.set_state(0.0, 0.0, SteeringParams(D_auto=False, D=13.0, NA=0.3))
        w._on_prepared('{"ok": true, "three": true, "webgl": true}')
        # after load: the travels and parameters, then the page's own trail (seeded 0 -> 4 um) restarts here
        assert len(calls) == 2 and calls[1] == m3d._CLEAR_TRAIL_JS
        vals, dauto = _pushed(calls[0])
        assert vals["pD"] == "13" and vals["numYaw"] == "0" and dauto is False
        # D auto on: the page computes its own D (9.2 mm here) and writes it into #pD
        w.set_state(0.0, 0.0, SteeringParams(D_auto=True, D=13.0, NA=0.3))
        w._push()
        assert _pushed(calls[-1]) == ({}, True)
        # auto off again with the same D: D must be sent again, before the checkbox
        w.set_state(0.0, 0.0, SteeringParams(D_auto=False, D=13.0, NA=0.3))
        w._push()
        assert _pushed(calls[-1]) == ({"pD": "13"}, False)
        n = len(calls)
        w.clear_trail()
        assert calls[n:] == [m3d._CLEAR_TRAIL_JS]
        # the travels go to the page as kappa * travel; the remembered state keeps the real ones
        w.set_state(4.0, -2.0, SteeringParams(D_auto=False, D=13.0, NA=0.3))
        w._push()
        assert _pushed(calls[-1]) == ({"numYaw": "1.33", "numPitch": "-0.67"}, None)
        assert w._state[:2] == (4.0, -2.0)
        assert "κ × travel" in w.info.toolTip() and "κ × travel" in w.banner.toolTip()
    finally:
        w._view = None
        w.deleteLater()
        app.processEvents()


def test_steering_page(tmp_path, monkeypatch):
    app = _app()
    from PySide6.QtCore import QSettings
    from ltemsim_gui.pages import steering as mod

    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(mod, "settings", lambda: QSettings(ini, QSettings.Format.IniFormat))
    page = mod.SteeringPage()
    try:
        page.resize(1200, 820)
        page.show()
        page.tabs.setCurrentIndex(0)
        _pump(app, 0.2)

        # the control column (with the calibration group) needs no horizontal scrolling
        area = page.calib_combo.parentWidget()
        while not isinstance(area, mod.QScrollArea):
            area = area.parentWidget()
        assert area.widget().minimumSizeHint().width() <= area.viewport().width()
        assert area.horizontalScrollBar().maximum() == 0

        # the calibrated defaults: one 0.2 um step of the upper actuator moves the spot 55 nm (exact trace)
        assert page.calib_result.text() == "κ = 0.33336 · per 0.2 µm step: upper 55.0 nm, lower 59.6 nm"
        assert "0.2 µm step → 55.0 nm" in page.act["upper"].per.text()
        assert "0.2 µm step → 59.6 nm" in page.act["lower"].per.text()
        # per um of REAL travel: lower (-0.283, -0.094), upper (+0.142, -0.236)
        assert "Per µm → spot <b>0.30 µm</b> (left 0.28 · front 0.09)" in page.act["lower"].per.text()
        assert "Per µm → spot <b>0.28 µm</b> (right 0.14 · front 0.24)" in page.act["upper"].per.text()
        assert page.sp.R == 16.0 and page.st0 is not None

        page.act["lower"].spin.setValue(4.0)
        # chief-ray readouts are synchronous (calibrated defaults: lower 4 um -> spot (-1.131, -0.375) um)
        assert page.ro["x"].text() == "left 1.13 µm"
        assert page.ro["y"].text() == "front 0.38 µm"
        assert page.ro["X"].text() == "−1.13 µm"
        assert page.act["lower"].slider.value() == 40
        assert "+138 counts" in page.act["lower"].kdc.text()          # the KDC101 counts of the real travel
        # the 3D model gets the effective travel kappa * 4 um
        from ltemsim_gui.widgets.steering3d import page_values
        assert page.view3d._state[:2] == (4.0, 0.0) and page_values(*page.view3d._state)[0]["numYaw"] == "1.33"

        # the fine spot at (4, 0) arrives from the worker thread
        assert _wait(app, lambda: page.cur_at == (4.0, 0.0, "fine") and page.cur is not None), page.shape_status.text()
        assert page.shape_ro["fwhm"].text() == "1.56 × 1.13 µm"
        assert page.shape_ro["peak"].text() == "97 %"
        assert page.D == pytest.approx(13.1)
        assert _wait(app, lambda: isinstance(page.curves, dict) and isinstance(page.gallery, dict)
                     and isinstance(page.link_rows, list))
        assert page.link_table.rowCount() == 7
        assert "at ±5 µm, peak" in page.act["lower"].per.text()

        # the slider pins at its end beyond ±30 um
        page.act["upper"].spin.setValue(40.0)
        assert page.act["upper"].slider.value() == page.act["upper"].slider.maximum()

        for i in range(page.tabs.count()):
            page.tabs.setCurrentIndex(i)
            _pump(app, 0.3)
        assert _wait(app, lambda: page.queue.idle())
        for i in (0, 1, 2, 4):                  # draw the plots once more with every result in
            page.tabs.setCurrentIndex(i)
            _pump(app, 0.2)
        assert not _plot_errors(page), _plot_errors(page)

        # moving to a row of the link table puts the spot at that X (travels rounded to the spin box's 0.01 um);
        # the travel ranges cover the X list's lower-only travels (X = -7.5 um: +26.6 um), so the slider holds it
        assert max(abs(r["lower_um"]) for r in page.link_rows) < mod.SLIDER_UM
        assert min(mod.CURVE_TRAVELS) <= -max(abs(r["lower_um"]) for r in page.link_rows)
        assert max(mod.CURVE_TRAVELS) >= max(abs(r["lower_um"]) for r in page.link_rows)
        page.link_table.selectRow(0)
        page._move_to_row()
        assert page.st.X_um == pytest.approx(-7.5, abs=0.01)
        assert page.act["lower"].slider.value() == 266
        assert page.link_table.item(0, 2).text() == "+920"                # KDC101 counts of the real travel

        # an invalid parameter keeps the last valid set
        page.form.rows["L"].editor.setText("0")
        page._apply_params()
        assert page.params_warn.isVisibleTo(page) and page.sp.L == pytest.approx(37.7)
    finally:
        page._shutdown()
        page.close()
        page.deleteLater()
        _pump(app, 0.1)
    s = QSettings(ini, QSettings.Format.IniFormat)                # state kept for the next start
    assert float(s.value("steering/lower_um")) == pytest.approx(26.634, abs=0.01)
    assert int(s.value("steering/params_version")) == mod.PARAMS_VERSION


def _page(tmp_path, monkeypatch, saved=None):
    app = _app()
    from PySide6.QtCore import QSettings
    from ltemsim_gui.pages import steering as mod
    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(mod, "settings", lambda: QSettings(ini, QSettings.Format.IniFormat))
    if saved:
        s = QSettings(ini, QSettings.Format.IniFormat)
        for k, v in saved.items():
            s.setValue(k, v)
        s.sync()
    page = mod.SteeringPage()
    page.resize(1200, 820)
    return app, mod, page, ini


def _close(app, page):
    page._shutdown()
    page.close()
    page.deleteLater()
    _pump(app, 0.1)


def test_steering_page_rejects_non_finite_values(tmp_path, monkeypatch):
    """NaN / inf in a field is invalid like any out-of-range value: the page keeps the last valid parameters, never
    saves them, and a saved NaN does not break the next start."""
    import json
    from PySide6.QtCore import QSettings
    app, mod, page, ini = _page(tmp_path, monkeypatch)
    try:
        for name, text in (("L", "nan"), ("bias_axis_deg", "nan"), ("R", "inf"), ("NA", "-inf")):
            page.form.rows[name].editor.setText(text)
            assert not page.form.rows[name].valid, name
            page._apply_params()
            assert page.params_warn.isVisibleTo(page)
            assert page.sp == SteeringParams()
            page.form.reset_field(name)
        page._apply_params()
        assert not page.params_warn.isVisibleTo(page)
        page.form.rows["L"].editor.setText("nan")
        page._save()
    finally:
        _close(app, page)
    saved = json.loads(QSettings(ini, QSettings.Format.IniFormat).value("steering/params"))
    assert all(not (isinstance(v, float) and v != v) for v in saved.values())

    # a NaN that reached the settings anyway (older version, hand edit) falls back to the default
    bad = json.dumps({"R": float("nan"), "L": 30.0, "beta": -30.0, "bias_axis_deg": float("inf")})
    app, mod, page, ini = _page(tmp_path, monkeypatch, {"steering/params": bad})
    try:
        assert page.form.is_valid()
        assert page.sp == SteeringParams(L=30.0)
        assert page._pkey is not None and page.st is not None
        assert page.view3d.banner.text() == ""
    finally:
        _close(app, page)


def test_steering_page_invalid_D_does_not_block_D_auto(tmp_path, monkeypatch):
    app, mod, page, ini = _page(tmp_path, monkeypatch)
    try:
        auto = page.form.rows["D_auto"].editor
        auto.setChecked(False)
        page._apply_params()
        assert page.sp.D_auto is False
        page.form.rows["D"].editor.setText("0.1")             # below the 0.5 mm minimum
        page._apply_params()
        assert page.params_warn.isVisibleTo(page) and page.sp.D_auto is False
        auto.setChecked(True)                                 # D is greyed out and unused now
        page._apply_params()
        assert not page.params_warn.isVisibleTo(page)
        assert page.sp.D_auto is True and page.sp.D == SteeringParams().D     # the last valid D is kept
        page.form.rows["R"].editor.setText("20")
        page._apply_params()
        assert page.sp.R == 20.0 and not page.params_warn.isVisibleTo(page)
        auto.setChecked(False)                                # D in use again: invalid again
        page._apply_params()
        assert page.params_warn.isVisibleTo(page) and page.sp.D_auto is True
    finally:
        _close(app, page)


def test_steering_page_link_rows_follow_the_x_list(tmp_path, monkeypatch):
    """An unparsable X list drops the rows of the old list, mode and parameters: no Move, no Export of them, no late
    result shown under the error; the status always says what the table holds."""
    import csv
    app, mod, page, ini = _page(tmp_path, monkeypatch)
    try:
        page.show()
        page.tabs.setCurrentIndex(4)
        assert _wait(app, lambda: isinstance(page.link_rows, list) and page.queue.idle())
        assert page._link_for == (page.sp, "lower")
        page.link_table.selectRow(0)
        assert page.move_btn.isEnabled()
        good = page.x_edit.text()

        page.x_edit.setText(good + ", x")
        page._request_link()
        assert page.link_rows is None and page._link_key is None and page._link_for is None
        assert page.link_table.rowCount() == 0 and not page.move_btn.isEnabled()
        assert page.link_status.text().startswith("⚠ Target X")
        page.mode_combo.setCurrentIndex(page.mode_combo.findData("upper"))      # still invalid: nothing computed
        page.form.rows["bias_axis_deg"].editor.setText("90")
        page._apply_params()
        assert page.link_rows is None and page.link_status.text().startswith("⚠ Target X")

        # back to the old text: computed afresh for the current mode and parameters, error gone
        page.x_edit.setText(good)
        page._request_link()
        assert page.link_status.text() == "Computing the actuator travels…"
        assert _wait(app, lambda: isinstance(page.link_rows, list) and page.queue.idle())
        assert page._link_for == (page.sp, "upper") and page.sp.bias_axis_deg == 90.0
        assert page.link_status.text().startswith("7 offsets · upper actuator only")
        assert all(r["lower_um"] == 0.0 for r in page.link_rows if "error" not in r)

        # an unchanged list re-checks the status (it does not keep an old error)
        page.link_status.setText("stale")
        page._request_link()
        assert page.link_status.text().startswith("7 offsets · upper actuator only")

        # a result still on its way when the list turns invalid is dropped
        page.x_edit.setText("-5, 0, 5")
        page._request_link()
        page.x_edit.setText("-5, 0, 5x")
        page._request_link()
        assert _wait(app, lambda: page.queue.idle())
        _pump(app, 0.1)
        assert page.link_rows is None and page.link_status.text().startswith("⚠ Target X")

        # Export writes the mode and parameters the rows were computed for
        page.x_edit.setText(good)
        page._request_link()
        assert _wait(app, lambda: isinstance(page.link_rows, list) and page.queue.idle())
        out = tmp_path / "rows.csv"
        monkeypatch.setattr(mod.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), "CSV (*.csv)"))
        page._export_csv()
        rows = list(csv.DictReader(open(out, encoding="utf-8")))
        assert len(rows) == 7
        assert {r["actuators"] for r in rows} == {"upper"} and {r["bias_axis_deg"] for r in rows} == {"90.0"}
    finally:
        _close(app, page)


def test_steering_page_move_refuses_travel_beyond_the_spin_range(tmp_path, monkeypatch):
    app, mod, page, ini = _page(tmp_path, monkeypatch)
    try:
        page.form.rows["L"].editor.setText("100")
        page._apply_params()
        page.x_edit.setText("-80, -5")
        page._request_link()
        assert _wait(app, lambda: isinstance(page.link_rows, list) and page.queue.idle())
        far, near = page.link_rows
        assert far["converged"] and far["lower_um"] > mod.SPIN_UM and mod.beyond_range(far)
        assert not mod.beyond_range(near)
        assert "1 beyond the ±500 µm actuator range" in page.link_status.text()
        assert "control range" in page.link_table.item(0, 1).toolTip()
        page.link_table.selectRow(0)
        assert not page.move_btn.isEnabled()
        page._move_to_row()
        assert page.lower == 0.0                                  # nothing moved (no silent clamp to 500 µm)
        page.link_table.selectRow(1)
        assert page.move_btn.isEnabled()
        page._move_to_row()
        assert page.st.X_um == pytest.approx(-5.0, abs=0.01)
    finally:
        _close(app, page)


def test_steering_page_readouts_with_the_module_defaults(tmp_path, monkeypatch):
    """The module's own defaults (R = 16 mm, theta = 60 deg) still give the README's numbers through the page."""
    from ltemsim import steering
    app, mod, page, ini = _page(tmp_path, monkeypatch)
    try:
        page.form.set_values(mod._form_values(steering.module_defaults()))
        page._apply_params()
        assert page.sp == steering.module_defaults()
        page.act["lower"].spin.setValue(4.0)
        # README: 4 um on the lower actuator -> spot (-3.387, -1.272) um
        assert (page.ro["x"].text(), page.ro["y"].text()) == ("left 3.39 µm", "front 1.27 µm")
        assert page.calib_result.text() == "κ = 1 · per 0.2 µm step: upper 235.8 nm, lower 181.6 nm"
    finally:
        _close(app, page)


def _saved_before_v3(sp: SteeringParams) -> dict:
    """steering/params as a version before 3 saved them: no tilt_scale field."""
    from ltemsim_gui.pages import steering as mod
    d = mod._form_values(sp)
    del d["tilt_scale"]
    return d


# the user's saved state as version 2 left it (R fitted to the 55 nm step, D auto on)
_V2_USER = {"R": 5.332, "theta": 45.0, "beta": 45.0, "NA": 0.42, "zLow": 1.0, "alpha": 45.0, "dist": 250.0, "L": 37.7,
            "yawSide": "1", "pitchSide": "1", "portX": 260.0, "foldY": 6.0, "lam": 820.0, "D_auto": True, "D": 4.4,
            "bias_axis_deg": 0.0}


def test_steering_page_migrates_params_saved_before_version_1_and_2(tmp_path, monkeypatch):
    """steering/params saved before params_version 2 hold the module's defaults R 16, theta 60, D 13 (and no
    tilt_scale): those become the defaults, every value the user changed survives, and the migrated set is saved
    with the current version."""
    import json
    from PySide6.QtCore import QSettings
    from ltemsim import steering
    from ltemsim_gui.pages import steering as mod
    assert mod.PARAMS_VERSION == 3

    # the user's settings: everything at the old defaults (no version key)
    old = _saved_before_v3(steering.module_defaults())
    app, mod, page, ini = _page(tmp_path, monkeypatch, {"steering/params": json.dumps(old)})
    try:
        assert page.sp == SteeringParams()
        assert not page.params_warn.isVisibleTo(page)
        s = QSettings(ini, QSettings.Format.IniFormat)
        assert int(s.value("steering/params_version")) == 3
        assert json.loads(s.value("steering/params"))["R"] == SteeringParams().R == 16.0
    finally:
        _close(app, page)

    # values changed away from the old defaults are kept; the ones still at an old default move.  A custom R saved
    # before version 2 was a physical choice (no 'Fit R' yet): kept as it is, with the default kappa and no note
    custom = dict(old, R=12.0, D=8.0, D_auto=False, beta=40.0, lam=1030.0)
    for sub in ("b", "d"):
        (tmp_path / sub).mkdir()
    app, mod, page, ini = _page(tmp_path / "b", monkeypatch, {"steering/params": json.dumps(custom),
                                                               "steering/params_version": 1})
    try:
        assert page.sp == SteeringParams(R=12.0, theta=45.0, D=8.0, D_auto=False, beta=40.0, lam=1030.0)
        assert not page.params_warn.isVisibleTo(page)
    finally:
        _close(app, page)
    saved = json.loads(QSettings(ini, QSettings.Format.IniFormat).value("steering/params"))
    assert (saved["R"], saved["theta"], saved["D"], saved["beta"]) == (12.0, 45.0, 8.0, 40.0)
    assert saved["tilt_scale"] == SteeringParams().tilt_scale

    # D = 13 in use (D auto off) with a custom R is the user's beam: kept; with R at its old default both move
    in_use = dict(old, R=12.0, D_auto=False)
    assert SteeringParams.from_dict(mod.migrate_params(in_use, 0)) == SteeringParams(R=12.0, D_auto=False, D=13.0)
    assert SteeringParams.from_dict(mod.migrate_params(dict(in_use, R=16.0), 0)) == SteeringParams(D_auto=False)
    assert SteeringParams.from_dict(mod.migrate_params(dict(in_use, D_auto="true"), 1)) == SteeringParams(R=12.0)
    assert mod.migrate_params({"R": 12.0, "theta": 60.0, "D_auto": False, "D": 13.0}, 0) == \
        {"R": 12.0, "theta": 45.0, "D_auto": False, "D": 13.0}
    app, mod, page, ini = _page(tmp_path / "d", monkeypatch, {"steering/params": json.dumps(in_use)})
    try:
        assert page.sp == SteeringParams(R=12.0, D_auto=False, D=13.0)
    finally:
        _close(app, page)


def test_steering_page_migrates_params_saved_by_version_2(tmp_path, monkeypatch):
    """Version 2 had R = 5.332 mm fitted to the spot step and D auto = 4.4 mm.  Its defaults become R = 16 mm and
    D 13.1 mm (D only while unused, or together with R), kappa takes its default; theta = 60 deg was a choice in
    version 2 and stays.  Any other R came from the old 'Fit R' (kappa = 1): it is converted to the kappa that gives
    the same spot step with R = 16 mm, and the page says so once."""
    import json
    from dataclasses import replace
    from PySide6.QtCore import QSettings
    from ltemsim import steering

    # the user's real state: exactly the new defaults afterwards, and no note
    app, mod, page, ini = _page(tmp_path, monkeypatch, {"steering/params": json.dumps(_V2_USER),
                                                         "steering/params_version": 2})
    try:
        assert page.sp == SteeringParams()
        assert not page.params_warn.isVisibleTo(page)
        assert page.calib_result.text() == "κ = 0.33336 · per 0.2 µm step: upper 55.0 nm, lower 59.6 nm"
        assert int(QSettings(ini, QSettings.Format.IniFormat).value("steering/params_version")) == 3
    finally:
        _close(app, page)
    saved = json.loads(QSettings(ini, QSettings.Format.IniFormat).value("steering/params"))
    assert SteeringParams.from_dict(saved) == SteeringParams() and saved["tilt_scale"] == 0.33336

    # rule by rule
    m = mod.migrate_params
    assert SteeringParams.from_dict(m(_V2_USER, 2)) == SteeringParams()
    assert m(dict(_V2_USER, theta=60.0), 2)["theta"] == 60.0                 # a choice in version 2
    assert m({"R": 5.332, "D_auto": False, "D": 4.4}, 2) == {"R": 16.0, "D_auto": False, "D": 13.1}
    k10 = mod.convert_fitted_R({"R": 10.0})["tilt_scale"]                   # a fitted R: kappa ~ R / 16 mm
    assert k10 == pytest.approx(10.0 / 16.0, rel=0.005) and k10 == mod.round_kappa(k10)
    assert m({"R": 10.0, "D_auto": False, "D": 4.4}, 2) == {"R": 16.0, "tilt_scale": k10, "D_auto": False,
                                                            "D": 4.4}         # the beam
    assert m({"R": 10.0, "D_auto": True, "D": 4.4}, 2) == {"R": 16.0, "tilt_scale": k10, "D_auto": True,
                                                           "D": 13.1}         # unused
    assert m({"R": 10.0, "D": 4.4}, 2) == {"R": 16.0, "tilt_scale": k10, "D": 13.1}    # D auto not saved: on
    assert m({"R": 16.0, "D_auto": False, "D": 13.0}, 2) == {"R": 16.0, "D_auto": False, "D": 13.1}
    assert m({"R": 10.0, "D_auto": False, "D": 6.0}, 2) == {"R": 16.0, "tilt_scale": k10, "D_auto": False, "D": 6.0}
    assert m({"tilt_scale": 0.5, "R": 5.332}, 2) == {"tilt_scale": 0.5, "R": 16.0}  # a kappa that is there stays
    assert m({"tilt_scale": 0.5, "R": 10.0}, 2) == {"tilt_scale": 0.5, "R": 10.0}
    assert m({"R": 10.0}, 1) == {"R": 10.0}                 # before version 2 an R was physical: kept, default kappa
    # no kappa in range (or no trace) keeps the fitted R at the nominal tilt: exactly the version-2 model
    assert m({"R": 170.0}, 2) == {"R": 170.0, "tilt_scale": 1.0}
    assert m({"R": 10.0, "L": "abc"}, 2) == {"R": 10.0, "L": "abc", "tilt_scale": 1.0}

    # the converted set moves the spot as the version-2 model did, step for step (the steps of a 'Fit R' to 78 nm of
    # the lower actuator, R = 6.9771 mm, and to 70 nm of the upper one, R = 6.7868 mm); the default kappa with the
    # fitted R would have scaled the step twice (26 / 24 nm)
    for R in (6.9771, 6.7868, 10.0):
        saved = dict(_V2_USER, R=R)
        v2 = mod.step_shifts_nm(SteeringParams.from_dict(dict(saved, tilt_scale=1.0)))
        new = SteeringParams.from_dict(m(saved, 2))
        assert new == SteeringParams(tilt_scale=new.tilt_scale) and new.tilt_scale == pytest.approx(R / 16, rel=0.005)
        got = mod.step_shifts_nm(new)
        assert got["upper"] == pytest.approx(v2["upper"], abs=0.005) and got["lower"] == pytest.approx(v2["lower"],
                                                                                                   abs=0.1)
        assert mod.step_shifts_nm(SteeringParams.from_dict(saved))["upper"] < 0.4 * v2["upper"]
    sp = SteeringParams(R=6.9771, tilt_scale=1.0)
    assert mod.step_shifts_nm(sp)["lower"] == pytest.approx(78.0, abs=0.01)
    assert steering.calibrate_tilt_scale(replace(sp, R=16.0), "lower", 0.2, 0.078) == pytest.approx(
        m(dict(_V2_USER, R=6.9771), 2)["tilt_scale"], rel=1e-3)

    # a v2 user who fitted R: converted, the rest migrated, and a note once (a warning-coloured line)
    from ltemsim_gui.theme import ThemeManager
    (tmp_path / "fit").mkdir()
    fitted = dict(_V2_USER, R=10.0)
    app, mod, page, ini = _page(tmp_path / "fit", monkeypatch, {"steering/params": json.dumps(fitted),
                                                                 "steering/params_version": 2})
    try:
        assert page.sp == SteeringParams(tilt_scale=k10)
        assert page.params_warn.isVisibleTo(page)
        assert ThemeManager.instance().tokens.warning in page.params_warn.styleSheet()
        assert page.params_warn.text() == mod.migration_note(fitted, m(fitted, 2))
        assert page.params_warn.text() == (f"Saved R = 10 mm was fitted to the spot step (the old 'Fit R'). The spot "
                                           f"step is now calibrated by κ: converted to κ = {k10:g} with R = 16 mm, "
                                           "the physical focal length. The spot steps are unchanged.")
        assert page.calib_result.text().startswith(f"κ = {k10:g} · per 0.2 µm step: upper 103.1 nm, lower 111.8 nm")
        page.form.rows["beta"].editor.setText("40")                 # gone once the parameters are next applied
        page._apply_params()
        assert not page.params_warn.isVisibleTo(page)
        page.form.rows["L"].editor.setText("0")                     # a later warning shows as before
        page._apply_params()
        assert page.params_warn.isVisibleTo(page) and "Lever arm L" in page.params_warn.text()
        page.form.reset_field("L")
    finally:
        _close(app, page)
    app, mod, page, ini = _page(tmp_path / "fit", monkeypatch)        # one time only: saved as version 3 now
    try:
        assert page.sp == SteeringParams(tilt_scale=k10, beta=40.0) and not page.params_warn.isVisibleTo(page)
    finally:
        _close(app, page)

    # the conversion failed: the fitted R stays, at the nominal tilt, and the note says so
    (tmp_path / "keep").mkdir()
    app, mod, page, ini = _page(tmp_path / "keep", monkeypatch, {"steering/params": json.dumps(dict(_V2_USER, R=170.0)),
                                                                  "steering/params_version": 2})
    try:
        assert page.sp == SteeringParams(R=170.0, tilt_scale=1.0)
        assert page.params_warn.text().startswith("Saved R = 170 mm was fitted to the spot step (the old 'Fit R'). "
                                                  "It could not be converted") and "κ = 1:" in page.params_warn.text()
    finally:
        _close(app, page)

    # R = 16 saved by version 2 (the module's value) needs no note; an unusable R gets the drop warning only
    for i, (params, warned) in enumerate(((dict(_V2_USER, R=16.0), False), (dict(_V2_USER, R="abc"), True))):
        (tmp_path / f"r{i}").mkdir()
        app, mod, page, ini = _page(tmp_path / f"r{i}", monkeypatch, {"steering/params": json.dumps(params),
                                                                       "steering/params_version": 2})
        try:
            assert page.sp == SteeringParams()
            assert page.params_warn.isVisibleTo(page) == warned
            assert "Fit R" not in page.params_warn.text()
        finally:
            _close(app, page)


def test_steering_page_keeps_params_saved_by_version_3(tmp_path, monkeypatch):
    """Saved by this version, every value is the user's choice: R = 5.332 mm, theta 60, D 4.4 and kappa stay."""
    import json
    from ltemsim import steering
    from ltemsim_gui.pages import steering as mod
    mine = SteeringParams(R=5.332, theta=60.0, D=4.4, D_auto=False, tilt_scale=0.5)
    assert mod.migrate_params(mod._form_values(mine), 3) == mod._form_values(mine)
    for i, sp in enumerate((mine, steering.module_defaults())):
        (tmp_path / str(i)).mkdir()
        saved = {"steering/params": json.dumps(mod._form_values(sp)), "steering/params_version": 3}
        app, mod, page, ini = _page(tmp_path / str(i), monkeypatch, saved)
        try:
            assert page.sp == sp and not page.params_warn.isVisibleTo(page)
        finally:
            _close(app, page)


def test_steering_page_migrates_a_preset_loaded_without_kappa(tmp_path, monkeypatch):
    """Load… in the parameter form: presets carry no version, so one without tilt_scale (saved before version 3) is
    migrated as a version-2 state: R 5.332 mm becomes 16 mm (no note), a fitted R becomes kappa (with the note).  A
    preset with tilt_scale is taken as it is; an unreadable file is a warning."""
    import json
    from PySide6.QtWidgets import QFileDialog, QPushButton
    v2 = tmp_path / "v2.json"
    v2.write_text(json.dumps(dict(_V2_USER, D_auto=False)))         # R 5.332 with D 4.4 in use: both move
    fitted = tmp_path / "fitted.json"
    fitted.write_text(json.dumps({"params": dict(_V2_USER, R=10.0, beta=40.0)}))      # a params.json layout
    v3 = tmp_path / "v3.json"
    mine = SteeringParams(R=5.332, theta=60.0, D=4.4, D_auto=False, tilt_scale=0.5)
    v3.write_text(json.dumps(mine.to_dict()))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    (tmp_path / "ini").mkdir()
    app, mod, page, ini = _page(tmp_path / "ini", monkeypatch)
    try:
        # the form's Load… button opens the page's loader
        monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(v2), "")))
        load = next(b for b in page.form.findChildren(QPushButton) if b.text() == "Load…")
        load.click()
        assert page.sp == SteeringParams(D_auto=False) and not page.params_warn.isVisibleTo(page)

        page.load_preset(str(fitted))
        m = mod.migrate_params(dict(_V2_USER, R=10.0, beta=40.0), 2)
        assert page.sp == SteeringParams(beta=40.0, tilt_scale=m["tilt_scale"])
        assert page.params_warn.isVisibleTo(page)
        assert page.params_warn.text() == mod.migration_note(dict(_V2_USER, R=10.0, beta=40.0), m)
        assert page.params_warn.text().startswith("Saved R = 10 mm was fitted to the spot step")

        page.load_preset(str(v3))
        assert page.sp == mine and not page.params_warn.isVisibleTo(page)

        page.load_preset(str(bad))
        assert page.params_warn.text().startswith("⚠ Could not load the preset: ") and page.sp == mine
    finally:
        _close(app, page)


def test_steering_page_fits_kappa_from_a_measured_step(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from ltemsim import steering
    app, mod, page, ini = _page(tmp_path, monkeypatch)
    try:
        # the group is the tilt calibration; the defaults reproduce steering.CALIBRATION
        c = steering.CALIBRATION
        assert c["fits"] == "tilt_scale"
        box = page.calib_combo.parentWidget()
        assert box.title() == "Calibrate the spot shift from a measured step"
        from PySide6.QtWidgets import QAbstractButton
        assert "Fit κ" in [b.text() for b in box.findChildren(QAbstractButton)]
        note = mod.calib_note()
        assert note.startswith("Default κ = 0.33336: the fraction of the nominal KS1-Z8 tilt (travel / L)")
        assert "R stays the physical focal length" in note and "camera calibration" in note
        assert "R stays the physical focal length" in mod.CALIB_HELP and c["source"] in mod.CALIB_HELP
        assert page.calib_combo.currentData() == c["actuator"] == "upper"
        assert page.calib_travel.value() == pytest.approx(c["travel_um"])
        assert page.calib_shift.value() == pytest.approx(c["spot_um"] * 1e3)
        page._fit_kappa()
        assert page.sp == SteeringParams() and not page.calib_warn.isVisibleTo(page)
        assert page.calib_result.text() == "κ = 0.33336 · per 0.2 µm step: upper 55.0 nm, lower 59.6 nm"

        # the same 55 nm read as a step of the lower actuator: kappa (5 decimals) goes into the form, R stays, and
        # everything follows it
        page.calib_combo.setCurrentIndex(page.calib_combo.findData("lower"))
        page._fit_kappa()
        k = mod.round_kappa(steering.calibrate_tilt_scale(SteeringParams(), "lower", 0.2, 0.055))
        assert k == 0.30738                                 # 5 significant figures
        assert page.form.rows["tilt_scale"].editor.text() == "0.30738" and page.sp.tilt_scale == k
        assert page.sp == SteeringParams(tilt_scale=k)
        assert page.calib_result.text() == "κ = 0.30738 · per 0.2 µm step: upper 50.7 nm, lower 55.0 nm"
        assert "0.2 µm step → 55.0 nm" in page.act["lower"].per.text()

        # no kappa fits: a warning line, kappa unchanged; editing the inputs clears it
        page.calib_shift.setValue(1e6)
        page._fit_kappa()
        assert page.calib_warn.isVisibleTo(page)
        assert page.calib_warn.text().startswith("⚠ Fit κ failed: the lower actuator cannot move the spot")
        assert page.sp.tilt_scale == k
        page.calib_shift.setValue(60.0)
        assert not page.calib_warn.isVisibleTo(page)

        # an invalid kappa is replaced by the fit; another invalid field blocks it
        page.form.rows["tilt_scale"].editor.setText("abc")
        page._fit_kappa()
        assert page.form.rows["tilt_scale"].valid and not page.calib_warn.isVisibleTo(page)
        assert page.sp.tilt_scale == mod.round_kappa(steering.calibrate_tilt_scale(SteeringParams(), "lower", 0.2,
                                                                                     0.060))
        page.form.rows["NA"].editor.setText("2")
        page._fit_kappa()
        assert "Aperture NA" in page.calib_warn.text() and page.calib_warn.isVisibleTo(page)
        page.form.reset_field("NA")

        # a kappa outside the field's range is refused
        kept = page.sp.tilt_scale
        page.calib_travel.setValue(0.001)
        page.calib_shift.setValue(50.0)
        page._fit_kappa()
        assert page.calib_warn.text() == "⚠ Fit κ: κ = 55.9 is outside the allowed 0.001–10; κ is unchanged."
        assert page.sp.tilt_scale == kept
        page.calib_travel.setValue(500.0)                   # a tiny kappa is shown as fitted, not rounded to 0
        page.calib_shift.setValue(0.1)
        page._fit_kappa()
        assert page.calib_warn.text() == "⚠ Fit κ: κ = 2.24e-07 is outside the allowed 0.001–10; κ is unchanged."
        assert page.sp.tilt_scale == kept
        # a shift just short of the mirror edge fits (the bracket of the fit jumps past the edge there): kappa = 0.9
        # moves the spot 363.85 um with +500 um of the lower actuator, the end of the spin box
        shift = steering.steer(SteeringParams(tilt_scale=0.9), 500.0, 0.0).shift_um
        assert shift == pytest.approx(363.85, abs=0.01)
        page.calib_shift.setValue(round(shift * 1e3, 1))
        page._fit_kappa()
        assert not page.calib_warn.isVisibleTo(page) and page.sp.tilt_scale == pytest.approx(0.9, abs=1e-5)
        page.form.set_values({"tilt_scale": kept})
        page._apply_params()
        assert page.sp.tilt_scale == kept
        page.calib_travel.setValue(0.4)
        page.calib_shift.setValue(60.0)
        page._save()
    finally:
        _close(app, page)
    s = QSettings(ini, QSettings.Format.IniFormat)
    assert (s.value("steering/calib_actuator"), float(s.value("steering/calib_travel_um")),
            float(s.value("steering/calib_spot_nm"))) == ("lower", 0.4, 60.0)
    app, mod, page, ini = _page(tmp_path, monkeypatch)                      # restored on the next start
    try:
        assert page.calib_combo.currentData() == "lower"
        assert (page.calib_travel.value(), page.calib_shift.value()) == (0.4, 60.0)
    finally:
        _close(app, page)


def test_steering_page_restores_the_usable_saved_values(tmp_path, monkeypatch):
    """One malformed value in steering/params costs that value only: the others are restored, the page says which
    value went back to its default, and the next save keeps the restored ones."""
    import json
    from PySide6.QtCore import QSettings
    for i, bad in enumerate(({"R": "abc", "beta": 40}, {"R": None, "theta": 60.0, "beta": 40},
                             {"yawSide": [1], "beta": 40, "lam": 1030.0})):
        (tmp_path / str(i)).mkdir()
        app, mod, page, ini = _page(tmp_path / str(i), monkeypatch,
                                    {"steering/params": json.dumps(bad), "steering/params_version": 2})
        try:
            assert page.sp.beta == 40.0 and page.form.is_valid()
            assert page.sp == SteeringParams.from_dict({k: v for k, v in bad.items() if k in ("beta", "theta", "lam")})
            assert page.params_warn.isVisibleTo(page)
            assert page.params_warn.text().startswith("⚠ Saved value(s) not usable, back to the default: ")
            label = page.form.rows["yawSide" if "yawSide" in bad else "R"].spec.label
            assert page.params_warn.text().endswith(f"{label}.")
            page._save()
        finally:
            _close(app, page)
        saved = json.loads(QSettings(ini, QSettings.Format.IniFormat).value("steering/params"))
        assert saved["beta"] == 40.0

    # the note stays until the parameters are next applied
    app, mod, page, ini = _page(tmp_path / "0", monkeypatch, {"steering/params": json.dumps({"R": "abc"})})
    try:
        assert page.params_warn.isVisibleTo(page)
        page.form.rows["beta"].editor.setText("40")
        page._apply_params()
        assert not page.params_warn.isVisibleTo(page)
    finally:
        _close(app, page)


def test_steering_page_fits_kappa_from_a_camera_step(tmp_path, monkeypatch):
    """The ~50 um camera calibration the help recommends: an extend or a retract of either actuator, fitted from the
    default kappa or from the nominal 1, gives back the kappa that produced the shift and leaves R alone.  The travel
    is signed (a saved retract is restored as such; a retract entered as an extend fits a different kappa); zero
    travel is refused; the warning goes when the parameters change."""
    from ltemsim import steering
    app, mod, page, ini = _page(tmp_path, monkeypatch, {"steering/calib_travel_um": -5.0})
    try:
        assert page.calib_travel.value() == -5.0                  # restored signed, not clamped
        assert page.calib_travel.minimum() == -mod.SPIN_UM
        k_true = 0.5
        for k_start in (SteeringParams().tilt_scale, 1.0):
            for actuator in ("upper", "lower"):
                for travel in (50.0, -50.0):
                    page.form.set_values({"tilt_scale": k_start})
                    page._apply_params()
                    tr = (travel, 0.0) if actuator == "lower" else (0.0, travel)
                    shift_nm = steering.steer(SteeringParams(tilt_scale=k_true), *tr).shift_um * 1e3
                    page.calib_combo.setCurrentIndex(page.calib_combo.findData(actuator))
                    page.calib_travel.setValue(travel)
                    page.calib_shift.setValue(round(shift_nm, 1))
                    page._fit_kappa()
                    assert not page.calib_warn.isVisibleTo(page), (k_start, actuator, travel, page.calib_warn.text())
                    assert page.sp.tilt_scale == pytest.approx(k_true, abs=2e-5), (k_start, actuator, travel)
                    assert page.sp.R == 16.0
                    # the step lands where it was measured
                    assert steering.steer(page.sp, *tr).shift_um * 1e3 == pytest.approx(shift_nm, abs=0.1)
                    # the same shift entered with the wrong sign fits another kappa (the response is asymmetric)
                    page.calib_travel.setValue(-travel)
                    page._fit_kappa()
                    assert abs(page.sp.tilt_scale - k_true) > 0.01, (actuator, travel)

        # zero travel: a warning, kappa unchanged
        k = page.sp.tilt_scale
        page.calib_travel.setValue(0.0)
        page._fit_kappa()
        assert page.calib_warn.text() == "⚠ Enter a non-zero actuator travel." and page.sp.tilt_scale == k

        # a warning about the parameters goes when they are fixed in the form
        page.calib_travel.setValue(50.0)
        page.form.rows["NA"].editor.setText("2")
        page._fit_kappa()
        assert "Aperture NA" in page.calib_warn.text() and page.calib_warn.isVisibleTo(page)
        page.form.rows["NA"].editor.setText("0.42")
        page._apply_params()
        assert not page.params_warn.isVisibleTo(page) and not page.calib_warn.isVisibleTo(page)
        # ... and so does a failed fit's warning when R or theta is edited
        page.calib_shift.setValue(1e6)
        page._fit_kappa()
        assert page.calib_warn.text().startswith("⚠ Fit κ failed: the ")
        assert "f(a)" not in page.calib_warn.text() and "leaves the parabolic mirror" in page.calib_warn.text()
        page.form.rows["theta"].editor.setText("60")
        assert not page.calib_warn.isVisibleTo(page)
    finally:
        _close(app, page)


def _drawn(fig):
    """(line data, texts) of every axes of a figure, insets included."""
    import numpy as np
    lines, texts = [], []

    def walk(ax):
        lines.extend(np.asarray(ln.get_xydata(), float) for ln in ax.get_lines())
        texts.extend(t.get_text() for t in ax.texts)
        for ch in ax.child_axes:
            walk(ch)
    for ax in fig.axes:
        walk(ax)
    return lines, texts


def test_schematics_trace_the_effective_tilt():
    """The Setup views trace kappa * travel (steering.core_travel), as the adapter does: kappa = 0.5 with 8 um
    draws what the nominal tilt draws with 4 um; the mirror view quotes the effective tilt and the real travel."""
    import numpy as np
    from matplotlib.figure import Figure
    from ltemsim import steering
    from ltemsim_gui.theme import LIGHT
    from ltemsim_gui.widgets import steering_views as views

    def draw(sp, lower, upper, view):
        fig = Figure(figsize=(9, 5))
        views.draw_schematic(fig, sp, lower, upper, view, 100, LIGHT)
        return _drawn(fig)

    half, nominal = SteeringParams(tilt_scale=0.5), SteeringParams(tilt_scale=1.0)
    for view in ("top", "inside", "side"):
        (la, ta), (lb, tb) = draw(half, 8.0, -4.0, view), draw(nominal, 4.0, -2.0, view)
        assert len(la) == len(lb) and all(a.shape == b.shape and np.allclose(a, b) for a, b in zip(la, lb)), view
        assert ta == tb, view
        # the nominal tilt at the same travel differs (in the top view the walk gain may halve instead: ×N text)
        lc, tc = draw(nominal, 8.0, -4.0, view)
        assert ta != tc or not all(a.shape == c.shape and np.allclose(a, c) for a, c in zip(la, lc)), view
    # the inside view's inset quotes the adapter's shift
    sp = SteeringParams()
    _, texts = draw(sp, 4.0, 0.0, "inside")
    assert f"shift {steering.steer(sp, 4.0, 0.0).shift_um:.2f} µm" in texts and "shift 1.19 µm" in texts

    # mirror view: effective tilt kappa * travel / L, labelled as such; KDC101 from the real travel
    _, texts = draw(sp, 4.0, 0.0, "mirror")
    d = sp.tilt_scale * 4.0 / sp.L                                  # mrad
    assert f"{d * 1e3:.1f}" == "35.4" and f"{d * 206.264806:.1f}" == "7.3"
    lower = next(t for t in texts if t.startswith("travel +4.000 µm"))
    assert "δ = +35.4 µrad = +7.3″ (κ × nominal)" in lower and "+138 encoder counts" in lower
    assert "lower Z812  +4.00 µm" in texts
    assert any("δ = κ × travel / L (effective tilt, κ = 0.33336 × nominal)" in t for t in texts)
    _, texts = draw(steering.module_defaults(), 4.0, 0.0, "mirror")
    assert any("δ = +106.1 µrad" in t for t in texts) and any("κ = 1 × nominal" in t for t in texts)
