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


def _app():
    from PySide6.QtWidgets import QApplication
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
    assert {s.name for s in specs if not s.advanced} == {"R", "theta", "beta", "NA", "L", "lam", "D_auto", "D",
                                                         "bias_axis_deg"}
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


def test_3d_page_values_show_the_calibrated_defaults():
    """The reference page's R slider (2-30 mm in 0.01 mm) holds the calibrated R: no banner with the defaults."""
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
    assert (vals["pR"], vals["pTheta"]) == ("5.33", "45")
    assert page_values(0.0, 0.0, steering.module_defaults())[2] == []
    # clipping and visible roundings are still reported
    assert page_values(0.0, 0.0, SteeringParams(R=1.5))[2] == ["R 1.5 mm clipped to 2–30 mm"]
    assert page_values(0.0, 0.0, SteeringParams(R=2.004))[2] == ["R 2.004 mm shown as 2 mm (slider step 0.01)"]
    assert page_values(0.0, 0.0, SteeringParams(theta=38.0))[2] == ["θ 38° clipped to 40–72°"]
    assert page_values(0.0, 0.0, SteeringParams(D_auto=False))[2] == ["D 4.4 mm shown as 4.5 mm (slider step 0.5)"]


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
        assert page.calib_result.text() == "R = 5.332 mm · per 0.2 µm step: upper 55.0 nm, lower 59.6 nm"
        assert "0.2 µm step → 55.0 nm" in page.act["upper"].per.text()
        assert "0.2 µm step → 59.6 nm" in page.act["lower"].per.text()

        page.act["lower"].spin.setValue(4.0)
        # chief-ray readouts are synchronous (calibrated defaults: lower 4 um -> spot (-1.124, -0.359) um)
        assert page.ro["x"].text() == "left 1.12 µm"
        assert page.ro["y"].text() == "front 0.36 µm"
        assert page.ro["X"].text() == "−1.12 µm"
        assert page.act["lower"].slider.value() == 40
        assert "+138 counts" in page.act["lower"].kdc.text()

        # the fine spot at (4, 0) arrives from the worker thread
        assert _wait(app, lambda: page.cur_at == (4.0, 0.0, "fine") and page.cur is not None), page.shape_status.text()
        assert page.shape_ro["fwhm"].text() == "1.55 × 1.13 µm"
        assert page.shape_ro["peak"].text() == "96 %"
        assert page.D == pytest.approx(4.4)
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
        # the travel ranges cover the X list's lower-only travels (X = -7.5 um: +27.6 um), so the slider holds it
        assert max(abs(r["lower_um"]) for r in page.link_rows) < mod.SLIDER_UM
        assert min(mod.CURVE_TRAVELS) <= -max(abs(r["lower_um"]) for r in page.link_rows)
        assert max(mod.CURVE_TRAVELS) >= max(abs(r["lower_um"]) for r in page.link_rows)
        page.link_table.selectRow(0)
        page._move_to_row()
        assert page.st.X_um == pytest.approx(-7.5, abs=0.01)
        assert page.act["lower"].slider.value() == 276

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
    assert float(s.value("steering/lower_um")) == pytest.approx(27.626, abs=0.01)
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
        assert page.calib_result.text().startswith("R = 16 mm · per 0.2 µm step: upper ")
    finally:
        _close(app, page)


def test_steering_page_migrates_params_saved_with_the_old_defaults(tmp_path, monkeypatch):
    """steering/params saved before params_version 2 hold the module's defaults R 16, theta 60, D 13: those become the
    calibrated defaults, every value the user changed survives, and the migrated set is saved with version 2."""
    import json
    from PySide6.QtCore import QSettings
    from ltemsim import steering
    from ltemsim_gui.pages import steering as mod

    # the user's settings: everything at the old defaults (no version key)
    old = json.dumps(mod._form_values(steering.module_defaults()))
    app, mod, page, ini = _page(tmp_path, monkeypatch, {"steering/params": old})
    try:
        assert page.sp == SteeringParams()
        s = QSettings(ini, QSettings.Format.IniFormat)
        assert int(s.value("steering/params_version")) == 2
        assert json.loads(s.value("steering/params"))["R"] == SteeringParams().R
    finally:
        _close(app, page)

    # values changed away from the old defaults are kept; the ones still at an old default move
    custom = dict(json.loads(old), R=12.0, D=8.0, D_auto=False, beta=40.0, lam=1030.0)
    for sub in ("b", "c"):
        (tmp_path / sub).mkdir()
    app, mod, page, ini = _page(tmp_path / "b", monkeypatch, {"steering/params": json.dumps(custom),
                                                               "steering/params_version": 1})
    try:
        assert page.sp == SteeringParams(R=12.0, theta=45.0, D=8.0, D_auto=False, beta=40.0, lam=1030.0)
    finally:
        _close(app, page)
    saved = json.loads(QSettings(ini, QSettings.Format.IniFormat).value("steering/params"))
    assert (saved["R"], saved["theta"], saved["D"], saved["beta"]) == (12.0, 45.0, 8.0, 40.0)

    # D = 13 in use (D auto off) with a custom R is the user's beam: kept; with R at its old default both move
    in_use = dict(json.loads(old), R=12.0, D_auto=False)
    assert SteeringParams.from_dict(mod.migrate_params(in_use, 0)) == SteeringParams(R=12.0, D_auto=False, D=13.0)
    assert SteeringParams.from_dict(mod.migrate_params(dict(in_use, R=16.0), 0)) == SteeringParams(D_auto=False)
    assert SteeringParams.from_dict(mod.migrate_params(dict(in_use, D_auto="true"), 1)) == SteeringParams(R=12.0)
    assert mod.migrate_params({"R": 12.0, "theta": 60.0, "D_auto": False, "D": 13.0}, 0) == \
        {"R": 12.0, "theta": 45.0, "D_auto": False, "D": 13.0}
    (tmp_path / "d").mkdir()
    app, mod, page, ini = _page(tmp_path / "d", monkeypatch, {"steering/params": json.dumps(in_use)})
    try:
        assert page.sp == SteeringParams(R=12.0, D_auto=False, D=13.0)
    finally:
        _close(app, page)

    # saved by this version: R = 16 is the user's choice now and stays
    app, mod, page, ini = _page(tmp_path / "c", monkeypatch, {"steering/params": old, "steering/params_version": 2})
    try:
        assert page.sp == steering.module_defaults()
    finally:
        _close(app, page)


def test_steering_page_fits_R_from_a_measured_step(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from ltemsim import steering
    app, mod, page, ini = _page(tmp_path, monkeypatch)
    try:
        # the defaults reproduce steering.CALIBRATION
        c = steering.CALIBRATION
        assert page.calib_combo.currentData() == c["actuator"] == "upper"
        assert page.calib_travel.value() == pytest.approx(c["travel_um"])
        assert page.calib_shift.value() == pytest.approx(c["spot_um"] * 1e3)
        page._fit_R()
        assert page.sp.R == pytest.approx(SteeringParams().R) and not page.calib_warn.isVisibleTo(page)
        assert page.calib_result.text() == "R = 5.332 mm · per 0.2 µm step: upper 55.0 nm, lower 59.6 nm"

        # the same 55 nm read as a step of the lower actuator: R goes into the form and everything follows it
        page.calib_combo.setCurrentIndex(page.calib_combo.findData("lower"))
        page._fit_R()
        R = round(steering.calibrate_R(SteeringParams(), "lower", 0.2, 0.055), 4)
        assert page.form.rows["R"].editor.text() == f"{R:g}" and page.sp.R == R
        assert page.calib_result.text() == f"R = {R:g} mm · per 0.2 µm step: upper 50.8 nm, lower 55.0 nm"
        assert "0.2 µm step → 55.0 nm" in page.act["lower"].per.text()

        # no focal length fits: a warning line, R unchanged; editing the inputs clears it
        page.calib_shift.setValue(1e6)
        page._fit_R()
        assert page.calib_warn.isVisibleTo(page) and page.calib_warn.text().startswith("⚠ Fit R failed")
        assert page.sp.R == R
        page.calib_shift.setValue(60.0)
        assert not page.calib_warn.isVisibleTo(page)

        # an invalid R is replaced by the fit; another invalid field blocks it
        page.form.rows["R"].editor.setText("abc")
        page._fit_R()
        assert page.form.rows["R"].valid and not page.calib_warn.isVisibleTo(page)
        assert page.sp.R == round(steering.calibrate_R(SteeringParams(), "lower", 0.2, 0.060), 4)
        page.form.rows["NA"].editor.setText("2")
        page._fit_R()
        assert "Aperture NA" in page.calib_warn.text() and page.calib_warn.isVisibleTo(page)
        page.form.reset_field("NA")
        page.calib_travel.setValue(0.4)
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


def test_steering_page_fits_R_from_a_camera_step(tmp_path, monkeypatch):
    """The ~50 um camera calibration the help recommends: an extend or a retract of either actuator, fitted from the
    default R or from R = 16 mm, gives back the R that produced the shift.  The travel is signed (a saved retract is
    restored as such); zero travel is refused; the warning goes when the parameters change."""
    from ltemsim import steering
    app, mod, page, ini = _page(tmp_path, monkeypatch, {"steering/calib_travel_um": -5.0})
    try:
        assert page.calib_travel.value() == -5.0                  # restored signed, not clamped
        assert page.calib_travel.minimum() == -mod.SPIN_UM
        for R_start in (SteeringParams().R, 16.0):
            for actuator in ("upper", "lower"):
                for travel in (50.0, -50.0):
                    page.form.set_values({"R": R_start})
                    page._apply_params()
                    R_true = 10.0
                    tr = (travel, 0.0) if actuator == "lower" else (0.0, travel)
                    shift_nm = steering.steer(SteeringParams(R=R_true), *tr).shift_um * 1e3
                    page.calib_combo.setCurrentIndex(page.calib_combo.findData(actuator))
                    page.calib_travel.setValue(travel)
                    page.calib_shift.setValue(round(shift_nm, 1))
                    page._fit_R()
                    assert not page.calib_warn.isVisibleTo(page), (R_start, actuator, travel, page.calib_warn.text())
                    assert page.sp.R == pytest.approx(R_true, abs=2e-3), (R_start, actuator, travel)

        # zero travel: a warning, R unchanged
        R = page.sp.R
        page.calib_travel.setValue(0.0)
        page._fit_R()
        assert page.calib_warn.text() == "⚠ Enter a non-zero actuator travel." and page.sp.R == R

        # a warning about the parameters goes when they are fixed in the form
        page.calib_travel.setValue(50.0)
        page.form.rows["NA"].editor.setText("2")
        page._fit_R()
        assert "Aperture NA" in page.calib_warn.text() and page.calib_warn.isVisibleTo(page)
        page.form.rows["NA"].editor.setText("0.42")
        page._apply_params()
        assert not page.params_warn.isVisibleTo(page) and not page.calib_warn.isVisibleTo(page)
        # ... and so does a failed fit's warning when R or theta is edited
        page.calib_shift.setValue(1e6)
        page._fit_R()
        assert page.calib_warn.text().startswith("⚠ Fit R failed: no focal length between")
        assert "f(a)" not in page.calib_warn.text() and " µm for 50 µm of the lower actuator" in page.calib_warn.text()
        page.form.rows["theta"].editor.setText("60")
        assert not page.calib_warn.isVisibleTo(page)
    finally:
        _close(app, page)
