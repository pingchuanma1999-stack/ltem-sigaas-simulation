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

        page.act["lower"].spin.setValue(4.0)
        # chief-ray readouts are synchronous (README: 4 um on the lower actuator -> spot (-3.387, -1.272) um)
        assert page.ro["x"].text() == "left 3.39 µm"
        assert page.ro["y"].text() == "front 1.27 µm"
        assert page.ro["X"].text() == "−3.39 µm"
        assert page.act["lower"].slider.value() == 40
        assert "+138 counts" in page.act["lower"].kdc.text()

        # the fine spot at (4, 0) arrives from the worker thread
        assert _wait(app, lambda: page.cur_at == (4.0, 0.0, "fine") and page.cur is not None), page.shape_status.text()
        assert page.shape_ro["fwhm"].text() == "2.18 × 1.27 µm"
        assert page.shape_ro["peak"].text() == "78 %"
        assert page.D == pytest.approx(13.0)
        assert _wait(app, lambda: isinstance(page.curves, dict) and isinstance(page.gallery, dict)
                     and isinstance(page.link_rows, list))
        assert page.link_table.rowCount() == 7
        assert "at ±5 µm, peak" in page.act["lower"].per.text()

        # the slider pins at its end beyond ±20 um
        page.act["upper"].spin.setValue(30.0)
        assert page.act["upper"].slider.value() == page.act["upper"].slider.maximum()

        for i in range(page.tabs.count()):
            page.tabs.setCurrentIndex(i)
            _pump(app, 0.3)
        assert _wait(app, lambda: page.queue.idle())
        for i in (0, 1, 2, 4):                  # draw the plots once more with every result in
            page.tabs.setCurrentIndex(i)
            _pump(app, 0.2)
        assert not _plot_errors(page), _plot_errors(page)

        # moving to a row of the link table puts the spot at that X (travels rounded to the spin box's 0.01 um)
        page.link_table.selectRow(0)
        page._move_to_row()
        assert page.st.X_um == pytest.approx(-7.5, abs=0.01)

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
    assert float(s.value("steering/lower_um")) == pytest.approx(8.885, abs=0.01)


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
        assert page.sp.D_auto is True and page.sp.D == 13.0   # the last valid D is kept
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
        page.x_edit.setText("-200, -5")
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
