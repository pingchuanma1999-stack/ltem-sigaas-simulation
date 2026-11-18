"""Checks of the pump-spot steering module: the vendored optics core against its golden file, and the adapter
ltemsim/steering.py (actuator names, signs, frames, solver, parameter handling) against the module's conventions.

Run with:  python -m pytest tests/test_steering.py -q

The golden part mirrors ltem-steering-module/python/test_golden.py, but goes through steering.optics (the core as
this package loads it) and, a second time, through the adapter, so that a swapped actuator, a dropped parameter or
a sign flip in the adapter cannot hide behind a passing core.

Two parameter sets: numbers the module documents (its handoff and README: R = 16 mm, theta = 60 deg, the nominal
mirror tilt travel / L) are checked on steering.module_defaults() (tilt_scale = 1).  SteeringParams() keeps the
physical R = 16 mm, assumes theta = 45 deg and carries the calibrated tilt_scale (steering.CALIBRATION: 55 nm per
0.2 um step of the upper actuator), checked in its own section; what tilt_scale does (the core is handed
tilt_scale * travel, the API keeps the real travel) has a section of its own.  Tests of adapter behaviour that does
not depend on the values run on SteeringParams().
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import fields, replace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ltemsim import steering  # noqa: E402
from ltemsim.steering import SteeringParams  # noqa: E402

O = steering.optics
GOLDEN = json.loads(steering.GOLDEN_PATH.read_text(encoding="utf-8"))
TOL = GOLDEN["tolerance"]
CASES = list(GOLDEN["cases"].items())
CASE_IDS = [c[0] for c in CASES]


# ------------------------------------------------------------------ comparison helpers (as in test_golden.py)
def _close(a, b, rel, abs_):
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def _cmp(path, got, exp, rel, abs_, errors):
    """Walk the expected structure and record every mismatch (None, bool and str must match exactly)."""
    if exp is None:
        if got is not None:
            errors.append(f"{path}: expected None, got {got!r}")
        return
    if isinstance(exp, (bool, str)):
        if got != exp:
            errors.append(f"{path}: got {got!r}, expected {exp!r}")
        return
    if isinstance(exp, (int, float)):
        if (isinstance(got, bool) or not isinstance(got, (int, float, np.floating, np.integer))
                or not _close(float(got), float(exp), rel, abs_)):
            errors.append(f"{path}: got {got!r}, expected {exp!r}")
        return
    if isinstance(exp, list):
        for i, e in enumerate(exp):
            _cmp(f"{path}[{i}]", got[i] if got is not None and i < len(got) else None, e, rel, abs_, errors)
        return
    for k, e in exp.items():
        _cmp(f"{path}.{k}", got.get(k) if isinstance(got, dict) else None, e, rel, abs_, errors)


def _summary(s):
    """The scalar summary of a spot that golden.json stores."""
    return {
        "M": s["M"], "W": s["W"], "N": s["N"], "nr": s["nr"], "rel": s["rel"], "peak": s["peak"], "IperP": s["IperP"],
        "fwhmL": s["fwhmL"], "fwhmS": s["fwhmS"], "ang": s["ang"], "area": s["area"], "peakAt": s["peakAt"],
        "cx": s["cx"], "cy": s["cy"], "border": s["border"], "x0_mm": s["x0"],
        "I_sum": float(np.sum(s["I"].astype(np.float64))),
        "halfmax_segments": int(steering.half_max_contour(s).shape[0]),
    }


def _check_spot(tag, got, case_spot, errors):
    exp = dict(case_spot["result"])
    samples = exp.pop("I_samples")
    _cmp(tag, _summary(got), exp, TOL["spot_rel"], TOL["spot_abs"], errors)
    for smp in samples:
        _cmp(f"{tag}.I[{smp['row']},{smp['col']}]", float(got["I"][smp["row"], smp["col"]]), smp["value"],
             TOL["spot_rel"], 1e-9, errors)


def _params_for(case) -> SteeringParams:
    """SteeringParams equivalent to a golden case's core params: the case's values on top of the module's defaults
    (a case lists only what it changes; the core fills in the rest), and a numeric D switches D_auto off."""
    d = dict(case["params_in"])
    if "D" in d and d["D"] not in ("auto", None):
        d["D_auto"] = False
    sp = SteeringParams.from_dict({**steering.module_defaults().to_dict(), **d})
    assert sp.tilt_scale == 1.0                                    # the core's nominal tilt
    return sp


def _as_core_response(st: steering.Steering) -> dict:
    """An adapter Steering in the core's naming (lower = yaw, upper = pitch), to compare with golden.json."""
    return {"spot_um": st.spot_um, "port_um": st.port_um,
            "perUm": {"yaw": st.per_um["lower"], "pitch": st.per_um["upper"]},
            "perUmPort": {"yaw": st.port_per_um["lower"], "pitch": st.port_per_um["upper"]},
            "incidencePlane": st.incidence_plane, "along_um": st.along_um, "across_um": st.across_um,
            "defocus_um": st.defocus_um, "arrival_fs": st.arrival_fs,
            "yawTurnsBeamToFront": st.lower_turns_beam_to_front}


def _travels(actuator: str, travel_um: float) -> tuple[float, float]:
    """(lower_um, upper_um) moving one actuator only."""
    return (travel_um, 0.0) if actuator == "lower" else (0.0, travel_um)


# ------------------------------------------------------------------ golden: the core as loaded by the adapter
@pytest.mark.parametrize("name,case", CASES, ids=CASE_IDS)
def test_golden_beam_diameter(name, case):
    p = O.resolve_params(case["params_in"])
    assert abs(p["D"] - case["params_resolved"]["D"]) <= TOL["D_abs_mm"]
    assert abs(O.optimal_beam_diameter(p) - case["optimal_D_mm"]) <= 1e-6


@pytest.mark.parametrize("name,case", CASES, ids=CASE_IDS)
def test_golden_steering_response(name, case):
    p = O.resolve_params(case["params_in"])
    errors = []
    for r in case["responses"]:
        got = O.steering_response(p, r["yaw_um"], r["pitch_um"])
        _cmp(f"{name}.response({r['yaw_um']},{r['pitch_um']})", got, r["result"], 1e-9, TOL["response_abs_um"], errors)
    assert not errors, "\n".join(errors[:20])


@pytest.mark.parametrize("name,case", CASES, ids=CASE_IDS)
def test_golden_spot_shape(name, case):
    p = O.resolve_params(case["params_in"])
    errors = []
    for s in case["spots"]:
        got = O.spot_shape(p, s["yaw_um"], s["pitch_um"], s["quality"])
        _check_spot(f"{name}.spot({s['yaw_um']},{s['pitch_um']},{s['quality']})", got, s, errors)
    assert not errors, "\n".join(errors[:20])


@pytest.mark.parametrize("name,case", CASES, ids=CASE_IDS)
def test_golden_peak_curves(name, case):
    p = O.resolve_params(case["params_in"])
    errors = []
    for ax in ("yaw", "pitch"):
        travels = [o["travel_um"] for o in case["curves"][ax]]
        _cmp(f"{name}.curve.{ax}", O.peak_curve(p, ax, travels, "trend"), case["curves"][ax],
             TOL["curve_rel"], TOL["spot_abs"], errors)
    assert not errors, "\n".join(errors[:20])


# ------------------------------------------------------------------ golden again, through the adapter
@pytest.mark.parametrize("name,case", CASES, ids=CASE_IDS)
def test_adapter_golden_params_and_responses(name, case):
    """SteeringParams -> to_core -> resolve gives the golden parameters; steer(lower=yaw, upper=pitch) the responses."""
    sp = _params_for(case)
    errors = []
    _cmp(f"{name}.params", steering.resolve(sp), case["params_resolved"], 0.0, TOL["D_abs_mm"], errors)
    assert steering.beam_diameter_mm(sp) == pytest.approx(case["params_resolved"]["D"], abs=TOL["D_abs_mm"])
    for r in case["responses"]:
        st = steering.steer(sp, lower_um=r["yaw_um"], upper_um=r["pitch_um"])
        _cmp(f"{name}.steer({r['yaw_um']},{r['pitch_um']})", _as_core_response(st), r["result"], 1e-9,
             TOL["response_abs_um"], errors)
        # bias_axis_deg = 0: the simulation frame is the lab frame
        assert (st.X_um, st.Y_um) == st.spot_um
    assert not errors, "\n".join(errors[:20])


@pytest.mark.parametrize("name,case", CASES, ids=CASE_IDS)
def test_adapter_golden_spots(name, case):
    sp = _params_for(case)
    errors = []
    for s in case["spots"]:
        got = steering.spot(sp, lower_um=s["yaw_um"], upper_um=s["pitch_um"], quality=s["quality"])
        _check_spot(f"{name}.spot(lower={s['yaw_um']},upper={s['pitch_um']},{s['quality']})", got, s, errors)
    assert not errors, "\n".join(errors[:20])


@pytest.mark.parametrize("name,case", CASES, ids=CASE_IDS)
def test_adapter_golden_peak_curves(name, case):
    sp = _params_for(case)
    errors = []
    for act, ax in (("lower", "yaw"), ("upper", "pitch")):
        travels = [o["travel_um"] for o in case["curves"][ax]]
        _cmp(f"{name}.peak_curve.{act}", steering.peak_curve(sp, act, travels), case["curves"][ax],
             TOL["curve_rel"], TOL["spot_abs"], errors)
    assert not errors, "\n".join(errors[:20])


@pytest.mark.parametrize("name,case", CASES, ids=CASE_IDS)
def test_adapter_golden_with_a_tilt_scale(name, case):
    """With tilt_scale = 1/2 the adapter reproduces the golden responses, spots and peak curves at twice the travel
    (exact in binary floating point): the travel reaches the core scaled, every per-um response is per um of real
    travel (half the golden one) and peak_curve() reports the real travel it was given."""
    k = 0.5
    sp = replace(_params_for(case), tilt_scale=k)
    errors = []
    for r in case["responses"]:
        st = steering.steer(sp, lower_um=r["yaw_um"] / k, upper_um=r["pitch_um"] / k)
        exp = dict(r["result"])
        exp["perUm"] = {ax: [k * v for v in exp["perUm"][ax]] for ax in ("yaw", "pitch")}
        exp["perUmPort"] = {ax: [k * v for v in exp["perUmPort"][ax]] for ax in ("yaw", "pitch")}
        _cmp(f"{name}.steer({r['yaw_um'] / k},{r['pitch_um'] / k})", _as_core_response(st), exp, 1e-9,
             TOL["response_abs_um"], errors)
        assert (st.lower_um, st.upper_um) == (r["yaw_um"] / k, r["pitch_um"] / k)
    for s in case["spots"]:
        got = steering.spot(sp, lower_um=s["yaw_um"] / k, upper_um=s["pitch_um"] / k, quality=s["quality"])
        _check_spot(f"{name}.spot(lower={s['yaw_um'] / k},upper={s['pitch_um'] / k},{s['quality']})", got, s, errors)
    for act, ax in (("lower", "yaw"), ("upper", "pitch")):
        travels = [o["travel_um"] / k for o in case["curves"][ax]]
        exp = [{**o, "travel_um": o["travel_um"] / k} for o in case["curves"][ax]]
        _cmp(f"{name}.peak_curve.{act}", steering.peak_curve(sp, act, travels), exp, TOL["curve_rel"],
             TOL["spot_abs"], errors)
    assert not errors, "\n".join(errors[:20])


# ------------------------------------------------------------------ physical sanity through the adapter (module values)
def test_lower_actuator_moves_spot_left_and_to_front():
    """The module's documented response (R = 16 mm, theta = 60 deg, nominal tilt): lower +1 um -> (-0.848, -0.321) um."""
    sp = steering.module_defaults()
    assert sp.tilt_scale == 1.0
    st = steering.steer(sp, lower_um=1.0)
    assert st.spot_um[0] == pytest.approx(-0.848, abs=1e-3)       # toward -X, away from the port wall
    assert st.spot_um[1] == pytest.approx(-0.321, abs=1e-3)       # toward -Y, the open front
    assert st.per_um["lower"][0] == pytest.approx(-0.848, abs=1e-3)
    assert st.per_um["lower"][1] == pytest.approx(-0.321, abs=1e-3)
    assert st.X_um == pytest.approx(-0.848, abs=1e-3)            # default bias axis = lab +X
    assert st.lower_turns_beam_to_front is True
    # independent small-angle result: dX = 2 R d / L per unit travel d (R, L as the adapter passes them)
    assert -st.per_um["lower"][0] == pytest.approx(2 * sp.R / sp.L, rel=2e-3)


def test_upper_actuator_moves_spot_right_and_to_front():
    """The module's documented response: upper +1 um -> (+0.737, -0.923) um."""
    sp = steering.module_defaults()
    st = steering.steer(sp, upper_um=1.0)
    assert st.spot_um[0] == pytest.approx(0.737, abs=1e-3)
    assert st.spot_um[1] == pytest.approx(-0.923, abs=1e-3)
    assert st.per_um["upper"][0] == pytest.approx(0.737, abs=1e-3)
    assert st.per_um["upper"][1] == pytest.approx(-0.923, abs=1e-3)


@pytest.mark.parametrize("yaw_side,pitch_side", [(-1, 1), (1, -1), (-1, -1)])
def test_mirrored_mount_flips_the_shift(yaw_side, pitch_side):
    """yawSide = -1 reverses the lower actuator's spot shift only, pitchSide = -1 the upper one's only; the other
    actuator keeps the upright response.  To within the geometric asymmetry, below 1 % per component with the
    module defaults and with the calibrated ones (0.8 % of the calibrated lower actuator's small front-back
    component, 0.3 % of its shift)."""
    for base in (steering.module_defaults(), SteeringParams()):
        up = steering.steer(base).per_um
        got = steering.steer(replace(base, yawSide=yaw_side, pitchSide=pitch_side)).per_um
        for a, side in (("lower", yaw_side), ("upper", pitch_side)):
            if side < 0:
                assert got[a] == pytest.approx((-up[a][0], -up[a][1]), rel=1e-2), a
                assert math.dist(got[a], (-up[a][0], -up[a][1])) < 5e-3 * math.hypot(*up[a]), a
            else:
                assert got[a] == pytest.approx(up[a], abs=1e-12), a


def test_upside_down_mount_needs_both_sides_flipped():
    """A mount turned 180 deg about the mirror normal (both sides -1): X = +5 um with the lower (horizontal-tilt)
    actuator alone needs +5.869 um (module defaults), the opposite of the upright mount; flipping pitchSide alone
    leaves it at -5.869.  The calibrated defaults behave the same way at +-17.613 um."""
    for base, travel in ((steering.module_defaults(), 5.869), (SteeringParams(), 17.613)):
        upright = steering.travel_for_offset(base, 5.0, actuators="lower")["lower_um"]
        flipped = steering.travel_for_offset(replace(base, yawSide=-1, pitchSide=-1), 5.0, actuators="lower")
        only_pitch = steering.travel_for_offset(replace(base, pitchSide=-1), 5.0, actuators="lower")["lower_um"]
        assert upright == pytest.approx(-travel, abs=1e-3)
        assert flipped["converged"] and flipped["lower_um"] == pytest.approx(travel, abs=1e-3)
        assert only_pitch == pytest.approx(upright, abs=1e-9)


@pytest.mark.parametrize("make,aspect_range", [(steering.module_defaults, (1.7, 2.3)), (SteeringParams, (1.2, 1.7))],
                         ids=["module", "calibrated"])
def test_aligned_spot_is_diffraction_limited_and_elongated_along_incidence(make, aspect_range):
    """Aligned spot: Strehl > 0.99, centred, long axis along the plane of incidence, aspect ~1/cos(theta)
    (module theta = 60 deg: 1.7-2.3; calibrated theta = 45 deg: 1.2-1.7)."""
    sp = make()
    s0 = steering.reference_spot(sp)
    assert s0["peak"] > 0.99
    assert math.hypot(*s0["peakAt"]) < 0.05
    plane = steering.steer(sp).incidence_plane
    inc = math.atan2(plane[1], plane[0])
    d = (math.degrees(s0["ang"] - inc) + 90) % 180 - 90           # axis directions are defined modulo pi
    assert abs(d) < 3
    aspect = s0["fwhmL"] / s0["fwhmS"]
    assert aspect_range[0] < aspect < aspect_range[1]
    assert aspect == pytest.approx(1 / math.cos(math.radians(sp.theta)), rel=0.05)


@pytest.mark.parametrize("lower,upper", [(5.0, 0.0), (0.0, 5.0), (-10.0, 3.0)])
def test_spot_window_follows_the_chief_ray(lower, upper):
    """The spot map sits where steer() puts the chief ray (same lab frame, um vs mm handled)."""
    sp = SteeringParams()
    s = steering.spot(sp, lower, upper, "fast")
    st = steering.steer(sp, lower, upper)
    dx, dy = steering.spot_offset_um(s, steering.reference_spot(sp))
    assert dx == pytest.approx(st.spot_um[0], abs=1e-6)
    assert dy == pytest.approx(st.spot_um[1], abs=1e-6)


# ------------------------------------------------------------------ calibration (the defaults' tilt_scale)
def test_defaults_reproduce_the_calibration():
    """One 0.2 um step of the upper (pitch) actuator moves the spot 55 nm at 45 deg incidence (steering.CALIBRATION),
    with the physical R = 16 mm and L at the KS1-Z8 spec: what is fitted is tilt_scale (~1/3), not the geometry.
    The step goes 28.3 nm toward the port wall (+X) and 47.2 nm toward the open front (-Y)."""
    cal = steering.CALIBRATION
    sp = SteeringParams()
    assert cal["actuator"] == "upper" and cal["fits"] == "tilt_scale"
    assert (cal["travel_um"], cal["spot_um"]) == (0.2, 0.055)
    assert sp.theta == cal["theta"] == 45.0
    assert sp.R == steering.module_defaults().R == 16.0           # the geometry stays physical
    assert sp.L == 37.7                                            # KS1-Z8: 0.2 um step = 1.1 arcsec
    assert sp.tilt_scale == pytest.approx(0.33336, abs=1e-12)
    st = steering.steer(sp, *_travels(cal["actuator"], cal["travel_um"]))
    assert st.shift_um == pytest.approx(cal["spot_um"], abs=1e-5)  # 55 nm to within 0.01 nm
    assert st.spot_um[0] * 1e3 == pytest.approx(28.30, abs=0.01)
    assert st.spot_um[1] * 1e3 == pytest.approx(-47.16, abs=0.01)
    # the default tilt_scale is the fit, to the 5 decimals it is stored with, whatever tilt_scale the fit starts from
    k = steering.calibrate_tilt_scale(sp, cal["actuator"], cal["travel_um"], cal["spot_um"])
    assert k == pytest.approx(sp.tilt_scale, abs=5e-6)
    for start in (1.0, 0.05, 3.0):
        assert steering.calibrate_tilt_scale(replace(sp, tilt_scale=start), cal["actuator"], cal["travel_um"],
                                             cal["spot_um"]) == pytest.approx(k, abs=1e-12)


def test_calibrated_lower_step_points_left_and_to_the_front():
    """The horizontal (lower) actuator with the calibrated tilt_scale: 59.65 nm per 0.2 um step, 56.6 nm left (-X,
    away from the port wall) and 18.9 nm toward the open front (-Y); per KDC101 count 8.6 nm (lower) and 8.0 nm
    (upper)."""
    sp = SteeringParams()
    st = steering.steer(sp, lower_um=0.2)
    assert st.shift_um * 1e3 == pytest.approx(59.65, abs=0.01)
    assert st.spot_um[0] * 1e3 == pytest.approx(-56.59, abs=0.01)
    assert st.spot_um[1] * 1e3 == pytest.approx(-18.86, abs=0.01)
    assert st.lower_turns_beam_to_front is True
    per = st.per_um                                                # at travel 0, per um of real travel
    assert per["lower"] == pytest.approx((-0.2828, -0.0939), abs=1e-4)
    assert per["upper"] == pytest.approx((0.1416, -0.2359), abs=1e-4)
    # small-angle check, as for the module: the mirror tilts by tilt_scale * travel / L
    assert -per["lower"][0] == pytest.approx(2 * sp.tilt_scale * sp.R / sp.L, rel=2e-3)
    nm_per_count = {a: math.hypot(*per[a]) / steering.KDC101_COUNTS_PER_UM * 1e3 for a in steering.ACTUATORS}
    assert nm_per_count["lower"] == pytest.approx(8.6, abs=0.05)
    assert nm_per_count["upper"] == pytest.approx(8.0, abs=0.05)


def test_calibrated_aligned_spot_size():
    """D auto with the calibrated defaults is 13.1 mm (the D field's default); the aligned spot is 1.562 x 1.115 um.
    tilt_scale does not change the aligned spot (it only scales the travel)."""
    sp = SteeringParams()
    s0 = steering.reference_spot(sp)
    assert s0["D"] == pytest.approx(13.1) and sp.D == pytest.approx(13.1)
    assert (s0["fwhmL"], s0["fwhmS"]) == pytest.approx((1.562, 1.115), abs=2e-3)
    s1 = steering.reference_spot(replace(sp, tilt_scale=1.0))
    assert (s1["fwhmL"], s1["fwhmS"], s1["peak"]) == (s0["fwhmL"], s0["fwhmS"], s0["peak"])


def test_calibrated_defaults_keep_the_physical_geometry():
    """The geometry the 3-D view draws is the physical one, whatever the calibration: the parabolic mirror's chief
    point is R = 16 mm from the focus and the fold mirror sits at x = R sin(theta) sin(beta) = 8.0 mm, 6 mm behind
    the tip (the reference page draws the collection fold mirror at -x0), far enough apart for the page's 10 mm
    square fold mirrors, turned 45 deg, not to overlap (they did at the fitted R = 5.3 mm: x0 = 2.7 mm)."""
    sp = SteeringParams()
    g = O.geometry(steering.resolve(sp))
    assert float(np.linalg.norm(g["P"])) == pytest.approx(16.0, abs=1e-12)
    assert g["x0"] == pytest.approx(16.0 * math.sin(math.radians(45)) ** 2, abs=1e-12)
    assert g["x0"] == pytest.approx(8.0, abs=1e-12)
    assert list(g["F1"][:2]) == pytest.approx([8.0, sp.foldY], abs=1e-12)
    assert g["x0"] - 5.0 * math.cos(math.radians(45)) > 4.0           # half-width of the mirror along x: 3.5 mm
    # tilt_scale never reaches the core's parameters or its geometry
    assert steering.to_core(replace(sp, tilt_scale=2.0)) == steering.to_core(sp)
    assert steering.resolve(replace(sp, tilt_scale=2.0)) == steering.resolve(sp)


def test_calibrated_lower_only_targets():
    """The tip-position offsets with the lower actuator alone (calibrated defaults, phi = 0): X = +2.5/+5/+7.5/-7.5 um
    need -8.821/-17.613/-26.376/+26.634 um of travel (-305/-609/-911/+920 KDC101 counts); the peak falls to
    84.8/54.2/34.3/35.5 % ('trend')."""
    rows = steering.plan_offsets(SteeringParams(), [2.5, 5.0, 7.5, -7.5])
    for r, travel, counts, rel in zip(rows, (-8.821, -17.613, -26.376, 26.634), (-305, -609, -911, 920),
                                      (0.848, 0.542, 0.343, 0.355)):
        assert r["converged"] and r["upper_um"] == 0.0
        assert r["lower_um"] == pytest.approx(travel, abs=1e-3)
        assert round(r["lower_counts"]) == counts
        assert r["rel"] == pytest.approx(rel, abs=0.002)
    assert rows[1]["Y_got_um"] == pytest.approx(1.70, abs=0.005)  # Y is not controlled with one actuator


@pytest.mark.parametrize("actuator", steering.ACTUATORS)
@pytest.mark.parametrize("travel", [0.2, -0.2, 5.0, 30.0, 50.0, -50.0])
@pytest.mark.parametrize("kappa_true", [0.2, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("theta", [45.0, 60.0])
def test_calibrate_tilt_scale_round_trip(actuator, travel, kappa_true, theta):
    """docs/steering.md: calibrate with a step from the aligned position measured on the AFM camera (extend or
    retract, signed travel).  The shift a true tilt_scale gives fits back to it, whatever tilt_scale sp starts from;
    with the fitted value the step reproduces the shift, and the geometry (R, theta, ...) is not touched."""
    sp = SteeringParams(theta=theta)
    shift = steering.steer(replace(sp, tilt_scale=kappa_true), *_travels(actuator, travel)).shift_um
    for start in (sp.tilt_scale, 1.0, 7.0):
        k = steering.calibrate_tilt_scale(replace(sp, tilt_scale=start), actuator, travel, shift)
        assert isinstance(k, float)
        assert k == pytest.approx(kappa_true, rel=1e-9)
    fitted = replace(sp, tilt_scale=k)
    assert steering.steer(fitted, *_travels(actuator, travel)).shift_um == pytest.approx(shift, abs=1e-9)
    assert steering.to_core(fitted) == steering.to_core(sp)


@pytest.mark.parametrize("actuator", steering.ACTUATORS)
def test_calibrate_tilt_scale_retract_differs_from_extend(actuator):
    """At a 50 um camera step the response is asymmetric (0.27-0.44 um between retract and extend with the defaults):
    a retract entered as an extend of the same size fits a tilt_scale 2-3 % off, so the sign of the travel matters
    (the GUI takes a signed travel).  The slope at travel 0 would be off by 1-1.6 %: the fit is an exact trace."""
    sp = SteeringParams()
    retract = steering.steer(sp, *_travels(actuator, -50.0)).shift_um
    extend = steering.steer(sp, *_travels(actuator, 50.0)).shift_um
    assert abs(retract - extend) > 0.2                             # um
    assert steering.calibrate_tilt_scale(sp, actuator, -50.0, retract) == pytest.approx(sp.tilt_scale, rel=1e-9)
    assert steering.calibrate_tilt_scale(sp, actuator, 50.0, extend) == pytest.approx(sp.tilt_scale, rel=1e-9)
    assert abs(steering.calibrate_tilt_scale(sp, actuator, 50.0, retract) / sp.tilt_scale - 1) > 0.015
    linear = retract / (50.0 * math.hypot(*steering.steer(replace(sp, tilt_scale=1.0)).per_um[actuator]))
    assert abs(linear / sp.tilt_scale - 1) > 0.008


@pytest.mark.parametrize("actuator,travel,shift", [
    ("yaw", 0.2, 0.055), ("pitch", 0.2, 0.055), ("", 0.2, 0.055), (None, 0.2, 0.055), ("Upper", 0.2, 0.055),
    ("upper", 0.0, 0.055), ("lower", -0.0, 0.055), ("lower", float("nan"), 0.055), ("upper", float("inf"), 0.055),
    ("upper", 0.2, 0.0), ("lower", 0.2, -0.055), ("upper", 0.2, float("nan")), ("lower", 0.2, float("inf"))])
def test_calibrate_tilt_scale_rejects_bad_input(actuator, travel, shift):
    with pytest.raises(ValueError):
        steering.calibrate_tilt_scale(SteeringParams(), actuator, travel, shift)


def test_calibrate_tilt_scale_unreachable_shift_raises():
    """With the defaults' geometry the lower actuator, extending, cannot move the spot more than ~447 um before the
    chief ray leaves the parabolic mirror (528 um of travel at the nominal tilt), the upper one, retracting, ~997 um:
    1 mm and 1.5 mm are out of reach whatever the tilt_scale.  The message is plain: no SciPy text."""
    for actuator, travel, shift in (("lower", 0.2, 1000.0), ("upper", -0.2, 1500.0)):
        with pytest.raises(ValueError, match=rf"^the {actuator} actuator cannot move the spot .*: the chief ray "
                                             r"leaves the parabolic mirror first$") as e:
            steering.calibrate_tilt_scale(SteeringParams(), actuator, travel, shift)
        msg = str(e.value).lower()
        assert not any(w in msg for w in ("f(a)", "f(b)", "brentq", "scipy", "sign")), msg


def test_calibrate_tilt_scale_finds_a_shift_near_the_mirror_edge():
    """A shift reachable shortly before the chief ray leaves the mirror is still fitted: 450 um of the lower actuator
    at the nominal tilt (85 % of the way to the edge at ~528 um) moves the spot 364 um.  The growing bracket jumps
    past the edge there (407 -> 610 um); the edge is then bisected instead of reporting the shift unreachable."""
    sp = SteeringParams()
    shift = steering.steer(replace(sp, tilt_scale=1.0), 450.0, 0.0).shift_um
    assert steering.calibrate_tilt_scale(sp, "lower", 450.0, shift) == pytest.approx(1.0, rel=1e-9)
    # the other directions, near their edges (lower +528 / -1916, upper +597 / -892 um at the nominal tilt), and
    # with the travel the GUI's spin box allows (kappa = 0.9: the 364 um of the lower actuator at +500 um)
    for actuator, travel, kappa in (("upper", 560.0, 1.0), ("upper", -850.0, 1.0), ("lower", -1500.0, 1.0),
                                    ("lower", 500.0, 0.9)):
        tr = (travel, 0.0) if actuator == "lower" else (0.0, travel)
        shift = steering.steer(replace(sp, tilt_scale=kappa), *tr).shift_um
        assert steering.calibrate_tilt_scale(sp, actuator, travel, shift) == pytest.approx(kappa, rel=1e-9), \
            (actuator, travel)
    # just beyond the largest shift the lower actuator reaches before the edge (~447 um): still unreachable, in um
    with pytest.raises(ValueError, match=r"^the lower actuator cannot move the spot 448 µm: the chief ray leaves"):
        steering.calibrate_tilt_scale(sp, "lower", 450.0, 448.0)


# ------------------------------------------------------------------ tilt_scale: the core sees tilt_scale * travel
@pytest.mark.parametrize("kappa", [0.2, 0.33336, 0.5, 2.0])
@pytest.mark.parametrize("lower,upper", [(3.0, -2.0), (0.2, 0.0), (0.0, 0.2), (-40.0, 15.0)])
def test_tilt_scale_scales_the_travel_handed_to_the_core(kappa, lower, upper):
    """steer() with tilt_scale kappa is steer() at the nominal tilt for kappa times the travel: the same chief-ray
    trace (spot, port, defocus, arrival time, also projected on a turned bias axis), the real travels reported back,
    and every per-um response per um of real travel (kappa times the nominal one)."""
    base = SteeringParams(theta=52.0, bias_axis_deg=30.0)
    got = steering.steer(replace(base, tilt_scale=kappa), lower, upper)
    ref = steering.steer(replace(base, tilt_scale=1.0), kappa * lower, kappa * upper)
    assert (got.lower_um, got.upper_um) == (lower, upper)
    for name in ("spot_um", "X_um", "Y_um", "port_um", "along_um", "across_um", "defocus_um", "arrival_fs",
                 "incidence_plane"):
        assert getattr(got, name) == pytest.approx(getattr(ref, name), abs=1e-12), name
    assert got.lower_turns_beam_to_front == ref.lower_turns_beam_to_front
    for a in steering.ACTUATORS:
        for name in ("per_um", "per_um_sim", "port_per_um"):
            assert getattr(got, name)[a] == pytest.approx(tuple(kappa * v for v in getattr(ref, name)[a]),
                                                          rel=1e-12, abs=1e-15), (name, a)
    # and straight from the core
    r = O.steering_response(steering.to_core(base), yaw_um=kappa * lower, pitch_um=kappa * upper)
    assert got.spot_um == pytest.approx(tuple(r["spot_um"]), abs=1e-12)


@pytest.mark.parametrize("kappa", [0.33336, 2.0])
def test_spot_uses_the_tilt_scale(kappa):
    """spot() hands the core tilt_scale * travel: the spot of a real travel t is the nominal-tilt spot of kappa * t,
    not the nominal-tilt spot of t."""
    base = SteeringParams()
    got = steering.spot(replace(base, tilt_scale=kappa), 6.0, -3.0, "fast")
    ref = steering.spot(replace(base, tilt_scale=1.0), kappa * 6.0, kappa * -3.0, "fast")
    np.testing.assert_allclose(got["I"], ref["I"], rtol=0, atol=1e-7)
    for key in ("x0", "W", "peak", "rel", "fwhmL", "fwhmS", "ang", "area", "peakAt", "D"):
        assert got[key] == pytest.approx(ref[key], abs=1e-12), key
    nominal = steering.spot(replace(base, tilt_scale=1.0), 6.0, -3.0, "fast")
    assert list(got["x0"]) != pytest.approx(list(nominal["x0"]), abs=1e-6)


@pytest.mark.parametrize("actuator", steering.ACTUATORS)
def test_peak_curve_uses_the_tilt_scale_and_reports_the_real_travel(actuator):
    """peak_curve() evaluates the nominal-tilt curve at tilt_scale * travel and reports the real travel; with the
    calibrated tilt_scale (~1/3) the peak falls more slowly per um of real travel than at the nominal tilt."""
    sp = SteeringParams()
    travels = [-30.0, -10.0, 0.0, 10.0, 30.0]
    got = steering.peak_curve(sp, actuator, travels)
    ref = steering.peak_curve(replace(sp, tilt_scale=1.0), actuator, [sp.tilt_scale * t for t in travels])
    assert [r["travel_um"] for r in got] == travels
    assert [r["travel_um"] for r in ref] == pytest.approx([sp.tilt_scale * t for t in travels], abs=1e-12)
    for g, r in zip(got, ref):
        for key in ("rel", "fwhmL", "fwhmS", "area", "ang"):
            assert g[key] == pytest.approx(r[key], abs=1e-12), key
    nominal = steering.peak_curve(replace(sp, tilt_scale=1.0), actuator, travels)
    for i in (0, 1, 3, 4):
        assert got[i]["rel"] > nominal[i]["rel"]
    assert got[2]["rel"] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("kappa", [0.2, 0.5, 2.0])
@pytest.mark.parametrize("actuators,X,Y", [("lower", 5.0, 0.0), ("upper", -4.0, 0.0), ("both", 6.0, -2.0)])
def test_travel_for_offset_returns_the_real_travel(kappa, actuators, X, Y):
    """The solver works in real travel: with tilt_scale kappa it lands on the same point with 1/kappa times the
    nominal-tilt travel, and its Steering reports that real travel."""
    base = SteeringParams(bias_axis_deg=20.0)
    sol = steering.travel_for_offset(replace(base, tilt_scale=kappa), X, Y, actuators)
    nom = steering.travel_for_offset(replace(base, tilt_scale=1.0), X, Y, actuators)
    assert sol["converged"] and nom["converged"]
    for a in steering.ACTUATORS:
        assert sol[f"{a}_um"] == pytest.approx(nom[f"{a}_um"] / kappa, abs=1e-4), a
    assert (sol["steering"].lower_um, sol["steering"].upper_um) == (sol["lower_um"], sol["upper_um"])
    st = steering.steer(replace(base, tilt_scale=kappa), sol["lower_um"], sol["upper_um"])
    assert st.X_um == pytest.approx(X, abs=1e-5)
    if actuators == "both":
        assert st.Y_um == pytest.approx(Y, abs=1e-5)


def test_plan_offsets_count_the_real_travel():
    """The KDC101 counts are those of the real push-rod travel: with the calibrated tilt_scale the lower actuator
    needs 1/tilt_scale times the nominal-tilt travel and counts for the same offset (X = +5 um: -609 counts, not
    -203), and the predicted spot and landing point are the same."""
    sp = SteeringParams()
    xs = [-5.0, 5.0]
    rows = steering.plan_offsets(sp, xs)
    nominal = steering.plan_offsets(replace(sp, tilt_scale=1.0), xs)
    for r, n in zip(rows, nominal):
        assert r["converged"] and n["converged"]
        assert r["lower_um"] == pytest.approx(n["lower_um"] / sp.tilt_scale, abs=1e-4)
        assert r["lower_counts"] == pytest.approx(steering.KDC101_COUNTS_PER_UM * r["lower_um"], rel=1e-12)
        assert r["lower_counts"] == pytest.approx(n["lower_counts"] / sp.tilt_scale, abs=5e-3)
        assert r["upper_um"] == r["upper_counts"] == 0.0
        for key in ("X_got_um", "Y_got_um", "defocus_um", "arrival_fs"):
            assert r[key] == pytest.approx(n[key], abs=1e-5), key
        for key in ("rel", "fwhmL", "fwhmS"):
            assert r[key] == pytest.approx(n[key], abs=1e-4), key
    assert (round(rows[1]["lower_counts"]), round(nominal[1]["lower_counts"])) == (-609, -203)


def test_core_travel():
    assert steering.core_travel(SteeringParams(tilt_scale=0.25), 8.0) == 2.0
    assert steering.core_travel(SteeringParams(tilt_scale=0.25), -8.0) == -2.0
    assert steering.core_travel(steering.module_defaults(), -3.5) == -3.5
    assert steering.core_travel(SteeringParams(), 0.2) == pytest.approx(0.2 * 0.33336, abs=1e-15)
    assert steering.core_travel(SteeringParams(), 0.0) == 0.0


@pytest.mark.parametrize("bad", [0.0, -0.0, -0.33336, float("nan"), float("inf"), float("-inf"), None])
def test_bad_tilt_scale_is_rejected(bad):
    """A tilt_scale that is not a positive number raises ValueError (naming tilt_scale) everywhere the travel
    reaches the core."""
    sp = SteeringParams(tilt_scale=bad)
    with pytest.raises(ValueError, match="tilt_scale"):
        steering.core_travel(sp, 1.0)
    for call in (lambda: steering.steer(sp), lambda: steering.steer(sp, 1.0, 0.0),
                 lambda: steering.spot(sp, 1.0, 0.0, "trend"), lambda: steering.reference_spot(sp),
                 lambda: steering.peak_curve(sp, "lower", [1.0]),
                 lambda: steering.travel_for_offset(sp, 1.0, actuators="lower"),
                 lambda: steering.plan_offsets(sp, [1.0])):
        with pytest.raises(ValueError, match="tilt_scale"):
            call()


def test_module_defaults_are_the_core_defaults():
    """module_defaults() is the vendored module's own parameter set: every key SteeringParams shares with the core
    matches optics.DEFAULTS (D 'auto' -> D_auto on, the D field at the auto value 13.0 mm), and the mirror tilts by
    the nominal travel / L (tilt_scale = 1); the bias axis keeps its default.  SteeringParams() differs from it only
    in the assumed theta, the calibrated tilt_scale and the D field that follows theta (auto value 13.1 vs 13.0 mm):
    R is the same physical 16 mm."""
    md = steering.module_defaults()
    shared = {f.name for f in fields(SteeringParams)} & set(O.DEFAULTS)
    assert shared == set(CORE_KEYS) | {"D"}
    for k in CORE_KEYS:
        assert getattr(md, k) == O.DEFAULTS[k], k
    assert O.DEFAULTS["D"] == "auto" and md.D_auto is True
    assert md.D == 13.0 == pytest.approx(steering.beam_diameter_mm(md))
    assert md.tilt_scale == 1.0
    assert md.bias_axis_deg == SteeringParams().bias_axis_deg == 0.0
    assert (md.R, md.theta) == (16.0, 60.0)
    differ = {f.name: (getattr(md, f.name), getattr(SteeringParams(), f.name)) for f in fields(SteeringParams)
              if getattr(md, f.name) != getattr(SteeringParams(), f.name)}
    assert differ == {"theta": (60.0, 45.0), "tilt_scale": (1.0, 0.33336), "D": (13.0, 13.1)}
    assert steering.module_defaults() is not md and steering.module_defaults() == md    # a fresh copy each call


# ------------------------------------------------------------------ lab <-> simulation frame
@pytest.mark.parametrize("phi", [0.0, 30.0, 90.0, 137.5, 180.0, -60.0, 270.0])
def test_lab_sim_round_trip(phi):
    sp = SteeringParams(bias_axis_deg=phi)
    for dx, dy in ((1.0, 0.0), (0.0, 1.0), (-3.2, 7.5), (12.0, -0.4)):
        X, Y = steering.lab_to_sim(dx, dy, sp)
        assert math.hypot(X, Y) == pytest.approx(math.hypot(dx, dy), abs=1e-12)   # a rotation
        bx, by = steering.sim_to_lab(X, Y, sp)
        assert (bx, by) == pytest.approx((dx, dy), abs=1e-12)
        fx, fy = steering.lab_to_sim(*steering.sim_to_lab(dx, dy, sp), sp)
        assert (fx, fy) == pytest.approx((dx, dy), abs=1e-12)


def test_bias_axis_special_angles():
    dX, dY = 2.0, -3.0
    assert steering.lab_to_sim(dX, dY, SteeringParams(bias_axis_deg=0.0)) == pytest.approx((dX, dY), abs=1e-12)
    assert steering.lab_to_sim(dX, dY, SteeringParams(bias_axis_deg=90.0)) == pytest.approx((dY, -dX), abs=1e-12)
    assert steering.lab_to_sim(dX, dY, SteeringParams(bias_axis_deg=180.0)) == pytest.approx((-dX, -dY), abs=1e-12)
    assert steering.lab_to_sim(dX, dY, SteeringParams(bias_axis_deg=-90.0)) == pytest.approx((-dY, dX), abs=1e-12)


@pytest.mark.parametrize("phi", [0.0, 30.0, 90.0, -120.0])
def test_steer_projects_on_the_bias_axis(phi):
    sp = SteeringParams(bias_axis_deg=phi)
    st = steering.steer(sp, lower_um=3.0, upper_um=-2.0)
    c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    dx, dy = st.spot_um
    assert st.X_um == pytest.approx(dx * c + dy * s, abs=1e-12)
    assert st.Y_um == pytest.approx(-dx * s + dy * c, abs=1e-12)
    # the lab shift itself does not depend on how the sample is mounted
    assert st.spot_um == steering.steer(SteeringParams(), 3.0, -2.0).spot_um
    for a in steering.ACTUATORS:
        px, py = st.per_um[a]
        assert st.per_um_sim[a] == pytest.approx((px * c + py * s, -px * s + py * c), abs=1e-12)


# ------------------------------------------------------------------ travel_for_offset
@pytest.mark.parametrize("phi,X,Y", [(0.0, 5.0, 1.0), (0.0, -12.0, 0.0), (0.0, 0.0, 0.0), (30.0, 8.0, -2.5),
                                     (90.0, -6.0, 4.0), (-70.0, -2.5, 0.0), (35.0, 7.5, 0.0)])
def test_travel_both_lands_on_target(phi, X, Y):
    sp = SteeringParams(bias_axis_deg=phi)
    sol = steering.travel_for_offset(sp, X, Y, "both")
    assert sol["converged"]
    assert sol["residual_um"] <= 1e-5
    st = steering.steer(sp, sol["lower_um"], sol["upper_um"])     # re-trace independently of the solver's result
    assert st.X_um == pytest.approx(X, abs=1e-5)
    assert st.Y_um == pytest.approx(Y, abs=1e-5)
    assert sol["steering"].X_um == pytest.approx(X, abs=1e-5)


@pytest.mark.parametrize("actuator", ["lower", "upper"])
@pytest.mark.parametrize("X", [-8.0, 5.0])
def test_travel_single_actuator_matches_X(actuator, X):
    sp = SteeringParams()
    sol = steering.travel_for_offset(sp, X, actuators=actuator)
    other = "upper" if actuator == "lower" else "lower"
    assert sol["converged"]
    assert sol[f"{other}_um"] == 0.0                              # the other actuator stays put
    st = steering.steer(sp, sol["lower_um"], sol["upper_um"])
    assert st.X_um == pytest.approx(X, abs=1e-5)
    assert sol["steering"].Y_um == pytest.approx(st.Y_um, abs=1e-12)


@pytest.mark.parametrize("phi,actuator,X,travel", [
    (-70.0, "lower", -2.5, -46.60),     # aligned slope 0.012 um/um: the first full Newton step (-212 um) misses
    (-65.0, "lower", -7.5, 65.08),      # aligned slope -0.067, ~0.16 at the target: a fixed Jacobian diverges
    (35.0, "upper", 5.0, 52.10),
    (35.0, "upper", 7.5, 71.27),
])
def test_travel_reaches_targets_far_from_the_aligned_slope(phi, actuator, X, travel):
    """Targets whose travel lies where the response is far from its slope at 0 (found with the module's defaults and
    checked by direct steer() scans): the solver must land on them and flag them reached, not miss or stall."""
    sp = replace(steering.module_defaults(), bias_axis_deg=phi)
    sol = steering.travel_for_offset(sp, X, actuators=actuator)
    assert sol["converged"]
    assert sol["residual_um"] <= steering.REACH_TOL_UM
    assert sol[f"{actuator}_um"] == pytest.approx(travel, abs=0.01)
    st = steering.steer(sp, sol["lower_um"], sol["upper_um"])
    assert st.X_um == pytest.approx(X, abs=1e-5)


def test_travel_unreachable_target_is_flagged_not_raised():
    """Module defaults at phi = -75 deg: the lower actuator alone cannot bring X above ~+2.34 um (its maximum; a scan
    over +-500 um finds no crossing of 7.5): the solver stops at the closest point and says so."""
    sp = replace(steering.module_defaults(), bias_axis_deg=-75.0)
    sol = steering.travel_for_offset(sp, 7.5, actuators="lower")
    assert not sol["converged"]
    assert sol["residual_um"] > 5.0
    assert sol["upper_um"] == 0.0
    best = max(steering.steer(sp, t, 0.0).X_um for t in np.arange(-100.0, 100.5, 0.5))
    assert sol["steering"].X_um == pytest.approx(best, abs=1e-3)


def test_travel_converged_means_within_the_reach_tolerance():
    """'converged' (the link table's 'reached') is a physical tolerance, REACH_TOL_UM = 1e-4 um, not the iteration's
    1e-6 um stop: with the module's defaults two Newton steps toward X = 20 um land 7e-5 um off, which is reached."""
    sp = steering.module_defaults()
    two = steering.travel_for_offset(sp, 20.0, actuators="lower", max_iter=2)
    assert 1e-6 < two["residual_um"] <= steering.REACH_TOL_UM
    assert two["converged"]
    one = steering.travel_for_offset(sp, 20.0, actuators="lower", max_iter=1)
    assert one["residual_um"] > steering.REACH_TOL_UM and not one["converged"]
    loose = steering.travel_for_offset(sp, 20.0, actuators="lower", tol_um=1.0, max_iter=1)
    assert loose["converged"] == (loose["residual_um"] <= 1.0)
    full = steering.travel_for_offset(sp, 20.0, actuators="lower")
    assert full["converged"] and full["residual_um"] <= 1e-6


@pytest.mark.parametrize("make,phi,actuator,per_um", [
    (steering.module_defaults, 20.7, "lower", 0.907), (steering.module_defaults, -159.3, "lower", 0.907),
    (steering.module_defaults, -51.4, "upper", 1.181), (steering.module_defaults, 128.6, "upper", 1.181),
    (SteeringParams, 18.38, "lower", 0.298), (SteeringParams, -161.62, "lower", 0.298),
    (SteeringParams, -59.02, "upper", 0.275), (SteeringParams, 120.98, "upper", 0.275),
], ids=["module-20.7-lower", "module--159.3-lower", "module--51.4-upper", "module-128.6-upper",
        "calibrated-18.38-lower", "calibrated--161.62-lower", "calibrated--59.02-upper", "calibrated-120.98-upper"])
def test_single_actuator_pure_x_mountings(make, phi, actuator, per_um):
    """The mountings for which one actuator alone scans along the bias axis (docs/steering.md); phi + 180 deg
    reverses the sign.  The directions differ between the module's theta = 60 deg and the assumed 45 deg (the
    tilt_scale only scales the rate: X per um of real travel)."""
    px, py = steering.steer(replace(make(), bias_axis_deg=phi)).per_um_sim[actuator]
    assert abs(py) < 2e-3 * abs(px)
    assert abs(px) == pytest.approx(per_um, abs=1e-3)


def test_travel_lower_sign_at_default_params():
    """+X (toward the port wall) needs the lower push rod to retract: about -5.87 um for X = +5 um with the module's
    defaults (-17.61 um with the calibrated ones, test_calibrated_lower_only_targets)."""
    sol = steering.travel_for_offset(steering.module_defaults(), 5.0, actuators="lower")
    assert sol["lower_um"] == pytest.approx(-5.87, abs=0.01)
    assert sol["upper_um"] == 0.0
    assert steering.kdc101_counts(sol["lower_um"]) < 0


def test_travel_rejects_unknown_mode():
    with pytest.raises(ValueError):
        steering.travel_for_offset(SteeringParams(), 1.0, actuators="yaw")
    with pytest.raises(ValueError):
        steering.travel_for_offset(SteeringParams(), 1.0, actuators="")


# ------------------------------------------------------------------ SteeringParams
def test_spot_and_peak_curve_use_the_auto_diameter():
    """With D auto on, spot() and peak_curve() use the resolved diameter, not the D field (module defaults with
    R = 12 mm: 9.8 mm, while the D field and the module's auto value are both 13.0 mm)."""
    sp = replace(steering.module_defaults(), R=12.0)
    assert sp.D_auto and sp.D == 13.0
    assert steering.beam_diameter_mm(sp) == pytest.approx(9.8)
    same = replace(sp, D_auto=False, D=9.8)
    other = replace(sp, D_auto=False, D=13.0)
    s, s_same, s_other = (steering.spot(p, 4.0, 0.0) for p in (sp, same, other))
    assert s["D"] == pytest.approx(9.8)
    assert s["fwhmL"] == pytest.approx(s_same["fwhmL"], abs=1e-12)
    assert s["fwhmL"] != pytest.approx(s_other["fwhmL"], abs=1e-3)
    c, c_same, c_other = (steering.peak_curve(p, a, [4.0])[0]["rel"] for p in (sp, same, other) for a in ("lower",))
    assert c == pytest.approx(c_same, abs=1e-12)
    assert c != pytest.approx(c_other, abs=1e-3)


def test_to_core_beam_diameter():
    """The D field's default is the auto value of its parameter set (13.1 mm calibrated, 13.0 mm module)."""
    assert steering.to_core(SteeringParams())["D"] == "auto"
    core = steering.to_core(SteeringParams(D_auto=False, D=9.5))
    assert core["D"] == 9.5 and isinstance(core["D"], float)
    assert steering.beam_diameter_mm(SteeringParams()) == pytest.approx(13.1) == SteeringParams().D
    assert steering.beam_diameter_mm(steering.module_defaults()) == pytest.approx(13.0) == steering.module_defaults().D
    assert steering.beam_diameter_mm(SteeringParams(D_auto=False, D=9.5)) == 9.5


CORE_KEYS = ("R", "theta", "beta", "alpha", "dist", "L", "lam", "NA", "zLow", "portX", "foldY", "yawSide", "pitchSide")


def test_to_core_defaults_agree_with_the_core():
    """module_defaults() reaches the core as the core's own defaults; SteeringParams() differs only in the assumed
    theta (tilt_scale is the adapter's: it never reaches the core's parameters)."""
    md = steering.module_defaults()
    core = steering.to_core(md)
    assert set(core) == set(O.DEFAULTS)
    for k in CORE_KEYS:
        assert core[k] == getattr(md, k)
        assert core[k] == O.DEFAULTS[k], k                         # the defaults agree with the module README
    assert core["D"] == O.DEFAULTS["D"] == "auto"
    sp = SteeringParams()
    core = steering.to_core(sp)
    for k in CORE_KEYS:
        assert core[k] == getattr(sp, k), k
        assert (core[k] == O.DEFAULTS[k]) == (k != "theta"), k
    assert (core["R"], core["theta"]) == (16.0, steering.CALIBRATION["theta"])
    assert steering.to_core(replace(sp, tilt_scale=1.0)) == core


def test_to_core_carries_every_core_parameter():
    """Every field set away from its default reaches the core (a constant equal to the default cannot pass), and
    tilt_scale reaches it through the travel."""
    sp = SteeringParams(R=12.5, theta=55.0, beta=30.0, NA=0.3, zLow=2.0, alpha=40.0, dist=180.0, L=35.0, yawSide=-1,
                        pitchSide=1, portX=300.0, foldY=8.0, lam=780.0, D_auto=False, D=9.0, bias_axis_deg=25.0,
                        tilt_scale=0.7)
    core = steering.to_core(sp)
    assert set(core) == set(O.DEFAULTS)
    for k in CORE_KEYS:
        assert core[k] == getattr(sp, k), k
        if k not in ("pitchSide",):
            assert core[k] != O.DEFAULTS[k], k                     # the test really moves every parameter,
            assert core[k] != getattr(SteeringParams(), k), k      # away from both parameter sets
    assert core["D"] == 9.0
    assert steering.to_core(SteeringParams(yawSide=1, pitchSide=-1))["pitchSide"] == -1
    assert steering.resolve(sp) == O.resolve_params({**{k: getattr(sp, k) for k in CORE_KEYS}, "D": 9.0})
    st = steering.steer(sp, 3.0, -2.0)
    r = O.steering_response({**{k: getattr(sp, k) for k in CORE_KEYS}, "D": 9.0}, yaw_um=0.7 * 3.0,
                            pitch_um=0.7 * -2.0)
    assert st.spot_um == pytest.approx(tuple(r["spot_um"]), abs=1e-12)
    assert st.per_um["lower"] == pytest.approx(tuple(0.7 * v for v in r["perUm"]["yaw"]), abs=1e-12)
    # zLow, portX and foldY change the results the adapter reports (with the defaults' R = 16 mm and theta = 45 deg
    # the mirror's lower edge clips the illumination cone from zLow ~ 4.5 mm on)
    base = SteeringParams()
    assert steering.spot(SteeringParams(zLow=6.0), 4.0, 0.0, "trend")["fwhmL"] != pytest.approx(
        steering.spot(base, 4.0, 0.0, "trend")["fwhmL"], abs=1e-6)
    assert steering.steer(SteeringParams(portX=300.0), 4.0).spot_um != pytest.approx(
        steering.steer(base, 4.0).spot_um, abs=1e-9)
    assert steering.steer(SteeringParams(foldY=8.0), 0.0, 4.0).spot_um != pytest.approx(
        steering.steer(base, 0.0, 4.0).spot_um, abs=1e-9)


@pytest.mark.parametrize("side,expected", [(1, 1), (-1, -1), (5, 1), (-3, -1), (0, 1)])
def test_to_core_normalises_sides(side, expected):
    core = steering.to_core(SteeringParams(yawSide=side, pitchSide=side))
    assert core["yawSide"] == expected and core["pitchSide"] == expected


@pytest.mark.parametrize("yaw_side,pitch_side", [(-1, 1), (1, -1), (-4, 2), (3, -2)])
def test_to_core_keeps_the_sides_apart(yaw_side, pitch_side):
    core = steering.to_core(SteeringParams(yawSide=yaw_side, pitchSide=pitch_side))
    assert core["yawSide"] == (1 if yaw_side >= 0 else -1)
    assert core["pitchSide"] == (1 if pitch_side >= 0 else -1)


def test_from_dict_coerces_strings_and_ignores_unknown_keys():
    sp = SteeringParams.from_dict({"D_auto": "false", "yawSide": "-1", "D": "15", "R": "12.5",
                                   "not_a_parameter": 3, "theta": 55, "tilt_scale": "0.25"})
    assert sp.D_auto is False
    assert sp.tilt_scale == 0.25 and isinstance(sp.tilt_scale, float)
    assert sp.yawSide == -1 and isinstance(sp.yawSide, int)
    assert sp.D == 15.0 and isinstance(sp.D, float)
    assert sp.R == 12.5
    assert sp.theta == 55.0 and isinstance(sp.theta, float)
    assert not hasattr(sp, "not_a_parameter")
    assert sp.beta == SteeringParams().beta                        # missing keys keep their defaults
    for s, b in (("true", True), ("True", True), ("1", True), ("on", True), ("0", False), ("no", False)):
        assert SteeringParams.from_dict({"D_auto": s}).D_auto is b
    assert SteeringParams.from_dict(None) == SteeringParams()
    assert SteeringParams.from_dict({}) == SteeringParams()


def test_params_round_trip():
    sp = SteeringParams(R=14.0, yawSide=-1, D_auto=False, D=11.0, bias_axis_deg=37.0, tilt_scale=0.41)
    assert SteeringParams.from_dict(sp.to_dict()) == sp
    # QSettings hands everything back as strings
    assert SteeringParams.from_dict({k: str(v) for k, v in sp.to_dict().items()}) == sp
    md = steering.module_defaults()
    assert SteeringParams.from_dict({k: str(v) for k, v in md.to_dict().items()}) == md


# ------------------------------------------------------------------ spots, encoder, curves, plan
def test_spot_returns_fresh_arrays():
    sp = SteeringParams()
    for get in (lambda: steering.reference_spot(sp), lambda: steering.spot(sp, 0.0, 0.0),
                lambda: steering.spot(sp, 3.0, -1.0, "fast")):
        a = get()
        peak, x0, peak_at = float(a["I"].max()), list(a["x0"]), list(a["peakAt"])
        a["I"][:] = 0
        a["x0"][0] = 1e9
        a["peakAt"][0] = 1e9
        b = get()
        assert float(b["I"].max()) == peak and peak > 0.5
        assert list(b["x0"]) == x0 and list(b["peakAt"]) == peak_at


def test_kdc101_counts():
    assert steering.kdc101_counts(1000) == pytest.approx(34555, abs=1e-9)
    assert steering.kdc101_counts(0) == 0
    assert steering.kdc101_counts(-1.0) == pytest.approx(-34.555)


def test_peak_curve_names_and_normalisation():
    sp = SteeringParams()
    for bad in ("yaw", "pitch", "left", ""):
        with pytest.raises(ValueError):
            steering.peak_curve(sp, bad, [0.0])
    for act in steering.ACTUATORS:
        assert steering.peak_curve(sp, act, [0.0])[0]["rel"] == pytest.approx(1.0, abs=1e-12)
        rows = steering.peak_curve(sp, act, [-4.0, 0.0, 4.0])
        assert [r["travel_um"] for r in rows] == [-4.0, 0.0, 4.0]
        assert rows[1]["rel"] == pytest.approx(1.0, abs=1e-3)       # window carried over from -4 um: not bit-exact
        assert rows[0]["rel"] < 1 and rows[2]["rel"] < 1


def test_plan_offsets_default():
    sp = SteeringParams()
    xs = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
    rows = steering.plan_offsets(sp, xs)
    assert [r["X_um"] for r in rows] == xs
    for r in rows:
        assert r["converged"]
        assert r["X_got_um"] == pytest.approx(r["X_um"], abs=1e-5)
        assert r["upper_um"] == 0.0
        assert r["lower_counts"] == pytest.approx(steering.kdc101_counts(r["lower_um"]))
        assert r["rel"] is not None
    assert rows[3]["rel"] == pytest.approx(1.0, abs=1e-3)          # 'trend' quality vs the 'fine' aligned spot
    rel = {r["X_um"]: r["rel"] for r in rows}
    for side in (1, -1):                                           # peak falls off monotonically with |X|
        seq = [rel[side * x] for x in (0.0, 5.0, 10.0, 15.0)]
        assert all(a > b for a, b in zip(seq, seq[1:])), seq
    # lower travel has the opposite sign of X at the default orientation
    assert all(r["lower_um"] * r["X_um"] < 0 for r in rows if r["X_um"] != 0)


def test_plan_offsets_both_actuators_hold_Y():
    rows = steering.plan_offsets(SteeringParams(), [-10.0, 10.0], actuators="both")
    for r in rows:
        assert r["converged"]
        assert r["X_got_um"] == pytest.approx(r["X_um"], abs=1e-5)
        assert r["Y_got_um"] == pytest.approx(0.0, abs=1e-5)
        assert r["upper_um"] != 0.0


# ------------------------------------------------------------------ docs
def _doc_numbers(comment: str) -> list[tuple[float, int]]:
    """(value, decimals) of each number in a docs comment; the docs may write the minus sign as U+2212."""
    out = []
    for m in re.findall(r"[+\-−]?\d+(?:\.\d+)?", comment):
        out.append((float(m.replace("−", "-")), len(m.split(".")[1]) if "." in m else 0))
    return out


def test_docs_python_example_runs():
    """The 'From Python' block of docs/steering.md runs as written and prints what its comments say: the lower-only
    solve for X = +5 um agrees with its comment to the digits shown, and is the defaults' solve."""
    import textwrap
    doc = (steering.MODULE_DIR.parent / "docs" / "steering.md").read_text(encoding="utf-8")
    block = re.search(r"## From Python\n\n((?:    .*\n|\n)+)", doc).group(1)
    ns = {}
    exec(compile(textwrap.dedent(block), "docs/steering.md", "exec"), ns)
    line = next((ln for ln in block.splitlines() if 'sol["steering"].Y_um' in ln and "#" in ln), None)
    assert line is not None, 'the block no longer prints sol["lower_um"], sol["steering"].Y_um with a comment'
    (lower, dec_l), (y, dec_y) = _doc_numbers(line.split("#", 1)[1])[:2]
    assert abs(ns["sol"]["lower_um"] - lower) <= 0.5 * 10 ** -dec_l + 1e-9, (ns["sol"]["lower_um"], line)
    assert abs(ns["sol"]["steering"].Y_um - y) <= 0.5 * 10 ** -dec_y + 1e-9, (ns["sol"]["steering"].Y_um, line)
    ref = steering.travel_for_offset(SteeringParams(), 5.0, actuators="lower")
    assert ns["sol"]["lower_um"] == pytest.approx(ref["lower_um"], abs=1e-9)
    assert len(ns["rows"]) == 7 and len(ns["curve"]) == 41
