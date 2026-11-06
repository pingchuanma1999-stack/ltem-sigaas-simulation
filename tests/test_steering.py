"""Checks of the pump-spot steering module: the vendored optics core against its golden file, and the adapter
ltemsim/steering.py (actuator names, signs, frames, solver, parameter handling) against the module's conventions.

Run with:  python -m pytest tests/test_steering.py -q

The golden part mirrors ltem-steering-module/python/test_golden.py, but goes through steering.optics (the core as
this package loads it) and, a second time, through the adapter, so that a swapped actuator, a dropped parameter or
a sign flip in the adapter cannot hide behind a passing core.
"""
from __future__ import annotations

import json
import math
import os
import sys

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
    """SteeringParams equivalent to a golden case's core params (a numeric D switches D_auto off)."""
    d = dict(case["params_in"])
    if "D" in d and d["D"] not in ("auto", None):
        d["D_auto"] = False
    return SteeringParams.from_dict(d)


def _as_core_response(st: steering.Steering) -> dict:
    """An adapter Steering in the core's naming (lower = yaw, upper = pitch), to compare with golden.json."""
    return {"spot_um": st.spot_um, "port_um": st.port_um,
            "perUm": {"yaw": st.per_um["lower"], "pitch": st.per_um["upper"]},
            "perUmPort": {"yaw": st.port_per_um["lower"], "pitch": st.port_per_um["upper"]},
            "incidencePlane": st.incidence_plane, "along_um": st.along_um, "across_um": st.across_um,
            "defocus_um": st.defocus_um, "arrival_fs": st.arrival_fs,
            "yawTurnsBeamToFront": st.lower_turns_beam_to_front}


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


# ------------------------------------------------------------------ physical sanity through the adapter
def test_lower_actuator_moves_spot_left_and_to_front():
    sp = SteeringParams()
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
    sp = SteeringParams()
    st = steering.steer(sp, upper_um=1.0)
    assert st.spot_um[0] == pytest.approx(0.737, abs=1e-3)
    assert st.spot_um[1] == pytest.approx(-0.923, abs=1e-3)
    assert st.per_um["upper"][0] == pytest.approx(0.737, abs=1e-3)
    assert st.per_um["upper"][1] == pytest.approx(-0.923, abs=1e-3)


@pytest.mark.parametrize("yaw_side,pitch_side", [(-1, 1), (1, -1), (-1, -1)])
def test_mirrored_mount_flips_the_shift(yaw_side, pitch_side):
    """yawSide = -1 reverses the lower actuator's spot shift only, pitchSide = -1 the upper one's only (to within the
    ~0.5 % geometric asymmetry); the other actuator keeps the upright response."""
    up = steering.steer(SteeringParams()).per_um
    got = steering.steer(SteeringParams(yawSide=yaw_side, pitchSide=pitch_side)).per_um
    for a, side in (("lower", yaw_side), ("upper", pitch_side)):
        if side < 0:
            assert got[a] == pytest.approx((-up[a][0], -up[a][1]), rel=1e-2), a
        else:
            assert got[a] == pytest.approx(up[a], abs=1e-12), a


def test_upside_down_mount_needs_both_sides_flipped():
    """A mount turned 180 deg about the mirror normal (both sides -1): X = +5 um with the lower (horizontal-tilt)
    actuator alone needs +5.869 um, the opposite of the upright mount; flipping pitchSide alone leaves it at -5.869."""
    upright = steering.travel_for_offset(SteeringParams(), 5.0, actuators="lower")["lower_um"]
    flipped = steering.travel_for_offset(SteeringParams(yawSide=-1, pitchSide=-1), 5.0, actuators="lower")
    only_pitch = steering.travel_for_offset(SteeringParams(pitchSide=-1), 5.0, actuators="lower")["lower_um"]
    assert upright == pytest.approx(-5.869, abs=1e-3)
    assert flipped["converged"] and flipped["lower_um"] == pytest.approx(5.869, abs=1e-3)
    assert only_pitch == pytest.approx(upright, abs=1e-9)


def test_aligned_spot_is_diffraction_limited_and_elongated_along_incidence():
    sp = SteeringParams()
    s0 = steering.reference_spot(sp)
    assert s0["peak"] > 0.99
    assert math.hypot(*s0["peakAt"]) < 0.05
    plane = steering.steer(sp).incidence_plane
    inc = math.atan2(plane[1], plane[0])
    d = (math.degrees(s0["ang"] - inc) + 90) % 180 - 90           # axis directions are defined modulo pi
    assert abs(d) < 3
    assert 1.7 < s0["fwhmL"] / s0["fwhmS"] < 2.3


@pytest.mark.parametrize("lower,upper", [(5.0, 0.0), (0.0, 5.0), (-10.0, 3.0)])
def test_spot_window_follows_the_chief_ray(lower, upper):
    """The spot map sits where steer() puts the chief ray (same lab frame, um vs mm handled)."""
    sp = SteeringParams()
    s = steering.spot(sp, lower, upper, "fast")
    st = steering.steer(sp, lower, upper)
    dx, dy = steering.spot_offset_um(s, steering.reference_spot(sp))
    assert dx == pytest.approx(st.spot_um[0], abs=1e-6)
    assert dy == pytest.approx(st.spot_um[1], abs=1e-6)


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
    """Targets whose travel lies where the response is far from its slope at 0 (checked by direct steer() scans):
    the solver must land on them and flag them reached, not miss or stall."""
    sp = SteeringParams(bias_axis_deg=phi)
    sol = steering.travel_for_offset(sp, X, actuators=actuator)
    assert sol["converged"]
    assert sol["residual_um"] <= steering.REACH_TOL_UM
    assert sol[f"{actuator}_um"] == pytest.approx(travel, abs=0.01)
    st = steering.steer(sp, sol["lower_um"], sol["upper_um"])
    assert st.X_um == pytest.approx(X, abs=1e-5)


def test_travel_unreachable_target_is_flagged_not_raised():
    """At phi = -75 deg the lower actuator alone cannot bring X above ~+2.34 um (its maximum; a scan over +-500 um
    finds no crossing of 7.5): the solver stops at the closest point and says so."""
    sp = SteeringParams(bias_axis_deg=-75.0)
    sol = steering.travel_for_offset(sp, 7.5, actuators="lower")
    assert not sol["converged"]
    assert sol["residual_um"] > 5.0
    assert sol["upper_um"] == 0.0
    best = max(steering.steer(sp, t, 0.0).X_um for t in np.arange(-100.0, 100.5, 0.5))
    assert sol["steering"].X_um == pytest.approx(best, abs=1e-3)


def test_travel_converged_means_within_the_reach_tolerance():
    """'converged' (the link table's 'reached') is a physical tolerance, REACH_TOL_UM = 1e-4 um, not the iteration's
    1e-6 um stop: two Newton steps toward X = 20 um land 7e-5 um off, which is reached."""
    sp = SteeringParams()
    two = steering.travel_for_offset(sp, 20.0, actuators="lower", max_iter=2)
    assert 1e-6 < two["residual_um"] <= steering.REACH_TOL_UM
    assert two["converged"]
    one = steering.travel_for_offset(sp, 20.0, actuators="lower", max_iter=1)
    assert one["residual_um"] > steering.REACH_TOL_UM and not one["converged"]
    loose = steering.travel_for_offset(sp, 20.0, actuators="lower", tol_um=1.0, max_iter=1)
    assert loose["converged"] == (loose["residual_um"] <= 1.0)
    full = steering.travel_for_offset(sp, 20.0, actuators="lower")
    assert full["converged"] and full["residual_um"] <= 1e-6


@pytest.mark.parametrize("phi,actuator", [(20.7, "lower"), (-159.3, "lower"), (-51.4, "upper"), (128.6, "upper")])
def test_single_actuator_pure_x_mountings(phi, actuator):
    """docs/steering.md: one actuator alone scans along the bias axis at these phi (phi + 180 deg reverses the sign)."""
    px, py = steering.steer(SteeringParams(bias_axis_deg=phi)).per_um_sim[actuator]
    assert abs(py) < 2e-3 and abs(px) > 0.9


def test_travel_lower_sign_at_default_params():
    """+X (toward the port wall) needs the lower push rod to retract: about -5.87 um for X = +5 um."""
    sol = steering.travel_for_offset(SteeringParams(), 5.0, actuators="lower")
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
    """With D auto on, spot() and peak_curve() use the resolved diameter, not the D field (R = 12 mm: 9.8 mm,
    while the default D field and the default auto value are both 13.0 mm)."""
    sp = SteeringParams(R=12.0)
    assert sp.D_auto and sp.D == 13.0
    assert steering.beam_diameter_mm(sp) == pytest.approx(9.8)
    same = SteeringParams(R=12.0, D_auto=False, D=9.8)
    other = SteeringParams(R=12.0, D_auto=False, D=13.0)
    s, s_same, s_other = (steering.spot(p, 4.0, 0.0) for p in (sp, same, other))
    assert s["D"] == pytest.approx(9.8)
    assert s["fwhmL"] == pytest.approx(s_same["fwhmL"], abs=1e-12)
    assert s["fwhmL"] != pytest.approx(s_other["fwhmL"], abs=1e-3)
    c, c_same, c_other = (steering.peak_curve(p, a, [4.0])[0]["rel"] for p in (sp, same, other) for a in ("lower",))
    assert c == pytest.approx(c_same, abs=1e-12)
    assert c != pytest.approx(c_other, abs=1e-3)


def test_to_core_beam_diameter():
    assert steering.to_core(SteeringParams())["D"] == "auto"
    core = steering.to_core(SteeringParams(D_auto=False, D=9.5))
    assert core["D"] == 9.5 and isinstance(core["D"], float)
    assert steering.beam_diameter_mm(SteeringParams()) == pytest.approx(13.0)
    assert steering.beam_diameter_mm(SteeringParams(D_auto=False, D=9.5)) == 9.5


CORE_KEYS = ("R", "theta", "beta", "alpha", "dist", "L", "lam", "NA", "zLow", "portX", "foldY", "yawSide", "pitchSide")


def test_to_core_defaults_agree_with_the_core():
    sp = SteeringParams()
    core = steering.to_core(sp)
    assert set(core) == set(O.DEFAULTS)
    for k in CORE_KEYS:
        assert core[k] == getattr(sp, k)
        assert core[k] == O.DEFAULTS[k]                            # the defaults agree with the module README


def test_to_core_carries_every_core_parameter():
    """Every field set away from its default reaches the core (a constant equal to the default cannot pass)."""
    sp = SteeringParams(R=12.5, theta=55.0, beta=30.0, NA=0.3, zLow=2.0, alpha=40.0, dist=180.0, L=35.0, yawSide=-1,
                        pitchSide=1, portX=300.0, foldY=8.0, lam=780.0, D_auto=False, D=9.0, bias_axis_deg=25.0)
    core = steering.to_core(sp)
    assert set(core) == set(O.DEFAULTS)
    for k in CORE_KEYS:
        assert core[k] == getattr(sp, k), k
        if k not in ("pitchSide",):
            assert core[k] != O.DEFAULTS[k], k                     # the test really moves every parameter
    assert core["D"] == 9.0
    assert steering.to_core(SteeringParams(yawSide=1, pitchSide=-1))["pitchSide"] == -1
    assert steering.resolve(sp) == O.resolve_params({**{k: getattr(sp, k) for k in CORE_KEYS}, "D": 9.0})
    st = steering.steer(sp, 3.0, -2.0)
    r = O.steering_response({**{k: getattr(sp, k) for k in CORE_KEYS}, "D": 9.0}, yaw_um=3.0, pitch_um=-2.0)
    assert st.spot_um == pytest.approx(tuple(r["spot_um"]), abs=1e-12)
    # zLow, portX and foldY change the results the adapter reports
    base = SteeringParams()
    assert steering.spot(SteeringParams(zLow=2.0), 4.0, 0.0, "trend")["fwhmL"] != pytest.approx(
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
                                   "not_a_parameter": 3, "theta": 55})
    assert sp.D_auto is False
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
    sp = SteeringParams(R=14.0, yawSide=-1, D_auto=False, D=11.0, bias_axis_deg=37.0)
    assert SteeringParams.from_dict(sp.to_dict()) == sp
    # QSettings hands everything back as strings
    assert SteeringParams.from_dict({k: str(v) for k, v in sp.to_dict().items()}) == sp


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
def test_docs_python_example_runs():
    """The 'From Python' block of docs/steering.md runs as written and prints what its comments say."""
    import re
    import textwrap
    doc = (steering.MODULE_DIR.parent / "docs" / "steering.md").read_text(encoding="utf-8")
    block = re.search(r"## From Python\n\n((?:    .*\n|\n)+)", doc).group(1)
    ns = {}
    exec(compile(textwrap.dedent(block), "docs/steering.md", "exec"), ns)
    assert ns["sol"]["lower_um"] == pytest.approx(-5.869, abs=1e-3)
    assert len(ns["rows"]) == 7 and len(ns["curve"]) == 41
