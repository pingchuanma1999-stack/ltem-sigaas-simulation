"""Pump-spot steering by the external KS1-Z8 mirror: adapter between this package and the vendored optics core.

The core, ``ltem-steering-module/python/ltem_optics.py``, is used unchanged: it reproduces
``ltem-steering-module/tests/golden.json`` (see that folder's README for the physics, the algorithm and the
validation).  Every conversion between its conventions and ours happens in this file and nowhere else.

Conventions
-----------
* Actuators.  ``lower_um`` is the lower Z812 of the mount: horizontal tilt, the core's ``yaw``.
  ``upper_um`` is the upper Z812: vertical tilt, the core's ``pitch``.  Travel is the push-rod extension in um;
  + = the rod extends = the KDC101 reading increases (1 mm = 34,555 encoder counts).
* Lab frame of the core, on the sample, in um: origin at the tip apex (= the aligned focus), +X toward the
  entrance-port wall ("right" seen from the open front of the AFM box), +Y toward the back panel.
* Simulation frame (ltemsim.params): x runs along the bias field, electrode to electrode, and the experimental
  offset X is the spot position relative to the tip.  The tip stays at the lab origin while the mirror moves the
  spot, so X is the spot shift projected on the device's +x axis.  How the sample sits on the AFM stage is the
  parameter ``bias_axis_deg``, the azimuth of +x in the lab frame from lab +X toward lab +Y:

      X = dX cos(phi) + dY sin(phi)        (along the bias axis)
      Y = -dX sin(phi) + dY cos(phi)       (across it, along the electrode edges; the 2-D model ignores it)

  With the default phi = 0 the lower actuator moves X by -0.848 um per um of travel.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent.parent / "ltem-steering-module"
CORE_PATH = MODULE_DIR / "python" / "ltem_optics.py"
GOLDEN_PATH = MODULE_DIR / "tests" / "golden.json"
REFERENCE_PAGE = MODULE_DIR / "reference" / "ltem-steering.html"
KDC101_COUNTS_PER_UM = 34.555          # Z812 encoder: 1 mm = 34,555 counts (~29 nm per count)
ACTUATORS = ("lower", "upper")
_CORE_AXIS = {"lower": "yaw", "upper": "pitch"}


def _load_core():
    """Import the vendored core under a private name (its caches then belong to this package)."""
    name = "ltemsim._ltem_optics"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, CORE_PATH)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


optics = _load_core()


@dataclass
class SteeringParams:
    # --------------------------------------------------- parabolic mirror and sample
    R: float = 16.0                 # mm, effective focal length of the parabolic mirror (chief point to focus).
                                    #   Reported for another neaspec THz-TDS s-SNOM; CALIBRATE (spot shifts scale with R/L)
    theta: float = 60.0             # deg, angle of incidence on the sample (neaspec standard)
    beta: float = 45.0              # deg, azimuth of the plane of incidence from the front-back axis toward the port side
                                    #   (user's sketch: 40-50 deg). Leaves the lower actuator's left-right (lab X) shift
                                    #   unchanged, but changes its front-back shift, the upper actuator's shift in both
                                    #   directions and the plane of incidence (along/across split, defocus): it matters
                                    #   for the upper actuator, for both, and for any bias_axis_deg other than 0
    NA: float = 0.42                # illumination aperture: cone of half-angle asin(NA) around the aligned chief ray
                                    #   (neaspec patent example, 50 deg cone; a published neaspec IR mirror has 0.46)
    zLow: float = 1.0               # mm, height of the parabolic mirror's lower edge above the sample (neaspec patent)
    # ----------------------------------------------------------------- steering mirror
    alpha: float = 45.0             # deg, angle of incidence on the steering mirror (a 90 deg fold; setup photo)
    dist: float = 250.0             # mm, steering mirror to entrance-port plane (sets the beam shift at the port)
    L: float = 37.7                 # mm, lever arm from actuator contact to pivot (KS1-Z8: 0.2 um step = 1.1 arcsec)
    yawSide: int = 1                # +1: lower (horizontal-tilt) actuator right of the pivot seen from the actuator side
                                    #   (upright mount); -1: left of it. A mount turned upside down (rotated 180 deg about
                                    #   the mirror normal) needs yawSide = pitchSide = -1, and its 'lower' (horizontal-tilt)
                                    #   actuator is then physically the upper one
    pitchSide: int = 1              # +1: upper (vertical-tilt) actuator above the pivot (upright mount); -1: below it
                                    #   (an upside-down mount needs yawSide = -1 as well)
    portX: float = 260.0            # mm, x of the entrance-port plane (model geometry, ~1 % effect)
    foldY: float = 6.0              # mm, y of the fold-mirror centre behind the tip (model geometry, negligible effect)
    # ------------------------------------------------------------------------ pump beam
    lam: float = 820.0              # nm, pump wavelength (Mai Tai, as Params.wavelength_nm)
    D_auto: bool = True             # use the input-beam diameter that gives the smallest spot (13.0 mm with the defaults)
    D: float = 13.0                 # mm, 1/e^2 diameter of the input beam, used when D_auto is off. Measure before the AFM
    # --------------------------------------------------------------- sample orientation
    bias_axis_deg: float = 0.0      # deg, azimuth of the simulation's +x (bias axis, toward the +x electrode) in the lab
                                    #   frame, from +X (port wall) toward +Y (back). ASSUMED: set it from the sample mounting

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SteeringParams":
        """Build from a (possibly partial) dict; unknown keys are ignored, values are coerced to the field types."""
        names = {f.name: f for f in fields(cls)}
        kw = {}
        for k, v in (d or {}).items():
            if k not in names:
                continue
            kind = type(names[k].default)
            if kind is bool and not isinstance(v, bool):      # QSettings and JSON presets may hand back strings
                v = str(v).strip().lower() in ("1", "true", "yes", "on")
            kw[k] = kind(float(v)) if kind is int else kind(v)
        return cls(**kw)


def to_core(sp: SteeringParams) -> dict:
    """Core parameter dict (geometry in mm, angles in deg, wavelength in nm, D in mm or 'auto')."""
    return {"R": float(sp.R), "theta": float(sp.theta), "beta": float(sp.beta), "alpha": float(sp.alpha),
            "dist": float(sp.dist), "L": float(sp.L), "lam": float(sp.lam), "D": "auto" if sp.D_auto else float(sp.D),
            "NA": float(sp.NA), "zLow": float(sp.zLow), "portX": float(sp.portX), "foldY": float(sp.foldY),
            "yawSide": 1 if sp.yawSide >= 0 else -1, "pitchSide": 1 if sp.pitchSide >= 0 else -1}


def resolve(sp: SteeringParams) -> dict:
    """Core parameters with D='auto' resolved (50-90 ms the first time per parameter set, then cached by the core)."""
    return optics.resolve_params(to_core(sp))


def beam_diameter_mm(sp: SteeringParams) -> float:
    return float(resolve(sp)["D"])


# ------------------------------------------------------------------ frames
def lab_to_sim(dx_um: float, dy_um: float, sp: SteeringParams) -> tuple[float, float]:
    """Lab-frame spot shift (um) -> (X along the bias axis, Y across it)."""
    phi = math.radians(sp.bias_axis_deg)
    c, s = math.cos(phi), math.sin(phi)
    return dx_um * c + dy_um * s, -dx_um * s + dy_um * c


def sim_to_lab(X_um: float, Y_um: float, sp: SteeringParams) -> tuple[float, float]:
    """Inverse of lab_to_sim."""
    phi = math.radians(sp.bias_axis_deg)
    c, s = math.cos(phi), math.sin(phi)
    return X_um * c - Y_um * s, X_um * s + Y_um * c


def kdc101_counts(travel_um: float) -> float:
    """Change of the KDC101 encoder reading for a push-rod travel (counts)."""
    return travel_um * KDC101_COUNTS_PER_UM


# ------------------------------------------------------------------ chief ray
@dataclass(frozen=True)
class Steering:
    """Where the spot goes for one pair of actuator travels (exact chief-ray trace)."""
    lower_um: float
    upper_um: float
    spot_um: tuple                  # (x, y) lab frame: + port wall, + back
    X_um: float                     # along the simulation's bias axis (= experimental spot-tip offset X)
    Y_um: float                     # across the bias axis
    port_um: tuple                  # (y, z) beam shift at the entrance port: + back, + up
    along_um: float                 # along the plane of incidence (+ = away from the parabolic mirror)
    across_um: float
    defocus_um: float
    arrival_fs: float               # change of the pump arrival time
    per_um: dict                    # {"lower": (x, y), "upper": (x, y)} lab spot shift per um of travel
    per_um_sim: dict                # same in (X, Y)
    port_per_um: dict               # {"lower": (y, z), "upper": (y, z)} port shift per um of travel
    incidence_plane: tuple          # horizontal unit vector of the aligned chief ray after the parabolic mirror
    lower_turns_beam_to_front: bool

    @property
    def shift_um(self) -> float:
        return math.hypot(*self.spot_um)


def steer(sp: SteeringParams, lower_um: float = 0.0, upper_um: float = 0.0) -> Steering:
    """Chief-ray response; raises ValueError when the chief ray misses the parabolic mirror."""
    r = optics.steering_response(to_core(sp), yaw_um=float(lower_um), pitch_um=float(upper_um))
    dx, dy = r["spot_um"]
    X, Y = lab_to_sim(dx, dy, sp)
    per = {a: tuple(r["perUm"][_CORE_AXIS[a]]) for a in ACTUATORS}
    return Steering(
        lower_um=float(lower_um), upper_um=float(upper_um), spot_um=(dx, dy), X_um=X, Y_um=Y,
        port_um=tuple(r["port_um"]), along_um=r["along_um"], across_um=r["across_um"],
        defocus_um=r["defocus_um"], arrival_fs=r["arrival_fs"], per_um=per,
        per_um_sim={a: lab_to_sim(*per[a], sp) for a in ACTUATORS},
        port_per_um={a: tuple(r["perUmPort"][_CORE_AXIS[a]]) for a in ACTUATORS},
        incidence_plane=tuple(r["incidencePlane"]), lower_turns_beam_to_front=r["yawTurnsBeamToFront"])


SOLVER_STEP_UM = 0.01          # travel step of the solver's finite-difference Jacobian
REACH_TOL_UM = 1e-4             # a landing point this close to the target counts as reached ('converged')
_MAX_HALVINGS = 30              # backtracking: a Newton step is halved at most this often


def _sim_spot(p: dict, Q0, sp: SteeringParams, t) -> np.ndarray | None:
    """(X, Y) of the chief-ray hit for travels t = (lower, upper) um, or None when it misses the parabolic mirror.
    The same trace as steer() (core parameters p, aligned hit Q0), without its per-um differences."""
    c = optics.trace_chief(p, float(t[1]) / 1000, float(t[0]) / 1000)
    if "Q" not in c:
        return None
    dQ = (c["Q"] - Q0) * 1000
    return np.array(lab_to_sim(float(dQ[0]), float(dQ[1]), sp))


def travel_for_offset(sp: SteeringParams, X_um: float, Y_um: float = 0.0, actuators: str = "both",
                      tol_um: float = 1e-6, max_iter: int = 50) -> dict:
    """Actuator travels that put the spot at (X, Y) in the simulation frame (exact trace, damped Newton).

    actuators: "both" solves for X and Y; "lower" or "upper" moves one actuator only, matches X and reports the
    Y it gives.  Returns {lower_um, upper_um, steering, residual_um, converged}.
    The Jacobian is re-evaluated by finite differences at every iterate (the per-um response of steer() holds at
    travel 0 only), and a step that overshoots, i.e. lands farther from the target or off the parabolic mirror, is
    halved until it gets closer.  The iteration stops at tol_um; 'converged' means the landing point is within
    max(tol_um, REACH_TOL_UM) of the target.  Raises ValueError when the aligned chief ray misses the parabolic
    mirror, or when the actuator(s) do not move the spot along the target direction(s) at travel 0.
    """
    if actuators not in ("both", "lower", "upper"):
        raise ValueError(f"actuators must be 'both', 'lower' or 'upper', not {actuators!r}")
    steer(sp)                                       # ValueError when the aligned chief ray misses
    p = to_core(sp)
    Q0 = optics.trace_chief(p, 0.0, 0.0)["Q"]
    both = actuators == "both"
    cols = (0, 1) if both else ((0,) if actuators == "lower" else (1,))
    goal = np.array([X_um, Y_um], dtype=float)

    def error(f):                                   # what is solved for: (X, Y) with both actuators, else X only
        e = goal - f
        return e if both else e[:1]

    t = np.zeros(2)
    f_t = _sim_spot(p, Q0, sp, t)
    err = error(f_t)
    res = float(np.linalg.norm(err))
    for it in range(max_iter):
        if res <= tol_um:
            break
        J = np.zeros((2, 2))
        for k in cols:                              # forward differences; backward ones at the edge of the mirror
            for h in (SOLVER_STEP_UM, -SOLVER_STEP_UM):
                tt = t.copy()
                tt[k] += h
                f_h = _sim_spot(p, Q0, sp, tt)
                if f_h is not None:
                    J[:, k] = (f_h - f_t) / h
                    break
        step = np.zeros(2)
        if both:
            singular = abs(np.linalg.det(J)) < 1e-12
            if not singular:
                step = np.linalg.solve(J, err)
        else:
            k = cols[0]
            singular = abs(J[0, k]) < 1e-12
            if not singular:
                step[k] = err[0] / J[0, k]
        if singular:
            if it == 0:
                raise ValueError(f"the {actuators} actuator{'s do' if both else ' does'} not move the spot along "
                                 "the bias axis" + (" and across it independently" if both else ""))
            break                                   # a stationary point of the landing position: stop there
        lam = 1.0
        for _ in range(_MAX_HALVINGS):
            t_new = t + lam * step
            f_new = _sim_spot(p, Q0, sp, t_new)
            if f_new is not None:
                e_new = error(f_new)
                r_new = float(np.linalg.norm(e_new))
                if r_new < res:
                    break
            lam *= 0.5
        else:
            break                                   # no step gets closer: the nearest the actuator(s) can get
        t, f_t, err, res = t_new, f_new, e_new, r_new
    s = steer(sp, t[0], t[1])
    res = float(np.hypot(X_um - s.X_um, Y_um - s.Y_um)) if both else abs(X_um - s.X_um)
    return {"lower_um": float(t[0]), "upper_um": float(t[1]), "steering": s, "residual_um": res,
            "converged": res <= max(tol_um, REACH_TOL_UM)}


# ------------------------------------------------------------------ spot shape (scalar diffraction)
def spot(sp: SteeringParams, lower_um: float = 0.0, upper_um: float = 0.0, quality="fine", W_start=None):
    """The core's spot dict, or None when fewer than 20 rays reach the sample.

    Keys (see the module README): I (float32 M x M, row = +y back, column = +x port side, relative to the window
    centre x0 in lab mm), W (window half-width, um), peak (Strehl-like), rel (peak relative to the aligned spot,
    per unit laser power), k (I * k = intensity relative to the aligned peak), fwhmL, fwhmS (um), ang (rad, long
    axis from +X toward +Y), area (um^2), peakAt, cx, cy (um relative to x0), D (mm).
    'fine' takes ~6-50 ms, 'fast' ~3-32 ms, 'trend' ~2-29 ms; keep 'fine' at large travel off the GUI thread.
    """
    return optics.spot_shape(resolve(sp), yaw_um=float(lower_um), pitch_um=float(upper_um), quality=quality,
                             W_start=W_start)


def reference_spot(sp: SteeringParams):
    """Aligned spot (both actuators at 0, 'fine'); a fresh copy, safe to modify."""
    return spot(sp, 0.0, 0.0, "fine")


def half_max_contour(s: dict) -> np.ndarray:
    """Half-maximum contour of a spot as (n, 4) segments [x1, y1, x2, y2] in um relative to s['x0']."""
    return optics.contour(s, 0.5 * s["peak"])


def spot_offset_um(s: dict, ref: dict) -> tuple[float, float]:
    """Lab-frame position (um) of spot s's window centre relative to the reference spot's centre."""
    return ((s["x0"][0] - ref["x0"][0]) * 1e3, (s["x0"][1] - ref["x0"][1]) * 1e3)


def peak_curve(sp: SteeringParams, actuator: str, travels_um, quality="trend") -> list[dict]:
    """[{travel_um, rel, fwhmL, fwhmS, area, ang}, ...] moving one actuator ('lower' or 'upper'); rel is None
    where no spot forms.  Normalised to an aligned spot of the same quality (travel 0 reads 1)."""
    if actuator not in _CORE_AXIS:
        raise ValueError(f"actuator must be 'lower' or 'upper', not {actuator!r}")
    return optics.peak_curve(resolve(sp), _CORE_AXIS[actuator], [float(t) for t in travels_um], quality=quality)


def plan_offsets(sp: SteeringParams, X_list_um, actuators: str = "lower", quality="trend") -> list[dict]:
    """For each target offset X (Y = 0 when both actuators are used): the travels, the exact landing point and
    the predicted spot (rel, FWHM, defocus).  This is the table that connects the mirror to the tip-position chain."""
    rows = []
    for X in X_list_um:
        sol = travel_for_offset(sp, float(X), 0.0, actuators)
        st = sol["steering"]
        s = spot(sp, sol["lower_um"], sol["upper_um"], quality)
        rows.append({
            "X_um": float(X), "lower_um": sol["lower_um"], "upper_um": sol["upper_um"],
            "lower_counts": kdc101_counts(sol["lower_um"]), "upper_counts": kdc101_counts(sol["upper_um"]),
            "X_got_um": st.X_um, "Y_got_um": st.Y_um, "defocus_um": st.defocus_um, "arrival_fs": st.arrival_fs,
            "rel": None if s is None else s["rel"], "fwhmL": None if s is None else s["fwhmL"],
            "fwhmS": None if s is None else s["fwhmS"], "converged": sol["converged"]})
    return rows
