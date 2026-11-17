# Pump steering (KS1-Z8 mirror)

In the non-local measurement the tip stays put and the pump spot is moved on the
sample by an external Thorlabs KS1-Z8 mirror mount: two Z812 actuators (KDC101
controllers) tilt the last mirror before the AFM entrance port. The steering module
answers, for a pair of push-rod travels:

* where the focused spot lands on the sample (exact chief-ray trace: steering mirror →
  port → fold mirror → off-axis semi-paraboloid → tilted sample);
* how its shape and peak intensity change (scalar diffraction with the parabolic
  mirror's clear aperture);
* which input-beam diameter gives the smallest spot;
* how the peak intensity falls off with travel.

Its link to this package is one number: the spot shift projected on the bias axis is
the experimental offset X that the tip-position chain and the steady-state tip scans
are plotted against (`docs/model.md`, `docs/pulse_chain.md`).

The physics, the full algorithm, the validation and the sources are in
[`ltem-steering-module/README.md`](../ltem-steering-module/README.md); the brief the
module came with is
[`ltem-steering-module/HANDOFF_PROMPT.md`](../ltem-steering-module/HANDOFF_PROMPT.md).
This page covers only the integration. The module README's numbers are for the module's
own defaults (R = 16 mm, θ = 60°, D = 13.0 mm); this package's defaults are calibrated
(R = 5.332 mm, θ = 45°, see "Calibration of the spot shift"), so its numbers differ, and
`steering.module_defaults()` gives the module's set as `SteeringParams`.

## Where things live

| path | content |
|---|---|
| `ltem-steering-module/` | the module as delivered: JS reference core, NumPy port, `tests/golden.json`, the interactive page `reference/ltem-steering.html` (three.js 3-D model, bilingual). The cores and `golden.json` are **unchanged**; do not edit them, and regenerate `golden.json` (`node tests/make_golden.mjs`) only for a deliberate physics change. The page has a display fix in its 3-D code and a widened R slider (`#pR` 2–30 mm in 0.01, was 6–30 in 0.5) for the calibrated R (see "Limitations" below). |
| `ltem-steering-module/python/ltem_optics.py` | the NumPy core used here (only dependency NumPy) |
| `ltemsim/steering.py` | the one adapter: parameters (calibrated defaults; `CALIBRATION`, `calibrate_R`, `module_defaults`), actuator names, signs, frames, the offset solver. Every conversion between the core's conventions and ours is in this file and nowhere else. It loads the core under the private name `ltemsim._ltem_optics` (`steering.optics`), so the core's caches belong to this package. |
| GUI: Simulate → Pump steering | the desktop page (tabs below); form fields from `paramspec.steering_specs()`, 3D tab in `ltemsim_gui/widgets/steering3d.py` |
| `tests/test_steering.py` | golden values through the core and again through the adapter; actuator signs, mirrored mounts, aligned-spot shape, frame round trips, the offset solver, parameter handling, `plan_offsets` |
| `tests/test_gui_steering.py` | the GUI page (offscreen) |

## Conventions

### Actuators

| adapter (`ltemsim.steering`, GUI) | core | KS1-Z8 position, seen from the actuator side | tilt |
|---|---|---|---|
| `lower` (`lower_um`) | `yaw` | lower right (the pivot, the manual knob, is lower left) | horizontal |
| `upper` (`upper_um`) | `pitch` | upper left | vertical |

Response to +1 µm of travel with the default (calibrated) parameters (lab frame, below):

| actuator | spot shift (ΔX, ΔY) | size | direction | port shift | defocus |
|---|---|---|---|---|---|
| lower | (−0.2824, −0.0932) µm | 0.2974 µm | left and toward the front, mostly across the plane of incidence | 13.3 µm toward the front | 0.095 µm |
| upper | (+0.1419, −0.2360) µm | 0.2754 µm | right and toward the front, along the plane of incidence toward the parabolic mirror | 9.4 µm down | 0.189 µm |

One 0.2 µm step (1.1″) moves the spot 59.6 nm with the lower and 55.0 nm with the upper
actuator; one KDC101 count 8.6 nm and 8.0 nm. With the module's own defaults (R = 16 mm,
θ = 60°) the same table reads (−0.848, −0.321) and (+0.737, −0.923) µm per µm, as in the
module README.

* Travel is push-rod extension in µm. **+ = the rod extends = the KDC101 reading
  increases.** 1 mm = 34,555 encoder counts (`KDC101_COUNTS_PER_UM = 34.555`, ≈ 29 nm per
  count); `kdc101_counts(travel_um)` converts. The reading comes from the motor encoder,
  not from the rod tip.
* The mirror turns by δ = travel / L. The two directions are 103° apart on the sample
  (108° with the module's θ = 60°), so either actuator alone also moves the spot across
  the bias axis.
* `yawSide` / `pitchSide` describe where the actuators sit, seen from the actuator side:
  `yawSide` = +1 / −1 puts the horizontal-tilt (lower) actuator right / left of the pivot,
  `pitchSide` = +1 / −1 the vertical-tilt (upper) actuator above / below it. Each −1 flips
  that actuator's shift. A mount turned upside down (rotated 180° about the mirror normal)
  needs **both** = −1; its "lower" (horizontal-tilt) actuator is then physically the upper
  one.
* Per-µm coefficients (`Steering.per_um`) are forward differences over +1 µm at the
  current geometry; the response is slightly nonlinear at large travel (X = +5 µm needs
  −17.136 µm of lower travel; the linear estimate is −5/0.2824 = −17.71 µm).
  The coefficients hold at travel 0; `travel_for_offset` iterates on the exact trace
  (damped Newton, Jacobian re-evaluated at every step, overshooting steps halved) and
  marks a target as reached (`converged`) when the spot lands within 1e-4 µm of it
  (`REACH_TOL_UM`).

### Lab frame (the core's, used unchanged)

* On the sample, origin at the tip apex = the aligned focus; the sample is z = 0.
* **+X** toward the entrance-port wall ("right" seen from the open front of the AFM box),
  **+Y** toward the back panel (−Y = "front", the open side), **+Z** up.
* The aligned chief ray leaves the parabolic mirror (front right) heading toward the back
  left; its horizontal direction (−0.707, +0.707) with the defaults is the plane of
  incidence (`Steering.incidence_plane`). `along_um` is the shift along it (+ = away from
  the parabolic mirror), `across_um` the shift perpendicular to it.

### Units

| quantity | unit |
|---|---|
| geometry: `R`, `dist`, `L`, `portX`, `foldY`, `zLow`, `D` | mm |
| angles: `theta`, `beta`, `alpha`, `bias_axis_deg` | deg (`s["ang"]` of a spot is in rad) |
| wavelength `lam` | nm |
| actuator travel, spot shift, port shift, `along`, `across`, defocus, FWHM, window half-width `W` | µm |
| spot area | µm² |
| arrival-time change | fs (defocus × 3.336 fs/µm) |
| spot window centre `s["x0"]` | **lab mm** (`spot_offset_um(s, ref)` gives it in µm relative to another spot) |

### Spot arrays

* `s["I"]` is a float32 M × M array: M = 72 (`'fine'`), 48 (`'fast'`), 40 (`'trend'`).
* Row index b runs along **+y (back)**, column index a along **+x (port side)**; grid
  u_a = −W + a·2W/(M − 1) µm, the same for v_b, around the window centre `s["x0"]`
  (the chief-ray hit point), so the pixel centres reach ±W. For matplotlib, with
  px = 2W/(M − 1): `imshow(s["I"], origin="lower", extent=(-W - px/2, W + px/2,
  -W - px/2, W + px/2))` in µm relative to `x0` (as `steering_views._heat`).
* `peakAt`, `cx`, `cy` are µm relative to `x0`; `ang` is the long-axis angle from +X
  toward +Y; `half_max_contour(s)` returns (n, 4) segments [x1, y1, x2, y2] in µm relative
  to `x0`.
* Normalisations: `peak` is Strehl-like (≈ 1 for the aligned beam); `I * s["k"]` is the
  intensity relative to the aligned peak; `rel` is the peak per unit laser power relative
  to the aligned spot (clipping included), i.e. "peak intensity, original = 100 %". The
  intensity is across the beam; per unit sample area it is smaller by ≈ cos θ (0.71 at
  the default θ = 45°), a constant at fixed geometry.

### Simulation frame and `bias_axis_deg`

The simulation (`ltemsim.params`) is a 2-D x–z cross-section with x along the bias field,
electrode to electrode. The experimental X is the spot position relative to the tip
along +x (the model holds the spot at the gap centre and puts the tip at x_tip = −X). In
the experiment the tip stays at the lab origin while the mirror moves the spot, so X is
the lab spot shift projected on the device's +x axis. How the sample sits on the AFM
stage is one parameter, φ = `bias_axis_deg`: the azimuth of the simulation's +x in the
lab frame, measured from +X (port wall) toward +Y (back).

    X =  ΔX cos φ + ΔY sin φ        (along the bias axis: the offset X of the simulation)
    Y = −ΔX sin φ + ΔY cos φ        (across it, along the electrode edges)

`lab_to_sim` and `sim_to_lab` implement this pair; `Steering.X_um`, `Y_um` and
`per_um_sim` are already projected. The 2-D model has no y, so Y is reported but not
used.

**The default φ = 0 is an assumption** (+x pointing at the port wall). Set it from the
sample mounting: φ is the lab azimuth of the line from the model's −x electrode to its +x
electrode. With the recorded wiring (`anode_side = -x`) that is the line from the
electrode carrying + at "+25 V" toward the other one. Everything below that mentions X
depends on it:

| φ | lower: (X, Y) per +1 µm | upper: (X, Y) per +1 µm |
|---|---|---|
| 0° (default) | (−0.2824, −0.0932) | (+0.1419, −0.2360) |
| 30° | (−0.2911, +0.0605) | (+0.0049, −0.2754) |
| 90° (+x toward the back) | (−0.0932, +0.2824) | (−0.2360, −0.1419) |
| 18.3° | (−0.2974, 0) | (+0.0606, −0.2687) |
| −59.0° | (−0.0656, −0.2900) | (+0.2754, 0) |

The last two rows are the mountings for which one actuator alone scans purely along the
bias axis; the same holds at φ + 180° with the travel sign reversed (−161.7° for the
lower, +121.0° for the upper actuator). For any other φ, a pure-X scan needs both
actuators (`actuators="both"`). These angles depend on θ (and β), hardly on R: with the
module's θ = 60° they are 20.7° / −159.3° (lower) and −51.4° / +128.6° (upper).

## Parameters

`SteeringParams` (dataclass in `ltemsim/steering.py`). `to_dict()` / `from_dict()` round
trip it; `from_dict` takes a partial dict, ignores unknown keys and coerces strings
(QSettings, JSON presets).

| field | default | unit | meaning | status |
|---|---|---|---|---|
| `R` | 5.332 | mm | effective focal length of the parabolic mirror (chief point → focus) | **calibrated** from the 55 nm per 0.2 µm step of Pizzuto et al. (upper actuator, θ = 45°; see "Calibration of the spot shift"); an effective value, about half a plausible real focal length: replace it with a camera calibration. Module default 16 (reported for another neaspec THz-TDS s-SNOM) |
| `theta` | 45.0 | deg | angle of incidence on the sample | **assumed** for the calibration; neaspec standard and module default 60 |
| `beta` | 45.0 | deg | azimuth of the plane of incidence from the front–back axis toward the port side (GUI: 0–89°; β < 0 would put the pump on the THz collection half) | user's sketch (40–50°). Leaves the lower actuator's left–right (lab X) shift unchanged, but changes its front–back shift, the upper actuator's shift in both directions and the plane of incidence (along/across split, defocus): it matters for the upper actuator, for both, and for any φ ≠ 0 |
| `NA` | 0.42 | – | illumination aperture: cone of half-angle asin(NA) around the aligned chief ray | neaspec patent example; a published neaspec IR mirror has 0.46 |
| `zLow` | 1.0 | mm | height of the parabolic mirror's lower edge above the sample | neaspec patent |
| `alpha` | 45.0 | deg | angle of incidence on the steering mirror (90° fold) | setup photo |
| `dist` | 250.0 | mm | steering mirror → entrance-port plane | setup photo; sets the port shift, ~1 % on the spot |
| `L` | 37.7 | mm | lever arm, actuator contact → pivot | Thorlabs (1.1″ per 0.2 µm step) |
| `yawSide` | +1 | ±1 | horizontal-tilt (lower) actuator right (+1) / left (−1) of the pivot, seen from the actuator side | KS1-Z8 drawing, setup photo |
| `pitchSide` | +1 | ±1 | vertical-tilt (upper) actuator above (+1) / below (−1) the pivot; an upside-down mount (180° about the mirror normal) needs `yawSide` = `pitchSide` = −1 | KS1-Z8 drawing, setup photo |
| `portX` | 260.0 | mm | x of the entrance-port plane | model geometry, ~1 % effect |
| `foldY` | 6.0 | mm | y of the fold-mirror centre behind the tip | model geometry, negligible |
| `lam` | 820.0 | nm | pump wavelength | as `Params.wavelength_nm` |
| `D_auto` | True | – | use the input-beam diameter for the smallest spot (4.4 mm with the defaults; 13.0 mm with the module's R = 16 mm, θ = 60°) | the optics are aligned for the smallest spot |
| `D` | 4.4 | mm | 1/e² input-beam diameter, used when `D_auto` is off | **measure** before the AFM |
| `bias_axis_deg` | 0.0 | deg | azimuth of the simulation's +x in the lab frame | **assumed**; set from the sample mounting |

The directions depend on θ, β and α, and hardly on R or L (≤ 0.2° for R = 5.3–16 mm or
L = 20–60 mm); the magnitudes scale with R/L; the spot shape depends strongly on D and NA.
R is the parameter calibrated against a measured step (next section).

## Calibration of the spot shift

**The data.** One 0.2 µm step of a Z812 (0.0002 mm on the KDC101, the KS1-Z8's minimum
repeatable step of 1.1″) moves the pump spot by ≈ 55 nm, the estimate of Pizzuto et al.
(both papers; module README, "Sources"). It is quoted for the pitch direction of the
mirror, so it is taken as the **upper** (vertical-tilt) actuator. The angle of incidence
on the sample inside the AFM is **assumed** to be 45° (θ = 45°; the neaspec standard and
the module's default is 60°). Both are recorded in `steering.CALIBRATION`:

    {"actuator": "upper", "travel_um": 0.2, "spot_um": 0.055, "theta": 45.0, "source": ...}

**Why R is the fitted parameter.** The lever arm is fixed by the KS1-Z8 specification,
L = 37.7 mm (module: 1.1″ ≈ 5.3 µrad per 0.2 µm step); the step tilts the mirror by
δ = 0.2 µm / 37.7 mm = 5.305 µrad ≈ 1.09″. The mirror turns by δ = travel / L,
the beam by ≈ 2δ, and the parabolic mirror turns that angle into a shift ≈ 2δ · R at the
focus, so every spot shift scales with R/L. With L fixed, the measured step sets R, the
one geometric number not known for this instrument. `steering.calibrate_R(sp, actuator,
travel_um, spot_um)` solves the exact chief-ray trace for R, every other parameter of `sp`
kept. The travel is signed and starts from the aligned position (both actuators at 0). R is
searched outward from the linear estimate R · spot / shift(R) in steps of × 1.25
(`CALIB_RATIO`); the sign change nearest the estimate is refined by Brent's method, and an R
for which the step takes the chief ray off the parabolic mirror is skipped. `ValueError` for a
bad input, or when no R in (R/20, 20 R) reproduces the shift:

    from ltemsim.steering import SteeringParams, calibrate_R, module_defaults
    calibrate_R(SteeringParams(), "upper", 0.2, 0.055)    # 5.332 mm  (θ = 45°: the default R)
    calibrate_R(module_defaults(), "upper", 0.2, 0.055)   # 3.728 mm  if θ were 60°
    calibrate_R(SteeringParams(), "lower", 0.2, 0.055)    # 4.921 mm  if the 55 nm were the horizontal step

The fitted R belongs to the assumed θ and to the choice of actuator: the pair (R, θ)
matters, not R alone.

**The result:** R = 5.332 mm, the new default (θ = 45°, D auto = 4.4 mm). Per 0.2 µm step,
exact trace, lab frame:

| actuator | spot shift (ΔX, ΔY) per 0.2 µm | size | per µm of travel | per KDC101 count |
|---|---|---|---|---|
| upper (pitch, calibrated) | (+28.31, −47.16) nm | 55.00 nm | (+0.1419, −0.2360) µm, 0.2754 µm | 8.0 nm |
| lower (yaw, derived) | (−56.55, −18.81) nm | 59.60 nm | (−0.2824, −0.0932) µm, 0.2974 µm | 8.6 nm |

So the horizontal (lower) actuator moves the spot **59.6 nm per step: 56.6 nm to the left
plus 18.8 nm toward the front** (per µm of travel: 0.282 µm left, 0.093 µm toward the
front). The per-µm values are forward differences over 1 µm, the per-count values are
per µm ÷ 34.555.

**Small-angle cross-check** (δ = 0.2 µm / 37.7 mm = 5.305 µrad):

* lower (horizontal tilt): the beam turns by 2δ in the horizontal plane, so
  |ΔX| = 2Rδ = 2 × 5.332 mm × 5.305 µrad = 56.6 nm (exact 56.55 nm). The tilted sample adds
  a front–back part, ΔY/ΔX = sin θ sin β / (1 + sin θ cos β) = 1/3 at θ = β = 45°, i.e.
  18.9 nm (exact 18.81 nm).
* upper (vertical tilt): the 45° fold reduces the vertical turn to 2δ cos α, and the
  sample stretches a shift along the plane of incidence by 1/cos θ:
  ≈ 2Rδ cos α / cos θ = 56.6 nm at α = θ = 45°, within 3 % of the exact 55.00 nm.

The same check on the module's defaults (R = 16 mm, θ = 60°) gives |ΔX| = 2Rδ = 170 nm and
ΔY/ΔX = 0.38, i.e. the 0.18 µm lower step of its README's literature comparison.

**Caveat: R = 5.33 mm is an effective value.** A neaspec parabolic mirror's focal length is
more like 10–16 mm (16 mm is reported for another neaspec THz-TDS s-SNOM, the module
default). 55 nm ÷ 5.3 µrad = 10.4 mm: the estimate looks like (focal length × mirror
angle) without the factor 2 by which a reflection turns the beam (module README,
"Literature comparison"). If so, a ~10 mm mirror moves the spot ≈ 0.11 µm per step, twice
the calibrated value, and the X ↔ travel conversion below is off by that factor. R also
enters the diffraction model, so the spot-shape results (D auto, FWHM, peak vs travel)
correspond to this effective R. What changes with R and θ (`'trend'` spot at X = +5 µm
with the lower actuator alone):

| R, θ | D auto | aligned FWHM | step: lower / upper | X = +5 µm, lower only: travel, `rel` |
|---|---|---|---|---|
| 5.332 mm, 45° (default) | 4.4 mm | 1.560 × 1.113 µm | 59.6 / 55.0 nm | −17.136 µm, 0.52 |
| 10.4 mm, 45° | 8.5 mm | 1.562 × 1.116 µm | 116.3 / 107.3 nm | −8.991 µm, 0.54 |
| 16 mm, 45° | 13.1 mm | 1.562 × 1.115 µm | 178.9 / 165.0 nm | −5.871 µm, 0.54 |
| 16 mm, 60° (module) | 13.0 mm | 2.177 × 1.115 µm | 181.6 / 235.8 nm | −5.869 µm, 0.60 |

With D auto the aligned spot is set by NA and θ, not by R; the travel needed for a given X
scales with L/R, and the peak at a given X depends on R only weakly.

**Replace it with a camera calibration.** From the aligned position (the spot on the tip,
both actuators at the model's 0), step one actuator by ~50 µm (≈ 1728 counts), measure the
spot shift on the AFM top-view camera, and enter the travel **with its sign** and the shift
in the page's "Calibrate R from a measured step" control, or call
`steering.calibrate_R(sp, actuator, travel_um, spot_um)` and set `R` to the result. Measure
θ if possible: R is fitted for the θ in `sp`. A large step makes the 29 nm encoder
resolution negligible; reach the starting point moving in the direction of the step
(backlash < 8 µm). At such steps the response is asymmetric: at the default R, ±50 µm moves
the spot 13.42 / 17.52 µm (lower, extend / retract) and 15.96 / 13.05 µm (upper), so a
retract entered as +50 µm fits R = 6.838 (lower) or 4.028 mm (upper) instead of 5.332 mm.
Do not go much beyond ~50 µm: the shift stops growing with R once the step is large (100 µm
of the upper actuator at R = 5.3 mm sits at the minimum of shift against R, and no R fits).

**Saved settings.** GUI settings saved before the calibration (settings key
`steering/params_version` < 2) are migrated automatically, field by field: a saved R, θ or
D equal to its old default (16 mm, 60°, 13.0 mm) becomes the new default; a value that
had been customised, and every other saved value, is kept (`migrate_params` in
`ltemsim_gui/pages/steering.py`). D = 13 mm in use (D auto off) next to a customised R is
the user's beam and stays; D moves only while it is unused or together with R. The
migrated set is saved back once. A saved value that cannot be read (wrong type, NaN, out of
range) goes back to its default on its own, and the page names it above the form.

## From Python

    import numpy as np
    from ltemsim.steering import (SteeringParams, kdc101_counts, peak_curve, plan_offsets,
                                  spot, steer, travel_for_offset)

    sp = SteeringParams()                     # defaults; set bias_axis_deg from the mounting
    r = steer(sp, lower_um=4, upper_um=0)     # chief ray
    r.spot_um                                 # (-1.124, -0.359) lab µm: + port wall, + back
    r.X_um, r.Y_um                            # (-1.124, -0.359) simulation frame (φ = 0)
    r.per_um                                  # {'lower': (-0.2824, -0.0932), 'upper': (0.1419, -0.2360)}
    r.port_um, r.defocus_um, r.arrival_fs     # (-53.05, 0.0) µm, 0.382 µm, 1.28 fs

    s = spot(sp, 4, 0, "fine")                # None when no spot forms
    s["fwhmL"], s["fwhmS"], s["rel"]          # 1.55 µm, 1.13 µm, 0.962
    s["I"].shape, s["W"]                      # (72, 72) float32, 3.5 µm

    sol = travel_for_offset(sp, 5.0)          # both actuators, Y = 0
    sol["lower_um"], sol["upper_um"]          # -13.707, +6.320 µm (converged, residual 4e-8 µm)
    sol = travel_for_offset(sp, 5.0, actuators="lower")
    sol["lower_um"], sol["steering"].Y_um     # -17.136 µm, +1.960 µm (Y is not controlled)
    kdc101_counts(sol["lower_um"])            # -592.1 counts

    rows = plan_offsets(sp, [-7.5, -5, -2.5, 0, 2.5, 5, 7.5], actuators="lower")
    curve = peak_curve(sp, "upper", np.linspace(-10, 10, 41))

The aligned spot (`reference_spot(sp)`): peak 1.0000, FWHM 1.560 × 1.113 µm (aspect
1.40 ≈ 1/cos θ = 1.41), long axis at −45.2°, i.e. along the plane of incidence, area
1.35 µm², D = 4.4 mm (module defaults: FWHM 2.177 × 1.115 µm, aspect 1.95, D = 13.0 mm). Up
to ~10 µm of travel both actuators keep it alike (`rel` 0.962 lower, 0.961 upper at +4 µm;
0.798 and 0.780 at +10 µm); beyond that the upper one degrades it faster: at +20 µm `rel` =
0.396 for the upper (FWHM 2.89 × 1.24 µm, defocus 3.96 µm, window grown to W = 5.6 µm)
against 0.484 for the lower.

`plan_offsets` for the X list of the tip-position chain (`scripts/run_tip_positions.py`),
default parameters, `'trend'` spots:

| X (µm) | −7.5 | −5 | −2.5 | 0 | +2.5 | +5 | +7.5 |
|---|---|---|---|---|---|---|---|
| lower only: travel (µm) | +27.626 | +18.185 | +8.968 | 0 | −8.704 | −17.136 | −25.293 |
| KDC101 counts | +954.6 | +628.4 | +309.9 | 0 | −300.8 | −592.1 | −874.0 |
| Y it gives (µm) | −1.79 | −1.36 | −0.76 | 0 | +0.91 | +1.96 | +3.15 |
| `rel` | 0.36 | 0.53 | 0.83 | 1.00 | 0.84 | 0.52 | 0.28 |
| FWHM (µm) | 2.11 × 1.47 | 1.76 × 1.47 | 1.52 × 1.19 | 1.59 × 1.14 | 1.61 × 1.19 | 1.80 × 1.55 | 3.29 × 1.77 |
| both, Y = 0: lower / upper (µm) | +24.563 / −7.197 | +15.820 / −5.231 | +7.635 / −2.797 | 0 / 0 | −7.104 / +3.066 | −13.707 / +6.320 | −19.842 / +9.696 |
| both: `rel` | 0.37 | 0.53 | 0.83 | 1.00 | 0.84 | 0.53 | 0.31 |

With these defaults, holding Y = 0 costs no peak intensity (0.53 against 0.52–0.53 for the
lower actuator alone at ±5 µm), although the defocus grows (2.50 µm instead of 1.52–1.82
µm). With the module's defaults it did (0.56 instead of 0.60). `rel` of a `'trend'` spot is
relative to the aligned `'fine'` spot (1.0007 at travel 0); `peak_curve` normalises to the
same quality, so its travel-0 point reads 1 to ~1e-4.

Peak intensity `rel` vs travel (`peak_curve`, `'trend'`; ±30 µm covers the lower-only
travels of the X list above):

| travel (µm) | −30 | −20 | −10 | −5 | 0 | +5 | +10 | +20 | +30 |
|---|---|---|---|---|---|---|---|---|---|
| lower | 0.23 | 0.42 | 0.79 | 0.95 | 1.00 | 0.94 | 0.80 | 0.48 | 0.33 |
| upper | 0.28 | 0.44 | 0.79 | 0.94 | 1.00 | 0.94 | 0.78 | 0.39 | 0.22 |

## Performance

Single thread, default (calibrated) parameters, measured on the Apple M5 Pro of this
checkout (the module README gives 2.8 GHz Xeon numbers for its own defaults, ~2–4× slower):

| call | time |
|---|---|
| `steer` | 0.5 ms (four chief rays) |
| `spot` `'fine'` | 3.5 ms (either actuator, 2–20 µm) – 7 ms (upper 20 µm, large window) |
| `spot` `'fast'` | 1.6 – 3.3 ms |
| `spot` `'trend'` | 1.3 – 2.8 ms |
| `D_auto` (first call per parameter set, then cached by the core) | 26 ms |
| `peak_curve`, 41 points (±10 µm), `'trend'` | 57 ms (either actuator) |
| `plan_offsets`, 7 rows, lower only | 26 ms |

The slow end of each range is large upper travel: the defocused spot needs a larger
window. With the module's defaults (R = 16 mm, θ = 60°) the upper actuator defocuses about
five times more per µm (1.02 against 0.19 µm), so its spots are slower at the same travel
(`'fine'` 17 ms against 3.5 ms at upper 10 µm, `peak_curve` upper 110 ms against 57 ms).
In interactive code compute `'fast'` while a value is being dragged and `'fine'`
when idle, keep `'fine'` at large travel off the GUI thread (`run_async`), and use
`'trend'` for sweeps. The core caches the auto D and the aligned reference spot per
full parameter set; every other result, including `reference_spot`, is a fresh copy.

## GUI page (Simulate → Pump steering)

A left column with the two actuators (spin box ±500 µm, slider ±30 µm in 0.1 µm steps, step
buttons −1 / −0.2 / +0.2 / +1 µm and Zero, the per-µm spot shift, the KDC101 change in mm and
counts) above the `SteeringParams` form (basic fields first; the ±1 sides as choices; D is
greyed out while D auto is on and the resolved value is shown beside the checkbox) and the
"Calibrate R from a measured step" box (actuator, signed travel in µm from the aligned
position, measured spot shift in nm, "Fit R", which runs `calibrate_R` with the current
parameters and puts the result in the form; below it the spot shift of one 0.2 µm step of
each actuator), and five tabs. The slider, the peak curves, the shape gallery and the data
table span ±30 µm, which covers the lower-only travels of the X list in "Link to
simulation" (+27.6 / −25.3 µm for X = ∓7.5 µm).

* **Spot.** The beam at the entrance port (seen from outside the box, facing the port) and the
  spot on the sample (top view: the diffraction spot with its half-maximum contour, the aligned
  spot dashed, the trail, the per-µm arrows of both actuators, the plane of incidence and the
  bias axis φ). Readouts in words (left/right, front/back, along · across the plane of
  incidence, port shift, defocus, arrival time) and in the simulation frame (X, Y).
* **Spot shape.** The current spot magnified around its chief-ray hit with its metrics (FWHM,
  long axis vs the plane of incidence, half-max area, peak = `rel`, brightest point vs the
  chief-ray hit); peak intensity against travel for each actuator (`peak_curve`, 'trend'),
  i.e. how far the spot can be moved before it degrades; the spot shapes from −30 to +30 µm;
  a data table.
* **Setup.** 2-D schematics from the exact chief-ray trace, tilts drawn ×10 / ×100 / ×400:
  the whole path from above (box interior | port wall, steering mirror and the two Z812s), the
  box interior near the tip, a side view in the plane of incidence, and the KS1-Z8 seen from
  the actuator side with δ = travel / L and the KDC101 change.
* **3D view.** The module's own interactive page (`reference/ltem-steering.html`) in an
  embedded browser: the three.js model of the box, the beams and the mount. Its trail on
  the sample starts at the current travels when the page loads and is cleared by "Clear
  trail" with the scopes; it records each actuator separately, so a jump of both travels
  (Zero both, Move to a row) shows as an L-shaped corner.
* **Link to simulation.** The `plan_offsets` table: for each target X, the actuator travels
  and KDC101 counts, the exact landing point (X, Y) and the predicted spot, for the lower
  actuator alone, the upper alone or both (Y = 0); a plot of the landing points and of the
  predicted peak and FWHM against X (optionally with the measured tip scan); "Move actuators
  to selected row" and "Export CSV…". Rows that miss their X by more than 1e-4 µm or need a
  travel beyond the ±500 µm actuator range are marked; Move is disabled for the latter. While the
  X list does not parse, the table is empty. This is where the mirror meets the
  tip-position chain.

Colours follow the reference page: lower / horizontal amber, upper / vertical teal, pump
red, THz / collection violet.

## Limitations

* **3D view needs internet.** The reference page loads three.js 0.128.0 (and its
  OrbitControls) from cdn.jsdelivr.net. Offline the stage keeps the page's own fallback
  text and the tab says so; the other tabs compute locally. Without QtWebEngine the tab
  offers to open the page in the default browser (`ltemsim_gui/widgets/steering3d.py`).
* **3D spot size (display fix in the page).** As delivered, the page drew the spot glyph at a
  constant on-screen size times FWHM / (original short-axis FWHM), so a defocused spot filled the
  view at every zoom. It is now the half-maximum ellipse at the same display gain as the spot shift
  and the trail (µm × N / 1000 → mm), so it scales with zoom, with a 3-pixel floor so it stays
  visible when zoomed out. The labels of the few-mm region around the tip (Tip, Spot, +horiz, +vert,
  Front, Back) are shown together only while that region is resolved (at least 6 px per mm at the
  tip), so they disappear when zoomed out and come back when zoomed in; any label that would cover
  an earlier one is hidden. Only the 3-D drawing changed; the page's numbers are the same.
* **Not every parameter reaches the 3D page.** The page is driven
  through its own inputs, which exist only for the two travels and R, θ, β, L, α, dist,
  λ, NA and D (or D auto). Its sliders clamp and round to their range and step (R 2–30 mm
  in 0.01, θ 40–72°, β 0–65°, L 15–60 mm, α 15–70°, dist 80–600 mm in 10, λ 700–1600 nm in
  10, NA 0.25–0.6, D 3–24 mm in 0.5; travel ±500 µm in 0.01 µm); the tab lists every value
  the model shows clipped, rounded or not at all. `zLow`, `portX`, `foldY`, `yawSide`, `pitchSide` and
  `bias_axis_deg` have no input there: the 3-D model always draws zLow = 1 mm, the port at
  260 mm, the fold mirror 6 mm behind the tip and an upright mount, in the lab frame, with
  no bias axis. The numbers in the other tabs come from the adapter and use every
  parameter. The page's R slider was widened from 6–30 mm in 0.5 to 2–30 mm in 0.01 mm, so
  the calibrated default R = 5.332 mm is drawn as 5.33 mm; roundings below 0.1 %
  (`SNAP_REL`) are not listed.
* **The sample orientation is assumed** (φ = 0) until `bias_axis_deg` is set; the X of
  every row in "Link to simulation" depends on it.
* **2-D link.** The simulation takes X only. The Y that a single-actuator scan adds
  (+1.96 µm at X = +5 µm), the defocus and the change of spot size are reported, not fed
  into the 2-D model. `Params.spot_w_um` (1/e² radius, assumed 3 µm ↔ FWHM 3.5 µm) stays a
  separate input; the aligned spot predicted here is FWHM 1.56 × 1.11 µm (1/e² radii
  ≈ 1.32 × 0.95 µm) for an ideal TEM00 beam at θ = 45° (2.18 × 1.12 µm at the module's
  θ = 60°).
* **The calibration is a literature estimate.** R = 5.332 mm reproduces 55 nm per 0.2 µm
  step of the upper actuator at an assumed θ = 45°; it is about half a plausible real focal
  length, so every travel ↔ X conversion may be off by up to a factor ~2 until a camera
  calibration replaces it ("Calibration of the spot shift").
* **From the module** (details in its README): scalar diffraction without polarisation;
  ideal TEM00 input and mirror surfaces; an aperture fixed in space; best focus at travel
  (0, 0) assumed; no actuator backlash (Z812 < 8 µm: approach each target from the same
  side) or load effects; magnitudes ∝ R/L with R fitted to a literature step, not measured
  on this instrument.
