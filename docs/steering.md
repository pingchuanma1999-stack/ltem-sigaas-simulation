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
This page covers only the integration.

## Where things live

| path | content |
|---|---|
| `ltem-steering-module/` | the module as delivered: JS reference core, NumPy port, `tests/golden.json`, the interactive page `reference/ltem-steering.html` (three.js 3-D model, bilingual). The cores and `golden.json` are **unchanged**; do not edit them, and regenerate `golden.json` (`node tests/make_golden.mjs`) only for a deliberate physics change. The page has one display fix in its 3-D code (see "3D view" below). |
| `ltem-steering-module/python/ltem_optics.py` | the NumPy core used here (only dependency NumPy) |
| `ltemsim/steering.py` | the one adapter: parameters, actuator names, signs, frames, the offset solver. Every conversion between the core's conventions and ours is in this file and nowhere else. It loads the core under the private name `ltemsim._ltem_optics` (`steering.optics`), so the core's caches belong to this package. |
| GUI: Simulate → Pump steering | the desktop page (tabs below); form fields from `paramspec.steering_specs()`, 3D tab in `ltemsim_gui/widgets/steering3d.py` |
| `tests/test_steering.py` | golden values through the core and again through the adapter; actuator signs, mirrored mounts, aligned-spot shape, frame round trips, the offset solver, parameter handling, `plan_offsets` |
| `tests/test_gui_steering.py` | the GUI page (offscreen) |

## Conventions

### Actuators

| adapter (`ltemsim.steering`, GUI) | core | KS1-Z8 position, seen from the actuator side | tilt |
|---|---|---|---|
| `lower` (`lower_um`) | `yaw` | lower right (the pivot, the manual knob, is lower left) | horizontal |
| `upper` (`upper_um`) | `pitch` | upper left | vertical |

Response to +1 µm of travel with the default parameters (lab frame, below):

| actuator | spot shift (ΔX, ΔY) | size | direction | port shift | defocus |
|---|---|---|---|---|---|
| lower | (−0.848, −0.321) µm | 0.907 µm | left and toward the front, mostly across the plane of incidence | 13.3 µm toward the front | 0.32 µm |
| upper | (+0.737, −0.923) µm | 1.181 µm | right and toward the front, along the plane of incidence toward the parabolic mirror | 9.4 µm down | 1.02 µm |

* Travel is push-rod extension in µm. **+ = the rod extends = the KDC101 reading
  increases.** 1 mm = 34,555 encoder counts (`KDC101_COUNTS_PER_UM = 34.555`, ≈ 29 nm per
  count); `kdc101_counts(travel_um)` converts. The reading comes from the motor encoder,
  not from the rod tip.
* The mirror turns by δ = travel / L. The two directions are 108° apart on the sample,
  so either actuator alone also moves the spot across the bias axis.
* `yawSide` / `pitchSide` describe where the actuators sit, seen from the actuator side:
  `yawSide` = +1 / −1 puts the horizontal-tilt (lower) actuator right / left of the pivot,
  `pitchSide` = +1 / −1 the vertical-tilt (upper) actuator above / below it. Each −1 flips
  that actuator's shift. A mount turned upside down (rotated 180° about the mirror normal)
  needs **both** = −1; its "lower" (horizontal-tilt) actuator is then physically the upper
  one.
* Per-µm coefficients (`Steering.per_um`) are forward differences over +1 µm at the
  current geometry; the response is slightly nonlinear at large travel (X = +5 µm needs
  −5.869 µm of lower travel; the linear estimate is −5/0.848 = −5.894 µm).
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
  intensity is across the beam; per unit sample area it is smaller by ≈ cos θ = 0.5, a
  constant at fixed geometry.

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
| 0° (default) | (−0.848, −0.321) | (+0.737, −0.923) |
| 30° | (−0.895, +0.146) | (+0.177, −1.167) |
| 90° (+x toward the back) | (−0.321, +0.848) | (−0.923, −0.737) |
| 20.7° | (−0.907, 0) | (+0.362, −1.124) |
| −51.4° | (−0.278, −0.863) | (+1.181, 0) |

The last two rows are the mountings for which one actuator alone scans purely along the
bias axis; the same holds at φ + 180° with the travel sign reversed (−159.3° for the
lower, +128.6° for the upper actuator). For any other φ, a pure-X scan needs both
actuators (`actuators="both"`).

## Parameters

`SteeringParams` (dataclass in `ltemsim/steering.py`). `to_dict()` / `from_dict()` round
trip it; `from_dict` takes a partial dict, ignores unknown keys and coerces strings
(QSettings, JSON presets).

| field | default | unit | meaning | status |
|---|---|---|---|---|
| `R` | 16.0 | mm | effective focal length of the parabolic mirror (chief point → focus) | reported for another neaspec THz-TDS s-SNOM; **calibrate** |
| `theta` | 60.0 | deg | angle of incidence on the sample | neaspec standard |
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
| `D_auto` | True | – | use the input-beam diameter for the smallest spot (13.0 mm with the defaults) | the optics are aligned for the smallest spot |
| `D` | 13.0 | mm | 1/e² input-beam diameter, used when `D_auto` is off | **measure** before the AFM |
| `bias_axis_deg` | 0.0 | deg | azimuth of the simulation's +x in the lab frame | **assumed**; set from the sample mounting |

The directions do not depend on the uncertain values; the magnitudes scale with R/L; the
spot shape depends strongly on D and NA. To calibrate R/L, step 50 µm and measure the
shift with the AFM top-view camera (the module README compares the model with the
≈ 55 nm per 1.1″ quoted by Pizzuto et al.).

## From Python

    import numpy as np
    from ltemsim.steering import (SteeringParams, kdc101_counts, peak_curve, plan_offsets,
                                  spot, steer, travel_for_offset)

    sp = SteeringParams()                     # defaults; set bias_axis_deg from the mounting
    r = steer(sp, lower_um=4, upper_um=0)     # chief ray
    r.spot_um                                 # (-3.387, -1.272) lab µm: + port wall, + back
    r.X_um, r.Y_um                            # (-3.387, -1.272) simulation frame (φ = 0)
    r.per_um                                  # {'lower': (-0.848, -0.321), 'upper': (0.737, -0.923)}
    r.port_um, r.defocus_um, r.arrival_fs     # (-53.05, 0.0) µm, 1.295 µm, 4.32 fs

    s = spot(sp, 4, 0, "fine")                # None when no spot forms
    s["fwhmL"], s["fwhmS"], s["rel"]          # 2.18 µm, 1.27 µm, 0.783
    s["I"].shape, s["W"]                      # (72, 72) float32, 3.5 µm

    sol = travel_for_offset(sp, 5.0)          # both actuators, Y = 0
    sol["lower_um"], sol["upper_um"]          # -4.480, +1.588 µm (converged, residual 2e-11 µm)
    sol = travel_for_offset(sp, 5.0, actuators="lower")
    sol["lower_um"], sol["steering"].Y_um     # -5.869 µm, +1.930 µm (Y is not controlled)
    kdc101_counts(sol["lower_um"])            # -202.8 counts

    rows = plan_offsets(sp, [-7.5, -5, -2.5, 0, 2.5, 5, 7.5], actuators="lower")
    curve = peak_curve(sp, "upper", np.linspace(-10, 10, 41))

The aligned spot (`reference_spot(sp)`): peak 0.99999, FWHM 2.177 × 1.115 µm (aspect
1.95 ≈ 1/cos θ), long axis at −45.1°, i.e. along the plane of incidence, area 1.89 µm²,
D = 13.0 mm. The upper actuator degrades it much faster than the lower one: at +4 µm of
upper travel `rel` = 0.477 (FWHM 2.51 × 1.31 µm, window grown to W = 5.6 µm) against
0.783 for the lower.

`plan_offsets` for the X list of the tip-position chain (`scripts/run_tip_positions.py`),
default parameters, `'trend'` spots:

| X (µm) | −7.5 | −5 | −2.5 | 0 | +2.5 | +5 | +7.5 |
|---|---|---|---|---|---|---|---|
| lower only: travel (µm) | +8.885 | +5.912 | +2.951 | 0 | −2.940 | −5.869 | −8.787 |
| KDC101 counts | +307.0 | +204.3 | +102.0 | 0 | −101.6 | −202.8 | −303.6 |
| Y it gives (µm) | −2.78 | −1.87 | −0.94 | 0 | +0.96 | +1.93 | +2.92 |
| `rel` | 0.41 | 0.60 | 0.87 | 1.00 | 0.87 | 0.60 | 0.40 |
| FWHM (µm) | 2.73 × 1.94 | 2.22 × 1.54 | 2.18 × 1.19 | 2.20 × 1.13 | 2.19 × 1.19 | 2.22 × 1.55 | 2.83 × 2.01 |
| both, Y = 0: lower / upper (µm) | | +4.562 / −1.575 | | 0 / 0 | | −4.480 / +1.588 | |
| both: `rel` | | 0.56 | | 1.00 | | 0.56 | |

Holding Y = 0 costs peak intensity (0.56 instead of 0.60 at ±5 µm) because the upper
actuator defocuses (3.06 µm instead of 1.9 µm). `rel` of a `'trend'` spot is relative to
the aligned `'fine'` spot (0.9998 at travel 0); `peak_curve` normalises to the same
quality, so its travel-0 point reads 1 to ~1e-4.

Peak intensity `rel` vs travel (`peak_curve`, `'trend'`):

| travel (µm) | −10 | −7.5 | −5 | −2.5 | 0 | +2.5 | +5 | +7.5 | +10 |
|---|---|---|---|---|---|---|---|---|---|
| lower | 0.35 | 0.47 | 0.69 | 0.91 | 1.00 | 0.91 | 0.69 | 0.48 | 0.37 |
| upper | 0.10 | 0.18 | 0.33 | 0.75 | 1.00 | 0.75 | 0.33 | 0.17 | 0.09 |

## Performance

Single thread, measured on the Apple M5 Pro of this checkout (the module README gives
2.8 GHz Xeon numbers, ~2–4× slower):

| call | time |
|---|---|
| `steer` | 0.5 ms (four chief rays) |
| `spot` `'fine'` | 2.7 ms (lower 2 µm) – 13 ms (upper 10 µm, large window) |
| `spot` `'fast'` | 1.5 – 8 ms |
| `spot` `'trend'` | 1.2 – 7 ms |
| `D_auto` (first call per parameter set, then cached by the core) | 29 ms |
| `peak_curve`, 41 points, `'trend'` | 55 ms (lower), 100 ms (upper) |
| `plan_offsets`, 7 rows, lower only | 26 ms |

The slow end of each range is large upper travel: the defocused spot needs a larger
window. In interactive code compute `'fast'` while a value is being dragged and `'fine'`
when idle, keep `'fine'` at large travel off the GUI thread (`run_async`), and use
`'trend'` for sweeps. The core caches the auto D and the aligned reference spot per
full parameter set; every other result, including `reference_spot`, is a fresh copy.

## GUI page (Simulate → Pump steering)

A left column with the two actuators (spin box ±500 µm, slider ±20 µm in 0.1 µm steps, step
buttons −1 / −0.2 / +0.2 / +1 µm and Zero, the per-µm spot shift, the KDC101 change in mm and
counts) above the `SteeringParams` form (basic fields first; the ±1 sides as choices; D is
greyed out while D auto is on and the resolved value is shown beside the checkbox), and five
tabs.

* **Spot.** The beam at the entrance port (seen from outside the box, facing the port) and the
  spot on the sample (top view: the diffraction spot with its half-maximum contour, the aligned
  spot dashed, the trail, the per-µm arrows of both actuators, the plane of incidence and the
  bias axis φ). Readouts in words (left/right, front/back, along · across the plane of
  incidence, port shift, defocus, arrival time) and in the simulation frame (X, Y).
* **Spot shape.** The current spot magnified around its chief-ray hit with its metrics (FWHM,
  long axis vs the plane of incidence, half-max area, peak = `rel`, brightest point vs the
  chief-ray hit); peak intensity against travel for each actuator (`peak_curve`, 'trend'),
  i.e. how far the spot can be moved before it degrades; the spot shapes from −20 to +20 µm;
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
  visible when zoomed out. Labels that would cover an earlier label are hidden (e.g. Front/Back in
  the top view when zoomed out). Only the 3-D drawing changed; the page's numbers are the same.
* **Not every parameter reaches the 3D page.** The page is driven
  through its own inputs, which exist only for the two travels and R, θ, β, L, α, dist,
  λ, NA and D (or D auto). Its sliders clamp and round to their range and step (R 6–30 mm
  in 0.5, θ 40–72°, β 0–65°, L 15–60 mm, α 15–70°, dist 80–600 mm in 10, λ 700–1600 nm in
  10, NA 0.25–0.6, D 3–24 mm in 0.5; travel ±500 µm in 0.01 µm); the tab lists every value
  the model shows clipped, rounded or not at all. `zLow`, `portX`, `foldY`, `yawSide`, `pitchSide` and
  `bias_axis_deg` have no input there: the 3-D model always draws zLow = 1 mm, the port at
  260 mm, the fold mirror 6 mm behind the tip and an upright mount, in the lab frame, with
  no bias axis. The numbers in the other tabs come from the adapter and use every
  parameter.
* **The sample orientation is assumed** (φ = 0) until `bias_axis_deg` is set; the X of
  every row in "Link to simulation" depends on it.
* **2-D link.** The simulation takes X only. The Y that a single-actuator scan adds
  (+1.93 µm at X = +5 µm), the defocus and the change of spot size are reported, not fed
  into the 2-D model. `Params.spot_w_um` (1/e² radius, assumed 3 µm ↔ FWHM 3.5 µm) stays a
  separate input; the aligned spot predicted here is FWHM 2.18 × 1.12 µm (1/e² radii
  ≈ 1.85 × 0.95 µm) for an ideal TEM00 beam.
* **From the module** (details in its README): scalar diffraction without polarisation;
  ideal TEM00 input and mirror surfaces; an aperture fixed in space; best focus at travel
  (0, 0) assumed; no actuator backlash (Z812 < 8 µm: approach each target from the same
  side) or load effects; magnitudes ∝ R/L with R not measured on this instrument.
