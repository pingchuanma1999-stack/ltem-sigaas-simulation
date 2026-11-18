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
own defaults (R = 16 mm, θ = 60°, D = 13.0 mm, the nominal mirror tilt travel / L); this
package keeps R = 16 mm but assumes θ = 45° and scales the mirror tilt by a calibrated
factor κ = `tilt_scale` = 0.33336 (see "Calibration of the spot shift"), so its numbers
differ, and `steering.module_defaults()` gives the module's set (κ = 1) as `SteeringParams`.

## Where things live

| path | content |
|---|---|
| `ltem-steering-module/` | the module as delivered: JS reference core, NumPy port, `tests/golden.json`, the interactive page `reference/ltem-steering.html` (three.js 3-D model, bilingual). The cores and `golden.json` are **unchanged**; do not edit them, and regenerate `golden.json` (`node tests/make_golden.mjs`) only for a deliberate physics change. The page has a display fix in its 3-D code and a widened R slider (`#pR` 2–30 mm in 0.01, was 6–30 in 0.5), made for an earlier calibration that fitted R (see "Limitations" below). |
| `ltem-steering-module/python/ltem_optics.py` | the NumPy core used here (only dependency NumPy) |
| `ltemsim/steering.py` | the one adapter: parameters (defaults with the calibrated tilt scale; `CALIBRATION`, `calibrate_tilt_scale`, `module_defaults`), actuator names, signs, frames, the tilt scale (`core_travel`: the core is handed κ × travel), the offset solver. Every conversion between the core's conventions and ours is in this file and nowhere else. It loads the core under the private name `ltemsim._ltem_optics` (`steering.optics`), so the core's caches belong to this package. |
| GUI: Simulate → Pump steering | the desktop page (tabs below); form fields from `paramspec.steering_specs()`, 3D tab in `ltemsim_gui/widgets/steering3d.py` |
| `tests/test_steering.py` | golden values through the core and again through the adapter (also with a tilt scale); actuator signs, mirrored mounts, aligned-spot shape, the calibration (defaults, `calibrate_tilt_scale`, κ in every call that traces), frame round trips, the offset solver, parameter handling, `plan_offsets`, the "From Python" block below (run as written) |
| `tests/test_gui_steering.py` | the GUI page (offscreen) |

## Conventions

### Actuators

| adapter (`ltemsim.steering`, GUI) | core | KS1-Z8 position, seen from the actuator side | tilt |
|---|---|---|---|
| `lower` (`lower_um`) | `yaw` | lower right (the pivot, the manual knob, is lower left) | horizontal |
| `upper` (`upper_um`) | `pitch` | upper left | vertical |

Response to +1 µm of travel with the default parameters (calibrated κ; lab frame, below):

| actuator | spot shift (ΔX, ΔY) | size | direction | port shift | defocus |
|---|---|---|---|---|---|
| lower | (−0.2828, −0.0939) µm | 0.2980 µm | left and toward the front, mostly across the plane of incidence | 4.42 µm toward the front | 0.094 µm |
| upper | (+0.1416, −0.2359) µm | 0.2752 µm | right and toward the front, along the plane of incidence toward the parabolic mirror | 3.13 µm down | 0.189 µm |

One 0.2 µm step (nominally 1.1″; κ × 1.1″ ≈ 0.36″ in the model) moves the spot 59.6 nm
with the lower and 55.0 nm with the upper actuator; one KDC101 count 8.6 nm and 8.0 nm. With the module's own defaults (R = 16 mm,
θ = 60°, κ = 1) the same table reads (−0.848, −0.321) and (+0.737, −0.923) µm per µm, as
in the module README.

* Travel is push-rod extension in µm. **+ = the rod extends = the KDC101 reading
  increases.** 1 mm = 34,555 encoder counts (`KDC101_COUNTS_PER_UM = 34.555`, ≈ 29 nm per
  count); `kdc101_counts(travel_um)` converts. The reading comes from the motor encoder,
  not from the rod tip. Every travel in the adapter's API, in the GUI and in the KDC101
  counts is this real travel.
* The nominal KS1-Z8 lever turns the mirror by δ = travel / L. The core is handed the
  effective travel κ · travel (`core_travel(sp, travel)`, κ = `tilt_scale`), so the mirror
  turns by κ · travel / L (see "Calibration of the spot shift"). The two directions are
  103° apart on the sample (108° with the module's θ = 60°), so either actuator alone also
  moves the spot across the bias axis.
* `yawSide` / `pitchSide` describe where the actuators sit, seen from the actuator side:
  `yawSide` = +1 / −1 puts the horizontal-tilt (lower) actuator right / left of the pivot,
  `pitchSide` = +1 / −1 the vertical-tilt (upper) actuator above / below it. Each −1 flips
  that actuator's shift. A mount turned upside down (rotated 180° about the mirror normal)
  needs **both** = −1; its "lower" (horizontal-tilt) actuator is then physically the upper
  one.
* Per-µm coefficients (`Steering.per_um`, `port_per_um`) are forward differences from the
  aligned position over +1 µm of the core's (effective) travel, i.e. over 1/κ = 3.0 µm of
  real travel, given per µm of real travel; the response is slightly nonlinear at large
  travel (X = +5 µm needs −17.613 µm of lower travel; the linear estimate is
  −5/0.2828 = −17.68 µm).
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
| tilt scale `tilt_scale` (κ) | – (fraction of the nominal tilt 1/L) |
| wavelength `lam` | nm |
| actuator travel (the real push-rod travel; the core gets κ × travel), spot shift, port shift, `along`, `across`, defocus, FWHM, window half-width `W` | µm |
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
| 0° (default) | (−0.2828, −0.0939) | (+0.1416, −0.2359) |
| 30° | (−0.2919, +0.0600) | (+0.0047, −0.2751) |
| 90° (+x toward the back) | (−0.0939, +0.2828) | (−0.2359, −0.1416) |
| 18.4° | (−0.2980, 0) | (+0.0600, −0.2685) |
| −59.0° | (−0.0650, −0.2908) | (+0.2752, 0) |

The last two rows are the mountings for which one actuator alone scans purely along the
bias axis; the same holds at φ + 180° with the travel sign reversed (−161.6° for the
lower, +121.0° for the upper actuator). For any other φ, a pure-X scan needs both
actuators (`actuators="both"`). These angles depend on θ (and β), hardly on R and not at
all on κ: with the module's θ = 60° they are 20.7° / −159.3° (lower) and −51.4° / +128.6°
(upper).

## Parameters

`SteeringParams` (dataclass in `ltemsim/steering.py`). `to_dict()` / `from_dict()` round
trip it; `from_dict` takes a partial dict, ignores unknown keys and coerces strings
(QSettings, JSON presets).

| field | default | unit | meaning | status |
|---|---|---|---|---|
| `R` | 16.0 | mm | effective focal length of the parabolic mirror (chief point → focus). Geometry: it places the parabolic mirror and the fold mirrors and sets the aperture of the diffraction model | reported for another neaspec THz-TDS s-SNOM (module default). Not fitted: the spot-shift calibration is `tilt_scale` |
| `theta` | 45.0 | deg | angle of incidence on the sample | **assumed** for the calibration; neaspec standard and module default 60 |
| `beta` | 45.0 | deg | azimuth of the plane of incidence from the front–back axis toward the port side (GUI: 0–89°; β < 0 would put the pump on the THz collection half) | user's sketch (40–50°). Leaves the lower actuator's left–right (lab X) shift unchanged, but changes its front–back shift, the upper actuator's shift in both directions and the plane of incidence (along/across split, defocus): it matters for the upper actuator, for both, and for any φ ≠ 0 |
| `NA` | 0.42 | – | illumination aperture: cone of half-angle asin(NA) around the aligned chief ray | neaspec patent example; a published neaspec IR mirror has 0.46 |
| `zLow` | 1.0 | mm | height of the parabolic mirror's lower edge above the sample | neaspec patent |
| `alpha` | 45.0 | deg | angle of incidence on the steering mirror (90° fold) | setup photo |
| `dist` | 250.0 | mm | steering mirror → entrance-port plane | setup photo; sets the port shift, ~1 % on the spot |
| `L` | 37.7 | mm | lever arm, actuator contact → pivot | Thorlabs (1.1″ per 0.2 µm step) |
| `tilt_scale` | 0.33336 | – | κ: effective mirror tilt per µm of travel as a fraction of the nominal 1/L; the core is handed κ × travel (GUI: "Tilt calibration κ", 0.001–10) | **calibrated** from the 55 nm per 0.2 µm step of Pizzuto et al. (upper actuator, θ = 45°, R = 16 mm; see "Calibration of the spot shift"); 1 = the nominal lever (module). Replace it with a camera calibration |
| `yawSide` | +1 | ±1 | horizontal-tilt (lower) actuator right (+1) / left (−1) of the pivot, seen from the actuator side | KS1-Z8 drawing, setup photo |
| `pitchSide` | +1 | ±1 | vertical-tilt (upper) actuator above (+1) / below (−1) the pivot; an upside-down mount (180° about the mirror normal) needs `yawSide` = `pitchSide` = −1 | KS1-Z8 drawing, setup photo |
| `portX` | 260.0 | mm | x of the entrance-port plane | model geometry, ~1 % effect |
| `foldY` | 6.0 | mm | y of the fold-mirror centre behind the tip | model geometry, negligible |
| `lam` | 820.0 | nm | pump wavelength | as `Params.wavelength_nm` |
| `D_auto` | True | – | use the input-beam diameter for the smallest spot (13.1 mm with the defaults; 13.0 mm with the module's θ = 60°) | the optics are aligned for the smallest spot |
| `D` | 13.1 | mm | 1/e² input-beam diameter, used when `D_auto` is off | **measure** before the AFM |
| `bias_axis_deg` | 0.0 | deg | azimuth of the simulation's +x in the lab frame | **assumed**; set from the sample mounting |

The directions depend on θ, β and α, hardly on R or L (≤ 0.3° for R = 5.3–25 mm and
L = 20–60 mm) and not at all on κ; the magnitudes scale with κR/L; the spot shape depends
strongly on D and NA. κ is the parameter calibrated against a measured step (next
section).

## Calibration of the spot shift

**The data.** One 0.2 µm step of a Z812 (0.0002 mm on the KDC101, the KS1-Z8's minimum
repeatable step of 1.1″) moves the pump spot by ≈ 55 nm, the estimate of Pizzuto et al.
(both papers; module README, "Sources"). It is quoted for the pitch direction of the
mirror, so it is taken as the **upper** (vertical-tilt) actuator. The angle of incidence
on the sample inside the AFM is **assumed** to be 45° (θ = 45°; the neaspec standard and
the module's default is 60°), and R = 16 mm is the focal length reported for another
neaspec THz-TDS s-SNOM (the module default). The data, the assumption and the fitted
parameter are recorded in `steering.CALIBRATION`:

    {"actuator": "upper", "travel_um": 0.2, "spot_um": 0.055, "theta": 45.0,
     "fits": "tilt_scale", "source": ...}

**What a measured step constrains.** The KS1-Z8 lever (L = 37.7 mm; Thorlabs: 1.1″ ≈
5.3 µrad per 0.2 µm step) tilts the mirror by δ = travel / L, 5.305 µrad ≈ 1.09″ per step;
the reflected beam turns by ≈ 2δ, and the parabolic mirror turns that angle into a shift
≈ 2δR at the focus. A measured step fixes one product only: travel → spot ∝ κR/L, where κ
scales the tilt that reaches the parabolic mirror. It cannot tell κ from R.

**Why the fitted parameter is the tilt scale, not R.** An earlier version put the whole
step into R and got R = 5.332 mm (with κ = 1). But R is geometry, and the model uses it as
such:

* The 3-D model places the pump fold mirror and the THz collection mirror (10 mm wide,
  6 mm behind the tip) at x = ±R sin θ sin β: at R = 5.33 mm that is ±2.7 mm, and the two
  mirrors crossed each other; at R = 16 mm they sit at ±8.0 mm.
* The parabolic mirror's chief point lies R sin θ from the tip and R cos θ above the
  sample: 3.8 mm and 3.8 mm at R = 5.33 mm, just above the tip (its lower edge stays at
  zLow = 1 mm), instead of 11.3 mm and 11.3 mm.
* The diffraction pupil scales with R: the NA cone at distance R is ≈ 2R tan(asin NA) =
  4.9 mm across instead of 14.8 mm, and the best input beam (D auto) came out 4.4 mm
  instead of 13.1 mm, so a measured beam diameter would be held against the wrong
  aperture. For a given spot shift the field angle (shift / R) is three times larger,
  which bends the response more and dims the spot more (table below).

So R stays a physical length, and the step calibrates a separate, dimensionless
`tilt_scale` κ: the effective mirror tilt is κ · travel / L. The adapter hands the core
`core_travel(sp, t)` = κ · t everywhere (`steer`, `spot`, `peak_curve`, which reports the
real travel back, and the offset solver). Every travel in the adapter's API, the per-µm
coefficients (`Steering.per_um`, `port_per_um`: per µm of real travel) and the KDC101
counts stay the real push-rod travel.

`steering.calibrate_tilt_scale(sp, actuator, travel_um, spot_um)` returns the κ for which
`travel_um` of one actuator, from the aligned position (both at 0), moves the spot by
`spot_um` (exact trace; every other parameter of `sp`, R included, is kept). The travel is
signed. It solves for the effective travel t = κ · travel at the nominal tilt: the shift
grows monotonically with |t| from 0 until the chief ray leaves the parabolic mirror, so the
first crossing is bracketed from t = 0 (the bracket grows × 1.5, `CALIB_GROW`) and refined
by Brent's method; κ = t / |travel|. A trial t that takes the chief ray off the mirror ends
the bracket at the mirror edge, bisected between the last t that hit the mirror and that
one, so a shift reachable just before the edge still fits. `ValueError` for a bad input
(unknown actuator, zero travel, a shift that is not positive), or when the actuator cannot
move the spot that far before the chief ray leaves the mirror:

    from ltemsim.steering import SteeringParams, calibrate_tilt_scale, module_defaults
    calibrate_tilt_scale(SteeringParams(), "upper", 0.2, 0.055)   # 0.33336  (θ = 45°: the default κ)
    calibrate_tilt_scale(module_defaults(), "upper", 0.2, 0.055)  # 0.23332  if θ were 60°
    calibrate_tilt_scale(SteeringParams(), "lower", 0.2, 0.055)   # 0.30738  if the 55 nm were the horizontal step

The fitted κ belongs to the assumed θ, to the R in `sp` and to the choice of actuator: the
product κR matters, not κ alone (κR = 5.334 mm here, practically the R of the earlier fit).

**The result:** κ = 0.33336, the new default (R = 16 mm, θ = 45°, D auto = 13.1 mm). Per
0.2 µm step, exact trace, lab frame:

| actuator | spot shift (ΔX, ΔY) per 0.2 µm | size | per µm of travel | per KDC101 count |
|---|---|---|---|---|
| upper (pitch, calibrated) | (+28.30, −47.16) nm | 55.00 nm | (+0.1416, −0.2359) µm, 0.2752 µm | 8.0 nm |
| lower (yaw, derived) | (−56.59, −18.86) nm | 59.65 nm | (−0.2828, −0.0939) µm, 0.2980 µm | 8.6 nm |

So the horizontal (lower) actuator moves the spot **59.65 nm per step: 56.6 nm to the left
plus 18.9 nm toward the front** (per µm of travel: 0.283 µm left, 0.094 µm toward the
front). The per-µm values are forward differences over 1 µm of effective travel (3.0 µm of
real travel), the per-count values are per µm ÷ 34.555.

**Small-angle cross-check** (δ = 0.2 µm / 37.7 mm = 5.305 µrad, κδ = 1.768 µrad):

* lower (horizontal tilt): the beam turns by 2κδ in the horizontal plane, so
  |ΔX| = 2Rκδ = 2 × 16 mm × 1.768 µrad = 56.6 nm (exact 56.59 nm). The tilted sample adds
  a front–back part, ΔY/ΔX = sin θ sin β / (1 + sin θ cos β) = 1/3 at θ = β = 45°, i.e.
  18.9 nm (exact 18.86 nm).
* upper (vertical tilt): the 45° fold reduces the vertical turn to 2κδ cos α, and the
  sample stretches a shift along the plane of incidence by 1/cos θ:
  ≈ 2Rκδ cos α / cos θ = 56.6 nm at α = θ = 45°, within 3 % of the exact 55.00 nm.

The same check on the module's defaults (R = 16 mm, θ = 60°, κ = 1) gives |ΔX| = 2Rδ =
170 nm and ΔY/ΔX = 0.38, i.e. the 0.18 µm lower step of its README's literature
comparison.

**What κ = 0.33336 means.** The spot moves a third of what the nominal KS1-Z8 tilt and
R = 16 mm predict: with κ = 1 one 0.2 µm step would move it 178.9 nm (lower) and 165.0 nm
(upper). Possible reasons, not yet told apart:

* The 55 nm is a literature estimate, not a measurement on this instrument, and it may
  leave out the factor 2 by which a reflection turns the beam: 55 nm ÷ 5.305 µrad =
  10.4 mm looks like (focal length × mirror angle) (module README, "Literature
  comparison"). With the factor 2 a ~10 mm mirror moves the spot ≈ 0.11 µm per step, which
  gives κ = 0.667 at R = 16 mm (`calibrate_tilt_scale(sp, "upper", 0.2, 0.110)`); the
  remaining factor would then be a focal length of ≈ 10.7 mm instead of 16 mm, which κ
  absorbs as well (only κR enters the step).
* An angular demagnification between the steering mirror and the AFM port (relay optics
  that widen the beam narrow its angle by the same factor) would reduce the tilt that
  reaches the parabolic mirror in exactly the same way.

Until a camera calibration settles it, every travel ↔ X conversion carries this
uncertainty: with κ = 1, X = +5 µm would need −5.871 µm of lower travel instead of
−17.613 µm. Only the spot is calibrated: the port shift is scaled by κ as well (4.42 µm per
µm of lower travel, 13.26 µm with κ = 1), which holds if the beam at the port really turns
less; if κ only corrects the 55 nm estimate, the real port shift is the κ = 1 value.

**What κ and R each change** (`'trend'` spot at X = +5 µm with the lower actuator alone):

| R, θ, κ | D auto | aligned FWHM | step: lower / upper | X = +5 µm, lower only: travel, Y, `rel` |
|---|---|---|---|---|
| 16 mm, 45°, 0.33336 (default) | 13.1 mm | 1.562 × 1.115 µm | 59.6 / 55.0 nm | −17.613 µm, +1.70 µm, 0.54 |
| 5.332 mm, 45°, 1 (the earlier R fit) | 4.4 mm | 1.560 × 1.113 µm | 59.6 / 55.0 nm | −17.136 µm, +1.96 µm, 0.52 |
| 16 mm, 45°, 1 (nominal tilt) | 13.1 mm | 1.562 × 1.115 µm | 178.9 / 165.0 nm | −5.871 µm, +1.70 µm, 0.54 |
| 16 mm, 60°, 1 (module) | 13.0 mm | 2.177 × 1.115 µm | 181.6 / 235.8 nm | −5.869 µm, +1.93 µm, 0.60 |

κ rescales the travel axis and nothing else: at R = 16 mm, θ = 45° the spot at X = +5 µm
(its Y, peak and shape) is the same for κ = 1 and κ = 0.33336, reached with 1/κ = 3.0 times
the travel. R changes the optics: the same step put into R = 5.332 mm bends the response
more (−17.136 instead of −17.613 µm), moves the spot farther off the bias axis (+1.96
instead of +1.70 µm) and dims it more (0.52 instead of 0.54). With D auto the aligned spot
is set by NA and θ, not by R or κ.

**Replace it with a camera calibration.** From the aligned position (the spot on the tip,
both actuators at the model's 0), step one actuator by ~50 µm (≈ 1728 counts), measure the
spot shift on the AFM top-view camera (≈ 14–15 µm at the default κ), and enter the travel
**with its sign** and the shift in the page's "Calibrate the spot shift from a measured
step" box ("Fit κ"), or call `steering.calibrate_tilt_scale(sp, actuator, travel_um,
spot_um)` and set `tilt_scale` to the result. Measure θ if possible: κ is fitted for the θ
(and the R) in `sp`. A large step makes the 29 nm encoder resolution negligible; reach the
starting point moving in the direction of the step (backlash < 8 µm). At such steps the
response is slightly asymmetric: at the defaults, ±50 µm moves the spot 14.70 / 15.14 µm
(lower, extend / retract) and 13.90 / 13.62 µm (upper), so a retract entered as +50 µm fits
κ = 0.3436 (lower) or 0.3268 (upper) instead of 0.3334. Larger steps fit as well (R is
kept, and at the defaults the chief ray leaves the parabolic mirror only beyond +1583 µm of
lower or +1790 / −2676 µm of upper travel, outside the ±500 µm range), but the spot grows
dim and wide (`rel` 0.12–0.15, FWHM 5–7 µm at ±50 µm), so its centre is harder to locate.

**Saved settings.** GUI settings carry a version (settings key `steering/params_version`,
now 3) and are migrated automatically, field by field, when the page loads
(`migrate_params` in `ltemsim_gui/pages/steering.py`). A value at an old default takes
the new one; a value that had been customised, and every other saved value, is kept:

* version 2 (R fitted to the step): R = 5.332 mm becomes 16 mm and D = 4.4 mm becomes
  13.1 mm; θ = 45° is still the default;
* before version 2 (the module's defaults): R = 16 mm stays, θ = 60° becomes 45° and
  D = 13.0 mm becomes 13.1 mm;
* `tilt_scale` is new in version 3: a set saved before has none and takes the default
  κ = 0.33336;
* any other R saved by version 2 came from the old "Fit R" (R was version 2's only
  spot-step calibration, at the nominal tilt κ = 1). Keeping it next to the default κ would
  calibrate the step twice (a 78 nm step fitted as R = 6.977 mm would read 26 nm) and keep
  the fold mirrors crossed (R < 10 mm at θ = β = 45°). It is converted instead
  (`convert_fitted_R`): R becomes 16 mm and κ is fitted so that one 0.2 µm step of the
  upper actuator moves the spot as before (the lower step then agrees to < 0.1 nm; κ comes
  out near R / 16 mm, e.g. 0.436 for R = 6.977 mm). Should that fail, R is kept with κ = 1,
  exactly the version-2 model. The page says so once, above the form;
* a customised R saved before version 2 was a physical choice: it stays, with the default κ.

A D in use (D auto off) next to a customised R is the user's beam and stays; D moves only
while it is unused or together with R. The migrated set is saved back once. A saved value
that cannot be read (wrong type, NaN, out of range) goes back to its default on its own,
and the page names it above the form. A JSON preset carries no version: one loaded without
`tilt_scale` (saved before version 3) is migrated as a version-2 set by the form's
Load…, with the same note when an R is converted.

## From Python

    import numpy as np
    from dataclasses import replace
    from ltemsim.steering import (SteeringParams, calibrate_tilt_scale, core_travel, kdc101_counts,
                                  peak_curve, plan_offsets, spot, steer, travel_for_offset)

    sp = SteeringParams()                     # defaults; set bias_axis_deg from the mounting
    r = steer(sp, lower_um=4, upper_um=0)     # chief ray; travels are real push-rod travels
    r.spot_um                                 # (-1.131, -0.375) lab µm: + port wall, + back
    r.X_um, r.Y_um                            # (-1.131, -0.375) simulation frame (φ = 0)
    r.per_um                                  # {'lower': (-0.2828, -0.0939), 'upper': (0.1416, -0.2359)}
    r.port_um, r.defocus_um, r.arrival_fs     # (-17.68, 0.0) µm, 0.378 µm, 1.26 fs

    s = spot(sp, 4, 0, "fine")                # None when no spot forms
    s["fwhmL"], s["fwhmS"], s["rel"]          # 1.56 µm, 1.13 µm, 0.966
    s["I"].shape, s["W"]                      # (72, 72) float32, 3.5 µm

    sol = travel_for_offset(sp, 5.0)          # both actuators, Y = 0
    sol["lower_um"], sol["upper_um"]          # -14.610, +5.948 µm (converged, residual 6e-12 µm)
    sol = travel_for_offset(sp, 5.0, actuators="lower")
    sol["lower_um"], sol["steering"].Y_um     # -17.613 µm, +1.700 µm (Y is not controlled)
    kdc101_counts(sol["lower_um"])            # -608.6 counts

    k = calibrate_tilt_scale(sp, "upper", 0.2, 0.055)   # 0.33336: κ from a measured step (travel, shift in µm)
    sp = replace(sp, tilt_scale=k)
    core_travel(sp, 10.0)                     # 3.3336 µm: the travel the core is handed for 10 µm

    rows = plan_offsets(sp, [-7.5, -5, -2.5, 0, 2.5, 5, 7.5], actuators="lower")
    curve = peak_curve(sp, "upper", np.linspace(-10, 10, 41))    # travel_um: the real travel

The aligned spot (`reference_spot(sp)`): peak 1.0000, FWHM 1.562 × 1.115 µm (aspect
1.40 ≈ 1/cos θ = 1.41), long axis at −45.2°, i.e. along the plane of incidence, area
1.36 µm², D = 13.1 mm (module defaults: FWHM 2.177 × 1.115 µm, aspect 1.95, D = 13.0 mm). Up
to ~10 µm of travel both actuators keep it alike (`rel` 0.966 lower, 0.963 upper at +4 µm;
0.810 and 0.792 at +10 µm); beyond that the upper one degrades it faster: at +20 µm `rel` =
0.426 for the upper (FWHM 2.82 × 1.21 µm, defocus 3.79 µm, window grown to W = 5.6 µm)
against 0.477 for the lower.

`plan_offsets` for the X list of the tip-position chain (`scripts/run_tip_positions.py`),
default parameters, `'trend'` spots:

| X (µm) | −7.5 | −5 | −2.5 | 0 | +2.5 | +5 | +7.5 |
|---|---|---|---|---|---|---|---|
| lower only: travel (µm) | +26.634 | +17.728 | +8.850 | 0 | −8.821 | −17.613 | −26.376 |
| KDC101 counts | +920.4 | +612.6 | +305.8 | 0 | −304.8 | −608.6 | −911.4 |
| Y it gives (µm) | −2.42 | −1.63 | −0.82 | 0 | +0.84 | +1.70 | +2.58 |
| `rel` | 0.35 | 0.54 | 0.85 | 1.00 | 0.85 | 0.54 | 0.34 |
| FWHM (µm) | 2.79 × 1.52 | 1.79 × 1.50 | 1.57 × 1.21 | 1.60 × 1.14 | 1.58 × 1.21 | 1.77 × 1.52 | 2.96 × 1.55 |
| both, Y = 0: lower / upper (µm) | +22.351 / −8.696 | +14.842 / −5.829 | +7.392 / −2.930 | 0 / 0 | −7.334 / +2.960 | −14.610 / +5.948 | −21.828 / +8.964 |
| both: `rel` | 0.36 | 0.54 | 0.85 | 1.00 | 0.85 | 0.54 | 0.35 |

With these defaults, holding Y = 0 costs no peak intensity (0.543 against 0.542–0.543
for the lower actuator alone at ±5 µm), although the defocus grows (2.50 µm instead of 1.65–1.68
µm). With the module's defaults it did (0.56 instead of 0.60). `rel` of a `'trend'` spot is
relative to the aligned `'fine'` spot (1.0007 at travel 0); `peak_curve` normalises to the
same quality, so its travel-0 point reads 1 to ~1e-4.

Peak intensity `rel` vs travel (`peak_curve`, `'trend'`; ±30 µm covers the lower-only
travels of the X list above):

| travel (µm) | −30 | −20 | −10 | −5 | 0 | +5 | +10 | +20 | +30 |
|---|---|---|---|---|---|---|---|---|---|
| lower | 0.30 | 0.47 | 0.81 | 0.95 | 1.00 | 0.95 | 0.81 | 0.48 | 0.31 |
| upper | 0.27 | 0.43 | 0.79 | 0.94 | 1.00 | 0.94 | 0.79 | 0.43 | 0.26 |

## Performance

Single thread, default parameters, measured on the Apple M5 Pro of this checkout (the
module README gives 2.8 GHz Xeon numbers for its own defaults, ~2–4× slower):

| call | time |
|---|---|
| `steer` | 0.5 ms (four chief rays) |
| `spot` `'fine'` | 2.7 ms (W = 3.5 µm: lower up to 20 µm, upper up to 10 µm) – 8.3 ms (upper 30 µm, W = 9 µm) |
| `spot` `'fast'` | 1.4 – 4.7 ms |
| `spot` `'trend'` | 1.2 – 4.1 ms |
| `D_auto` (first call per parameter set, then cached by the core) | 27 ms |
| `peak_curve`, 41 points (±10 µm), `'trend'` | 52–55 ms (either actuator); 61 points (±30 µm, as on the page) 85 ms lower, 98 ms upper |
| `plan_offsets`, 7 rows, lower only | 21 ms |

The slow end of each range is large travel, upper first: the defocused spot needs a larger
window. With the module's defaults (θ = 60°, κ = 1) the upper actuator defocuses about
five times more per µm (1.02 against 0.19 µm), so its spots are slower at the same travel
(`'fine'` 13 ms against 2.7 ms at upper 10 µm, `peak_curve` upper 101 ms against 52 ms).
In interactive code compute `'fast'` while a value is being dragged and `'fine'`
when idle, keep `'fine'` at large travel off the GUI thread (`run_async`), and use
`'trend'` for sweeps. The core caches the auto D and the aligned reference spot per
full parameter set; every other result, including `reference_spot`, is a fresh copy.

## GUI page (Simulate → Pump steering)

A left column with the two actuators (spin box ±500 µm, slider ±30 µm in 0.1 µm steps, step
buttons −1 / −0.2 / +0.2 / +1 µm and Zero, the per-µm spot shift, the KDC101 change in mm and
counts) above the `SteeringParams` form (basic fields first; the ±1 sides as choices; D is
greyed out while D auto is on and the resolved value is shown beside the checkbox; κ is
"Tilt calibration κ" in the Steering mirror group) and the "Calibrate the spot shift from a
measured step" box (actuator, signed travel in µm from the aligned position, measured spot
shift in nm, "Fit κ", which runs `calibrate_tilt_scale` with the current parameters, R
included, and puts κ in the form; below it κ and the spot shift of one 0.2 µm step of each
actuator), and five tabs. Every travel on the page (spin boxes, slider, curves, tables,
KDC101 counts) is the real push-rod travel. The slider, the peak curves, the shape gallery
and the data table span ±30 µm, which covers the lower-only travels of the X list in "Link
to simulation" (+26.6 / −26.4 µm for X = ∓7.5 µm).

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
  the actuator side with the effective tilt δ = κ · travel / L and the KDC101 change of the
  real travel. The traces use the effective travels κ · travel, as the adapter does.
* **3D view.** The module's own interactive page (`reference/ltem-steering.html`) in an
  embedded browser: the three.js model of the box, the beams and the mount. The page tilts
  the mirror by travel / L (the nominal lever) and has no tilt scale, so it is driven with
  the effective travels κ · travel (`core_travel`): the mirror tilt, the beams and the spot
  it draws match the other tabs. Its trail on the sample starts at the current travels when
  the page loads and is cleared by "Clear trail" with the scopes; it records each actuator
  separately, so a jump of both travels (Zero both, Move to a row) shows as an L-shaped
  corner.
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
  λ, NA and D (or D auto). κ has no input there; it reaches the page through the travels,
  which are handed over as κ · travel. Its sliders clamp and round to their range and step
  (R 2–30 mm in 0.01, θ 40–72°, β 0–65°, L 15–60 mm, α 15–70°, dist 80–600 mm in 10,
  λ 700–1600 nm in 10, NA 0.25–0.6, D 3–24 mm in 0.5; travel ±500 µm in 0.01 µm, applied
  to the effective travel: 0.01 µm of it is 0.03 µm of real travel at the default κ, and the
  ±500 µm real range is ±167 µm of it); the tab lists every value the model shows clipped,
  rounded or not at all. `zLow`, `portX`, `foldY`, `yawSide`, `pitchSide` and
  `bias_axis_deg` have no input there: the 3-D model always draws zLow = 1 mm, the port at
  260 mm, the fold mirror 6 mm behind the tip and an upright mount, in the lab frame, with
  no bias axis. The numbers in the other tabs come from the adapter and use every
  parameter. The page's R slider was widened from 6–30 mm in 0.5 to 2–30 mm in 0.01 mm for
  the earlier calibration that fitted R; the default R = 16 mm lies on both grids.
  Roundings below 0.1 % (`SNAP_REL`) are not listed.
* **The sample orientation is assumed** (φ = 0) until `bias_axis_deg` is set; the X of
  every row in "Link to simulation" depends on it.
* **2-D link.** The simulation takes X only. The Y that a single-actuator scan adds
  (+1.70 µm at X = +5 µm), the defocus and the change of spot size are reported, not fed
  into the 2-D model. `Params.spot_w_um` (1/e² radius, assumed 3 µm ↔ FWHM 3.5 µm) stays a
  separate input; the aligned spot predicted here is FWHM 1.56 × 1.12 µm (1/e² radii
  ≈ 1.33 × 0.95 µm) for an ideal TEM00 beam at θ = 45° (2.18 × 1.12 µm at the module's
  θ = 60°).
* **The calibration is a literature estimate.** κ = 0.33336 reproduces 55 nm per 0.2 µm
  step of the upper actuator at an assumed θ = 45° with R = 16 mm: the spot moves a third
  of what the nominal tilt predicts, so every travel ↔ X conversion may be off by up to a
  factor ~3 (κ = 1 would need a third of the travel) until a camera calibration replaces
  it ("Calibration of the spot shift"). The port shift is scaled by the same κ.
* **From the module** (details in its README): scalar diffraction without polarisation;
  ideal TEM00 input and mirror surfaces; an aperture fixed in space; best focus at travel
  (0, 0) assumed; no actuator backlash (Z812 < 8 µm: approach each target from the same
  side) or load effects; magnitudes ∝ κR/L with κ fitted to a literature step, not measured
  on this instrument.
