# LTEM pump-spot steering module

> **中文简介** 这个模块计算非局域 LTEM 实验里，盒外 KS1-Z8 转向镜上的两根 Z812 电机移动时，泵浦光斑在样品面上往哪走、形状怎么变。下面那根电机管水平方向，上面那根管竖直方向。计算核心与交互页面完全一致，有 JavaScript 参考实现和 NumPy 移植两份，用同一组标准答案（`tests/golden.json`）互相校验。集成说明见下文英文部分；给 Claude Code 的任务说明在 `HANDOFF_PROMPT.md`。

A self-contained physics core for the neaspec THz s-SNOM used in non-local laser THz emission microscopy (LTEM). It takes the push-rod travel of the two Z812 actuators on the external KS1-Z8 steering mirror and returns:

1. **Steering response** from an exact chief-ray trace:
   - spot displacement on the sample;
   - beam shift at the AFM entrance port;
   - the along/across split relative to the plane of incidence;
   - defocus and the change in pump arrival time;
   - the per-µm coefficients of both actuators.
2. **Spot shape** from scalar diffraction. The model takes a circular Gaussian pump beam through steering mirror → fold mirror → off-axis semi-paraboloid onto the tilted sample plane, including the mirror's clear aperture. It returns:
   - the intensity map on the sample;
   - the half-maximum ellipse: FWHM long × short and orientation;
   - the half-max area;
   - the peak intensity relative to the aligned spot;
   - the offset of the brightest point;
   - a half-max contour.
3. **Optimal beam diameter**: the diameter that gives the smallest spot for a given aperture, i.e. the highest peak intensity per unit laser power.
4. **Peak-intensity curves** versus actuator travel.

The core contains no UI code. `reference/ltem-steering.html` is the interactive bilingual page built on the same core. It shows the intended behaviour. Open it in a browser with internet access, since it loads three.js from a CDN.

---

## Package layout

```
ltem-steering-module/
├── README.md                 this file: conventions, parameters, API, algorithm spec, validation
├── HANDOFF_PROMPT.md         prompt to give Claude Code for the integration
├── js/
│   ├── ltem_optics.mjs       reference implementation (ES module, no dependencies)
│   ├── test_golden.mjs       regression test against tests/golden.json  → node js/test_golden.mjs
│   └── example.mjs           usage example                              → node js/example.mjs
├── python/
│   ├── ltem_optics.py        NumPy port with the same API (snake_case)
│   ├── test_golden.py        pytest: golden values + physical sanity checks → python -m pytest python -q
│   ├── example.py            usage example (+ --plot with matplotlib)    → python python/example.py --plot
│   └── requirements.txt
├── tests/
│   ├── golden.json           reference outputs (3 parameter sets: responses, spots, curves, optimal D)
│   └── make_golden.mjs       regenerates golden.json from the JS reference
└── reference/
    └── ltem-steering.html    the interactive page (3D view, scopes, spot panel; 中文/English)
```

Requirements:
- JS: Node ≥ 18. Tested with Node 22.
- Python: Python ≥ 3.9 with NumPy. pytest is needed for the tests; matplotlib only for `--plot`. Tested with Python 3.13, NumPy 2.5 and pytest 9.

---

## Quick start

**Python.** Put `python/` on `sys.path`, or copy `ltem_optics.py` into your package.
```python
import ltem_optics as O

p = O.resolve_params({})                      # defaults; D='auto' → diameter for the smallest spot (13.0 mm)
r = O.steering_response(p, yaw_um=4, pitch_um=0)
r["spot_um"]          # [-3.387, -1.272]  µm: x (+ = port side), y (+ = back)
r["perUm"]["yaw"]     # [-0.848, -0.321]  µm of spot per µm of lower-actuator travel
s = O.spot_shape(p, yaw_um=4, pitch_um=0, quality="fine")
s["fwhmL"], s["fwhmS"], s["rel"]   # 2.18 µm, 1.27 µm, 0.78 (peak vs aligned spot)
s["I"]                # float32 (M, M) intensity map, see "Spot arrays" below
O.contour(s, 0.5 * s["peak"])      # half-max contour, (n, 4) segments
O.peak_curve(p, "yaw", [-5, 0, 5]) # [{'travel_um': -5, 'rel': 0.687, ...}, ...]
```

**JavaScript**
```js
import * as O from './js/ltem_optics.mjs';
const p = O.resolveParams({});
const r = O.steeringResponse(p, 4, 0);
const s = O.spotShape(p, 4, 0, 'fine');
const segs = O.contour(s, 0.5 * s.peak);      // flat [x1,y1,x2,y2, ...]
const curve = O.peakCurve(p, 'yaw', [-5, 0, 5]);
```

---

## Conventions (read before integrating)

**Lab frame.** Right-handed, in mm. The origin is the focus, which is the tip apex on the sample surface.
- **+X** points toward the entrance-port wall. This is "right" for someone standing at the open front of the AFM box.
- **+Y** points toward the back panel; **−Y** is the "front", the open side where the user stands.
- **+Z** points up. The sample surface is the plane z = 0.

**Optical path of the pump.**
1. The input beam travels −Y on the table. It reflects off the steering mirror at S = (portX + dist, foldY, h). The mirror face is vertical and α = 45° by default.
2. The beam travels −X into the port in the right wall, at the plane x = portX.
3. A vertical fold mirror behind the tip, at (P_x, foldY, h) with normal ∝ (1, −1, 0), turns it to −Y.
4. It travels toward the front and hits the **right half** of a semi-paraboloid, x² + z² = 4·fp·(y + fp). The mirror axis runs along Y, parallel to the sample. The focus is at the origin, and the vertex is at y = −fp, in front of the tip.
5. The semi-paraboloid focuses the beam onto the sample at angle of incidence θ.

The chief point on the paraboloid is P = R·(sinθ·sinβ, −sinθ·cosβ, cosθ), with fp = R(1 + sinθ·cosβ)/2 and h = R·cosθ.

**Actuators.** The mount is a Thorlabs KS1-Z8, installed upright as Thorlabs draws it. Seen from the actuator side, the manual knob is the pivot, at the lower left.
- `yaw_um` is the **lower** Z812, at the lower right. It gives a pure horizontal tilt.
- `pitch_um` is the **upper** Z812, at the upper left. It gives a pure vertical tilt.

Travel is in µm. **Positive means the push rod extends**, so the KDC101 reading rises. In the Z8 manual, the forward limit is at full extension.
- The mirror rotation is δ = travel / L. Extending an actuator tilts the mirror normal away from that actuator.
- Yaw extension turns the mirror counter-clockwise seen from above (`yawSide = +1`).
- Pitch extension tilts the mirror face down (`pitchSide = +1`).
- If a mount is installed mirrored, set `yawSide`/`pitchSide` to −1.

Inside the code, `dp` is the pitch travel and `dy` is the yaw travel, both in mm.

**Resulting directions** with the default parameters:

| Actuator | Spot moves per +1 µm of travel | Size | Character |
|---|---|---|---|
| Lower (yaw) | 0.848 µm left, 0.321 µm toward the front | 0.907 µm | mostly across the plane of incidence |
| Upper (pitch) | 0.737 µm right, 0.923 µm toward the front | 1.181 µm | along the plane of incidence, toward the parabolic mirror; defocuses strongly |

The two directions are about 108° apart on the sample.

**Spot arrays.** `spot["I"]` is an M × M float32 array. In JS it is a flat, row-major `Float32Array`.
- Row index b runs along +y (back). Column index a runs along +x (port side).
- Grid coordinates relative to the window centre are u_a = −W + a·2W/(M−1) µm, and the same for v_b.
- The window centre `x0` = [x, y, z] (z ≈ 0) is the chief-ray hit point on the sample, in lab mm.
- `peakAt`, `cx` and `cy` are in µm relative to `x0`.
- `ang` is the long-axis angle in radians, measured from +X toward +Y.

**Intensity normalisations.**
- `peak` is the maximum of `I`. `I` is normalised to a perfect focus of the light that passes the aperture, so `peak` is Strehl-like and ≈ 1 for the aligned beam.
- `IperP` is the peak |E|² per unit laser power, in 1/µm². It includes clipping losses. It is the quantity that is maximised for the "smallest spot".
- `rel` = `IperP / IperP(aligned spot)`, the peak intensity relative to the original spot. The page shows this as "peak intensity (original = 100 %)".
- `k` turns `I` into intensity relative to the aligned peak: `I_rel = I·k`.

`I` and `IperP` are the intensity across the beam, i.e. power per area perpendicular to it. The power per unit *sample* area is smaller by about cos θ = 0.5, a constant factor at fixed geometry. The *shape* (FWHM, ellipse, contour) is measured in the sample plane, so the aligned spot is about 1/cos θ longer along the plane of incidence.

---

## Parameters

| Key | Unit | Default | Meaning | Source / confidence |
|---|---|---|---|---|
| `R` | mm | 16 | effective focal length of the parabolic mirror (chief point → focus) | reported for another neaspec THz-TDS s-SNOM; **calibrate** |
| `theta` | deg | 60 | angle of incidence on the sample | neaspec standard |
| `beta` | deg | 45 | azimuth of the plane of incidence, measured from the front–back axis toward the port side | from the user's sketch (40–50°); affects only the front–back component |
| `alpha` | deg | 45 | angle of incidence on the steering mirror (a 90° fold) | setup photo |
| `dist` | mm | 250 | steering mirror → port plane | setup photo; sets the beam shift at the port (∝ dist); the spot moves by ~1 % when dist doubles |
| `L` | mm | 37.7 | lever arm from actuator contact to pivot | Thorlabs: KS1-Z8 makes 1.1″ (5.3 µrad) per 0.2 µm Z812 step |
| `lam` | nm | 820 | pump wavelength | both Pizzuto papers (Ti:sapphire) |
| `D` | mm or `'auto'` | `'auto'` | 1/e² diameter of the input beam; `'auto'` is the diameter giving the smallest spot (13.0 mm with the defaults) | the user's optics are optimised for the smallest spot |
| `NA` | – | 0.42 | illumination aperture: a cone of half-angle asin(NA) around the aligned chief ray | neaspec patent example (50° cone); a published neaspec IR mirror has 0.46 |
| `zLow` | mm | 1 | height of the mirror's lower edge above the sample | neaspec patent ("about 1 mm") |
| `portX` | mm | 260 | x of the port plane | model geometry; together with `dist` it sets the path to the fold mirror (~1 % effect) |
| `foldY` | mm | 6 | y of the fold-mirror centre behind the tip | model geometry; negligible effect |
| `yawSide`, `pitchSide` | ±1 | +1, +1 | positions of the actuators relative to the pivot | Thorlabs KS1-Z8 3D model and the setup photo |

The **directions** do not depend on the uncertain values. The **magnitudes** scale with R/L. The **shape** results depend strongly on D and NA. To calibrate R/L, step 50 µm and measure the spot shift with the AFM top-view camera. Measure D before the AFM.

---

## API

The Python and JS APIs mirror each other: snake_case in Python, camelCase in JS.

| Python | JS | Returns |
|---|---|---|
| `resolve_params(params)` | `resolveParams(params)` | Full parameter dict. Resolves `D='auto'`, rounded half-up to 0.1 mm and cached per parameter set. |
| `steering_response(params, yaw_um, pitch_um)` | `steeringResponse(params, yaw, pitch)` | `spot_um[x,y]`, `port_um[y,z]`, `perUm{yaw,pitch}`, `perUmPort{yaw,pitch}`, `incidencePlane[x,y]`, `along_um`, `across_um`, `defocus_um`, `arrival_fs`, `yawTurnsBeamToFront`. Raises if the chief ray misses the parabolic mirror. |
| `spot_shape(params, yaw_um, pitch_um, quality, W_start)` | `spotShape(params, yaw, pitch, quality, Wstart)` | Spot dict (see Conventions) plus `rel`, `k`, `D`. Returns `None`/`null` if fewer than 20 rays reach the sample. `W_start` is optional; it defaults to the aligned spot's window. |
| `peak_curve(params, axis, travels_um, quality)` | `peakCurve(params, axis, travels, quality)` | List of `{travel_um, rel, fwhmL, fwhmS, area, ang}`; `axis` is `'yaw'` or `'pitch'`. `quality` defaults to `'trend'`. |
| `optimal_beam_diameter(params)` | `optimalBeamDiameter(params)` | Diameter in mm, unrounded. |
| `contour(spot, level)` | `contour(spot, level)` | Segments `[x1,y1,x2,y2]` in µm relative to `x0`. Python returns an `(n,4)` array; JS returns a flat list. |
| `reference_spot`, `compute_spot_raw`, `adaptive_spot_raw`, `geometry`, `mirror_normal`, `trace_chief`, `trace_after_steer` | same names in camelCase | Lower-level building blocks. |

`quality` is `'fine'` (M = 72, Nmin = 40), `'fast'` (48, 30), `'trend'` (40, 28), or a `{M, Nmin}` dict.

Pass a resolved parameter dict (`resolve_params(...)`) into the other functions. Raw dicts also work, and `D='auto'` is cached, but resolving once is clearer.

The module keeps two caches, both keyed by the full parameter set:
- the auto D;
- the aligned reference spot, which `reference_spot` returns. Treat it as read-only.

Everything else is returned as a fresh object.

The building blocks form four layers. Reuse whichever matches your software:
1. **Actuators → mirror normal**: `geometry`, `mirror_normal`.
2. **Ray trace**: `trace_after_steer` and `trace_chief`, through the fold mirror and the semi-paraboloid to the sample.
3. **Spot engine**: `compute_spot_raw` and `adaptive_spot_raw`, the Debye sum on the tilted sample.
4. **High-level calls**: `steering_response`, `spot_shape`, `peak_curve`, `optimal_beam_diameter`.

---

## Algorithm (specification for porting)

All steps are deterministic. A port that follows them reproduces `golden.json` (see Validation for the observed agreement).

### 1. Geometry
- th = θ and be = β, in radians.
- P = R(sin th·sin be, −sin th·cos be, cos th).
- fp = R(1 + sin th·cos be)/2 and h = P_z.
- Fold mirror: F1 = (P_x, foldY, h), with normal nF1 = (1, −1, 0)/√2.
- Steering mirror: S = (portX + dist, foldY, h).
- uIn = (−1, 0, 0) is the aligned outgoing beam.
- d0 = uIn rotated about +Z by π − 2α. This is the input beam.
- n0 = norm(uIn − d0) is the aligned mirror normal.
- uBack = norm(n0 × Z).

### 2. Mirror normal
n = rotZ(ay) ∘ rot_uBack(ap) applied to n0, with:
- ap = −pitchSide·dp/L;
- ay = yawSide·dy/L;
- dp and dy in mm.

Use Rodrigues' formula. The pitch rotation is applied first, then the yaw rotation.

### 3. Chief ray
1. Reflect: d1 = norm(reflect(d0, n)), starting from S.
2. Intersect the fold plane: t = ((F1 − o)·nF1)/(d·nF1). Return no hit if |d·nF1| < 1e-12; accept t > 1e-9.
3. Reflect off the fold: d2 = norm(reflect(d1, nF1)).
4. Intersect the paraboloid. Its coefficients are:
   - a = d_x² + d_z²;
   - b = 2(o_x d_x + o_z d_z) − 4fp·d_y;
   - c = o_x² + o_z² − 4fp(o_y + fp).

   Use the stable roots q = −½(b + sign(b)·√disc), giving t ∈ {q/a, c/q}. Skip a candidate whose divisor is 0; a can be exactly 0. Take the smallest root > 1e-9.
5. The paraboloid normal is ∝ (2x, −4fp, 2z). Reflect off it; the reflected ray must point downward (r_z < 0).
6. Intersect z = 0. The hit point is Q.

### 4. Steering response
- Trace four chief rays: at (0, 0), at the requested travel, at yaw +1 µm and at pitch +1 µm.
- `perUm` = (Q(+1 µm) − Q(0))·1000. These are forward differences.
- The port point is S + d1·(portX − S_x)/d1_x.
- The plane of incidence is the horizontal unit vector `ph` of the aligned chief ray after the paraboloid.
- along = ΔQ·ph and across = ΔQ·(−ph_y, ph_x, 0).
- defocus = |along|·sin θ. Arrival time = defocus·3.33564 fs/µm.

### 5. Spot on one window
1. **Aperture angle and pupil.**
   - apA = asin(min(0.95, NA)), w = D/2.
   - rmax = min(1.8w, 1.3·R·tan apA).
   - Aliasing period: per = (λ[µm]·R)/(2·rmax).
   - Sample count: N = max(Nmin, min(120, ⌈2W/(0.8·per)⌉)).
2. **Ray grid.**
   - a_i = −rmax + 2rmax(i + ½)/N for i = 0 … N−1, and b_j likewise. Loop i outer, j inner, and keep a² + b² ≤ rmax².
   - Each ray has amplitude A0 = exp(−(a²+b²)/w²).
   - a2all = Σ A0², summed over all of these rays before any tracing or clipping.
   - The ray origin is o = S + a·e1 + b·e2. Here e1 = norm(d1 × Z), falling back to d1 × Y if that vanishes, and e2 = norm(e1 × d1).
   - Trace each ray as in step 3 and drop the ones that miss.
3. **Soft aperture.**
   - cell = 2rmax/N, Lp = |P_hit| and p0 = norm(P_chief,aligned).
   - mEdge = (apA − acos(min(1, P_hit·p0/Lp)))·Lp and mLow = P_hit,z − zLow.
   - wAp = clamp(½ + mEdge/cell, 0, 1)·clamp(½ + mLow/cell, 0, 1).
   - Drop rays with wAp ≤ 0. Return null if fewer than 20 rays are left.
4. **Phase and amplitude.**
   - x0 = Q of this travel's chief ray.
   - OPL = |A − o| + |P − A|, where A is the fold hit and P the paraboloid hit.
   - ph = OPL + r·(x0 − P).
   - Phase = k(ph − ph_first accepted ray), with k = 2π/λ[mm].
   - Amplitude = A0·wAp/|x0 − P|. asum = Σ amplitude.
5. **Field on the grid.**
   - E(u_a, v_b) = Σ_i c_i·exp(ik(r_x,i u_a + r_y,i v_b)) on an M × M grid of half-width W, where c_i = amplitude·e^(i·phase).
   - The sum separates in u and v, so it is a matrix product.
   - I = |E|²/asum², stored as float32.
6. **Peak refinement.**
   - Start at the grid maximum. Take the first index in row-major order, comparing the float64 values.
   - Run a pattern search with step h = du/2, trying in order (+h,0), (−h,0), (0,+h), (0,−h). Evaluate the sum directly; move to the best improving point.
   - If no step improves, halve h.
   - Stop after 16 iterations or when h ≤ du/40. The result is `peak`.
7. **Half-max ellipse.**
   - Weight each pixel: t = 0 for I/peak ≤ 0.45, 1 for I/peak ≥ 0.55, and linear in between.
   - S, Σt·x and Σt·y use the double-precision t.
   - The second moments use t rounded to float32. This is a JS detail, kept for bit-compatibility.
   - fwhmL, fwhmS = 4√λ₁, 4√λ₂ (the eigenvalues of the second-moment matrix).
   - `ang` = ½·atan2(2xy, xx − yy).
   - `area` = S·px², with px = 2W/(M−1).
8. **Border.** `border` = the maximum of I on the four window edges, divided by `peak`.
9. **Intensity per power.** `IperP` = peak·asum²·dA/(λ_mm²·a2all)·1e-6, with dA = (2rmax/N)².

### 6. Adaptive window
Start from W = clamp(W_start, 2.4, 18), with 3.5 if no start is given. Then repeat up to 5 times:
- If border > 0.03 and W < 18, set W ← min(18, 1.6W) and recompute.
- Otherwise, if no shrink has happened yet, border < 0.004 and W > 2·want, where want = max(2.4, 1.3·fwhmL): shrink once to W ← max(2.4, 1.25·want) and recompute.
- Otherwise stop.

The aligned reference spot is computed at 'fine' quality from W = 3.5 and cached. `spot_shape` starts other spots from the reference's W. `spot_shape(0, 0, 'fine')` returns a copy of the reference itself.

### 7. Optimal D
1. Use the `peakOnly` variant of step 5: IperP = |Σc|²/asum²·perPow at the chief point. Use Nmin = 44 and W = 0.5 µm, which keeps N = 44 for any realistic parameters.
2. Scan D = 3, 3.5, … 26 mm. On a tie, keep the first maximum.
3. Fit a parabola through the best value and its neighbours at ±0.5 mm. If the curvature is negative, the result is D_best + 0.25·(a − c)/(a − 2b + c).
4. Clamp the result to [3, 26]. `resolve_params` rounds it half-up to 0.1 mm.

### 8. Peak curve
- Normalise by an aligned spot computed with the *same* quality, so travel 0 reads 1 to within about 1e-4.
- Carry W from point to point. A point with no spot returns `{travel_um, rel: null}`.

### 9. Contour
- Marching squares on I with strict `> level`. The corner bits are (a,b) = 1, (a+1,b) = 2, (a+1,b+1) = 4 and (a,b+1) = 8.
- Interpolate linearly on the edges e0 = (00–10), e1 = (10–11), e2 = (11–01) and e3 = (01–00).
- The case table is `MS` in the JS code and `_MS` in the Python code.

---

## Validation

- `node js/test_golden.mjs` checks the JS reference against `tests/golden.json` (919 checks).
- `python -m pytest python -q` checks the NumPy port against the same file and tolerances. It also runs independent sanity checks:
  - the analytic small-angle results Δx = 2R·Δ/L and Δy = 2R·Δ/L·sinθ·sinβ/(1 + sinθ·cosβ);
  - the aligned spot is diffraction-limited (Strehl > 0.99) and centred on the chief ray;
  - its long axis lies along the plane of incidence;
  - its aspect ratio is ≈ 1/cosθ;
  - returned arrays do not alias the internal caches.
- Observed agreement between the Python port and the JS reference:
  - chief-ray results within 1e-10 µm;
  - all spot and curve quantities within 2e-7. The float32 intensity map sets this floor;
  - window sizes, pupil sample counts and ray counts are identical;
  - the sampled intensity values are bit-identical.

  The tolerances in `golden.json` (1e-6 for spots and curves) leave headroom for other summation orders (BLAS, SIMD). A larger deviation means a porting bug.
- Regenerate `golden.json` with `node tests/make_golden.mjs` **only** when the physics changes on purpose, and say so in the commit.

## Performance

Measured single-threaded on a 2.8 GHz Xeon core:

| Call | JS (Node 22) | Python (NumPy 2.5) |
|---|---|---|
| `spot_shape` 'fine' | 14–120 ms | 6–50 ms |
| `spot_shape` 'fast' | 4–55 ms | 3–32 ms |
| `spot_shape` 'trend' | 3–42 ms | 2–29 ms |
| 7-point `peak_curve` ('trend') | ~30 ms | ~30 ms |
| `optimal_beam_diameter` (cached by `resolve_params`) | 90 ms | 50 ms |
| `steering_response` | 20 µs | 1 ms |

The slow end of each spot range comes from large windows: large vertical travel, which defocuses strongly.

For interactive use, compute 'fast' on every change and refine with 'fine' when idle, as the page does. Keep 'fine' at large travel off a UI thread. Use 'trend' for sweeps.

## Assumptions and limitations

- **Geometry inside the box.**
  - Two vertical fold mirrors sit behind the tip and the parabolic mirror sits in front of it. Its right half illuminates and its left half collects.
  - The parabolic mirror is the semi-paraboloid of the neaspec patents, with its axis parallel to the sample.
  - The port element is treated as a flat window.
- **Steering mirror.**
  - The face is vertical and the beams run parallel to the table.
  - The rotation is applied to the normal only. The pivot is at the corner, so the reflection point also shifts by about a micrometre. This shift is ignored; it does not move the focus.
- **Spot shape.**
  - Scalar diffraction: a Debye plane-wave superposition near the focus, without polarisation.
  - Ideal TEM00 input and ideal mirror surfaces.
  - The aperture is a cone of half-angle asin(NA) around the aligned chief ray, plus the mirror's lower edge at height zLow. It stays fixed in space while the beam walks.
  - At travel (0, 0), the beam is assumed parallel to the paraboloid axis, with best focus on the sample under the tip. A residual tilt at the "local" position would make the deformation start earlier.
- **Mechanics.**
  - Actuator backlash (Z812 < 8 µm) and load effects are not modelled.
  - Travel is push-rod extension, i.e. the change in the KDC101 reading. That reading is computed from the motor encoder (1 mm = 34,555 counts, ≈ 29 nm per count), not measured at the rod tip.
- **Literature comparison.**
  - Both Pizzuto papers quote ≈ 55 nm per 1.1″.
  - This model gives ≈ 0.18 µm (lower actuator) and ≈ 0.24 µm (upper actuator) per 0.2 µm step.
  - 55 nm ÷ 5.3 µrad ≈ 10 mm, i.e. the mirror angle times a ≈ 10 mm focal length. But the beam turns by twice the mirror angle, so even R = 10 mm would give ≈ 0.11 µm.
  - Trust a camera calibration.

## Sources

- [Thorlabs KS1-Z8](https://www.thorlabs.com/item/KS1-Z8) and its [3D model](https://media.thorlabs.com/globalassets/items/k/ks/ks1/ks1-z8/19520-e0w.html?v=0116120059):
  - seen from the back, the two Z812s are at the upper left and lower right, and the manual knob is at the lower left;
  - minimum repeatable angular step 1.1″ (5.3 µrad).
- [Thorlabs Z8-series actuator manual](https://media.thorlabs.com/globalassets/items/z/z8/z80/z806/17630-d01.pdf): forward limit at full extension; 29 nm/count; backlash < 8 µm.
- [Pizzuto, Castro-Camus, Mittleman, J. Infrared Millim. Terahertz Waves 44, 769 (2023)](https://par.nsf.gov/servlets/purl/10411087):
  - external motorized mirror, pitch resolution 1.1″, ray-matrix estimate ≈ 55 nm;
  - steps of 200–250 nm, up to ≈ 2 µm;
  - pump spot ≈ 1.3 µm.
- [Pizzuto et al., ACS Photonics 8, 2904 (2021)](https://par.nsf.gov/servlets/purl/10320065):
  - 1.1″ ≈ 55 nm (simple ray optics);
  - non-local offsets ≈ 2.5 and 5 µm;
  - pump and probe focused by the off-axis parabolic mirror in the AFM.
- [neaspec patent US 8,832,861 (Ocelic)](https://patentimages.storage.googleapis.com/4b/58/77/f872ed9e5255b4/US8832861.pdf): the semi-paraboloid's lower and upper edges subtend 50° at the focus; the lower edge is ≈ 1 mm above the sample.
- [Nanoscale Advances (2025)](https://pubs.rsc.org/en/content/articlehtml/2025/na/d5na00307e): neaspec parabolic mirror, NA 0.46, 11 mm focal length.
- [Keilmann & Hillenbrand, US 2007/0183060 (US 7,591,858)](https://patents.justia.com/patent/20070183060): semi-paraboloid with its axis parallel to the sample.
- [neaspec THz-TDS s-SNOM: off-axis parabolic mirror, 16 mm focal length](https://pmc.ncbi.nlm.nih.gov/articles/PMC11435917/)
- [attocube nano-THz-TDS: parabolic mirrors on both sides, one focusing and one collecting](https://www.attocube.com/en/products/microscopes/nanoscale-imaging-spectroscopy-1/neascope/technology/nanoTHz-TDS)
