# ltem-sigaas-simulation

Self-consistent 2-D drift-diffusion + EL2-trap model of a continuously pumped,
laterally biased SI-GaAs gap, and the near-field (AFM-tip) THz emission signal
it predicts. Built to test why the near-field LTEM signal of our 40 µm gap at
±25 V is **one-sided**: strongest with the pump spot displaced ~5 µm from the tip
toward the side set by the bias polarity, without a sign change across the spot.

中文摘要：对连续泵浦、横向加偏压的 SI-GaAs 间隙做 2-D 自洽漂移-扩散 + EL2 深能级
稳态模拟（Poisson + 电子/空穴连续性 + EL2 占据态，全耦合 Newton，伪瞬态延拓），
再用"下一个脉冲的载流子 × 局域稳态场"作为发射源，计算针尖处 E_z 随光斑-针尖
相对位置 X 的变化，并与 ±25 V 的实测 RMS 幅度对比。所有参数在 `ltemsim/params.py`
中给出来源和范围；泵浦功率、光斑尺寸是假设值，用实测光电流 (0.6 mA @ 25 V) 来标定。

## Layout

```
ltemsim/
  params.py      parameter sheet with units, ranges and sources (edit here)
  grid.py        nonuniform tensor grid, GaAs (z<0) + air (z>0)
  fv.py          finite-volume faces, Bernoulli function
  physics.py     scaling, EL2 kinetics, mobility, generation
  poisson.py     FV Poisson operator (dielectric interface, electrode faces)
  continuity.py  Scharfetter–Gummel fluxes and contact currents
  newton.py      coupled Newton (psi, ln n, ln p, f) + backward-Euler marching
  solver.py      Simulation driver, Solution container (save/load .npz)
  tip.py         emission proxy P = n_pulse·E, tip E_z response, tip scan
  plots.py       figures
scripts/
  run_baseline.py        default run -> results/baseline/
  sweep.py               parameter sweeps (pump power, EL2, sigma_p, spot size)
data/measured_tip_scan.csv   measured RMS vs X for ±25 V (Oct 2026 notes)
tests/test_numerics.py       Poisson vs analytic, equilibrium, Jacobian FD, dark current
docs/model.md                equations, discretisation, assumptions
```

## Run

```bash
pip install -r requirements.txt
python tests/test_numerics.py          # ~2 min
python scripts/run_baseline.py         # ~10-20 min on a laptop; writes results/baseline/
python scripts/run_baseline.py out=results/anode_plus_x anode_side=+x   # any Params field can be overridden
python scripts/sweep.py power          # P_avg = 0.5 ... 8 mW
```

Every `Params` field can be given on the command line as `name=value`.

## Geometry and conventions

* x along the bias field, z vertical, GaAs at z < 0; thin Au electrodes on the
  surface at |x| ≥ 20 µm.
* `anode_side` = which electrode carries the supply's + terminal at "+25 V".
  **Recorded 2026-10-26 from the wiring sketch: the −X electrode.** So at +25 V
  the anode is at −x, the cathode at +x. (`anode_side=+x` mirrors everything.)
* The spot sits at the gap centre; the tip is scanned. Experimental
  X (spot relative to tip) = −x_tip. The strongest measured +25 V signal
  (X = +5 µm) therefore has the tip 5 µm on the **anode** side of the spot.

## Parameters (defaults, see `params.py` for ranges and sources)

| quantity | value | status |
|---|---|---|
| N_EL2 | 1.5e16 cm⁻³ | literature (~1e16) |
| E_T | E_c − 0.75 eV | literature (0.75–0.82) |
| σ_n (EL2⁺ + e) | 1e-14 cm² | literature range 1e-15 (low field) – 1e-13 (1e4 V/cm) |
| σ_p (EL2⁰ + h) | 2e-18 cm² | from memory (Mitonneau/Lang), not re-verified |
| σ_n^o, σ_p^o | 1e-16, 5e-17 cm² | from memory (Chantre 1981 scale) |
| N_A − N_D | 1.5e15 cm⁻³ | sets EL2⁺ in the dark (10 % ionised) |
| μ_n, μ_p | 6000, 350 cm²/Vs | Hall values for SI-GaAs |
| v_sat | 1e7 cm/s | monotone μ(E) |
| B (radiative) | 1e-10 cm³/s | Stout & Kushner 1996 |
| α (820 nm), R | 1.2e4 cm⁻¹, 0.33 | standard tables |
| P_avg | 2 mW | **assumed**; unity-gain current 0.44 mA/mW |
| spot w (1/e²) | 3 µm | **assumed** (0 V step fit: 3.2 µm incl. tip) |

## Results so far (2026-10-26)

Production grid: `results/baseline/` (maps, profiles, polarisation, tip scan, csv).
Quick scenario tests on a coarse grid (0.5 µm cells, ~1 min each):
`results/coarse_scenarios/` (figures, tables, and the saved solutions). Numbers
below are from the production baseline unless marked "coarse".

**1. Baseline (25 V, anode −x, 2 mW, w = 3 µm, literature EL2 parameters, blocking
Au contacts).** The continuously pumped spot does not stay a local conductive
island: with the radiative-limited lifetime (10–100 ns) the photo-plasma spreads
over the whole 40 µm gap (n ≈ p ≈ 5×10¹⁷ cm⁻³ at the centre, 2–5×10¹⁶ at the
electrodes), EL2 is fully neutral everywhere in the plasma, and the bias field
collapses in the entire gap (|E_x| ≈ 0.05 kV/cm vs 4 kV/cm bare). The 25 V drops
within ~1 µm of the **cathode**, in a layer where EL2⁺ ≈ 5×10¹⁵ cm⁻³ (a
trap-charged, blocking-contact layer). Photocurrent 0.12 mA (gain 0.14; measured
0.6 mA). The only fields left at the spot are the ambipolar (Dember-type) fields,
which are symmetric, so the predicted tip response is symmetric and peaked at
X = 0 — it does not reproduce the one-sided measurement (peak at X = +5 µm).

**2. Pump power 0.02–2 mW (coarse) and 0.5 / 8 mW (production, `results/sweeps/power/`).**
I ∝ P^0.6 (0.5 mW: 0.063 mA; 2 mW: 0.124 mA; 8 mW: 0.205 mA); the gap is
shorted at every power (even 20 µW, n(0) ≈ 5×10¹⁵), and the response stays symmetric.
Reaching the measured 0.6 mA by power alone would need tens of mW in this model.

**3. Confined plasma (coarse): extra SRH lifetime 0.1–1 ns, or S = 10⁶ cm/s.** The
electrons are confined to the spot, but the carrier that is in excess leaks into
the dark gap (holes to ~10¹⁵ cm⁻³ out to the anode) and shorts it from the other
side; |E_x(±5 µm)| ≤ 0.3 kV/cm, I ≈ 0, response symmetric (two lobes at ±2.5 µm).

**4. Injecting contacts (coarse, n = p = 10¹⁶–10¹⁷ at the metal).** Restores a
field in the gap and makes it one-sided — but on the **cathode** side (hole-rich
flank: E = J/(e p μ_p) is larger there), with the tip response peaking for the tip
on the cathode side, i.e. opposite to the measurement, and with 17–140 mA of
dark double-injection current, which is unphysical for Au/SI-GaAs.

**5. Field bending only (no carriers, no EL2; `scripts/field_bending.py`, `results/field_bending/`).**
Wafer 350 or 500 µm, back surface floating or held at the bias midpoint: at the
surface the four cases are identical to 3 digits (the back surface is ≥ 9 gap
widths away). At x = 5 µm from the spot centre: E_x = 3.9 kV/cm, E_z = 5 V/cm at
the surface, 49 V/cm at 1 µm, 0.33 kV/cm at 20 µm (tilt 0.7° / 6.9°); the
exp(−αz)-weighted vertical polarisation is 0 / 0.5 / 1.0 / 1.6 % of the in-plane one
at x = 0 / 2.5 / 5 / 7.5 µm. The tip response is antisymmetric in X (lobes at
±1.5–2.5 µm, zero at X = 0, 0.13 at |X| = 5) with ~0 % vertical share, so field
bending cannot give the one-sidedness, the nonzero X = 0 value, or the peak at 5 µm.

**Reading.** Within this steady-state model, any plasma channel that connects the
spot to an electrode screens the bias field in the gap, and the voltage then drops
in a thin trap-charged layer at a blocking contact. The measured facts — strong
emission with the spot 5 µm from the tip on the anode side, ~0.5–1 collected
carrier per photon — require a bias field that survives near the spot while a
substantial current flows. None of the explored variants does both. What decides
between the regimes is not known yet: the pump power on the sample, the
steady-state carrier density and lifetime (a 0 V reflectivity decay out to
~1 ns; photocurrent vs chopping frequency), and the contact behaviour (dark I–V,
linearity of photocurrent vs V, change under polarity reversal). On the model
side the 2-D line spot overestimates how far the plasma reaches, the contacts
are either fully blocking or fully injecting (a Schottky contact is neither),
and the 80 MHz pulse train is replaced by its average.

## Single-pulse emission chain (added 2026-10-26 evening; `docs/pulse_chain.md`)

Requested scope: one pulse, bare field, no traps; Maxwell solver for the optical pump;
Pizzuto-style self-consistent carrier dynamics; tip scattering demodulated at the
68 kHz tapping. Three stages, all 2-D:

1. **Pump FDTD** (`ltemsim/optics_fdtd.py`, `scripts/run_fdtd_pump.py`, `results/fdtd_pump/`):
   820 nm, 100 fs, p-pol, 45° incidence, 15 nm Yee grid, CPML, lossy GaAs, Drude PtIr tip
   (R = 40 nm, 15° cone, 20 nm gap). Validation without tip: absorbed fraction 0.73 vs
   Fresnel 0.80. With the (2-D wedge) tip: apex enhancement ×1–4 in interference bands on
   the illuminated side, full shadow on the far side (absorbed fraction 0.39) — an upper
   bound for a real cone.
2. **Carrier dynamics** (`ltemsim/pulse_dynamics.py`): hydrodynamic electrons/holes with
   inertia (τ_n = τ_p = 0.1 ps), self-consistent Poisson every 0.5 fs, 4 ps; quasi-static
   THz near field above the tip position vs height from the instantaneous charge.
3. **Tip + demodulation** (`ltemsim/tip_demod.py`): finite-dipole tip in Pizzuto's
   conventions (a 40 nm, l 600 nm, g 0.8e^{0.06i}, 0.84 THz), Drude-GaAs β under the tip,
   tapping A = 180 nm (as in the experiment), harmonics 1–4; detected waveform
   ∝ d S_n/dt, 2.2 THz low-pass; RMS vs beam position X (`scripts/run_pulse_chain.py`,
   `scripts/redemodulate.py`, `results/pulse_chain/`).

Results (RMS of the detected 3rd-harmonic waveform, A = 180 nm, normalised; anode at −x;
beam travels toward +x; harmonics 1–4 in `results/pulse_chain/*/redemod_A180_h0.json`):

| run | X = −7.5 | −5 | −2.5 | 0 | +2.5 | +5 | +7.5 |
|---|---|---|---|---|---|---|---|
| 2 mW, symmetric spot (no tip map) | 0.11 | 0.76 | 0.22 | 0.02 | 0.24 | **1.00** | 0.14 |
| 2 mW, with 2-D tip shadow map | 0.11 | 0.70 | **1.00** | 0.64 | 0.54 | 0.13 | 0.09 |
| 30 mW (saturated, 2×10¹⁹), tip map | 0.19 | **1.00** | 0.46 | 0.12 | 0.12 | 0.31 | 0.11 |
| measured +25 V | – | (≈0) | – | 0.24 | 0.59 | 1.00 | 0.15 |
| measured −25 V | 0.15 | 0.45 | 0.52 | 0.33 | 0.20 | 0.32 | – |

With a symmetric spot the chain is **one-sided by itself**: the signal peaks with the
tip 5 µm from the spot on the **anode side** (X = +5 for anode at −x), 1.3× the
cathode side (1.6× with a 60 nm tapping amplitude), and drops sharply at 7.5 µm — the same side and shape as the +25 V
data. The mechanism is the electron front: the light electrons accelerate toward the
anode-side flank and build the sharpest, fastest charge front there, which is what a
tip sensing dE_z/dt picks up; the hole front on the cathode side is slower. This is
also a polarity-flipping effect, unlike the illumination-direction asymmetry. The
spot-centre value is too small (0.07 vs 0.24). The 2-D tip map over-shadows the +x
side and rearranges the pattern (its 3-D counterpart is a ~1 µm stripe); the truth is
between the two 2 mW rows. At 30 mW the per-pulse density is pinned at the saturation
value and the flanks become flat-topped.

## Status / to do

* [x] solver, validation tests
* [x] emission proxy + tip scan vs X
* [x] baseline + coarse scenario tests (power, lifetime, surface recombination, contacts)
* [x] single-pulse chain: pump FDTD → hydrodynamic carriers + Poisson → FDM tip → tapping demodulation
* [ ] 3-D (or body-of-revolution) pump FDTD for a realistic tip shadow; Monte Carlo carriers (intervalley) behind the same Poisson
* [ ] Schottky (thermionic) contact model
* [ ] pulsed (80 MHz) carrier dynamics instead of the time average
* [ ] 3-D (finite spot length) estimate of the space-charge field
* [ ] once P_avg, lifetime and contact I–V are measured: sweeps of N_EL2, σ_p, σ_n, spot size
