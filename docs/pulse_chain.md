# Single-pulse emission chain (pump FDTD → carrier dynamics → tip scattering → demodulation)

Scope requested on 2026-10-26: one laser pulse only (no steady state from earlier
pulses), bare applied field (no EL2 or other traps), Maxwell solver for the optical
field, carrier dynamics in the spirit of Pizzuto et al. (Monte Carlo + Poisson), and the
tip-scattered THz field demodulated at the tapping frequency (68 kHz).

All stages are 2-D (x–z cross-section; the tip is a wedge and the spot a strip along y).

## Stage 1 — pump field, 2-D FDTD (`ltemsim/optics_fdtd.py`, `scripts/run_fdtd_pump.py`)

* Yee grid, 15 nm cells, Courant 0.5 (25 as), CPML (20 cells, m = 3) on all sides.
* p-polarisation (Ex, Ez, Hy); Gaussian pulse (100 fs intensity FWHM, 820 nm) launched by a
  soft current line source 2.2 µm above the surface with a phase tilt for 45° incidence
  (the beam travels toward +x). Beam 1/e² radius 3 µm projected on the surface.
* GaAs: lossy dielectric, n = 3.67, κ = 0.078 (α = 1.2×10⁴ cm⁻¹). Tip: PtIr ≈ Pt at 820 nm,
  ε = −17.6 + 29.6 i, as a Drude ADE fit (ε∞ = 1, ωp = 1.86×10¹⁶ s⁻¹, γ = 3.66×10¹⁵ s⁻¹);
  apex radius 40 nm, half-angle 15°, apex 20 nm above the surface.
* Output: absorbed energy density Q(x,z) = ∫ σ|E|² dt; the ratio Q_tip/Q_notip is the
  per-pulse generation correction (enhancement under the apex, shadow and shank
  reflections).
* Validation (no tip): absorbed fraction 0.73 vs Fresnel 1 − R_p(45°) = 0.80 (−9 %, finite
  beam and 15 nm grid); depth decay 0.74 µm vs 0.82 µm expected at 30 nm cells.
* Caveat: a 2-D tip is an infinitely long wedge. It blocks the whole beam on the +x side
  (absorbed fraction with tip 0.39) and reflects a standing-wave pattern (2–4× bands) on
  the −x side; a real cone shadows only a ~1 µm stripe and scatters into 3-D. The map is
  therefore an upper bound for shadowing; results are given with and without it.

## Stage 2 — carrier dynamics (`ltemsim/pulse_dynamics.py`)

Two-moment hydrodynamic model (density + drift velocity with inertia), electrons and
holes, explicit 0.5 fs steps, staggered finite volumes (face velocities, upwind fluxes),
Poisson at every step with the existing finite-volume operator (GaAs/air interface,
electrodes as equipotentials, superposed on the bare applied field):

    ∂n/∂t = G − ∇·(n v_n) + ∇·(D_n ∇n),   ∂v_n/∂t = −(e/m_n) E − v_n (1 + |v_n|/v_sat)/τ_n
    ∂p/∂t = G − ∇·(p v_p) + ∇·(D_p ∇p),   ∂v_p/∂t = +(e/m_p) E − v_p (1 + |v_p|/v_sat)/τ_p

m_n = 0.067, m_p = 0.5, τ_n = 0.1 ps (plasmon damping scale; 0.2 ps is the low-field
mobility value), τ_p = 0.1 ps, v_sat = 10⁷ cm/s.

Diffusion (added 2026-10-27, `degenerate_diffusion=True`): generalised Einstein relation
D = μ (kT/q) F_1/2(η)/F_-1/2(η), applied face by face with η from the local density
(parabolic band, N_c = 4.35×10¹⁷ cm⁻³ at 300 K). At 10¹⁹ cm⁻³ the electron gas is
strongly degenerate (E_F ≈ 0.25 eV) and D_n ≈ 450 cm²/s in the spot core; at the flanks
(n ≲ 10¹⁷) it is the classical 68 cm²/s. The carrier temperature T (which also sets N_c)
remains a separate parameter: 300 K unless stated. Generation G(x,z,t) = N(x,z)·g(t) with
N the Beer–Lambert Gaussian × FDTD ratio map, soft-capped at n_sat = 2×10¹⁹ cm⁻³
(absorption saturation: at 30 mW the linear estimate is 1.5×10²⁰ cm⁻³ at the surface,
far above the Mott density; the real density saturates in the 10¹⁹ range).

Grid: 0.1 µm in x (|x| ≤ 8 µm), 20 nm in z within 0.3 µm of the surface. Results are
converged in dt (0.25 vs 0.5 fs) and dx (0.05 vs 0.1 µm). The plasma oscillations seen
at the spot flanks are a property of the model (weakly damped Langmuir waves at
ωp ≈ 26 THz for 8×10¹⁸ cm⁻³), not of the grid; stronger damping removes them.

Relation to Pizzuto et al.: they solve the same problem with an ensemble Monte Carlo
(band structure, intervalley scattering) coupled to Poisson. The hydrodynamic model
keeps the acceleration/inertia and the self-consistent screening but not velocity
overshoot by Γ→L transfer or non-parabolicity; it can be replaced by a Monte Carlo
sampler behind the same Poisson operator.

Outputs: J_x(t), J_z(t) per unit y length (far-field reference ∝ dJ/dt), snapshots, and
the quasi-static THz near field E_z(x_tip, h, t) in the air above the tip position for
h = 1 nm … 1 µm, computed from the instantaneous charge distribution with the
dielectric half-space rule (a charge in GaAs acts on the air side as 2/(ε_r + 1) of
itself). Quasi-statics is justified: the source is ~10 µm across (λ_THz ≈ 300 µm) and the
retardation across it (30 fs) is small against the 100 fs rise.

## Stage 3 — tip scattering and demodulation (`ltemsim/tip_demod.py`)

* Finite-dipole model in the conventions of A. Pizzuto's MATLAB script
  (`docs/reference/pizzuto_fdm_demod_reference.m`): original Cvitkovic 2007 form,
  a = 40 nm, l = 600 nm, g = 0.8 e^{0.06 i}, 0.84 THz, collection factor (1 + β);
  sample response β = (ε − 1)/(ε + 1) with ε the Drude permittivity of photoexcited
  GaAs using the instantaneous density under the tip. Her LTEM (1 − β)·t_pump factor
  is available as an option but off by default, because the carrier simulation already
  supplies the emitted field below the tip (its static part, 2/(ε_r + 1), is in E_loc).
* The complex α acts on the analytic signal of E_loc(h, t): E_sc = Re[α (E_loc + iH[E_loc])].
* Tapping h(τ) = h_min + A(1 + cos Ωτ), Ω = 2π·68 kHz, **A = 180 nm** (experiment ≈ 180 nm),
  h_min = 0; harmonics **n = 1–4** all reported, S_n(t) = (1/π)∫E_sc cos(nθ)dθ (exact
  projection; equivalent to the cosine-series fit of the script). Ω drops out; A, h_min
  and n matter.
* The detected waveform is taken ∝ d S_n/dt (the static dipole left behind after the
  pulse does not radiate), low-passed at 2.2 THz (detector bandwidth). Amplitude metric:
  RMS of that trace, compared with the measured RMS vs X.

Not included: the tip as a THz antenna (frequency-dependent outcoupling), the tip's
effect on the carrier dynamics, the EO detector response beyond a 2.2 THz low-pass.

## Results added 2026-10-27 — diffusion model and spot size (`scripts/run_pulse_chain.py`, `scripts/fig_slides.py`)

dS3/dt RMS, A = 180 nm, h_min = 0, normalised to the +25 V maximum of each run (anode at −x).

| run | X = −10 | −7.5 | −5 | −2.5 | 0 | +2.5 | +5 | +7.5 | +10 |
|---|---|---|---|---|---|---|---|---|---|
| 2 mW, w = 3 µm, classical 300 K | – | 0.09 | 0.76 | 0.07 | 0.11 | 0.42 | 1.00 | 0.11 | – |
| 2 mW, w = 3 µm, degenerate 300 K | – | 0.10 | 0.73 | 0.11 | 0.15 | 0.35 | 1.00 | 0.14 | – |
| 2 mW, w = 3 µm, degenerate 600 K | – | 0.09 | 0.62 | 0.12 | 0.10 | 0.30 | 1.00 | 0.14 | – |
| 2 mW, w = 3 µm, degenerate 1500 K | – | 0.05 | 0.38 | 0.16 | 0.07 | 0.23 | 1.00 | 0.15 | – |
| 2 mW, w = 3 µm, classical 1500 K | – | 0.05 | 0.39 | 0.20 | 0.10 | 0.20 | 1.00 | 0.15 | – |
| 30 mW, w = 3 µm, degenerate 300 K | – | 0.32 | 0.60 | 0.19 | 0.15 | 0.15 | 1.00 | 0.45 | – |
| 30 mW, w = 3 µm, degenerate 600 K | – | 0.28 | 0.59 | 0.22 | 0.17 | 0.12 | 1.00 | 0.46 | – |
| 30 mW, w = 3 µm, degenerate 1000 K | – | 0.23 | 0.57 | 0.28 | 0.20 | 0.12 | 1.00 | 0.47 | – |
| 30 mW, w = 3 µm, degenerate 300 K, FDTD tip map | – | 0.07 | 0.29 | 1.00 | 0.75 | 0.83 | 0.14 | 0.05 | – |
| 30 mW, w = 7 µm, degenerate 300 K | 0.85 | 0.24 | 0.04 | 0.06 | 0.10 | 0.19 | 0.30 | 0.45 | 1.00 |
| 30 mW, w = 7 µm, degenerate 600 K | 0.81 | 0.25 | 0.04 | 0.04 | 0.08 | 0.16 | 0.26 | 0.42 | 1.00 |
| 30 mW, w = 7 µm, classical 1500 K | 0.72 | 0.27 | 0.13 | 0.07 | 0.04 | 0.06 | 0.16 | 0.36 | 1.00 |
| measured +25 V | – | – | – | – | 0.24 | 0.59 | 1.00 | 0.15 | – |
| measured −25 V, X → −X | – | – | 0.32 | 0.20 | 0.33 | 0.52 | 0.45 | 0.15 | – |

0 V maximum relative to the 25 V maximum (size of the even, bias-independent term):
classical 300 K 0.08; classical 1500 K 0.26; degenerate 300 K: 0.16 (2 mW), 0.23 (30 mW);
degenerate 600 K, 30 mW: 0.23; degenerate 300 K, w = 7 µm, 30 mW: 0.15.

Conclusions:

* Fermi-pressure (degenerate) diffusion acts only in the dense core. It sets the size of the
  even term (0 V signal ≈ 0.2 of the biased peak, as large as the hot-carrier bracket) but
  leaves the weak side almost unchanged (0.76 → 0.73). The weak side is set by the diffusion at
  the flanks, where the drift fronts are, i.e. by the carrier temperature: 0.73 (300 K) → 0.62
  (600 K) → 0.38 (1500 K). At 1500 K the gas is no longer degenerate, so "degenerate 1500 K" and
  "classical 1500 K" coincide. Temperature therefore stays a parameter; the physically expected
  T_e is ≈ 600–800 K (0.08 eV excess energy; field heating is small because the field is
  screened in 0.2 ps).
* The peak position follows the spot edge. With w = 3 µm the strong-side peak is at +5 µm and
  gone by +7.5 µm, as measured; with w = 7 µm the fronts sit at ±10 µm and the scan is nearly
  symmetric (0.85 vs 1.00; at X = ±10 the spot tail also reaches the electrodes, which inflates
  dJ/dt there). The measured shape requires an effective 1/e² radius of 3–4 µm; the spot size on
  the sample must be measured (knife edge, or photocurrent vs X with the tip retracted).
* At 30 mW the 3 µm spot is flat-topped by the n_sat cap and its fronts move out to 5–7 µm
  (+7.5 rises to 0.45, +2.5 falls to 0.15), worse than at 2 mW. The cap is a density clip rather
  than bleaching in the FDTD; a bleached absorber would send light deeper rather than wider.
