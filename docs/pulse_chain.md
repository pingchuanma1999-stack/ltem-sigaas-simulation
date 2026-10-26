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
mobility value), τ_p = 0.1 ps, v_sat = 10⁷ cm/s. Generation G(x,z,t) = N(x,z)·g(t) with
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

* Finite-dipole model (Cvitkovic et al. 2007; Hauer et al. 2012 form), R = 40 nm,
  L = 300 nm, sample response β = (ε − 1)/(ε + 1) with ε the Drude permittivity of
  photoexcited GaAs at 1 THz using the instantaneous density under the tip.
* Scattered field E_sc(h, t) = Re α_eff(h, t)·E_loc(h, t); tapping h(τ) = h_min + A(1 + cos Ωτ),
  Ω = 2π·68 kHz, A = 60 nm, h_min = 1 nm; lock-in harmonics S_n(t) = (1/π)∫E_sc cos(nθ)dθ.
  Ω drops out (time scales separated by 10⁷); A, h_min and n matter.
* The detected waveform is taken ∝ d S_n/dt (the static dipole left behind after the
  pulse does not radiate), low-passed at 2.2 THz (detector bandwidth). Amplitude metric:
  RMS of that trace, compared with the measured RMS vs X.

Not included: the tip as a THz antenna (frequency-dependent outcoupling), the tip's
effect on the carrier dynamics, the EO detector response beyond a 2.2 THz low-pass.
