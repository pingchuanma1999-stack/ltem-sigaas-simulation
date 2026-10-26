# Model description

## Question

Near-field LTEM on a biased SI-GaAs coplanar gap (40 µm, ±25 V) shows an emission
waveform whose strength peaks with the pump spot displaced ~5 µm from the tip, on
the side set by the bias polarity, with no sign change across the spot. Why is the
emission one-sided, and is a vertical field component involved?

The hypothesis under test: the continuously pumped spot (80 MHz) sits in a
steady state in which its own photocarriers and the EL2 trap charge they create
redistribute the applied field, so that the next pulse sees a strongly
non-uniform field whose geometry is set by the anode/cathode assignment.

## Equations (steady state, 2-D cross-section x–z)

Semiconductor (z < 0), scaled units (De Mari, see `ltemsim/physics.py`):

* Poisson, GaAs and air: ∇·(ε ∇ψ) = −(q/ε₀)(p − n + N_T⁺ + N_D − N_A)
* Electrons: ∇·J_n = q (R_n − G), J_n = −q μ_n n ∇ψ + q D_n ∇n
* Holes: ∇·J_p = q (G − R_p), J_p = −q μ_p p ∇ψ − q D_p ∇p
* EL2 (0/+), occupancy f = N_T⁰/N_T:
  dN_T⁰/dt = c_n n N_T⁺ − e_n N_T⁰ − c_p p N_T⁰ + e_p N_T⁺ = 0
  with e_n = c_n n₁ + σ_n^o Φ, e_p = c_p p₁ + σ_p^o Φ (thermal + optical emission, Φ = local photon flux)
  * R_n = c_n n N_T⁺ − e_n N_T⁰ + B(np − n_i²)
  * R_p = c_p p N_T⁰ − e_p N_T⁺ + B(np − n_i²)
* Generation: G(x,z) = α Φ₀ exp(−2x²/w²) exp(−α|z|), Φ₀ = 2P(1−R)/(π w² hν) — the
  time-averaged rate of the 80 MHz pulse train.
* Mobility: μ(E) = μ₀/(1 + μ₀|E|/v_sat), monotone (GaAs NDR is not modelled).

Boundary conditions: thin Au electrodes on the surface for |x| ≥ 20 µm as ohmic
contacts (ψ = ±V/2 + ψ_eq, n = n_eq, p = p_eq); free GaAs surface with zero
normal current (optionally a surface recombination velocity S); Neumann on the
box sides, bottom and air top. Neither electrode is grounded (floating supply);
the symmetric split ±V/2 is a convention and does not change the fields.

## Discretisation and solver

* Cell-centred finite volumes on a tensor grid (0.125 µm around the spot, 0.1 µm
  near the surface, graded to 2–3 µm far away). Box 120 × (30 + 30) µm: with it the
  bare field at the gap centre is 4.04 kV/cm, within 2 % of the thin-strip value
  V/(πa) = 3.98 kV/cm.
* Scharfetter–Gummel fluxes for the carriers; harmonic permittivity across the
  GaAs/air interface; electrode faces as Dirichlet faces.
* Fully coupled Newton in (ψ, ln n, ln p, f) with elementwise step limiting and
  backtracking. Gummel decoupling does **not** converge for this problem: a 1 %
  change of n ≈ 10⁶ cm⁻³ changes the EL2⁺ charge by ~10¹⁴ cm⁻³, enough to
  redistribute kV/cm fields (the trap-enhanced-field physics is exactly this
  stiffness).
* Pseudo-transient continuation: backward Euler with the EL2 occupancy as a
  dynamic variable, dt growing geometrically from 10⁻¹³ s, following the physical
  sequence (plasma formation ~ps, carrier transit ~µs, trap charging up to s);
  a pure steady-state Newton is attempted periodically and finishes the solve.
* Validation (`tests/test_numerics.py`): bare field vs analytic strip field,
  equilibrium residual, finite-difference check of the full Jacobian, dark
  current.

## Emission proxy and tip response (`ltemsim/tip.py`)

The next pulse's carriers n_pulse ∝ G(x,z) are accelerated by the local
steady-state field; the initial polarisation current is P ∝ n_pulse E. The THz
field radiated in the first ~100 fs (before the plasma screens the local field)
is ∝ dP/dt ∝ P. The tip is a passive E_z sensor at height h = 50 nm: the
quasi-static field of the bound charge −∇·P (electrodes grounded, dielectric
contrast included) is solved with the same Poisson operator and E_z is read at
the apex. The in-plane (P_x) and vertical (P_z) contributions are reported
separately.

Experimental X = spot position relative to the tip = −x_tip in the model frame
(spot at the gap centre). At "+25 V" the + terminal is on the −X electrode
(anode at −x), so the strongest measured signal (X = +5 µm) has the tip on the
anode side of the spot.

## What the model does not include

* Finite spot length along y (a 2-D model is a line source; space-charge
  effects are stronger than for a round spot).
* Pulsed dynamics inside the 12.5 ns period: the steady-state densities are
  time averages. Radiative recombination makes the average nonlinear in the
  pulse density; treat P_avg as an effective parameter and calibrate it against
  the measured photocurrent.
* Gunn-type negative differential mobility, hot-carrier effects, the tip's own
  field enhancement and shadowing, surface states, and the Schottky character
  of the Au/SI-GaAs contacts (treated as ohmic).
* Screening of the emitted field by the photo-plasma (impulse approximation).
