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

## Results

See `results/baseline/` (figures + csv) and the "Results" section at the end of
this file, which is updated as runs complete.

## Status / to do

* [x] solver, validation tests
* [x] emission proxy + tip scan vs X
* [ ] calibrate P_avg against the 0.6 mA photocurrent
* [ ] sweeps: P_avg, N_EL2, σ_p, σ_n, spot size, anode side
* [ ] 3-D (finite spot length) estimate of the space-charge field
