"""Lateral photo-Dember (even in X, bias-independent) + drift front (odd, ∝ V): one-sidedness from their interference.
Runs X = -5..5 at 25 V and 0 V with cold (300 K) and hot (1500 K) carrier diffusion. Output: dS3/dt RMS, normalised."""
import sys, time; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from ltemsim.pulse_dynamics import PulseConfig, PulseSim
from ltemsim.tip_demod import demodulate_pizzuto, lowpass, amplitude_metrics
for T in (300.0, 1500.0):
    out = {}
    for V in (25.0, 0.0):
        for X in (-5.0, -2.5, 0.0, 2.5, 5.0):
            cfg = PulseConfig(P_avg_mW=2.0, x_beam_um=X, t_end_ps=3.0, T=T, V_bias=V)
            sim = PulseSim(cfg, verbose=False); N = sim.generation_profile(); rec = sim.run(N)
            dem = demodulate_pizzuto(rec["t"], rec["heights_nm"], rec["Ez_tip"], rec["n_center"], A_nm=180, h_min_nm=0, harmonics=(3,))
            pk, rms = amplitude_metrics(rec["t"], lowpass(rec["t"], np.gradient(dem[3], rec["t"])))
            out[(V, X)] = rms
    m = max(out.values())
    print(f"T_carrier = {T:.0f} K (D_n = {sim.D_n:.0f} cm^2/s):")
    for V in (25.0, 0.0):
        print(f"  V = {V:4.0f} V: " + "  ".join(f"X={X:+.1f}: {out[(V,X)]/m:.2f}" for X in (-5.0, -2.5, 0.0, 2.5, 5.0)), flush=True)
