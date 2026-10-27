"""THz amplitude vs pump power at fixed beam offset (single-pulse chain). Usage: python scripts/power_dependence.py [X=5] [n_sat=2e19]"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ltemsim.pulse_dynamics import PulseConfig, PulseSim
from ltemsim.tip_demod import demodulate, lowpass, amplitude_metrics

def main(X=5.0, n_sat=2e19, powers=(0.25, 0.5, 1, 2, 4, 8, 16, 30, 60, 120), out="results/pulse_chain"):
    rows = []
    for P in powers:
        cfg = PulseConfig(P_avg_mW=P, x_beam_um=X, n_sat=n_sat, t_end_ps=3.0)
        sim = PulseSim(cfg, verbose=False)
        N = sim.generation_profile()
        t0 = time.time()
        rec = sim.run(N)
        dem = demodulate(rec["t"], rec["heights_nm"], rec["Ez_tip"], rec["n_center"], A_nm=60, h_min_nm=1, harmonics=(2,))
        t = rec["t"]
        pk2, rms2 = amplitude_metrics(t, lowpass(t, np.gradient(dem[2], t)))
        pkJ, rmsJ = amplitude_metrics(t, lowpass(t, np.gradient(rec["Jx"], t)))
        sheet = float(np.sum(N * sim.area))     # pairs per cm of y
        rows.append(dict(P_mW=P, N_surface=float(N.max()), pairs_per_cm=sheet, dS2dt_rms=rms2, dS2dt_peak=pk2, dJxdt_rms=rmsJ))
        print(f"P={P:6.2f} mW: N_surf={N.max():.2e} pairs/cm={sheet:.2e}  dS2/dt rms={rms2:.3e} peak={pk2:.3e}  dJx/dt rms={rmsJ:.3e}  ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"power_dependence_X{X:+.1f}_nsat{n_sat:.0e}.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    return rows

if __name__ == "__main__":
    kw = {}
    for a in sys.argv[1:]:
        k, v = a.split("="); kw[k] = float(v)
    main(**kw)
