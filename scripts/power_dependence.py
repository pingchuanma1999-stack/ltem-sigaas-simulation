"""THz amplitude vs pump power at fixed beam offset (tip-free near field + FDM route, degenerate diffusion, band-limited metric).
Usage: python scripts/power_dependence.py [X=5] [n_sat=2e19] [degenerate=1]"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ltemsim.pulse_dynamics import PulseConfig, PulseSim
from ltemsim.tip_demod import demodulate_pizzuto, band_metric

def main(X=5.0, n_sat=2e19, powers=(0.25, 0.5, 1, 2, 4, 8, 16, 30, 60, 120), out="results/pulse_chain/power", degenerate=1.0):
    os.makedirs(out, exist_ok=True)
    rows = []
    for P in powers:
        cfg = PulseConfig(P_avg_mW=P, x_beam_um=X, n_sat=n_sat, t_end_ps=4.0, degenerate_diffusion=bool(degenerate))
        sim = PulseSim(cfg, verbose=False)
        N = sim.generation_profile()
        t0 = time.time()
        rec = sim.run(N)
        dem = demodulate_pizzuto(rec["t"], rec["heights_nm"], rec["Ez_tip"], rec["n_center"], A_nm=180, h_min_nm=0, harmonics=(2, 3))
        t = rec["t"]
        # band-limited (0.3-2 THz) spectral amplitude of the detected waveform (∝ second derivative), and of the far field dJ/dt
        a2 = band_metric(t, dem[2], deriv=2); a3 = band_metric(t, dem[3], deriv=2)
        aJ = band_metric(t, np.cumsum(rec["Jx"]) * (t[1] - t[0]), deriv=2)
        sheet = float(np.sum(N * sim.area))     # pairs per cm of y
        rows.append(dict(P_mW=P, N_surface=float(N.max()), pairs_per_cm=sheet, det2_band=a2, det3_band=a3, far_band=aJ,
                         S2_peak=float(np.abs(dem[2]).max())))
        np.savez_compressed(os.path.join(out, f"power_X{X:+.1f}_nsat{n_sat:.0e}_P{P:g}.npz"), t=t, S2=dem[2], S3=dem[3], Jx=rec["Jx"],
                            Ez_tip=rec["Ez_tip"], heights_nm=rec["heights_nm"], n_center=rec["n_center"])
        print(f"P={P:6.2f} mW: N_surf={N.max():.2e} pairs/cm={sheet:.2e}  det2 {a2:.3e} det3 {a3:.3e}  far {aJ:.3e}  ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(out, f"power_dependence_X{X:+.1f}_nsat{n_sat:.0e}.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    return rows

if __name__ == "__main__":
    kw = {}
    for a in sys.argv[1:]:
        k, v = a.split("="); kw[k] = float(v)
    main(**kw)
