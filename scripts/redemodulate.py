"""Re-demodulate saved chain runs (results/pulse_chain/<run>/X*.npz) with the Pizzuto FDM conventions
(A = 200 nm, z_min = 0, L = 600 nm, g = 0.8 e^{0.06i}, 0.84 THz, harmonics 1-4) and print the X-scan table.
Usage: python scripts/redemodulate.py results/pulse_chain/P2mW_notipmap [A_nm=200] [harmonic=3]"""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ltemsim.tip_demod import demodulate_pizzuto, lowpass, amplitude_metrics


def main(run_dir, A_nm=180.0, h_min_nm=0.0, harmonics=(1, 2, 3, 4), ltem_factor=False):
    rows = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "X*.npz"))):
        d = np.load(f)
        X = float(os.path.basename(f)[1:-4])
        t = d["t"]
        dem = demodulate_pizzuto(t, d["heights_nm"], d["Ez_tip"], d["n_center"], A_nm=A_nm, h_min_nm=h_min_nm,
                                 harmonics=harmonics, ltem_transmission_factor=ltem_factor)
        rows[X] = {}
        for n in harmonics:
            pk, rms = amplitude_metrics(t, lowpass(t, np.gradient(dem[n], t)))
            pk0, rms0 = amplitude_metrics(t, lowpass(t, dem[n]))
            rows[X][f"dS{n}dt_rms"] = rms; rows[X][f"dS{n}dt_peak"] = pk; rows[X][f"S{n}_rms"] = rms0
    Xs = sorted(rows)
    print(f"{run_dir}: A = {A_nm:g} nm, h_min = {h_min_nm:g} nm, LTEM (1-beta) factor = {ltem_factor}")
    print("metric      | " + " | ".join(f"X={X:+.1f}" for X in Xs))
    for key in [f"dS{n}dt_rms" for n in harmonics] + [f"S{n}_rms" for n in harmonics]:
        v = np.array([rows[X][key] for X in Xs]); m = v.max()
        print(f"{key:11s} | " + " | ".join(f"{x/m:6.2f}" for x in v))
    with open(os.path.join(run_dir, f"redemod_A{A_nm:g}_h{h_min_nm:g}{'_ltemfactor' if ltem_factor else ''}.json"), "w") as fh:
        json.dump({str(X): rows[X] for X in Xs}, fh, indent=1)


if __name__ == "__main__":
    kw = {}
    for a in sys.argv[2:]:
        k, v = a.split("="); kw[k] = (v.lower() in ("1", "true")) if k == "ltem_factor" else float(v)
    main(sys.argv[1], **kw)
