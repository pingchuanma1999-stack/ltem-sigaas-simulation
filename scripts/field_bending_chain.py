"""Contribution of the electrodes' field-line bending to the near-field signal, evaluated with the single-pulse
chain (route 1): the same runs with the full applied field (E_x + E_z from the finite-volume Poisson solve of the
coplanar gap) and with its vertical component removed. Also writes the applied-field tilt vs x and depth.
Usage: python scripts/field_bending_chain.py [P_avg_mW=2] [out=results/field_bending_chain]"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ltemsim.pulse_dynamics import PulseConfig, PulseSim
from ltemsim.tip_demod import demodulate_pizzuto, band_metric

X_LIST = (-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5)


def main(P_avg_mW=2.0, out="results/field_bending_chain"):
    os.makedirs(out, exist_ok=True)
    res = {}
    for vert in (True, False):
        key = "bent" if vert else "horizontal"
        res[key] = {}
        for X in X_LIST:
            fn = os.path.join(out, f"{key}_X{X:+.1f}.npz")
            if os.path.exists(fn):
                d = np.load(fn); res[key][X] = {k: float(d[k]) for k in ("a1", "a2", "a3", "a4", "far")}
                continue
            cfg = PulseConfig(P_avg_mW=P_avg_mW, x_beam_um=X, degenerate_diffusion=True, applied_vertical=vert, t_end_ps=4.0)
            sim = PulseSim(cfg, verbose=False)
            if vert and X == 0.0:
                # applied field tilt at the surface and at 1 um depth, from the chain's own Poisson solve
                xc, zc = sim.xc * 1e4, sim.zc * 1e4
                k0 = sim.Nzs - 1; k1 = int(np.argmin(np.abs(zc + 1.0))); k5 = int(np.argmin(np.abs(zc + 5.0)))
                tilt = {}
                for xx in (0.0, 2.5, 5.0, 7.5, 10.0, 15.0):
                    i = int(np.argmin(np.abs(xc - xx)))
                    tilt[xx] = {f"z={zc[k]:.2f}um": dict(Ex=float(sim.Ex_app[k, i]), Ez=float(sim.Ez_app[k, i]),
                                                       tilt_deg=float(np.degrees(np.arctan2(sim.Ez_app[k, i], sim.Ex_app[k, i])))) for k in (k0, k1, k5)}
                np.savez_compressed(os.path.join(out, "applied_field.npz"), xc=xc, zc=zc, Ex=sim.Ex_app, Ez=sim.Ez_app)
                json.dump(tilt, open(os.path.join(out, "tilt.json"), "w"), indent=1)
            N = sim.generation_profile()
            t0 = time.time()
            rec = sim.run(N)
            dem = demodulate_pizzuto(rec["t"], rec["heights_nm"], rec["Ez_tip"], rec["n_center"], A_nm=180, h_min_nm=0, harmonics=(1, 2, 3, 4))
            a = {f"a{n}": band_metric(rec["t"], dem[n]) for n in (1, 2, 3, 4)}
            a["far"] = band_metric(rec["t"], np.cumsum(rec["Jx"]) * (rec["t"][1] - rec["t"][0]))
            np.savez_compressed(fn, t=rec["t"], S3=dem[3], Jx=rec["Jx"], Ez_tip=rec["Ez_tip"], heights_nm=rec["heights_nm"], **a)
            res[key][X] = a
            print(f"{key:10s} X={X:+.1f}: a3 {a['a3']:.3e}  ({time.time()-t0:.0f}s)", flush=True)
    summ = {k: {str(X): v for X, v in d.items()} for k, d in res.items()}
    a3b = np.array([res["bent"][X]["a3"] for X in X_LIST]); a3h = np.array([res["horizontal"][X]["a3"] for X in X_LIST])
    summ["ratio_bent_over_horizontal_a3"] = (a3b / a3h).tolist()
    summ["X"] = list(X_LIST)
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print("a3 bent/horizontal:", np.round(a3b / a3h, 4).tolist())


if __name__ == "__main__":
    kw = {}
    for a in sys.argv[1:]:
        k, v = a.split("="); kw[k] = v if k == "out" else float(v)
    main(**kw)
