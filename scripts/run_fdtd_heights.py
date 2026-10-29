"""Pump FDTD with the tip apex at each tapping height. Writes results/fdtd_pump/heights/ratio_h<nm>.npz
(ratio = absorbed energy density with tip / without, same grid as results/fdtd_pump/notip.npz).
Usage: python scripts/run_fdtd_heights.py [heights=10,15,25,45,80,130,200,270,330,370]"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ltemsim.optics_fdtd import FDTDConfig, run

HEIGHTS_NM = (10, 15, 25, 45, 80, 130, 200, 270, 330, 370)


def main(out="results/fdtd_pump/heights", heights=HEIGHTS_NM, ref="results/fdtd_pump/notip.npz", tip_shape="cone"):
    os.makedirs(out, exist_ok=True)
    base = dict(dx=15e-9, x_half=4.0e-6, z_min=-3.0e-6, z_max=2.5e-6, z_source=2.2e-6, tip_shape=tip_shape)
    rn = np.load(ref, allow_pickle=True)
    Qn = rn["Q"]
    summary = {}
    for h in heights:
        fn = os.path.join(out, f"ratio_h{int(h)}.npz")
        if os.path.exists(fn):
            print(f"h = {h} nm: exists, skipping", flush=True)
            continue
        t0 = time.time()
        r = run(FDTDConfig(tip=True, tip_gap=h * 1e-9, **base), verbose=False)
        ratio = np.where(Qn > 1e-3 * Qn.max(), r.Q / np.maximum(Qn, 1e-300), 1.0)
        np.savez_compressed(fn, x=r.x, z=r.z, ratio=ratio, Qtip=r.Q, Qnotip=Qn, absorbed=r.absorbed_energy, launched=r.launched_energy)
        k0 = np.argmin(np.abs(r.z + 0.02e-6)); i0 = np.argmin(np.abs(r.x))
        summary[int(h)] = dict(absorbed_fraction=float(r.absorbed_energy / rn["launched"]), surface_ratio_under_apex=float(ratio[i0, k0]))
        print(f"h = {h} nm: absorbed {summary[int(h)]['absorbed_fraction']:.3f}, ratio under apex {summary[int(h)]['surface_ratio_under_apex']:.2f}, {time.time()-t0:.0f}s", flush=True)
    summary["tip_shape"] = tip_shape
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    kw = {}
    for a in sys.argv[1:]:
        k, v = a.split("=")
        kw[k] = tuple(float(x) for x in v.split(",")) if k == "heights" else v
    main(**kw)
