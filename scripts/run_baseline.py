"""Baseline run: default parameters (25 V, anode at -x, 2 mW, w = 3 um). Writes results/baseline/."""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ltemsim import Params, Simulation
from ltemsim.tip import tip_scan, surface_profiles
from ltemsim.plots import fig_maps, fig_profiles, fig_tip_scan, fig_polarisation_map

def load_measured(path):
    import csv
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append((float(r["bias_V"]), float(r["X_um"]), float(r["rms"]), float(r["rms_err"])))
    return np.array(rows, dtype=[("bias_V", float), ("X_um", float), ("rms", float), ("rms_err", float)])

def main(out="results/baseline", **overrides):
    os.makedirs(out, exist_ok=True)
    p = Params(**overrides)
    sim = Simulation(p, verbose=True)
    t = time.time()
    sol = sim.run()
    print(f"solved in {time.time()-t:.0f} s; current into -x electrode {sol.I_left_mA:.3f} mA, +x {sol.I_right_mA:.3f} mA "
          f"(unity-gain {p.unity_gain_current_mA:.2f} mA)")
    sol.save(os.path.join(out, "solution.npz"))
    with open(os.path.join(out, "params.json"), "w") as fh:
        fh.write(p.to_json())
    measured = load_measured(os.path.join(os.path.dirname(__file__), "..", "data", "measured_tip_scan.csv"))
    fig_maps(sim, sol, path=os.path.join(out, "maps.png"))
    fig_profiles(sim, sol, path=os.path.join(out, "profiles.png"))
    fig_polarisation_map(sim, sol, path=os.path.join(out, "polarisation.png"))
    fig, ts = fig_tip_scan(sim, sol, measured=measured, path=os.path.join(out, "tip_scan.png"))
    pr = surface_profiles(sim, sol)
    summary = dict(
        I_mA=dict(left=sol.I_left_mA, right=sol.I_right_mA, unity_gain=p.unity_gain_current_mA),
        tip_scan=dict(X=ts.X_exp.tolist(), S=ts.S.tolist(), Sx=ts.Sx.tolist(), Sz=ts.Sz.tolist(), Dx=ts.Dx, Dz=ts.Dz),
        log=sol.log,
    )
    np.savetxt(os.path.join(out, "tip_scan.csv"), np.c_[ts.X_exp, ts.S, ts.Sx, ts.Sz], delimiter=",",
               header="X_um,S,S_from_Px,S_from_Pz", comments="")
    np.savetxt(os.path.join(out, "profiles_depth0.5um.csv"),
               np.c_[pr["x"], pr["Ex"], pr["Ez"], pr["n"], pr["p"], pr["NTp"], pr["f"], pr["w_int"]], delimiter=",",
               header="x_um,Ex_kVcm,Ez_kVcm,n_cm3,p_cm3,EL2plus_cm3,f_neutral,emission_weight_int", comments="")
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    # headline numbers
    S = np.abs(ts.S)
    i = np.argmax(S)
    print(f"tip scan: peak |S| at X = {ts.X_exp[i]:+.1f} um; S(X=0)/max = {np.abs(np.interp(0, ts.X_exp[::-1], ts.S[::-1]))/S.max():.2f}; "
          f"S(+5)/S(-5) = {abs(np.interp(5, ts.X_exp[::-1], ts.S[::-1]))/max(abs(np.interp(-5, ts.X_exp[::-1], ts.S[::-1])),1e-30):.2f}")
    return sim, sol, ts

if __name__ == "__main__":
    kw = {}
    for a in sys.argv[1:]:
        k, v = a.split("=")
        kw[k] = type(getattr(Params(), k))(v) if not isinstance(getattr(Params(), k), bool) else v.lower() in ("1", "true")
    out = kw.pop("out", "results/baseline")
    main(out, **kw)
