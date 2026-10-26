"""Parameter sweeps.  Usage:  python scripts/sweep.py power|el2|sigma_p|sigma_n|spot|anode [out=results/sweeps]

Each run writes results/<sweep>/<label>/ (figures + csv + summary.json) and a
combined figure results/<sweep>/compare.png: emission profile and |S|(X) for all runs.
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ltemsim import Params, Simulation
from ltemsim.tip import tip_scan, surface_profiles
from ltemsim.plots import fig_maps, fig_profiles, fig_tip_scan
from run_baseline import load_measured

SWEEPS = {
    "power": ("P_avg_mW", [0.5, 1.0, 2.0, 4.0, 8.0]),
    "el2": ("N_T", [5e15, 1.5e16, 3e16]),
    "sigma_p": ("sigma_p", [1e-18, 2e-18, 1e-17, 1e-16]),
    "sigma_n": ("sigma_n", [1e-15, 1e-14, 1e-13]),
    "spot": ("spot_w_um", [2.0, 3.0, 4.5]),
    "anode": ("anode_side", ["-x", "+x"]),
}


def run_one(out, **kw):
    os.makedirs(out, exist_ok=True)
    p = Params(**kw)
    sim = Simulation(p, verbose=False)
    t = time.time()
    sol = sim.run()
    sol.save(os.path.join(out, "solution.npz"))
    ts = tip_scan(sim, sol)
    pr = surface_profiles(sim, sol)
    fig_profiles(sim, sol, path=os.path.join(out, "profiles.png")); plt.close("all")
    fig_maps(sim, sol, path=os.path.join(out, "maps.png")); plt.close("all")
    measured = load_measured(os.path.join(os.path.dirname(__file__), "..", "data", "measured_tip_scan.csv"))
    fig_tip_scan(sim, sol, measured=measured, path=os.path.join(out, "tip_scan.png")); plt.close("all")
    summary = dict(params=kw, I_mA=dict(left=sol.I_left_mA, right=sol.I_right_mA, unity_gain=p.unity_gain_current_mA),
                   converged=[l.get("converged") for l in sol.log], wall_s=time.time() - t,
                   tip_scan=dict(X=ts.X_exp.tolist(), S=ts.S.tolist(), Sx=ts.Sx.tolist(), Sz=ts.Sz.tolist()),
                   profile=dict(x=pr["x"].tolist(), w_int=pr["w_int"].tolist(), Ex=pr["Ex"].tolist(), Ez=pr["Ez"].tolist(),
                                NTp=pr["NTp"].tolist()))
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, default=float)
    print(f"  {out}: I={sol.I_left_mA:.3f} mA, converged={summary['converged']}, {summary['wall_s']:.0f}s", flush=True)
    return summary


def main(name, root="results/sweeps", values=None):
    key, default_values = SWEEPS[name]
    values = default_values if not values else values
    out = os.path.join(root, name)
    os.makedirs(out, exist_ok=True)
    measured = load_measured(os.path.join(os.path.dirname(__file__), "..", "data", "measured_tip_scan.csv"))
    results = []
    for v in values:
        label = f"{key}={v}"
        results.append((label, run_one(os.path.join(out, label.replace('/', '_')), **{key: v})))
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for label, s in results:
        x = np.array(s["profile"]["x"]); w = np.array(s["profile"]["w_int"])
        axs[0].plot(x, w / w.max(), label=label)
        axs[1].plot(x, s["profile"]["NTp"], label=label)
        X = np.array(s["tip_scan"]["X"]); S = np.abs(np.array(s["tip_scan"]["S"]))
        axs[2].plot(X, S / S.max(), label=f"{label} (I={s['I_mA']['left']:.2f} mA)")
    m = measured[measured["bias_V"] == 25]
    axs[2].errorbar(m["X_um"], m["rms"], yerr=m["rms_err"], fmt="ko", capsize=3, label="measured +25 V")
    axs[0].set_xlim(-12, 12); axs[0].set_xlabel("x (um)"); axs[0].set_ylabel("emission weight n_pulse|E| (norm.)")
    axs[1].set_xlim(-25, 25); axs[1].set_yscale("log"); axs[1].set_xlabel("x (um)"); axs[1].set_ylabel("EL2+ at 0.5 um depth (cm^-3)")
    axs[2].set_xlim(-10, 10); axs[2].set_xlabel("X = spot - tip (um)"); axs[2].set_ylabel("|S| (norm.)")
    for ax in axs:
        ax.legend(fontsize=7)
    fig.suptitle(f"sweep: {name}")
    fig.savefig(os.path.join(out, "compare.png"), dpi=130)
    with open(os.path.join(out, "sweep_summary.json"), "w") as fh:
        json.dump({l: {k: v for k, v in s.items() if k not in ("profile", "tip_scan")} for l, s in results}, fh, indent=1, default=float)


if __name__ == "__main__":
    name = sys.argv[1]
    root = "results/sweeps"
    vals = []
    for a in sys.argv[2:]:
        if a.startswith("out="):
            root = a[4:]
        else:
            vals.append(a if name == "anode" else float(a))
    main(name, root, vals)
