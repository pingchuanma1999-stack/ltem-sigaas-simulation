"""Compare saved solutions side by side: field profile, carrier/EL2+ profiles, emission weight, tip scan.
Usage: python scripts/compare_scenarios.py out.png label1=path1.npz label2=path2.npz ..."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ltemsim import Simulation, Solution
from ltemsim.tip import tip_scan, surface_profiles
from run_baseline import load_measured


def main(out, items):
    measured = load_measured(os.path.join(os.path.dirname(__file__), "..", "data", "measured_tip_scan.csv"))
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    rows = []
    for label, path in items:
        sol = Solution.load(path)
        sim = Simulation(sol.params, verbose=False)
        pr = surface_profiles(sim, sol, 0.5)
        ts = tip_scan(sim, sol)
        x = pr["x"]
        axs[0, 0].plot(x, pr["Ex"], label=label)
        axs[0, 1].semilogy(x, np.maximum(pr["n"], 1), label=f"{label}: n")
        axs[0, 1].semilogy(x, np.maximum(pr["NTp"], 1), ls="--", label=f"{label}: EL2+")
        axs[1, 0].plot(x, pr["w_int"] / pr["w_int"].max(), label=label)
        S = np.abs(ts.S); axs[1, 1].plot(ts.X_exp, S / S.max(), label=f"{label} (I={abs(sol.I_left_mA):.3f} mA)")
        i = np.argmax(S)
        rows.append((label, sol.params.P_avg_mW, sol.params.tau_srh_ns, sol.params.S_surface, abs(sol.I_left_mA),
                     pr["n"][np.argmin(abs(x))], pr["Ex"][np.argmin(abs(x + 5))], pr["Ex"][np.argmin(abs(x - 5))], ts.X_exp[i]))
    m = measured[measured["bias_V"] == 25]
    axs[1, 1].errorbar(m["X_um"], m["rms"], yerr=m["rms_err"], fmt="ko", capsize=3, label="measured +25 V")
    axs[0, 0].set_xlim(-20, 20); axs[0, 0].set_ylim(-1, 6); axs[0, 0].set_ylabel("E_x at 0.5 um depth (kV/cm)"); axs[0, 0].axhline(0, color="k", lw=0.5)
    axs[0, 0].text(0.01, 0.95, "bare field at centre: 4.0 kV/cm", transform=axs[0, 0].transAxes, fontsize=8, va="top")
    axs[0, 1].set_xlim(-20, 20); axs[0, 1].set_ylim(1e10, 1e18); axs[0, 1].set_ylabel("density at 0.5 um (cm^-3)")
    axs[1, 0].set_xlim(-12, 12); axs[1, 0].set_ylabel("emission weight n_pulse|E| (norm.)")
    axs[1, 1].set_xlim(-10, 10); axs[1, 1].set_ylabel("|tip signal| (norm.)"); axs[1, 1].set_xlabel("X = spot - tip (um)")
    for ax in axs.ravel():
        ax.legend(fontsize=7); ax.set_xlabel(ax.get_xlabel() or "x (um)")
    fig.savefig(out, dpi=130)
    with open(os.path.splitext(out)[0] + ".md", "w") as fh:
        fh.write("| scenario | P (mW) | tau_srh (ns) | S (cm/s) | I (mA) | n(0) (cm^-3) | E_x(-5) | E_x(+5) (kV/cm) | peak X (um) |\n|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r[0]} | {r[1]:.2g} | {r[2]:.2g} | {r[3]:.1g} | {r[4]:.3f} | {r[5]:.2e} | {r[6]:.2f} | {r[7]:.2f} | {r[8]:+.1f} |\n")
    print(open(os.path.splitext(out)[0] + ".md").read())


if __name__ == "__main__":
    out = sys.argv[1]
    items = [a.split("=", 1) for a in sys.argv[2:]]
    main(out, items)
