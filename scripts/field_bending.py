"""Pure electrostatics of the coplanar gap: no carriers, no EL2 — only the bending of
the applied field with depth, for a GaAs wafer of finite thickness, with the back
surface either floating (E_z = 0) or held at the bias midpoint (grounded stage).

For each case: E_x, E_z vs depth under the gap, the vertical fraction of the
polarisation of the next pulse (P = n_pulse·E with n_pulse ∝ exp(-alpha z)), and the
tip E_z response vs X (in-plane and vertical parts).

Usage: python scripts/field_bending.py [out_dir]        (default results/field_bending)
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ltemsim import Params, Simulation, Solution
from ltemsim.tip import tip_scan, polarisation
from run_baseline import load_measured

CASES = [
    ("30 um, floating (reference)", dict(depth=30.0, air_height=30.0, back_boundary="neumann")),
    ("350 um, floating back", dict(depth=350.0, back_boundary="neumann")),
    ("350 um, grounded back", dict(depth=350.0, back_boundary="dirichlet")),
    ("500 um, floating back", dict(depth=500.0, back_boundary="neumann")),
    ("500 um, grounded back", dict(depth=500.0, back_boundary="dirichlet")),
]
BOX = dict(x_half=200.0, air_height=200.0, dx_coarse=10.0, dz_coarse_semi=20.0, dz_coarse_air=20.0,
           x_fine=9.0, dx_fine=0.125, dz_fine=0.1, z_fine=2.0)


def bare_field_solution(p: Params):
    sim = Simulation(p, verbose=False)
    V_left, V_right = sim.electrode_voltages(p.V_bias)
    Vl, Vr, Ve, b = sim.scaled_contacts(V_left, V_right)
    psi = sim.pois.solve_linear(b, np.zeros(sim.top.N_all))
    G, phi = sim.generation_arrays(1.0)
    Ns = sim.Ns
    sol = Solution(params=p, grid=sim.grid, psi=psi, n=np.zeros(Ns), p=np.zeros(Ns), NTp=np.zeros(Ns), G=G,
                   gen_factor=1.0, V_left=V_left, V_right=V_right, I_left_mA=0.0, I_right_mA=0.0, log=[], f=np.ones(Ns))
    return sim, sol


def main(out="results/field_bending"):
    os.makedirs(out, exist_ok=True)
    measured = load_measured(os.path.join(os.path.dirname(__file__), "..", "data", "measured_tip_scan.csv"))
    rows = []
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    xs_probe = (0.0, 2.5, 5.0, 7.5)
    for label, kw in CASES:
        p = Params(**{**BOX, **kw})
        sim, sol = bare_field_solution(p)
        g = sim.grid
        Ex, Ez = sol.fields_physical(sim)
        Px, Pz, npulse = polarisation(sim, sol)
        ts = tip_scan(sim, sol)
        dz = g.dz[: g.ks][:, None]
        ratio = {}
        for x in xs_probe:
            i = np.argmin(np.abs(g.xc - x))
            # depth-integrated |P_z| / |P_x| at this x
            ratio[x] = float(np.sum(Pz[:, i] * dz[:, 0]) / np.sum(Px[:, i] * dz[:, 0]))
        i5 = np.argmin(np.abs(g.xc - 5.0)); i0 = np.argmin(np.abs(g.xc))
        ksurf = g.ks - 1
        k1 = np.argmin(np.abs(g.zc[: g.ks] + 1.0))
        k20 = np.argmin(np.abs(g.zc[: g.ks] + 20.0))
        S = ts.S; m = np.abs(S).max()
        Sx = ts.Sx / m; Sz = ts.Sz / m; Sn = S / m
        def at(arr, X):
            o = np.argsort(ts.X_exp); return float(np.interp(X, ts.X_exp[o], arr[o]))
        rows.append(dict(case=label, Ex0_surf=float(Ex[ksurf, i0]), Ex5_surf=float(Ex[ksurf, i5]), Ez5_surf=float(Ez[ksurf, i5]),
                         Ez5_1um=float(Ez[k1, i5]), Ez5_20um=float(Ez[k20, i5]), Ex5_20um=float(Ex[k20, i5]),
                         tilt5_1um=float(np.degrees(np.arctan2(Ez[k1, i5], Ex[k1, i5]))),
                         tilt5_20um=float(np.degrees(np.arctan2(Ez[k20, i5], Ex[k20, i5]))),
                         PzPx=ratio, peakX=float(ts.X_exp[np.argmax(np.abs(S))]),
                         S=[at(Sn, X) for X in (-7.5, -5, -2.5, 0, 2.5, 5, 7.5)],
                         Sz_share_at5=float(at(Sz, 5.0) / at(Sn, 5.0)) if at(Sn, 5.0) != 0 else 0.0,
                         Dz_over_Dx=float(ts.Dz / ts.Dx) if ts.Dx else 0.0))
        # figures
        zc = g.zc[: g.ks]
        sel = zc > -60
        axs[0, 0].plot(np.degrees(np.arctan2(Ez[: g.ks, i5], Ex[: g.ks, i5]))[sel], -zc[sel], label=label)
        axs[0, 1].plot(g.xc, Ez[ksurf, :], label=label)
        axs[1, 0].plot(ts.X_exp, Sn, label=label)
        axs[1, 1].plot(ts.X_exp, np.abs(Sn), label=label)
    m25 = measured[measured["bias_V"] == 25]
    axs[1, 1].errorbar(m25["X_um"], m25["rms"], yerr=m25["rms_err"], fmt="ko", capsize=3, label="measured +25 V")
    axs[0, 0].set_xlabel("field tilt from horizontal at x = 5 um (deg)"); axs[0, 0].set_ylabel("depth (um)")
    axs[0, 0].set_ylim(60, 0); axs[0, 0].axhline(0.83, color="gray", lw=0.5); axs[0, 0].text(axs[0, 0].get_xlim()[0], 1.0, " 1/alpha", fontsize=8, va="top")
    axs[0, 1].set_xlim(-20, 20); axs[0, 1].set_xlabel("x (um)"); axs[0, 1].set_ylabel("E_z at the surface, 0.05 um deep (kV/cm, + up)")
    axs[0, 1].axhline(0, color="k", lw=0.5)
    axs[1, 0].set_xlim(-10, 10); axs[1, 0].set_xlabel("X = spot - tip (um)"); axs[1, 0].set_ylabel("signed tip E_z (norm.)"); axs[1, 0].axhline(0, color="k", lw=0.5)
    axs[1, 1].set_xlim(-10, 10); axs[1, 1].set_xlabel("X = spot - tip (um)"); axs[1, 1].set_ylabel("|tip E_z| (norm.)")
    for ax in axs.ravel():
        ax.legend(fontsize=7)
    fig.suptitle("Bare applied field only (no carriers, no EL2): wafer thickness and back-surface condition")
    fig.savefig(os.path.join(out, "field_bending.png"), dpi=130)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    # markdown table
    lines = ["| case | E_x(0, surf) | E_x(5, surf) | E_z(5, surf) | E_z(5, 1 um) | tilt at 5 um, 1 um / 20 um deep | ∫P_z/∫P_x at x = 0 / 2.5 / 5 / 7.5 | tip peak X | S(X) at -7.5…+7.5 (step 2.5) | vertical share of S at X=5 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['case']} | {r['Ex0_surf']:.2f} | {r['Ex5_surf']:.2f} | {r['Ez5_surf']:.3f} | {r['Ez5_1um']:.3f} | "
                     f"{r['tilt5_1um']:.1f}° / {r['tilt5_20um']:.1f}° | " + " / ".join(f"{100*r['PzPx'][x]:.1f}%" for x in xs_probe) +
                     f" | {r['peakX']:+.1f} | " + " ".join(f"{v:+.2f}" for v in r["S"]) + f" | {100*r['Sz_share_at5']:.0f}% |")
    md = "\n".join(lines)
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("Fields in kV/cm (E_z positive = up); x = position in the gap, spot centred at 0; tip height 50 nm.\n\n" + md + "\n")
    print(md)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/field_bending")
