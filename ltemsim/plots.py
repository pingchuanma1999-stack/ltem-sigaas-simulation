"""Figures for a solved steady state."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

from .solver import Simulation, Solution
from .tip import tip_scan, surface_profiles, polarisation

C_ANODE = "#c0392b"
C_CATH = "#2471a3"
C_N = "#1f77b4"
C_P = "#d62728"
C_T = "#7d3c98"
C_W = "#2e7d32"


def _electrode_labels(sol: Solution):
    return ("anode (+)", "cathode (−)") if sol.V_left > sol.V_right else ("cathode (−)", "anode (+)")


def fig_maps(sim: Simulation, sol: Solution, xlim=(-20, 20), zlim=(-4, 1), path=None):
    g = sim.grid
    Ex, Ez = sol.fields_physical(sim)
    n, p, NTp = sol.densities_physical(sim)
    V = sol.potential_V(sim)
    X, Z = np.meshgrid(g.xe, g.ze)
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    left_lab, right_lab = _electrode_labels(sol)
    items = [
        (axs[0, 0], V, "potential (V)", None, "RdBu_r"),
        (axs[0, 1], Ex, "E_x (kV/cm)", None, "RdBu_r"),
        (axs[1, 0], Ez, "E_z (kV/cm, + up)", None, "RdBu_r"),
        (axs[1, 1], np.log10(np.maximum(n, 1)), "log10 n (cm^-3)", None, "viridis"),
        (axs[2, 0], np.log10(np.maximum(p, 1)), "log10 p (cm^-3)", None, "viridis"),
        (axs[2, 1], NTp / 1e15, "EL2+ (1e15 cm^-3)", None, "magma"),
    ]
    for ax, data, title, norm, cmap in items:
        if data.shape[0] == g.Nzs:
            Xs, Zs = np.meshgrid(g.xe, g.ze[: g.ks + 1])
            if "E_" in title or "potential" in title:
                pass
            vmax = np.nanmax(np.abs(data[:, (g.xc > xlim[0]) & (g.xc < xlim[1])]))
            if cmap == "RdBu_r":
                pc = ax.pcolormesh(Xs, Zs, data, cmap=cmap, vmin=-vmax, vmax=vmax, shading="flat")
            else:
                pc = ax.pcolormesh(Xs, Zs, data, cmap=cmap, shading="flat")
        else:
            sel = (g.xc > xlim[0]) & (g.xc < xlim[1])
            vmax = np.nanmax(np.abs(data[:, sel]))
            if cmap == "RdBu_r":
                pc = ax.pcolormesh(X, Z, data, cmap=cmap, vmin=-vmax, vmax=vmax, shading="flat")
            else:
                pc = ax.pcolormesh(X, Z, data, cmap=cmap, shading="flat")
        fig.colorbar(pc, ax=ax, shrink=0.9)
        ax.set_xlim(*xlim); ax.set_ylim(*zlim)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(title)
        ax.set_xlabel("x (um)"); ax.set_ylabel("z (um)")
    fig.suptitle(f"{sol.params.V_bias:.0f} V, {left_lab} at −x / {right_lab} at +x, "
                 f"P = {sol.params.P_avg_mW * sol.gen_factor:.2f} mW, w = {sol.params.spot_w_um} um")
    if path:
        fig.savefig(path, dpi=130)
    return fig


def fig_profiles(sim: Simulation, sol: Solution, depth_um=0.5, xlim=(-20, 20), path=None):
    pr = surface_profiles(sim, sol, depth_um)
    x = pr["x"]
    sel = (x > xlim[0]) & (x < xlim[1])
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True, constrained_layout=True)
    ax = axs[0]
    ax.plot(x[sel], pr["Ex"][sel], color="k", label="E_x")
    ax.plot(x[sel], pr["Ez"][sel], color="gray", ls="--", label="E_z (+ up)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel(f"field at depth {pr['depth']:.2f} um (kV/cm)")
    ax.legend()
    ax = axs[1]
    ax.semilogy(x[sel], np.maximum(pr["n"][sel], 1), color=C_N, label="n")
    ax.semilogy(x[sel], np.maximum(pr["p"][sel], 1), color=C_P, label="p")
    ax.semilogy(x[sel], np.maximum(pr["NTp"][sel], 1), color=C_T, label="EL2+")
    ax.set_ylabel("density (cm^-3)")
    ax.legend()
    ax = axs[2]
    w = pr["w_int"]
    ax.plot(x[sel], w[sel] / w.max(), color=C_W, label="n_pulse·|E| (depth-integrated)")
    ax.plot(x[sel], pr["wx_int"][sel] / np.abs(w).max(), color="k", lw=0.8, label="n_pulse·E_x")
    ax.plot(x[sel], pr["wz_int"][sel] / np.abs(w).max(), color="gray", ls="--", lw=0.8, label="n_pulse·E_z")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("emission weight (norm.)")
    ax.set_xlabel("x (um)   [spot centred at 0]")
    ax.legend()
    left_lab, right_lab = _electrode_labels(sol)
    for ax in axs:
        ax.text(0.01, 0.95, f"← {left_lab}", transform=ax.transAxes, va="top", fontsize=8)
        ax.text(0.99, 0.95, f"{right_lab} →", transform=ax.transAxes, va="top", ha="right", fontsize=8)
    if path:
        fig.savefig(path, dpi=130)
    return fig


def fig_tip_scan(sim: Simulation, sol: Solution, measured=None, path=None):
    """|S| vs experimental X, with the mirrored curve for the opposite polarity, and measured points."""
    ts = tip_scan(sim, sol)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    ax = axs[0]
    ax.plot(ts.X_exp, ts.S / np.abs(ts.S).max(), color="k", label="total E_z at apex")
    ax.plot(ts.X_exp, ts.Sx / np.abs(ts.S).max(), color=C_N, lw=0.8, label="from in-plane P")
    ax.plot(ts.X_exp, ts.Sz / np.abs(ts.S).max(), color=C_P, lw=0.8, ls="--", label="from vertical P")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("X = spot position relative to tip (um)")
    ax.set_ylabel("signed tip signal (norm.)")
    ax.set_title(f"model, {sol.params.V_bias:+.0f} V (tip height {ts.h_um} um)")
    ax.legend(fontsize=8)
    ax = axs[1]
    Sn = np.abs(ts.S) / np.abs(ts.S).max()
    V = sol.params.V_bias
    ax.plot(ts.X_exp, Sn, color=C_ANODE, label=f"model {V:+.0f} V")
    ax.plot(-ts.X_exp, Sn, color=C_CATH, ls="--", label=f"model {-V:+.0f} V (mirror)")
    if measured is not None:
        for Vb, col, mk in ((V, C_ANODE, "o"), (-V, C_CATH, "s")):
            m = measured[measured["bias_V"] == Vb]
            if len(m):
                ax.errorbar(m["X_um"], m["rms"], yerr=m["rms_err"], fmt=mk, color=col, capsize=3,
                            label=f"measured {Vb:+.0f} V (RMS)")
    ax.set_xlabel("X = spot position relative to tip (um)")
    ax.set_ylabel("|signal| (norm. to +25 V max)")
    ax.set_xlim(-10, 10)
    ax.set_title("model vs measured")
    ax.legend(fontsize=8)
    if path:
        fig.savefig(path, dpi=130)
    return fig, ts


def fig_polarisation_map(sim: Simulation, sol: Solution, xlim=(-10, 10), zlim=(-3, 0.2), path=None):
    g = sim.grid
    Px, Pz, npulse = polarisation(sim, sol)
    Xs, Zs = np.meshgrid(g.xe, g.ze[: g.ks + 1])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, data, title in ((axs[0], Px, "P_x ∝ n_pulse E_x"), (axs[1], Pz, "P_z ∝ n_pulse E_z")):
        vmax = np.max(np.abs(data))
        pc = ax.pcolormesh(Xs, Zs, data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
        fig.colorbar(pc, ax=ax)
        ax.set_xlim(*xlim); ax.set_ylim(*zlim)
        ax.set_title(title); ax.set_xlabel("x (um)"); ax.set_ylabel("z (um)")
    if path:
        fig.savefig(path, dpi=130)
    return fig
