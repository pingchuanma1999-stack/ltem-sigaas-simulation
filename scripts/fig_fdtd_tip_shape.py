"""Pump FDTD: straight-rod tip vs 2-D wedge (cone) tip. Generation-correction maps, column-integrated generation vs x,
and the absorbed fraction vs apex height. Writes results/fdtd_pump/fdtd_tip_shape.png.

Inputs: results/fdtd_pump/ratio.npz + tip.npz (cone, apex 20 nm), results/fdtd_pump/heights_rod/ratio_h<nm>.npz (rod),
results/fdtd_pump/heights/summary.json and heights_rod/summary.json (absorbed fractions), summary.json (no tip).
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon

ROOT = os.path.join(os.path.dirname(__file__), "..")
FP = os.path.join(ROOT, "results", "fdtd_pump")
C = dict(orange="#d9541f", blue="#1f6fb2", green="#1b8a5a", amber="#b07a00", ink="#1b2430", grey="#8a8984", muted="#6a7179")
for name in ("IBM Plex Sans", "Noto Sans", "DejaVu Sans"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update({"axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9aa0a6", "axes.labelcolor": C["ink"], "xtick.color": C["muted"],
                     "ytick.color": C["muted"], "text.color": C["ink"], "axes.titlecolor": C["ink"],
                     "grid.color": "#e4e1d8", "axes.grid": True, "grid.linewidth": 0.6, "legend.frameon": False,
                     "font.size": 11.5, "axes.titlesize": 12.5, "axes.labelsize": 11.5, "legend.fontsize": 9.5,
                     "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white"})
R_NM, HALF_ANGLE = 40.0, 15.0


def rod_outline(h_nm, z_top=3.0):
    """Polygon of the rod tip (apex sphere + straight shank of width 2R), µm."""
    R = R_NM * 1e-3; zc = h_nm * 1e-3 + R
    th = np.linspace(np.pi, 2 * np.pi, 60)
    arc = np.stack([R * np.cos(th), zc + R * np.sin(th)], axis=1)       # lower half-circle from (-R, zc) to (+R, zc)
    return np.vstack([[[-R, z_top]], arc, [[R, z_top]]])


def main(out=os.path.join(FP, "fdtd_tip_shape.png"), h_rod=25):
    cone = np.load(os.path.join(FP, "ratio.npz")); tipc = np.load(os.path.join(FP, "tip.npz"), allow_pickle=True)
    rod = np.load(os.path.join(FP, "heights_rod", f"ratio_h{int(h_rod)}.npz"))
    x, z = cone["x"] * 1e6, cone["z"] * 1e6
    sc = json.load(open(os.path.join(FP, "heights", "summary.json"))); sr = json.load(open(os.path.join(FP, "heights_rod", "summary.json")))
    s0 = json.load(open(os.path.join(FP, "summary.json")))

    fig, axs = plt.subplots(2, 2, figsize=(13.5, 10), constrained_layout=True)
    sel_x = np.abs(x) <= 2.5; sel_z = (z >= -1.0) & (z <= 1.0)
    for ax, d, title in ((axs[0, 0], cone, "(a) cone / 2-D wedge tip (15°), apex 20 nm"), (axs[0, 1], rod, f"(b) straight-rod tip (2R = 80 nm), apex {h_rod} nm")):
        r = np.clip(d["ratio"].T, 1e-2, 50.0)
        pc = ax.pcolormesh(x[sel_x], z[sel_z], np.log10(r[sel_z][:, sel_x]), cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest", rasterized=True)
        if d is cone:
            ax.contourf(x[sel_x], z[sel_z], tipc["tip"].T[sel_z][:, sel_x], levels=[0.5, 1.5], colors=["#4a5260"])
        else:
            ax.add_patch(Polygon(rod_outline(h_rod), closed=True, facecolor="#4a5260", edgecolor="none", zorder=4))
        ax.axhline(0, color=C["ink"], lw=1)
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-1.0, 1.0); ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)"); ax.set_title(title); ax.grid(False)
        ax.annotate("", xy=(-1.3, 0.55), xytext=(-2.2, 0.95), arrowprops=dict(arrowstyle="->", lw=2.0, color=C["blue"]))
        ax.text(2.4, -0.95, "GaAs", ha="right", fontsize=11); ax.text(2.4, 0.85, "air", ha="right", fontsize=11)
    fig.colorbar(pc, ax=axs[0, :], label="log₁₀ (absorbed energy with tip / without)", shrink=0.8, pad=0.01)

    # column-integrated generation (with tip / without) vs x
    ax = axs[1, 0]
    gaas = z < 0
    for d, lab, col in ((cone, "cone tip, apex 20 nm", C["blue"]), (rod, f"rod tip, apex {h_rod} nm", C["orange"])):
        col_ratio = d["Qtip"][:, gaas].sum(axis=1) / np.maximum(d["Qnotip"][:, gaas].sum(axis=1), 1e-300)
        ok = d["Qnotip"][:, gaas].sum(axis=1) > 1e-3 * d["Qnotip"][:, gaas].sum(axis=1).max()
        ax.plot(x[ok], col_ratio[ok], "-", color=col, lw=2.0, label=lab)
    ax.axhline(1, color=C["grey"], lw=1, ls="--")
    ax.set_yscale("log"); ax.set_ylim(5e-3, 10); ax.set_xlim(-4, 4)
    ax.set_xlabel("x (µm); beam travels toward +x"); ax.set_ylabel("column generation, with tip / without")
    ax.set_title("(c) depth-integrated generation across the spot"); ax.legend(loc="upper right")
    ax.text(3.9, 0.0085, "+x side: shadow\n(2-D: the tip is a sheet along y)", color=C["ink"], fontsize=10, ha="right", va="bottom")
    ax.text(-3.9, 0.0085, "−x side: reflection\nbands off the shank", color=C["ink"], fontsize=10, ha="left", va="bottom")

    # absorbed fraction vs height
    ax = axs[1, 1]
    hc = sorted(int(k) for k in sc if k.isdigit()); hr = sorted(int(k) for k in sr if k.isdigit())
    ax.plot(hc, [sc[str(h)]["absorbed_fraction"] for h in hc], "s-", color=C["blue"], ms=7, lw=1.8, label="cone tip")
    ax.plot(hr, [sr[str(h)]["absorbed_fraction"] for h in hr], "o-", color=C["orange"], ms=7, lw=2.0, label="rod tip")
    ax.axhline(s0["absorbed_fraction_notip"], color=C["ink"], lw=1.2, ls="--", label=f"no tip ({s0['absorbed_fraction_notip']:.2f})")
    ax.axhline(s0["fresnel_p"], color=C["grey"], lw=1.0, ls=":", label=f"Fresnel 1 − R_p(45°) ({s0['fresnel_p']:.2f})")
    ax.set_xlim(0, 400); ax.set_ylim(0.3, 0.85); ax.set_xlabel("apex height h (nm)"); ax.set_ylabel("absorbed fraction of the launched energy")
    ax.set_title("(d) absorbed fraction vs tapping height"); ax.legend(loc="lower right")
    fig.suptitle("Pump FDTD (820 nm, 45°, p-pol, 3 µm beam): straight-rod tip vs 2-D wedge tip", fontsize=13)
    fig.savefig(out, dpi=140)
    print("figure:", out)
    for h in hr:
        print(f"h = {h:3d} nm: absorbed rod {sr[str(h)]['absorbed_fraction']:.3f}" + (f", cone {sc[str(h)]['absorbed_fraction']:.3f}" if str(h) in sc else "") +
              f"; ratio under apex rod {sr[str(h)]['surface_ratio_under_apex']:.2f}" + (f", cone {sc[str(h)]['surface_ratio_under_apex']:.2f}" if str(h) in sc else ""))


if __name__ == "__main__":
    kw = dict(a.split("=", 1) for a in sys.argv[1:])
    main(**({"h_rod": int(kw["h_rod"])} if "h_rod" in kw else {}))
