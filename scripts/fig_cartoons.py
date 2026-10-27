"""Schematic physics cartoons for the slide deck (no data; drawn to scale only loosely). Writes results/slides/cartoon_*.png.
Usage: python scripts/fig_cartoons.py [which=all|distortions|chain|oddeven|tippot|power|lobes]"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch, Ellipse, Circle
from matplotlib import font_manager

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "slides")
os.makedirs(OUT, exist_ok=True)
C = dict(orange="#d9541f", blue="#1f6fb2", green="#1b8a5a", amber="#b07a00", ink="#1b2430", grey="#8a8984", muted="#6a7179",
         gaas="#f3e6c8", gaas_edge="#d8c9a3", electrode="#e0a030", tip="#4a5260", paper="#ffffff")
for name in ("IBM Plex Sans", "Noto Sans", "DejaVu Sans"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update({"axes.unicode_minus": False, "savefig.facecolor": "white", "figure.facecolor": "white", "font.size": 12})


def cross_section(ax, xlim=(-10, 10), zlim=(-4, 3), electrodes=True, gap_edge=None):
    """GaAs below z = 0, air above; optional electrode bars at the gap edges (gap_edge = ±x position)."""
    ax.set_xlim(*xlim); ax.set_ylim(*zlim); ax.set_aspect("auto"); ax.axis("off")
    ax.add_patch(Rectangle((xlim[0], zlim[0]), xlim[1] - xlim[0], -zlim[0], facecolor=C["gaas"], edgecolor="none", zorder=0))
    ax.plot(xlim, [0, 0], color=C["ink"], lw=1.4, zorder=2)
    if electrodes and gap_edge is not None:
        w = (xlim[1] - xlim[0]) * 0.06
        ax.add_patch(Rectangle((xlim[0], -0.08), gap_edge - xlim[0], 0.3, facecolor=C["electrode"], edgecolor="none", zorder=3))
        ax.add_patch(Rectangle((-gap_edge, -0.08), xlim[1] + gap_edge, 0.3, facecolor=C["electrode"], edgecolor="none", zorder=3))
    ax.text(xlim[1] - 0.2, zlim[0] + 0.25, "GaAs", ha="right", va="bottom", fontsize=11, color=C["muted"])
    ax.text(xlim[1] - 0.2, zlim[1] - 0.25, "air", ha="right", va="top", fontsize=11, color=C["muted"])


def tip(ax, x, h=0.35, size=1.6, color=None, label=None, lw=2.2):
    color = color or C["tip"]
    ax.add_patch(Polygon([[x, h], [x - 0.32 * size, h + size], [x + 0.32 * size, h + size]], closed=True, facecolor=color, edgecolor="none", zorder=5))
    if label:
        ax.text(x, h + size + 0.15, label, ha="center", va="bottom", fontsize=11, color=color)


def cloud(ax, x, z, w, hgt, color, alpha=0.55, label=None, lz=None):
    ax.add_patch(Ellipse((x, z), w, hgt, facecolor=color, edgecolor="none", alpha=alpha, zorder=1))
    if label:
        ax.text(x, lz if lz is not None else z, label, ha="center", va="center", fontsize=12, color="white", fontweight="bold", zorder=6)


def arrow(ax, x0, z0, x1, z1, color, lw=2.0, ms=14, zorder=6, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, z0), (x1, z1), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=zorder, linestyle=ls))


def arrow_label(ax, x, z, text, color, fontsize=11, ha="center", va="center"):
    ax.text(x, z, text, ha=ha, va=va, fontsize=fontsize, color=color, zorder=7)


# ---------------------------------------------------------------- 1. three kinds of field distortion
def fig_distortions():
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    # (a) electrodes: bowed field lines over the whole gap, carriers in a thin top layer
    ax = axs[0]
    cross_section(ax, xlim=(-22, 22), zlim=(-18, 6), gap_edge=-20)
    ax.add_patch(Rectangle((-22, -1.0), 44, 1.0, facecolor=C["green"], alpha=0.35, edgecolor="none", zorder=1))
    for d in (1.5, 4, 7.5, 12, 17):
        xs = np.linspace(-20, 20, 100)
        zs = -d * (1 - (xs / 20) ** 2) ** 0.5 * 1.0 - 0.02
        ax.plot(xs, zs, color=C["grey"], lw=1.1, zorder=2)
        i = 55; ax.annotate("", xy=(xs[i + 2], zs[i + 2]), xytext=(xs[i], zs[i]), arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.1, mutation_scale=10))
    ax.text(-20.5, 0.9, "anode (+)", color=C["electrode"], fontsize=11, ha="left")
    ax.text(20.5, 0.9, "cathode (−)", color=C["electrode"], fontsize=11, ha="right")
    ax.text(0, -0.45, "carriers: top 0.9 µm, field tilt ≈ 1°", ha="center", va="center", fontsize=11, color=C["green"], fontweight="bold", zorder=7)
    ax.text(0, -11, "field lines bow on the gap scale\n(40 µm) — far below the carriers", ha="center", fontsize=11.5, color=C["ink"])
    ax.set_title("(a) electrode geometry — static, < 1 % of the signal", fontsize=13, color=C["ink"])
    # (b) within the pulse: displaced electron cloud, ± fronts, E_z above the fronts, tip
    ax = axs[1]
    cross_section(ax, xlim=(-10, 10), zlim=(-4, 3.2))
    ax.add_patch(Ellipse((0, -0.9), 9.0, 2.2, facecolor="#f0c9b0", edgecolor="none", alpha=0.6, zorder=1))   # holes (stay)
    ax.add_patch(Ellipse((-0.7, -0.9), 9.0, 2.2, facecolor="#b9d4ee", edgecolor="none", alpha=0.6, zorder=1))  # electrons, shifted
    cloud(ax, -4.6, -0.9, 1.6, 2.0, C["blue"], alpha=0.8, label="−")
    cloud(ax, 4.2, -0.9, 1.6, 2.0, C["orange"], alpha=0.8, label="+")
    arrow(ax, 1.5, -2.8, -2.5, -2.8, C["blue"], lw=2.2); arrow_label(ax, -0.5, -3.4, "electrons drift to the anode (−x)", C["blue"])
    ax.text(0, -0.9, "inside: screened,\nE ≈ 0", ha="center", va="center", fontsize=11, color=C["ink"], zorder=7)
    for x, dz, col in ((-4.6, -1.0, C["blue"]), (4.2, 1.0, C["orange"])):
        for dx in (-1.0, 1.0):
            z0, z1 = (1.6, 0.4) if dz < 0 else (0.4, 1.6)
            arrow(ax, x + dx, z0, x + dx, z1, col, lw=1.6, ms=11)
    ax.text(-4.6, 2.05, "E_z down", ha="center", fontsize=11, color=C["blue"])
    ax.text(4.2, 2.05, "E_z up", ha="center", fontsize=11, color=C["orange"])
    tip(ax, -4.6, h=0.25, size=1.3, label=None)
    ax.text(-7.6, 2.5, "tip reads E_z\nabove a front", ha="center", fontsize=11, color=C["tip"])
    ax.set_title("(b) the carriers' own field, 0.2 ps — this IS the signal", fontsize=13, color=C["ink"])
    # (c) between pulses: plasma fills the gap, field collapsed
    ax = axs[2]
    cross_section(ax, xlim=(-22, 22), zlim=(-18, 6), gap_edge=-20)
    ax.add_patch(Rectangle((-20, -3.0), 40, 3.0, facecolor="#c9c3e6", alpha=0.6, edgecolor="none", zorder=1))
    ax.text(0, -1.5, "standing plasma ≈ 10¹⁷ cm⁻³ fills the gap", ha="center", va="center", fontsize=11.5, color=C["ink"], zorder=7)
    ax.text(0, -7, "E inside ≈ 0;\nthe 25 V drop at the cathode contact", ha="center", fontsize=11.5, color=C["ink"])
    arrow(ax, 20.5, -0.5, 20.5, -4.5, C["orange"], lw=2.4); ax.text(18.8, -6.2, "voltage\ndrops here", ha="center", fontsize=10.5, color=C["orange"])
    ax.text(-20.5, 0.9, "anode (+)", color=C["electrode"], fontsize=11, ha="left")
    ax.text(20.5, 0.9, "cathode (−)", color=C["electrode"], fontsize=11, ha="right")
    ax.set_title("(c) between pulses (ns–ms) — symmetric, not the signal", fontsize=13, color=C["ink"])
    fig.savefig(os.path.join(OUT, "cartoon_distortions.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 2. the chain
def fig_chain():
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    # stage 1: pump FDTD
    ax = axs[0]
    cross_section(ax, xlim=(-5, 5), zlim=(-2.5, 3.2))
    tip(ax, 0, h=0.12, size=2.6)
    for k in range(5):
        x0 = -4.2 + k * 0.4
        arrow(ax, x0 - 1.2, 3.1, x0 + 1.2, 0.05, "#c0392b", lw=1.6, ms=10, zorder=4)
    ax.text(-1.6, 2.7, "820 nm · 100 fs · 45°", color="#c0392b", fontsize=11, ha="left")
    xs = np.linspace(-3.2, 3.2, 200); ax.fill_between(xs, 0, -0.9 * np.exp(-(xs / 1.8) ** 2), color="#c0392b", alpha=0.35)
    ax.text(0, -1.7, "absorbed-energy map Q(x, z)\nwith the tip in place", ha="center", fontsize=11.5, color=C["ink"])
    ax.set_title("1 · pump: 2-D FDTD", fontsize=13, color=C["ink"])
    # stage 2: carriers + Poisson
    ax = axs[1]
    cross_section(ax, xlim=(-5, 5), zlim=(-2.5, 3.2))
    ax.add_patch(Ellipse((0, -0.6), 5.6, 1.4, facecolor="#f0c9b0", edgecolor="none", alpha=0.6, zorder=1))
    ax.add_patch(Ellipse((-0.5, -0.6), 5.6, 1.4, facecolor="#b9d4ee", edgecolor="none", alpha=0.6, zorder=1))
    cloud(ax, -3.1, -0.6, 1.0, 1.3, C["blue"], alpha=0.8, label="−"); cloud(ax, 2.6, -0.6, 1.0, 1.3, C["orange"], alpha=0.8, label="+")
    arrow(ax, -4.6, 2.4, -1.4, 2.4, C["ink"], lw=1.8); ax.text(-3.0, 2.75, "E_bias", ha="center", fontsize=11)
    ax.text(0, -1.8, "electrons + holes with inertia,\nPoisson every 0.5 fs, 4 ps", ha="center", fontsize=11.5, color=C["ink"])
    ax.text(0, 1.2, "ρ(x, z, t), J(t), near field E_z(h, t)", ha="center", fontsize=11.5, color=C["ink"])
    ax.set_title("2 · carriers: hydrodynamic + Poisson", fontsize=13, color=C["ink"])
    # stage 3: tip + lock-in
    ax = axs[2]
    cross_section(ax, xlim=(-5, 5), zlim=(-2.5, 3.2))
    for h, a in ((0.1, 1.0), (0.7, 0.35), (1.3, 0.15)):
        t_ = Polygon([[0, h], [-0.5, h + 1.6], [0.5, h + 1.6]], closed=True, facecolor=C["tip"], alpha=a, edgecolor="none", zorder=5 - int(h * 2)); ax.add_patch(t_)
    arrow(ax, 1.1, 0.1, 1.1, 1.3, C["grey"], lw=1.4, ms=10); arrow(ax, 1.1, 1.3, 1.1, 0.1, C["grey"], lw=1.4, ms=10)
    ax.text(1.4, 0.7, "tapping: 10 runs\nover 10–370 nm", fontsize=10.5, color=C["grey"], va="center")
    cloud(ax, -2.6, -0.6, 1.0, 1.3, C["blue"], alpha=0.8, label="−")
    tt = np.linspace(-4.5, 4.5, 300); ax.plot(tt, -1.5 + 0.6 * np.exp(-((tt + 0.5) / 1.2) ** 2) * np.sin(2.2 * (tt + 0.5)), color=C["orange"], lw=2)
    ax.text(0, -2.3, "S_n(t) → detected ∝ d²S_n/dt², 0.3–2 THz", ha="center", fontsize=11.5, color=C["ink"])
    ax.text(-3.5, 2.7, "finite-dipole tip, or the tip\nitself in the Poisson solve", fontsize=10.5, color=C["tip"], ha="center")
    ax.set_title("3 · tip + lock-in: harmonics n = 1–4", fontsize=13, color=C["ink"])
    fig.savefig(os.path.join(OUT, "cartoon_chain.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 3. odd + even
def fig_oddeven():
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    X = np.linspace(-8, 8, 400)
    O = np.exp(-((X - 4.5) / 1.6) ** 2) - np.exp(-((X + 4.5) / 1.6) ** 2)
    E = 0.45 * np.exp(-((X) / 3.5) ** 2) + 0.7 * (np.exp(-((X - 4.5) / 1.6) ** 2) + np.exp(-((X + 4.5) / 1.6) ** 2))
    for ax, title, y, col, lab in ((axs[0], "(a) drift front: odd in X, ∝ V", O, C["blue"], "O(X)"),
                                   (axs[1], "(b) even term: photo-Dember, or the tip's DC potential", E, C["green"], "E(X)"),
                                   (axs[2], "(c) sum O + E: one-sided, and −V gives the mirror image", None, None, None)):
        ax.axhline(0, color=C["grey"], lw=0.8); ax.axvline(0, color=C["grey"], lw=0.6)
        ax.set_xlim(-8, 8); ax.set_ylim(-1.5, 2.2); ax.set_xlabel("X = beam position relative to the tip (µm)")
        ax.set_yticks([]); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.spines["left"].set_visible(False)
        ax.set_ylabel("signal (sign arbitrary)")
        ax.set_title(title, fontsize=12.5, color=C["ink"])
        if y is not None:
            ax.plot(X, y, color=col, lw=2.5, label=lab); ax.legend(loc="upper left", frameon=False)
    ax = axs[2]
    ax.plot(X, O + E, color=C["orange"], lw=2.5, label="+V: O + E")
    ax.plot(X, -O + E, color=C["amber"], lw=2.0, ls="--", label="−V: −O + E (mirror)")
    ax.legend(loc="upper left", frameon=False)
    ax.text(4.5, 2.0, "same sign → strong", ha="center", fontsize=11, color=C["orange"])
    ax.text(-4.5, -0.75, "opposite sign → weak", ha="center", fontsize=11, color=C["orange"])
    ax.text(0, -1.3, "centre: E alone, ≠ 0", ha="center", fontsize=11, color=C["green"])
    axs[0].text(4.5, 1.3, "tip on the anode side:\nabove the electron front", ha="center", fontsize=10.5, color=C["blue"])
    axs[0].text(-4.5, -1.4, "tip on the cathode side:\nabove the hole front, opposite sign", ha="center", fontsize=10.5, color=C["blue"])
    axs[1].text(0, 1.5, "keeps its sign when V flips", ha="center", fontsize=11, color=C["green"])
    fig.savefig(os.path.join(OUT, "cartoon_oddeven.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 4. tip DC potential
def fig_tippot():
    fig, axs = plt.subplots(1, 2, figsize=(17, 4.8), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1]))
    for ax, title, Vt in ((axs[0], "(a) tip at the potential of the surface below it → two equal lobes", 0),
                          (axs[1], "(b) tip at one electrode's potential: 12.5 V across ≈ 100 nm → one side wins", 1)):
        cross_section(ax, xlim=(-10, 10), zlim=(-4, 3.4))
        ax.add_patch(Ellipse((0, -0.9), 9.0, 2.2, facecolor="#f0c9b0", edgecolor="none", alpha=0.6, zorder=1))
        ax.add_patch(Ellipse((-0.7, -0.9), 9.0, 2.2, facecolor="#b9d4ee", edgecolor="none", alpha=0.6, zorder=1))
        cloud(ax, -4.6, -0.9, 1.6, 2.0, C["blue"], alpha=0.8, label="−"); cloud(ax, 4.2, -0.9, 1.6, 2.0, C["orange"], alpha=0.8, label="+")
        tip(ax, 0, h=0.3, size=1.6)
        ax.set_title(title, fontsize=12, color=C["ink"])
        if Vt:
            for dx in (-0.9, 0.9):
                arrow(ax, dx, 0.3, dx, -0.2, C["amber"], lw=1.8, ms=11)
            ax.text(1.3, 0.55, "E_DC", fontsize=11, color=C["amber"], ha="left")
            cloud(ax, 0, -0.25, 1.4, 0.5, C["blue"], alpha=0.9, label="−")
            cloud(ax, 0, -1.6, 1.4, 0.5, C["orange"], alpha=0.9, label="+")
            ax.text(0, 2.3, "tip at the anode's potential (grounded electrode);\nthe surface below it sits 12.5 V lower", ha="center", fontsize=11, color=C["amber"])
            ax.text(0, -2.5, "DC field under the apex pulls the photo-carriers apart vertically:\na second, even dipole right under the tip", ha="center", fontsize=11, color=C["ink"])
            ax.text(0, -3.5, "sign set by (V_tip − V_surface), i.e. by the wiring — not by the beam position", ha="center", fontsize=11, color=C["amber"])
        else:
            ax.text(0, 2.3, "tip and surface at the same potential: no DC field under the apex", ha="center", fontsize=11, color=C["grey"])
            ax.text(0, -2.6, "only the lateral fronts are read: equal lobes at ±5 µm, nothing at the centre", ha="center", fontsize=11, color=C["ink"])
    fig.savefig(os.path.join(OUT, "cartoon_tippot.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 5. power: the spot widens, the fronts move away from the tip
def fig_power():
    fig, axs = plt.subplots(1, 2, figsize=(17, 4.6), constrained_layout=True)
    for ax, title, w, flat in ((axs[0], "(a) low power: the fronts sit at the spot edge, ±4 µm — right under the tip", 3.0, False),
                               (axs[1], "(b) high power: the dense core is screened and spreads; the fronts move out to ±7 µm", 3.0, True)):
        cross_section(ax, xlim=(-11, 11), zlim=(-4, 3.2))
        xs = np.linspace(-11, 11, 400)
        prof = np.exp(-2 * (xs / w) ** 2)
        if flat:
            prof = np.minimum(prof * 8, 1.0)
        rgba = np.zeros((1, len(xs), 4)); rgba[..., :3] = (0.42, 0.37, 0.66); rgba[..., 3] = 0.75 * prof
        ax.imshow(rgba, extent=(-11, 11, -2.0, 0), aspect="auto", zorder=1)
        ax.plot(xs, -3.0 + 0.8 * prof, color="#6b5fa8", lw=1.8); ax.text(-10.5, -2.4, "n(x)", fontsize=11, color="#6b5fa8")
        edge = 4.3 if not flat else 7.4
        cloud(ax, -edge, -1.0, 1.4, 1.8, C["blue"], alpha=0.85, label="−"); cloud(ax, edge, -1.0, 1.4, 1.8, C["orange"], alpha=0.85, label="+")
        tip(ax, 4.5, h=0.3, size=1.5)
        ax.text(4.5, 2.1, "tip at 5 µm from the spot centre", ha="center", fontsize=11, color=C["tip"])
        if flat:
            ax.text(0, -1.0, "screened interior: E ≈ 0", ha="center", fontsize=11, color=C["ink"])
            arrow(ax, 4.4, -1.0, 6.3, -1.0, C["orange"], lw=1.6, ms=11); arrow(ax, -4.4, -1.0, -6.3, -1.0, C["blue"], lw=1.6, ms=11)
        ax.set_title(title, fontsize=12, color=C["ink"])
    fig.savefig(os.path.join(OUT, "cartoon_power.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1].split("=")[-1] if len(sys.argv) > 1 else "all"
    todo = dict(distortions=fig_distortions, chain=fig_chain, oddeven=fig_oddeven, tippot=fig_tippot, power=fig_power)
    for k, f in todo.items():
        if which in ("all", k):
            f(); print("saved", k)
