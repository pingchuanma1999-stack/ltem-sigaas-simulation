"""Schematic: what a tip (an E_z sensor above the surface) sees from a vertical dipole (classic photo-Dember)
and from a horizontal dipole (drift separation in the lateral bias field). Fields of 2-D line charges."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for name in ("Noto Sans CJK SC", "Noto Serif CJK SC"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name; break
plt.rcParams["axes.unicode_minus"] = False

def field(charges, X, Z):
    Ex = np.zeros_like(X); Ez = np.zeros_like(Z)
    for q, x0, z0 in charges:
        dx, dz = X - x0, Z - z0; r2 = dx ** 2 + dz ** 2 + 1e-6
        Ex += q * dx / r2; Ez += q * dz / r2
    return Ex, Ez

x = np.linspace(-6, 6, 241); z = np.linspace(-2.5, 2.5, 161)
X, Z = np.meshgrid(x, z)
cases = [
    ("垂直偶极（经典 photo-Dember：电子扩散到深处，空穴留在表面）", [(+1, 0.0, -0.3), (-1, 0.0, -1.3)]),
    ("横向偶极（我们的情形：电子整体向阳极移，边缘露出正负前沿）", [(-1, -2.8, -0.5), (+1, 2.8, -0.5)]),
]
fig, axs = plt.subplots(2, 2, figsize=(14, 9), gridspec_kw=dict(width_ratios=[1.6, 1]), constrained_layout=True)
for row, (title, charges) in enumerate(cases):
    Ex, Ez = field(charges, X, Z)
    ax = axs[row, 0]
    ax.streamplot(x, z, Ex, Ez, color="0.55", density=1.4, linewidth=0.8, arrowsize=0.9)
    ax.axhline(0, color="k", lw=1.2); ax.fill_between(x, -2.5, 0, color="#f3e6c8", alpha=0.6, zorder=0)
    for q, x0, z0 in charges:
        ax.scatter([x0], [z0], s=260, c="C3" if q > 0 else "C0", edgecolor="k", zorder=5)
        ax.text(x0, z0, "+" if q > 0 else "−", ha="center", va="center", color="w", fontsize=14, fontweight="bold", zorder=6)
    for xt, lab in ((-2.8, "针尖在 x=−2.8"), (0.0, "针尖在 x=0"), (2.8, "针尖在 x=+2.8")):
        ax.plot([xt, xt], [0.12, 0.9], color="k", lw=3); ax.plot([xt - 0.25, xt, xt + 0.25], [0.9, 0.12, 0.9], color="k", lw=1)
        ax.text(xt, 1.05, lab, ha="center", fontsize=8)
    ax.set_xlim(-6, 6); ax.set_ylim(-2.5, 1.6); ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)")
    ax.set_title(title, fontsize=11)
    ax.text(5.9, -2.3, "GaAs", ha="right", fontsize=10); ax.text(5.9, 1.3, "空气", ha="right", fontsize=10)
    # E_z above the surface vs tip position
    ax2 = axs[row, 1]
    for h, c in ((0.05, "k"), (0.2, "C2"), (0.5, "C4")):
        Ex1, Ez1 = field(charges, x, np.full_like(x, h))
        ax2.plot(x, Ez1 / np.abs(Ez1).max(), color=c, label=f"h = {h*1000:.0f} nm")
    ax2.axhline(0, color="k", lw=0.5); ax2.axvline(0, color="k", lw=0.5)
    ax2.set_xlim(-6, 6); ax2.set_xlabel("针尖位置 x (µm)"); ax2.set_ylabel("针尖处 E_z（归一化）")
    ax2.set_title("针尖感受到的垂直场 vs 位置", fontsize=11); ax2.legend(fontsize=8)
axs[0, 1].text(0, 0.55, "中心最强", ha="center", fontsize=10)
axs[1, 1].text(-2.8, -0.9, "负前沿上方：E_z 向下", ha="center", fontsize=9)
axs[1, 1].text(2.8, 0.9, "正前沿上方：E_z 向上", ha="center", fontsize=9)
axs[1, 1].text(0, 0.15, "中心：E_z = 0", ha="center", fontsize=9)
fig.suptitle("LTEM 的源是正负电荷中心分离；针尖测的是这个偶极在它正下方产生的 E_z，所以偶极的取向决定了扫描图样", fontsize=12)
fig.savefig("results/dipole_geometry.png", dpi=140)
print("saved")
