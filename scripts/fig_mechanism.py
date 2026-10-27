"""Illustration of the single-pulse mechanism from the model itself: electron front on the anode
side, hole front on the cathode side, the (odd) near field a tip would sense when scanned over the
spot, and how an even term (tip–sample DC potential) turns it one-sided. Writes results/mechanism.png"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ltemsim.pulse_dynamics import PulseConfig, PulseSim, QE, EPS0_CM

for name in ("Noto Sans CJK SC", "Noto Serif CJK SC"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

cfg = PulseConfig(P_avg_mW=2.0, x_beam_um=0.0, t_end_ps=0.6, snapshot_times_ps=(0.08, 0.15, 0.3, 0.5))
sim = PulseSim(cfg, verbose=False)
N = sim.generation_profile()
rec = sim.run(N)
xc, zc = rec["xc_um"], rec["zc_um"]
snap = rec["snapshots"][0.15]
n, p = snap["n"], snap["p"]
rho = QE * (p - n)                                   # C/cm^3
dz = sim.dz                                          # cm
sheet_n = np.sum(n * dz[:, None], axis=0); sheet_p = np.sum(p * dz[:, None], axis=0)
sheet_rho = np.sum(rho * dz[:, None], axis=0)        # C/cm^2
# near field E_z at height h above the surface vs tip position x_t (odd term), from the charge distribution
h = 100e-7
xt = np.linspace(-8, 8, 161)
Ez = np.array([sim.nearfield_Ez(rho * sim.area, np.array([h]), x * 1e-4)[0] for x in xt])
# illustrative even term (schematic): an X-even contribution such as the vertical dipole induced under the
# apex by a tip–sample DC potential; width (4 um) and sign (chosen to favour the anode side) are NOT from the model
even = -0.5 * np.abs(Ez).max() * np.exp(-2 * (xt / 4.0) ** 2)

fig = plt.figure(figsize=(14, 10), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
# ---- (a) net charge map ----
ax = fig.add_subplot(gs[0, :])
sel = (np.abs(xc) < 6); zsel = zc > -1.2
vmax = np.abs(rho[zsel][:, sel]).max()
pc = ax.pcolormesh(xc[sel], zc[zsel], rho[zsel][:, sel] / vmax, cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest")
cb = fig.colorbar(pc, ax=ax, pad=0.01); cb.set_label("净电荷 e(p−n)（归一化）")
ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)")
ax.set_title("(a) 脉冲后 0.15 ps 的净空间电荷（模型，2 mW，阳极在 −x）")
ax.annotate("", xy=(-5.3, 0.25), xytext=(-2.3, 0.25), arrowprops=dict(arrowstyle="->", lw=2.5, color="C0"))
ax.text(-3.8, 0.42, "电子向阳极漂移", ha="center", color="C0", fontsize=11)
ax.annotate("", xy=(5.3, 0.25), xytext=(2.3, 0.25), arrowprops=dict(arrowstyle="->", lw=2.0, color="C3"))
ax.text(3.8, 0.42, "空穴向阴极（慢 15×）", ha="center", color="C3", fontsize=11)
ax.text(-5.9, -1.05, "← 阳极 (+12.5 V, x = −20 µm)", fontsize=10, va="bottom")
ax.text(5.9, -1.05, "阴极 (−12.5 V, x = +20 µm) →", fontsize=10, va="bottom", ha="right")
ax.text(-2.9, -0.55, "负前沿：\n电子堆积", color="navy", fontsize=10, ha="center",
        bbox=dict(boxstyle="round", fc="white", ec="navy", alpha=0.85))
ax.text(2.9, -0.55, "正前沿：\n电子被抽走，留下空穴", color="darkred", fontsize=10, ha="center",
        bbox=dict(boxstyle="round", fc="white", ec="darkred", alpha=0.85))
ax.text(0, -0.55, "光斑内部：\n等离子体屏蔽，净电荷≈0", color="k", fontsize=10, ha="center",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
ax.set_ylim(-1.2, 0.55)
# ---- (b) sheet densities along x ----
ax = fig.add_subplot(gs[1, 0])
ax.plot(xc, sheet_n / sheet_n.max(), color="C0", label="电子面密度 n(x)")
ax.plot(xc, sheet_p / sheet_n.max(), color="C3", ls="--", label="空穴面密度 p(x)")
ax2 = ax.twinx()
ax2.plot(xc, sheet_rho / np.abs(sheet_rho).max(), color="k", lw=1.2, label="净面电荷 (p−n)")
ax2.axhline(0, color="k", lw=0.5)
ax2.set_ylabel("净面电荷（归一化）")
ax.set_xlim(-6, 6); ax.set_xlabel("x (µm)"); ax.set_ylabel("面密度（归一化）")
ax.set_title("(b) 同一时刻沿表面的分布：电子整体向阳极（左）移 → 左负右正，几乎反对称")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
# ---- (c) near field vs tip position: odd + even ----
ax = fig.add_subplot(gs[1, 1])
Ezn = Ez / np.abs(Ez).max()
ax.plot(-xt, Ezn, color="k", lw=2, label="横向前沿的近场 E_z（奇项，模型）")
ax.plot(-xt, even / np.abs(Ez).max(), color="C2", lw=2, ls="--", label="随极性翻转的偶项（例如针尖-样品直流电势产生的垂直偶极，示意）")
ax.plot(-xt, Ezn + even / np.abs(Ez).max(), color="C1", lw=2.5, label="两者之和 → 单侧")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlim(-8, 8); ax.set_xlabel("X = 光斑相对针尖的位置 (µm)"); ax.set_ylabel("针尖处 E_z（h = 100 nm，归一化）")
ax.set_title("(c) 针尖扫描看到的近场：奇项本身左右等大，偶项使它单侧")
ax.legend(fontsize=8, loc="lower left")
ax.text(5.0, -0.95, "X > 0：针尖在阳极侧\n偶项与奇项同号 → 增强", ha="center", fontsize=9, color="C1")
ax.text(-5.0, 0.85, "X < 0：针尖在阴极侧\n偶项与奇项反号 → 削弱", ha="center", fontsize=9, color="C1")
ax.text(0.3, -0.45, "偶项的宽度和符号\n在此为示意", fontsize=8, color="C2")
fig.savefig("results/mechanism.png", dpi=140)
print("saved results/mechanism.png")
