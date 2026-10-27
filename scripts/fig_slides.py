"""Slide figures (English) for the single-pulse chain, from the saved results.
Writes results/slides/*.png. Usage: python scripts/fig_slides.py [which=all|pump|dynamics|nearfield|xscan|dember|power|dipole]
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ltemsim.tip_demod import demodulate_pizzuto, lowpass, amplitude_metrics
from run_baseline import load_measured

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "slides")
os.makedirs(OUT, exist_ok=True)

# palette (validated for CVD separation): orange, blue, green, amber; ink and grey for reference series
C = dict(orange="#d9541f", blue="#1f6fb2", green="#1b8a5a", amber="#b07a00", ink="#1b2430", grey="#8a8984", muted="#6a7179")
HARM = {1: C["orange"], 2: C["blue"], 3: C["green"], 4: C["amber"]}
MARK = {1: "o", 2: "s", 3: "^", 4: "D"}
for name in ("IBM Plex Sans", "Noto Sans", "DejaVu Sans"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update({"axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9aa0a6", "axes.labelcolor": C["ink"], "xtick.color": C["muted"],
                     "ytick.color": C["muted"], "text.color": C["ink"], "axes.titlecolor": C["ink"],
                     "grid.color": "#e4e1d8", "axes.grid": True, "grid.linewidth": 0.6, "legend.frameon": False,
                     "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10.5,
                     "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white"})
A_NM, H_MIN = 180.0, 0.0
X_LIST = (-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5)
measured = load_measured(os.path.join(ROOT, "data", "measured_tip_scan.csv"))


def load_run(folder, X):
    return np.load(os.path.join(ROOT, "results", "pulse_chain", folder, f"X{X:+.1f}.npz"))


def redemod(d, harmonics=(1, 2, 3, 4)):
    dem = demodulate_pizzuto(d["t"], d["heights_nm"], d["Ez_tip"], d["n_center"], A_nm=A_NM, h_min_nm=H_MIN, harmonics=harmonics)
    return dem


def plot_measured(ax, mirror_neg=True):
    m = measured[measured["bias_V"] == 25]
    ax.errorbar(m["X_um"], m["rms"], yerr=m["rms_err"], fmt="o", color=C["ink"], ms=8, capsize=3, lw=1.5, label="measured +25 V", zorder=6)
    m = measured[measured["bias_V"] == -25]
    xm = -np.asarray(m["X_um"]) if mirror_neg else np.asarray(m["X_um"])
    ax.errorbar(xm, m["rms"], yerr=m["rms_err"], fmt="s", mfc="white", mec=C["ink"], ecolor=C["ink"], ms=8, capsize=3, lw=1.2,
                label="measured −25 V (X → −X)" if mirror_neg else "measured −25 V", zorder=6)


# ----------------------------------------------------------------------------- pump FDTD
def fig_pump():
    tip = np.load(os.path.join(ROOT, "results", "fdtd_pump", "tip.npz"), allow_pickle=True)
    notip = np.load(os.path.join(ROOT, "results", "fdtd_pump", "notip.npz"), allow_pickle=True)
    rat = np.load(os.path.join(ROOT, "results", "fdtd_pump", "ratio.npz"))
    x, z = tip["x"] * 1e6, tip["z"] * 1e6
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    ref = np.percentile(notip["Emax2"], 99.5)
    for ax, d, title in ((axs[0], notip, "(a) pump |E|² envelope, no tip"), (axs[1], tip, "(b) with the PtIr tip (2-D wedge)")):
        E = d["Emax2"].T / ref
        pc = ax.pcolormesh(x, z, np.log10(np.maximum(E, 1e-4)), cmap="Oranges", vmin=-1.6, vmax=0.3, shading="nearest", rasterized=True)
        ax.contour(x, z, d["gaas"].T, levels=[0.5], colors=[C["ink"]], linewidths=1.0)
        if d["tip"].any():
            ax.contourf(x, z, d["tip"].T, levels=[0.5, 1.5], colors=["#4a5260"])
        ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)"); ax.set_title(title); ax.grid(False)
        ax.set_xlim(-4, 4); ax.set_ylim(-2.5, 2.5)
        ax.annotate("", xy=(-2.3, 1.0), xytext=(-3.5, 2.2), arrowprops=dict(arrowstyle="->", lw=2.2, color=C["blue"]))
        ax.text(-3.9, 0.15, "820 nm, 100 fs, 45°\np-pol, toward +x", fontsize=10.5, color=C["blue"], va="bottom")
        ax.text(3.8, -2.3, "GaAs", ha="right", fontsize=11); ax.text(3.8, 0.15, "air", ha="right", fontsize=11)
        ax.text(3.9, 2.3, "source line", ha="right", fontsize=9, color=C["muted"])
    fig.colorbar(pc, ax=axs[:2], label="log₁₀ |E|²max (normalised to the incident beam)", shrink=0.85, pad=0.01)
    ax = axs[2]
    r = np.clip(rat["ratio"].T, 1e-2, 50.0)
    sel_x = np.abs(x) <= 2.0; sel_z = (z >= -1.0) & (z <= 0.5)
    pc = ax.pcolormesh(x[sel_x], z[sel_z], np.log10(r[sel_z][:, sel_x]), cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest", rasterized=True)
    ax.contourf(x[sel_x], z[sel_z], tip["tip"].T[sel_z][:, sel_x], levels=[0.5, 1.5], colors=["#4a5260"])
    ax.axhline(0, color=C["ink"], lw=1)
    fig.colorbar(pc, ax=ax, label="log₁₀ (absorbed energy with tip / without)", shrink=0.85, pad=0.01)
    ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)"); ax.set_title("(c) generation correction under the apex"); ax.grid(False)
    ax.text(-1.9, 0.3, "−x side: shank reflection,\n2–4× standing-wave bands", fontsize=10.5, color=C["ink"], va="center")
    ax.text(1.9, 0.3, "+x side: wedge\nshadow (upper bound)", fontsize=10.5, color=C["ink"], ha="right", va="center")
    fig.savefig(os.path.join(OUT, "sim_pump_fdtd.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- carrier dynamics
def fig_dynamics():
    from ltemsim.pulse_dynamics import PulseConfig, PulseSim, QE
    cfg = PulseConfig(P_avg_mW=2.0, x_beam_um=0.0, t_end_ps=0.6, snapshot_times_ps=(0.15,), degenerate_diffusion=True)
    sim = PulseSim(cfg, verbose=False)
    N = sim.generation_profile()
    rec = sim.run(N)
    xc, zc = rec["xc_um"], rec["zc_um"]
    snap = rec["snapshots"][0.15]
    n, p = snap["n"], snap["p"]
    rho = QE * (p - n)
    dz = sim.dz
    sheet_n = np.sum(n * dz[:, None], axis=0); sheet_p = np.sum(p * dz[:, None], axis=0); sheet_rho = np.sum(rho * dz[:, None], axis=0)
    d0 = load_run("P2mW_notipmap_degen", 0.0)
    fig = plt.figure(figsize=(17, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, :])
    sel = np.abs(xc) < 6; zsel = zc > -1.2
    vmax = np.abs(rho[zsel][:, sel]).max()
    pc = ax.pcolormesh(xc[sel], zc[zsel], rho[zsel][:, sel] / vmax, cmap="RdBu_r", vmin=-1, vmax=1, shading="nearest", rasterized=True)
    cb = fig.colorbar(pc, ax=ax, pad=0.01); cb.set_label("net charge e(p − n), normalised")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)"); ax.grid(False)
    ax.set_title("(a) net space charge 0.15 ps after the pulse — 2 mW, symmetric 3 µm spot, +25 V with the anode at −x, degenerate diffusion")
    ax.annotate("", xy=(-5.3, 0.25), xytext=(-2.3, 0.25), arrowprops=dict(arrowstyle="->", lw=2.5, color=C["blue"]))
    ax.text(-3.8, 0.4, "electrons drift to the anode (fast)", ha="center", color=C["blue"], fontsize=11.5)
    ax.annotate("", xy=(5.3, 0.25), xytext=(2.3, 0.25), arrowprops=dict(arrowstyle="->", lw=2.0, color=C["orange"]))
    ax.text(3.8, 0.4, "holes drift to the cathode (15× slower)", ha="center", color=C["orange"], fontsize=11.5)
    ax.text(-5.9, -1.1, "← anode, +12.5 V at x = −20 µm", fontsize=10.5, va="bottom")
    ax.text(5.9, -1.1, "cathode, −12.5 V at x = +20 µm →", fontsize=10.5, va="bottom", ha="right")
    for xx, txt, col in ((-2.9, "negative front:\nelectron pile-up", C["blue"]), (0, "inside the spot:\nplasma screens, net charge ≈ 0", C["muted"]),
                         (2.9, "positive front:\nelectrons withdrawn, holes left", C["orange"])):
        ax.text(xx, -0.6, txt, color=col, fontsize=10.5, ha="center", bbox=dict(boxstyle="round", fc="white", ec=col, alpha=0.9))
    ax.set_ylim(-1.2, 0.55)
    # (b) sheet densities
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(xc, sheet_n / sheet_n.max(), color=C["blue"], lw=2, label="electrons n(x)")
    ax.plot(xc, sheet_p / sheet_n.max(), color=C["orange"], lw=2, ls="--", label="holes p(x)")
    ax.plot(xc, 0.5 * sheet_rho / np.abs(sheet_rho).max(), color=C["ink"], lw=1.6, label="net (p − n), own scale")
    ax.axhline(0, color=C["grey"], lw=0.6)
    ax.set_xlim(-6, 6); ax.set_xlabel("x (µm)"); ax.set_ylabel("sheet density, normalised"); ax.legend(loc="upper left")
    ax.set_title("(b) along the surface at 0.15 ps: electrons shifted left")
    # (c) field at the spot centre
    t = d0["t"] * 1e12
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, d0["Ex_center"] / 1e3, color=C["ink"], lw=2)
    ax.axhline(0, color=C["grey"], lw=0.6)
    ax.set_xlim(0, 2); ax.set_xlabel("t (ps)"); ax.set_ylabel("E_x at the spot centre (kV/cm)")
    ax.set_title("(c) the applied field is screened within ≈ 0.2 ps")
    ax.text(0.9, 3.0, "4 kV/cm applied\n→ ≈ 0 after the plasma forms", fontsize=10.5, color=C["muted"])
    # (d) current and its derivative (far-field reference)
    ax = fig.add_subplot(gs[1, 2])
    Jx = d0["Jx"]; dJ = lowpass(d0["t"], np.gradient(Jx, d0["t"]))
    ax.plot(t, Jx / np.abs(Jx).max(), color=C["green"], lw=2, label="J_x (normalised)")
    ax.plot(t, dJ / np.abs(dJ).max(), color=C["amber"], lw=2, ls="--", label="dJ_x/dt, 2.2 THz low-pass (far-field reference)")
    ax.axhline(0, color=C["grey"], lw=0.6)
    ax.set_xlim(0, 2); ax.set_xlabel("t (ps)"); ax.set_ylabel("normalised"); ax.legend(loc="upper right")
    ax.set_title("(d) one current surge, then a frozen dipole")
    fig.savefig(os.path.join(OUT, "sim_dynamics.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- near field + demodulation
def fig_nearfield():
    d5 = load_run("P2mW_notipmap_degen", 5.0)
    dem5 = redemod(d5)
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True)
    ax = axs[0]
    t = d5["t"] * 1e12
    for tt, col in ((0.1, C["orange"]), (0.2, C["blue"]), (0.5, C["green"]), (1.0, C["amber"]), (3.0, C["ink"])):
        i = np.argmin(np.abs(t - tt))
        ax.semilogx(d5["heights_nm"], d5["Ez_tip"][i], color=col, lw=2, label=f"{tt:g} ps")
    ax.axvspan(H_MIN, H_MIN + 2 * A_NM, color="#ece8de", alpha=0.7, lw=0)
    y0, y1 = ax.get_ylim(); ax.set_ylim(y0, y1 + 0.16 * (y1 - y0))
    ax.text(1.3, y1 + 0.08 * (y1 - y0), "shaded: tapping range h = 0–360 nm (A = 180 nm)", fontsize=10.5, color=C["muted"], va="center")
    ax.set_xlabel("height above the surface h (nm)"); ax.set_ylabel("E_z,loc (V/cm), tip absent")
    ax.set_title("(a) THz near field above the tip position, X = +5 µm"); ax.legend(title="time after the pulse", loc="lower left")
    ax = axs[1]
    s1 = np.abs(dem5[1]).max()
    for nh in (1, 2, 3, 4):
        ax.plot(t, dem5[nh] / s1, color=HARM[nh], lw=2, label=f"n = {nh}")
    ax.axhline(0, color=C["grey"], lw=0.6)
    ax.set_xlim(0, 3); ax.set_xlabel("t (ps)"); ax.set_ylabel("S_n(t) / max |S₁|")
    ax.set_title("(b) demodulated scattered field, harmonics 1–4 (A = 180 nm)"); ax.legend(loc="upper right")
    ax = axs[2]
    for X, col, ls in ((5.0, C["orange"], "-"), (2.5, C["blue"], "-"), (0.0, C["ink"], "-"), (-2.5, C["green"], "--"), (-5.0, C["amber"], "--")):
        d = load_run("P2mW_notipmap_degen", X)
        dem = redemod(d, harmonics=(3,))
        y = lowpass(d["t"], np.gradient(dem[3], d["t"]))
        ax.plot(d["t"] * 1e12, y / 1e12, color=col, lw=2, ls=ls, label=f"X = {X:+.1f} µm")
    ax.axhline(0, color=C["grey"], lw=0.6)
    ax.set_xlim(0, 2.5); ax.set_xlabel("t (ps)"); ax.set_ylabel("dS₃/dt (V cm⁻¹ ps⁻¹), 2.2 THz low-pass")
    ax.set_title("(c) detected waveform ∝ dS₃/dt vs beam position"); ax.legend(loc="upper right")
    fig.savefig(os.path.join(OUT, "sim_nearfield_demod.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- X scan
def xscan_table(folder):
    out = {nh: [] for nh in (1, 2, 3, 4)}; ff = []
    for X in X_LIST:
        d = load_run(folder, X)
        dem = redemod(d)
        for nh in (1, 2, 3, 4):
            out[nh].append(amplitude_metrics(d["t"], lowpass(d["t"], np.gradient(dem[nh], d["t"])))[1])
        ff.append(amplitude_metrics(d["t"], lowpass(d["t"], np.gradient(d["Jx"], d["t"])))[1])
    return {nh: np.array(v) for nh, v in out.items()}, np.array(ff)


def summary_rms(folder, key="dS3_dt"):
    """RMS vs X from a run's summary.json (A = 180 nm, h_min = 0, harmonics 1-4 as run)."""
    d = json.load(open(os.path.join(ROOT, "results", "pulse_chain", folder, "summary.json")))
    return np.array(d["X"]), np.array(d["rms"][key]), np.array(d["rms"]["dJx_dt"])


def fig_xscan():
    fig, axs = plt.subplots(1, 2, figsize=(17, 6.8), constrained_layout=True, sharey=True)
    fig.get_layout_engine().set(rect=(0, 0.13, 1, 0.87))
    tables = {}
    # (a) 3 um spot, 2 mW: classical (old) vs degenerate diffusion, harmonics 1-4
    ax = axs[0]
    Xs = np.array(X_LIST)
    tab_old, ff_old = xscan_table("P2mW_notipmap")
    tables["P2mW_notipmap_classical"] = dict(rms={str(nh): v.tolist() for nh, v in tab_old.items()}, dJx_dt=ff_old.tolist(), X=list(X_LIST))
    ax.plot(Xs, tab_old[3] / tab_old[3].max(), color=C["grey"], lw=1.8, ls="--", marker="^", ms=7, mfc="white",
            label="2 mW, w = 3 µm: classical D = µkT/q, n = 3")
    d = json.load(open(os.path.join(ROOT, "results", "pulse_chain", "P2mW_notipmap_degen", "summary.json")))
    tables["P2mW_notipmap_degen"] = d["rms"]
    for nh in (1, 2, 3, 4):
        v = np.array(d["rms"][f"dS{nh}_dt"])
        ax.plot(d["X"], v / v.max(), color=HARM[nh], lw=2, marker=MARK[nh], ms=7, mfc="white", label=f"2 mW, w = 3 µm: degenerate (Fermi-pressure) diffusion, n = {nh}")
    plot_measured(ax)
    ax.set_title("(a) 3 µm spot, 2 mW: Fermi-pressure diffusion changes little")
    # (b) 30 mW: spot size 3 um vs 7 um, 3rd harmonic
    ax = axs[1]
    for folder, col, mk, ls, lab in (("P30mW_notipmap_degen", C["orange"], "o", "-", "30 mW, w = 3 µm: degenerate, 300 K"),
                                     ("P30mW_w7_degen", C["blue"], "s", "-", "30 mW, w = 7 µm: degenerate, 300 K"),
                                     ("P30mW_w7_T1500", C["amber"], "D", "--", "30 mW, w = 7 µm: classical, 1500 K")):
        try:
            X, r, ff = summary_rms(folder)
        except FileNotFoundError:
            continue
        tables[folder] = dict(X=X.tolist(), dS3_dt=r.tolist(), dJx_dt=ff.tolist())
        ax.plot(X, r / r.max(), color=col, lw=2, ls=ls, marker=mk, ms=7, mfc="white", label=lab + ", n = 3")
    plot_measured(ax)
    ax.set_title("(b) 30 mW: the peak follows the spot edge — 1/e² radius 3 µm vs 7 µm")
    for ax in axs:
        ax.axvline(0, color=C["grey"], lw=0.6); ax.set_xlim(-11, 11); ax.set_ylim(0, 1.15)
        ax.set_xlabel("X = beam position relative to the tip (µm);  X > 0: tip on the anode side of the spot")
    axs[0].set_ylabel("RMS of the detected waveform, normalised")
    h0, l0 = axs[0].get_legend_handles_labels(); h1, l1 = axs[1].get_legend_handles_labels()
    keep = [i for i, l in enumerate(l1) if l.startswith("30 mW")]
    fig.legend(h0 + [h1[i] for i in keep], l0 + [l1[i] for i in keep], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=10, columnspacing=1.0)
    with open(os.path.join(OUT, "xscan_A180.json"), "w") as fh:
        json.dump(tables, fh, indent=1)
    fig.savefig(os.path.join(OUT, "sim_xscan.png"), dpi=150)
    plt.close(fig)
    return tables


# ----------------------------------------------------------------------------- diffusion model / Dember interference
DEMBER = {  # from scripts/dember_interference.py (dS3/dt RMS, A = 180 nm, 2 mW, 3 um), normalised to the 25 V maximum of each temperature
    (300, 25): [0.75, 0.46, 0.02, 0.54, 1.00], (300, 0): [0.08, 0.07, 0.03, 0.07, 0.08],
    (1500, 25): [0.37, 0.29, 0.07, 0.52, 1.00], (1500, 0): [0.26, 0.16, 0.08, 0.16, 0.28]}
DX = [-5, -2.5, 0, 2.5, 5]


def fig_dember():
    fig, axs = plt.subplots(1, 2, figsize=(17, 5.8), constrained_layout=True, sharey=True)
    # (a) 25 V, 3 um, 2 mW: the weak side vs the diffusion model
    ax = axs[0]
    ax.plot(DX, DEMBER[(300, 25)], color=C["grey"], lw=1.6, ls=":", marker="o", ms=7, mfc="white", label="classical, 300 K (D_n = 68 cm²/s)")
    for folder, col, mk, ls, lab in (("P2mW_notipmap_degen", C["blue"], "^", "-", "degenerate, 300 K (D_n ≈ 450 cm²/s in the core)"),
                                     ("P2mW_notipmap_degen_T600", C["green"], "s", "-", "degenerate, 600 K"),
                                     ("P2mW_notipmap_degen_T1500", C["orange"], "D", "-", "degenerate, 1500 K (≈ classical 1500 K)")):
        try:
            X, r, _ = summary_rms(folder)
        except FileNotFoundError:
            continue
        ax.plot(X, r / r.max(), color=col, lw=2, ls=ls, marker=mk, ms=7, mfc="white", label=lab)
    plot_measured(ax)
    ax.set_title("(a) 25 V, 3 µm, 2 mW: the weak side is set by the flank diffusion, i.e. the temperature")
    ax.legend(loc="upper left", fontsize=10)
    # (b) 0 V: the even term alone, normalised to the same case's 25 V maximum
    ax = axs[1]
    ax.plot(DX, DEMBER[(300, 0)], color=C["grey"], lw=1.6, ls=":", marker="o", ms=7, mfc="white", label="classical, 300 K, 2 mW")
    ax.plot(DX, DEMBER[(1500, 0)], color=C["amber"], lw=1.8, ls="--", marker="D", ms=7, mfc="white", label="classical, 1500 K, 2 mW")
    notes = []
    for f25, f0, col, mk, lab in (("P2mW_notipmap_degen", "P2mW_notipmap_degen_0V", C["blue"], "^", "degenerate, 300 K, 2 mW"),
                                  ("P30mW_notipmap_degen", "P30mW_notipmap_degen_0V", C["orange"], "s", "degenerate, 300 K, 30 mW"),
                                  ("P30mW_notipmap_degen_T600", "P30mW_notipmap_degen_T600_0V", C["green"], "v", "degenerate, 600 K, 30 mW")):
        try:
            X25, r25, _ = summary_rms(f25); X0, r0, _ = summary_rms(f0)
        except FileNotFoundError:
            continue
        ax.plot(X0, r0 / r25.max(), color=col, lw=2, marker=mk, ms=7, mfc="white", label=lab)
        notes.append(f"{lab}: {r0.max() / r25.max():.2f}")
    ax.set_title("(b) 0 V: the even term alone — symmetric; its size relative to the 25 V peak is the test")
    ax.legend(loc="upper left", fontsize=10)
    ax.text(7.8, 1.12, "0 V maximum / 25 V maximum\n" + "\n".join(["classical 300 K, 2 mW: 0.08", "classical 1500 K, 2 mW: 0.26"] + notes),
            ha="right", va="top", fontsize=10, color=C["muted"])
    for ax in axs:
        ax.axvline(0, color=C["grey"], lw=0.6); ax.set_xlim(-8, 8); ax.set_ylim(0, 1.15)
        ax.set_xlabel("X = beam position relative to the tip (µm)")
    axs[0].set_ylabel("dS₃/dt RMS, normalised to each case's 25 V maximum")
    fig.savefig(os.path.join(OUT, "sim_dember.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- power dependence
def fig_power():
    rows = {}
    for key, fn in (("2e19", "power_dependence_X+5.0_nsat2e+19.json"), ("5e18", "power_dependence_X+5.0_nsat5e+18.json")):
        rows[key] = json.load(open(os.path.join(ROOT, "results", "pulse_chain", fn)))
    fig, axs = plt.subplots(1, 2, figsize=(17, 5.6), constrained_layout=True)
    ax = axs[0]
    for key, col, mk, lab in (("2e19", C["orange"], "o", "near field, n_sat = 2×10¹⁹ cm⁻³"), ("5e18", C["blue"], "s", "near field, n_sat = 5×10¹⁸ cm⁻³")):
        P = np.array([r["P_mW"] for r in rows[key]]); y = np.array([r["dS2dt_rms"] for r in rows[key]])
        ax.semilogx(P, y / y.max(), color=col, lw=2, marker=mk, ms=7, mfc="white", label=lab)
    P = np.array([r["P_mW"] for r in rows["2e19"]]); y = np.array([r["dJxdt_rms"] for r in rows["2e19"]])
    ax.semilogx(P, y / y.max(), color=C["grey"], lw=1.6, ls=":", marker="x", ms=7, label="far-field reference dJ_x/dt")
    ax.axvline(16, color=C["green"], lw=1.2, ls="--"); ax.text(15, 0.12, "model: peak ≈ 16 mW,\nthen a slow decline", color=C["green"], fontsize=11, ha="right")
    ax.text(0.27, 1.28, "both n_sat curves coincide: the saturation is screening of the bias field\n(dipole limit p_sat = ε₀ε_r E), not absorption saturation", fontsize=10.5, color=C["muted"], va="top")
    ax.axvspan(30, 50, color="#ece8de", alpha=0.8, lw=0); ax.text(31, 0.3, "experiment:\nsaturates ≈ 40 mW", color=C["muted"], fontsize=11)
    ax.set_xlabel("average pump power (mW), 3 µm spot, 80 MHz"); ax.set_ylabel("dS₂/dt RMS at X = +5 µm, normalised")
    ax.set_title("(a) THz amplitude vs pump power"); ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.28)); ax.set_ylim(0, 1.32)
    ax = axs[1]
    for key, col, mk, lab in (("2e19", C["orange"], "o", "n_sat = 2×10¹⁹ cm⁻³"), ("5e18", C["blue"], "s", "n_sat = 5×10¹⁸ cm⁻³")):
        P = np.array([r["P_mW"] for r in rows[key]]); Ns = np.array([r["N_surface"] for r in rows[key]])
        ax.loglog(P, Ns, color=col, lw=2, marker=mk, ms=7, mfc="white", label=lab)
    Nlin = rows["2e19"][0]["N_surface"] / rows["2e19"][0]["P_mW"] * P * 1.02
    ax.loglog(P, Nlin, color=C["grey"], lw=1.4, ls="--", label="linear absorption (no saturation)")
    ax.set_xlabel("average pump power (mW)"); ax.set_ylabel("peak pair density per pulse at the surface (cm⁻³)")
    ax.set_title("(b) peak pair density per pulse vs power (soft cap n_sat)"); ax.legend(loc="upper left")
    fig.savefig(os.path.join(OUT, "sim_power.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- dipole geometry (English)
def fig_dipole():
    def field(charges, X, Z):
        Ex = np.zeros_like(X); Ez = np.zeros_like(Z)
        for q, x0, z0 in charges:
            r2 = (X - x0) ** 2 + (Z - z0) ** 2 + 0.02
            Ex += q * (X - x0) / r2; Ez += q * (Z - z0) / r2
        return Ex, Ez
    x = np.linspace(-6, 6, 241); z = np.linspace(-2.5, 2.5, 161)
    X, Z = np.meshgrid(x, z)
    cases = [("vertical dipole (classic photo-Dember: electrons diffuse down, holes stay near the surface)", [(+1, 0.0, -0.3), (-1, 0.0, -1.3)]),
             ("lateral dipole (our case: the electron cloud shifts toward the anode, exposing ± fronts)", [(-1, -2.8, -0.5), (+1, 2.8, -0.5)])]
    fig, axs = plt.subplots(2, 2, figsize=(17, 8.6), gridspec_kw=dict(width_ratios=[1.6, 1]), constrained_layout=True)
    for row, (title, charges) in enumerate(cases):
        Ex, Ez = field(charges, X, Z)
        ax = axs[row, 0]
        ax.streamplot(x, z, Ex, Ez, color="0.6", density=1.4, linewidth=0.8, arrowsize=0.9)
        ax.axhline(0, color=C["ink"], lw=1.2); ax.fill_between(x, -2.5, 0, color="#f3e6c8", alpha=0.6, zorder=0)
        for q, x0, z0 in charges:
            ax.scatter([x0], [z0], s=300, c=C["orange"] if q > 0 else C["blue"], edgecolor=C["ink"], zorder=5)
            ax.text(x0, z0, "+" if q > 0 else "−", ha="center", va="center", color="w", fontsize=15, fontweight="bold", zorder=6)
        for xt, lab in ((-2.8, "tip at x = −2.8"), (0.0, "tip at x = 0"), (2.8, "tip at x = +2.8")):
            ax.plot([xt, xt], [0.12, 0.9], color=C["ink"], lw=3); ax.plot([xt - 0.25, xt, xt + 0.25], [0.9, 0.12, 0.9], color=C["ink"], lw=1)
            ax.text(xt, 1.05, lab, ha="center", fontsize=9.5)
        ax.set_xlim(-6, 6); ax.set_ylim(-2.5, 1.6); ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)"); ax.grid(False)
        ax.set_title(title)
        ax.text(5.9, -2.3, "GaAs", ha="right", fontsize=11); ax.text(5.9, 1.3, "air", ha="right", fontsize=11)
        ax2 = axs[row, 1]
        for h, col in ((0.05, C["ink"]), (0.2, C["green"]), (0.5, C["amber"])):
            Ex1, Ez1 = field(charges, x, np.full_like(x, h))
            ax2.plot(x, Ez1 / np.abs(Ez1).max(), color=col, lw=2, label=f"h = {h*1000:.0f} nm")
        ax2.axhline(0, color=C["grey"], lw=0.6); ax2.axvline(0, color=C["grey"], lw=0.6)
        ax2.set_xlim(-6, 6); ax2.set_xlabel("tip position x (µm)"); ax2.set_ylabel("E_z at the tip, normalised")
        ax2.set_title("vertical field the tip senses vs its position"); ax2.legend()
    axs[0, 1].text(0.3, 0.6, "strongest at the centre", fontsize=10.5)
    axs[1, 1].text(-2.8, -0.9, "above the − front: E_z down", ha="center", fontsize=10)
    axs[1, 1].text(2.8, 0.9, "above the + front: E_z up", ha="center", fontsize=10)
    axs[1, 1].text(0, 0.15, "centre: E_z = 0", ha="center", fontsize=10)
    fig.savefig(os.path.join(OUT, "sim_dipole_geometry.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- steady-state DD + EL2 (why it fails)
def fig_steady():
    import csv
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "baseline", "profiles_depth0.5um.csv"))))
    x = np.array([float(r["x_um"]) for r in rows]); Ex = np.array([float(r["Ex_kVcm"]) for r in rows])
    n = np.array([float(r["n_cm3"]) for r in rows]); p = np.array([float(r["p_cm3"]) for r in rows]); el2 = np.array([float(r["EL2plus_cm3"]) for r in rows])
    scan = list(csv.DictReader(open(os.path.join(ROOT, "results", "baseline", "tip_scan.csv"))))
    Xs = np.array([float(r["X_um"]) for r in scan]); S = np.abs(np.array([float(r["S"]) for r in scan]))
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    ax = axs[0]
    ax.plot(x, Ex, color=C["orange"], lw=2, label="steady state, 2 mW CW-equivalent pumping")
    ax.axhline(4.0, color=C["grey"], lw=1.4, ls="--", label="bare electrode field, 4.0 kV/cm")
    ax.axhline(0, color=C["grey"], lw=0.6)
    ax.set_xlim(-20, 20); ax.set_ylim(-1, 8); ax.set_xlabel("x (µm), anode at −20, cathode at +20"); ax.set_ylabel("E_x at 0.5 µm depth (kV/cm)")
    ax.set_title("(a) the bias field collapses inside the gap"); ax.legend(loc="upper left")
    ax.text(0, 1.2, "the 25 V drop moves into an\nEL2⁺ layer at the cathode contact", ha="center", fontsize=10.5, color=C["muted"])
    ax = axs[1]
    ax.semilogy(x, n, color=C["blue"], lw=2, label="electrons n")
    ax.semilogy(x, p, color=C["orange"], lw=2, ls="--", label="holes p")
    ax.semilogy(x, el2, color=C["green"], lw=2, ls=":", label="ionised traps EL2⁺")
    ax.set_xlim(-20, 20); ax.set_ylim(1e11, 1e18); ax.set_xlabel("x (µm)"); ax.set_ylabel("density at 0.5 µm depth (cm⁻³)")
    ax.set_title("(b) a 10¹⁷ cm⁻³ photo-plasma fills the whole gap"); ax.legend(loc="center left")
    ax = axs[2]
    ax.plot(Xs, S / S.max(), color=C["orange"], lw=2, label="model |tip signal|, n_pulse·E")
    plot_measured(ax)
    ax.axvline(0, color=C["grey"], lw=0.6); ax.set_xlim(-10, 10); ax.set_ylim(0, 1.15)
    ax.set_xlabel("X = beam position relative to the tip (µm)"); ax.set_ylabel("amplitude, normalised")
    ax.set_title("(c) the predicted scan is symmetric; the data are not"); ax.legend(loc="upper left")
    fig.savefig(os.path.join(OUT, "sim_steady_state.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1].split("=")[-1] if len(sys.argv) > 1 else "all"
    todo = dict(steady=fig_steady, pump=fig_pump, dynamics=fig_dynamics, nearfield=fig_nearfield, xscan=fig_xscan, dember=fig_dember, power=fig_power, dipole=fig_dipole)
    for k, f in todo.items():
        if which in ("all", k):
            f(); print("saved", k, flush=True)
