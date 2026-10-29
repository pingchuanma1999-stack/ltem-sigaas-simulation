"""Straight-rod tip vs cone tip in the per-tip-position chain, against the measured ±25 V X scans.

Reads the demodulated summaries written by scripts/run_tip_positions.py (demod_A180_h10.json) for pairs of runs
that differ only in PulseConfig.tip_shape, prints the band-limited amplitude tables (n = 1-4 and the far-field
reference) and writes results/tip_positions/compare_tip_shape.png.

Usage: python scripts/compare_tip_shape.py [out=results/tip_positions/compare_tip_shape.png]
       [pairs=label:cone_dir:rod_dir,label:cone_dir:rod_dir,...]
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_baseline import load_measured

ROOT = os.path.join(os.path.dirname(__file__), "..")
TP = os.path.join(ROOT, "results", "tip_positions")
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
                     "font.size": 11.5, "axes.titlesize": 12.5, "axes.labelsize": 11.5, "legend.fontsize": 9.5,
                     "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white"})

DEFAULT_PAIRS = (("2 mW, grounded tip,\nsymmetric pump (no FDTD map)", "P2mW_sym", "P2mW_sym_rod"),
                 ("2 mW, tip at the anode potential\n(V_tip = +12.5 V), symmetric pump", "P2mW_sym_Vtip12", "P2mW_sym_Vtip12_rod"),
                 ("2 mW, grounded tip,\nFDTD pump map (beam → +x)", "P2mW_tipmap", "P2mW_tipmap_rod"),
                 ("30 mW, grounded tip,\nFDTD pump map (beam → +x)", "P30mW_tipmap", "P30mW_tipmap_rod"),
                 ("2 mW, rod FDTD map mirrored: beam → −x,\nshadow on −X (dashed: cone map, beam → +x)", "P2mW_tipmap", "P2mW_tipmap_rod_mirror"))

measured = load_measured(os.path.join(ROOT, "data", "measured_tip_scan.csv"))


def load(name):
    fn = os.path.join(TP, name, "demod_A180_h10.json")
    return json.load(open(fn)) if os.path.exists(fn) else None


def plot_measured(ax):
    m = measured[measured["bias_V"] == 25]
    ax.errorbar(m["X_um"], m["rms"], yerr=m["rms_err"], fmt="o", color=C["ink"], ms=7, capsize=3, lw=1.4, label="measured +25 V", zorder=6)
    m = measured[measured["bias_V"] == -25]
    ax.errorbar(-np.asarray(m["X_um"]), m["rms"], yerr=m["rms_err"], fmt="s", mfc="white", mec=C["ink"], ecolor=C["ink"], ms=7,
                capsize=3, lw=1.1, label="measured −25 V (X → −X)", zorder=6)


def table(pairs):
    lines = []
    for label, cone, rod in pairs:
        sc, sr = load(cone), load(rod)
        if sc is None and sr is None:
            continue
        lines.append(f"\n## {label.replace(chr(10), ' ')}")
        X = (sr or sc)["X"]
        lines.append("| run | " + " | ".join(f"X = {x:+g}" for x in X) + " | abs. max (n=3) |")
        lines.append("|---|" + "---|" * (len(X) + 1))
        for tag, s in (("cone", sc), ("rod", sr)):
            if s is None:
                continue
            for k in ("det1", "det2", "det3", "det4", "far"):
                nm = k.replace("det", "n = ") if k != "far" else "far field dJ/dt"
                lines.append(f"| {tag}, {nm} | " + " | ".join(f"{v:.2f}" for v in s["rms_norm"][k]) +
                             (f" | {max(s['rms']['det3']):.3e} |" if k == "det3" else " | |"))
        if sc is not None and sr is not None:
            r = max(sr["rms"]["det3"]) / max(sc["rms"]["det3"])
            lines.append(f"| rod / cone, absolute n = 3 maximum | " + " | ".join("" for _ in X) + f" | {r:.2f} |")
    for Vb in (25, -25):
        m = measured[measured["bias_V"] == Vb]
        lines.append(f"measured {Vb:+d} V: " + ", ".join(f"X = {(x if Vb == 25 else (-x if x else 0.0)):+g}: {v:.2f} ± {e:.2f}" for x, v, e in zip(m["X_um"], m["rms"], m["rms_err"])) +
                     ("" if Vb == 25 else "  (X → −X)"))
    return "\n".join(lines)


def figure(pairs, out):
    pairs = [p for p in pairs if load(p[2]) is not None]
    n = len(pairs); ncol = min(3, n); nrow = (n + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.6 * nrow), constrained_layout=True, squeeze=False)
    for j, (label, cone, rod) in enumerate(pairs):
        sc, sr = load(cone), load(rod)
        X = np.array(sr["X"])
        ax = axs[j // ncol, j % ncol]
        if sc is not None:
            ax.plot(sc["X"], sc["rms_norm"]["det3"], "^--", color=C["blue"], ms=7, lw=1.6, mfc="white", label="cone tip (15°), n = 3")
        ax.plot(X, sr["rms_norm"]["det1"], "o-", color=C["amber"], ms=4.5, lw=1.1, label="rod tip, n = 1")
        ax.plot(X, sr["rms_norm"]["det3"], "^-", color=C["orange"], ms=7, lw=2.0, label="straight-rod tip (2R = 80 nm), n = 3")
        plot_measured(ax)
        ax.set_title(f"({'abcdefgh'[j]}) {label}", fontsize=11.5)
        ax.set_xlabel("X = spot − tip (µm);  anode at −x")
        if j % ncol == 0:
            ax.set_ylabel("band-limited amplitude (0.3–2 THz), normalised")
        ax.set_ylim(0, 1.12); ax.set_xticks(X)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    for k in range(n, nrow * ncol):
        axs[k // ncol, k % ncol].axis("off")
    if n < nrow * ncol:                       # legend in the empty slot
        axs[-1, -1].legend(handles, labels, loc="center", fontsize=11.5, labelspacing=1.1)
    else:
        fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=10.5)
    fig.suptitle("Per-tip-position chain: straight-rod tip vs cone tip vs measurement (A = 180 nm, h_min = 10 nm, 3 µm spot, 25 V)", fontsize=13)
    fig.savefig(out, dpi=140)
    return out


if __name__ == "__main__":
    kw = dict(a.split("=", 1) for a in sys.argv[1:])
    pairs = DEFAULT_PAIRS
    if "pairs" in kw:
        pairs = tuple(tuple(p.split(":")) for p in kw["pairs"].split(","))
    out = kw.get("out", os.path.join(TP, "compare_tip_shape.png"))
    txt = table(pairs)
    print(txt)
    with open(os.path.splitext(out)[0] + ".md", "w") as fh:
        fh.write("# Straight-rod tip vs cone tip (per-tip-position chain)\n" + txt + "\n")
    print("figure:", figure(pairs, out))
