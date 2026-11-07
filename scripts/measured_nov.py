"""November 3-6 LTEM scans: the weak side at +25 V (X = -7.5, -5, -2.5 um, means of 3) and -25 V at X = -2.5 and +5 um.

Metric as in the October analysis: noise-corrected RMS of lock-in A in the stage window 82.80-83.20 mm. Two noise
estimators: (i) point-to-point differences (the October estimator; under-corrects correlated noise), (ii) for the
"Mean of 3" files, the recorded scan-to-scan STD / sqrt(3). The reported value uses (ii) when available, else (i).
Scale: the October table is relative to +25 V, X = +5 um (Oct 19). No reference scan exists in the November sessions,
so the new data are put on that scale through the one condition measured in both sets, -25 V at X = -2.5 um
(Oct 19 single scan 0.52 +- 0.10; Nov 3 two single scans). The scale factor therefore carries about +-20 %.

Outputs: results/slides/data_nov_weak_side.png, data/measured_tip_scan.csv (rewritten with the new rows),
results/measured_nov_table.md.  Usage: python scripts/measured_nov.py
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io as sio

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "nov2026")
OUT = os.path.join(ROOT, "results", "slides")
os.makedirs(OUT, exist_ok=True)
C = dict(orange="#d9541f", blue="#1f6fb2", green="#1b8a5a", ink="#1b2430", grey="#8a8984", muted="#6a7179", amber="#b07a00")
WIN = (82.80, 83.20)          # amplitude window (mm), as in October
CORR = (82.78, 83.26)         # correlation window (mm), as in October
PS_PER_MM = 2.0 / 0.299792458  # double pass
ANCHOR_OLD = (0.52, 0.10)     # -25 V, X = -2.5 um, Oct 19 single scan, relative units

# the October table (relative units), kept verbatim
OCT = [  # bias, X, rms, err, note
    (25, 0, 0.24, 0.08, "Oct 19 single scan"),
    (25, 2.5, 0.59, 0.07, "Oct 21 16h30m35s single scan"),
    (25, 5, 1.00, 0.08, "reference (Oct 19 11h15m48s; Oct 8 gives 1.09)"),
    (25, 7.5, 0.15, 0.04, "Oct 21 mean of 3"),
    (-25, -7.5, 0.15, 0.05, "Oct 22 mean of 3"),
    (-25, -5, 0.45, 0.04, "Oct 22 mean of 3 runs"),
    (-25, -2.5, 0.52, 0.10, "Oct 19 single scan (error nominal)"),
    (-25, 0, 0.33, 0.10, "Oct 19 single scan (error nominal)"),
    (-25, 2.5, 0.20, 0.10, "Oct 19 single scan (error nominal)"),
    (-25, 5, 0.32, 0.10, "Oct 19 two runs 0.42 / 0.22 (mean; error nominal)"),
]


def load():
    runs = []
    with open(os.path.join(DATA, "labels.csv")) as fh:
        for r in csv.DictReader(fh):
            d = sio.loadmat(os.path.join(DATA, r["file"]))
            pos = float(d["iPos"].ravel()[0]) + float(d["step_size"].ravel()[0]) * np.arange(100)
            if "MeanTimeTraceA" in d:
                y = d["MeanTimeTraceA"].ravel().astype(float)
                sd = d["STDTimeTraceA"].ravel().astype(float)
            else:
                y = d["data_A"].ravel().astype(float)
                sd = None
            runs.append(dict(file=r["file"], tag=r["file"][:9], date=r["date"], bias=int(r["bias_V"]), X=float(r["X_um"]),
                             kind=r["kind"], note=r["note"], pos=pos, y=y, sd=sd))
    return runs


def metrics(r):
    m = (r["pos"] >= WIN[0]) & (r["pos"] <= WIN[1])
    y = r["y"]
    raw = np.sqrt(np.mean(y[m] ** 2))
    sig_diff = np.std(np.diff(y)) / np.sqrt(2)
    sig_std = np.mean(r["sd"][m]) / np.sqrt(3) if r["sd"] is not None else np.nan
    sig = sig_std if np.isfinite(sig_std) else sig_diff
    n = int(m.sum())
    corr = np.sqrt(max(raw ** 2 - sig ** 2, 0.0))
    # 1-sigma uncertainty of the corrected RMS: d(rms^2) ~ sig^2 * sqrt(2/n) (noise-only) + 2*sig*rms/sqrt(n) (cross term)
    var_rms2 = (sig ** 2 * np.sqrt(2.0 / n)) ** 2 + (2 * sig * corr / np.sqrt(n)) ** 2
    err = np.sqrt(var_rms2) / (2 * corr) if corr > 0.5 * sig else np.sqrt(np.sqrt(var_rms2))  # linear / floor regime
    return dict(raw=raw, sig_diff=sig_diff, sig_std=sig_std, sig=sig, corr=corr, err=err, n=n,
                corr_diff=np.sqrt(max(raw ** 2 - sig_diff ** 2, 0.0)))


def correlation(a, b):
    m = (a["pos"] >= CORR[0]) & (a["pos"] <= CORR[1])
    x = a["y"][m] - a["y"][m].mean()
    y = b["y"][m] - b["y"][m].mean()
    return float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)))


def main():
    runs = load()
    for r in runs:
        r.update(metrics(r))
    by = lambda bias, X: [r for r in runs if r["bias"] == bias and r["X"] == X]

    # ---- scale: Nov 3 -25 V, X = -2.5 (two single scans) <-> Oct 19 0.52
    anchor_V = np.mean([r["corr"] for r in by(-25, -2.5)])
    anchor_uV = anchor_V * 1e6
    k = ANCHOR_OLD[0] / anchor_V                        # relative units per volt
    k_err = k * ANCHOR_OLD[1] / ANCHOR_OLD[0]
    lines = ["# November 3-6 scans: amplitude metric", "",
             f"Scale: -25 V, X = -2.5 um = {anchor_uV:.2f} uV (mean of two Nov 3 single scans) == {ANCHOR_OLD[0]} (Oct 19) -> "
             f"1.00 (the +25 V, X = +5 um reference) == {1e6 / k:.2f} uV, +-{100 * ANCHOR_OLD[1] / ANCHOR_OLD[0]:.0f} %.", "",
             "| file | date | bias | X (um) | kind | raw RMS (uV) | noise diff (uV) | noise STD/sqrt3 (uV) | corrected RMS (uV) | relative | note |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        lines.append(f"| {r['tag']} | {r['date']} | {r['bias']:+d} | {r['X']:+g} | {r['kind']} | {r['raw'] * 1e6:.2f} | {r['sig_diff'] * 1e6:.2f} | "
                     f"{'' if not np.isfinite(r['sig_std']) else f'{r['sig_std'] * 1e6:.2f}'} | {r['corr'] * 1e6:.2f} +- {r['err'] * 1e6:.2f} | "
                     f"{r['corr'] * k:.2f} +- {r['err'] * k:.2f} | {r['note']} |")
    # ---- waveform correlations
    R = {(r["tag"], r["X"]): r for r in runs}
    pairs = [(("22h45m12s", -2.5), ("23h05m15s", -2.5), "-25 V X=-2.5: before vs after moving the tip"),
             (("23h27m21s", 5.0), ("23h27m21s", 5.0), "-25 V X=+5: scan 2 vs scan 3"),
             (("00h28m32s", -5.0), ("04h25m30s", -5.0), "+25 V X=-5: Nov 4 vs Nov 6"),
             (("02h07m42s", -7.5), ("01h48m18s", -7.5), "+25 V X=-7.5: Nov 4 vs Nov 6"),
             (("00h28m32s", -5.0), ("01h18m30s", -2.5), "+25 V X=-5 vs X=-2.5 (Nov 4)"),
             (("00h28m32s", -5.0), ("23h05m15s", -2.5), "+25 V X=-5 vs the -25 V X=-2.5 template"),
             (("01h18m30s", -2.5), ("23h05m15s", -2.5), "same X=-2.5: +25 V vs -25 V"),
             (("00h28m32s", -5.0), ("23h27m21s", 5.0), "mirror pair: +25 V X=-5 vs -25 V X=+5 (scan 2)")]
    s2, s3 = by(-25, 5.0)
    lines += ["", "Waveform correlation in 82.78-83.26 mm (pure noise: |r| < 0.31 at 95 % for 40 points):", ""]
    for a, b, lab in pairs:
        if lab.startswith("-25 V X=+5"):
            rr = correlation(s2, s3)
        elif "scan 2" in lab:
            rr = correlation(R[a], s2)
        else:
            rr = correlation(R[a], R[b])
        lines.append(f"- {lab}: r = {rr:+.2f}")
    # ---- combined points (relative units)
    def combine(rs):
        v = np.array([r["corr"] for r in rs]) * k
        e = np.array([r["err"] for r in rs]) * k
        wts = 1 / e ** 2
        return float(np.sum(wts * v) / np.sum(wts)), float(1 / np.sqrt(np.sum(wts)))
    # error floor for noise-level points: the single-scan and 3-scan noise is partly correlated (October note), so a
    # point that is not detected is quoted with +-0.08, roughly 0.7 sigma of a 3-scan mean on this scale
    FLOOR = 0.08
    new = {}
    for X in (-7.5, -5.0, -2.5):
        v, e = combine(by(25, X))
        e = max(np.hypot(e, v * k_err / k), FLOOR)
        new[(25, X)] = (v, e)
    v, e = combine(by(-25, 5.0)); new[(-25, 5.0)] = (v, max(np.hypot(e, v * k_err / k), 0.10))
    lines += ["", "Combined (relative to +25 V, X = +5 um; error includes the 20 % scale uncertainty):", ""]
    for key, (v, e) in new.items():
        lines.append(f"- {key[0]:+d} V, X = {key[1]:+g} um: {v:.2f} +- {e:.2f}")
    # ---- rewrite the measured csv
    rows = []
    for bias, X, rms, err, note in OCT:
        if (bias, X) == (-25, 5):
            v, e = new[(-25, 5.0)]
            vv = 0.5 * (rms + v); ee = 0.10
            rows.append((bias, X, round(vv, 2), round(ee, 2), f"Oct 19 two runs 0.42 / 0.22 and Nov 3 two scans ({v:.2f}); all at noise level"))
        elif (bias, X) == (-25, -2.5):
            rows.append((bias, X, rms, 0.07, "Oct 19 single scan; Nov 3 two scans agree by construction (scale anchor)"))
        else:
            rows.append((bias, X, rms, err, note))
    for X in (-7.5, -5.0, -2.5):
        v, e = new[(25, X)]
        n = len(by(25, X))
        rows.append((25, X, round(v, 2), round(e, 2), f"Nov 4/6, {n} x mean of 3; at noise level"))
    rows.sort(key=lambda r: (-r[0], r[1]))
    with open(os.path.join(ROOT, "data", "measured_tip_scan.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["bias_V", "X_um", "rms", "rms_err", "note"])
        for r in rows:
            w.writerow([int(r[0]), f"{r[1]:g}", f"{r[2]:.2f}", f"{r[3]:.2f}", r[4]])
    open(os.path.join(ROOT, "results", "measured_nov_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- figure
    fig, axs = plt.subplots(1, 2, figsize=(17, 6.0), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    ax = axs[0]
    t = lambda r: (r["pos"] - 82.70) * PS_PER_MM
    strong = by(-25, -2.5)
    ys = np.mean([r["y"] for r in strong], axis=0) * 1e6
    ax.plot(t(strong[0]), ys, color=C["blue"], lw=2.4, label="−25 V, X = −2.5 µm (Nov 3, mean of 2 scans)")
    cols = [C["green"], "#7cc5a0", C["grey"], C["ink"], C["amber"]]
    for r, c in zip(by(25, -5.0) + by(25, -2.5) + by(25, -7.5), cols):
        ax.plot(t(r), r["y"] * 1e6, color=c, lw=1.3, label=f"+25 V, X = {r['X']:+g} µm ({r['date'][5:].replace('-', '/')} mean of 3)")
    ax.axvspan((WIN[0] - 82.7) * PS_PER_MM, (WIN[1] - 82.7) * PS_PER_MM, color="#e9b56b", alpha=0.18, label="amplitude window")
    ax.axhline(0, color=C["grey"], lw=0.6)
    ax.set_xlabel("delay (ps), 2Δx/c from 82.70 mm"); ax.set_ylabel("lock-in A (µV)")
    ax.set_title("(a) +25 V on the −X side: nothing above the noise, on two days")
    ax.legend(fontsize=8.5, loc="lower right", ncol=2)
    ax.set_ylim(-3.6, 3.0)
    ax = axs[1]
    for bias, col, mk in ((25, C["orange"], "o"), (-25, C["blue"], "s")):
        old = [(X, v, e) for b, X, v, e, _ in OCT if b == bias and (b, X) not in ((-25, 5),)]
        X_, v_, e_ = zip(*old)
        ax.errorbar(X_, v_, yerr=e_, fmt=mk, color=col, ms=9, capsize=3, lw=1.5, label=f"{bias:+d} V, Oct 19–22")
    nx = [X for (b, X) in new if b == 25]; nv = [new[(25, X)][0] for X in nx]; ne = [new[(25, X)][1] for X in nx]
    ax.errorbar(nx, nv, yerr=ne, fmt="o", mfc="white", mec=C["orange"], ecolor=C["orange"], ms=10, capsize=3, lw=1.6, label="+25 V, Nov 4–6 (means of 3)")
    v, e = new[(-25, 5.0)]
    ax.errorbar([5.0], [0.32], yerr=[0.10], fmt="s", color=C["blue"], ms=9, capsize=3, lw=1.5)
    ax.errorbar([5.0 + 0.25], [v], yerr=[e], fmt="s", mfc="white", mec=C["blue"], ecolor=C["blue"], ms=10, capsize=3, lw=1.6, label="−25 V, Nov 3 (two scans)")
    ax.axhspan(0, 0.15, color=C["grey"], alpha=0.15, label="noise level of a 3-scan mean")
    ax.axvline(0, color=C["grey"], lw=0.6)
    ax.set_xlim(-9, 9); ax.set_ylim(-0.05, 1.2); ax.set_xticks([-7.5, -5, -2.5, 0, 2.5, 5, 7.5])
    ax.set_xlabel("X = beam position relative to the tip (µm)"); ax.set_ylabel("amplitude, relative to +25 V at X = +5 µm")
    ax.set_title("(b) both polarities: one side only, mirror image under V → −V")
    ax.legend(fontsize=9, loc="upper left")
    fig.savefig(os.path.join(OUT, "data_nov_weak_side.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
