"""Odd + even decomposition of the per-tip-position (route 2) runs, and the scale of the even (tip-potential) term that
the measured scan asks for.

O(X): grounded tip (results/tip_positions/P2mW_sym).  E(X): the change when the tip is put at +12.5 V
(P2mW_sym_Vtip12) minus the grounded run; checked to be even in X and odd in V_tip against the -12.5 V runs at X = +-5.
Amplitudes are signed: the in-band (0.3-2 THz) spectrum of d2S3/dt2 is projected on the grounded X = +5 waveform.
The scan for a scaled tip term is |O + alpha E|, alpha = 1 being the 12.5 V run; alpha is fitted to the measured
+25 V points (data/measured_tip_scan.csv, November points included).

Outputs: results/slides/sim_oddeven_fit.png, results/oddeven_fit.json.  Usage: python scripts/oddeven_fit.py
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from ltemsim.tip_demod import demodulate_positions  # noqa: E402
from run_baseline import load_measured  # noqa: E402

BASE = os.path.join(ROOT, "results", "tip_positions")
C = dict(orange="#d9541f", blue="#1f6fb2", green="#1b8a5a", ink="#1b2430", grey="#8a8984", amber="#b07a00")


def traces(folder, n=3):
    meta = json.load(open(os.path.join(BASE, folder, "meta.json")))
    out = {}
    for X in meta["X"]:
        Q = []
        for h in meta["heights_nm"]:
            d = np.load(os.path.join(BASE, folder, f"X{X:+.1f}_h{int(h)}.npz"))
            Q.append(d["Q_tip"]); t = d["t"]
        out[X] = demodulate_positions(meta["heights_nm"], np.array(Q))[n]
    return t, out


def band_spectrum(t, S, band=(0.3e12, 2.0e12)):
    S = np.asarray(S, float)
    S = np.concatenate([S, np.full(len(S), S[-1])]) - S[-1]
    f = np.fft.rfftfreq(len(S), t[1] - t[0])
    Y = np.fft.rfft(S) * (2j * np.pi * f) ** 2
    m = (f >= band[0]) & (f <= band[1])
    return Y[m]


def main():
    t, G = traces("P2mW_sym"); _, T = traces("P2mW_sym_Vtip12"); _, M = traces("P2mW_sym_Vtipm12")
    Xs = np.array(sorted(G))
    ref = band_spectrum(t, G[5.0])
    signed = lambda S: float(np.vdot(ref, band_spectrum(t, S)).real / np.linalg.norm(ref))
    A0 = abs(signed(G[5.0]))
    O = np.array([signed(G[X]) for X in Xs]) / A0
    St = np.array([signed(T[X]) for X in Xs]) / A0
    E = St - O
    # checks against the -12.5 V runs
    chk = {}
    for X in (-5.0, 5.0):
        sp, sm = signed(T[X]) / A0, signed(M[X]) / A0
        chk[X] = dict(tip_term=(sp - sm) / 2, drift_term=(sp + sm) / 2, grounded=float(O[Xs == X][0]))
    even = {float(X): float(E[Xs == -X][0] / E[Xs == X][0]) for X in (2.5, 5.0, 7.5)}
    # measured +25 V
    meas = load_measured(os.path.join(ROOT, "data", "measured_tip_scan.csv"))
    mp = meas[meas["bias_V"] == 25]
    mx, mv, me = np.asarray(mp["X_um"]), np.asarray(mp["rms"]), np.asarray(mp["rms_err"])

    def scan(alpha):
        s = np.abs(O + alpha * E); return s / s.max()

    def chi2(alpha):
        s = scan(alpha); mod = np.array([s[Xs == X][0] for X in mx])
        return float(np.sum(((mod - mv) / me) ** 2))
    alphas = np.linspace(0, 1.2, 241)
    c2 = np.array([chi2(a) for a in alphas])
    best = float(alphas[np.argmin(c2)])
    inside = alphas[c2 <= c2.min() + 1.0]
    out = dict(X=Xs.tolist(), O=O.tolist(), E=E.tolist(), evenness=even, checks=chk, alpha_best=best,
               alpha_1sigma=[float(inside.min()), float(inside.max())], chi2_best=float(c2.min()), chi2_alpha1=chi2(1.0), chi2_alpha0=chi2(0.0),
               V_tip_equiv_V=12.5 * best, scan_best=scan(best).tolist(), scan_alpha1=scan(1.0).tolist(), scan_alpha0=scan(0.0).tolist(),
               note="alpha = 1 is the 12.5 V run; V_tip_equiv assumes the tip term is linear in the tip-surface voltage (checked at +-12.5 V only)")
    json.dump(out, open(os.path.join(ROOT, "results", "oddeven_fit.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("scan_best", "scan_alpha1", "scan_alpha0")}, indent=1))
    print("scan alpha=0  :", np.round(scan(0.0), 2)); print("scan alpha=best:", np.round(scan(best), 2)); print("scan alpha=1  :", np.round(scan(1.0), 2))

    fig, axs = plt.subplots(1, 2, figsize=(17, 5.6), constrained_layout=True)
    ax = axs[0]
    ax.axhline(0, color=C["grey"], lw=0.6); ax.axvline(0, color=C["grey"], lw=0.6)
    ax.plot(Xs, O, "o-", color=C["blue"], lw=2.2, ms=7, label="O(X): grounded tip — the drift fronts, odd in X")
    ax.plot(Xs, E, "s-", color=C["amber"], lw=2.2, ms=7, label="E(X): tip at +12.5 V minus grounded — even in X")
    ax.plot(Xs, best * E, "s--", color=C["green"], lw=1.8, ms=6, label=f"{best:.2f} × E(X): the size the data ask for")
    ax.set_xlabel("X = beam position relative to the tip (µm)"); ax.set_ylabel("signed in-band amplitude (grounded X = +5 = 1)")
    ax.set_title("(a) the two terms, from the per-tip-position runs (2 mW, n = 3)"); ax.legend(loc="upper left", fontsize=9.5)
    ax = axs[1]
    for alpha, col, lab, ls in ((0.0, C["blue"], "grounded tip (α = 0): two equal lobes", "-"),
                                (1.0, C["amber"], "tip at the anode potential, 12.5 V (α = 1)", "-"),
                                (best, C["green"], f"α = {best:.2f}  (≈ {12.5 * best:.0f} V tip–surface if linear): fitted", "-")):
        ax.plot(Xs, scan(alpha), marker="o", ls=ls, color=col, lw=2.4 if alpha == best else 1.8, ms=6, label=lab)
    ax.errorbar(mx, mv, yerr=me, fmt="o", color=C["ink"], ms=9, capsize=3, lw=1.5, label="measured +25 V (Oct + Nov)", zorder=6)
    mm = meas[meas["bias_V"] == -25]
    ax.errorbar(-np.asarray(mm["X_um"]), mm["rms"], yerr=mm["rms_err"], fmt="s", mfc="white", mec=C["ink"], ecolor=C["ink"], ms=9, capsize=3, lw=1.2, label="measured −25 V (X → −X)", zorder=6)
    ax.axvline(0, color=C["grey"], lw=0.6); ax.set_xlim(-9, 9); ax.set_ylim(0, 1.5)
    ax.set_xlabel("X = beam position relative to the tip (µm)"); ax.set_ylabel("|O + αE|, normalised")
    ax.set_title("(b) one parameter, the size of the even term, sets the weak side and the centre"); ax.legend(loc="upper left", fontsize=9.5)
    fig.savefig(os.path.join(ROOT, "results", "slides", "sim_oddeven_fit.png"), dpi=150)


if __name__ == "__main__":
    main()
