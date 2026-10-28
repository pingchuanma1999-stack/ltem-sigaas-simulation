"""Is the weak side's emission simply outside the detection band? Spectra of d^2 S_n/dt^2 for the two lobes and the centre,
and the weak/strong ratio in several frequency bands, for route 1 (FDM on the tip-free field) and route 2 (tip in Poisson).
Usage: python scripts/band_check.py  -> prints a table and writes results/slides/sim_band_check.png"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from ltemsim.tip_demod import demodulate_positions, band_metric  # noqa: E402

BANDS = [(0.3, 1.0), (0.3, 2.0), (1.0, 2.0), (2.0, 5.0), (5.0, 15.0), (0.3, 30.0)]


def spectrum(t, S, deriv=2):
    S = np.asarray(S, dtype=float)
    S = np.concatenate([S, np.full(len(S), S[-1])]) - S[-1]
    f = np.fft.rfftfreq(len(S), t[1] - t[0])
    Y = np.fft.rfft(S) * (2j * np.pi * f) ** deriv
    return f, np.abs(Y)


def route1(folder, X, n=3):
    d = np.load(os.path.join(ROOT, "results", "pulse_chain", folder, f"X{X:+.1f}.npz"))
    return d["t"], d[f"S{n}"]


def route2(folder, X, n=3):
    meta = json.load(open(os.path.join(ROOT, "results", "tip_positions", folder, "meta.json")))
    Q = []
    for h in meta["heights_nm"]:
        d = np.load(os.path.join(ROOT, "results", "tip_positions", folder, f"X{X:+.1f}_h{int(h)}.npz")); Q.append(d["Q_tip"]); t = d["t"]
    dem = demodulate_positions(meta["heights_nm"], np.array(Q))
    return t, dem[n]


def rise_time(t, S):
    S = np.asarray(S) - S[0]; A = S[np.argmax(np.abs(S))]
    s = S / A
    i10 = np.argmax(s >= 0.1); i90 = np.argmax(s >= 0.9)
    return (t[i90] - t[i10]) * 1e15


def main():
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    print(f"{'route':8s} {'X':>5s} " + " ".join(f"{a:g}-{b:g}THz" for a, b in BANDS) + "   rise10-90 (fs)")
    for ax, (name, getter, folder) in zip(axs, (("route 1", route1, "P2mW_notipmap_degen"), ("route 2", route2, "P2mW_sym"))):
        vals = {}
        for X, col in ((5.0, "#d9541f"), (-5.0, "#1f6fb2"), (0.0, "#1b8a5a")):
            t, S = getter(folder, X)
            f, Y = spectrum(t, S)
            vals[X] = [band_metric(t, S, deriv=2, band_thz=b) for b in BANDS]
            ax.semilogy(f * 1e-12, Y / Y.max(), color=col, lw=2, label=f"X = {X:+g} µm")
            print(f"{name:8s} {X:+5.1f} " + " ".join(f"{v:9.3g}" for v in vals[X]) + f"   {rise_time(t, S):8.0f}")
        ratio = [vals[-5.0][i] / vals[5.0][i] for i in range(len(BANDS))]
        print(f"{name:8s} ratio -5/+5 " + " ".join(f"{r:9.2f}" for r in ratio))
        ax.axvspan(0.3, 2.0, color="#e9b56b", alpha=0.25, label="detector band 0.3–2 THz")
        ax.set_xlim(0, 20); ax.set_ylim(1e-3, 1.5); ax.set_xlabel("frequency (THz)"); ax.set_ylabel("|d²S₃/dt²| spectrum, normalised")
        ax.set_title(f"{name}: spectral shape of the two lobes and the centre"); ax.legend()
    fig.savefig(os.path.join(ROOT, "results", "slides", "sim_band_check.png"), dpi=150)


if __name__ == "__main__":
    main()
