"""Per-tip-position chain: for every beam position X and every tip height h of the tapping cycle, run the
carrier dynamics with the tip as an equipotential body in Poisson (and, optionally, the pump map from the
FDTD run with the tip at that height); record the charge induced on the tip Q_tip(t). Then demodulate all
heights together (cos nθ projection along h(θ) = h_min + A(1 + cos θ)) and form the detected waveform
∝ d²S_n/dt² with a steep 2.2 THz detector low-pass.

Usage:
  python scripts/run_tip_positions.py simulate out=results/tip_positions/<name> [fdtd=results/fdtd_pump/heights|none]
         [P_avg_mW=2] [degenerate=1] [T=300] [V_bias=25] [V_tip=0] [heights=10,15,...] [x_list=-7.5,...] [t_end_ps=4]
  python scripts/run_tip_positions.py demod out=results/tip_positions/<name> [A_nm=180] [h_min_nm=10]
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ltemsim.pulse_dynamics import PulseConfig, PulseSim
from ltemsim.tip_demod import demodulate_positions, detected_waveform, amplitude_metrics, lowpass_steep, band_metric
from run_baseline import load_measured

HEIGHTS_NM = (10, 15, 25, 45, 80, 130, 200, 270, 330, 370)
X_LIST = (-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5)


def load_ratio_file(path):
    d = np.load(path)
    x, z, r = d["x"] * 1e6, d["z"] * 1e6, np.clip(d["ratio"], 0.0, 50.0)
    interp = RegularGridInterpolator((x, z), r, bounds_error=False, fill_value=1.0)
    return lambda Xum, Zum: interp(np.stack([Xum.ravel(), Zum.ravel()], axis=1)).reshape(Xum.shape)


def simulate(out, fdtd="none", P_avg_mW=2.0, degenerate=1.0, T=300.0, V_bias=25.0, V_tip=0.0, heights=HEIGHTS_NM,
             x_list=X_LIST, t_end_ps=4.0, tip_shape="cone"):
    os.makedirs(out, exist_ok=True)
    meta = dict(P_avg_mW=P_avg_mW, degenerate=bool(degenerate), T=T, V_bias=V_bias, V_tip=V_tip, heights_nm=list(heights),
                X=list(x_list), fdtd=fdtd, t_end_ps=t_end_ps, tip_shape=tip_shape)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    for X in x_list:
        for h in heights:
            fn = os.path.join(out, f"X{X:+.1f}_h{int(h)}.npz")
            if os.path.exists(fn):
                continue
            ratio_map = None
            if fdtd and fdtd != "none":
                rf = os.path.join(fdtd, f"ratio_h{int(h)}.npz")
                while not os.path.exists(rf):          # the FDTD runs may still be in progress
                    time.sleep(20)
                time.sleep(1)
                ratio_map = load_ratio_file(rf)
            cfg = PulseConfig(P_avg_mW=P_avg_mW, x_beam_um=X, t_end_ps=t_end_ps, degenerate_diffusion=bool(degenerate), T=T,
                              V_bias=V_bias, tip_in_poisson=True, tip_height_nm=h, V_tip=V_tip, tip_shape=tip_shape)
            sim = PulseSim(cfg, verbose=False)
            N = sim.generation_profile(ratio_map)
            t0 = time.time()
            rec = sim.run(N)
            np.savez_compressed(fn, t=rec["t"], Q_tip=rec["Q_tip"], Q_tip_static=rec["Q_tip_static"], E_gap=rec["E_gap"],
                                Jx=rec["Jx"], Jz=rec["Jz"], n_center=rec["n_center"], Ex_center=rec["Ex_center"],
                                Ez_tip=rec["Ez_tip"], heights_nm=rec["heights_nm"], N_max=N.max())
            print(f"X={X:+.1f} h={h:>4.0f} nm: N_max {N.max():.2e}, |Q_tip| max {np.abs(rec['Q_tip']).max():.2e} C/cm, "
                  f"|E_gap| max {np.abs(rec['E_gap']).max()/1e3:.1f} kV/cm, {time.time()-t0:.0f}s", flush=True)


def demod(out, A_nm=180.0, h_min_nm=10.0, harmonics=(1, 2, 3, 4), f_cut_thz=2.2, deriv=2, band=(0.3, 2.0)):
    meta = json.load(open(os.path.join(out, "meta.json")))
    heights = [h for h in meta["heights_nm"]]
    measured = load_measured(os.path.join(os.path.dirname(__file__), "..", "data", "measured_tip_scan.csv"))
    res = {}
    for X in meta["X"]:
        Q = []; t = None; Jx = None
        for h in heights:
            d = np.load(os.path.join(out, f"X{X:+.1f}_h{int(h)}.npz"))
            Q.append(d["Q_tip"]); t = d["t"]; Jx = d["Jx"] if Jx is None else Jx
        Q = np.array(Q)
        dem = demodulate_positions(heights, Q, A_nm=A_nm, h_min_nm=h_min_nm, harmonics=harmonics)
        traces = {}
        for n in harmonics:
            traces[f"S{n}"] = dem[n]
            traces[f"det{n}"] = detected_waveform(t, dem[n], order=deriv, f_cut_thz=f_cut_thz)
            traces[f"det{n}_d1"] = detected_waveform(t, dem[n], order=1, f_cut_thz=f_cut_thz)
        traces["far"] = lowpass_steep(t, np.gradient(Jx, t), f_cut_thz=f_cut_thz)
        # metrics: band-limited (0.3-2 THz) spectral amplitude of d^deriv S_n/dt^deriv (robust to the detector roll-off)
        metrics = {}
        for n in harmonics:
            metrics[f"det{n}"] = band_metric(t, dem[n], deriv=deriv, band_thz=band)
            metrics[f"det{n}_d1"] = band_metric(t, dem[n], deriv=1, band_thz=band)
            metrics[f"S{n}"] = amplitude_metrics(t, dem[n])[0]
        metrics["far"] = band_metric(t, np.cumsum(Jx) * (t[1] - t[0]), deriv=2, band_thz=band)     # dJ/dt = d^2(∫J dt)/dt^2
        res[X] = dict(t=t, traces=traces, metrics=metrics, Q=Q)
    Xs = np.array(meta["X"])
    table = {k: np.array([res[X]["metrics"][k] for X in meta["X"]]) for k in res[meta["X"][0]]["metrics"]}
    summ = dict(meta, A_nm=A_nm, h_min_nm=h_min_nm, f_cut_thz=f_cut_thz, deriv=deriv, band_thz=list(band), rms={k: v.tolist() for k, v in table.items()},
                rms_norm={k: (v / v.max()).tolist() for k, v in table.items()})
    json.dump(summ, open(os.path.join(out, f"demod_A{A_nm:g}_h{h_min_nm:g}.json"), "w"), indent=1)
    for k in ("det1", "det2", "det3", "det4", "det3_d1", "far"):
        print(f"{k:8s}", np.round(table[k] / table[k].max(), 2).tolist())
    # figure
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    ax = axs[0]
    X0 = 5.0 if 5.0 in res else meta["X"][0]
    for k, h in enumerate(heights):
        ax.plot(res[X0]["t"] * 1e12, res[X0]["Q"][k] * 1e12, color=plt.cm.viridis(k / max(1, len(heights) - 1)), label=f"h = {h:g} nm")
    ax.set_xlim(0, 3); ax.set_xlabel("t (ps)"); ax.set_ylabel("charge induced on the tip (pC/cm)"); ax.set_title(f"(a) Q_tip(t) per tip height, X = {X0:+g} µm"); ax.legend(fontsize=7, ncol=2)
    ax = axs[1]
    for X, c in ((5.0, "C1"), (2.5, "C0"), (0.0, "k"), (-2.5, "C2"), (-5.0, "C3")):
        if X in res:
            ax.plot(res[X]["t"] * 1e12, res[X]["traces"]["det3"], color=c, label=f"X = {X:+g}")
    ax.set_xlim(0, 2.5); ax.set_xlabel("t (ps)"); ax.set_title("(b) detected ∝ d²S₃/dt², 2.2 THz steep low-pass"); ax.legend(fontsize=8)
    ax = axs[2]
    for n, c in ((1, "C1"), (2, "C0"), (3, "C2"), (4, "C8")):
        ax.plot(Xs, table[f"det{n}"] / table[f"det{n}"].max(), "o-", color=c, label=f"n = {n}")
    ax.plot(Xs, table["far"] / table["far"].max(), "x:", color="0.5", label="far field dJ/dt")
    ax.plot(Xs, table["det3_d1"] / table["det3_d1"].max(), "^--", color="0.3", label="n = 3, first-derivative convention")
    for Vb, col, mk in ((25, "k", "o"), (-25, "0.5", "s")):
        m = measured[measured["bias_V"] == Vb]
        xm = m["X_um"] if Vb == 25 else -np.asarray(m["X_um"])
        ax.errorbar(xm, m["rms"], yerr=m["rms_err"], fmt=mk, color=col, capsize=3, label=f"measured {Vb:+d} V" + ("" if Vb == 25 else " (X→−X)"))
    ax.set_xlabel("X (µm)"); ax.set_ylabel("band-limited amplitude (0.3-2 THz), normalised"); ax.set_title("(c) amplitude vs X"); ax.legend(fontsize=7)
    shape = {"rod": "straight-rod tip (2R = 80 nm)", "cone": "cone tip (15°)"}[meta.get("tip_shape", "cone")]
    fig.suptitle(f"per-tip-position chain: {meta['P_avg_mW']:g} mW, {meta['V_bias']:g} V, V_tip = {meta['V_tip']:g} V, {shape}, "
                 f"{'FDTD tip map' if meta['fdtd'] not in (None, 'none') else 'symmetric pump'}, A = {A_nm:g} nm, h_min = {h_min_nm:g} nm")
    fig.savefig(os.path.join(out, f"tip_positions_A{A_nm:g}.png"), dpi=130)
    return summ


if __name__ == "__main__":
    mode = sys.argv[1]
    kw = {}
    for a in sys.argv[2:]:
        k, v = a.split("=")
        if k in ("out", "fdtd", "tip_shape"):
            kw[k] = v
        elif k in ("heights", "x_list"):
            kw[k] = tuple(float(x) for x in v.split(","))
        else:
            kw[k] = float(v)
    (simulate if mode == "simulate" else demod)(**kw)
