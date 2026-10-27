"""Single-pulse chain: generation (analytic Beer–Lambert Gaussian × FDTD tip ratio map)
-> hydrodynamic carrier dynamics + Poisson -> THz near field at the tip -> FDM tip
scattering -> tapping demodulation. Scans the beam position X relative to the tip.

Usage: python scripts/run_pulse_chain.py [P_avg_mW=2] [A_nm=180] [out=results/pulse_chain] [fdtd=results/fdtd_pump/ratio.npz|none]
       [degenerate=1] [T=300] [V_bias=25] [x_list=-5,0,5] [w_um=3] [x_fine=8]
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ltemsim.pulse_dynamics import PulseConfig, PulseSim
from ltemsim.tip_demod import demodulate_pizzuto, lowpass, amplitude_metrics
from run_baseline import load_measured

X_LIST = (-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5)


def load_ratio(path):
    if not path or path == "none" or not os.path.exists(path):
        return None, None
    d = np.load(path)
    x, z, r = d["x"] * 1e6, d["z"] * 1e6, d["ratio"]
    r = np.clip(r, 0.0, 50.0)
    interp = RegularGridInterpolator((x, z), r, bounds_error=False, fill_value=1.0)

    def ratio_map(Xum, Zum):
        pts = np.stack([Xum.ravel(), Zum.ravel()], axis=1)
        return interp(pts).reshape(Xum.shape)
    return ratio_map, d


def main(out="results/pulse_chain", fdtd="results/fdtd_pump/ratio.npz", P_avg_mW=2.0, A_nm=180.0, h_min_nm=0.0,
         t_end_ps=4.0, harmonics=(1, 2, 3, 4), degenerate=0.0, T=300.0, V_bias=25.0, x_list=None, w_um=3.0, x_fine=None):
    # tapping amplitude 180 nm (user, 2026-10-26: ~180 nm in the experiment), harmonics 1-4, Pizzuto FDM conventions
    os.makedirs(out, exist_ok=True)
    ratio_map, dr = load_ratio(fdtd)
    measured = load_measured(os.path.join(os.path.dirname(__file__), "..", "data", "measured_tip_scan.csv"))
    results = {}
    X_LIST = tuple(x_list) if x_list else globals()["X_LIST"]
    for X in X_LIST:
        xf = x_fine if x_fine else max(8.0, 2.0 * w_um + max(abs(x) for x in X_LIST) - 4.0)   # fine grid covers the spot at every offset
        cfg = PulseConfig(P_avg_mW=P_avg_mW, x_beam_um=X, t_end_ps=t_end_ps, degenerate_diffusion=bool(degenerate), T=T, V_bias=V_bias,
                          w_um=w_um, x_fine=xf)
        sim = PulseSim(cfg, verbose=(X == 0.0))
        N = sim.generation_profile(ratio_map)
        t0 = time.time()
        rec = sim.run(N)
        dem = demodulate_pizzuto(rec["t"], rec["heights_nm"], rec["Ez_tip"], rec["n_center"], A_nm=A_nm,
                                 h_min_nm=h_min_nm, harmonics=harmonics)
        t = rec["t"]
        dJx = np.gradient(rec["Jx"], t)
        traces = {f"S{n}": dem[n] for n in harmonics}
        for n in harmonics:                                   # radiated/detected waveform ∝ time derivative (removes the static dipole step)
            traces[f"dS{n}_dt"] = np.gradient(dem[n], t)
        traces["E_loc_mid"] = dem["E_loc_mid"]
        traces["dJx_dt"] = dJx
        traces["dJz_dt"] = np.gradient(rec["Jz"], t)
        metrics = {}
        for k, y in traces.items():
            pk, rms = amplitude_metrics(t, lowpass(t, y))
            metrics[k] = dict(peak=pk, rms=rms)
        results[X] = dict(t=t, traces=traces, metrics=metrics, rec=rec, N=N)
        np.savez_compressed(os.path.join(out, f"X{X:+.1f}.npz"), t=t, **{k: v for k, v in traces.items()},
                            Ez_tip=rec["Ez_tip"], heights_nm=rec["heights_nm"], Jx=rec["Jx"], Jz=rec["Jz"],
                            n_center=rec["n_center"], Ex_center=rec["Ex_center"], N=N, xc=rec["xc_um"], zc=rec["zc_um"])
        print(f"X={X:+.1f}: N_max {N.max():.2e}, S2 rms {metrics['S2']['rms']:.3e} peak {metrics['S2']['peak']:.3e}, "
              f"dJx/dt rms {metrics['dJx_dt']['rms']:.3e}, {time.time()-t0:.0f}s", flush=True)
    # ---------------- summary table and figures ----------------
    Xs = np.array(X_LIST)
    Xref = 0.0 if 0.0 in results else X_LIST[0]
    table = {k: np.array([results[X]["metrics"][k]["rms"] for X in X_LIST]) for k in results[Xref]["metrics"]}
    peaks = {k: np.array([results[X]["metrics"][k]["peak"] for X in X_LIST]) for k in results[Xref]["metrics"]}
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(dict(X=Xs.tolist(), rms={k: v.tolist() for k, v in table.items()}, peak={k: v.tolist() for k, v in peaks.items()},
                       P_avg_mW=P_avg_mW, A_nm=A_nm, h_min_nm=h_min_nm, fdtd_ratio=bool(ratio_map), degenerate=bool(degenerate), T=T, V_bias=V_bias,
                       w_um=w_um), fh, indent=1)
    fig, axs = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    # (a) generation map at X = 0
    r0 = results[Xref]
    ax = axs[0, 0]
    N = r0["N"]; xc, zc = r0["rec"]["xc_um"], r0["rec"]["zc_um"]
    sel = np.abs(xc) < 8
    pc = ax.pcolormesh(xc[sel], zc[zc > -2.5], np.log10(np.maximum(N[zc > -2.5][:, sel], 1e10)), cmap="viridis", shading="nearest")
    fig.colorbar(pc, ax=ax, label="log10 pairs per pulse (cm^-3)")
    ax.set_title(f"generation, X = 0, {P_avg_mW:g} mW{' (FDTD tip map)' if ratio_map else ''}"); ax.set_xlabel("x (um)"); ax.set_ylabel("z (um)")
    # (b) snapshot n and E_x at 0.3 ps
    ax = axs[0, 1]
    snap = r0["rec"]["snapshots"].get(0.3) or list(r0["rec"]["snapshots"].values())[0]
    pc = ax.pcolormesh(xc[sel], zc[zc > -2.5], snap["Ex"][zc > -2.5][:, sel] / 1e3, cmap="RdBu_r", vmin=-6, vmax=6, shading="nearest")
    fig.colorbar(pc, ax=ax, label="E_x (kV/cm)")
    ax.set_title("E_x at 0.3 ps (applied + space charge), X = 0"); ax.set_xlabel("x (um)"); ax.set_ylabel("z (um)")
    # (c) traces at X = 0, 2.5, 5
    ax = axs[0, 2]
    for X, c in ((0.0, "k"), (2.5, "C0"), (5.0, "C1"), (-2.5, "C2"), (-5.0, "C3")):
        if X not in results:
            continue
        y = lowpass(results[X]["t"], results[X]["traces"]["dS3_dt"])
        ax.plot(results[X]["t"] * 1e12, y / 1e12, color=c, label=f"X={X:+.1f}")
    ax.set_xlabel("t (ps)"); ax.set_ylabel("d S3/dt (V/cm/ps, 2.2 THz low-pass)"); ax.legend(fontsize=8); ax.set_title("detected waveform ∝ d/dt of the demodulated near field (3rd harmonic)")
    # (d) E_loc vs height at a few times, X = 0
    ax = axs[1, 0]
    rec = r0["rec"]
    for tt in (0.1, 0.2, 0.5, 1.0, 3.0):
        i = np.argmin(np.abs(rec["t"] * 1e12 - tt))
        ax.semilogx(rec["heights_nm"], rec["Ez_tip"][i], label=f"{tt} ps")
    ax.set_xlabel("tip height h (nm)"); ax.set_ylabel("E_z,loc (V/cm)"); ax.legend(fontsize=8); ax.set_title("near field above the spot centre vs height")
    # (e) far-field reference and centre field
    ax = axs[1, 1]
    ax.plot(rec["t"] * 1e12, rec["Ex_center"] / 1e3, label="E_x at spot centre (kV/cm)")
    ax2 = ax.twinx(); ax2.plot(rec["t"] * 1e12, lowpass(rec["t"], np.gradient(rec["Jx"], rec["t"])), color="C3", label="dJx/dt (far-field ref., lp)")
    ax.set_xlabel("t (ps)"); ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8); ax.set_title("screening of the applied field, X = 0")
    # (f) amplitude vs X
    ax = axs[1, 2]
    for k, c in (("dS1_dt", "C9"), ("dS2_dt", "C0"), ("dS3_dt", "C1"), ("dS4_dt", "C2"), ("dJx_dt", "C3")):
        if k in table:
            ax.plot(Xs, table[k] / table[k].max(), "o-", color=c, label=f"model {k} RMS")
    for Vb, col, mk in ((25, "k", "o"), (-25, "C7", "s")):
        m = measured[measured["bias_V"] == Vb]
        ax.errorbar(m["X_um"], m["rms"], yerr=m["rms_err"], fmt=mk, color=col, capsize=3, label=f"measured {Vb:+d} V")
    ax.set_xlabel("X = beam position relative to tip (um)"); ax.set_ylabel("amplitude (norm.)"); ax.legend(fontsize=7); ax.set_title("amplitude vs X")
    fig.suptitle(f"single pulse, bare field, {P_avg_mW:g} mW, w = {w_um:g} um, {V_bias:g} V, 45 deg incidence (beam travels to +x), tapping A = {A_nm:g} nm"
                 + (", degenerate diffusion" if degenerate else "") + (f", T_carrier = {T:g} K" if T != 300 else ""))
    fig.savefig(os.path.join(out, "pulse_chain.png"), dpi=130)
    print("RMS vs X:", {k: np.round(v / v.max(), 3).tolist() for k, v in table.items()})
    return results


if __name__ == "__main__":
    kw = {}
    for a in sys.argv[1:]:
        k, v = a.split("=")
        kw[k] = v if k in ("out", "fdtd") else ([float(a) for a in v.split(",")] if k == "x_list" else float(v))
    main(**kw)
