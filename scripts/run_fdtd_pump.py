"""Pump-field FDTD: vacuum reference, GaAs without tip, GaAs with tip. Writes results/fdtd_pump/*.npz and a figure.
Usage: python scripts/run_fdtd_pump.py [theta_deg=60] [w_um=3] [dx_nm=15]"""
import os, sys, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ltemsim.optics_fdtd import FDTDConfig, run, fresnel_p_absorption


def main(out="results/fdtd_pump", **kw):
    os.makedirs(out, exist_ok=True)
    dx = kw.pop("dx_nm", 15.0) * 1e-9
    base = dict(dx=dx, x_half=4.0e-6, z_min=-3.0e-6, z_max=2.5e-6, z_source=2.2e-6, **kw)
    res = {}
    for name, extra in (("vacuum", dict(gaas=False, tip=False)), ("notip", dict(tip=False)), ("tip", dict(tip=True))):
        t = time.time()
        r = run(FDTDConfig(**base, **extra), verbose=True)
        res[name] = r
        np.savez_compressed(os.path.join(out, f"{name}.npz"), x=r.x, z=r.z, Q=r.Q, Emax2=r.Emax2,
                            launched=r.launched_energy, absorbed=r.absorbed_energy, tip=r.tip_mask_cc, gaas=r.gaas_mask_cc,
                            cfg=json.dumps(r.cfg.__dict__))
        print(f"{name}: launched {r.launched_energy:.3e} J/m, absorbed {r.absorbed_energy:.3e} J/m, {time.time()-t:.0f}s", flush=True)
    L = res["vacuum"].launched_energy
    cfg = res["notip"].cfg
    summary = dict(absorbed_fraction_notip=res["notip"].absorbed_energy / L, absorbed_fraction_tip=res["tip"].absorbed_energy / L,
                   fresnel_p=fresnel_p_absorption(cfg.n_gaas, cfg.kappa_gaas, cfg.theta_deg), theta_deg=cfg.theta_deg, w_um=cfg.w_um)
    # enhancement / shadow map: ratio of absorbed energy density with tip to without
    Qt, Qn = res["tip"].Q, res["notip"].Q
    ratio = np.where(Qn > 1e-3 * Qn.max(), Qt / np.maximum(Qn, 1e-300), 1.0)
    np.savez_compressed(os.path.join(out, "ratio.npz"), x=res["tip"].x, z=res["tip"].z, ratio=ratio, Qtip=Qt, Qnotip=Qn)
    x, z = res["tip"].x * 1e6, res["tip"].z * 1e6
    gz = z < 0
    # column sums (generation per unit x) with/without tip, and the ratio near the surface
    col_t = Qt[:, gz].sum(axis=1); col_n = Qn[:, gz].sum(axis=1)
    summary["column_ratio_min"] = float((col_t / col_n).min()); summary["column_ratio_max"] = float((col_t / col_n).max())
    k0 = np.argmin(np.abs(z + 0.02))
    summary["surface_ratio_under_apex"] = float(ratio[np.argmin(np.abs(x)), k0])
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ext = [x[0], x[-1], z[0], z[-1]]
    im = axs[0, 0].imshow(np.log10(np.maximum(res["tip"].Emax2.T, 1e-30) / res["notip"].Emax2[:, z > 0.5].max()), origin="lower", extent=ext, cmap="magma", vmin=-3, vmax=2)
    axs[0, 0].set_title("log10 max|E|^2 / incident (with tip)"); fig.colorbar(im, ax=axs[0, 0])
    im = axs[0, 1].imshow(np.log10(np.maximum(Qt.T, 1e-30) / Qn.max()), origin="lower", extent=[x[0], x[-1], z[0], 0], cmap="viridis", vmin=-4, vmax=0.5)
    axs[0, 1].set_title("log10 absorbed energy density (with tip)"); axs[0, 1].set_ylim(-2, 0.3); fig.colorbar(im, ax=axs[0, 1])
    im = axs[1, 0].imshow(np.log10(ratio.T), origin="lower", extent=ext, cmap="RdBu_r", vmin=-1, vmax=1)
    axs[1, 0].set_ylim(-1.5, 0.3); axs[1, 0].set_xlim(-2, 2); axs[1, 0].set_title("log10 (Q_tip / Q_notip): enhancement (red) and shadow (blue)"); fig.colorbar(im, ax=axs[1, 0])
    axs[1, 1].plot(x, col_n / col_n.max(), label="no tip"); axs[1, 1].plot(x, col_t / col_n.max(), label="with tip")
    axs[1, 1].set_xlabel("x (um)"); axs[1, 1].set_ylabel("depth-integrated absorption (norm.)"); axs[1, 1].legend()
    for ax in axs.ravel()[:3]:
        ax.set_xlabel("x (um)"); ax.set_ylabel("z (um)")
    fig.suptitle(f"820 nm pump, p-pol, {cfg.theta_deg:.0f} deg incidence (beam travels to +x), w = {cfg.w_um} um, tip R = {cfg.tip_radius*1e9:.0f} nm, gap {cfg.tip_gap*1e9:.0f} nm")
    fig.savefig(os.path.join(out, "fdtd_pump.png"), dpi=130)


if __name__ == "__main__":
    kw = {}
    for a in sys.argv[1:]:
        k, v = a.split("="); kw[k] = float(v)
    main(**kw)
