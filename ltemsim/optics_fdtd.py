"""2-D FDTD (x–z plane, p-polarisation: Ex, Ez, Hy) for the 820 nm pump field around the
AFM tip on GaAs.

Yee grid, CPML absorbing boundaries, GaAs as a lossy dielectric (n + i kappa at the pump
wavelength), the PtIr tip as a Drude metal (ADE), a p-polarised Gaussian pulse beam at
oblique incidence launched by a soft current line source in the air.

Output: the absorbed energy density Q(x, z) = ∫ sigma |E|^2 dt in the GaAs — the
per-pulse carrier generation profile including the tip's near-field enhancement and shadow.

Units: SI (metres, seconds). Arrays are (Nx, Nz) with Ex at (i+1/2, k), Ez at (i, k+1/2),
Hy at (i+1/2, k+1/2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
import numpy as np

C0 = 2.99792458e8
EPS0 = 8.8541878128e-12
MU0 = 4e-7 * np.pi


@dataclass
class FDTDConfig:
    dx: float = 15e-9               # m, cell size
    x_half: float = 4.0e-6          # m, half width of the physical domain
    z_min: float = -3.0e-6          # m, bottom of the GaAs region
    z_max: float = 2.5e-6           # m, top of the air region
    n_pml: int = 20
    pml_m: int = 3
    # pump
    wavelength: float = 820e-9
    fwhm_fs: float = 100.0          # intensity FWHM
    theta_deg: float = 45.0         # incidence angle from the surface normal (beam travels toward +x); user: 45 deg
    w_um: float = 3.0               # 1/e^2 intensity radius of the beam (projected on the surface)
    x_beam_um: float = 0.0          # where the beam axis hits the surface
    z_source: float = 2.2e-6        # m, height of the source line (air)
    # GaAs at 820 nm
    gaas: bool = True               # False: vacuum run (used to measure the launched energy)
    n_gaas: float = 3.67
    kappa_gaas: float = 0.078       # alpha = 4 pi kappa / lambda = 1.2e4 cm^-1
    # tip (PtIr ~ Pt at 820 nm: eps ~ -17.6 + 29.6 i)
    tip: bool = True
    tip_radius: float = 40e-9
    tip_half_angle_deg: float = 15.0
    tip_gap: float = 20e-9          # apex–surface distance
    tip_shape: str = "cone"         # "cone": apex sphere + cone of tip_half_angle_deg; "rod": apex sphere + straight
                                    #   shank of width 2 * tip_radius (half-angle unused)
    tip_x: float = 0.0
    eps_inf_tip: float = 1.0
    omega_p_tip: float = 1.86e16    # rad/s   (single-frequency Drude fit, see docs)
    gamma_tip: float = 3.66e15      # rad/s
    # run control
    courant: float = 0.5
    t_end_fs: float | None = None   # default: covers the whole pulse plus propagation
    snapshot_every: int = 0


@dataclass
class FDTDResult:
    cfg: FDTDConfig
    x: np.ndarray            # cell-centre x (m)
    z: np.ndarray            # cell-centre z (m)
    Q: np.ndarray            # absorbed energy density per unit y length (J/m^3), cell centres (Nx, Nz)
    Emax2: np.ndarray        # max |E|^2 over time (V/m)^2, cell centres — for the enhancement map
    launched_energy: float   # J/m, energy that crossed a plane just below the source heading down (vacuum estimate)
    absorbed_energy: float   # J/m
    tip_mask_cc: np.ndarray
    gaas_mask_cc: np.ndarray
    snapshots: list = field(default_factory=list)


def _pml_sigma(N, n_pml, m, dx, half_offset):
    """1-D CPML conductivity profile along an axis for N positions (plus `half_offset` cells)."""
    sigma_max = 0.8 * (m + 1) / (376.73 * dx)
    s = np.zeros(N)
    idx = np.arange(N) + half_offset
    # left PML: positions 0 .. n_pml
    rho_l = (n_pml - idx) / n_pml
    s = np.where(rho_l > 0, sigma_max * np.clip(rho_l, 0, 1) ** m, s)
    rho_r = (idx - (N - 1 - n_pml)) / n_pml
    s = np.where(rho_r > 0, sigma_max * np.clip(rho_r, 0, 1) ** m, s)
    return s


def run(cfg: FDTDConfig, verbose=True) -> FDTDResult:
    dx = cfg.dx
    dt = cfg.courant * dx / C0
    Nx = int(round(2 * cfg.x_half / dx)) + 2 * cfg.n_pml
    Nz = int(round((cfg.z_max - cfg.z_min) / dx)) + 2 * cfg.n_pml
    x0 = -cfg.x_half - cfg.n_pml * dx       # x of node i = 0
    z0 = cfg.z_min - cfg.n_pml * dx         # z of node k = 0
    xn = x0 + np.arange(Nx) * dx            # node positions
    zn = z0 + np.arange(Nz) * dx
    xc, zc = xn + dx / 2, zn + dx / 2       # cell centres / half positions
    # ---- material masks at the component positions -----------------------------------------
    eps_gaas = cfg.n_gaas ** 2 - cfg.kappa_gaas ** 2
    omega0 = 2 * np.pi * C0 / cfg.wavelength
    sig_gaas = 2 * cfg.n_gaas * cfg.kappa_gaas * EPS0 * omega0

    def in_tip(X, Z):
        if not cfg.tip:
            return np.zeros_like(X, dtype=bool)
        R = cfg.tip_radius
        th = np.radians(cfg.tip_half_angle_deg)
        zc_apex = cfg.tip_gap + R
        sphere = (X - cfg.tip_x) ** 2 + (Z - zc_apex) ** 2 <= R ** 2
        if cfg.tip_shape == "rod":
            return sphere | ((Z >= zc_apex) & (np.abs(X - cfg.tip_x) <= R))
        if cfg.tip_shape != "cone":
            raise ValueError("tip_shape must be 'cone' or 'rod'")
        z_t = zc_apex + R * np.sin(th)
        cone = (Z >= z_t) & (np.abs(X - cfg.tip_x) <= R * np.cos(th) + (Z - z_t) * np.tan(th))
        return sphere | cone

    Xex, Zex = np.meshgrid(xc, zn, indexing="ij")    # Ex positions
    Xez, Zez = np.meshgrid(xn, zc, indexing="ij")    # Ez positions
    gaas_ex = (Zex < 0) & cfg.gaas
    gaas_ez = (Zez < 0) & cfg.gaas
    tip_ex = in_tip(Xex, Zex) & ~gaas_ex
    tip_ez = in_tip(Xez, Zez) & ~gaas_ez
    eps_ex = np.where(gaas_ex, eps_gaas, 1.0)
    eps_ez = np.where(gaas_ez, eps_gaas, 1.0)
    sig_ex = np.where(gaas_ex, sig_gaas, 0.0)
    sig_ez = np.where(gaas_ez, sig_gaas, 0.0)
    # lossy-dielectric update coefficients
    def coeffs(eps, sig):
        a = sig * dt / (2 * EPS0 * eps)
        return (1 - a) / (1 + a), (dt / (EPS0 * eps)) / (1 + a)
    ca_x, cb_x = coeffs(eps_ex, sig_ex)
    ca_z, cb_z = coeffs(eps_ez, sig_ez)
    # Drude ADE for the tip (J at half time steps)
    g = cfg.gamma_tip
    ka = (1 - g * dt / 2) / (1 + g * dt / 2)
    kb = EPS0 * cfg.omega_p_tip ** 2 * dt / (1 + g * dt / 2)
    # ---- CPML coefficients ------------------------------------------------------------------
    m, npml = cfg.pml_m, cfg.n_pml
    # x-derivatives: in the Hy update (d Ez/dx at i+1/2) and the Ez update (d Hy/dx at i)
    sx_h = _pml_sigma(Nx, npml, m, dx, 0.5)
    sx_e = _pml_sigma(Nx, npml, m, dx, 0.0)
    sz_h = _pml_sigma(Nz, npml, m, dx, 0.5)
    sz_e = _pml_sigma(Nz, npml, m, dx, 0.0)
    def bc(s):
        b = np.exp(-s * dt / EPS0)
        c = np.where(s > 0, b - 1.0, 0.0)
        return b, c
    bx_h, cx_h = bc(sx_h); bx_e, cx_e = bc(sx_e)
    bz_h, cz_h = bc(sz_h); bz_e, cz_e = bc(sz_e)
    # ---- fields -----------------------------------------------------------------------------
    Ex = np.zeros((Nx, Nz)); Ez = np.zeros((Nx, Nz)); Hy = np.zeros((Nx, Nz))
    Jx = np.zeros((Nx, Nz)); Jz = np.zeros((Nx, Nz))
    psi_hy_x = np.zeros((Nx, Nz)); psi_hy_z = np.zeros((Nx, Nz))
    psi_ex_z = np.zeros((Nx, Nz)); psi_ez_x = np.zeros((Nx, Nz))
    Qx = np.zeros((Nx, Nz)); Qz = np.zeros((Nx, Nz))
    Emax2 = np.zeros((Nx, Nz))
    # ---- source -----------------------------------------------------------------------------
    th = np.radians(cfg.theta_deg)
    ks = int(round((cfg.z_source - z0) / dx))
    sigE = cfg.fwhm_fs * 1e-15 / 2.3548 * np.sqrt(2.0)      # field envelope sigma
    t0 = 4 * sigE
    w = cfg.w_um * 1e-6
    xs = cfg.x_beam_um * 1e-6 - cfg.z_source * np.tan(th)    # beam axis at the source height
    i_in = np.arange(npml, Nx - npml)
    ampx = np.exp(-(((xc[i_in] - xs) * np.cos(th)) / w) ** 2)   # Ex component sits at xc
    ampz = np.exp(-(((xn[i_in] - xs) * np.cos(th)) / w) ** 2)
    ex_dir, ez_dir = np.cos(th), np.sin(th)                     # polarisation ⊥ (sin th, -cos th)
    delay_x = xc[i_in] * np.sin(th) / C0
    delay_z = xn[i_in] * np.sin(th) / C0
    J0 = 1.0
    t_end = cfg.t_end_fs * 1e-15 if cfg.t_end_fs else (t0 + 4 * sigE + (cfg.z_source - cfg.z_min) * 2 / C0 + 2e-14)
    nsteps = int(t_end / dt)
    # launched-energy estimate: downward Poynting flux through the plane 2 cells below the source
    k_flux = ks - 3
    Sz_acc = 0.0
    t_wall = time.time()
    if verbose:
        print(f"FDTD: {Nx}x{Nz} cells, dx={dx*1e9:.0f} nm, dt={dt*1e18:.1f} as, {nsteps} steps, "
              f"source line k={ks}, tip={cfg.tip}")
    for n in range(nsteps):
        t = n * dt
        # ---- H update ---------------------------------------------------------------------
        dEx_dz = np.zeros_like(Hy); dEz_dx = np.zeros_like(Hy)
        dEx_dz[:, :-1] = (Ex[:, 1:] - Ex[:, :-1]) / dx
        dEz_dx[:-1, :] = (Ez[1:, :] - Ez[:-1, :]) / dx
        psi_hy_z = bz_h[None, :] * psi_hy_z + cz_h[None, :] * dEx_dz
        psi_hy_x = bx_h[:, None] * psi_hy_x + cx_h[:, None] * dEz_dx
        Hy -= (dt / MU0) * ((dEx_dz + psi_hy_z) - (dEz_dx + psi_hy_x))
        # ---- Drude currents (half step) ---------------------------------------------------
        Jx = ka * Jx + kb * Ex * tip_ex
        Jz = ka * Jz + kb * Ez * tip_ez
        # ---- soft source --------------------------------------------------------------------
        env_x = np.exp(-((t - t0 - delay_x) ** 2) / (2 * sigE ** 2)) * np.cos(omega0 * (t - t0 - delay_x))
        env_z = np.exp(-((t - t0 - delay_z) ** 2) / (2 * sigE ** 2)) * np.cos(omega0 * (t - t0 - delay_z))
        Jsx = np.zeros(Nx); Jsz = np.zeros(Nx)
        Jsx[i_in] = J0 * ex_dir * ampx * env_x
        Jsz[i_in] = J0 * ez_dir * ampz * env_z
        # ---- E update ---------------------------------------------------------------------
        dHy_dz = np.zeros_like(Ex); dHy_dx = np.zeros_like(Ez)
        dHy_dz[:, 1:] = (Hy[:, 1:] - Hy[:, :-1]) / dx
        dHy_dx[1:, :] = (Hy[1:, :] - Hy[:-1, :]) / dx
        psi_ex_z = bz_e[None, :] * psi_ex_z + cz_e[None, :] * dHy_dz
        psi_ez_x = bx_e[:, None] * psi_ez_x + cx_e[:, None] * dHy_dx
        curl_x = -(dHy_dz + psi_ex_z)
        curl_z = (dHy_dx + psi_ez_x)
        Jx_tot = Jx.copy(); Jz_tot = Jz.copy()
        Jx_tot[:, ks] += Jsx
        Jz_tot[:, ks] += Jsz
        Ex = ca_x * Ex + cb_x * (curl_x - Jx_tot)
        Ez = ca_z * Ez + cb_z * (curl_z - Jz_tot)
        # ---- diagnostics --------------------------------------------------------------------
        Qx += sig_ex * Ex ** 2 * dt
        Qz += sig_ez * Ez ** 2 * dt
        E2 = Ex ** 2 + Ez ** 2
        np.maximum(Emax2, E2, out=Emax2)
        Sz_acc += -np.sum(Ex[i_in, k_flux] * Hy[i_in, k_flux]) * dx * dt   # S_z = -Ex Hy for (Ex, Ez, Hy)
        if cfg.snapshot_every and n % cfg.snapshot_every == 0:
            pass
        if verbose and n % 2000 == 0:
            print(f"  step {n}/{nsteps}  t={t*1e15:.0f} fs  max|E|={np.sqrt(E2.max()):.3e}  ({time.time()-t_wall:.0f}s)")
    # ---- outputs at cell centres --------------------------------------------------------------
    Q = np.zeros((Nx, Nz)); Em = np.zeros((Nx, Nz))
    Q[:, :-1] = 0.5 * (Qx[:, :-1] + Qx[:, 1:])
    Q[:-1, :] += 0.5 * (Qz[:-1, :] + Qz[1:, :])
    Em[:, :-1] = Emax2[:, :-1]
    gaas_cc = ((zc < 0)[None, :] & np.ones((Nx, 1), bool)) & cfg.gaas
    Xcc, Zcc = np.meshgrid(xc, zc, indexing="ij")
    tip_cc = in_tip(Xcc, Zcc) & ~gaas_cc
    sl = (slice(npml, Nx - npml), slice(npml, Nz - npml))
    absorbed = float(np.sum(Q[sl][gaas_cc[sl]]) * dx * dx)
    return FDTDResult(cfg=cfg, x=xc[npml:Nx - npml], z=zc[npml:Nz - npml], Q=Q[sl], Emax2=Em[sl],
                      launched_energy=float(Sz_acc), absorbed_energy=absorbed,
                      tip_mask_cc=tip_cc[sl], gaas_mask_cc=gaas_cc[sl])


def fresnel_p_absorption(n, kappa, theta_deg):
    """1 - |r_p|^2 for air -> (n + i kappa) at angle theta (from the normal)."""
    th = np.radians(theta_deg)
    N = n + 1j * kappa
    cos_t = np.sqrt(1 - (np.sin(th) / N) ** 2)
    rp = (N * np.cos(th) - cos_t) / (N * np.cos(th) + cos_t)
    return 1 - abs(rp) ** 2
