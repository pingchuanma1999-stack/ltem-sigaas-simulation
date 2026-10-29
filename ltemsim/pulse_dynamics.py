"""Single-pulse carrier dynamics in the biased gap (bare applied field, no traps):
a two-moment hydrodynamic model (density + drift velocity with inertia) for electrons
and holes, self-consistent with Poisson at every step.

    dn/dt = G(x,z,t) - div(n v_n) + div(D_n grad n)
    dv_n/dt = -(e/m_n) E - v_n (1 + |v_n|/v_sat) / tau_n
    dp/dt = G - div(p v_p) + div(D_p grad p)
    dv_p/dt = +(e/m_p) E - v_p (1 + |v_p|/v_sat) / tau_p
    E = E_applied + E_space_charge,   div(eps grad phi_sc) = -e (p - n) / eps0

Diffusion: D = mu kT/q by default; with degenerate_diffusion=True the generalised
Einstein relation D(n) = mu (kT/q) F_1/2(eta)/F_-1/2(eta) is applied face by face with
eta from the local density (parabolic band, N_c = 4.4e17 cm^-3 for electrons at 300 K),
so the Fermi pressure of the 1e18-1e19 cm^-3 photo-plasma drives the lateral spreading
(D_n ~ 6-10x the classical value at the spot centre, classical at the flanks).

Explicit time stepping (0.5 fs), first-order upwind fluxes, cell-centred finite
volumes on the same kind of tensor grid as the steady-state solver; Poisson is the
existing finite-volume operator (GaAs/air interface, electrodes as equipotentials).

Outputs: total current per unit y length J(t) (far-field reference ∝ dJ/dt) and the
quasi-static THz near field E_z(x_tip, h, t) above the surface for a set of heights
(computed from the instantaneous charge distribution with the dielectric half-space
image rule), ready for the tip-scattering/demodulation stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
import numpy as np

from .params import Params, Q as QE, EPS0 as EPS0_CM
from .grid import Grid, _graded
from .fv import Topology
from .physics import Material
from .poisson import PoissonOperator

ME = 9.1093837e-31   # kg


def _fermi_dirac(j, eta):
    """Complete Fermi-Dirac integral F_j(eta) = (1/Gamma(j+1)) int_0^inf x^j / (1 + exp(x - eta)) dx,
    by quadrature in u = sqrt(x) (removes the x^-1/2 singularity): F_j = (2/Gamma(j+1)) int u^(2j+1) / (1 + exp(u^2 - eta)) du."""
    from scipy.special import gamma
    u = np.linspace(0, np.sqrt(max(80.0, eta.max() + 40.0)), 40001)
    out = np.empty_like(eta)
    for i, e in enumerate(eta):
        with np.errstate(over="ignore"):
            f = 2.0 * u ** (2 * j + 1) / (1.0 + np.exp(u ** 2 - e))
        out[i] = np.trapezoid(f, u)
    return out / gamma(j + 1)


def _einstein_factor_table(eta_min=-15.0, eta_max=80.0, n_eta=400):
    """Table of ln(n/N_c) -> F_1/2(eta)/F_-1/2(eta) (generalised Einstein relation, parabolic band)."""
    eta = np.linspace(eta_min, eta_max, n_eta)
    f12 = _fermi_dirac(0.5, eta)
    fm12 = _fermi_dirac(-0.5, eta)
    r = np.log(f12)                     # ln(n/N_c) = ln F_1/2(eta), monotone in eta
    return r, f12 / fm12


@dataclass
class PulseConfig:
    V_bias: float = 25.0
    anode_side: str = "-x"
    # excitation
    P_avg_mW: float = 2.0
    rep_rate: float = 80e6
    fwhm_fs: float = 100.0
    w_um: float = 3.0                 # 1/e^2 intensity radius projected on the surface along x
    theta_deg: float = 45.0           # sets the spot extent along y: w_y = w_x cos(theta)
    absorbed_fraction: float = 0.8    # (1-R) for p-pol at theta, from the FDTD/Fresnel
    alpha_cm: float = 1.2e4
    x_beam_um: float = 0.0            # beam centre; the tip is at x = 0
    n_sat: float = 2e19               # cm^-3, soft cap of the per-pulse density (absorption saturation)
    # transport
    m_n: float = 0.067
    m_p: float = 0.5
    tau_n_ps: float = 0.1             # momentum relaxation; 0.2 ps = low-field mobility value, 0.1 ps ~ plasmon damping at 1e18-1e19
    tau_p_ps: float = 0.1
    v_sat: float = 1.0e7              # cm/s
    T: float = 300.0
    degenerate_diffusion: bool = False   # generalised Einstein relation D = mu (kT/q) F_1/2(eta)/F_-1/2(eta), eta from the local density
    # tip as an equipotential body in the Poisson solve (per-tip-position chain)
    tip_in_poisson: bool = False
    tip_height_nm: float = 100.0      # apex-surface distance
    tip_half_angle_deg: float = 15.0
    tip_apex_radius_um: float = 0.04
    tip_shape: str = "cone"           # "cone" (half-angle above) or "rod": straight shank of width 2 * tip_apex_radius_um
    V_tip: float = 0.0                # tip potential relative to the gap midpoint (V); 0 = grounded tip, symmetric bias
    applied_vertical: bool = True     # False: drop the vertical component of the applied (electrode) field -> isolates field-line bending
    # run
    dt_fs: float = 0.5
    t_end_ps: float = 4.0
    record_every_fs: float = 5.0
    tip_x_um: float = 0.0
    heights_nm: tuple = tuple(np.concatenate([np.arange(1, 30, 2), np.arange(30, 200, 5), np.arange(200, 1001, 50)]).tolist())
    snapshot_times_ps: tuple = (0.1, 0.3, 1.0, 3.0)
    # grid
    x_fine: float = 8.0
    dx_fine: float = 0.1
    x_half: float = 60.0
    depth: float = 30.0
    air_height: float = 30.0


def build_grid(cfg: PulseConfig) -> Grid:
    xf = np.arange(-cfg.x_fine, cfg.x_fine + 1e-9, cfg.dx_fine)
    right = _graded(cfg.x_fine, cfg.x_half, cfg.dx_fine, 2.0)
    xe = np.concatenate([(-right)[::-1], xf, right])
    # z (GaAs): 0.02 um within 0.3 um of the surface, 0.1 um to 2 um, graded beyond
    z1 = np.arange(0.0, 0.3 + 1e-9, 0.02)
    z2 = np.arange(0.3, 2.0 + 1e-9, 0.1)[1:]
    z3 = _graded(2.0, cfg.depth, 0.1, 3.0)
    zs = np.concatenate([z1, z2, z3])                 # positive depths 0 .. depth
    # z (air): 0.02 um to 0.5 um (tapping range), 0.05 um to 1 um, 0.1 um to 2 um, graded beyond
    za1 = np.arange(0.0, 0.5 + 1e-9, 0.02)
    za2 = np.arange(0.5, 1.0 + 1e-9, 0.05)[1:]
    za2b = np.arange(1.0, 2.0 + 1e-9, 0.1)[1:]
    za3 = _graded(2.0, cfg.air_height, 0.1, 3.0)
    za = np.concatenate([za1, za2, za2b, za3])
    ze = np.concatenate([(-zs)[::-1], za[1:]])
    ks = int(np.argmin(np.abs(ze)))
    ze[ks] = 0.0
    return Grid(xe=xe, ze=ze, ks=ks)


class PulseSim:
    def __init__(self, cfg: PulseConfig, verbose=True):
        self.cfg = cfg
        self.verbose = verbose
        p = Params(V_bias=cfg.V_bias, anode_side=cfg.anode_side, T=cfg.T)
        self.p = p
        self.grid = build_grid(cfg)
        g = self.grid
        self.top = Topology(g, p.gap)
        self.mat = Material(p)
        self.top.set_eps(p.eps_r, 1.0)
        self.pois = PoissonOperator(self.top, self.mat.sc.lam2)
        self.Ns = self.top.N_semi
        self.Nx, self.Nzs = g.Nx, g.Nzs
        # geometry in cm
        self.dx = g.dx * 1e-4
        self.dz = g.dz[: g.ks] * 1e-4
        self.xc = g.xc * 1e-4
        self.zc = g.zc[: g.ks] * 1e-4
        self.area = np.outer(self.dz, self.dx)            # cm^2 (per unit y), shape (Nzs, Nx)
        # applied (bare) field, V/cm, on semiconductor cells
        V_left, V_right = (+cfg.V_bias / 2, -cfg.V_bias / 2) if cfg.anode_side == "-x" else (-cfg.V_bias / 2, +cfg.V_bias / 2)
        VT = self.mat.sc.VT
        Vl, Vr = self.mat.psi_eq + V_left / VT, self.mat.psi_eq + V_right / VT
        self.b_applied = self.pois.dirichlet_rhs(Vl, Vr)
        self._setup_tip(VT)
        psi0 = self.solve_poisson(self.b_applied, np.zeros(self.top.N_all), applied=True)
        Ex0, Ez0 = self.pois.field(psi0)
        E0 = self.mat.sc.E0
        self.Ex_app = Ex0[: g.ks] * E0
        self.Ez_app = Ez0[: g.ks] * E0
        self.b_zero = np.zeros(self.top.N_all)
        # material
        kT = 1.380649e-23 * cfg.T
        self.VT = VT
        self.mu_n = QE * cfg.tau_n_ps * 1e-12 / (cfg.m_n * ME) * 1e4     # cm^2/Vs
        self.mu_p = QE * cfg.tau_p_ps * 1e-12 / (cfg.m_p * ME) * 1e4
        self.D_n, self.D_p = self.mu_n * VT, self.mu_p * VT
        self.qm_n = QE / (cfg.m_n * ME) * 1e4        # (cm/s^2) per (V/cm): (e/m)[C/kg] * 100 [V/m per V/cm] * 100 [cm per m]
        self.qm_p = QE / (cfg.m_p * ME) * 1e4
        # degenerate (Fermi-Dirac) diffusion: D(n) = D0 * g(n/N_c), g = F_1/2(eta)/F_-1/2(eta) -> 1 for n << N_c, (2/3) eta for n >> N_c
        h_planck = 6.62607015e-34
        self.N_c = 2 * (2 * np.pi * cfg.m_n * ME * kT / h_planck ** 2) ** 1.5 * 1e-6      # cm^-3
        self.N_v = 2 * (2 * np.pi * cfg.m_p * ME * kT / h_planck ** 2) ** 1.5 * 1e-6
        self._g_table = _einstein_factor_table()
        if verbose:
            print(f"pulse grid: Nx={g.Nx}, Nz semi={g.Nzs} (dz_min {g.dz.min():.3f} um), cells {self.Ns}; "
                  f"mu_n={self.mu_n:.0f}, mu_p={self.mu_p:.0f} cm^2/Vs; E_applied at centre {self.Ex_app[-1, np.argmin(np.abs(self.xc))]/1e3:.2f} kV/cm")
            if cfg.degenerate_diffusion:
                print(f"  degenerate diffusion on: N_c={self.N_c:.2e}, N_v={self.N_v:.2e} cm^-3; D_n at 1e18/1e19/2e19 = "
                      + "/".join(f"{self.D_n * self.einstein_factor(nn, self.N_c):.0f}" for nn in (1e18, 1e19, 2e19)) + " cm^2/s")

    def einstein_factor(self, dens, N_eff):
        """F_1/2(eta)/F_-1/2(eta) for the local reduced density dens/N_eff (parabolic band, lattice temperature)."""
        r, gval = self._g_table
        return np.interp(np.log(np.maximum(dens, 1e-30) / N_eff), r, gval, left=1.0, right=gval[-1])

    # ------------------------------------------------------------ tip as an equipotential in Poisson
    def _setup_tip(self, VT):
        """Mark the air cells inside the tip (apex at tip_height above the surface, half-angle cone, one cell wide
        at least) and build the Poisson matrix with those cells held at a fixed potential (Dirichlet rows)."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        cfg, g, top = self.cfg, self.grid, self.top
        self.tip_cells = np.zeros(0, dtype=int)
        self.V_tip_scaled = cfg.V_tip / VT
        if not cfg.tip_in_poisson:
            self._lu_tip = None
            return
        h = cfg.tip_height_nm * 1e-3                       # um
        th = np.radians(cfg.tip_half_angle_deg)
        R = cfg.tip_apex_radius_um
        zb = g.ze[:-1]                                     # bottom edge of each cell row (um)
        rows = np.where((zb >= h - 1e-9) & (np.arange(g.Nz) >= g.ks))[0]
        cells = []
        if cfg.tip_shape not in ("cone", "rod"):
            raise ValueError("tip_shape must be 'cone' or 'rod'")
        for k in rows:
            if cfg.tip_shape == "rod":
                half = R
            else:
                half = max(R, R + (g.zc[k] - h - R) * np.tan(th)) if g.zc[k] > h + R else R
            half = max(half, 0.5 * g.dx[np.argmin(np.abs(g.xc - cfg.tip_x_um))] + 1e-9)
            cols = np.where(np.abs(g.xc - cfg.tip_x_um) <= half)[0]
            cells.append(k * g.Nx + cols)
        self.tip_cells = np.concatenate(cells) if cells else np.zeros(0, dtype=int)
        A = (self.pois.lam2 * self.pois.L).tolil()
        for c in self.tip_cells:
            A.rows[c] = [int(c)]
            A.data[c] = [1.0]
        self._A_tip = A.tocsc()
        self._lu_tip = spla.splu(self._A_tip)
        self._tip_mask = np.zeros(top.N_all, dtype=bool); self._tip_mask[self.tip_cells] = True
        if self.verbose:
            print(f"  tip in Poisson: apex {cfg.tip_height_nm:.0f} nm above the surface, {self.tip_cells.size} cells, V_tip = {cfg.V_tip:g} V")

    def solve_poisson(self, b, rho, applied=False):
        """Linear Poisson with the electrodes (via b) and, if enabled, the tip cells at V_tip (applied=True) or 0 (space charge)."""
        if self._lu_tip is None:
            return self.pois.solve_linear(b, rho)
        rhs = -(self.pois.lam2 * b + self.pois.vol * rho)
        rhs[self.tip_cells] = self.V_tip_scaled if applied else 0.0
        return self._lu_tip.solve(rhs)

    def tip_charge(self, psi, b, rho):
        """Charge induced on the tip (C per cm of y) = minus the residual of the free Poisson equation on the tip cells."""
        if self.tip_cells.size == 0:
            return 0.0
        res = self.pois.lam2 * (self.pois.L @ psi + b) + self.pois.vol * rho      # = 0 on free cells
        q_scaled = -res[self.tip_cells].sum()                                      # units: C0 * um^2 (per unit y)
        return float(q_scaled * QE * self.mat.sc.C0 * 1e-8)

    # ------------------------------------------------------------ generation
    def generation_profile(self, ratio_map=None):
        """Pairs per cm^3 per pulse, (Nzs, Nx). ratio_map(x_um, z_um) -> tip correction factor (optional)."""
        cfg = self.cfg
        E_pulse = cfg.P_avg_mW * 1e-3 / cfg.rep_rate
        hv = 6.62607015e-34 * 2.99792458e8 / 820e-9
        wx = cfg.w_um * 1e-4
        wy = wx * np.cos(np.radians(cfg.theta_deg))
        F_peak = 2 * E_pulse / (np.pi * wx * wy)                         # J/cm^2
        N0 = F_peak * cfg.absorbed_fraction / hv * cfg.alpha_cm          # cm^-3 at the surface
        X, Z = np.meshgrid(self.xc, self.zc)
        N_lin = N0 * np.exp(-2 * ((X - cfg.x_beam_um * 1e-4) / wx) ** 2) * np.exp(cfg.alpha_cm * Z)   # Z < 0
        if ratio_map is not None:
            N_lin = N_lin * ratio_map(X * 1e4, Z * 1e4)
        N = cfg.n_sat * (1 - np.exp(-N_lin / cfg.n_sat))                 # absorption saturation
        self.N0_lin = N0
        return N

    # ------------------------------------------------------------ helpers
    def _face_fields(self, psi_scaled):
        """E (V/cm) on x faces (Nzs, Nx-1) and z faces (Nzs-1, Nx) from the scaled potential on semiconductor cells."""
        VT = self.VT
        P = psi_scaled[: self.Ns].reshape(self.Nzs, self.Nx) * VT
        Exf = -(P[:, 1:] - P[:, :-1]) / self.dcx[None, :]
        Ezf = -(P[1:, :] - P[:-1, :]) / self.dcz[:, None]
        return Exf, Ezf

    def _div_flux(self, dens, vxf, vzf, D, N_eff=None):
        """div(dens v) - div(D grad dens): face velocities, upwind advection, central diffusion (per cm^3 per s).
        With N_eff given, D is multiplied on every face by the generalised Einstein factor of the face density."""
        dx, dz = self.dx, self.dz
        up = np.where(vxf > 0, dens[:, :-1], dens[:, 1:])
        Dx = D if N_eff is None else D * self.einstein_factor(0.5 * (dens[:, 1:] + dens[:, :-1]), N_eff)
        Fx = vxf * up - Dx * (dens[:, 1:] - dens[:, :-1]) / self.dcx[None, :]
        divx = np.zeros_like(dens)
        divx[:, :-1] += Fx * dz[:, None]
        divx[:, 1:] -= Fx * dz[:, None]
        upz = np.where(vzf > 0, dens[:-1, :], dens[1:, :])
        Dz = D if N_eff is None else D * self.einstein_factor(0.5 * (dens[1:, :] + dens[:-1, :]), N_eff)
        Fz = vzf * upz - Dz * (dens[1:, :] - dens[:-1, :]) / self.dcz[:, None]
        divz = np.zeros_like(dens)
        divz[:-1, :] += Fz * dx[None, :]
        divz[1:, :] -= Fz * dx[None, :]
        return (divx + divz) / self.area, Fx, Fz

    def nearfield_Ez(self, rho_sheet, heights_cm, x_t):
        """E_z (V/cm) in the air at (x_t, h) from the charge distribution rho_sheet = e(p-n)*area (C/cm per cell),
        dielectric half-space: a charge inside GaAs acts on the air side as 2/(eps_r+1) times itself."""
        eps_r = self.p.eps_r
        f = 2.0 / (eps_r + 1.0)
        X, Z = np.meshgrid(self.xc, self.zc)
        dxs = x_t - X.ravel()
        q = rho_sheet.ravel()
        out = np.empty(len(heights_cm))
        for j, h in enumerate(heights_cm):
            dzs = h - Z.ravel()
            out[j] = np.sum(q * dzs / (dxs ** 2 + dzs ** 2))
        return f * out / (2 * np.pi * EPS0_CM)

    # ------------------------------------------------------------ run
    def run(self, N_pairs):
        cfg = self.cfg
        g = self.grid
        shape = (self.Nzs, self.Nx)
        self.dcx = 0.5 * (self.dx[1:] + self.dx[:-1])
        self.dcz = 0.5 * (self.dz[1:] + self.dz[:-1])
        n = np.zeros(shape); p = np.zeros(shape)
        vnx = np.zeros((self.Nzs, self.Nx - 1)); vnz = np.zeros((self.Nzs - 1, self.Nx))
        vpx = np.zeros_like(vnx); vpz = np.zeros_like(vnz)
        dt = cfg.dt_fs * 1e-15
        nsteps = int(cfg.t_end_ps * 1e-12 / dt)
        sigI = cfg.fwhm_fs * 1e-15 / 2.3548
        t0 = 3.5 * sigI
        rec_every = max(1, int(round(cfg.record_every_fs * 1e-15 / dt)))
        heights = np.array(cfg.heights_nm) * 1e-7      # cm
        C0 = self.mat.sc.C0
        tau_n, tau_p = cfg.tau_n_ps * 1e-12, cfg.tau_p_ps * 1e-12
        rec = dict(t=[], Jx=[], Jz=[], Ez_tip=[], Ex_center=[], n_center=[], snapshots={})
        snap_steps = {int(round(ts * 1e-12 / dt)): ts for ts in cfg.snapshot_times_ps}
        ic = np.argmin(np.abs(self.xc)); kc = self.Nzs - 1
        t_wall = time.time()
        rho_all = np.zeros(self.top.N_all)
        psi_app = self.solve_poisson(self.b_applied, np.zeros(self.top.N_all), applied=True)
        Exf, Ezf = self._face_fields(psi_app)
        if not cfg.applied_vertical:
            Ezf = np.zeros_like(Ezf)
        Exf_app, Ezf_app = Exf.copy(), Ezf.copy()
        rec["Q_tip"] = []; rec["E_gap"] = []
        rec["Q_tip_static"] = self.tip_charge(psi_app, self.b_applied, np.zeros(self.top.N_all))
        # gap cell: the air cell just below the apex at the tip column (for E_gap); surface cell below it
        ic_tip = np.argmin(np.abs(self.xc - cfg.tip_x_um * 1e-4))
        if self.tip_cells.size:
            k_apex = int(self.tip_cells.min() // self.Nx)
            k_gap = max(k_apex - 1, g.ks)
        else:
            k_gap = g.ks + int(np.argmin(np.abs(g.zc[g.ks:] - cfg.tip_height_nm * 1e-3)))
        i_gap = k_gap * self.Nx + ic_tip; i_surf = (g.ks - 1) * self.Nx + ic_tip
        dz_gap = (g.zc[k_gap] - g.zc[g.ks - 1]) * 1e-4
        for step in range(nsteps):
            t = step * dt
            gt = np.exp(-((t - t0) ** 2) / (2 * sigI ** 2)) / (sigI * np.sqrt(2 * np.pi))
            G = N_pairs * gt
            # momentum (face velocities, semi-implicit damping with velocity-dependent relaxation)
            dn_x = (1 + np.abs(vnx) / cfg.v_sat) / tau_n; dn_z = (1 + np.abs(vnz) / cfg.v_sat) / tau_n
            vnx = (vnx - dt * self.qm_n * Exf) / (1 + dt * dn_x); vnz = (vnz - dt * self.qm_n * Ezf) / (1 + dt * dn_z)
            dp_x = (1 + np.abs(vpx) / cfg.v_sat) / tau_p; dp_z = (1 + np.abs(vpz) / cfg.v_sat) / tau_p
            vpx = (vpx + dt * self.qm_p * Exf) / (1 + dt * dp_x); vpz = (vpz + dt * self.qm_p * Ezf) / (1 + dt * dp_z)
            # continuity
            divn, Fnx, Fnz = self._div_flux(n, vnx, vnz, self.D_n, self.N_c if cfg.degenerate_diffusion else None)
            divp, Fpx, Fpz = self._div_flux(p, vpx, vpz, self.D_p, self.N_v if cfg.degenerate_diffusion else None)
            n = np.maximum(n + dt * (G - divn), 0.0)
            p = np.maximum(p + dt * (G - divp), 0.0)
            # Poisson: space-charge potential (electrodes at the applied potentials by superposition)
            rho_all[: self.Ns] = ((p - n) / C0).ravel()
            psi_sc = self.solve_poisson(self.b_zero, rho_all)
            Exs, Ezs = self._face_fields(psi_sc)
            Exf = Exf_app + Exs; Ezf = Ezf_app + Ezs
            if step % rec_every == 0:
                rec["Q_tip"].append(self.tip_charge(psi_sc, self.b_zero, rho_all))
                rec["E_gap"].append(-(psi_sc[i_gap] - psi_sc[i_surf]) * self.VT / dz_gap)
                # total current per unit y length: the x-current through a vertical cut, averaged over cuts (A/cm);
                # for the far field only its time derivative matters
                Jx = QE * np.sum((Fpx - Fnx) * self.dz[:, None], axis=0).mean()
                Jz = QE * np.sum((Fpz - Fnz) * self.dx[None, :], axis=1).mean()
                rho_sheet = QE * (p - n) * self.area
                rec["t"].append(t); rec["Jx"].append(Jx); rec["Jz"].append(Jz)
                rec["Ez_tip"].append(self.nearfield_Ez(rho_sheet, heights, cfg.tip_x_um * 1e-4))
                rec["Ex_center"].append(0.5 * (Exf[kc, ic - 1] + Exf[kc, ic])); rec["n_center"].append(n[kc, ic])
            if step in snap_steps:
                Exc = np.zeros(shape); Exc[:, 1:-1] = 0.5 * (Exf[:, 1:] + Exf[:, :-1]); Exc[:, 0] = Exf[:, 0]; Exc[:, -1] = Exf[:, -1]
                Ezc = np.zeros(shape); Ezc[1:-1, :] = 0.5 * (Ezf[1:, :] + Ezf[:-1, :]); Ezc[0] = Ezf[0]; Ezc[-1] = Ezf[-1]
                rec["snapshots"][snap_steps[step]] = dict(n=n.copy(), p=p.copy(), Ex=Exc, Ez=Ezc)
            if self.verbose and step % 2000 == 0:
                print(f"  t={t*1e12:.2f} ps  n_max={n.max():.2e}  E_x(centre,surf)={0.5*(Exf[kc, ic-1]+Exf[kc, ic])/1e3:.2f} kV/cm  ({time.time()-t_wall:.0f}s)")
        for k in ("t", "Jx", "Jz", "Ex_center", "n_center", "Q_tip", "E_gap"):
            rec[k] = np.array(rec[k])
        rec["Ez_tip"] = np.array(rec["Ez_tip"])          # (nt, nh), V/cm
        rec["heights_nm"] = np.array(cfg.heights_nm)
        rec["N_pairs"] = N_pairs
        rec["xc_um"] = self.xc * 1e4; rec["zc_um"] = self.zc * 1e4
        rec["Ex_app"] = self.Ex_app; rec["Ez_app"] = self.Ez_app
        return rec
