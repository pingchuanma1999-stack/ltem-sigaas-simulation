"""Emission proxy and near-field tip response.

Next-pulse emission (impulse approximation): the fresh photocarriers of the
next pulse, n_pulse(x, z) ∝ generation profile, are accelerated by the local
steady-state field E(x, z).  The initial polarisation current is therefore
P(x, z) ∝ n_pulse(x, z) · E(x, z); the THz field radiated in the first ~100 fs
(before the plasma screens the field) is ∝ dP/dt ∝ P.

The tip is treated as a passive E_z sensor at height h above the surface: the
quasi-static field of the bound charge -div P (with the Au electrodes as
grounded conductors and the GaAs/air dielectric contrast included) is solved
with the same finite-volume Poisson operator, and E_z is read at the apex.

Mapping to the experiment: the spot is fixed at the gap centre in the model and
the tip is moved; the experimental coordinate X (spot position relative to the
tip) is X = -x_tip.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .solver import Simulation, Solution


@dataclass
class TipScan:
    x_tip: np.ndarray        # um, tip position in the model frame
    X_exp: np.ndarray        # um, experimental coordinate (= -x_tip)
    S: np.ndarray            # signed E_z at the apex (arbitrary units)
    Sx: np.ndarray           # contribution of the in-plane polarisation only
    Sz: np.ndarray           # contribution of the vertical polarisation only
    h_um: float
    Dx: float                # total in-plane dipole (far-field proxy, arbitrary units)
    Dz: float


def polarisation(sim: Simulation, sol: Solution):
    """P = n_pulse * E on semiconductor cells (arbitrary units; E in kV/cm, n_pulse peak-normalised)."""
    g = sim.grid
    Ex, Ez = sol.fields_physical(sim)
    npulse = (sol.G / sol.G.max()).reshape(g.Nzs, g.Nx)
    Px = npulse * Ex[: g.ks, :]
    Pz = npulse * Ez[: g.ks, :]
    return Px, Pz, npulse


def bound_charge(sim: Simulation, Px, Pz):
    """rho_b = -div P on all cells (zero in air; the surface charge P_z lands in the top GaAs row)."""
    g = sim.grid
    ks, Nx = g.ks, g.Nx
    dx, dz = g.dx, g.dz[:ks]
    xc, zc = g.xc, g.zc[:ks]
    rho = np.zeros((g.Nz, Nx))
    # x faces (interior): flux = average P_x * dz
    Pxf = 0.5 * (Px[:, 1:] + Px[:, :-1])
    flux_x = Pxf * dz[:, None]
    divx = np.zeros((ks, Nx))
    divx[:, :-1] += flux_x
    divx[:, 1:] -= flux_x
    # z faces (interior)
    Pzf = 0.5 * (Pz[1:, :] + Pz[:-1, :])
    flux_z = Pzf * dx[None, :]
    divz = np.zeros((ks, Nx))
    divz[:-1, :] += flux_z
    divz[1:, :] -= flux_z
    # surface face: P outside = 0 -> flux = P_z(top cell) * dx
    divz[-1, :] += Pz[-1, :] * dx
    vol = np.outer(dz, dx)
    rho[:ks, :] = -(divx + divz) / vol
    return rho.ravel()


def apex_field(sim: Simulation, rho_b, x_tip_um, h_um):
    """E_z (arbitrary units) at (x_tip, h) from the bound charge, electrodes grounded."""
    pois = sim.pois
    g = sim.grid
    b0 = np.zeros(sim.top.N_all)
    phi = pois.solve_linear(b0, rho_b)
    _, Ez = pois.field(phi)
    k = g.ks + int(np.argmin(np.abs(g.zc[g.ks:] - h_um)))
    return np.interp(x_tip_um, g.xc, Ez[k, :])


def tip_scan(sim: Simulation, sol: Solution, x_tip=None, h_um=None) -> TipScan:
    p = sim.p
    g = sim.grid
    x_tip = np.asarray(p.tip_x_scan_um if x_tip is None else x_tip, dtype=float)
    h = p.tip_height_um if h_um is None else h_um
    Px, Pz, npulse = polarisation(sim, sol)
    rho_total = bound_charge(sim, Px, Pz)
    rho_x = bound_charge(sim, Px, np.zeros_like(Pz))
    rho_z = bound_charge(sim, np.zeros_like(Px), Pz)
    S = apex_field(sim, rho_total, x_tip, h)
    Sx = apex_field(sim, rho_x, x_tip, h)
    Sz = apex_field(sim, rho_z, x_tip, h)
    vol = np.outer(g.dz[: g.ks], g.dx)
    return TipScan(x_tip=x_tip, X_exp=-x_tip, S=S, Sx=Sx, Sz=Sz, h_um=h,
                   Dx=float(np.sum(Px * vol)), Dz=float(np.sum(Pz * vol)))


def surface_profiles(sim: Simulation, sol: Solution, depth_um=0.5):
    """1-D profiles vs x at a given depth: E_x, E_z (kV/cm), n, p, N_T+ (cm^-3), emission weight n_pulse*|E|."""
    g = sim.grid
    k = int(np.argmin(np.abs(g.zc[: g.ks] + depth_um)))
    Ex, Ez = sol.fields_physical(sim)
    n, p, NTp = sol.densities_physical(sim)
    npulse = (sol.G / sol.G.max()).reshape(g.Nzs, g.Nx)
    # depth-integrated emission weight (what the far field would see from each x)
    w_int = np.sum(npulse * np.hypot(Ex[: g.ks], Ez[: g.ks]) * g.dz[: g.ks][:, None], axis=0)
    wx_int = np.sum(npulse * Ex[: g.ks] * g.dz[: g.ks][:, None], axis=0)
    wz_int = np.sum(npulse * Ez[: g.ks] * g.dz[: g.ks][:, None], axis=0)
    return dict(x=g.xc, depth=-g.zc[k], Ex=Ex[k], Ez=Ez[k], n=n[k], p=p[k], NTp=NTp[k], f=sol.f.reshape(g.Nzs, g.Nx)[k],
                w=npulse[k] * np.hypot(Ex[k], Ez[k]), w_int=w_int, wx_int=wx_int, wz_int=wz_int)
