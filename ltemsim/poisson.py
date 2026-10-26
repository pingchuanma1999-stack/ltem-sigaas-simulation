"""Nonlinear Poisson solver (scaled units) on the full GaAs + air domain.

lambda^2 [ L psi + b(V) ] + Vol * rho(psi) = 0
with L the FV Laplacian with per-cell permittivity, b the Dirichlet
(electrode) contribution and rho the scaled charge density
(p - n + N_T+ + N_D - N_A on semiconductor cells, 0 in air).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .fv import Topology, assemble


class PoissonOperator:
    def __init__(self, top: Topology, lam2: float):
        self.top = top
        self.lam2 = lam2
        f = top.faces_all
        eps = top.eps
        g = f.A / (f.d1 / eps[f.c] + f.d2 / eps[f.nb])      # face conductance
        rows = [f.c, f.nb, f.c, f.nb]
        cols = [f.c, f.nb, f.nb, f.c]
        vals = [-g, -g, g, g]
        # Dirichlet electrode faces (+ optional back-surface Dirichlet faces)
        bfaces = [top.contact_semi, top.contact_air] + ([top.bottom] if top.back_dirichlet else [])
        for bf in bfaces:
            gb = bf.A * eps[bf.cell] / bf.dhalf
            rows.append(bf.cell); cols.append(bf.cell); vals.append(-gb)
        self.L = assemble(top.N_all, rows, cols, vals)
        self._gb_semi = top.contact_semi.A * eps[top.contact_semi.cell] / top.contact_semi.dhalf
        self._gb_air = top.contact_air.A * eps[top.contact_air.cell] / top.contact_air.dhalf
        self._gb_bottom = top.bottom.A * eps[top.bottom.cell] / top.bottom.dhalf if top.back_dirichlet else None
        self.V_back_scaled = 0.0
        self.vol = top.vol_all
        self._lu = None

    def dirichlet_rhs(self, V_left: float, V_right: float) -> np.ndarray:
        """Vector b: contribution of the fixed electrode potentials to L psi + b."""
        top = self.top
        b = np.zeros(top.N_all)
        Ve = top.electrode_values(V_left, V_right)
        np.add.at(b, top.contact_semi.cell, self._gb_semi * Ve)
        np.add.at(b, top.contact_air.cell, self._gb_air * Ve)
        if self._gb_bottom is not None:
            np.add.at(b, top.bottom.cell, self._gb_bottom * self.V_back_scaled)
        return b

    def residual(self, psi, b, rho):
        return self.lam2 * (self.L @ psi + b) + self.vol * rho

    def solve_linear(self, b, rho):
        """Linear Poisson: lambda^2 (L psi + b) + Vol rho = 0."""
        if self._lu is None:
            self._lu = spla.splu((self.lam2 * self.L).tocsc())
        return self._lu.solve(-(self.lam2 * b + self.vol * rho))

    def newton(self, psi0, b, charge_fn, max_iter=60, tol=1e-8, dpsi_max=10.0):
        """Damped Newton for the nonlinear Poisson equation.

        charge_fn(psi) -> (rho, drho_dpsi) on all cells (zeros in air).
        Returns (psi, n_iter, final residual norm).
        """
        psi = psi0.copy()
        res_norm = np.inf
        for it in range(max_iter):
            rho, drho = charge_fn(psi)
            F = self.residual(psi, b, rho)
            J = self.lam2 * self.L + sp.diags(self.vol * drho)
            dpsi = spla.spsolve(J.tocsc(), -F)
            m = np.max(np.abs(dpsi))
            if m > dpsi_max:
                dpsi *= dpsi_max / m
            psi += dpsi
            res_norm = m
            if m < tol:
                break
        return psi, it + 1, res_norm

    def field(self, psi):
        """E_x, E_z at cell centres (scaled: -grad psi~ in 1/um -> multiply by E0*L0... caller handles),
        computed from one-sided differences at the boundaries and central differences inside.
        Returns arrays shaped (Nz, Nx), in units of (kT/q)/um.
        """
        g = self.top.g
        P = psi.reshape(g.Nz, g.Nx)
        xc, zc = g.xc, g.zc
        Ex = np.zeros_like(P)
        Ez = np.zeros_like(P)
        Ex[:, 1:-1] = -(P[:, 2:] - P[:, :-2]) / (xc[2:] - xc[:-2])[None, :]
        Ex[:, 0] = -(P[:, 1] - P[:, 0]) / (xc[1] - xc[0])
        Ex[:, -1] = -(P[:, -1] - P[:, -2]) / (xc[-1] - xc[-2])
        Ez[1:-1, :] = -(P[2:, :] - P[:-2, :]) / (zc[2:] - zc[:-2])[:, None]
        Ez[0, :] = -(P[1, :] - P[0, :]) / (zc[1] - zc[0])
        Ez[-1, :] = -(P[-1, :] - P[-2, :]) / (zc[-1] - zc[-2])
        # do not difference across the GaAs/air interface (E_z jumps there): one-sided on both sides
        ks = g.ks
        Ez[ks - 1, :] = -(P[ks - 1, :] - P[ks - 2, :]) / (zc[ks - 1] - zc[ks - 2])
        Ez[ks, :] = -(P[ks + 1, :] - P[ks, :]) / (zc[ks + 1] - zc[ks])
        return Ex, Ez
