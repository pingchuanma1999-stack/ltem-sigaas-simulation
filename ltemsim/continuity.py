"""Scharfetter–Gummel discretisation of the steady-state carrier continuity equations.

Scaled units; see physics.py.  Electron current J~_n = mu (grad n - n grad psi),
hole current J~_p = -mu (grad p + p grad psi).
div J~_n = R - G,   div J~_p = G - R.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .fv import Topology, assemble, bernoulli
from .physics import Material


class Continuity:
    def __init__(self, top: Topology, mat: Material):
        self.top = top
        self.mat = mat
        self.vol = top.vol_semi
        self.N = top.N_semi

    def _face_data(self, psi, mu0, vsat):
        f = self.top.faces_semi
        dpsi = psi[f.nb] - psi[f.c]                 # scaled potential difference nb - c
        d = f.d
        E = dpsi / d                                # scaled field along the face normal (VT/um -> /E0 handled by vsat scaling)
        mu = self.mat.mobility(mu0, vsat, E)
        k = f.A * mu / d
        return f, dpsi, k

    def _contact_data(self, psi, mu0, vsat, Ve):
        bf = self.top.contact_semi
        dpsi = Ve - psi[bf.cell]
        E = dpsi / bf.dhalf
        mu = self.mat.mobility(mu0, vsat, E)
        k = bf.A * mu / bf.dhalf
        return bf, dpsi, k

    # ------------------------------------------------------------ electrons
    def solve_electrons(self, psi_semi, n0, p0, G, en, ep, Ve):
        mat = self.mat
        f, dpsi, k = self._face_data(psi_semi, mat.mu_n, mat.vsat_n)
        Bp = bernoulli(dpsi)
        Bm = bernoulli(-dpsi)
        rows = [f.c, f.c, f.nb, f.nb]
        cols = [f.nb, f.c, f.nb, f.c]
        vals = [k * Bp, -k * Bm, -k * Bp, k * Bm]
        # recombination linearisation
        NTp, R0, dRdn, dRdp, _, _ = mat.trap_terms(n0, p0, en, ep)
        diag = -self.vol * dRdn
        rhs = self.vol * (R0 - dRdn * n0 - G)
        # contacts (Dirichlet n = n_eq at the electrode face)
        bf, dpb, kb = self._contact_data(psi_semi, mat.mu_n, mat.vsat_n, Ve)
        diag_c = np.zeros(self.N)
        np.add.at(diag_c, bf.cell, -kb * bernoulli(-dpb))
        np.add.at(rhs, bf.cell, -kb * mat.n_contact * bernoulli(dpb))
        # surface recombination on the free surface
        if mat.S > 0:
            fs = self.top.free_surface
            np.add.at(diag_c, fs.cell, -fs.A * mat.S)
        rows.append(np.arange(self.N)); cols.append(np.arange(self.N)); vals.append(diag + diag_c)
        M = assemble(self.N, rows, cols, vals)
        n = spla.spsolve(M.tocsc(), rhs)
        return n

    # ---------------------------------------------------------------- holes
    def solve_holes(self, psi_semi, n0, p0, G, en, ep, Ve):
        mat = self.mat
        f, dpsi, k = self._face_data(psi_semi, mat.mu_p, mat.vsat_p)
        Bp = bernoulli(dpsi)
        Bm = bernoulli(-dpsi)
        # flux (c->nb) = -k [ p_nb B(-dpsi) - p_c B(dpsi) ]
        rows = [f.c, f.c, f.nb, f.nb]
        cols = [f.nb, f.c, f.nb, f.c]
        vals = [-k * Bm, k * Bp, k * Bm, -k * Bp]
        NTp, R0, dRdn, dRdp, _, _ = mat.trap_terms(n0, p0, en, ep)
        diag = self.vol * dRdp
        rhs = self.vol * (G - R0 + dRdp * p0)
        bf, dpb, kb = self._contact_data(psi_semi, mat.mu_p, mat.vsat_p, Ve)
        diag_c = np.zeros(self.N)
        np.add.at(diag_c, bf.cell, kb * bernoulli(dpb))
        np.add.at(rhs, bf.cell, kb * mat.p_contact * bernoulli(-dpb))
        if mat.S > 0:
            fs = self.top.free_surface
            np.add.at(diag_c, fs.cell, fs.A * mat.S)
        rows.append(np.arange(self.N)); cols.append(np.arange(self.N)); vals.append(diag + diag_c)
        M = assemble(self.N, rows, cols, vals)
        p = spla.spsolve(M.tocsc(), rhs)
        return p

    # ------------------------------------------------------------- currents
    def face_currents(self, psi_semi, n, p):
        """Scaled electron and hole currents through every interior semiconductor face (c -> nb)."""
        mat = self.mat
        f, dpsi, kn = self._face_data(psi_semi, mat.mu_n, mat.vsat_n)
        _, _, kp = self._face_data(psi_semi, mat.mu_p, mat.vsat_p)
        Bp = bernoulli(dpsi)
        Bm = bernoulli(-dpsi)
        Jn = kn / f.A * (n[f.nb] * Bp - n[f.c] * Bm)            # current density (scaled), c->nb
        Jp = -kp / f.A * (p[f.nb] * Bm - p[f.c] * Bp)
        return f, Jn, Jp

    def contact_currents(self, psi_semi, n, p, Ve):
        """Total scaled current (per unit y length, units J0*L0) entering each electrode column."""
        mat = self.mat
        bf, dpb, kb_n = self._contact_data(psi_semi, mat.mu_n, mat.vsat_n, Ve)
        _, _, kb_p = self._contact_data(psi_semi, mat.mu_p, mat.vsat_p, Ve)
        Jn = kb_n * (mat.n_contact * bernoulli(dpb) - n[bf.cell] * bernoulli(-dpb))   # already times A
        Jp = -kb_p * (mat.p_contact * bernoulli(-dpb) - p[bf.cell] * bernoulli(dpb))
        left = bf.x < 0
        I_left = np.sum((Jn + Jp)[left])
        I_right = np.sum((Jn + Jp)[~left])
        return I_left, I_right
