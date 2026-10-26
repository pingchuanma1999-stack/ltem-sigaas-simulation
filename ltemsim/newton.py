"""Fully coupled Newton solver for (psi, ln n, ln p, f) with backward-Euler time
stepping (pseudo-transient continuation to the steady state).

Why: the semi-insulating problem is stiff.  A 1 % change of the free-electron
density (~1e6 cm^-3) changes the EL2 space charge by ~1e14 cm^-3, enough to
redistribute kV/cm fields, so neither Gummel decoupling nor plain steady-state
Newton converge from a crude guess.  Treating the EL2 occupancy f = N_T0/N_T
as a dynamic variable and marching in time (dt growing geometrically from
~1e-13 s to seconds) follows the physical charging sequence instead.

Residuals (scaled units, see physics.py)
    F_psi[c] = lam2 * ( sum_f g_f (psi_nb - psi_c) + b_c ) + Vol_c * rho_c
    F_n[c]   = sum_f k_f [ n_nb B(d) - n_c B(-d) ]  - Vol_c (R_n - G) - Vol_c (n - n_old)/dt
    F_p[c]   = sum_f -k_f [ p_nb B(-d) - p_c B(d) ] - Vol_c (G - R_p) + Vol_c (p - p_old)/dt
    F_f[c]   = Vol_c [ (f - f_old)/dt - (R_n - R_p)/N_T ]
with   rho = p - n + N_T (1 - f) + N_D - N_A
       R_n = c_n n N_T (1-f) - e_n N_T f + B (n p - ni^2)      (net electron loss)
       R_p = c_p p N_T f - e_p N_T (1-f) + B (n p - ni^2)      (net hole loss)
dt = inf gives the steady-state equations (R_n = R_p, quasi-static traps).
"""
from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .fv import Topology, bernoulli
from .physics import Material
from .poisson import PoissonOperator


def dbernoulli(x: np.ndarray) -> np.ndarray:
    """B'(x) = [B(x) - B(x) B(-x)] / x, numerically safe."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-3
    out[small] = -0.5 + x[small] / 6.0 - x[small] ** 3 / 180.0
    m = ~small
    xm = x[m]
    out[m] = (bernoulli(xm) - bernoulli(xm) * bernoulli(-xm)) / xm
    return out


class State:
    __slots__ = ("psi", "n", "p", "f")

    def __init__(self, psi, n, p, f):
        self.psi, self.n, self.p, self.f = psi, n, p, f

    def copy(self):
        return State(self.psi.copy(), self.n.copy(), self.p.copy(), self.f.copy())


class CoupledNewton:
    def __init__(self, top: Topology, mat: Material, pois: PoissonOperator):
        self.top = top
        self.mat = mat
        self.pois = pois
        self.Na = top.N_all
        self.Ns = top.N_semi
        self.vol_s = top.vol_semi
        self.lam2 = mat.sc.lam2
        self.f = top.faces_semi
        self.bf = top.contact_semi
        self.fs = top.free_surface
        self._Lcoo = self.pois.L.tocoo()

    # ------------------------------------------------------------ fluxes
    def _mob(self, mu0, vsat, E):
        if not self.mat.p.field_dependent_mobility:
            return mu0 * np.ones_like(E), np.zeros_like(E)
        den = 1.0 + mu0 * np.abs(E) / vsat
        mu = mu0 / den
        dmu_dE = -mu0 * (mu0 / vsat) * np.sign(E) / den ** 2
        return mu, dmu_dE

    def _carrier_flux(self, psi_s, dens, mu0, vsat, Ve, dens_eq, sign):
        f, bf = self.f, self.bf
        d = psi_s[f.nb] - psi_s[f.c]
        dist = f.d
        mu, dmu = self._mob(mu0, vsat, d / dist)
        k = f.A * mu / dist
        dk_dd = f.A * dmu / dist / dist
        xc, xn = dens[f.c], dens[f.nb]
        Bp, Bm, dBp, dBm = bernoulli(d), bernoulli(-d), dbernoulli(d), dbernoulli(-d)
        if sign > 0:
            core = xn * Bp - xc * Bm
            J = k * core
            dJ_dd = k * (xn * dBp + xc * dBm) + dk_dd * core
            dJ_dxc, dJ_dxn = -k * Bm, k * Bp
        else:
            core = xn * Bm - xc * Bp
            J = -k * core
            dJ_dd = -(k * (-xn * dBm - xc * dBp) + dk_dd * core)
            dJ_dxc, dJ_dxn = k * Bp, -k * Bm
        db = Ve - psi_s[bf.cell]
        mub, dmub = self._mob(mu0, vsat, db / bf.dhalf)
        kb = bf.A * mub / bf.dhalf
        dkb_ddb = bf.A * dmub / bf.dhalf / bf.dhalf
        xb = dens[bf.cell]
        if sign > 0:
            coreb = dens_eq * bernoulli(db) - xb * bernoulli(-db)
            Jb = kb * coreb
            dJb_ddb = kb * (dens_eq * dbernoulli(db) + xb * dbernoulli(-db)) + dkb_ddb * coreb
            dJb_dxb = -kb * bernoulli(-db)
        else:
            coreb = dens_eq * bernoulli(-db) - xb * bernoulli(db)
            Jb = -kb * coreb
            dJb_ddb = -(kb * (-dens_eq * dbernoulli(-db) - xb * dbernoulli(db)) + dkb_ddb * coreb)
            dJb_dxb = kb * bernoulli(db)
        return dict(J=J, dJ_dd=dJ_dd, dJ_dxc=dJ_dxc, dJ_dxn=dJ_dxn, Jb=Jb, dJb_ddb=dJb_ddb, dJb_dxb=dJb_dxb)

    # ------------------------------------------------------------ recombination
    def _recomb(self, n, p, f, en, ep):
        mat = self.mat
        cn, cp, NT, B, ni = mat.cn, mat.cp, mat.NT, mat.B, mat.ni
        rad = B * (n * p - ni ** 2)
        Rn = cn * n * NT * (1 - f) - en * NT * f + rad
        Rp = cp * p * NT * f - ep * NT * (1 - f) + rad
        d = dict(
            Rn=Rn, Rp=Rp,
            dRn_dn=cn * NT * (1 - f) + B * p, dRn_dp=B * n, dRn_df=-NT * (cn * n + en),
            dRp_dn=B * p, dRp_dp=cp * NT * f + B * n, dRp_df=NT * (cp * p + ep),
        )
        return d

    # ------------------------------------------------------------ residual/Jacobian
    def residual_jacobian(self, st: State, G, en, ep, b, Ve, old: State | None, inv_dt: float, want_jac=True):
        mat, top = self.mat, self.top
        Na, Ns = self.Na, self.Ns
        f_, bf, fs = self.f, self.bf, self.fs
        vol = self.vol_s
        psi, n, p, f = st.psi, st.n, st.p, st.f
        psi_s = psi[:Na][:Ns]
        NT = mat.NT
        rho = np.zeros(Na)
        rho[:Ns] = p - n + NT * (1 - f) + mat.ND - mat.NA
        F_psi = self.lam2 * (self.pois.L @ psi + b) + top.vol_all * rho
        rc = self._recomb(n, p, f, en, ep)
        fn = self._carrier_flux(psi_s, n, mat.mu_n, mat.vsat_n, Ve, mat.n_eq, +1)
        fp = self._carrier_flux(psi_s, p, mat.mu_p, mat.vsat_p, Ve, mat.p_eq, -1)
        F_n = np.zeros(Ns); F_p = np.zeros(Ns)
        np.add.at(F_n, f_.c, fn["J"]); np.add.at(F_n, f_.nb, -fn["J"])
        np.add.at(F_p, f_.c, fp["J"]); np.add.at(F_p, f_.nb, -fp["J"])
        np.add.at(F_n, bf.cell, fn["Jb"]); np.add.at(F_p, bf.cell, fp["Jb"])
        if mat.S > 0:
            np.add.at(F_n, fs.cell, -fs.A * mat.S * n[fs.cell])
            np.add.at(F_p, fs.cell, fs.A * mat.S * p[fs.cell])
        F_n -= vol * (rc["Rn"] - G)
        F_p -= vol * (G - rc["Rp"])
        F_f = -vol * (rc["Rn"] - rc["Rp"]) / NT
        if inv_dt > 0 and old is not None:
            F_n -= vol * (n - old.n) * inv_dt
            F_p += vol * (p - old.p) * inv_dt
            F_f += vol * (f - old.f) * inv_dt
        F = np.concatenate([F_psi, F_n, F_p, F_f])
        if not want_jac:
            return F, None
        rows, cols, vals = [], [], []
        off_n, off_p, off_f = Na, Na + Ns, Na + 2 * Ns
        idx = np.arange(Ns)

        def add(r, c, v):
            rows.append(np.asarray(r)); cols.append(np.asarray(c)); vals.append(np.asarray(v, dtype=float))

        # --- Poisson rows
        add(self._Lcoo.row, self._Lcoo.col, self.lam2 * self._Lcoo.data)
        add(idx, off_n + idx, -vol * n)
        add(idx, off_p + idx, vol * p)
        add(idx, off_f + idx, -vol * NT)
        # --- carrier rows (fluxes)
        for fl, off_x, dens, rowoff, sgn in ((fn, off_n, n, off_n, +1), (fp, off_p, p, off_p, -1)):
            rowc = rowoff + f_.c
            rownb = rowoff + f_.nb
            add(rowc, f_.nb, fl["dJ_dd"]); add(rowc, f_.c, -fl["dJ_dd"])
            add(rownb, f_.nb, -fl["dJ_dd"]); add(rownb, f_.c, fl["dJ_dd"])
            add(rowc, off_x + f_.c, fl["dJ_dxc"] * dens[f_.c]); add(rowc, off_x + f_.nb, fl["dJ_dxn"] * dens[f_.nb])
            add(rownb, off_x + f_.c, -fl["dJ_dxc"] * dens[f_.c]); add(rownb, off_x + f_.nb, -fl["dJ_dxn"] * dens[f_.nb])
            rowb = rowoff + bf.cell
            add(rowb, bf.cell, -fl["dJb_ddb"])
            add(rowb, off_x + bf.cell, fl["dJb_dxb"] * dens[bf.cell])
            if mat.S > 0:
                add(rowoff + fs.cell, off_x + fs.cell, (-sgn) * fs.A * mat.S * dens[fs.cell])
        # --- recombination and time terms
        rn, rp, rf = off_n + idx, off_p + idx, off_f + idx
        add(rn, off_n + idx, -vol * (rc["dRn_dn"] * n + inv_dt * n))
        add(rn, off_p + idx, -vol * rc["dRn_dp"] * p)
        add(rn, off_f + idx, -vol * rc["dRn_df"])
        add(rp, off_n + idx, vol * rc["dRp_dn"] * n)
        add(rp, off_p + idx, vol * (rc["dRp_dp"] * p + inv_dt * p))
        add(rp, off_f + idx, vol * rc["dRp_df"])
        add(rf, off_n + idx, -vol * (rc["dRn_dn"] - rc["dRp_dn"]) * n / NT)
        add(rf, off_p + idx, -vol * (rc["dRn_dp"] - rc["dRp_dp"]) * p / NT)
        add(rf, off_f + idx, vol * (inv_dt - (rc["dRn_df"] - rc["dRp_df"]) / NT))
        N = Na + 3 * Ns
        J = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsc()
        return F, J

    # ------------------------------------------------------------ Newton for one (pseudo-)time step
    def newton_step(self, st: State, G, en, ep, b, Ve, old, inv_dt, max_iter=25, tol_psi=1e-5, tol_log=1e-5,
                    dpsi_max=20.0, dlog_max=3.0, df_max=0.3, verbose=False, tag=""):
        mat = self.mat
        nfloor = mat.p.n_floor / mat.sc.C0
        Na, Ns = self.Na, self.Ns
        F, J = self.residual_jacobian(st, G, en, ep, b, Ve, old, inv_dt)
        normF = np.linalg.norm(F)
        hist = []
        converged = False
        for it in range(max_iter):
            delta = spla.spsolve(J, -F)
            dpsi, du, dv, df = delta[:Na], delta[Na:Na + Ns], delta[Na + Ns:Na + 2 * Ns], delta[Na + 2 * Ns:]
            s = 1.0
            for arr, lim in ((dpsi, dpsi_max), (du, dlog_max), (dv, dlog_max), (df, df_max)):
                m = np.max(np.abs(arr))
                if m > lim:
                    s = min(s, lim / m)
            accepted = False
            for _ in range(10):
                trial = State(st.psi + s * dpsi, np.maximum(st.n * np.exp(s * du), nfloor),
                              np.maximum(st.p * np.exp(s * dv), nfloor), np.clip(st.f + s * df, 0.0, 1.0))
                F_t, _ = self.residual_jacobian(trial, G, en, ep, b, Ve, old, inv_dt, want_jac=False)
                normF_t = np.linalg.norm(F_t)
                if normF_t < (1.0 - 1e-4 * s) * normF or normF_t < 1e-300:
                    accepted = True
                    break
                s *= 0.5
            st = trial
            F, J = self.residual_jacobian(st, G, en, ep, b, Ve, old, inv_dt)
            normF = np.linalg.norm(F)
            step_psi, step_log, step_f = s * np.max(np.abs(dpsi)), s * max(np.max(np.abs(du)), np.max(np.abs(dv))), s * np.max(np.abs(df))
            hist.append((it, float(step_psi), float(step_log), float(step_f), float(normF), float(s), bool(accepted)))
            if verbose:
                print(f"    [{tag}] newton {it:2d}: dpsi {step_psi:.1e} VT, dlog {step_log:.1e}, df {step_f:.1e}, "
                      f"s={s:.3f}{'' if accepted else ' (no decrease)'}, |F|={normF:.2e}")
            if step_psi < tol_psi and step_log < tol_log and step_f < tol_psi:
                converged = True
                break
        return st, converged, hist

    # ------------------------------------------------------------ pseudo-transient driver
    def march(self, st: State, G, phi, b, Ve, dt0=1e-13, dt_max=10.0, t_end=30.0, grow=2.5,
              max_steps=600, step_tol=1e-4, polish_every=12, polish_after=1e-10, verbose=False, tag="", log=None):
        """Backward-Euler time stepping from state `st` towards the steady state.

        dt0, dt_max, t_end, polish_after in seconds.  Every `polish_every` steps
        (once t > polish_after) a pure steady-state Newton is attempted from the
        current state; the march ends when that converges, or at t_end.
        """
        mat = self.mat
        t0 = mat.sc.t0
        en, ep = mat.emission_rates(phi)
        dt = dt0 / t0
        t = 0.0
        step = 0
        t_wall = time.time()
        info = []
        ok_ss = False
        st_ss = None
        while step < max_steps and t < t_end / t0:
            old = st.copy()
            inv_dt = 1.0 / dt
            new, ok, hist = self.newton_step(st, G, en, ep, b, Ve, old, inv_dt, tol_psi=step_tol, tol_log=step_tol,
                                             verbose=False, tag=tag)
            nit = len(hist)
            if ok:
                st = new
                t += dt
                step += 1
                Fss, _ = self.residual_jacobian(st, G, en, ep, b, Ve, None, 0.0, want_jac=False)
                w = self.weighted_norm(Fss)
                info.append((step, t * t0, dt * t0, nit, w))
                if verbose and (step % 10 == 0 or step < 3):
                    print(f"  [{tag}] step {step:3d} t={t*t0:.2e}s dt={dt*t0:.2e}s newton {nit:2d} "
                          f"ss-res psi {w[0]:.1e} n {w[1]:.1e} p {w[2]:.1e} f {w[3]:.1e} ({time.time()-t_wall:.0f}s)")
                if nit <= 8:
                    dt = min(dt * grow, dt_max / t0)
                elif nit > 14:
                    dt = max(dt * 0.5, dt0 / t0)
                if step % polish_every == 0 and t * t0 > polish_after:
                    st_try, ok_ss, hist_ss = self.newton_step(st, G, en, ep, b, Ve, None, 0.0, max_iter=25,
                                                              verbose=False, tag=tag + " ss")
                    if ok_ss:
                        st_ss = st_try
                        if verbose:
                            print(f"  [{tag}] steady-state Newton converged after step {step} ({len(hist_ss)} it)")
                        break
                    elif verbose:
                        print(f"  [{tag}] steady-state Newton not yet converging (t={t*t0:.1e}s); continue marching")
            else:
                dt *= 0.25
                if verbose:
                    print(f"  [{tag}] step rejected (newton {nit} it), dt -> {dt*t0:.2e}s")
                if dt * t0 < 1e-18:
                    raise RuntimeError("time step collapsed")
        if st_ss is None:
            st_ss, ok_ss, hist_ss = self.newton_step(st, G, en, ep, b, Ve, None, 0.0, max_iter=40, verbose=False, tag=tag + " ss")
        st = st_ss
        Fss, _ = self.residual_jacobian(st, G, en, ep, b, Ve, None, 0.0, want_jac=False)
        w = self.weighted_norm(Fss)
        if verbose:
            print(f"  [{tag}] final: steady-state converged={ok_ss}, ss-res psi {w[0]:.1e} n {w[1]:.1e} "
                  f"p {w[2]:.1e} f {w[3]:.1e}, t_sim={t*t0:.2e}s, {step} steps, wall {time.time()-t_wall:.0f}s")
        return st, dict(steps=step, t_sim=t * t0, converged=bool(ok_ss), ss_res=w, info=info)

    def weighted_norm(self, F):
        Na, Ns = self.Na, self.Ns
        blocks = [F[:Na] / self.lam2, F[Na:Na + Ns], F[Na + Ns:Na + 2 * Ns], F[Na + 2 * Ns:]]
        return tuple(float(np.max(np.abs(x))) for x in blocks)
