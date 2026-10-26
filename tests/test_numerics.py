"""Numerical checks: Poisson against the analytic coplanar-strip field, equilibrium
consistency, and a finite-difference check of the coupled Jacobian.

Run with:  python -m pytest tests/   (or simply python tests/test_numerics.py)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ltemsim import Params, Simulation
from ltemsim.newton import State


def small_params(**kw):
    base = dict(x_half=60, x_fine=6, dx_fine=0.5, dx_coarse=2.0, gap=40, depth=30, air_height=30,
                dz_fine=0.25, z_fine=1.0, dz_coarse_semi=3.0, dz_coarse_air=3.0)
    base.update(kw)
    return Params(**base)


def test_poisson_bare_field_vs_analytic():
    """No charge: E_x at the gap centre of thin coplanar strips, half-space dielectric.

    Analytic (2-D, strips at |x| >= a, potentials ±V/2, dielectric below, air above):
    E_x(x, 0) = V / (pi sqrt(a^2 - x^2)) — the same in both media for thin strips.
    """
    p = small_params()
    sim = Simulation(p, verbose=False)
    Vl, Vr, Ve, b = sim.scaled_contacts(-12.5, 12.5)
    psi = sim.pois.solve_linear(b, np.zeros(sim.top.N_all))
    Ex, Ez = sim.pois.field(psi)
    E0 = sim.mat.sc.E0
    g = sim.grid
    a = p.gap / 2
    for x in (0.0, 5.0, 10.0):   # semi-infinite strips need a box well beyond the gap (x_half = 3a, depth 1.5a)
        i = np.argmin(np.abs(g.xc - x))
        num = Ex[g.ks - 1, i] * E0            # V/cm, first GaAs row (0.125 um deep)
        ana = 25.0 / (np.pi * np.sqrt((a * 1e-4) ** 2 - (x * 1e-4) ** 2))
        assert abs(abs(num) / ana - 1) < 0.12, (x, num, ana)
        # the field is nearly horizontal at the surface in the gap centre
        assert abs(Ez[g.ks - 1, i] / Ex[g.ks - 1, i]) < 0.05


def test_equilibrium_is_steady_state():
    """At V = 0 and G = 0 the equilibrium state has zero residual."""
    p = small_params()
    sim = Simulation(p, verbose=False)
    st = sim.equilibrium_state()
    Vl, Vr, Ve, b = sim.scaled_contacts(0.0, 0.0)
    G, phi = np.zeros(sim.Ns), np.zeros(sim.Ns)
    en, ep = sim.mat.emission_rates(phi)
    F, _ = sim.newton.residual_jacobian(st, G, en, ep, b, Ve, None, 0.0, want_jac=False)
    w = sim.newton.weighted_norm(F)
    assert max(w) < 1e-9, w


def test_jacobian_finite_difference():
    p = small_params()
    sim = Simulation(p, verbose=False)
    nw, mat, Ns, Na = sim.newton, sim.mat, sim.Ns, sim.top.N_all
    Vl, Vr, Ve, b = sim.scaled_contacts(6, -6)
    G, phi = sim.generation_arrays(1e-3)
    en, ep = mat.emission_rates(phi)
    rng = np.random.default_rng(1)
    psi = sim.pois.solve_linear(b, np.zeros(Na)) + 0.3 * rng.standard_normal(Na)
    n = mat.n_eq * np.exp(2 * rng.standard_normal(Ns)) + 1e-3 * G
    pp = mat.p_eq * np.exp(2 * rng.standard_normal(Ns)) + 1e-3 * G
    f = rng.uniform(0.1, 0.9, Ns)
    st = State(psi, n, pp, f)
    old = State(psi * 1.01, n * 1.3, pp * 0.7, f * 0.9)
    for inv_dt in (0.0, 3.0):
        F, J = nw.residual_jacobian(st, G, en, ep, b, Ve, old, inv_dt)
        eps = 1e-6
        d = rng.standard_normal(Na + 3 * Ns)
        d /= np.abs(d).max()
        st2 = State(psi + eps * d[:Na], n * np.exp(eps * d[Na:Na + Ns]), pp * np.exp(eps * d[Na + Ns:Na + 2 * Ns]),
                    f + eps * d[Na + 2 * Ns:])
        F2, _ = nw.residual_jacobian(st2, G, en, ep, b, Ve, old, inv_dt, want_jac=False)
        fd = (F2 - F) / eps
        jd = J @ d
        for sl in (slice(0, Na), slice(Na, Na + Ns), slice(Na + Ns, Na + 2 * Ns), slice(Na + 2 * Ns, None)):
            rel = np.abs(fd[sl] - jd[sl]).max() / max(np.abs(fd[sl]).max(), 1e-300)
            assert rel < 1e-5, rel


def test_dark_current_is_ohmic():
    """Dark gap at 10 V: tiny current, nearly uniform EL2+ far from the electrodes."""
    p = small_params()
    sim = Simulation(p, verbose=False)
    sol = sim.run(V_total=10.0, gen_factor=0.0, dark_t_end=5.0, light_t_end=1e-3,
                  march_kwargs=dict(verbose=False, max_steps=200))
    assert sol.log[0]["converged"]
    assert abs(sol.I_left_mA) < 1e-3          # dark current below 1 uA
    n, pp, NTp = sol.densities_physical(sim)
    assert 0.5e15 < NTp.mean() < 3e15


if __name__ == "__main__":
    test_poisson_bare_field_vs_analytic(); print("poisson ok")
    test_equilibrium_is_steady_state(); print("equilibrium ok")
    test_jacobian_finite_difference(); print("jacobian ok")
    test_dark_current_is_ohmic(); print("dark ok")
