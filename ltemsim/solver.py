"""Steady-state drift-diffusion + EL2 solver with continuation.

Workflow
    1. equilibrium (V = 0, no light): psi = psi_eq, n = n_eq, p = p_eq
    2. ramp the bias to V_bias in the dark (linear Poisson guess, then coupled Newton)
    3. ramp the generation rate from 1e-6 * G to G (coupled Newton at each step)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
import numpy as np

from .params import Params
from .grid import Grid
from .fv import Topology
from .physics import Material
from .poisson import PoissonOperator
from .continuity import Continuity
from .newton import CoupledNewton, State


@dataclass
class Solution:
    params: Params
    grid: Grid
    psi: np.ndarray          # scaled potential on all cells (Nz*Nx)
    n: np.ndarray            # scaled electron density on semiconductor cells
    p: np.ndarray
    NTp: np.ndarray          # scaled ionised EL2 density on semiconductor cells
    G: np.ndarray            # scaled generation on semiconductor cells
    gen_factor: float
    V_left: float            # V (physical) on the -x electrode
    V_right: float
    I_left_mA: float         # current into the -x electrode (mA, after multiplying by the spot y-length)
    I_right_mA: float
    log: list = field(default_factory=list)
    f: np.ndarray = None    # EL2 neutral fraction on semiconductor cells

    # ---- convenient physical fields ----------------------------------------
    def fields_physical(self, sim: "Simulation"):
        """E_x, E_z in kV/cm on all cells (Nz, Nx)."""
        Ex, Ez = sim.pois.field(self.psi)
        E0 = sim.mat.sc.E0  # V/cm per (VT/um)
        return Ex * E0 / 1e3, Ez * E0 / 1e3

    def densities_physical(self, sim: "Simulation"):
        C0 = sim.mat.sc.C0
        g = self.grid
        shape = (g.Nzs, g.Nx)
        return (self.n * C0).reshape(shape), (self.p * C0).reshape(shape), (self.NTp * C0).reshape(shape)

    def potential_V(self, sim: "Simulation"):
        return (self.psi - sim.mat.psi_eq).reshape(self.grid.Nz, self.grid.Nx) * sim.mat.sc.VT

    def save(self, path):
        np.savez_compressed(path, psi=self.psi, n=self.n, p=self.p, NTp=self.NTp, G=self.G,
                            gen_factor=self.gen_factor, V_left=self.V_left, V_right=self.V_right,
                            I_left_mA=self.I_left_mA, I_right_mA=self.I_right_mA, f=self.f,
                            params_json=self.params.to_json(), xe=self.grid.xe, ze=self.grid.ze, ks=self.grid.ks,
                            log=json.dumps(self.log))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        params = Params.from_json(str(d["params_json"]))
        grid = Grid(xe=d["xe"], ze=d["ze"], ks=int(d["ks"]))
        return cls(params=params, grid=grid, psi=d["psi"], n=d["n"], p=d["p"], NTp=d["NTp"], G=d["G"],
                   gen_factor=float(d["gen_factor"]), V_left=float(d["V_left"]), V_right=float(d["V_right"]),
                   I_left_mA=float(d["I_left_mA"]), I_right_mA=float(d["I_right_mA"]), log=json.loads(str(d["log"])), f=d["f"])


class Simulation:
    def __init__(self, params: Params, verbose: bool = True):
        self.p = params
        self.verbose = verbose
        self.grid = Grid.build(params)
        self.top = Topology(self.grid, params.gap)
        self.mat = Material(params)
        self.top.set_eps(params.eps_r, 1.0)
        self.pois = PoissonOperator(self.top, self.mat.sc.lam2)
        self.cont = Continuity(self.top, self.mat)
        self.newton = CoupledNewton(self.top, self.mat, self.pois)
        self.Ns = self.top.N_semi
        if verbose:
            print(self.grid.describe())
            print(self.mat.describe())

    # ------------------------------------------------------------ helpers
    def electrode_voltages(self, V_total: float):
        """Physical voltages on the -x and +x electrodes for a given total bias (symmetric split)."""
        if self.p.anode_side == "-x":
            return +V_total / 2.0, -V_total / 2.0
        elif self.p.anode_side == "+x":
            return -V_total / 2.0, +V_total / 2.0
        raise ValueError("anode_side must be '+x' or '-x'")

    def scaled_contacts(self, V_left, V_right):
        VT = self.mat.sc.VT
        off = self.mat.psi_eq
        Vl, Vr = off + V_left / VT, off + V_right / VT
        Ve = self.top.electrode_values(Vl, Vr)      # per contact column
        b = self.pois.dirichlet_rhs(Vl, Vr)
        return Vl, Vr, Ve, b

    def generation_arrays(self, factor=1.0):
        g = self.grid
        G, phi = self.mat.generation(g.xc, g.zc[: g.ks], factor)
        return G.ravel(), phi.ravel()

    def charge_density(self, psi, n, p, phi):
        mat, Ns = self.mat, self.Ns
        en, ep = mat.emission_rates(phi)
        NTp = mat.trap_terms(n, p, en, ep)[0]
        rho = np.zeros(self.top.N_all)
        rho[:Ns] = p - n + NTp + mat.ND - mat.NA
        return rho, NTp

    # ------------------------------------------------------------ driver
    def equilibrium_state(self) -> State:
        mat, Ns = self.mat, self.Ns
        f_eq = (mat.cn * mat.n_eq + mat.ep_th) / (mat.cn * mat.n_eq + mat.en_th + mat.cp * mat.p_eq + mat.ep_th)
        return State(np.full(self.top.N_all, mat.psi_eq), np.full(Ns, mat.n_eq), np.full(Ns, mat.p_eq), np.full(Ns, f_eq))

    def run(self, V_total=None, gen_factor=1.0, warm=None, dark_t_end=30.0, light_t_end=10.0,
            dt0=1e-13, march_kwargs=None) -> Solution:
        """Dark bias turn-on followed by light turn-on, each marched to the steady state."""
        p_ = self.p
        mat = self.mat
        V_total = p_.V_bias if V_total is None else V_total
        V_left, V_right = self.electrode_voltages(V_total)
        g = self.grid
        Ns = self.Ns
        mk = dict(verbose=self.verbose)
        if march_kwargs:
            mk.update(march_kwargs)
        log = []
        t_start = time.time()
        Vl, Vr, Ve, b = self.scaled_contacts(V_left, V_right)
        if warm is None:
            st = self.equilibrium_state()
            G0, phi0 = np.zeros(Ns), np.zeros(Ns)
            # dark: linear Poisson guess for the potential, then march (trap charge screens the surface field)
            rho = np.zeros(self.top.N_all)
            rho[:Ns] = st.p - st.n + mat.NT * (1 - st.f) + mat.ND - mat.NA
            st.psi = self.pois.solve_linear(b, rho)
            st, info = self.newton.march(st, G0, phi0, b, Ve, dt0=dt0, t_end=dark_t_end, tag="dark", **mk)
            log.append({"stage": f"dark V={V_total:.2f}", "t_wall": time.time() - t_start, **{k: v for k, v in info.items() if k != "info"}})
        else:
            st = State(warm.psi.copy(), warm.n.copy(), warm.p.copy(), warm.f.copy())
        # light on
        G, phi = self.generation_arrays(gen_factor)
        st, info = self.newton.march(st, G, phi, b, Ve, dt0=dt0, t_end=light_t_end, tag=f"light x{gen_factor:.2g}", **mk)
        log.append({"stage": f"light x{gen_factor:.3g}", "t_wall": time.time() - t_start, **{k: v for k, v in info.items() if k != "info"}})
        NTp = mat.NT * (1 - st.f)
        I_l, I_r = self.cont.contact_currents(st.psi[:Ns], st.n, st.p, Ve)
        conv = mat.sc.J0 * mat.sc.L0 * p_.spot_length_y_cm * 1e3   # scaled current/length -> mA for the spot
        return Solution(params=p_, grid=g, psi=st.psi, n=st.n, p=st.p, NTp=NTp, G=G, gen_factor=gen_factor,
                        V_left=V_left, V_right=V_right, I_left_mA=I_l * conv, I_right_mA=I_r * conv, log=log,
                        f=st.f)
