"""Material physics: scaling, EL2 trap kinetics, recombination, mobility, generation.

Scaled units (De Mari):
    potential   psi~ = psi / V_T
    length      x~   = x / L0,            L0 = 1 um
    density     n~   = n / C0,            C0 = 1e16 cm^-3
    mobility    mu~  = mu / mu0,          mu0 = 1 cm^2/Vs
    time        t~   = t / t0,            t0 = L0^2 / (mu0 V_T)
    rate        R~   = R * t0 / C0
    current     J~   = J / J0,            J0 = q mu0 V_T C0 / L0
Poisson:  lambda^2 div(eps_r grad psi~) = -(p~ - n~ + N_T+~ + N_D~ - N_A~),
          lambda^2 = eps0 V_T / (q C0 L0^2).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .params import Params, Q, EPS0


@dataclass
class Scale:
    VT: float
    L0: float = 1e-4          # cm
    C0: float = 1e16          # cm^-3
    mu0: float = 1.0          # cm^2/Vs

    @property
    def t0(self) -> float:
        return self.L0 ** 2 / (self.mu0 * self.VT)

    @property
    def R0(self) -> float:
        return self.C0 / self.t0

    @property
    def J0(self) -> float:    # A/cm^2
        return Q * self.mu0 * self.VT * self.C0 / self.L0

    @property
    def E0(self) -> float:    # V/cm
        return self.VT / self.L0

    @property
    def lam2(self) -> float:
        return EPS0 * self.VT / (Q * self.C0 * self.L0 ** 2)


class Material:
    """All scaled coefficients needed by the solver."""

    def __init__(self, p: Params):
        self.p = p
        self.sc = Scale(VT=p.VT)
        sc = self.sc
        # scaled densities
        self.ni = p.ni / sc.C0
        self.NT = p.N_T / sc.C0
        self.NA = p.N_A / sc.C0
        self.ND = p.N_D / sc.C0
        self.n1 = p.n1 / sc.C0
        self.p1 = p.p1 / sc.C0
        # scaled capture coefficients: c [cm^3/s] * C0 * t0 -> 1/(scaled density * scaled time)
        self.cn = p.c_n * sc.C0 * sc.t0
        self.cp = p.c_p * sc.C0 * sc.t0
        self.B = p.B_rad * sc.C0 * sc.t0
        self.en_th = self.cn * self.n1
        self.ep_th = self.cp * self.p1
        self.mu_n = p.mu_n / sc.mu0
        self.mu_p = p.mu_p / sc.mu0
        self.vsat_n = p.v_sat_n / (sc.mu0 * sc.E0)   # scaled: v/(mu0*E0) so that mu*E is a velocity in the same units
        self.vsat_p = p.v_sat_p / (sc.mu0 * sc.E0)
        self.S = p.S_surface / (sc.mu0 * sc.E0)      # surface recombination velocity, same scaling as a velocity
        # equilibrium
        self.psi_eq, self.n_eq, self.p_eq = self._equilibrium()

    # ------------------------------------------------------------ equilibrium
    def _equilibrium(self):
        """Dark Fermi level from charge neutrality with the EL2 level (scaled)."""
        ni, NT, NA, ND, n1, p1 = self.ni, self.NT, self.NA, self.ND, self.n1, self.p1

        def rho(psi):   # psi = (E_F - E_i)/kT ; n = ni e^psi, p = ni e^-psi
            n = ni * np.exp(psi)
            pp = ni * np.exp(-psi)
            f = (self.cn * n + self.ep_th) / (self.cn * n + self.en_th + self.cp * pp + self.ep_th)
            return pp - n + NT * (1 - f) + ND - NA

        lo, hi = -25.0, 25.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if rho(mid) > 0:
                lo = mid
            else:
                hi = mid
        psi = 0.5 * (lo + hi)
        return psi, ni * np.exp(psi), ni * np.exp(-psi)

    # --------------------------------------------------------- trap kinetics
    def emission_rates(self, phi_scaled):
        """Total emission rates (scaled) including optical emission.

        phi_scaled: photon flux * t0 (so that sigma[cm^2] * phi gives a scaled rate).
        """
        p = self.p
        if p.optical_emission_on:
            en = self.en_th + p.sigma_n_opt * phi_scaled
            ep = self.ep_th + p.sigma_p_opt * phi_scaled
        else:
            en = self.en_th + 0.0 * phi_scaled
            ep = self.ep_th + 0.0 * phi_scaled
        return en, ep

    def trap_terms(self, n, pp, en, ep):
        """Return (N_T+ , R, dR/dn, dR/dp, dNT+/dn, dNT+/dp), all scaled.

        R = net recombination through EL2 (electron net capture = hole net capture)
            + radiative B(np - ni^2).
        """
        cn, cp, NT, B, ni = self.cn, self.cp, self.NT, self.B, self.ni
        den = cn * n + en + cp * pp + ep
        NTp = NT * (en + cp * pp) / den
        num = cn * cp * n * pp - en * ep
        R = NT * num / den + B * (n * pp - ni ** 2)
        dR_dn = NT * cn * (cp * pp * (en + cp * pp + ep) + en * ep) / den ** 2 + B * pp
        dR_dp = NT * cp * (cn * n * (cn * n + en + ep) + en * ep) / den ** 2 + B * n
        dNTp_dn = -NT * (en + cp * pp) * cn / den ** 2
        dNTp_dp = NT * cp * (cn * n + ep) / den ** 2
        return NTp, R, dR_dn, dR_dp, dNTp_dn, dNTp_dp

    def mobility(self, mu0, vsat, E_scaled):
        """Monotone field-dependent mobility mu0/(1 + mu0 |E|/vsat) (scaled)."""
        if not self.p.field_dependent_mobility:
            return mu0 * np.ones_like(E_scaled)
        return mu0 / (1.0 + mu0 * np.abs(E_scaled) / vsat)

    # ------------------------------------------------------------ generation
    def generation(self, xc, zc_semi, scale_factor=1.0):
        """Scaled generation rate G~(x, z) on semiconductor cell centres and the photon flux (scaled).

        xc: (Nx,), zc_semi: (Nzs,) negative. Returns arrays of shape (Nzs, Nx).
        """
        p, sc = self.p, self.sc
        X, Z = np.meshgrid(xc, zc_semi)
        depth_cm = -Z * 1e-4
        lateral = np.exp(-2.0 * (X - p.spot_x0_um) ** 2 / p.spot_w_um ** 2)
        G = p.G0 * lateral * np.exp(-p.alpha_cm * depth_cm) * scale_factor / sc.R0
        phi = p.photon_flux_peak * lateral * np.exp(-p.alpha_cm * depth_cm) * scale_factor * sc.t0
        return G, phi

    def describe(self) -> str:
        p, sc = self.p, self.sc
        return (f"ni={p.ni:.2e} cm^-3, n_eq={self.n_eq*sc.C0:.2e}, p_eq={self.p_eq*sc.C0:.2e} cm^-3, "
                f"EL2+ dark={self.NT*sc.C0*(1-(self.cn*self.n_eq+self.ep_th)/(self.cn*self.n_eq+self.en_th+self.cp*self.p_eq+self.ep_th)):.2e} cm^-3, "
                f"e_n^th={self.en_th/sc.t0:.2e} s^-1, c_n={p.c_n:.2e}, c_p={p.c_p:.2e} cm^3/s, "
                f"G0={p.G0:.2e} cm^-3 s^-1, unity-gain I={p.unity_gain_current_mA:.2f} mA")
