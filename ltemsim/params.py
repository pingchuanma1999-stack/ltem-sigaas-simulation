"""Parameter sheet for the biased SI-GaAs near-field LTEM simulation.

Every number carries its unit, a plausible range and where it comes from.
Values marked "assumed" are educated guesses that should be replaced by a
measurement as soon as one exists (spot size, pump power, electrode wiring).

Conventions
-----------
* Cross-section in the (x, z) plane. x runs along the bias field (electrode to
  electrode), z is vertical, z = 0 is the GaAs surface, GaAs occupies z < 0.
* The electrodes are thin Au films on the surface at |x| >= gap/2.
* ``anode_side`` says which electrode is wired to the + terminal of the supply
  when the user calls the condition "+25 V".  Recorded by the user on
  2026-10-26 from the hand-drawn wiring sketch: at "+25 V" the + terminal is on
  the **-X electrode**, i.e. the anode is at -x and the cathode at +x.  (The
  Oct 25 note had guessed the opposite.)  This is a single switch here.
* Experimental X (spot position relative to the tip) maps onto the model as
  x_tip = -X with the spot fixed at the gap centre.  The strongest +25 V
  signal (X = +5 um) therefore has the tip at x = -5 um, i.e. on the
  **anode side** of the spot.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import math

# ----------------------------------------------------------------------------
# Physical constants (SI unless noted)
# ----------------------------------------------------------------------------
Q = 1.602176634e-19        # C
KB = 1.380649e-23          # J/K
EPS0 = 8.8541878128e-14    # F/cm
H_PLANCK = 6.62607015e-34  # J s
C_LIGHT = 2.99792458e8     # m/s


@dataclass
class Params:
    # ------------------------------------------------------------------ bias
    V_bias: float = 25.0            # V, total voltage across the gap
    anode_side: str = "-x"          # "+x" or "-x": electrode on the + terminal at "+V_bias"
                                    #   (user, 2026-10-26: + terminal on the -X electrode)
    T: float = 300.0                # K

    # -------------------------------------------------------------- geometry
    gap: float = 40.0               # um, electrode-to-electrode distance
    x_half: float = 60.0            # um, half width of the simulated box (electrodes from 20 to 60 um)
    depth: float = 30.0             # um, GaAs thickness simulated (bottom: E_z = 0)
    air_height: float = 30.0        # um, air box above the surface
    # With this box the bare field at the gap centre is 4.04 kV/cm, within 2 % of the
    # semi-infinite-strip value V/(pi a) = 3.98 kV/cm (a 40x15 um box gives 4.8, 20 % high).
    # grid control
    dx_fine: float = 0.125          # um, cell size for |x| <= x_fine
    x_fine: float = 9.0             # um
    dx_coarse: float = 2.0          # um, cell size far from the spot
    dz_fine: float = 0.1            # um, cell size near the surface (both sides)
    z_fine: float = 2.0             # um, extent of the fine z region
    dz_coarse_semi: float = 3.0     # um
    dz_coarse_air: float = 3.0      # um

    # ----------------------------------------------------- GaAs bulk, 300 K
    eps_r: float = 12.9             # static permittivity
    Eg: float = 1.424               # eV
    Nc: float = 4.7e17              # cm^-3
    Nv: float = 9.0e18              # cm^-3
    mu_n: float = 6000.0            # cm^2/Vs  (SI-GaAs Hall: 4000-8000; vendor 7500 in McGregor 1994)
    mu_p: float = 350.0             # cm^2/Vs  (200-400; vendor 350 in McGregor 1994)
    v_sat_n: float = 1.0e7          # cm/s, used in the monotone mu(E) = mu0/(1 + mu0 E/v_sat)
    v_sat_p: float = 1.0e7          # cm/s
    field_dependent_mobility: bool = True
    B_rad: float = 1.0e-10          # cm^3/s, band-to-band (Stout & Kushner use 1e-10; literature 1-7e-10)
    S_surface: float = 0.0          # cm/s, surface recombination velocity at the free GaAs surface (0 = off)
    contact_density: float = 0.0    # cm^-3; 0 = equilibrium (blocking) contacts n_eq, p_eq at the metal.
                                    #   >0 = injecting ("doped-ohmic") contacts with n = p = this value at the
                                    #   electrode faces, so the plasma can draw current from the electrodes.
    tau_srh_ns: float = 0.0         # ns, extra bulk SRH lifetime from other defects (0 = off). The EL2 path
                                    #   alone gives a radiative-limited plasma (10-100 ns); set ~0.1-1 ns to
                                    #   test the confined-plasma regime.

    # --------------------------------------------------- shallow compensation
    # LEC SI-GaAs: carbon acceptors ~1e15-1e16, residual donors ~1e15.
    # Only N_A - N_D matters for the dark state; it sets the ionized EL2 fraction
    # (1-2e15 of ~1e16 EL2 ionized in McGregor 1994, i.e. ~10-20 %).
    N_A: float = 2.5e15             # cm^-3
    N_D: float = 1.0e15             # cm^-3

    # ----------------------------------------------------------- EL2 (0/+)
    N_T: float = 1.5e16             # cm^-3, total EL2 (0.5-2e16; "~1e16" McGregor 1994)
    E_T_below_Ec: float = 0.75      # eV   (0.75-0.82 quoted; DFT (+/0) at 0.80 above VBM ~ Ec-0.71)
    sigma_n: float = 1.0e-14        # cm^2, electron capture by EL2+ at 300 K, low field.
                                    #   Thermally activated (barrier ~66 meV); 1e-15 at low field
                                    #   rising to ~1e-13 near 1e4 V/cm (McGregor 1994). 1e-14 is a
                                    #   middle value for our 4-6 kV/cm; sweep 1e-15 .. 1e-13.
    sigma_p: float = 2.0e-18        # cm^2, hole capture by EL2^0 (Mitonneau/Lang value from memory,
                                    #   ~2e-18; not re-verified online. Sweep 1e-18 .. 1e-16.)
    sigma_n_opt: float = 1.0e-16    # cm^2, EL2^0 + hv -> e + EL2+ at ~1.5 eV (Chantre 1981 scale, from memory)
    sigma_p_opt: float = 5.0e-17    # cm^2, EL2+ + hv -> h + EL2^0
    m_eff_n: float = 0.067          # thermal velocity
    m_eff_p: float = 0.5

    # ----------------------------------------------------------------- pump
    # Spectra-Physics Mai Tai: 80 MHz, ~100 fs, 820 nm.  The steady-state model
    # uses the time-averaged generation rate.
    wavelength_nm: float = 820.0
    rep_rate: float = 80.0e6        # Hz
    P_avg_mW: float = 2.0           # mW on the sample. ASSUMED. Calibrate against the
                                    #   measured photocurrent (0.6 mA at 25 V, Oct 2 note):
                                    #   unity-gain current = e P(1-R)/hv = 0.44 A/W * P.
    spot_w_um: float = 3.0          # um, 1/e^2 intensity radius. ASSUMED (0 V step fit gave a
                                    #   Gaussian w = 3.19 um incl. the tip response).
    spot_x0_um: float = 0.0         # um, spot centre relative to the gap centre
    reflectivity: float = 0.33      # at 820 nm (n ~ 3.67)
    alpha_cm: float = 1.2e4         # cm^-1 at 820 nm (standard tables, 1.0-1.5e4; delta ~ 0.83 um)
    optical_emission_on: bool = True

    # ------------------------------------------------------------- tip model
    tip_height_um: float = 0.05     # um above the surface where E_z is sampled
    tip_x_scan_um: tuple = tuple([-10.0 + 0.5 * i for i in range(41)])

    # ------------------------------------------------------------- numerics
    gummel_max_iter: int = 400
    gummel_tol: float = 1e-4        # max |dpsi| in units of kT/q
    poisson_max_iter: int = 60
    poisson_tol: float = 1e-7
    generation_ramp: tuple = (1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
    bias_ramp_steps: int = 2
    n_floor: float = 1.0            # cm^-3, densities are never allowed below this

    # ---------------------------------------------------------- derived
    @property
    def VT(self) -> float:
        return KB * self.T / Q

    @property
    def ni(self) -> float:
        return math.sqrt(self.Nc * self.Nv) * math.exp(-self.Eg / (2 * self.VT))

    @property
    def v_th_n(self) -> float:
        m = self.m_eff_n * 9.1093837e-31
        return math.sqrt(3 * KB * self.T / m) * 100.0   # cm/s

    @property
    def v_th_p(self) -> float:
        m = self.m_eff_p * 9.1093837e-31
        return math.sqrt(3 * KB * self.T / m) * 100.0

    @property
    def c_n(self) -> float:          # cm^3/s
        return self.sigma_n * self.v_th_n

    @property
    def c_p(self) -> float:
        return self.sigma_p * self.v_th_p

    @property
    def n1(self) -> float:           # cm^-3, SRH parameter for the EL2 level
        return self.Nc * math.exp(-self.E_T_below_Ec / self.VT)

    @property
    def p1(self) -> float:
        E_T_above_Ev = self.Eg - self.E_T_below_Ec
        return self.Nv * math.exp(-E_T_above_Ev / self.VT)

    @property
    def photon_energy_J(self) -> float:
        return H_PLANCK * C_LIGHT / (self.wavelength_nm * 1e-9)

    @property
    def photon_flux_peak(self) -> float:
        """Peak photon flux at the surface, cm^-2 s^-1 (after reflection)."""
        area = math.pi * (self.spot_w_um * 1e-4) ** 2 / 2.0     # cm^2, peak intensity = 2P/(pi w^2)
        return self.P_avg_mW * 1e-3 * (1 - self.reflectivity) / self.photon_energy_J / area

    @property
    def G0(self) -> float:
        """Peak generation rate, cm^-3 s^-1."""
        return self.alpha_cm * self.photon_flux_peak

    @property
    def unity_gain_current_mA(self) -> float:
        return Q * self.P_avg_mW * 1e-3 * (1 - self.reflectivity) / self.photon_energy_J * 1e3

    @property
    def spot_length_y_cm(self) -> float:
        """Effective extent of the spot along y (integral of the Gaussian line profile)."""
        return self.spot_w_um * 1e-4 * math.sqrt(math.pi / 2.0)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=list)

    @classmethod
    def from_json(cls, s: str) -> "Params":
        d = json.loads(s)
        d["tip_x_scan_um"] = tuple(d["tip_x_scan_um"])
        d["generation_ramp"] = tuple(d["generation_ramp"])
        return cls(**d)


SOURCES = {
    "McGregor1994": "McGregor et al., 'Evidence for field enhanced electron capture by EL2 centers in "
                    "semi-insulating GaAs...', Univ. Michigan Deep Blue (1994): EL2 ~1e16 cm^-3, level ~Ec-0.8 eV, "
                    "sigma_n ~1e-15 (low field) -> ~1e-13 (1e4 V/cm), e_n ~0.1 s^-1, mu_n 7500, mu_p 350, ionized EL2 1-2e15.",
    "StoutKushner1996": "Stout & Kushner, J. Appl. Phys. 79, 2084 (1996): radiative coefficient 1e-10 cm^3/s used for SI-GaAs switches.",
    "Pizzuto2023": "Pizzuto, Castro-Camus, Mittleman, J. Infrared Millim. THz Waves 44, 769 (2023): 80 MHz, 100 fs, 820 nm, "
                   "near-field LTEM; pump power not stated.",
    "RalphGrischkowsky1991": "Ralph & Grischkowsky, APL 59, 1972 (1991): trap-enhanced field in SI-GaAs near the anode.",
    "Murakami2016": "Murakami et al., Photonics Research 4, A9 (2016): LTEM of SI-GaAs PC switches, screening time scales.",
    "memory": "sigma_p ~2e-18 cm^2 (Mitonneau 1979 / Lang) and EL2 optical cross sections (Chantre 1981) are "
              "quoted from memory and were not re-verified online in this session.",
}

DEFAULT = Params()
