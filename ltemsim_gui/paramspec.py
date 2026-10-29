"""Parameter descriptions for the forms, read from the dataclasses' own source.

The inline comments of ltemsim/params.py (units, ranges, sources) are the
documentation of every parameter; they are parsed here so the GUI tooltips stay in
sync with the code.  Chinese labels and the basic/advanced split are added on top.
"""
from __future__ import annotations

import dataclasses
import inspect
import io
import re
import tokenize
from dataclasses import dataclass


@dataclass
class FieldSpec:
    name: str
    default: object
    type: str                 # "float" | "int" | "bool" | "str" | "tuple" | "choice"
    label: str = ""
    unit: str = ""
    help: str = ""
    section: str = ""
    advanced: bool = False
    choices: tuple = ()
    minimum: float | None = None
    maximum: float | None = None


def _kind(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, (tuple, list)):
        return "tuple"
    return "str"


def _source_comments(cls):
    """{field: comment text}, and {field: section title} from the class body."""
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return {}, {}
    comments, sections = {}, {}
    names = [f.name for f in dataclasses.fields(cls)]
    lines = src.splitlines()
    current_section = ""
    last_field = None
    for line in lines:
        s = line.strip()
        m_sec = re.match(r"#\s*-{3,}\s*(.*?)\s*-*\s*$", s)
        if m_sec and m_sec.group(1):
            current_section = m_sec.group(1).strip()
            last_field = None
            continue
        m = re.match(r"(\w+)\s*:\s*[^=]+=", s)
        if m and m.group(1) in names:
            name = m.group(1)
            sections[name] = current_section
            c = ""
            try:
                toks = list(tokenize.generate_tokens(io.StringIO(s + "\n").readline))
                c = " ".join(t.string[1:].strip() for t in toks if t.type == tokenize.COMMENT)
            except (tokenize.TokenError, IndentationError, SyntaxError):
                if "#" in s:
                    c = s.split("#", 1)[1].strip()
            comments[name] = c
            last_field = name
            continue
        if s.startswith("#") and last_field is not None and line.startswith(" " * 12):
            # continuation comment lines are indented under the inline comment
            comments[last_field] = (comments[last_field] + " " + s.lstrip("#").strip()).strip()
            continue
        if s.startswith("#") and last_field is None:
            continue
        if not s:
            last_field = None
    return comments, sections


_UNIT_RE = re.compile(r"^(V|K|um|nm|ns|ps|fs|eV|Hz|mW|m|cm/s|cm\^-?1|cm\^2/Vs|cm\^3/s|cm\^-3|cm\^2|cm\^-1|m/s|rad/s|deg|cm\^-3 ?\(.*?\))(?=[,;\s]|$)")


def _split_unit(comment: str):
    c = comment.strip()
    m = _UNIT_RE.match(c)
    if m:
        unit = m.group(1)
        rest = c[m.end():].lstrip(",; ").strip()
        return unit, rest
    return "", c


def specs_for(cls, labels: dict | None = None, basic: set | None = None, choices: dict | None = None,
              units: dict | None = None, section_zh: dict | None = None, skip: set | None = None,
              bounds: dict | None = None, section_of: dict | None = None) -> list[FieldSpec]:
    labels = labels or {}
    choices = choices or {}
    units = units or {}
    section_zh = section_zh or {}
    skip = skip or set()
    bounds = bounds or {}
    comments, sections = _source_comments(cls)
    inst = cls()
    out = []
    for f in dataclasses.fields(cls):
        if f.name in skip:
            continue
        v = getattr(inst, f.name)
        kind = "choice" if f.name in choices else _kind(v)
        unit, rest = _split_unit(comments.get(f.name, ""))
        sec = (section_of or {}).get(f.name) or sections.get(f.name, "")
        lo, hi = bounds.get(f.name, (None, None))
        out.append(FieldSpec(
            name=f.name, default=v, type=kind, label=labels.get(f.name, f.name),
            unit=units.get(f.name, unit), help=rest,
            section=section_zh.get(sec, sec) or "Other", advanced=(basic is not None and f.name not in basic),
            choices=tuple(choices.get(f.name, ())), minimum=lo, maximum=hi))
    return out


# ----------------------------------------------------------------------------- Params (steady state)
PARAMS_SECTIONS_ZH = {          # section titles shown in the form (name kept for compatibility)
    "bias": "Bias & temperature",
    "geometry": "Geometry & grid",
    "GaAs bulk, 300 K": "GaAs bulk",
    "shallow compensation": "Shallow compensation",
    "EL2 (0/+)": "EL2 deep level",
    "pump": "Pump",
    "tip model": "Tip",
    "numerics": "Numerics",
}

PARAMS_LABELS = {
    "V_bias": "Bias V_bias", "anode_side": "Anode side", "T": "Temperature T",
    "gap": "Electrode gap", "x_half": "Box half-width x_half", "depth": "GaAs depth", "air_height": "Air height",
    "back_boundary": "Back-surface boundary", "V_back": "Back-surface potential V_back",
    "dx_fine": "Fine grid dx", "x_fine": "Fine grid for |x| ≤", "dx_coarse": "Coarse grid dx", "dz_fine": "Fine grid dz",
    "z_fine": "Fine grid z extent", "dz_coarse_semi": "Coarse dz (GaAs)", "dz_coarse_air": "Coarse dz (air)",
    "eps_r": "Permittivity ε_r", "Eg": "Band gap E_g", "Nc": "Conduction-band DOS N_c", "Nv": "Valence-band DOS N_v",
    "mu_n": "Electron mobility μ_n", "mu_p": "Hole mobility μ_p", "v_sat_n": "Electron v_sat", "v_sat_p": "Hole v_sat",
    "field_dependent_mobility": "Field-dependent mobility μ(E)", "B_rad": "Radiative coefficient B",
    "S_surface": "Surface recombination S", "contact_density": "Injecting-contact density",
    "tau_srh_ns": "Extra SRH lifetime",
    "N_A": "Shallow acceptors N_A", "N_D": "Shallow donors N_D",
    "N_T": "EL2 density N_T", "E_T_below_Ec": "EL2 level E_c − E_T", "sigma_n": "Electron capture σ_n",
    "sigma_p": "Hole capture σ_p", "sigma_n_opt": "Optical cross-section σ_n°", "sigma_p_opt": "Optical cross-section σ_p°",
    "m_eff_n": "Electron effective mass", "m_eff_p": "Hole effective mass",
    "wavelength_nm": "Wavelength λ", "rep_rate": "Repetition rate", "P_avg_mW": "Average power P_avg",
    "spot_w_um": "Spot radius w (1/e²)", "spot_x0_um": "Spot centre x0", "reflectivity": "Reflectivity R",
    "alpha_cm": "Absorption α", "optical_emission_on": "EL2 photo-ionisation",
    "tip_height_um": "Tip sampling height", "tip_x_scan_um": "Tip scan positions",
    "gummel_max_iter": "Gummel max iterations", "gummel_tol": "Gummel tolerance", "poisson_max_iter": "Poisson max iterations",
    "poisson_tol": "Poisson tolerance", "generation_ramp": "Generation ramp", "bias_ramp_steps": "Bias ramp steps",
    "n_floor": "Density floor",
}

PARAMS_BASIC = {
    "V_bias", "anode_side", "T", "gap", "back_boundary", "mu_n", "mu_p", "B_rad", "S_surface", "contact_density",
    "tau_srh_ns", "N_A", "N_D", "N_T", "E_T_below_Ec", "sigma_n", "sigma_p", "P_avg_mW", "spot_w_um", "spot_x0_um",
    "reflectivity", "alpha_cm", "tip_height_um",
}

PARAMS_CHOICES = {"anode_side": ("-x", "+x"), "back_boundary": ("neumann", "dirichlet")}


PARAMS_UNITS = {"wavelength_nm": "nm", "m_eff_n": "m₀", "m_eff_p": "m₀", "gummel_tol": "kT/q"}


def params_specs() -> list[FieldSpec]:
    from ltemsim.params import Params
    return specs_for(Params, labels=PARAMS_LABELS, basic=PARAMS_BASIC, choices=PARAMS_CHOICES,
                     section_zh=PARAMS_SECTIONS_ZH, units=PARAMS_UNITS)


# ----------------------------------------------------------------------------- PulseConfig (single pulse)
PULSE_LABELS = {
    "V_bias": "Bias V_bias", "anode_side": "Anode side", "P_avg_mW": "Average power P_avg", "rep_rate": "Repetition rate",
    "fwhm_fs": "Pulse width (FWHM)", "w_um": "Spot radius w", "theta_deg": "Incidence angle θ",
    "absorbed_fraction": "Absorbed fraction (1−R)", "alpha_cm": "Absorption α", "x_beam_um": "Beam position X",
    "n_sat": "Saturation density n_sat", "m_n": "Electron effective mass", "m_p": "Hole effective mass",
    "tau_n_ps": "Electron momentum relaxation τ_n", "tau_p_ps": "Hole momentum relaxation τ_p", "v_sat": "Saturation velocity",
    "T": "Carrier temperature T", "degenerate_diffusion": "Degenerate diffusion (Fermi–Dirac)",
    "tip_in_poisson": "Tip as equipotential", "tip_height_nm": "Tip height", "tip_half_angle_deg": "Tip half-angle",
    "tip_apex_radius_um": "Tip apex radius", "V_tip": "Tip potential V_tip", "applied_vertical": "Keep vertical applied field",
    "dt_fs": "Time step dt", "t_end_ps": "Simulated time", "record_every_fs": "Record interval", "tip_x_um": "Tip x",
    "heights_nm": "Near-field heights", "snapshot_times_ps": "Snapshot times", "x_fine": "Fine grid extent",
    "dx_fine": "Fine grid dx", "x_half": "Box half-width", "depth": "GaAs depth", "air_height": "Air height",
}

PULSE_UNITS = {
    "V_bias": "V", "P_avg_mW": "mW", "rep_rate": "Hz", "fwhm_fs": "fs", "w_um": "um", "theta_deg": "deg",
    "alpha_cm": "cm^-1", "x_beam_um": "um", "n_sat": "cm^-3", "m_n": "m₀", "m_p": "m₀", "tau_n_ps": "ps",
    "tau_p_ps": "ps", "v_sat": "cm/s", "T": "K", "tip_height_nm": "nm", "tip_half_angle_deg": "deg",
    "tip_apex_radius_um": "um", "V_tip": "V", "dt_fs": "fs", "t_end_ps": "ps", "record_every_fs": "fs",
    "tip_x_um": "um", "heights_nm": "nm", "snapshot_times_ps": "ps", "x_fine": "um", "dx_fine": "um",
    "x_half": "um", "depth": "um", "air_height": "um",
}
PULSE_SECTION_OF = {}
for _sec, _names in (
        ("Bias", ("V_bias", "anode_side")),
        ("Excitation", ("P_avg_mW", "rep_rate", "fwhm_fs", "w_um", "theta_deg", "absorbed_fraction", "alpha_cm", "x_beam_um", "n_sat")),
        ("Transport", ("m_n", "m_p", "tau_n_ps", "tau_p_ps", "v_sat", "T", "degenerate_diffusion")),
        ("Tip", ("tip_in_poisson", "tip_height_nm", "tip_half_angle_deg", "tip_apex_radius_um", "V_tip", "applied_vertical", "tip_x_um")),
        ("Run", ("dt_fs", "t_end_ps", "record_every_fs", "heights_nm", "snapshot_times_ps")),
        ("Grid", ("x_fine", "dx_fine", "x_half", "depth", "air_height"))):
    for _n in _names:
        PULSE_SECTION_OF[_n] = _sec
PULSE_BASIC = {"V_bias", "anode_side", "P_avg_mW", "w_um", "theta_deg", "x_beam_um", "n_sat", "tau_n_ps", "tau_p_ps",
               "T", "degenerate_diffusion", "V_tip", "t_end_ps"}


def pulse_specs() -> list[FieldSpec]:
    from ltemsim.pulse_dynamics import PulseConfig
    return specs_for(PulseConfig, labels=PULSE_LABELS, choices={"anode_side": ("-x", "+x")}, units=PULSE_UNITS,
                     section_of=PULSE_SECTION_OF, basic=PULSE_BASIC)


# ----------------------------------------------------------------------------- value parsing
def parse_value(spec: FieldSpec, text: str):
    """Text from an editor -> typed value; raises ValueError."""
    t = text.strip()
    if spec.type == "float":
        return float(t)
    if spec.type == "int":
        return int(float(t))
    if spec.type == "bool":
        return t.lower() in ("1", "true", "yes", "on")
    if spec.type == "tuple":
        if not t:
            return ()
        parts = [p for p in re.split(r"[,\s;]+", t) if p]
        vals = []
        for p in parts:
            v = float(p)
            vals.append(int(v) if v.is_integer() and all(isinstance(x, int) for x in (spec.default or ())) else v)
        return tuple(vals)
    return t


def format_value(spec: FieldSpec, v) -> str:
    if spec.type == "float":
        return format_float(v)
    if spec.type == "tuple":
        return ", ".join(format_float(x) if isinstance(x, float) else str(x) for x in v)
    return str(v)


def format_float(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e5 or a < 1e-3:
        s = f"{v:.4g}"
        if "e" in s:
            m, e = s.split("e")
            return f"{m}e{int(e)}"
        return s
    return f"{v:.6g}"
