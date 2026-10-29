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
            section=section_zh.get(sec, sec) or "其他", advanced=(basic is not None and f.name not in basic),
            choices=tuple(choices.get(f.name, ())), minimum=lo, maximum=hi))
    return out


# ----------------------------------------------------------------------------- Params (steady state)
PARAMS_SECTIONS_ZH = {
    "bias": "偏压与温度",
    "geometry": "几何与网格",
    "GaAs bulk, 300 K": "GaAs 体材料",
    "shallow compensation": "浅能级补偿",
    "EL2 (0/+)": "EL2 深能级",
    "pump": "泵浦光",
    "tip model": "针尖",
    "numerics": "数值求解",
}

PARAMS_LABELS = {
    "V_bias": "偏压 V_bias", "anode_side": "阳极位置 anode_side", "T": "温度 T",
    "gap": "电极间隙 gap", "x_half": "计算区半宽 x_half", "depth": "GaAs 厚度 depth", "air_height": "空气层高度",
    "back_boundary": "背面边界条件", "V_back": "背面电位 V_back",
    "dx_fine": "细网格 dx", "x_fine": "细网格范围 |x|≤", "dx_coarse": "粗网格 dx", "dz_fine": "细网格 dz",
    "z_fine": "细网格 z 范围", "dz_coarse_semi": "粗网格 dz (GaAs)", "dz_coarse_air": "粗网格 dz (空气)",
    "eps_r": "相对介电常数 ε_r", "Eg": "带隙 E_g", "Nc": "导带有效态密度 N_c", "Nv": "价带有效态密度 N_v",
    "mu_n": "电子迁移率 μ_n", "mu_p": "空穴迁移率 μ_p", "v_sat_n": "电子饱和速度", "v_sat_p": "空穴饱和速度",
    "field_dependent_mobility": "场依赖迁移率 μ(E)", "B_rad": "辐射复合系数 B", "S_surface": "表面复合速度 S",
    "contact_density": "注入接触密度", "tau_srh_ns": "额外 SRH 寿命",
    "N_A": "浅受主 N_A", "N_D": "浅施主 N_D",
    "N_T": "EL2 总浓度 N_T", "E_T_below_Ec": "EL2 能级 E_c−E_T", "sigma_n": "电子俘获截面 σ_n",
    "sigma_p": "空穴俘获截面 σ_p", "sigma_n_opt": "光电离截面 σ_n°", "sigma_p_opt": "光电离截面 σ_p°",
    "m_eff_n": "电子有效质量", "m_eff_p": "空穴有效质量",
    "wavelength_nm": "波长 λ", "rep_rate": "重复频率", "P_avg_mW": "平均功率 P_avg", "spot_w_um": "光斑半径 w (1/e²)",
    "spot_x0_um": "光斑中心 x0", "reflectivity": "反射率 R", "alpha_cm": "吸收系数 α", "optical_emission_on": "EL2 光电离",
    "tip_height_um": "针尖采样高度", "tip_x_scan_um": "针尖扫描位置",
    "gummel_max_iter": "Gummel 最大迭代", "gummel_tol": "Gummel 容差", "poisson_max_iter": "Poisson 最大迭代",
    "poisson_tol": "Poisson 容差", "generation_ramp": "光生成率爬升序列", "bias_ramp_steps": "偏压爬升步数",
    "n_floor": "密度下限",
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
    "V_bias": "偏压 V_bias", "anode_side": "阳极位置", "P_avg_mW": "平均功率 P_avg", "rep_rate": "重复频率",
    "fwhm_fs": "脉宽 FWHM", "w_um": "光斑半径 w", "theta_deg": "入射角 θ", "absorbed_fraction": "吸收比例 (1−R)",
    "alpha_cm": "吸收系数 α", "x_beam_um": "光束位置 X", "n_sat": "饱和密度 n_sat", "m_n": "电子有效质量",
    "m_p": "空穴有效质量", "tau_n_ps": "电子动量弛豫 τ_n", "tau_p_ps": "空穴动量弛豫 τ_p", "v_sat": "饱和速度",
    "T": "载流子温度 T", "degenerate_diffusion": "简并扩散 (Fermi–Dirac)", "tip_in_poisson": "针尖作为等势体",
    "tip_height_nm": "针尖高度", "tip_half_angle_deg": "针尖半角", "tip_apex_radius_um": "针尖曲率半径",
    "V_tip": "针尖电位 V_tip", "applied_vertical": "保留外场竖直分量", "dt_fs": "时间步长 dt", "t_end_ps": "模拟时长",
    "record_every_fs": "记录间隔", "tip_x_um": "针尖 x", "heights_nm": "近场高度列表", "snapshot_times_ps": "快照时刻",
    "x_fine": "细网格范围", "dx_fine": "细网格 dx", "x_half": "计算区半宽", "depth": "GaAs 厚度", "air_height": "空气层高度",
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
        ("偏压", ("V_bias", "anode_side")),
        ("激发", ("P_avg_mW", "rep_rate", "fwhm_fs", "w_um", "theta_deg", "absorbed_fraction", "alpha_cm", "x_beam_um", "n_sat")),
        ("输运", ("m_n", "m_p", "tau_n_ps", "tau_p_ps", "v_sat", "T", "degenerate_diffusion")),
        ("针尖", ("tip_in_poisson", "tip_height_nm", "tip_half_angle_deg", "tip_apex_radius_um", "V_tip", "applied_vertical", "tip_x_um")),
        ("运行", ("dt_fs", "t_end_ps", "record_every_fs", "heights_nm", "snapshot_times_ps")),
        ("网格", ("x_fine", "dx_fine", "x_half", "depth", "air_height"))):
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
