"""Pump-field FDTD page (scripts/run_fdtd_pump.py and scripts/run_fdtd_heights.py)."""
from __future__ import annotations

import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGroupBox,
                               QHBoxLayout, QLabel, QLayout, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QScrollArea, QSplitter, QStackedWidget, QTabBar, QTabWidget, QVBoxLayout,
                               QWidget)

from .. import icons, nav, paths
from ..jobs import CANCELLED, FAILED, FINISHED, JobManager
from ..paramspec import FieldSpec, format_float
from ..theme import ThemeManager
from ..widgets.common import MetricTile, Page, button, error_box, label, reveal
from ..widgets.imageview import ImageView
from ..widgets.mpl import PlotWidget
from ..widgets.paramform import ParamForm
from ..widgets.runpanel import RunPanel
from ..workers import run_async

C0 = 2.99792458e8
PUMP_DIR = paths.RESULTS / "fdtd_pump"
HEIGHTS_DIR = PUMP_DIR / "heights"
DEFAULT_REF = PUMP_DIR / "notip.npz"
HEIGHTS_DEFAULT = (10, 15, 25, 45, 80, 130, 200, 270, 330, 370)
BOX = dict(x_half=4.0e-6, z_min=-3.0e-6, z_max=2.5e-6, z_source=2.2e-6)    # hard-coded by both scripts
# measured cost of optics_fdtd.run on this machine: ~0.04 ms per step + ~14.5 ns per cell and step (incl. PML);
# dx = 15 nm: 70–80 s per run, dx = 40 nm: 5 s (whole dx = 40 nm job: 16 s).  JOB_OVERHEAD = start-up + figure.
STEP_OVERHEAD, CELL_STEP, RUN_OVERHEAD, JOB_OVERHEAD = 4e-5, 14.5e-9, 0.5, 3.0
SPREAD = (0.8, 1.35)            # machine load; shown as a range
PUMP_PROGRESS = {"type": "count", "pattern": r"^(vacuum|notip|tip): launched", "total": 3, "sub": r"step (\d+)/(\d+)"}
CMAPS = ("RdBu_r", "coolwarm", "viridis", "magma", "inferno", "plasma", "cividis", "turbo", "gray")

# form field -> (FDTDConfig field, factor to SI); every other field keeps its name and unit
SCALE = {"wavelength_nm": ("wavelength", 1e-9), "tip_radius_nm": ("tip_radius", 1e-9),
         "tip_gap_nm": ("tip_gap", 1e-9), "tip_x_um": ("tip_x", 1e-6)}
UNSCALE = {si: (form, f) for form, (si, f) in SCALE.items()}
# arrays every ratio_h*.npz of run_fdtd_heights.py holds (an interrupted save leaves some of them missing or empty)
HEIGHT_KEYS = ("x", "z", "ratio", "Qtip", "Qnotip", "absorbed", "launched")


def pump_specs() -> list[FieldSpec]:
    """FDTDConfig fields that run_fdtd_pump.main can pass through, in convenient units.

    dx goes through dx_nm; x_half, z_min, z_max, z_source, gaas and tip are fixed by the script
    (passing them raises a duplicate-keyword error); eps_inf_tip and snapshot_every do nothing.
    """
    B, G, T, N = "光束", "GaAs（820 nm）", "针尖（PtIr，Drude 模型）", "网格与数值"
    return [
        FieldSpec("dx_nm", 15.0, "float", "网格步长 dx", "nm",
                  "Yee 网格边长。耗时 ∝ dx⁻³。高度序列固定使用 15 nm，因此其参考场也必须是 15 nm。", N, minimum=2, maximum=100),
        FieldSpec("courant", 0.5, "float", "Courant 数", "", "dt = courant·dx/c。2-D 稳定极限 1/√2 ≈ 0.707。", N,
                  advanced=True, minimum=0.05, maximum=0.7),
        FieldSpec("t_end_fs", 0.0, "float", "模拟时长 t_end", "fs", "0 = 自动（脉冲 + 两倍传播时间，默认约 535 fs）。", N,
                  advanced=True, minimum=0),
        FieldSpec("n_pml", 20, "int", "PML 层数", "", "每侧 CPML 吸收层的网格数。", N, advanced=True, minimum=4, maximum=80),
        FieldSpec("pml_m", 3, "int", "PML 阶数 m", "", "CPML 电导率多项式阶数。", N, advanced=True, minimum=1, maximum=6),
        FieldSpec("theta_deg", 45.0, "float", "入射角 θ", "deg", "相对表面法线；光束向 +x 方向传播。", B, minimum=0, maximum=80),
        FieldSpec("w_um", 3.0, "float", "光斑半径 w (1/e²)", "µm", "投影到表面上的 1/e² 强度半径。", B, minimum=0.05),
        FieldSpec("x_beam_um", 0.0, "float", "光束位置 X", "µm", "光轴与表面交点的 x 坐标（针尖位于 tip_x）。", B),
        FieldSpec("wavelength_nm", 820.0, "float", "波长 λ", "nm",
                  "FDTDConfig.wavelength 以米为单位，这里自动换算。注意 n、κ 和针尖 Drude 参数是 820 nm 的值，不会随 λ 改变。",
                  B, minimum=100, maximum=5000),
        FieldSpec("fwhm_fs", 100.0, "float", "脉宽 FWHM", "fs", "强度半高全宽。脉宽越长，模拟时长越长。", B, minimum=5, maximum=1000),
        FieldSpec("n_gaas", 3.67, "float", "折射率 n", "", "GaAs 在泵浦波长的折射率实部。", G, minimum=1),
        FieldSpec("kappa_gaas", 0.078, "float", "消光系数 κ", "", "α = 4πκ/λ；0.078 对应 1.2×10⁴ cm⁻¹ @ 820 nm。", G, minimum=0),
        FieldSpec("tip_radius_nm", 40.0, "float", "针尖曲率半径 R", "nm", "FDTDConfig.tip_radius（米），自动换算。", T, minimum=1),
        FieldSpec("tip_half_angle_deg", 15.0, "float", "针尖半角", "deg",
                  "2-D 模型中针尖是 y 方向无限长的楔形，阴影是上限。", T, minimum=1, maximum=60),
        FieldSpec("tip_gap_nm", 20.0, "float", "针尖–表面间距 gap", "nm", "针尖顶点到 GaAs 表面的距离（FDTDConfig.tip_gap，米）。",
                  T, minimum=0),
        FieldSpec("tip_x_um", 0.0, "float", "针尖位置 x", "µm", "FDTDConfig.tip_x（米），自动换算。", T, advanced=True),
        FieldSpec("omega_p_tip", 1.86e16, "float", "等离子体频率 ω_p", "rad/s", "单频 Drude 拟合 (Pt @ 820 nm: ε ≈ −17.6 + 29.6i)。",
                  T, advanced=True, minimum=0),
        FieldSpec("gamma_tip", 3.66e15, "float", "阻尼 γ", "rad/s", "Drude 阻尼率。", T, advanced=True, minimum=0),
    ]


def pump_kwargs(ov: dict) -> dict:
    """Changed form values -> run_fdtd_pump.main keyword arguments (FDTDConfig units)."""
    kw = {}
    for k, v in ov.items():
        if k == "t_end_fs":
            if v and v > 0:
                kw[k] = float(v)
        elif k in SCALE:
            name, f = SCALE[k]
            kw[name] = float(f"{float(v) * f:.12g}")       # 3e-08, not 3.0000000000000004e-08
        else:
            kw[k] = v
    return kw


def form_kwarg(k: str, v) -> str:
    """A run_fdtd_pump kwarg as shown to the user: in the form's units (tip_gap=3e-08 -> tip_gap_nm=30)."""
    if k in UNSCALE and isinstance(v, (int, float)):
        k, v = UNSCALE[k][0], float(f"{v / UNSCALE[k][1]:.10g}")
    return f"{k}={format_float(v) if isinstance(v, float) else v}"


def cfg_to_form(cfg: dict) -> dict:
    """FDTDConfig dict (from an npz) -> form values."""
    r = lambda v: float(f"{v:.6g}")
    out = {}
    for spec in pump_specs():
        n = spec.name
        if n == "dx_nm" and "dx" in cfg:
            out[n] = r(cfg["dx"] * 1e9)
        elif n in SCALE and SCALE[n][0] in cfg:
            out[n] = r(cfg[SCALE[n][0]] / SCALE[n][1])
        elif n == "t_end_fs":
            out[n] = float(cfg.get("t_end_fs") or 0.0)
        elif n in cfg:
            out[n] = cfg[n] if spec.type == "int" else r(float(cfg[n]))
    return out


def fdtd_cost(dx_nm=15.0, courant=0.5, fwhm_fs=100.0, t_end_fs=0.0, n_pml=20, **_):
    """(Nx, Nz incl. PML, number of steps, estimated seconds per run) — mirrors optics_fdtd.run."""
    if not (dx_nm > 0 and courant > 0 and fwhm_fs > 0 and n_pml >= 0):
        raise ValueError("non-positive grid parameter")
    dx = dx_nm * 1e-9
    nx = int(round(2 * BOX["x_half"] / dx)) + 2 * n_pml
    nz = int(round((BOX["z_max"] - BOX["z_min"]) / dx)) + 2 * n_pml
    dt = courant * dx / C0
    sig = fwhm_fs * 1e-15 / 2.3548 * math.sqrt(2.0)
    t_end = t_end_fs * 1e-15 if t_end_fs and t_end_fs > 0 else 8 * sig + 2 * (BOX["z_source"] - BOX["z_min"]) / C0 + 2e-14
    steps = int(t_end / dt)
    return nx, nz, steps, steps * (STEP_OVERHEAD + CELL_STEP * nx * nz)


def fmt_dur(s: float) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def fmt_range(s: float) -> str:
    a, b = SPREAD[0] * s, SPREAD[1] * s
    if b < 90:
        return f"{a:.0f}–{b:.0f} s"
    if b < 5400:
        return f"{a / 60:.1f}–{b / 60:.1f} min"
    return f"{a / 3600:.1f}–{b / 3600:.1f} h"


def pump_total(values: dict) -> float:
    return 3 * (fdtd_cost(**values)[3] + RUN_OVERHEAD) + JOB_OVERHEAD


def heights_total(n: int) -> float:
    return n * (fdtd_cost()[3] + RUN_OVERHEAD) + (JOB_OVERHEAD if n else 0.0)


def heights_in(d: Path) -> dict[int, Path]:
    out = {}
    for f in Path(d).glob("ratio_h*.npz"):
        try:
            out[int(f.stem[7:])] = f
        except ValueError:
            pass
    return dict(sorted(out.items()))


def read_json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def read_dict(p: Path) -> dict:
    """A JSON object file, or {} when it is missing, unreadable or not an object."""
    d = read_json(p)
    return d if isinstance(d, dict) else {}


def height_npz_problem(f: Path, deep=False) -> str | None:
    """None if `f` is a complete ratio_h*.npz, else a short reason.

    The quick check reads only the zip directory: a killed process leaves no directory (BadZipFile), a
    KeyboardInterrupt inside np.savez leaves a valid zip whose later members are missing or empty.
    deep=True also decompresses every array.
    """
    try:
        with zipfile.ZipFile(f) as zf:
            size = {i.filename: i.file_size for i in zf.infolist()}
        bad = [k for k in HEIGHT_KEYS if size.get(f"{k}.npy", 0) < 64]       # 64 = smallest .npy header
        if bad:
            return "缺少或空的数组：" + ", ".join(bad)
        if deep:
            with np.load(f, allow_pickle=False) as d:
                for k in HEIGHT_KEYS:
                    np.asarray(d[k])
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


def quarantine(files) -> list[str]:
    """Rename broken height files to *.npz.partial (run_fdtd_heights.py would skip them as 'exists')."""
    moved = []
    for f in files:
        f = Path(f)
        dst = f.with_suffix(".npz.partial")
        i = 2
        while dst.exists():
            dst = f.with_suffix(f".npz.partial{i}")
            i += 1
        try:
            f.rename(dst)
            moved.append(f.name)
        except OSError:
            pass
    return moved


def npz_cfg(p: Path) -> dict | None:
    try:
        with np.load(p, allow_pickle=False) as d:
            return json.loads(str(d["cfg"]))
    except Exception:
        return None


def check_ref(p: Path) -> tuple[str, str]:
    """('ok' | 'warn' | 'err', Chinese message) for a reference notip.npz of run_fdtd_heights."""
    from ltemsim.optics_fdtd import FDTDConfig
    p = Path(p)
    if not p.exists():
        return "err", f"文件不存在：{paths.rel(p)}。运行时可以先排队一次默认参数的泵浦 FDTD 来生成参考场。"
    if p.is_dir():
        return "err", f"{paths.rel(p)} 是文件夹，请选择其中的 notip.npz。"
    cfg = npz_cfg(p)
    if cfg is None:
        return "err", "无法读取该文件的 cfg（不是 run_fdtd_pump.py 写出的 npz，或文件不完整）。"
    if not cfg.get("gaas", True):
        return "err", "这是真空参考（vacuum.npz，gaas = False）：没有 GaAs 吸收，Q_notip ≡ 0，比值无意义。请选择 notip.npz。"
    if cfg.get("tip"):
        return "err", ("这是有针尖的结果（tip.npz）。ratio_h*.npz 会被下游（针尖位置链）当作 Q_tip/Q_notip 使用，"
                       "参考场必须是无针尖的 notip.npz。")
    if abs(cfg.get("dx", 0) - 15e-9) > 1e-12 or any(abs(cfg.get(k, 0) - v) > 1e-12 for k, v in BOX.items()):
        return "err", (f"网格不匹配：dx = {cfg.get('dx', 0) * 1e9:.1f} nm。高度序列固定使用 15 nm 默认网格和计算区，"
                       "用这个参考场会因数组形状不一致而失败。")
    d0 = FDTDConfig()
    diffs = []
    for k, unit, f in (("theta_deg", "°", 1), ("w_um", " µm", 1), ("x_beam_um", " µm", 1), ("wavelength", " nm", 1e9),
                       ("fwhm_fs", " fs", 1), ("n_gaas", "", 1), ("kappa_gaas", "", 1), ("courant", "", 1)):
        v, v0 = cfg.get(k), getattr(d0, k)
        if v is not None and abs(v - v0) > 1e-9 * max(1.0, abs(v0)):
            diffs.append(f"{k} = {format_float(v * f)}{unit}（默认 {format_float(v0 * f)}{unit}）")
    if cfg.get("t_end_fs"):
        diffs.append(f"t_end = {format_float(cfg['t_end_fs'])} fs（默认自动）")
    if diffs:
        return "warn", ("⚠ 光束/材料参数与高度序列（固定为 FDTDConfig 默认值）不同：" + "；".join(diffs)
                        + "。比值会混合两种光束。")
    return "ok", f"✓ 15 nm 默认网格 · θ = {cfg['theta_deg']:g}° · w = {cfg['w_um']:g} µm · 与高度序列一致"


def tip_outline(x0, gap, R, half_deg, z_top=20.0):
    """Outline (µm) of the 2-D tip: apex circle + wedge, as in optics_fdtd.in_tip."""
    th = math.radians(half_deg)
    zc = gap + R
    phi = np.linspace(math.pi - th, 2 * math.pi + th, 64)
    z_t = zc + R * math.sin(th)
    half_top = R * math.cos(th) + (z_top - z_t) * math.tan(th)
    xs = np.concatenate([[x0 - half_top], x0 + R * np.cos(phi), [x0 + half_top]])
    zs = np.concatenate([[z_top], zc + R * np.sin(phi), [z_top]])
    return xs, zs


def _log10(a, floor=1e-30):
    return np.log10(np.maximum(a, floor))


def _air_nan(a, z):
    a = np.array(a, float)
    a[:, z >= 0] = np.nan
    return a


# ----------------------------------------------------------------------------- loading (worker threads)
def load_pump(d: Path) -> dict:
    """Everything the pump-results view needs, from whatever files the folder has."""
    from ltemsim.optics_fdtd import fresnel_p_absorption
    d = Path(d)
    res = {"dir": d, "summary": read_dict(d / "summary.json"), "runs": {}, "layers": [], "gui": read_json(d / "gui_run.json")}
    for name in ("vacuum", "notip", "tip"):
        p = d / f"{name}.npz"
        if not p.exists():
            continue
        try:
            with np.load(p, allow_pickle=False) as z:
                res["runs"][name] = dict(x=z["x"] * 1e6, z=z["z"] * 1e6, Q=z["Q"], Emax2=z["Emax2"],
                                         launched=float(z["launched"]), absorbed=float(z["absorbed"]),
                                         cfg=json.loads(str(z["cfg"])))
        except Exception as e:                        # partially written file of a cancelled run
            res.setdefault("errors", []).append(f"{name}.npz: {type(e).__name__}: {e}")
    runs = res["runs"]
    ratio = Qt = Qn = None
    x = z = None
    if (d / "ratio.npz").exists():
        try:
            with np.load(d / "ratio.npz", allow_pickle=False) as r:
                x, z, ratio, Qt, Qn = r["x"] * 1e6, r["z"] * 1e6, r["ratio"], r["Qtip"], r["Qnotip"]
        except Exception as e:
            res.setdefault("errors", []).append(f"ratio.npz: {type(e).__name__}: {e}")
    if Qt is None and "tip" in runs and "notip" in runs:
        x, z = runs["tip"]["x"], runs["tip"]["z"]
        Qt, Qn = runs["tip"]["Q"], runs["notip"]["Q"]
        ratio = np.where(Qn > 1e-3 * Qn.max(), Qt / np.maximum(Qn, 1e-300), 1.0)
    if x is None and runs:
        r0 = next(iter(runs.values()))
        x, z = r0["x"], r0["z"]
    cfg = (runs.get("tip") or runs.get("notip") or runs.get("vacuum") or {}).get("cfg")
    res["cfg"] = cfg
    tip = None
    tcfg = (runs.get("tip") or {}).get("cfg") or cfg
    if tcfg:
        tip = dict(x0=tcfg["tip_x"] * 1e6, gap=tcfg["tip_gap"] * 1e6, R=tcfg["tip_radius"] * 1e6,
                   half=tcfg["tip_half_angle_deg"])

    # headline numbers: summary.json, else recomputed like the script does
    m = dict(res["summary"])
    L = runs.get("vacuum", {}).get("launched")
    if L:
        if "absorbed_fraction_notip" not in m and "notip" in runs:
            m["absorbed_fraction_notip"] = runs["notip"]["absorbed"] / L
        if "absorbed_fraction_tip" not in m and "tip" in runs:
            m["absorbed_fraction_tip"] = runs["tip"]["absorbed"] / L
    if "fresnel_p" not in m and cfg:
        m["fresnel_p"] = float(fresnel_p_absorption(cfg["n_gaas"], cfg["kappa_gaas"], cfg["theta_deg"]))
    if Qt is not None:
        gz = z < 0
        col_t, col_n = Qt[:, gz].sum(axis=1), Qn[:, gz].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            col_r = col_t / col_n
        res["columns"] = dict(x=x, t=col_t, n=col_n, r=col_r)
        if "column_ratio_min" not in m:
            m["column_ratio_min"] = float(np.nanmin(col_r))
            m["column_ratio_max"] = float(np.nanmax(col_r))
        if "surface_ratio_under_apex" not in m:
            m["surface_ratio_under_apex"] = float(ratio[np.argmin(np.abs(x)), np.argmin(np.abs(z + 0.02))])
    res["metrics"] = m

    # colour-map layers (already log10'd), the defaults follow fdtd_pump.png
    L_ = res["layers"]
    if ratio is not None:
        with np.errstate(divide="ignore"):
            L_.append(dict(key="ratio", group="ratio", label="log₁₀(Q_tip / Q_notip) — 增强 / 阴影", x=x, z=z,
                           data=_air_nan(np.log10(np.maximum(ratio, 1e-30)), z), cmap="RdBu_r", v=(-1.0, 1.0),
                           xlim=(-2.0, 2.0), zlim=(-1.5, 0.3), title="log10(Q_tip / Q_notip): enhancement (red), shadow (blue)",
                           cbar="log10 ratio", tip=tip))
    if Qn is not None:
        qmax = Qn.max()
        L_.append(dict(key="Qtip", group="Q", label="log₁₀ Q 有针尖（/ max Q_notip）", x=x, z=z,
                       data=_air_nan(_log10(Qt / qmax), z), cmap="viridis", v=(-4.0, 0.5), xlim=(-4.0, 4.0),
                       zlim=(-2.0, 0.3), title="log10 absorbed energy density, with tip (/ max Q_notip)", cbar="log10 Q", tip=tip))
        L_.append(dict(key="Qnotip", group="Q", label="log₁₀ Q 无针尖（/ max Q_notip）", x=x, z=z,
                       data=_air_nan(_log10(Qn / qmax), z), cmap="viridis", v=(-4.0, 0.5), xlim=(-4.0, 4.0),
                       zlim=(-2.0, 0.3), title="log10 absorbed energy density, no tip (/ max Q_notip)", cbar="log10 Q", tip=None))
    ref_run = runs.get("notip") or runs.get("vacuum")
    if ref_run is not None:
        zz = ref_run["z"]
        inc = ref_run["Emax2"][:, zz > 0.5].max()
        for name, zh, en in (("tip", "有针尖", "with tip"), ("notip", "无针尖", "no tip"), ("vacuum", "真空", "vacuum")):
            if name in runs:
                rr = runs[name]
                L_.append(dict(key=f"E_{name}", group="E", label=f"log₁₀ max|E|² / 入射（{zh}）", x=rr["x"], z=rr["z"],
                               data=_log10(rr["Emax2"] / inc), cmap="magma", v=(-3.0, 2.0),
                               xlim=(float(rr["x"][0]), float(rr["x"][-1])), zlim=(float(rr["z"][0]), float(rr["z"][-1])),
                               title=f"log10 max|E|^2 / incident ({en})", cbar="log10 |E|^2 / incident",
                               tip=tip if name == "tip" else None))
    for r in runs.values():           # the raw arrays are no longer needed
        r.pop("Q", None)
        r.pop("Emax2", None)
    return res


def heights_ref(d: Path) -> Path:
    """The no-tip reference a heights folder was computed against."""
    d = Path(d)
    g = read_dict(d / "gui_run.json")
    ref = (g.get("kwargs") or {}).get("ref")
    if ref:
        return paths.absolute(ref)
    if d.name == "heights" and (d.parent / "notip.npz").exists():
        return d.parent / "notip.npz"
    return DEFAULT_REF


def _launched_absorbed(p: Path) -> tuple[float | None, float | None]:
    try:
        with np.load(p, allow_pickle=False) as z:
            return float(z["launched"]), float(z["absorbed"])
    except Exception:
        return None, None


def load_heights(d: Path) -> dict:
    """A heights folder: per-height summary, ratio-map layers, broken files and the absorbed-fraction basis.

    run_fdtd_heights.py divides the absorbed energy by the reference's `launched`, which for the notip run is
    the NET downward flux below the source (incident minus the reflected part, ~0.86 x the vacuum value),
    whereas run_fdtd_pump.py divides by the vacuum run's.  With vacuum.npz next to the reference the fractions
    are rescaled by launched_notip / launched_vacuum so that they compare with the pump result (and Fresnel).
    """
    d = Path(d)
    files = heights_in(d)
    summ = {}
    for k, v in read_dict(d / "summary.json").items():
        try:
            summ[int(float(k))] = dict(v)
        except (TypeError, ValueError):
            pass
    ref = heights_ref(d)
    ref_summary = read_dict(ref.parent / "summary.json")
    L_ref, A_ref = _launched_absorbed(ref)
    L_vac = _launched_absorbed(ref.parent / "vacuum.npz")[0]
    if L_ref and L_vac:
        norm = dict(basis="vacuum", factor=L_ref / L_vac, notip=A_ref / L_vac)
    else:
        norm = dict(basis="notip", factor=1.0, notip=A_ref / L_ref if L_ref else None)
    res = {"dir": d, "files": files, "summary": summ, "ref": ref, "ref_summary": ref_summary, "layers": [],
           "from_npz": [], "bad": {}, "norm": norm, "gui": read_json(d / "gui_run.json")}
    tcfg = npz_cfg(ref.parent / "tip.npz")       # the pump run's own tip point, if it used the series' tip shape
    if (norm["basis"] == "vacuum" and tcfg and isinstance(ref_summary.get("absorbed_fraction_tip"), (int, float))
            and abs(tcfg.get("tip_radius", 0) - 40e-9) < 1e-12 and tcfg.get("tip_half_angle_deg") == 15.0):
        res["pump_tip"] = (tcfg["tip_gap"] * 1e9, float(ref_summary["absorbed_fraction_tip"]))
    for h, f in files.items():
        prob = height_npz_problem(f)
        if prob:
            res["bad"][h] = prob
            continue
        try:
            with np.load(f, allow_pickle=False) as z:
                x, zz, ratio = z["x"] * 1e6, z["z"] * 1e6, z["ratio"]
                if h not in summ:
                    k0, i0 = np.argmin(np.abs(zz + 0.02)), np.argmin(np.abs(x))
                    summ[h] = dict(absorbed_fraction=float(z["absorbed"]) / L_ref if L_ref else float("nan"),
                                   surface_ratio_under_apex=float(ratio[i0, k0]))
                    res["from_npz"].append(h)
        except Exception as e:
            res["bad"][h] = f"{type(e).__name__}: {e}"
            continue
        with np.errstate(divide="ignore"):
            data = _air_nan(np.log10(np.maximum(ratio, 1e-30)), zz)
        res["layers"].append(dict(key=f"h{h}", group="ratio", label=f"h = {h} nm · log₁₀(Q_tip/Q_notip)", x=x, z=zz, data=data,
                                  cmap="RdBu_r", v=(-1.0, 1.0), xlim=(-2.0, 2.0), zlim=(-1.5, 0.3 + h / 1000),
                                  title=f"log10(Q_tip(h) / Q_notip), h = {h} nm", cbar="log10 ratio",
                                  tip=dict(x0=0.0, gap=h / 1000, R=0.04, half=15.0)))
    res["summary"] = dict(sorted(summ.items()))
    return res


# ----------------------------------------------------------------------------- responsive helpers
def _visible(items):
    return [it for it in items if not it.isEmpty()]


class FlowLayout(QLayout):
    """Items left to right, wrapping onto further lines when the width runs out (narrow windows).

    lead=True: when everything fits on one line the first item (a title) takes the free width and the others are
    right-aligned; otherwise the first item gets a line of its own and the others flow below it.
    """

    def __init__(self, parent=None, hspacing=6, vspacing=6, lead=False, lead_min=240):
        super().__init__(parent)
        self._items = []
        self._hs, self._vs, self._lead, self._lead_min = hspacing, vspacing, lead, lead_min
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return self._place(QRect(0, 0, w, 0), False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._place(rect, True)

    def sizeHint(self):
        items = _visible(self._items)
        m = self.contentsMargins()
        w = sum(it.sizeHint().width() for it in items) + self._hs * max(0, len(items) - 1)
        h = max((it.sizeHint().height() for it in items), default=0)
        return QSize(w + m.left() + m.right(), h + m.top() + m.bottom())

    def minimumSize(self):
        m = self.contentsMargins()
        s = QSize(0, 0)
        for it in _visible(self._items):
            s = s.expandedTo(it.minimumSize())
        return s + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _place(self, rect, apply):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        width = max(1, r.width())
        rest = _visible(self._items)
        lines = []
        if self._lead and rest:
            head, rest = rest[0], rest[1:]
            tail = sum(it.sizeHint().width() for it in rest) + self._hs * len(rest)
            if min(head.sizeHint().width(), self._lead_min) + tail <= width:
                lines.append([(head, width - tail)] + [(it, it.sizeHint().width()) for it in rest])
                rest = []
            else:
                lines.append([(head, width)])
        line, used = [], 0
        for it in rest:
            w = min(it.sizeHint().width(), width)
            if line and used + self._hs + w > width:
                lines.append(line)
                line, used = [], 0
            used += (self._hs if line else 0) + w
            line.append((it, w))
        if line:
            lines.append(line)
        y = r.y()
        for i, ln in enumerate(lines):
            hgt = max(it.sizeHint().height() for it, _ in ln)
            if apply:
                x = r.x()
                for it, w in ln:
                    h = min(it.sizeHint().height(), hgt)
                    it.setGeometry(QRect(QPoint(x, y + (hgt - h) // 2), QSize(w, h)))
                    x += w + self._hs
            y += hgt + (self._vs if i < len(lines) - 1 else 0)
        return y - r.y() + m.top() + m.bottom()


class TileLayout(QLayout):
    """Equal-width MetricTiles in one row; 2 (then 1) per row when a tile would get narrower than `min_w`."""

    def __init__(self, parent=None, min_w=150, spacing=8):
        super().__init__(parent)
        self._items = []
        self._min_w, self._sp = min_w, spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation.Horizontal

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return self._place(QRect(0, 0, w, 0), False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._place(rect, True)

    def sizeHint(self):
        items = _visible(self._items)
        return QSize(len(items) * (self._min_w + self._sp), max((it.sizeHint().height() for it in items), default=0))

    def minimumSize(self):
        return QSize(self._min_w, max((it.minimumSize().height() for it in _visible(self._items)), default=0))

    def _cols(self, width):
        n = c = len(_visible(self._items))
        while c > 1 and (width - (c - 1) * self._sp) / c < self._min_w:
            c = (c + 1) // 2
        return max(1, c) if n else 1

    def _place(self, rect, apply):
        items = _visible(self._items)
        if not items:
            return 0
        cols = self._cols(rect.width())
        cw = (rect.width() - (cols - 1) * self._sp) / cols
        y = rect.y()
        for r0 in range(0, len(items), cols):
            row = items[r0:r0 + cols]
            hgt = max(it.heightForWidth(int(cw)) if it.hasHeightForWidth() else it.sizeHint().height() for it in row)
            if apply:
                for j, it in enumerate(row):
                    x = rect.x() + round(j * (cw + self._sp))
                    it.setGeometry(QRect(x, y, int(cw), hgt))
            y += hgt + self._sp
        return y - self._sp - rect.y()


def _group(*widgets, spacing=4) -> QWidget:
    """Widgets kept together on one line of a FlowLayout."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for x in widgets:
        lay.addWidget(x)
    return w


def _scrolled(w: QWidget) -> QScrollArea:
    """Result-tab content that scrolls vertically when the window is too short for the plots' minimum height."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(w)
    return area


class ElidedLabel(QLabel):
    """Single-line label that elides in the middle when narrow; the full text is in the tooltip."""

    def __init__(self, text="", obj=None):
        super().__init__()
        if obj:
            self.setObjectName(obj)
        self._full = ""
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setText(text)

    def setText(self, text):
        self._full = text
        self.setToolTip(text)
        self._elide()

    def text(self):
        return self._full

    def sizeHint(self):
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + 6, super().sizeHint().height())

    def minimumSizeHint(self):
        return QSize(60, super().minimumSizeHint().height())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._elide()

    def _elide(self):
        QLabel.setText(self, self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideMiddle, max(20, self.width())))


class _PlotStack(QStackedWidget):
    """Plot / empty-state stack whose size hint is its minimum height: the scroll area of a results tab sizes
    its content by height-for-width from the hints, and PlotWidget's own hint (~450 px) would force scrolling."""

    def sizeHint(self):
        return QSize(super().sizeHint().width(), self.minimumHeight())


# ----------------------------------------------------------------------------- colour-map view
class MapView(QWidget):
    """Colour map of one of several (x, z) layers with layer / colormap / range controls."""

    def __init__(self, empty_text="暂无数据", parent=None):
        super().__init__(parent)
        self.layers: list[dict] = []
        self.cur: dict | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        r1 = FlowLayout(hspacing=10)            # both control rows wrap in narrow windows
        self.layer = QComboBox()
        self.layer.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.layer.setMinimumContentsLength(24)
        self.layer.currentIndexChanged.connect(self._on_layer)
        self.cmap = QComboBox()
        self.cmap.addItems(CMAPS)
        self.cmap.currentTextChanged.connect(self._schedule)
        self.tip = QCheckBox("针尖轮廓")
        self.tip.setChecked(True)
        self.tip.toggled.connect(self._schedule)
        self.equal = QCheckBox("等比例")
        self.equal.setChecked(True)
        self.equal.setToolTip("x 与 z 使用相同的长度比例")
        self.equal.toggled.connect(self._schedule)
        r1.addWidget(_group(label("物理量", "muted"), self.layer))
        r1.addWidget(_group(label("色图", "muted"), self.cmap))
        r1.addWidget(_group(self.tip, self.equal, spacing=10))
        rb = button("重置视图", self._reset_view, "flat", "恢复该物理量的默认范围与色图")
        r1.addWidget(rb)
        lay.addLayout(r1)
        r2 = FlowLayout(hspacing=14)
        self.spins = {}
        for key, text, lo, hi, step in (("x", "x", -20, 20, 0.25), ("z", "z", -20, 20, 0.25), ("v", "色标", -40, 40, 0.25)):
            parts = [label(text, "muted")]
            pair = []
            for j in range(2):
                s = QDoubleSpinBox()
                s.setRange(lo, hi)
                s.setDecimals(2)
                s.setSingleStep(step)
                s.setKeyboardTracking(False)
                s.setFixedWidth(74)
                s.valueChanged.connect(self._schedule)
                pair.append(s)
                parts.append(s)
                if j == 0:
                    parts.append(label("–", "muted"))
            if key != "v":
                parts.append(label("µm", "muted"))
            r2.addWidget(_group(*parts))
            self.spins[key] = pair
        lay.addLayout(r2)
        self.stack = _PlotStack()
        self.plot = PlotWidget()
        self.empty = label(empty_text, "muted", wrap=True)
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stack.addWidget(self.plot)
        self.stack.addWidget(self.empty)
        self.stack.setMinimumHeight(260)
        lay.addWidget(self.stack, 1)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._draw)
        self._controls = [self.layer, self.cmap, self.tip, self.equal, rb] + [s for p in self.spins.values() for s in p]
        self.set_layers([], empty_text)

    def set_layers(self, layers, empty_text=None):
        prev = self.cur["key"] if self.cur else None
        self.layers = list(layers)
        self.layer.blockSignals(True)
        self.layer.clear()
        for L in self.layers:
            self.layer.addItem(L["label"])
        idx = next((i for i, L in enumerate(self.layers) if L["key"] == prev), 0)
        self.layer.setCurrentIndex(idx if self.layers else -1)
        self.layer.blockSignals(False)
        for w in self._controls:
            w.setEnabled(bool(self.layers))
        if not self.layers:
            if empty_text:
                self.empty.setText(empty_text)
            self.cur = None
            self.stack.setCurrentWidget(self.empty)
            return
        self.stack.setCurrentWidget(self.plot)
        self.cur = None
        self._on_layer(idx)

    def select(self, key):
        for i, L in enumerate(self.layers):
            if L["key"] == key:
                self.layer.setCurrentIndex(i)
                return

    def _on_layer(self, i):
        if not (0 <= i < len(self.layers)):
            return
        new = self.layers[i]
        if self.cur is None or self.cur.get("group") != new.get("group"):
            self.cur = new
            self._reset_view()
        else:
            self.cur = new
            self._schedule()

    def _set_pair(self, key, lo, hi):
        for s, v in zip(self.spins[key], (lo, hi)):
            s.blockSignals(True)
            s.setValue(v)
            s.blockSignals(False)

    def _reset_view(self):
        L = self.cur
        if L is None:
            return
        self._set_pair("x", *L["xlim"])
        self._set_pair("z", *L["zlim"])
        self._set_pair("v", *L["v"])
        self.cmap.blockSignals(True)
        self.cmap.setCurrentText(L["cmap"])
        self.cmap.blockSignals(False)
        self._schedule()

    def _schedule(self, *_):
        self._timer.start()

    def _draw(self):
        L = self.cur
        if L is None:
            return
        import matplotlib
        x, z = L["x"], L["z"]
        hx = (x[1] - x[0]) / 2 if len(x) > 1 else 0.01
        hz = (z[1] - z[0]) / 2 if len(z) > 1 else 0.01
        ext = [x[0] - hx, x[-1] + hx, z[0] - hz, z[-1] + hz]
        x0, x1 = (s.value() for s in self.spins["x"])
        z0, z1 = (s.value() for s in self.spins["z"])
        v0, v1 = (s.value() for s in self.spins["v"])
        cmap = matplotlib.colormaps[self.cmap.currentText()].copy()
        cmap.set_bad(alpha=0.0)
        show_tip, equal = self.tip.isChecked(), self.equal.isChecked()

        def draw(fig):
            t = ThemeManager.instance().tokens
            ax = fig.add_subplot()
            im = ax.imshow(L["data"].T, origin="lower", extent=ext, cmap=cmap, vmin=min(v0, v1), vmax=max(v0, v1),
                           interpolation="nearest", aspect="equal" if equal else "auto")
            ax.axhline(0.0, color=t.muted, lw=0.8, ls="--")
            if show_tip and L.get("tip"):
                tp = L["tip"]
                xs, zs = tip_outline(tp["x0"], tp["gap"], tp["R"], tp["half"])
                ax.fill(xs, zs, facecolor="#9aa0a6", edgecolor=t.text, alpha=0.45, lw=0.9, zorder=3)
            ax.set_xlim(min(x0, x1), max(x0, x1) if x1 != x0 else x0 + 0.1)
            ax.set_ylim(min(z0, z1), max(z0, z1) if z1 != z0 else z0 + 0.1)
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("z (µm)")
            ax.set_title(L["title"])
            fig.colorbar(im, ax=ax, label=L["cbar"], shrink=0.9)

        self.plot.plot(draw)


# ----------------------------------------------------------------------------- page
class FdtdPage(Page):
    title = "泵浦 FDTD"
    subtitle = ("2-D FDTD：820 nm、100 fs p 偏振泵浦依次照射真空、无针尖、有针尖（PtIr Drude）的 GaAs，"
                "得到吸收能量密度 Q(x,z) 与针尖增强/阴影比值 Q_tip/Q_notip。默认网格约 4–5 分钟。")

    def __init__(self):
        super().__init__()
        self.job = None
        self.post_job = None
        self.pump = None            # loaded pump result dict
        self.heights = None         # loaded heights result dict
        self._pump_token = self._heights_token = 0
        self._levels = {}           # label -> colour level, re-applied on theme change
        self._loaded_once = False

        b = button("文档", lambda: nav.navigate("docs", path=str(paths.DOCS / "pulse_chain.md")), "flat",
                   "docs/pulse_chain.md — Stage 1 pump field")
        self.header.add_action(b)
        self.run_btn = button("  运行泵浦 FDTD", self.run, "primary")
        self.header.add_action(self.run_btn)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._build_left())
        split.addWidget(self._build_right())
        split.setSizes([440, 900])
        self.body.addWidget(split, 1)

        self._update_pump_estimate()
        self._refresh_refs()
        self._update_heights_info()
        self._retheme()
        ThemeManager.instance().changed.connect(self._retheme)
        JobManager.instance().job_finished.connect(self._on_any_finished)

    # ================================================================== layout
    def _build_left(self):
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)
        self.mode_bar = QTabBar()
        self.mode_bar.setDrawBase(False)
        self.mode_bar.setExpanding(False)
        self.mode_bar.addTab("泵浦场（无针尖 / 有针尖）")
        self.mode_bar.addTab("针尖高度序列")
        self.mode_bar.currentChanged.connect(self._on_mode)
        ll.addWidget(self.mode_bar)
        self.mode_stack = QStackedWidget()
        ll.addWidget(self.mode_stack, 1)

        # ---------------- mode 1: pump field
        m1 = QWidget()
        l1 = QVBoxLayout(m1)
        l1.setContentsMargins(0, 0, 0, 0)
        l1.setSpacing(6)
        self.form = ParamForm(pump_specs())
        self.form.changed.connect(self._update_pump_estimate)
        l1.addWidget(self.form, 1)
        self.est1 = MetricTile("预计耗时", "—")
        l1.addWidget(self.est1)
        self.warn1 = label("", wrap=True)
        self.warn1.setVisible(False)
        l1.addWidget(self.warn1)
        orow = QHBoxLayout()
        orow.addWidget(QLabel("输出文件夹"))
        self.out1 = QLineEdit()
        self.out1.setPlaceholderText("自动：results/gui_runs/fdtd_<时间>")
        orow.addWidget(self.out1, 1)
        b = QPushButton("选择…")
        b.clicked.connect(lambda: self._pick_dir(self.out1))
        orow.addWidget(b)
        l1.addLayout(orow)
        self.mode_stack.addWidget(m1)

        # ---------------- mode 2: tip-height series
        m2 = QWidget()
        m2.setObjectName("page")
        l2 = QVBoxLayout(m2)
        l2.setContentsMargins(0, 0, 12, 0)        # room for the overlay scroll bar
        l2.setSpacing(8)
        self.hform = ParamForm([FieldSpec("heights", HEIGHTS_DEFAULT, "tuple", "高度 h 列表", "nm",
                                          "针尖顶点到表面的距离，逗号分隔。文件名用 int(h)，所以 12.5 与 12 会冲突。", "针尖高度")],
                               toolbar=False, scroll=False)
        self.hform.changed.connect(self._update_heights_info)
        self.hform.rows["heights"].editor.setCursorPosition(0)
        hrow = self.hform.rows["heights"]
        gg = hrow.group.layout()                 # the form's own group box: add presets + note below the field
        gg.setColumnMinimumWidth(0, 90)
        prow = QHBoxLayout()
        prow.addWidget(label("预设", "muted"))
        for text, hs in (("默认 10 点", HEIGHTS_DEFAULT), ("快速 3 点", (10, 80, 370)),
                         ("近场加密", (5, 10, 15, 20, 25, 35, 45, 60, 80))):
            pb = button(text, lambda _=False, v=hs: self._set_heights(v), "flat")
            prow.addWidget(pb)
        prow.addStretch(1)
        r0 = gg.rowCount()
        gg.addLayout(prow, r0, 0, 1, 3)
        gg.addWidget(label("每个高度一次完整 FDTD（有针尖）。脚本固定 dx = 15 nm、θ = 45°、w = 3 µm 及其余 FDTDConfig 默认值，"
                           "“泵浦场”表单中的参数不作用于高度序列。", "muted", wrap=True), r0 + 1, 0, 1, 3)
        l2.addWidget(self.hform)

        g = QGroupBox("参考场（无针尖 notip.npz）")
        gl = QVBoxLayout(g)
        rrow = QHBoxLayout()
        self.ref = QComboBox()
        self.ref.setMinimumWidth(150)
        self.ref.currentIndexChanged.connect(self._update_heights_info)
        rrow.addWidget(self.ref, 1)
        b = QPushButton("选择…")
        b.clicked.connect(self._pick_ref)
        rrow.addWidget(b)
        gl.addLayout(rrow)
        self.ref_info = label("", wrap=True)
        gl.addWidget(self.ref_info)
        gl.addWidget(label("比值 = Q_tip(h) / Q_notip，并用参考场的入射能量归一化吸收比例。参考场必须来自 15 nm 默认网格。",
                           "muted", wrap=True))
        l2.addWidget(g)

        g = QGroupBox("输出")
        gl = QVBoxLayout(g)
        orow = QHBoxLayout()
        self.out2 = QLineEdit()
        self.out2.setPlaceholderText("自动：results/gui_runs/fdtd_heights_<时间>")
        self.out2.textChanged.connect(self._update_heights_info)
        orow.addWidget(self.out2, 1)
        b = QPushButton("选择…")
        b.clicked.connect(lambda: self._pick_dir(self.out2))
        orow.addWidget(b)
        gl.addLayout(orow)
        self.out2_info = label("", "muted", wrap=True)
        gl.addWidget(self.out2_info)
        self.rebuild = QCheckBox("完成后重建 summary.json")
        self.rebuild.setChecked(True)
        self.rebuild.setToolTip("汇总文件夹内所有高度。run_fdtd_heights.py 只把本次计算的高度写进 summary.json"
                                "（跳过的高度会丢失）；\n勾选后排队运行 gui_tasks.fdtd_heights_summary，从所有 ratio_h*.npz 重建。")
        gl.addWidget(self.rebuild)
        l2.addWidget(g)
        self.est2 = MetricTile("预计耗时", "—")
        l2.addWidget(self.est2)
        l2.addStretch(1)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setWidget(m2)
        self.mode_stack.addWidget(area)
        left.setMinimumWidth(350)
        return left

    def _build_right(self):
        self.tabs = QTabWidget()
        # ---------------- run
        run_tab = QWidget()
        rl = QVBoxLayout(run_tab)
        rl.setContentsMargins(12, 12, 12, 12)
        rl.setSpacing(8)
        self.run_info = label("尚未提交任务。在左侧选择模式、设置参数，然后点击右上角的运行按钮。"
                              "任务在后台子进程中运行，可在“任务中心”查看全部任务。", "muted", wrap=True)
        rl.addWidget(self.run_info)
        self.panel = RunPanel()
        self.panel.job_finished.connect(self._on_finished)
        rl.addWidget(self.panel, 1)
        self.tabs.addTab(run_tab, "运行")

        # ---------------- pump results
        pt = QWidget()
        pl = QVBoxLayout(pt)
        pl.setContentsMargins(12, 12, 12, 12)
        pl.setSpacing(8)
        top = FlowLayout(lead=True)              # buttons move under the title in narrow windows
        self.p_title = ElidedLabel("尚未载入结果", "cardTitle")
        top.addWidget(self.p_title)
        self.btn_params = button("载入参数", self._load_params, "flat", "把该结果的 FDTDConfig 填入左侧“泵浦场”表单")
        self.btn_as_ref = button("设为高度参考", self._use_as_ref, "flat", "把该结果的 notip.npz 设为针尖高度序列的参考场")
        self.btn_p_heights = button("高度序列", lambda: self.pump and self.show_heights(self.pump["dir"] / "heights"), "flat",
                                    "查看该文件夹下 heights/ 中的针尖高度序列结果")
        self.btn_p_folder = button("打开文件夹", lambda: self.pump and reveal(self.pump["dir"]), "flat")
        for b in (self.btn_params, self.btn_as_ref, self.btn_p_heights, self.btn_p_folder):
            top.addWidget(b)
        pl.addLayout(top)
        self.p_cfg = ElidedLabel("", "muted")
        pl.addWidget(self.p_cfg)
        grid = TileLayout()
        self.m_notip = MetricTile("吸收比例 · 无针尖", "—")
        self.m_tip = MetricTile("吸收比例 · 有针尖", "—")
        self.m_apex = MetricTile("针尖正下方 Q_tip/Q_notip", "—")
        self.m_col = MetricTile("深度积分比值 min – max", "—")
        for m in (self.m_notip, self.m_tip, self.m_apex, self.m_col):
            grid.addWidget(m)
        pl.addLayout(grid)
        self.p_tabs = QTabWidget()
        self.p_tabs.setMinimumHeight(330)
        self.p_img = ImageView()
        self.p_img.setMinimumWidth(240)          # its toolbar row alone asks for ~505 px; do not widen the page
        self.p_tabs.addTab(self.p_img, "概览图")
        self.p_tabs.setTabToolTip(0, "脚本输出的 fdtd_pump.png")
        self.p_map = MapView("该文件夹没有 .npz 数据文件，无法绘制场分布。")
        mw = QWidget()
        ml = QVBoxLayout(mw)
        ml.setContentsMargins(8, 8, 8, 4)
        ml.addWidget(self.p_map)
        self.p_tabs.addTab(mw, "场分布（交互）")
        self.p_cols = PlotWidget()
        self.p_tabs.addTab(self.p_cols, "深度积分 vs x")
        pl.addWidget(self.p_tabs, 1)
        self.tabs.addTab(_scrolled(pt), "泵浦场结果")

        # ---------------- heights results
        ht = QWidget()
        hl = QVBoxLayout(ht)
        hl.setContentsMargins(12, 12, 12, 12)
        hl.setSpacing(8)
        top = FlowLayout(lead=True)
        self.h_title = ElidedLabel("尚未载入高度序列", "cardTitle")
        top.addWidget(self.h_title)
        self.btn_resume = button("在此文件夹续算", self._resume_heights, "flat",
                                 "把该文件夹设为高度序列的输出（已有 ratio_h*.npz 会被跳过）")
        self.btn_summary = button("重建 summary.json", self._rebuild_summary, "flat",
                                  "从文件夹内全部 ratio_h*.npz 重建 summary.json（gui_tasks.fdtd_heights_summary）")
        self.btn_h_folder = button("打开文件夹", lambda: self.heights and reveal(self.heights["dir"]), "flat")
        for b in (self.btn_resume, self.btn_summary, self.btn_h_folder):
            top.addWidget(b)
        hl.addLayout(top)
        grid = TileLayout()
        self.h_n = MetricTile("高度点数", "—")
        self.h_abs = MetricTile("吸收比例 h_min → h_max", "—")
        self.h_apex = MetricTile("针尖下 Q_tip/Q_notip", "—")
        self.h_ref = MetricTile("参考场", "—")
        for m in (self.h_n, self.h_abs, self.h_apex, self.h_ref):
            grid.addWidget(m)
        hl.addLayout(grid)
        self.h_tabs = QTabWidget()
        self.h_tabs.setMinimumHeight(330)
        self.h_plot = PlotWidget()
        self.h_tabs.addTab(self.h_plot, "随高度变化")
        self.h_map = MapView()
        mw = QWidget()
        ml = QVBoxLayout(mw)
        ml.setContentsMargins(8, 8, 8, 4)
        ml.addWidget(self.h_map)
        self.h_tabs.addTab(mw, "比值图（按高度）")
        hl.addWidget(self.h_tabs, 1)
        self.tabs.addTab(_scrolled(ht), "高度序列结果")

        # ---------------- history
        hist = QWidget()
        hy = QVBoxLayout(hist)
        hy.setContentsMargins(12, 12, 12, 12)
        hrow = QHBoxLayout()
        hrow.addWidget(label("已有的 FDTD 结果（双击载入）", "cardTitle"), 1)
        for text, slot in (("在访达中显示", self._reveal_selected), ("刷新", self.refresh_history)):
            hrow.addWidget(button(text, slot, "flat"))
        hy.addLayout(hrow)
        self.hist = QListWidget()
        self.hist.setSpacing(3)
        self.hist.itemDoubleClicked.connect(self._open_item)
        hy.addWidget(self.hist, 1)
        self.tabs.addTab(hist, "历史结果")
        return self.tabs

    def _retheme(self, *_):
        t = ThemeManager.instance().tokens
        self.run_btn.setIcon(icons.icon("play", t.on_accent))
        for w, level in self._levels.items():
            self._paint(w, level)

    def _paint(self, w: QLabel, level: str):
        """Colour a message label by level ('ok' | 'warn' | 'err' | 'muted')."""
        self._levels[w] = level
        t = ThemeManager.instance().tokens
        col = {"ok": t.success, "warn": t.warning, "err": t.danger}.get(level, t.muted)
        w.setStyleSheet(f"color: {col};")

    # ================================================================== mode 1: pump field
    def _on_mode(self, i):
        self.mode_stack.setCurrentIndex(i)
        self.run_btn.setText("  运行泵浦 FDTD" if i == 0 else "  运行高度序列")
        if i == 1:
            self._update_heights_info()

    def _pick_dir(self, edit: QLineEdit):
        start = paths.absolute(edit.text().strip()) if edit.text().strip() else paths.GUI_RUNS
        d = QFileDialog.getExistingDirectory(self, "选择输出文件夹", str(start if start.exists() else paths.RESULTS))
        if d:
            edit.setText(paths.rel(d))

    def _pump_values(self) -> dict:
        return self.form.values()

    def _update_pump_estimate(self, *_):
        v = self._pump_values()
        bad = self.form.invalid_fields()
        try:
            if bad:                       # values() would silently fall back to the defaults
                raise ValueError
            nx, nz, steps, per = fdtd_cost(**v)
        except (TypeError, ValueError, ZeroDivisionError):
            self.est1.set("—", "参数无效" + ("：" + ", ".join(bad) if bad else ""))
            self.warn1.setVisible(False)
            return
        total = 3 * (per + RUN_OVERHEAD) + JOB_OVERHEAD
        self.est1.set(f"≈ {fmt_dur(total)}",
                      f"范围 {fmt_range(total)} · 3 次 FDTD（真空 / 无针尖 / 有针尖）× {steps} 步 · "
                      f"网格 {nx}×{nz}（含 PML）· 耗时 ∝ dx⁻³")
        warns = []
        dx = v.get("dx_nm", 15.0)
        ppw = v.get("wavelength_nm", 820.0) / (max(v.get("n_gaas", 3.67), 1.0) * dx)
        if ppw < 10:
            warns.append(f"网格较粗：GaAs 中每个波长只有 {ppw:.1f} 个网格（默认 15 nm 时为 14.9），结果只适合快速测试。")
        if v.get("tip_radius_nm", 40.0) / dx < 2:
            warns.append(f"针尖曲率半径只有 {v.get('tip_radius_nm', 40.0) / dx:.1f} 个网格，针尖形状分辨不足。")
        if abs(dx - 15.0) > 1e-9:
            warns.append("dx ≠ 15 nm：此结果的 notip.npz 不能作为针尖高度序列的参考场。")
        if total > 1800:
            warns.append(f"运行时间较长（约 {fmt_dur(total)}）。耗时 ∝ dx⁻³。")
        self.warn1.setText("\n".join("⚠ " + w for w in warns))
        self.warn1.setVisible(bool(warns))
        self._paint(self.warn1, "warn")

    def run(self):
        if self.mode_bar.currentIndex() == 1:
            self.run_heights()
            return
        if not self.form.is_valid():
            error_box(self, "参数无效", "以下参数的值无法解析或超出范围：\n" + ", ".join(self.form.invalid_fields()))
            return
        ov = self.form.overrides()
        kw = pump_kwargs(ov)
        txt = self.out1.text().strip()
        out = paths.absolute(txt) if txt else paths.new_run_dir("fdtd")
        if out.exists() and not out.is_dir():
            error_box(self, "输出文件夹无效", f"{paths.rel(out)} 是一个文件，不是文件夹。")
            return
        if out.exists() and any(out.iterdir()):
            committed = out.resolve() == PUMP_DIR.resolve()
            msg = (f"{paths.rel(out)} 已有文件，运行会覆盖其中的 npz、图和 summary.json。"
                   + ("\n\n这是仓库提交的参考结果：高度序列和单脉冲链默认读取其中的 notip.npz / ratio.npz，"
                      "用其他参数覆盖会悄悄改变下游结果。" if committed else "") + "\n\n继续吗？")
            if QMessageBox.question(self, "覆盖已有结果？", msg) != QMessageBox.StandardButton.Yes:
                return
        total = pump_total(self._pump_values())
        if total > 1800 and QMessageBox.question(
                self, "运行时间较长", f"按当前网格估计需要约 {fmt_dur(total)}。确定运行吗？") != QMessageBox.StandardButton.Yes:
            return
        desc = ", ".join(f"{k}={format_float(v) if isinstance(v, float) else v}" for k, v in ov.items()) or "默认参数"
        self.job = JobManager.instance().submit(
            f"泵浦 FDTD · {desc}", "fdtd", "scripts/run_fdtd_pump.py", "main",
            kwargs=dict(out=paths.rel(out), **kw), out=out, progress=PUMP_PROGRESS)
        self.post_job = None
        self.run_info.setText(f"泵浦场 FDTD（3 次运行）· 预计 ≈ {fmt_dur(total)}（{fmt_range(total)}）· "
                              f"输出到 {paths.rel(out)}。中途停止后没有续算，再次运行会重算全部三次。")
        self.panel.attach(self.job)
        self.tabs.setCurrentIndex(0)

    def _load_params(self):
        if self.pump and self.pump.get("cfg"):
            self.form.set_values(cfg_to_form(self.pump["cfg"]), reset_others=True)
            self.mode_bar.setCurrentIndex(0)

    def _use_as_ref(self):
        if not self.pump:
            return
        p = self.pump["dir"] / "notip.npz"
        self._refresh_refs(select=p)
        self.mode_bar.setCurrentIndex(1)

    # ================================================================== mode 2: heights
    def _set_heights(self, hs):
        self.hform.set_values({"heights": tuple(hs)})
        self.hform.rows["heights"].editor.setCursorPosition(0)       # show the start of a long list

    def _heights(self):
        try:
            return tuple(self.hform.values()["heights"])
        except (KeyError, TypeError):
            return ()

    def _refresh_refs(self, select: Path | None = None):
        cur = Path(self.ref.currentData()) if self.ref.currentData() else None
        want = select or cur or DEFAULT_REF
        cands = [DEFAULT_REF] + sorted((p for p in paths.GUI_RUNS.glob("fdtd_*/notip.npz")
                                        if not p.parent.name.startswith("fdtd_heights")), reverse=True)
        if want not in cands:
            cands.append(Path(want))
        self.ref.blockSignals(True)
        self.ref.clear()
        for p in cands:
            note = "（仓库参考结果）" if p == DEFAULT_REF else ("" if p.exists() else "（不存在）")
            self.ref.addItem(paths.rel(p) + note, str(p))
        self.ref.setCurrentIndex(max(0, cands.index(Path(want))))
        self.ref.blockSignals(False)
        self._update_heights_info()

    def _pick_ref(self):
        start = self.ref.currentData() or str(PUMP_DIR)
        f, _ = QFileDialog.getOpenFileName(self, "选择无针尖参考场 notip.npz", start, "notip.npz (*.npz)")
        if f:
            self._refresh_refs(select=Path(f))

    def _out2(self) -> Path | None:
        t = self.out2.text().strip()
        return paths.absolute(t) if t else None

    def _update_heights_info(self, *_):
        hs = self._heights()
        out = self._out2()
        existing = heights_in(out) if out and out.is_dir() else {}
        todo = [h for h in hs if int(h) not in existing]
        if out is None:
            self.out2_info.setText("每次运行新建一个文件夹，不会写入 results/fdtd_pump/heights。")
        elif not out.exists():
            self.out2_info.setText("文件夹不存在，将新建。")
        elif not out.is_dir():
            self.out2_info.setText("⚠ 这是一个文件，不是文件夹。")
        else:
            skip = [h for h in existing if h in {int(x) for x in hs}]
            self.out2_info.setText(f"已有 {len(existing)} 个 ratio_h*.npz" + (
                f"，其中 {len(skip)} 个高度（{', '.join(map(str, skip))} nm）会被跳过（可续算）。" if skip else "。")
                + ("\n⚠ 这是仓库中的高度结果文件夹，将写入 npz 并覆盖 summary.json。" if out.resolve() == HEIGHTS_DIR.resolve() else ""))
        ref = Path(self.ref.currentData()) if self.ref.currentData() else DEFAULT_REF
        level, msg = self._check_ref(ref)
        self.ref_info.setText(msg)
        self._paint(self.ref_info, level)
        per = fdtd_cost()[3] + RUN_OVERHEAD
        total = heights_total(len(todo))
        bad = [h for h in hs if h <= 0]
        dup = len({int(h) for h in hs}) != len(hs)
        if not hs or bad or dup:
            self.est2.set("—", "高度列表无效：需要正数且 int(h) 互不相同。")
        else:
            self.est2.set(f"≈ {fmt_dur(total)}" if todo else "无需计算",
                          (f"范围 {fmt_range(total)} · " if todo else "")
                          + f"{len(todo)} 个高度待算 × 约 {fmt_dur(per)}（每个一次完整 FDTD，15 nm 网格）"
                          + (f"，{len(hs) - len(todo)} 个已存在将跳过" if len(hs) > len(todo) else ""))

    def _check_ref(self, ref: Path):
        """check_ref, memoised on (path, mtime) — it is called on every keystroke of the heights field."""
        try:
            key = (str(ref), ref.stat().st_mtime)
        except OSError:
            key = (str(ref), None)
        if getattr(self, "_ref_memo", (None,))[0] != key:
            self._ref_memo = (key, check_ref(ref))
        return self._ref_memo[1]

    def run_heights(self):
        if not self.hform.is_valid():
            error_box(self, "高度列表无效", "无法解析高度列表，请用逗号分隔的数字，例如 10, 15, 25。")
            return
        hs = self._heights()
        if not hs or any(h <= 0 for h in hs) or len({int(h) for h in hs}) != len(hs):
            error_box(self, "高度列表无效", "高度必须为正数，且取整后互不相同（输出文件名为 ratio_h{int(h)}.npz）。")
            return
        ref = Path(self.ref.currentData()) if self.ref.currentData() else DEFAULT_REF
        out = self._out2() or paths.new_run_dir("fdtd_heights")
        if out.exists() and not out.is_dir():
            error_box(self, "输出文件夹无效", f"{paths.rel(out)} 是一个文件，不是文件夹。")
            return
        jm = JobManager.instance()
        pre = None
        if not ref.exists():
            m = QMessageBox(self)
            m.setIcon(QMessageBox.Icon.Question)
            m.setWindowTitle("缺少参考场")
            m.setText(f"参考场 {paths.rel(ref)} 不存在。\n\n可以先排队运行一次默认参数的泵浦 FDTD（dx = 15 nm，约 4–5 分钟），"
                      "完成后自动开始高度序列，并以它的 notip.npz 为参考。")
            yes = m.addButton("先运行泵浦 FDTD", QMessageBox.ButtonRole.AcceptRole)
            m.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            m.exec()
            if m.clickedButton() is not yes:
                return
            pdir = paths.new_run_dir("fdtd")
            pre = jm.submit("泵浦 FDTD · 默认参数（高度序列参考）", "fdtd", "scripts/run_fdtd_pump.py", "main",
                            kwargs=dict(out=paths.rel(pdir)), out=pdir, progress=PUMP_PROGRESS)
            ref = pdir / "notip.npz"
        else:
            level, msg = check_ref(ref)
            if level == "err":
                error_box(self, "参考场不可用", msg)
                return
            if level == "warn" and QMessageBox.question(self, "参考场参数不一致", msg + "\n\n仍然继续吗？") \
                    != QMessageBox.StandardButton.Yes:
                return
        if out.exists() and out.resolve() == HEIGHTS_DIR.resolve() and QMessageBox.question(
                self, "写入仓库结果文件夹？", "results/fdtd_pump/heights 是仓库中的结果文件夹（其中的 summary.json 已提交）。\n"
                "运行会写入 ratio_h*.npz、gui_run*.json / gui_log*.txt 并覆盖 summary.json。继续吗？") \
                != QMessageBox.StandardButton.Yes:
            return
        if not self._clear_broken(out):
            return
        existing = heights_in(out) if out.is_dir() else {}
        todo = [h for h in hs if int(h) not in existing]
        rel_out, rel_ref = paths.rel(out), paths.rel(ref)
        if not todo:
            if not existing:
                return
            self.post_job = self.job = self._submit_summary(out, ref)
            self.run_info.setText(f"所选高度在 {rel_out} 中都已存在，只重建 summary.json（几秒钟）。")
            self.panel.attach(self.job)
            self.tabs.setCurrentIndex(0)
            return
        hs_json = [int(h) if float(h).is_integer() else float(h) for h in hs]
        self.job = jm.submit(f"针尖高度序列 · {len(hs)} 个高度 ({min(hs_json)}–{max(hs_json)} nm)", "fdtd_heights",
                             "scripts/run_fdtd_heights.py", "main",
                             kwargs=dict(out=rel_out, heights=hs_json, ref=rel_ref), out=out,
                             progress={"type": "count", "pattern": r"^h = ", "total": len(hs)}, after=pre)
        # the follow-up summary job shares the folder: it logs to gui_run_fdtd_heights_summary.json / gui_log_….txt
        self.post_job = self._submit_summary(out, ref, after=self.job) if self.rebuild.isChecked() else None
        total = heights_total(len(todo))
        self.run_info.setText(
            ("已先排队默认参数的泵浦 FDTD（约 4–5 min），完成后开始高度序列。" if pre else "")
            + f"针尖高度序列：{len(todo)} 个高度待算（{len(hs) - len(todo)} 个已存在跳过）· 预计 ≈ {fmt_dur(total)} · "
            f"输出到 {rel_out}。可随时停止，已完成的高度会保留并在下次续算时跳过。")
        self.panel.attach(pre or self.job)
        self.tabs.setCurrentIndex(0)

    def _resume_heights(self):
        if not self.heights:
            return
        d = self.heights["dir"]
        self.out2.setText(paths.rel(d))
        hs = sorted(set(self.heights["summary"]) | set(self.heights["files"]))
        if hs:
            self._set_heights(tuple(hs))
        self._refresh_refs(select=self.heights["ref"])
        self.mode_bar.setCurrentIndex(1)

    def _submit_summary(self, d: Path, ref: Path, after=None):
        """Queue gui_tasks.fdtd_heights_summary for folder `d` (a follow-up job may share the series' folder)."""
        return JobManager.instance().submit(
            f"重建高度汇总 · {paths.rel(d)}", "fdtd_heights", "ltemsim_gui/gui_tasks.py", "fdtd_heights_summary",
            kwargs=dict(out=paths.rel(d), ref=paths.rel(ref)), out=d, after=after)

    def _clear_broken(self, d: Path) -> bool:
        """Offer to move incomplete ratio_h*.npz aside (the script skips them as 'exists' and the summary task
        crashes on them).  False if the user cancelled."""
        files = heights_in(d) if d.is_dir() else {}
        bad = {h: p for h, f in files.items() if (p := height_npz_problem(f))}
        if not bad:
            return True
        detail = "\n".join(f"ratio_h{h}.npz：{p}" for h, p in bad.items())
        if QMessageBox.question(
                self, "发现不完整的结果文件",
                f"{paths.rel(d)} 中有 {len(bad)} 个 ratio_h*.npz 不完整（多半是被中断的运行留下的）：\n\n{detail}\n\n"
                "run_fdtd_heights.py 会把它们当作已完成而跳过，重建 summary.json 也会因此失败。\n"
                "把它们改名为 *.npz.partial（不删除）后继续吗？") != QMessageBox.StandardButton.Yes:
            return False
        quarantine(files[h] for h in bad)
        return True

    def _rebuild_summary(self):
        if not self.heights:
            return
        d, ref = self.heights["dir"], self.heights["ref"]
        if not ref.exists():
            error_box(self, "缺少参考场", f"参考场 {paths.rel(ref)} 不存在，无法计算吸收比例。")
            return
        if d.resolve() == HEIGHTS_DIR.resolve() and QMessageBox.question(
                self, "覆盖仓库中的 summary.json？", "将用文件夹中的 ratio_h*.npz 重写 results/fdtd_pump/heights/summary.json，"
                "并在其中写入 gui_run*.json / gui_log*.txt。继续吗？") != QMessageBox.StandardButton.Yes:
            return
        if not self._clear_broken(d):
            return
        n = len(heights_in(d))
        if not n:
            error_box(self, "无法重建", f"{paths.rel(d)} 中没有完整的 ratio_h*.npz，重建会把 summary.json 清空。")
            self.show_heights(d, switch=False)
            return
        self.post_job = self.job = self._submit_summary(d, ref)
        self.run_info.setText(f"从 {paths.rel(d)} 中的 {n} 个 ratio_h*.npz 重建 summary.json（几秒钟）。")
        self.panel.attach(self.job)
        self.tabs.setCurrentIndex(0)

    # ================================================================== job completion
    def _on_finished(self, job):
        """The job shown in the run panel ended (every fdtd job has an `out` folder)."""
        d = paths.absolute(job.out) if job.out else None
        if job.status == FINISHED and d is not None:
            if job is self.job:
                (self.show_pump if job.kind == "fdtd" else self.show_heights)(d)
            elif job.kind == "fdtd" and self.job is not None and self.job.after is job:
                self.panel.attach(self.job)          # prerequisite pump run done: follow the heights series
                self._refresh_refs(select=d / "notip.npz")
        self.refresh_history()

    def _on_any_finished(self, job):
        if job.kind not in ("fdtd", "fdtd_heights"):
            return
        d = paths.absolute(job.out) if job.out else None
        if job.kind == "fdtd_heights" and job.status in (FAILED, CANCELLED) and d is not None:
            # any stopped series, also one no longer shown in the run panel (e.g. cancelled in the job centre)
            run_async(self._quarantine_partial, d, job.finished, on_done=lambda moved, d=d: self._after_quarantine(d, moved))
        if job is self.post_job and job is not self.job and job.status == FINISHED and d is not None:
            self.show_heights(d, switch=False)
        if job is not self.job and self.isVisible():
            self.refresh_history()
            self._refresh_refs()

    @staticmethod
    def _quarantine_partial(d: Path, before: float | None = None) -> list[str]:
        """Worker: move ratio_h*.npz that a stopped series left incomplete to *.npz.partial (the script would
        skip them as 'exists').  Files newer than `before` belong to a later job and are left alone."""
        bad = []
        for f in heights_in(d).values():
            try:
                if before and f.stat().st_mtime > before + 2:
                    continue
            except OSError:
                continue
            if height_npz_problem(f, deep=True):
                bad.append(f)
        return quarantine(bad)

    def _after_quarantine(self, d: Path, moved: list[str]):
        if moved:
            self.run_info.setText(self.run_info.text() + f"\n已把中断时未写完的 {', '.join(moved)} 改名为 *.npz.partial，"
                                  "续算时会重新计算这些高度。")
            if self.heights and Path(self.heights["dir"]) == d:
                self.show_heights(d, switch=False)
        self._update_heights_info()

    # ================================================================== results: pump
    def show_pump(self, d, switch=True):
        d = Path(d)
        self._pump_token += 1
        tok = self._pump_token
        self.p_title.setText(f"{paths.rel(d)}  ·  载入中…")
        if switch:
            self.tabs.setCurrentIndex(1)
        run_async(load_pump, d, on_done=lambda r: tok == self._pump_token and self._show_pump(r),
                  on_error=lambda e: tok == self._pump_token and self._load_failed(self.p_title, d, e))

    def _load_failed(self, title: QLabel, d, err):
        title.setText(f"{paths.rel(d)}  ·  载入失败")
        error_box(self, "载入失败", f"无法读取 {paths.rel(d)}", err)

    def _show_pump(self, r: dict):
        self.pump = r
        d, m, cfg = r["dir"], r["metrics"], r.get("cfg")
        state = []
        if not r["runs"]:
            state.append("没有 npz 数据")
        elif len(r["runs"]) < 3:
            state.append("未完成：只有 " + " / ".join(r["runs"]))
        if r.get("errors"):
            state.append("部分文件损坏")
        self.p_title.setText(paths.rel(d) + ("  ·  " + "，".join(state) if state else ""))
        f = lambda k, fmt: fmt.format(m[k]) if isinstance(m.get(k), (int, float)) and np.isfinite(m[k]) else "—"
        fr = m.get("fresnel_p")
        self.m_notip.set(f("absorbed_fraction_notip", "{:.3f}"), f"Fresnel 1−|r_p|² = {fr:.3f}" if fr is not None else "")
        rel_tip = (m["absorbed_fraction_tip"] / m["absorbed_fraction_notip"]
                   if m.get("absorbed_fraction_tip") and m.get("absorbed_fraction_notip") else None)
        self.m_tip.set(f("absorbed_fraction_tip", "{:.3f}"), f"无针尖的 {rel_tip:.2f} 倍" if rel_tip else "")
        self.m_tip.setToolTip("相对真空入射能量。2-D 模型中针尖是无限长楔形，阴影是上限。")
        self.m_apex.set(f("surface_ratio_under_apex", "{:.2f}"), "x = 0，z = −20 nm")
        if isinstance(m.get("column_ratio_min"), float):
            self.m_col.set(f"{m['column_ratio_min']:.3f} – {m['column_ratio_max']:.2f}", "阴影 – 增强")
        else:
            self.m_col.set("—", "")
        self.m_col.setToolTip("∫Q_tip dz / ∫Q_notip dz 随 x 的最小值（阴影）与最大值（增强）")
        if cfg:
            self.p_cfg.setText(
                f"θ {cfg['theta_deg']:g}° · w {cfg['w_um']:g} µm · X {cfg['x_beam_um']:g} µm · λ {cfg['wavelength'] * 1e9:.0f} nm · "
                f"FWHM {cfg['fwhm_fs']:g} fs · n {cfg['n_gaas']:g}, κ {cfg['kappa_gaas']:g} · 针尖 R {cfg['tip_radius'] * 1e9:.0f} nm, "
                f"半角 {cfg['tip_half_angle_deg']:g}°, gap {cfg['tip_gap'] * 1e9:.0f} nm · dx {cfg['dx'] * 1e9:g} nm")
        else:
            self.p_cfg.setText(f"θ = {m['theta_deg']:g}° · w = {m['w_um']:g} µm" if "theta_deg" in m else "")
        png = d / "fdtd_pump.png"
        if png.exists():
            self.p_img.load(png)
        else:
            self.p_img.clear("该结果没有 fdtd_pump.png（运行未完成或被中断；图在三次运行全部结束后才生成）。")
        self.p_map.set_layers(r["layers"], "该文件夹没有可用的 .npz 数据文件，无法绘制场分布。")
        cols = r.get("columns")
        if cols is not None:
            tip_x = cfg["tip_x"] * 1e6 if cfg else 0.0
            beam_x = cfg["x_beam_um"] if cfg else 0.0
            self.p_cols.plot(lambda fig: self._draw_columns(fig, cols, tip_x, beam_x))
        else:
            self.p_cols.plot(lambda fig: fig.text(0.5, 0.5, "needs ratio.npz, or tip.npz + notip.npz", ha="center", va="center",
                                                  color=ThemeManager.instance().tokens.muted))
        has_ref = (d / "notip.npz").exists() and check_ref(d / "notip.npz")[0] != "err"
        self.btn_as_ref.setEnabled(has_ref)
        self.btn_as_ref.setToolTip("把该结果的 notip.npz 设为高度序列的参考场" if has_ref
                                   else "需要 15 nm 默认网格的 notip.npz")
        self.btn_params.setEnabled(bool(cfg))
        self.btn_p_heights.setVisible((d / "heights").is_dir())

    @staticmethod
    def _draw_columns(fig, c, tip_x, beam_x):
        t = ThemeManager.instance().tokens
        a1, a2 = fig.subplots(2, 1, sharex=True, height_ratios=[2, 1])
        nmax = np.nanmax(c["n"]) or 1.0
        a1.plot(c["x"], c["n"] / nmax, label="no tip", lw=1.6)
        a1.plot(c["x"], c["t"] / nmax, label="with tip", lw=1.6)
        a1.set_ylabel("depth-integrated absorption\n(norm. to max, no tip)")
        a2.semilogy(c["x"], c["r"], color=t.accent, lw=1.4)
        a2.axhline(1.0, color=t.muted, lw=0.8, ls="--")
        a2.set_ylabel("tip / no tip")
        a2.set_xlabel("x (µm)")
        for ax in (a1, a2):
            ax.axvline(tip_x, color=t.muted, lw=0.8, ls=":")
            ax.grid(True, alpha=0.4)
        a1.axvline(beam_x, color=t.warning, lw=0.8, ls="-.", label="beam axis on surface")
        a1.text(tip_x, a1.get_ylim()[1] * 0.97, " tip", color=t.muted, va="top", fontsize=8.5)
        a1.legend(loc="upper right")
        a1.set_title("Depth-integrated absorption ∫Q dz (beam → +x)")

    # ================================================================== results: heights
    def show_heights(self, d, switch=True):
        d = Path(d)
        self._heights_token += 1
        tok = self._heights_token
        self.h_title.setText(f"{paths.rel(d)}  ·  载入中…")
        if switch:
            self.tabs.setCurrentIndex(2)
        run_async(load_heights, d, on_done=lambda r: tok == self._heights_token and self._show_heights(r),
                  on_error=lambda e: tok == self._heights_token and self._load_failed(self.h_title, d, e))

    @staticmethod
    def _fractions(r: dict):
        """Heights, absorbed fractions on the display basis of load_heights, and apex ratios (NaN when missing)."""
        s, k = r["summary"], r["norm"]["factor"]
        num = lambda v: float(v) if isinstance(v, (int, float)) else float("nan")
        hs = sorted(s)
        return (hs, [num(s[h].get("absorbed_fraction")) * k for h in hs],
                [num(s[h].get("surface_ratio_under_apex")) for h in hs])

    def _show_heights(self, r: dict):
        self.heights = r
        d, files, bad, norm = r["dir"], r["files"], r["bad"], r["norm"]
        note = []
        if r["from_npz"]:
            note.append(f"{len(r['from_npz'])} 个高度不在 summary.json 中，已由 npz 计算")
        if bad:
            note.append(f"{len(bad)} 个 npz 不完整（h = {', '.join(map(str, bad))} nm）")
        self.h_title.setText(paths.rel(d) + ("  ·  " + "，".join(note) if note else ""))
        hs, af, ra = self._fractions(r)
        f = lambda v, fmt: fmt.format(v) if np.isfinite(v) else "—"
        n_ok = len(files) - len(bad)
        if hs:
            self.h_n.set(str(len(hs)), f"{hs[0]}–{hs[-1]} nm · npz {n_ok} 个" + (f"，{len(bad)} 个不完整" if bad else ""))
            nt, k = norm["notip"], norm["factor"]
            if norm["basis"] == "vacuum":
                hint = f"无针尖 {nt:.3f}"
                tip = ("吸收能量 / 真空参考的入射能量，与“泵浦场结果”和 Fresnel 值可直接比较。\n"
                       "summary.json 中的值除以的是 notip 运行的净入射通量（入射减去部分反射），"
                       f"这里乘以 launched_notip / launched_vacuum = {k:.3f}；\n"
                       f"summary.json 原值：{f(af[0] / k, '{:.3f}')} → {f(af[-1] / k, '{:.3f}')}。")
            else:
                hint = f"无针尖 {nt:.3f}（÷ notip 净入射）" if nt else "÷ notip 运行的净入射"
                tip = ("参考场旁没有 vacuum.npz，无法换算为相对真空入射：数值与 summary.json 相同，"
                       "除以 notip 运行的净入射通量（入射减去部分反射），不能与“泵浦场结果”的吸收比例直接比较。")
            self.h_abs.set(f"{f(af[0], '{:.3f}')} → {f(af[-1], '{:.3f}')}", hint)
            self.h_abs.setToolTip(tip)
            self.h_apex.set(f"{f(ra[0], '{:.2f}')} → {f(ra[-1], '{:.2f}')}", "z = −20 nm，<1 为阴影")
        else:
            self.h_n.set("0", f"npz {n_ok} 个" + (f"，{len(bad)} 个不完整" if bad else ""))
            self.h_abs.set("—", "")
            self.h_abs.setToolTip("")
            self.h_apex.set("—", "")
        ref = r["ref"]
        self.h_ref.set(ref.parent.name, ref.name + ("" if ref.exists() else "（不存在）"))
        self.h_ref.setToolTip(paths.rel(ref))
        if hs:
            self.h_plot.plot(lambda fig: self._draw_heights(fig, r))
        else:
            self.h_plot.plot(lambda fig: fig.text(0.5, 0.5, "No summary.json and no complete ratio_h*.npz in this folder",
                                                  ha="center", va="center", color=ThemeManager.instance().tokens.muted))
        self.h_map.set_layers(r["layers"], "该文件夹没有可用的 ratio_h*.npz（这些文件被 .gitignore 排除，仓库中只有 summary.json）。"
                                           "运行一次高度序列后即可在这里查看每个高度的比值图。")
        self.btn_summary.setEnabled(bool(files))
        self.btn_summary.setToolTip(("先把不完整的 npz 改名为 *.npz.partial，再从其余 ratio_h*.npz 重建 summary.json" if bad
                                     else "从文件夹内全部 ratio_h*.npz 重建 summary.json") if files
                                    else "没有 ratio_h*.npz，无法重建（会把 summary.json 清空）")

    @staticmethod
    def _draw_heights(fig, r):
        t = ThemeManager.instance().tokens
        hs, af, ra = FdtdPage._fractions(r)
        norm, rs = r["norm"], r["ref_summary"]
        vac = norm["basis"] == "vacuum"
        a1, a2 = fig.subplots(1, 2)
        a1.plot(hs, af, "o-", lw=1.6, ms=5, label="with tip at h")
        if norm["notip"]:
            a1.axhline(norm["notip"], color=t.muted, ls="--", lw=1, label="no tip")
        if vac and rs.get("fresnel_p"):                 # plane-wave value: a fraction of the incident energy
            a1.axhline(rs["fresnel_p"], color=t.muted, ls=":", lw=1, label="Fresnel 1-|r_p|^2")
        if r.get("pump_tip"):
            g, v = r["pump_tip"]
            a1.plot([g], [v], "D", ms=7, mfc="none", mew=1.4, color=t.text, label=f"pump run (gap {g:.0f} nm)")
        a1.set_xlabel("tip–surface gap h (nm)")
        a1.set_ylabel("absorbed / incident (vacuum run)" if vac else "absorbed / launched (notip run)")
        a1.set_title("Absorbed fraction")
        a1.legend(loc="best")
        a2.plot(hs, ra, "o-", lw=1.6, ms=5, color=t.warning)
        a2.axhline(1.0, color=t.muted, ls="--", lw=1)
        a2.set_xlabel("tip–surface gap h (nm)")
        a2.set_ylabel("Q_tip / Q_notip  (x = 0, z = -20 nm)")
        a2.set_title("Under the apex")
        for ax in (a1, a2):
            ax.grid(True, alpha=0.4)

    # ================================================================== history
    def refresh_history(self):
        self.hist.clear()

        def header(text):
            it = QListWidgetItem(text)
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            f = it.font()
            f.setBold(True)
            it.setFont(f)
            self.hist.addItem(it)

        def when(p: Path):
            try:
                return time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
            except OSError:
                return ""

        header("泵浦场（vacuum / notip / tip）")
        pumps = [PUMP_DIR] + sorted((p for p in paths.GUI_RUNS.glob("fdtd_*")
                                     if p.is_dir() and not p.name.startswith("fdtd_heights")), reverse=True)
        n = 0
        for d in pumps:
            if not d.exists():
                continue
            s = read_dict(d / "summary.json")
            npz = [k for k in ("vacuum", "notip", "tip") if (d / f"{k}.npz").exists()]
            g = read_dict(d / "gui_run.json")
            kw = {k: v for k, v in (g.get("kwargs") or {}).items() if k != "out"}
            if s:
                line = (f"无针尖 {s.get('absorbed_fraction_notip', float('nan')):.3f} · 有针尖 "
                        f"{s.get('absorbed_fraction_tip', float('nan')):.3f} · θ {s.get('theta_deg', 45):g}° · w {s.get('w_um', 3):g} µm")
            elif npz:
                line = "未完成：只有 " + " / ".join(npz)
            else:
                line = "空文件夹（运行失败或仍在进行）"
            line += f" · dx {kw.get('dx_nm', 15):g} nm" if d != PUMP_DIR else " · 仓库参考结果"
            extra = ", ".join(form_kwarg(k, v) for k, v in kw.items() if k != "dx_nm")
            it = QListWidgetItem(f"{paths.rel(d)}\n    {line}" + (f" · {extra}" if extra else "") + f" · {when(d)}")
            it.setData(Qt.ItemDataRole.UserRole, ("pump", str(d)))
            self.hist.addItem(it)
            n += 1
        if not n:
            self.hist.addItem("    还没有泵浦场结果")

        header("针尖高度序列（ratio_h*.npz）")
        n = 0
        for d in [HEIGHTS_DIR] + sorted((p for p in paths.GUI_RUNS.glob("fdtd_heights_*") if p.is_dir()), reverse=True):
            if not d.exists():
                continue
            s = read_dict(d / "summary.json")
            files = heights_in(d)
            hs = sorted({int(float(k)) for k in s} | set(files))
            line = (f"{len(hs)} 个高度：{', '.join(map(str, hs))} nm" if hs else "还没有完成的高度") + \
                   f" · npz {len(files)} 个" + (" · 仓库结果（npz 未提交）" if d == HEIGHTS_DIR and not files else "")
            it = QListWidgetItem(f"{paths.rel(d)}\n    {line} · {when(d)}")
            it.setData(Qt.ItemDataRole.UserRole, ("heights", str(d)))
            self.hist.addItem(it)
            n += 1
        if not n:
            self.hist.addItem("    还没有高度序列结果")

    def _open_item(self, it):
        data = it.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        kind, d = data
        (self.show_heights if kind == "heights" else self.show_pump)(Path(d))

    def _reveal_selected(self):
        it = self.hist.currentItem()
        data = it.data(Qt.ItemDataRole.UserRole) if it else None
        reveal(Path(data[1]) if data else paths.GUI_RUNS if paths.GUI_RUNS.exists() else PUMP_DIR)

    # ================================================================== page API
    @staticmethod
    def _is_heights(d: Path) -> bool:
        if d.name == "heights" or d.name.startswith("fdtd_heights") or heights_in(d):
            return True
        s = read_json(d / "summary.json")
        return isinstance(s, dict) and bool(s) and all(str(k).isdigit() for k in s)

    def on_show(self):
        if not self._loaded_once:
            self._loaded_once = True
            if PUMP_DIR.exists() and self.pump is None:
                self.show_pump(PUMP_DIR, switch=False)
            if HEIGHTS_DIR.exists() and self.heights is None:
                self.show_heights(HEIGHTS_DIR, switch=False)
        self.refresh_history()
        self._refresh_refs()

    def open(self, path=None, **_):
        if not path:
            return
        p = paths.absolute(path)
        d = p if p.is_dir() else p.parent
        self._loaded_once = True
        if self._is_heights(d):
            self.show_heights(d)
            if self.pump is None and PUMP_DIR.exists():
                self.show_pump(PUMP_DIR, switch=False)
        else:
            self.show_pump(d)
            if self.heights is None and (d / "heights").is_dir():
                self.show_heights(d / "heights", switch=False)
