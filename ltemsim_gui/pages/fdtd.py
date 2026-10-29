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
from ..widgets.cores import CoresSpin, estimate_s
from ..widgets.device import FieldView, draw_fdtd_device
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
# both modes run through ltemsim_gui/parallel.py: one FDTD run per core, "@@PAR i/n done" after each
PAR_SCRIPT = "ltemsim_gui/parallel.py"
PAR_PATTERN = r"^@@PAR \d+/\d+ done"
PUMP_PROGRESS = {"type": "count", "pattern": PAR_PATTERN, "total": 3}
CMAPS = ("RdBu_r", "coolwarm", "viridis", "magma", "inferno", "plasma", "cividis", "turbo", "gray")

# tip shape: form label <-> FDTDConfig.tip_shape. The app default is the rod; FDTDConfig's default (and every
# result computed before the choice existed, incl. the committed results/fdtd_pump) is the cone.
ROD, CONE = "Rod (width 2R)", "Cone"
SHAPES = {ROD: "rod", CONE: "cone"}
SHAPE_LABEL = {v: k for k, v in SHAPES.items()}
LEGACY_SHAPE = "cone"

# form field -> (FDTDConfig field, factor to SI); every other field keeps its name and unit
SCALE = {"wavelength_nm": ("wavelength", 1e-9), "tip_radius_nm": ("tip_radius", 1e-9),
         "tip_gap_nm": ("tip_gap", 1e-9), "tip_x_um": ("tip_x", 1e-6)}
UNSCALE = {si: (form, f) for form, (si, f) in SCALE.items()}
# arrays every ratio_h*.npz of run_fdtd_heights.py holds (an interrupted save leaves some of them missing or empty)
HEIGHT_KEYS = ("x", "z", "ratio", "Qtip", "Qnotip", "absorbed", "launched")

# quantities of the COMSOL-style optical-field view
E_TIP = "Electric field norm |E|max / |E0| (field enhancement)"
E_NOTIP = "Electric field norm |E|max / |E0|, no tip"
Q_TIP = "Absorbed energy density Q (normalised)"
Q_NOTIP = "Absorbed energy density Q (normalised), no tip"
RATIO = "Enhancement Q_tip/Q_notip"
WITH_TIP = (E_TIP, Q_TIP, RATIO)             # quantities that get the tip outline
FIELD_NOTE = "|E0| = peak incident field (vacuum run) · Q normalised to max Q_notip — hover for details"
FIELD_HELP = ("|E|max: the peak of |E(t)| in each cell during the pulse (max over time of Ex² + Ez²).\n"
              "|E0|: the peak incident field, the maximum of |E|max in the vacuum run between z = 0.5 µm and the source "
              "plane.\n(fdtd_pump.png divides by the no-tip maximum over all z > 0.5 µm, which includes the source row "
              "itself\nand is ~60× |E0|² on the default grid.)\n"
              "Q: absorbed energy density ∫σ|E|² dt, normalised to its maximum without the tip; zero (blank) in the air.\n"
              "Black lines: the exact tip outline and the GaAs surface z = 0. Cells the FDTD treats as PtIr are blank.")
E0_Z = 0.5                                   # µm; |E0| is measured in the air between this height and the source


SHAPE_HELP = ("Straight rod: the apex sphere of radius R plus a straight shank of width 2R. Cone: the apex sphere plus "
              "a cone of the half-angle below (the repository result results/fdtd_pump used the cone). In the 2-D model "
              "either shank is infinitely long in y.")


def shape_spec(section: str) -> FieldSpec:
    return FieldSpec("tip_shape", ROD, "choice", "Tip shape", "", SHAPE_HELP, section, choices=(ROD, CONE))


def shape_of(d: dict | None) -> str:
    """'rod' | 'cone' of an FDTDConfig dict or run kwargs; without the key: the cone (the library default)."""
    s = (d or {}).get("tip_shape") or LEGACY_SHAPE
    return s if s in SHAPE_LABEL else LEGACY_SHAPE


def shape_text(shape: str, R_nm: float | None = None, half_deg: float | None = None) -> str:
    """'rod tip, 2R = 80 nm' / 'cone tip, R = 40 nm, 15°' (short, for captions and history rows)."""
    if shape == "rod":
        return "rod tip" + (f", 2R = {2 * R_nm:.0f} nm" if R_nm else "")
    return "cone tip" + (f", R = {R_nm:.0f} nm" if R_nm else "") + (f", {half_deg:g}°" if half_deg else "")


def pump_specs() -> list[FieldSpec]:
    """FDTDConfig fields that run_fdtd_pump.main can pass through, in convenient units.

    dx goes through dx_nm; x_half, z_min, z_max, z_source, gaas and tip are fixed by the script
    (passing them raises a duplicate-keyword error); eps_inf_tip and snapshot_every do nothing.
    """
    B, G, T, N = "Beam", "GaAs (820 nm)", "Tip (PtIr, Drude model)", "Grid and numerics"     # no "&": QGroupBox mnemonic
    return [
        FieldSpec("dx_nm", 15.0, "float", "Cell size dx", "nm",
                  "Yee cell edge. Runtime ∝ dx⁻³. The tip-height series always uses 15 nm, so its reference field "
                  "must be computed at 15 nm too.", N, minimum=2, maximum=100),
        FieldSpec("courant", 0.5, "float", "Courant number", "", "dt = courant·dx/c. The 2-D stability limit is 1/√2 ≈ 0.707.",
                  N, advanced=True, minimum=0.05, maximum=0.7),
        FieldSpec("t_end_fs", 0.0, "float", "Simulated time t_end", "fs",
                  "0 = automatic (pulse plus twice the propagation time, ≈ 535 fs by default).", N, advanced=True, minimum=0),
        FieldSpec("n_pml", 20, "int", "PML layers", "", "CPML absorbing cells on each side.", N, advanced=True,
                  minimum=4, maximum=80),
        FieldSpec("pml_m", 3, "int", "PML grading order m", "", "Polynomial order of the CPML conductivity profile.", N,
                  advanced=True, minimum=1, maximum=6),
        FieldSpec("theta_deg", 45.0, "float", "Incidence angle θ", "deg",
                  "From the surface normal; the beam travels toward +x.", B, minimum=0, maximum=80),
        FieldSpec("w_um", 3.0, "float", "Spot radius w (1/e²)", "µm", "1/e² intensity radius projected on the surface.", B,
                  minimum=0.05),
        FieldSpec("x_beam_um", 0.0, "float", "Beam position X", "µm",
                  "x where the beam axis meets the surface (the tip sits at tip_x).", B),
        FieldSpec("wavelength_nm", 820.0, "float", "Wavelength λ", "nm",
                  "FDTDConfig.wavelength is in metres; converted automatically. n, κ and the tip's Drude parameters "
                  "are 820 nm values and do not follow λ.", B, minimum=100, maximum=5000),
        FieldSpec("fwhm_fs", 100.0, "float", "Pulse width (FWHM)", "fs",
                  "Intensity full width at half maximum. Longer pulses need a longer simulated time.", B, minimum=5,
                  maximum=1000),
        FieldSpec("n_gaas", 3.67, "float", "Refractive index n", "", "Real part of the GaAs index at the pump wavelength.",
                  G, minimum=1),
        FieldSpec("kappa_gaas", 0.078, "float", "Extinction coefficient κ", "",
                  "α = 4πκ/λ; 0.078 gives 1.2×10⁴ cm⁻¹ at 820 nm.", G, minimum=0),
        shape_spec(T),
        FieldSpec("tip_radius_nm", 40.0, "float", "Apex radius R", "nm",
                  "FDTDConfig.tip_radius (metres), converted automatically. The straight rod is 2R wide.", T, minimum=1),
        FieldSpec("tip_half_angle_deg", 15.0, "float", "Cone half-angle", "deg",
                  "Cone only (unused for the straight rod). In the 2-D model the tip is a wedge, infinitely long in y, "
                  "so its shadow is an upper bound.", T, minimum=1, maximum=60),
        FieldSpec("tip_gap_nm", 20.0, "float", "Tip–surface gap", "nm",
                  "Distance from the apex to the GaAs surface (FDTDConfig.tip_gap, metres).", T, minimum=0),
        FieldSpec("tip_x_um", 0.0, "float", "Tip position x", "µm", "FDTDConfig.tip_x (metres), converted automatically.",
                  T, advanced=True),
        FieldSpec("omega_p_tip", 1.86e16, "float", "Plasma frequency ω_p", "rad/s",
                  "Single-frequency Drude fit (Pt at 820 nm: ε ≈ −17.6 + 29.6i).", T, advanced=True, minimum=0),
        FieldSpec("gamma_tip", 3.66e15, "float", "Damping γ", "rad/s", "Drude damping rate.", T, advanced=True, minimum=0),
    ]


def pump_kwargs(ov: dict) -> dict:
    """Form values -> run_fdtd_pump.main keyword arguments (FDTDConfig units; dx stays dx_nm).

    tip_shape is always passed (the app default, the rod, is not FDTDConfig's default); the half-angle is
    dropped for the rod, which does not use it."""
    kw = {}
    for k, v in ov.items():
        if k == "t_end_fs":
            if v and v > 0:
                kw[k] = float(v)
        elif k in SCALE:
            name, f = SCALE[k]
            kw[name] = float(f"{float(v) * f:.12g}")       # 3e-08, not 3.0000000000000004e-08
        elif k == "tip_shape":
            kw[k] = SHAPES.get(v, v)
        else:
            kw[k] = v
    kw.setdefault("tip_shape", SHAPES[ROD])
    if kw["tip_shape"] == "rod":
        kw.pop("tip_half_angle_deg", None)
    return kw


def device_cfg(values: dict) -> dict:
    """Form values -> FDTDConfig fields for draw_fdtd_device (lengths in metres, w_um / x_beam_um in µm)."""
    cfg = pump_kwargs(values)
    cfg["dx"] = float(f"{float(cfg.pop('dx_nm', 15.0)) * 1e-9:.12g}")
    return cfg


def kwargs_cfg(kw: dict) -> dict:
    """The FDTDConfig dict of a pump run from its run_fdtd_pump / parallel.fdtd_pump kwargs (gui_run.json).
    Without tip_shape it is FDTDConfig's default, the cone, which is what such runs used."""
    from ltemsim.optics_fdtd import FDTDConfig
    c = {**FDTDConfig().__dict__, **BOX}
    c.update({k: v for k, v in kw.items() if k in c})
    c["dx"] = float(kw.get("dx_nm", 15.0)) * 1e-9
    return c


def form_kwarg(k: str, v) -> str:
    """A run_fdtd_pump kwarg as shown to the user: in the form's units (tip_gap=3e-08 -> tip_gap_nm=30)."""
    if k in UNSCALE and isinstance(v, (int, float)):
        k, v = UNSCALE[k][0], float(f"{v / UNSCALE[k][1]:.10g}")
    return f"{k}={format_float(v) if isinstance(v, float) else v}"


def cfg_to_form(cfg: dict) -> dict:
    """FDTDConfig dict (from an npz, or a run's kwargs with dx_nm) -> form values. A missing tip_shape means the
    cone: every run before the shape choice existed, the repository result included, used it."""
    r = lambda v: float(f"{v:.6g}")
    out = {}
    for spec in pump_specs():
        n = spec.name
        if n == "dx_nm" and ("dx" in cfg or "dx_nm" in cfg):
            out[n] = r(cfg["dx"] * 1e9) if "dx" in cfg else r(float(cfg["dx_nm"]))
        elif n == "tip_shape":
            out[n] = SHAPE_LABEL[shape_of(cfg)]
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


def pump_total(values: dict, workers: int = 3) -> float:
    """Wall time of the three FDTD runs (vacuum, no tip, with tip) on `workers` cores."""
    return estimate_s(fdtd_cost(**values)[3] + RUN_OVERHEAD, 3, workers) + JOB_OVERHEAD


def heights_total(n: int, workers: int = 1) -> float:
    """Wall time of n tip heights (one full 15 nm FDTD run each) on `workers` cores."""
    return estimate_s(fdtd_cost()[3] + RUN_OVERHEAD, n, workers) + (JOB_OVERHEAD if n else 0.0)


def speedup_text(n: int, unit: str, workers: int, total: float, serial: float) -> str:
    """'3 runs on 3 cores ≈ 1.4 min (≈ 3.8 min on one core)'."""
    k = max(1, min(workers, n))
    s = f"{n} {unit if n != 1 else unit.rstrip('s')} on {k} core{'s' if k > 1 else ''} ≈ {fmt_dur(total)}"
    return s + (f" (≈ {fmt_dur(serial)} on one core)" if k > 1 else "")


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
            return "missing or empty arrays: " + ", ".join(bad)
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
    """('ok' | 'warn' | 'err', message) for a reference notip.npz of run_fdtd_heights."""
    from ltemsim.optics_fdtd import FDTDConfig
    p = Path(p)
    if not p.exists():
        return "err", (f"File not found: {paths.rel(p)}. You can queue a default pump FDTD first to create "
                       "the reference field.")
    if p.is_dir():
        return "err", f"{paths.rel(p)} is a folder — choose the notip.npz inside it."
    cfg = npz_cfg(p)
    if cfg is None:
        return "err", "Cannot read this file's cfg (not an npz written by run_fdtd_pump.py, or the file is incomplete)."
    if not cfg.get("gaas", True):
        return "err", ("This is the vacuum reference (vacuum.npz, gaas = False): there is no GaAs absorption, "
                       "Q_notip ≡ 0 and the ratio is meaningless. Choose notip.npz.")
    if cfg.get("tip"):
        return "err", ("This is a with-tip result (tip.npz). The Tip-position chain uses ratio_h*.npz as "
                       "Q_tip/Q_notip, so the reference must be the no-tip notip.npz.")
    if abs(cfg.get("dx", 0) - 15e-9) > 1e-12 or any(abs(cfg.get(k, 0) - v) > 1e-12 for k, v in BOX.items()):
        return "err", (f"Grid mismatch: dx = {cfg.get('dx', 0) * 1e9:.1f} nm. The height series always uses the "
                       "default 15 nm grid and domain; with this reference it would fail on mismatched array shapes.")
    d0 = FDTDConfig()
    diffs = []
    for k, unit, f in (("theta_deg", "°", 1), ("w_um", " µm", 1), ("x_beam_um", " µm", 1), ("wavelength", " nm", 1e9),
                       ("fwhm_fs", " fs", 1), ("n_gaas", "", 1), ("kappa_gaas", "", 1), ("courant", "", 1)):
        v, v0 = cfg.get(k), getattr(d0, k)
        if v is not None and abs(v - v0) > 1e-9 * max(1.0, abs(v0)):
            diffs.append(f"{k} = {format_float(v * f)}{unit} (default {format_float(v0 * f)}{unit})")
    if cfg.get("t_end_fs"):
        diffs.append(f"t_end = {format_float(cfg['t_end_fs'])} fs (default: automatic)")
    if diffs:
        return "warn", ("⚠ Beam/material parameters differ from the height series (fixed to the FDTDConfig "
                        "defaults): " + "; ".join(diffs) + ". The ratio would mix two different beams.")
    return "ok", f"✓ Default 15 nm grid · θ = {cfg['theta_deg']:g}° · w = {cfg['w_um']:g} µm · matches the height series"


def tip_outline(x0, gap, R, half_deg, z_top=20.0, shape=LEGACY_SHAPE):
    """Outline (µm) of the 2-D tip, as in optics_fdtd.in_tip: apex circle + wedge (cone) or + straight shank of
    width 2R (rod)."""
    zc = gap + R
    if shape == "rod":
        phi = np.linspace(math.pi, 2 * math.pi, 64)
        return (np.concatenate([[x0 - R], x0 + R * np.cos(phi), [x0 + R]]),
                np.concatenate([[z_top], zc + R * np.sin(phi), [z_top]]))
    th = math.radians(half_deg)
    phi = np.linspace(math.pi - th, 2 * math.pi + th, 64)
    z_t = zc + R * math.sin(th)
    half_top = R * math.cos(th) + (z_top - z_t) * math.tan(th)
    xs = np.concatenate([[x0 - half_top], x0 + R * np.cos(phi), [x0 + half_top]])
    zs = np.concatenate([[z_top], zc + R * np.sin(phi), [z_top]])
    return xs, zs


def tip_mask(cfg: dict, z_top: float):
    """(xs, zs, f[z, x]) in µm for FieldView's outlines: f = 0.5 − (signed distance to the tip of FDTDConfig `cfg`,
    the shape of optics_fdtd.in_tip), so its 0.5 contour is the exact, smooth geometry edge (a boolean mask would
    give a staircase). The grid is fine around the apex and coarse along the straight shank."""
    R, gap, xt = cfg["tip_radius"] * 1e6, cfg["tip_gap"] * 1e6, cfg["tip_x"] * 1e6
    rod = shape_of(cfg) == "rod"
    th = 0.0 if rod else math.radians(cfg["tip_half_angle_deg"])
    zc = gap + R
    z_t = zc + R * math.sin(th)                         # the shank is tangent to the apex sphere here
    h = max(min(R / 10, cfg.get("dx", 15e-9) * 1e6 / 2), 5e-4)
    half = R * math.cos(th) + max(z_top - z_t, 0.0) * math.tan(th) + 4 * h
    hx = max(h, 2 * half / 800)
    xs = np.arange(xt - half, xt + half + hx / 2, hx)
    z_fine = max(min(zc + 6 * R, z_top), gap - 3 * h)
    zs = np.concatenate([np.arange(gap - 4 * h, z_fine, h), np.linspace(z_fine, max(z_top, z_fine + h), 80)])
    X, Z = np.meshgrid(xs, zs)
    d_sphere = np.hypot(X - xt, Z - zc) - R
    # rod (th = 0): the slab |x − x_t| ≤ R above the sphere centre; cone: the wedge tangent to the sphere
    d_shank = np.maximum((np.abs(X - xt) - R * math.cos(th) - (Z - z_t) * math.tan(th)) * math.cos(th), z_t - Z)
    return xs, zs, 0.5 - np.minimum(d_sphere, d_shank)


def _edges(c) -> np.ndarray:
    """Cell edges from cell centres (midpoints, the outer edges mirrored)."""
    c = np.asarray(c, float)
    if c.size < 2:
        c0 = float(c[0]) if c.size else 0.0
        return np.array([c0 - 0.005, c0 + 0.005])
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[2 * c[0] - mid[0]], mid, [2 * c[-1] - mid[-1]]])


def tip_text(cfg: dict) -> str:
    """The tip of an FDTDConfig dict, e.g. 'rod tip, 2R = 80 nm' or 'cone tip, R = 40 nm, 15°'."""
    shape = shape_of(cfg)
    return shape_text(shape, cfg["tip_radius"] * 1e9, cfg.get("tip_half_angle_deg") if shape == "cone" else None)


def cfg_caption(cfg: dict) -> str:
    return (f"θ {cfg['theta_deg']:g}° · w {cfg['w_um']:g} µm · λ {cfg['wavelength'] * 1e9:.0f} nm · {tip_text(cfg)} · "
            f"gap {cfg['tip_gap'] * 1e9:.0f} nm · dx {cfg['dx'] * 1e9:g} nm")


def source_cut(cfg: dict | None) -> float:
    """µm: top of the useful air region, just below the soft source line (its own row carries a near field ~8x
    the propagating beam, and above it there is only the source's backward emission)."""
    c = cfg or {}
    return c.get("z_source", BOX["z_source"]) * 1e6 - max(0.1, 4 * c.get("dx", 15e-9) * 1e6)


def incident_e2(runs: dict) -> tuple[float, str]:
    """(|E0|², run name): the peak incident |E|², i.e. the maximum of the vacuum run's max|E|² (the no-tip run's
    when vacuum.npz is missing) in the air between z = 0.5 µm and the source line.

    run_fdtd_pump.py's figure divides by the no-tip maximum over all z > 0.5 µm, which includes the source row
    itself and so is ~60x the incident |E|² on the default grid.
    """
    for name in ("vacuum", "notip"):
        r = runs.get(name)
        if r is None or "Emax2" not in r:
            continue
        sel = (r["z"] > E0_Z) & (r["z"] < source_cut(r.get("cfg")))
        if sel.any():
            e2 = float(r["Emax2"][:, sel].max())
            if e2 > 0:
                return e2, name
    return 0.0, ""


def field_data(runs: dict, x, z, Qt, Qn, ratio, cfg: dict | None) -> dict | None:
    """Data for the COMSOL-style optical-field view (FieldView scalar mode), or None without field arrays.

    The npz arrays are (Nx, Nz) on cell centres (x, z already in µm here); FieldView takes [z, x] on cell edges.
    |E| is normalised to the peak incident field (incident_e2); the metal tip's interior is left blank.
    """
    base = runs.get("tip") or runs.get("notip") or runs.get("vacuum")
    if base is not None:
        xs, zs = base["x"], base["z"]
    elif x is not None:
        xs, zs = x, z
    else:
        return None
    shape = (len(xs), len(zs))
    ok = lambda a: a is not None and np.shape(a) == shape
    air = zs >= 0
    q = {}
    e0, e0_run = incident_e2(runs)
    if e0 > 0:
        for name, key in (("tip", E_TIP), ("notip", E_NOTIP)):
            r = runs.get(name)
            if r is not None and ok(r["Emax2"]):
                a = np.sqrt(np.maximum(r["Emax2"], 0.0) / e0)
                if ok(r.get("tip_mask")):
                    a[r["tip_mask"]] = np.nan          # inside the PtIr: tiny field, would swamp the log scale
                q[key] = (a.T, "", False)

    def gaas_only(a):
        a = np.array(a, float).T                 # [z, x]
        a[air, :] = np.nan                       # Q and the ratio only live in the GaAs
        return a

    if Qt is None and "tip" in runs:                 # interrupted run: no ratio.npz, but the run files hold Q
        Qt = runs["tip"]["Q"]
    if Qn is None and "notip" in runs:
        Qn = runs["notip"]["Q"]
    qmax = float(np.max(Qn)) if ok(Qn) else float(np.max(Qt)) if ok(Qt) else 0.0
    if qmax > 0:
        if ok(Qt):
            q[Q_TIP] = (gaas_only(Qt / qmax), "", False)
        if ok(Qn):
            q[Q_NOTIP] = (gaas_only(Qn / qmax), "", False)
    if ok(ratio):
        q[RATIO] = (gaas_only(ratio), "", False)
    if not q:
        return None
    xe, ze = _edges(xs), _edges(zs)
    xt = cfg["tip_x"] * 1e6 if cfg else 0.0
    za = cfg["tip_gap"] * 1e6 if cfg else 0.0
    top = min(float(ze[-1]), source_cut(cfg or base and base.get("cfg")))
    outline = [tip_mask(cfg, float(ze[-1]) + 0.2)] if cfg and ("tip" in runs or ok(Qt)) else []
    return dict(xe=xe, ze=ze, quantities=q, log_default=True, e0=math.sqrt(e0) if e0 else None, e0_run=e0_run,
                outlines=outline, outlines_by={k: (outline if k in WITH_TIP else []) for k in q},
                views={"Whole domain": ((float(xe[0]), float(xe[-1])), (float(ze[0]), top)),
                       "Near tip (±1 µm)": ((xt - 1.0, xt + 1.0), (za - 1.0, za + 1.0)),
                       "Apex (±0.2 µm)": ((xt - 0.2, xt + 0.2), (za - 0.2, za + 0.2))},
                title=cfg_caption(cfg) if cfg else "")


def _log10(a, floor=1e-30):
    return np.log10(np.maximum(a, floor))


def _air_nan(a, z):
    a = np.array(a, float)
    a[:, z >= 0] = np.nan
    return a


# ----------------------------------------------------------------------------- loading (worker threads)
def load_pump(d: Path) -> dict:
    """Everything the pump-results views need, from whatever files the folder has."""
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
                                         cfg=json.loads(str(z["cfg"])),
                                         tip_mask=z["tip"] if name == "tip" and "tip" in z.files else None)
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
    gkw = (res["gui"] or {}).get("kwargs") if isinstance(res["gui"], dict) else None
    if tcfg is None and ratio is not None and isinstance(gkw, dict):
        tcfg = kwargs_cfg(gkw)          # no readable run npz: the tip from the run's arguments (outline, layers)
    if tcfg:
        tip = dict(x0=tcfg["tip_x"] * 1e6, gap=tcfg["tip_gap"] * 1e6, R=tcfg["tip_radius"] * 1e6,
                   half=tcfg["tip_half_angle_deg"], shape=shape_of(tcfg))
    res["field"] = field_data(runs, x, z, Qt, Qn, ratio, tcfg)

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
            L_.append(dict(key="ratio", group="ratio", label="log₁₀(Q_tip / Q_notip) — enhancement / shadow", x=x, z=z,
                           data=_air_nan(np.log10(np.maximum(ratio, 1e-30)), z), cmap="RdBu_r", v=(-1.0, 1.0),
                           xlim=(-2.0, 2.0), zlim=(-1.5, 0.3), title="log10(Q_tip / Q_notip): enhancement (red), shadow (blue)",
                           cbar="log10 ratio", tip=tip))
    if Qn is not None:
        qmax = Qn.max()
        L_.append(dict(key="Qtip", group="Q", label="log₁₀ Q with tip (/ max Q_notip)", x=x, z=z,
                       data=_air_nan(_log10(Qt / qmax), z), cmap="viridis", v=(-4.0, 0.5), xlim=(-4.0, 4.0),
                       zlim=(-2.0, 0.3), title="log10 absorbed energy density, with tip (/ max Q_notip)", cbar="log10 Q", tip=tip))
        L_.append(dict(key="Qnotip", group="Q", label="log₁₀ Q without tip (/ max Q_notip)", x=x, z=z,
                       data=_air_nan(_log10(Qn / qmax), z), cmap="viridis", v=(-4.0, 0.5), xlim=(-4.0, 4.0),
                       zlim=(-2.0, 0.3), title="log10 absorbed energy density, no tip (/ max Q_notip)", cbar="log10 Q", tip=None))
    inc = incident_e2(runs)[0]
    if inc > 0:
        for name, en in (("tip", "with tip"), ("notip", "no tip"), ("vacuum", "vacuum")):
            if name in runs:
                rr = runs[name]
                L_.append(dict(key=f"E_{name}", group="E", label=f"log₁₀ max|E|² / |E0|² ({en})", x=rr["x"], z=rr["z"],
                               data=_log10(rr["Emax2"] / inc), cmap="magma", v=(-3.0, 1.5),
                               xlim=(float(rr["x"][0]), float(rr["x"][-1])), zlim=(float(rr["z"][0]), float(rr["z"][-1])),
                               title=f"log10 max|E|^2 / incident |E0|^2 ({en})", cbar="log10 |E|^2 / |E0|^2",
                               tip=tip if name == "tip" else None))
    for r in runs.values():           # the raw arrays are no longer needed
        for k in ("Q", "Emax2", "tip_mask"):
            r.pop(k, None)
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


def heights_shapes(d: Path) -> set[str]:
    """Tip shapes the height runs in folder `d` used, from its gui_run*.json (ratio_h*.npz do not record the tip).

    Folders without a GUI record (the repository result) and runs of scripts/run_fdtd_heights.py used the cone;
    a resumed folder can hold both shapes."""
    shapes = set()
    for p in Path(d).glob("gui_run*.json"):
        kw = read_dict(p).get("kwargs") or {}
        if isinstance(kw, dict) and "heights" in kw:
            shapes.add(shape_of(kw))
    return shapes or {LEGACY_SHAPE}


def heights_shape(d: Path) -> str:
    """'rod' | 'cone' | 'mixed' for a heights folder."""
    s = heights_shapes(d)
    return next(iter(s)) if len(s) == 1 else "mixed"


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
    shape = heights_shape(d)
    res = {"dir": d, "files": files, "summary": summ, "ref": ref, "ref_summary": ref_summary, "layers": [],
           "from_npz": [], "bad": {}, "norm": norm, "gui": read_json(d / "gui_run.json"), "shape": shape}
    # the pump run's own with-tip point, if its tip is the series' tip (R = 40 nm and the same shape; a cone also
    # needs the default 15°). npz without tip_shape — the repository result included — used the cone.
    tcfg = npz_cfg(ref.parent / "tip.npz")
    if (norm["basis"] == "vacuum" and tcfg and isinstance(ref_summary.get("absorbed_fraction_tip"), (int, float))
            and abs(tcfg.get("tip_radius", 0) - 40e-9) < 1e-12 and shape_of(tcfg) == shape
            and (shape == "rod" or tcfg.get("tip_half_angle_deg") == 15.0)):
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
                                  tip=None if shape == "mixed" else dict(x0=0.0, gap=h / 1000, R=0.04, half=15.0,
                                                                         shape=shape)))
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


class _Tab(QWidget):
    """Tab page with a modest preferred height. The app hosts pages in a resizable scroll area that, because the
    page layout has height-for-width, makes the page as tall as its preferred height: a FigureCanvas asks for its
    whole figsize (~500 px), which would push a 700 px window into scrolling."""

    def sizeHint(self):
        return QSize(super().sizeHint().width(), max(self.minimumSizeHint().height(), 320))


# ----------------------------------------------------------------------------- COMSOL-style optical field
class OpticalFieldView(FieldView):
    """FieldView for the FDTD maps: the tip outline only on the with-tip quantities, 1:1 aspect by default and the
    selected quantity / view kept when another result is loaded. Scalar maps only: the vector-overlay control
    (disabled by FieldView for scalar data) is hidden."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.equal.setChecked(True)
        self.overlay.hide()
        for w in self.findChildren(QLabel):
            if w.text() == "Overlay":
                w.hide()
        self.qty.setToolTip("Quantity shown as the color surface")
        self.view.setToolTip("Zoom presets; the views end just below the source plane at z = 2.2 µm")
        self.log.setToolTip("Logarithmic color scale (on by default: the field and the absorption span decades)")
        self.equal.setToolTip("Same length scale on x and z (to scale, as in COMSOL)")

    def set_fields(self, data, message=None):
        keep = (self.qty.currentText(), self.view.currentText()) if self.data else None
        super().set_fields(data, message)
        if not data:
            return
        if keep and (keep[0] != self.qty.currentText() or keep[1] != self.view.currentText()):
            self._busy = True
            try:
                for combo, text in zip((self.qty, self.view), keep):
                    j = combo.findText(text)
                    if j >= 0:
                        combo.setCurrentIndex(j)
            finally:
                self._busy = False
            self._redraw()

    def _redraw(self, *_):
        d = self.data
        if d and "outlines_by" in d:
            d["outlines"] = d["outlines_by"].get(self.qty.currentText(), ())
        super()._redraw()


# ----------------------------------------------------------------------------- colour-map view
class MapView(QWidget):
    """Colour map of one of several (x, z) layers with layer / colormap / range controls."""

    def __init__(self, empty_text="No data", parent=None):
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
        self.tip = QCheckBox("Tip outline")
        self.tip.setChecked(True)
        self.tip.toggled.connect(self._schedule)
        self.equal = QCheckBox("1:1 aspect")
        self.equal.setChecked(True)
        self.equal.setToolTip("Same length scale on x and z")
        self.equal.toggled.connect(self._schedule)
        r1.addWidget(_group(label("Quantity", "muted"), self.layer))
        r1.addWidget(_group(label("Colormap", "muted"), self.cmap))
        r1.addWidget(_group(self.tip, self.equal, spacing=10))
        rb = button("Reset view", self._reset_view, "flat", "Restore this quantity's default ranges and colormap")
        r1.addWidget(rb)
        lay.addLayout(r1)
        r2 = FlowLayout(hspacing=14)
        self.spins = {}
        for key, text, lo, hi, step in (("x", "x", -20, 20, 0.25), ("z", "z", -20, 20, 0.25),
                                        ("v", "Color range", -40, 40, 0.25)):
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
                xs, zs = tip_outline(tp["x0"], tp["gap"], tp["R"], tp["half"], shape=tp.get("shape", LEGACY_SHAPE))
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
    title = "Pump FDTD (optics)"
    subtitle = ("2-D FDTD of the 820 nm, 100 fs p-polarised pump on GaAs: vacuum reference, no tip, and a PtIr (Drude) "
                "tip. Gives the optical field |E|, the absorbed energy density Q(x, z) and the tip's enhancement / "
                f"shadow Q_tip/Q_notip. The three runs execute side by side on 3 cores: default grid ≈ "
                f"{fmt_dur(pump_total({}))}.")

    def __init__(self):
        super().__init__()
        self.job = None
        self._runs2 = None          # heights to compute, last passed to cores2.set_runs
        self.pump = None            # loaded pump result dict
        self.heights = None         # loaded heights result dict
        self._pump_token = self._heights_token = 0
        self._levels = {}           # label -> colour level, re-applied on theme change
        self._loaded_once = False
        self._dev_cfg = None        # FDTDConfig fields of the last device drawing
        self._dev_dirty = True

        b = button("Docs", lambda: nav.navigate("docs", path=str(paths.DOCS / "pulse_chain.md")), "flat",
                   "docs/pulse_chain.md — stage 1: the pump field")
        self.header.add_action(b)
        self.run_btn = button("  Run pump FDTD", self.run, "primary")
        self.header.add_action(self.run_btn)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._build_left())
        split.addWidget(self._build_right())
        split.setSizes([440, 900])
        self.body.addWidget(split, 1)

        self._dev_timer = QTimer(self)
        self._dev_timer.setSingleShot(True)
        self._dev_timer.setInterval(300)
        self._dev_timer.timeout.connect(self._draw_device)
        self.form.changed.connect(self._schedule_device)
        self.hform.changed.connect(self._schedule_device)
        self.tabs.currentChanged.connect(self._on_tab)

        self._sync_shape()
        self._update_pump_estimate()
        self._refresh_refs()
        self._update_heights_info()
        self._retheme()
        self._schedule_device()
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
        self.mode_bar.addTab("Pump field")
        self.mode_bar.addTab("Tip-height series")
        self.mode_bar.setTabToolTip(0, "Vacuum reference, no tip and with tip, with the parameters below")
        self.mode_bar.setTabToolTip(1, "One with-tip run per apex height (fixed default grid and beam)")
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
        self.form.changed.connect(self._sync_shape)
        self.form.changed.connect(self._update_pump_estimate)
        self._compact_form(self.form)
        for sig in (self.form.adv.toggled, self.form.search.textChanged, self.form.changed):    # after ParamForm's filter
            sig.connect(self._align_labels)
        self._align_labels()
        l1.addWidget(self.form, 1)
        self.cores1 = CoresSpin()
        self.cores1.set_runs(3)
        self.cores1.changed.connect(self._update_pump_estimate)
        self.est1 = self._estimate_tile(self.cores1, "The three FDTD runs (vacuum, no tip, with tip) are independent; "
                                                     "each one runs on one core.")
        l1.addWidget(self.est1)
        self.warn1 = label("", wrap=True)
        self.warn1.setVisible(False)
        l1.addWidget(self.warn1)
        orow = QHBoxLayout()
        orow.addWidget(QLabel("Output folder"))
        self.out1 = QLineEdit()
        self.out1.setPlaceholderText("Automatic: results/gui_runs/fdtd_<time>")
        orow.addWidget(self.out1, 1)
        b = QPushButton("Choose…")
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
        self.hform = ParamForm([shape_spec("Tip heights"),
                                FieldSpec("heights", HEIGHTS_DEFAULT, "tuple", "Heights h", "nm",
                                          "Apex–surface distances, comma separated. Files are named by int(h), "
                                          "so 12.5 and 12 collide.", "Tip heights")],
                               toolbar=False, scroll=False)
        self.hform.changed.connect(self._update_heights_info)
        self._compact_form(self.hform)
        self.hform.rows["heights"].editor.setCursorPosition(0)
        hrow = self.hform.rows["heights"]
        gg = hrow.group.layout()                 # the form's own group box: add presets + note below the field
        gg.setColumnMinimumWidth(0, 90)
        presets = QWidget()
        prow = FlowLayout(presets, hspacing=4, vspacing=2)          # wraps in the narrow column of small windows
        prow.addWidget(label("Presets", "muted"))
        for text, hs in (("Default (10)", HEIGHTS_DEFAULT), ("Quick (3)", (10, 80, 370)),
                         ("Dense near field", (5, 10, 15, 20, 25, 35, 45, 60, 80))):
            pb = button(text, lambda _=False, v=hs: self._set_heights(v), "flat")
            prow.addWidget(pb)
        r0 = gg.rowCount()
        gg.addWidget(presets, r0, 0, 1, 3)
        gg.addWidget(label("One full FDTD run (with tip) per height, one height per core. The script fixes dx = 15 nm, "
                           "θ = 45°, w = 3 µm, R = 40 nm and all other FDTDConfig defaults except the tip shape chosen "
                           "here; the “Pump field” form does not apply to the series.",
                           "muted", wrap=True), r0 + 1, 0, 1, 3)
        l2.addWidget(self.hform)

        g = QGroupBox("Reference field (no-tip notip.npz)")
        gl = QVBoxLayout(g)
        rrow = QHBoxLayout()
        self.ref = QComboBox()
        self.ref.setMinimumWidth(150)
        self.ref.currentIndexChanged.connect(self._update_heights_info)
        rrow.addWidget(self.ref, 1)
        b = QPushButton("Choose…")
        b.clicked.connect(self._pick_ref)
        rrow.addWidget(b)
        gl.addLayout(rrow)
        self.ref_info = label("", wrap=True)
        gl.addWidget(self.ref_info)
        gl.addWidget(label("Ratio = Q_tip(h) / Q_notip; absorbed fractions are normalised by the reference's incident "
                           "energy. The reference must come from the default 15 nm grid.", "muted", wrap=True))
        l2.addWidget(g)

        g = QGroupBox("Output")
        gl = QVBoxLayout(g)
        orow = QHBoxLayout()
        self.out2 = QLineEdit()
        self.out2.setPlaceholderText("Automatic: results/gui_runs/fdtd_heights_<time>")
        self.out2.textChanged.connect(self._update_heights_info)
        orow.addWidget(self.out2, 1)
        b = QPushButton("Choose…")
        b.clicked.connect(lambda: self._pick_dir(self.out2))
        orow.addWidget(b)
        gl.addLayout(orow)
        self.out2_info = label("", "muted", wrap=True)
        gl.addWidget(self.out2_info)
        gl.addWidget(label("summary.json is rebuilt from every ratio_h*.npz in the folder when the series ends, so "
                           "heights from earlier runs are kept.", "muted", wrap=True))
        l2.addWidget(g)
        self.cores2 = CoresSpin()
        self.cores2.changed.connect(self._update_heights_info)
        self.est2 = self._estimate_tile(self.cores2, "One tip height per core; each height is a full FDTD run on one "
                                                     "core.")
        l2.addWidget(self.est2)
        l2.addStretch(1)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setWidget(m2)
        self.mode_stack.addWidget(area)
        left.setMinimumWidth(330)
        return left

    def _build_right(self):
        self.tabs = QTabWidget()
        # ---------------- device (model geometry of the parameters that will be run)
        self.tab_device = _Tab()
        dl = QVBoxLayout(self.tab_device)
        dl.setContentsMargins(12, 12, 12, 8)
        dl.setSpacing(6)
        self.dev_note = label("", "muted", wrap=True)
        dl.addWidget(self.dev_note)
        self.dev_plot = PlotWidget(figsize=(8, 5))
        dl.addWidget(self.dev_plot, 1)
        self.dev_info = label("", "muted", wrap=True)
        dl.addWidget(self.dev_info)
        self.tabs.addTab(self.tab_device, "Device")
        self.tabs.setTabToolTip(0, "The FDTD model to scale: GaAs, air, CPML, source, pump beam and the tip")

        # ---------------- run
        self.tab_run = QWidget()
        rl = QVBoxLayout(self.tab_run)
        rl.setContentsMargins(12, 12, 12, 12)
        rl.setSpacing(8)
        self.run_info = label("No job submitted yet. Choose a mode and set the parameters on the left, then press the run "
                              "button at the top right. Jobs run in a background process; the Job center lists all of them.",
                              "muted", wrap=True)
        rl.addWidget(self.run_info)
        self.panel = RunPanel()
        self.panel.job_finished.connect(self._on_finished)
        rl.addWidget(self.panel, 1)
        self.tabs.addTab(self.tab_run, "Run")

        # ---------------- optical field (COMSOL style)
        self.tab_field = _Tab()
        fl = QVBoxLayout(self.tab_field)
        fl.setContentsMargins(12, 12, 12, 8)
        fl.setSpacing(6)
        top = FlowLayout(lead=True)
        self.f_title = ElidedLabel("No result loaded", "cardTitle")
        top.addWidget(self.f_title)
        self.btn_f_folder = button("Show in Finder", lambda: self.pump and reveal(self.pump["dir"]), "flat")
        top.addWidget(self.btn_f_folder)
        fl.addLayout(top)
        self.f_note = ElidedLabel(FIELD_NOTE, "muted")
        self.f_note.setToolTip(FIELD_HELP)
        fl.addWidget(self.f_note)
        self.field = OpticalFieldView()
        self.field.set_fields(None, "Loading the repository result…")
        fl.addWidget(self.field, 1)
        self.tabs.addTab(self.tab_field, "Optical field")
        self.tabs.setTabToolTip(2, "COMSOL-style colour maps of the optical field, the absorbed energy and the tip enhancement")

        # ---------------- pump results
        pt = QWidget()
        pl = QVBoxLayout(pt)
        pl.setContentsMargins(12, 12, 12, 12)
        pl.setSpacing(8)
        top = FlowLayout(lead=True)              # buttons move under the title in narrow windows
        self.p_title = ElidedLabel("No result loaded", "cardTitle")
        top.addWidget(self.p_title)
        self.btn_params = button("Load parameters", self._load_params, "flat",
                                 "Copy this result's FDTDConfig into the “Pump field” form")
        self.btn_as_ref = button("Use as height reference", self._use_as_ref, "flat",
                                 "Use this result's notip.npz as the tip-height series reference")
        self.btn_p_heights = button("Height series", lambda: self.pump and self.show_heights(self.pump["dir"] / "heights"),
                                    "flat", "Show the tip-height series in this folder's heights/")
        self.btn_p_folder = button("Show in Finder", lambda: self.pump and reveal(self.pump["dir"]), "flat")
        for b in (self.btn_params, self.btn_as_ref, self.btn_p_heights, self.btn_p_folder):
            top.addWidget(b)
        pl.addLayout(top)
        self.p_cfg = ElidedLabel("", "muted")
        pl.addWidget(self.p_cfg)
        grid = TileLayout()
        self.m_notip = MetricTile("Absorbed fraction · no tip", "—")
        self.m_tip = MetricTile("Absorbed fraction · with tip", "—")
        self.m_apex = MetricTile("Q_tip/Q_notip under the apex", "—")
        self.m_col = MetricTile("Depth-integrated ratio min – max", "—")
        for m in (self.m_notip, self.m_tip, self.m_apex, self.m_col):
            grid.addWidget(m)
        pl.addLayout(grid)
        self.p_tabs = QTabWidget()
        self.p_tabs.setMinimumHeight(330)
        self.p_img = ImageView()
        self.p_img.setMinimumWidth(240)          # its toolbar row alone asks for ~505 px; do not widen the page
        self.p_tabs.addTab(self.p_img, "Overview")
        self.p_tabs.setTabToolTip(0, "fdtd_pump.png written by the script")
        self.p_map = MapView("This folder has no .npz data files, so the field maps cannot be drawn.")
        mw = QWidget()
        ml = QVBoxLayout(mw)
        ml.setContentsMargins(8, 8, 8, 4)
        ml.addWidget(self.p_map)
        self.p_tabs.addTab(mw, "Log₁₀ maps")
        self.p_tabs.setTabToolTip(1, "log₁₀ maps with editable x / z / color ranges (defaults as in fdtd_pump.png, but |E|² "
                                     "is divided by the true incident |E0|², not by the source-plane maximum)")
        self.p_cols = PlotWidget()
        self.p_tabs.addTab(self.p_cols, "Depth-integrated vs x")
        pl.addWidget(self.p_tabs, 1)
        self.tab_pump = _scrolled(pt)
        self.tabs.addTab(self.tab_pump, "Absorption")
        self.tabs.setTabToolTip(3, "Absorbed fractions, the script's figure, log maps and depth-integrated absorption")

        # ---------------- heights results
        ht = QWidget()
        hl = QVBoxLayout(ht)
        hl.setContentsMargins(12, 12, 12, 12)
        hl.setSpacing(8)
        top = FlowLayout(lead=True)
        self.h_title = ElidedLabel("No height series loaded", "cardTitle")
        top.addWidget(self.h_title)
        self.btn_resume = button("Resume in this folder", self._resume_heights, "flat",
                                 "Use this folder as the series output (existing ratio_h*.npz are skipped)")
        self.btn_summary = button("Rebuild summary.json", self._rebuild_summary, "flat",
                                  "Rebuild summary.json from every ratio_h*.npz in the folder (gui_tasks.fdtd_heights_summary)")
        self.btn_h_folder = button("Show in Finder", lambda: self.heights and reveal(self.heights["dir"]), "flat")
        for b in (self.btn_resume, self.btn_summary, self.btn_h_folder):
            top.addWidget(b)
        hl.addLayout(top)
        grid = TileLayout()
        self.h_n = MetricTile("Heights", "—")
        self.h_abs = MetricTile("Absorbed fraction h_min → h_max", "—")
        self.h_apex = MetricTile("Q_tip/Q_notip under the tip", "—")
        self.h_ref = MetricTile("Reference field", "—")
        for m in (self.h_n, self.h_abs, self.h_apex, self.h_ref):
            grid.addWidget(m)
        hl.addLayout(grid)
        self.h_tabs = QTabWidget()
        self.h_tabs.setMinimumHeight(330)
        self.h_plot = PlotWidget()
        self.h_tabs.addTab(self.h_plot, "Versus height")
        self.h_map = MapView()
        mw = QWidget()
        ml = QVBoxLayout(mw)
        ml.setContentsMargins(8, 8, 8, 4)
        ml.addWidget(self.h_map)
        self.h_tabs.addTab(mw, "Ratio map per height")
        hl.addWidget(self.h_tabs, 1)
        self.tab_heights = _scrolled(ht)
        self.tabs.addTab(self.tab_heights, "Heights")
        self.tabs.setTabToolTip(4, "Results of the tip-height series (ratio_h*.npz)")

        # ---------------- history
        self.tab_hist = QWidget()
        hy = QVBoxLayout(self.tab_hist)
        hy.setContentsMargins(12, 12, 12, 12)
        hrow = QHBoxLayout()
        hrow.addWidget(ElidedLabel("Existing FDTD results (double-click to load)", "cardTitle"), 1)
        for text, slot in (("Show in Finder", self._reveal_selected), ("Refresh", self.refresh_history)):
            hrow.addWidget(button(text, slot, "flat"))
        hy.addLayout(hrow)
        self.hist = QListWidget()
        self.hist.setSpacing(3)
        self.hist.itemDoubleClicked.connect(self._open_item)
        hy.addWidget(self.hist, 1)
        self.tabs.addTab(self.tab_hist, "History")
        return self.tabs

    def _retheme(self, *_):
        t = ThemeManager.instance().tokens
        self.run_btn.setIcon(icons.icon("play", t.on_accent))
        for w, level in self._levels.items():
            self._paint(w, level)
        self._sync_shape()

    def _paint(self, w: QLabel, level: str):
        """Colour a message label by level ('ok' | 'warn' | 'err' | 'muted')."""
        self._levels[w] = level
        t = ThemeManager.instance().tokens
        col = {"ok": t.success, "warn": t.warning, "err": t.danger}.get(level, t.muted)
        w.setStyleSheet(f"color: {col};")

    @staticmethod
    def _estimate_tile(spin: CoresSpin, tip: str) -> MetricTile:
        """'Estimated runtime' tile with 'Cores [k] of n' at the right of its title line (saves a form row)."""
        general = spin.spin.toolTip().split(" A single steady-state")[0]     # drop the steady-state remark
        spin.spin.setToolTip(general + "\n" + tip)
        tile = MetricTile("Estimated runtime", "—")
        lay = tile.layout()
        lay.removeWidget(tile.lab)
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(tile.lab, 1, Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(spin, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.insertLayout(0, row)
        return tile

    @staticmethod
    def _compact_form(form: ParamForm):
        """Let the form fit the 330 px left column of a 1120 px window. ParamForm's 110 px editor minimum (and the
        long 'Rod (width 2R)' item) make the content wider than its scroll area, which then clips the
        right-aligned values; the editors still take all the width they get."""
        for r in form.rows.values():
            e = r.editor
            if isinstance(e, QComboBox):
                e.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                e.setMinimumContentsLength(6)
                e.setMinimumWidth(64)
            elif isinstance(e, QLineEdit):
                e.setMinimumWidth(64)

    def _align_labels(self, *_):
        """Label column of the pump form = its widest visible label (ParamForm's fixed 178 px leaves the editors
        ~65 px in a 1120 px window), the same in every group so that the editors stay aligned."""
        rows = [r for r in self.form.rows.values() if not r.label.isHidden()]
        w = max((r.label.sizeHint().width() for r in rows), default=120)
        for g in {r.group for r in self.form.rows.values()}:
            g.layout().setColumnMinimumWidth(0, w)

    def _shape(self, form: ParamForm) -> str:
        """'rod' | 'cone' chosen in `form` (the pump form or the heights form)."""
        return SHAPES.get(form.values().get("tip_shape"), SHAPES[ROD])

    def _sync_shape(self, *_):
        """The cone half-angle means nothing for the straight rod: grey it out."""
        cone = self._shape(self.form) == "cone"
        row = self.form.rows["tip_half_angle_deg"]
        row.editor.setEnabled(cone)
        row.label.setEnabled(cone)
        row.editor.setToolTip("" if cone else "Unused: the straight rod has no half-angle. Choose “Cone” to set it.")
        t = ThemeManager.instance().tokens            # the app style sheet has no :disabled look for these two
        row.editor.setStyleSheet("" if cone else f"color: {t.muted}; background: {t.surface2};")
        row.label.setStyleSheet("" if cone else f"color: {t.muted};")

    # ================================================================== device view
    def _schedule_device(self, *_):
        self._dev_timer.start()

    def _on_tab(self, _i):
        if self.tabs.currentWidget() is self.tab_device and self._dev_dirty:
            self._draw_device()

    def device_config(self) -> tuple[dict, str, dict]:
        """(FDTDConfig fields to draw, caption, form values for the grid statistics) for the current mode."""
        if self.mode_bar.currentIndex() == 1:
            hs = sorted(h for h in self._heights() if h > 0)
            h = hs[0] if hs else 20.0
            cfg = dict(dx=15e-9, tip_gap=float(f"{h * 1e-9:.12g}"), tip_shape=self._shape(self.hform))
            more = f" (lowest of {len(hs)}, up to {hs[-1]:g} nm)" if len(hs) > 1 else ""
            note = ("Tip-height series: one run per height with the fixed 15 nm grid and the default beam, GaAs and tip "
                    f"(FDTDConfig); the apex is drawn at h = {h:g} nm{more}.")
            return cfg, note, {}
        v = self.form.values()
        bad = self.form.invalid_fields()
        note = ("The model to scale, from the current form values — it follows your edits. Hatched border: CPML absorbing "
                "layers; dashed red line: source plane.")
        if bad:
            note += f" Invalid values are drawn at their defaults: {', '.join(bad)}."
        return device_cfg(v), note, v

    def _draw_device(self):
        if self.tabs.currentWidget() is not self.tab_device:
            self._dev_dirty = True            # drawn when the tab is opened
            return
        self._dev_dirty = False
        cfg, note, v = self.device_config()
        self._dev_cfg = cfg
        self.dev_note.setText(note)
        try:
            nx, nz, steps, _ = fdtd_cost(**{k: v[k] for k in ("dx_nm", "courant", "fwhm_fs", "t_end_fs", "n_pml") if k in v})
            dx = cfg["dx"] * 1e9
            n_gaas = v.get("n_gaas", 3.67)
            lam = cfg.get("wavelength", 820e-9) * 1e9
            R = cfg.get("tip_radius", 40e-9) * 1e9
            self.dev_info.setText(f"Grid {nx} × {nz} cells incl. CPML · {lam / (n_gaas * dx):.1f} cells per wavelength in "
                                  f"GaAs · apex radius = {R / dx:.1f} cells · {steps:,} time steps".replace(",", " "))
        except (TypeError, ValueError, ZeroDivisionError):
            self.dev_info.setText("")
        self.dev_plot.plot(lambda fig: draw_fdtd_device(fig, cfg))

    # ================================================================== mode 1: pump field
    def _on_mode(self, i):
        self.mode_stack.setCurrentIndex(i)
        self.run_btn.setText("  Run pump FDTD" if i == 0 else "  Run height series")
        if i == 1:
            self._update_heights_info()
        self._schedule_device()

    def _pick_dir(self, edit: QLineEdit):
        start = paths.absolute(edit.text().strip()) if edit.text().strip() else paths.GUI_RUNS
        d = QFileDialog.getExistingDirectory(self, "Choose output folder", str(start if start.exists() else paths.RESULTS))
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
            self.est1.set("—", "Invalid parameters" + (": " + ", ".join(bad) if bad else ""))
            self.warn1.setVisible(False)
            return
        w = self.cores1.value()
        total, serial = pump_total(v, w), pump_total(v, 1)
        self.est1.set(f"≈ {fmt_dur(total)}",
                      f"{speedup_text(3, 'runs', w, total, serial)} · range {fmt_range(total)} · {steps} steps on "
                      f"{nx}×{nz} cells per run")
        self.est1.setToolTip("Vacuum reference, no tip and with tip: three independent FDTD runs, each on one core. "
                             "Runtime ∝ dx⁻³ (grid incl. PML).")
        warns = []
        dx = v.get("dx_nm", 15.0)
        ppw = v.get("wavelength_nm", 820.0) / (max(v.get("n_gaas", 3.67), 1.0) * dx)
        if ppw < 10:
            warns.append(f"Coarse grid: only {ppw:.1f} cells per wavelength in GaAs (14.9 at the default 15 nm) — "
                         "good for quick tests only.")
        if v.get("tip_radius_nm", 40.0) / dx < 2:
            warns.append(f"The apex radius spans only {v.get('tip_radius_nm', 40.0) / dx:.1f} cells; the tip shape is "
                         "under-resolved.")
        if abs(dx - 15.0) > 1e-9:
            warns.append("dx ≠ 15 nm: this result's notip.npz cannot serve as the tip-height series reference.")
        if total > 1800:
            warns.append(f"Long run (≈ {fmt_dur(total)}). Runtime ∝ dx⁻³.")
        self.warn1.setText("\n".join("⚠ " + w for w in warns))
        self.warn1.setVisible(bool(warns))
        self._paint(self.warn1, "warn")

    def run(self):
        if self.mode_bar.currentIndex() == 1:
            self.run_heights()
            return
        if not self.form.is_valid():
            error_box(self, "Invalid parameters", "These values cannot be parsed or are out of range:\n"
                      + ", ".join(self.form.invalid_fields()))
            return
        ov = self.form.overrides()
        kw = pump_kwargs(ov)
        txt = self.out1.text().strip()
        out = paths.absolute(txt) if txt else paths.new_run_dir("fdtd")
        if out.exists() and not out.is_dir():
            error_box(self, "Invalid output folder", f"{paths.rel(out)} is a file, not a folder.")
            return
        if out.exists() and any(out.iterdir()):
            committed = out.resolve() == PUMP_DIR.resolve()
            msg = (f"{paths.rel(out)} already contains files; the run overwrites its npz files, figure and summary.json."
                   + ("\n\nThis is the repository's reference result (computed with the cone tip): the height series "
                      "and the Single-pulse chain read its notip.npz / ratio.npz by default, so overwriting it with "
                      "other parameters silently changes downstream results." if committed else "") + "\n\nContinue?")
            if QMessageBox.question(self, "Overwrite existing results?", msg) != QMessageBox.StandardButton.Yes:
                return
        w = self.cores1.value()
        total, serial = pump_total(self._pump_values(), w), pump_total(self._pump_values(), 1)
        if total > 1800 and QMessageBox.question(
                self, "Long run", f"With the current grid this takes about {fmt_dur(total)}. Run it?") \
                != QMessageBox.StandardButton.Yes:
            return
        shape = kw["tip_shape"]
        hidden = {"tip_shape"} | ({"tip_half_angle_deg"} if shape == "rod" else set())       # shape is in the title
        desc = ", ".join(f"{k}={format_float(v) if isinstance(v, float) else v}" for k, v in ov.items()
                         if k not in hidden) or "default parameters"
        k = min(w, 3)
        self.job = JobManager.instance().submit(
            f"Pump FDTD · {shape_text(shape)} · {desc} · {k} core{'s' if k > 1 else ''}", "fdtd", PAR_SCRIPT,
            "fdtd_pump", kwargs=dict(out=paths.rel(out), workers=w, **kw), out=out, progress=PUMP_PROGRESS)
        self.run_info.setText(f"Pump-field FDTD with the {shape_text(shape)}: {speedup_text(3, 'runs', w, total, serial)} "
                              f"(range {fmt_range(total)}) · output: {paths.rel(out)}. Each FDTD run uses one core. "
                              "A stopped run cannot be resumed; running again recomputes all three. The optical field "
                              "opens automatically when the run finishes.")
        self.panel.attach(self.job)
        self.tabs.setCurrentWidget(self.tab_run)

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
            note = "  (repository reference)" if p == DEFAULT_REF else ("" if p.exists() else "  (missing)")
            self.ref.addItem(paths.rel(p) + note, str(p))
        self.ref.setCurrentIndex(max(0, cands.index(Path(want))))
        self.ref.blockSignals(False)
        self._update_heights_info()

    def _pick_ref(self):
        start = self.ref.currentData() or str(PUMP_DIR)
        f, _ = QFileDialog.getOpenFileName(self, "Choose the no-tip reference notip.npz", start, "notip.npz (*.npz)")
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
            self.out2_info.setText("Each run creates a new folder; nothing is written to results/fdtd_pump/heights.")
        elif not out.exists():
            self.out2_info.setText("The folder does not exist yet and will be created.")
        elif not out.is_dir():
            self.out2_info.setText("⚠ This is a file, not a folder.")
        else:
            skip = [h for h in existing if h in {int(x) for x in hs}]
            other = self._other_shapes(out) if existing else set()
            self.out2_info.setText(f"{len(existing)} ratio_h*.npz already here" + (
                f"; {len(skip)} height(s) ({', '.join(map(str, skip))} nm) will be skipped (resume)." if skip else ".")
                + (f"\n⚠ They were computed with the {' and '.join(sorted(other))} tip; new heights would use the "
                   f"{self._shape(self.hform)} — choose that shape or another folder." if other else "")
                + ("\n⚠ This is the repository's height-results folder: npz files will be written and summary.json "
                   "overwritten." if out.resolve() == HEIGHTS_DIR.resolve() else ""))
        ref = Path(self.ref.currentData()) if self.ref.currentData() else DEFAULT_REF
        level, msg = self._check_ref(ref)
        self.ref_info.setText(msg)
        self._paint(self.ref_info, level)
        if len(todo) != self._runs2:
            self._runs2 = len(todo)
            self.cores2.set_runs(max(1, len(todo)))         # re-enters here through cores2.changed with the same n
        w = self.cores2.value()
        per = fdtd_cost()[3] + RUN_OVERHEAD
        total, serial = heights_total(len(todo), w), heights_total(len(todo), 1)
        bad = [h for h in hs if h <= 0]
        dup = len({int(h) for h in hs}) != len(hs)
        if not hs or bad or dup:
            self.est2.set("—", "Invalid height list: the heights must be positive with distinct int(h).")
        else:
            self.est2.set(f"≈ {fmt_dur(total)}" if todo else "Nothing to compute",
                          (f"{speedup_text(len(todo), 'heights', w, total, serial)} · range {fmt_range(total)} · "
                           f"≈ {fmt_dur(per)} per height on one core (full 15 nm FDTD)" if todo else "")
                          + (f"{'; ' if todo else ''}{len(hs) - len(todo)} already done, skipped" if len(hs) > len(todo)
                             else ""))

    def _other_shapes(self, d: Path) -> set[str]:
        """Tip shapes of the heights already in folder `d` that differ from the one chosen in the heights form."""
        return heights_shapes(d) - {self._shape(self.hform)}

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
            error_box(self, "Invalid height list", "Cannot parse the height list. Use comma-separated numbers, e.g. 10, 15, 25.")
            return
        hs = self._heights()
        if not hs or any(h <= 0 for h in hs) or len({int(h) for h in hs}) != len(hs):
            error_box(self, "Invalid height list", "The heights must be positive and distinct after rounding down "
                      "(the files are named ratio_h{int(h)}.npz).")
            return
        ref = Path(self.ref.currentData()) if self.ref.currentData() else DEFAULT_REF
        out = self._out2() or paths.new_run_dir("fdtd_heights")
        if out.exists() and not out.is_dir():
            error_box(self, "Invalid output folder", f"{paths.rel(out)} is a file, not a folder.")
            return
        jm = JobManager.instance()
        shape, w = self._shape(self.hform), self.cores2.value()
        pre = None
        pre_s = pump_total({}, min(w, 3))
        if not ref.exists():
            m = QMessageBox(self)
            m.setIcon(QMessageBox.Icon.Question)
            m.setWindowTitle("Reference field missing")
            m.setText(f"The reference field {paths.rel(ref)} does not exist.\n\nA default pump FDTD (dx = 15 nm, "
                      f"≈ {fmt_dur(pre_s)}) can be queued first; the height series then starts automatically with its "
                      "notip.npz as the reference.")
            yes = m.addButton("Run pump FDTD first", QMessageBox.ButtonRole.AcceptRole)
            m.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            m.exec()
            if m.clickedButton() is not yes:
                return
            pdir = paths.new_run_dir("fdtd")
            k = min(w, 3)
            pre = jm.submit(f"Pump FDTD · {shape_text(shape)} · default parameters (height-series reference) · {k} "
                            f"core{'s' if k > 1 else ''}", "fdtd", PAR_SCRIPT, "fdtd_pump",
                            kwargs=dict(out=paths.rel(pdir), workers=k, tip_shape=shape), out=pdir, progress=PUMP_PROGRESS)
            ref = pdir / "notip.npz"
        else:
            level, msg = check_ref(ref)
            if level == "err":
                error_box(self, "Reference field unusable", msg)
                return
            if level == "warn" and QMessageBox.question(self, "Reference parameters differ", msg + "\n\nContinue anyway?") \
                    != QMessageBox.StandardButton.Yes:
                return
        if out.exists() and out.resolve() == HEIGHTS_DIR.resolve() and QMessageBox.question(
                self, "Write into the repository results folder?",
                "results/fdtd_pump/heights is a repository results folder (its summary.json is committed).\n"
                "The run writes ratio_h*.npz and gui_run*.json / gui_log*.txt there and overwrites summary.json. Continue?") \
                != QMessageBox.StandardButton.Yes:
            return
        if not self._clear_broken(out):
            return
        existing = heights_in(out) if out.is_dir() else {}
        todo = [h for h in hs if int(h) not in existing]
        other = self._other_shapes(out) if existing else set()
        if todo and other and QMessageBox.question(
                self, "Different tip shape",
                f"The heights already in {paths.rel(out)} were computed with the {' and '.join(sorted(other))} tip; the "
                f"new ones would use the {shape} tip, so one summary would mix two tip shapes.\n\nContinue anyway?") \
                != QMessageBox.StandardButton.Yes:
            return
        rel_out, rel_ref = paths.rel(out), paths.rel(ref)
        if not todo:
            if not existing:
                return
            self.job = self._submit_summary(out, ref)
            self.run_info.setText(f"All selected heights already exist in {rel_out}; only summary.json is rebuilt "
                                  "(a few seconds).")
            self.panel.attach(self.job)
            self.tabs.setCurrentWidget(self.tab_run)
            return
        hs_json = [int(h) if float(h).is_integer() else float(h) for h in hs]
        k = min(w, len(todo))
        # parallel.fdtd_heights computes the missing heights, one per core, then rebuilds summary.json from every
        # ratio_h*.npz in the folder (heights of earlier runs included)
        self.job = jm.submit(f"Tip-height series · {shape_text(shape)} · {len(hs)} heights ({min(hs_json)}–{max(hs_json)} "
                             f"nm) · {k} core{'s' if k > 1 else ''}", "fdtd_heights", PAR_SCRIPT, "fdtd_heights",
                             kwargs=dict(out=rel_out, heights=hs_json, ref=rel_ref, tip_shape=shape, workers=w), out=out,
                             progress={"type": "count", "pattern": PAR_PATTERN, "total": len(todo)}, after=pre)
        total, serial = heights_total(len(todo), w), heights_total(len(todo), 1)
        self.run_info.setText(
            (f"A default pump FDTD (≈ {fmt_dur(pre_s)}) is queued first; the height series starts when it finishes. "
             if pre else "")
            + f"Tip-height series with the {shape_text(shape)}: {speedup_text(len(todo), 'heights', w, total, serial)}"
            + (f" ({len(hs) - len(todo)} already done, skipped)" if len(hs) > len(todo) else "")
            + f" · output: {rel_out}. Each height is one FDTD run on one core. You can stop at any time; finished "
            "heights are kept and skipped when you resume.")
        self.panel.attach(pre or self.job)
        self.tabs.setCurrentWidget(self.tab_run)

    def _resume_heights(self):
        if not self.heights:
            return
        d = self.heights["dir"]
        self.out2.setText(paths.rel(d))
        hs = sorted(set(self.heights["summary"]) | set(self.heights["files"]))
        if hs:
            self._set_heights(tuple(hs))
        if self.heights.get("shape") in SHAPE_LABEL:          # resume with the folder's tip shape
            self.hform.set_values({"tip_shape": SHAPE_LABEL[self.heights["shape"]]})
        self._refresh_refs(select=self.heights["ref"])
        self.mode_bar.setCurrentIndex(1)

    def _submit_summary(self, d: Path, ref: Path):
        """Queue gui_tasks.fdtd_heights_summary: rebuild summary.json of folder `d` without computing any height."""
        return JobManager.instance().submit(
            f"Rebuild height summary · {paths.rel(d)}", "fdtd_heights", "ltemsim_gui/gui_tasks.py", "fdtd_heights_summary",
            kwargs=dict(out=paths.rel(d), ref=paths.rel(ref)), out=d)

    def _clear_broken(self, d: Path) -> bool:
        """Offer to move incomplete ratio_h*.npz aside (the script skips them as 'exists' and the summary task
        crashes on them).  False if the user cancelled."""
        files = heights_in(d) if d.is_dir() else {}
        bad = {h: p for h, f in files.items() if (p := height_npz_problem(f))}
        if not bad:
            return True
        detail = "\n".join(f"ratio_h{h}.npz: {p}" for h, p in bad.items())
        if QMessageBox.question(
                self, "Incomplete result files found",
                f"{len(bad)} ratio_h*.npz in {paths.rel(d)} are incomplete (most likely left by an interrupted run):\n\n"
                f"{detail}\n\nrun_fdtd_heights.py would skip them as finished, and rebuilding summary.json would fail on "
                "them.\nRename them to *.npz.partial (nothing is deleted) and continue?") != QMessageBox.StandardButton.Yes:
            return False
        quarantine(files[h] for h in bad)
        return True

    def _rebuild_summary(self):
        if not self.heights:
            return
        d, ref = self.heights["dir"], self.heights["ref"]
        if not ref.exists():
            error_box(self, "Reference field missing", f"The reference field {paths.rel(ref)} does not exist, so the "
                      "absorbed fractions cannot be computed.")
            return
        if d.resolve() == HEIGHTS_DIR.resolve() and QMessageBox.question(
                self, "Overwrite the committed summary.json?",
                "This rewrites results/fdtd_pump/heights/summary.json from the folder's ratio_h*.npz and writes "
                "gui_run*.json / gui_log*.txt there. Continue?") != QMessageBox.StandardButton.Yes:
            return
        if not self._clear_broken(d):
            return
        n = len(heights_in(d))
        if not n:
            error_box(self, "Cannot rebuild", f"{paths.rel(d)} has no complete ratio_h*.npz; rebuilding would empty "
                      "summary.json.")
            self.show_heights(d, switch=False)
            return
        self.job = self._submit_summary(d, ref)
        self.run_info.setText(f"Rebuilding summary.json from the {n} ratio_h*.npz in {paths.rel(d)} (a few seconds).")
        self.panel.attach(self.job)
        self.tabs.setCurrentWidget(self.tab_run)

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
            self.run_info.setText(self.run_info.text() + f"\nRenamed {', '.join(moved)}, left incomplete by the "
                                  "interruption, to *.npz.partial; these heights are recomputed when you resume.")
            if self.heights and Path(self.heights["dir"]) == d:
                self.show_heights(d, switch=False)
        self._update_heights_info()

    # ================================================================== results: pump
    def show_pump(self, d, switch=True):
        """Load a pump result; with `switch`, open the optical field (or, without field data, the absorption tab)."""
        d = Path(d)
        self._pump_token += 1
        tok = self._pump_token
        self.p_title.setText(f"{paths.rel(d)}  ·  loading…")
        self.f_title.setText(f"{paths.rel(d)}  ·  loading…")
        run_async(load_pump, d, on_done=lambda r: tok == self._pump_token and self._show_pump(r, switch),
                  on_error=lambda e: tok == self._pump_token and self._load_failed((self.p_title, self.f_title), d, e))

    def _load_failed(self, titles, d, err):
        for t in titles:
            t.setText(f"{paths.rel(d)}  ·  failed to load")
        if self.f_title in titles:
            self.field.set_fields(None, f"Cannot read {paths.rel(d)} — see the error message for details.")
        error_box(self, "Loading failed", f"Cannot read {paths.rel(d)}", err)

    def _show_pump(self, r: dict, switch=False):
        self.pump = r
        d, m, cfg = r["dir"], r["metrics"], r.get("cfg")
        state = []
        if not r["runs"]:
            state.append("no npz data")
        elif len(r["runs"]) < 3:
            state.append("incomplete: only " + " / ".join(r["runs"]))
        if r.get("errors"):
            state.append("some files are damaged")
        title = paths.rel(d) + ("  ·  " + ", ".join(state) if state else "")
        self.p_title.setText(title)
        self.f_title.setText(title)
        f = lambda k, fmt: fmt.format(m[k]) if isinstance(m.get(k), (int, float)) and np.isfinite(m[k]) else "—"
        fr = m.get("fresnel_p")
        self.m_notip.set(f("absorbed_fraction_notip", "{:.3f}"), f"Fresnel 1−|r_p|² = {fr:.3f}" if fr is not None else "")
        rel_tip = (m["absorbed_fraction_tip"] / m["absorbed_fraction_notip"]
                   if m.get("absorbed_fraction_tip") and m.get("absorbed_fraction_notip") else None)
        self.m_tip.set(f("absorbed_fraction_tip", "{:.3f}"), f"{rel_tip:.2f}× the no-tip value" if rel_tip else "")
        self.m_tip.setToolTip("Relative to the vacuum run's incident energy. In 2-D the tip is an infinite wedge, so its "
                              "shadow is an upper bound.")
        self.m_apex.set(f("surface_ratio_under_apex", "{:.2f}"), "x = 0, z = −20 nm")
        if isinstance(m.get("column_ratio_min"), float):
            self.m_col.set(f"{m['column_ratio_min']:.3f} – {m['column_ratio_max']:.2f}", "shadow – enhancement")
        else:
            self.m_col.set("—", "")
        self.m_col.setToolTip("Minimum (shadow) and maximum (enhancement) over x of ∫Q_tip dz / ∫Q_notip dz")
        if cfg:
            self.p_cfg.setText(
                f"θ {cfg['theta_deg']:g}° · w {cfg['w_um']:g} µm · X {cfg['x_beam_um']:g} µm · λ {cfg['wavelength'] * 1e9:.0f} nm · "
                f"FWHM {cfg['fwhm_fs']:g} fs · n {cfg['n_gaas']:g}, κ {cfg['kappa_gaas']:g} · {tip_text(cfg)}, "
                f"gap {cfg['tip_gap'] * 1e9:.0f} nm · dx {cfg['dx'] * 1e9:g} nm")
        else:
            self.p_cfg.setText(f"θ = {m['theta_deg']:g}° · w = {m['w_um']:g} µm" if "theta_deg" in m else "")
        png = d / "fdtd_pump.png"
        if png.exists():
            self.p_img.load(png)
        else:
            self.p_img.clear("This result has no fdtd_pump.png (the run is unfinished or was interrupted; the figure is "
                             "written after all three runs).")
        self.p_map.set_layers(r["layers"], "This folder has no usable .npz data files, so the field maps cannot be drawn.")
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
        self.btn_as_ref.setToolTip("Use this result's notip.npz as the height-series reference" if has_ref
                                   else "Needs a notip.npz on the default 15 nm grid")
        self.btn_params.setEnabled(bool(cfg))
        self.btn_p_heights.setVisible((d / "heights").is_dir())
        fd = r.get("field")
        if fd:
            self.field.set_fields(fd)
            e0 = fd.get("e0")
            self.f_note.setToolTip(FIELD_HELP + (f"\n\n|E0| = {e0:.3g} V/m per unit source current, from "
                                                 f"{fd.get('e0_run')}.npz." if e0 else ""))
        else:
            fast, full = pump_total({"dx_nm": 40.0}), pump_total({})
            self.field.set_fields(None, (
                "No optical-field data in this folder. The maps need vacuum/notip/tip.npz (or ratio.npz), which "
                "run_fdtd_pump.py writes but git does not track.\n\nRun the pump FDTD to compute them: a coarse "
                f"dx = 40 nm grid takes ≈ {fmt_dur(fast)}, the default 15 nm grid ≈ {fmt_dur(full)}."))
        if switch:
            self.tabs.setCurrentWidget(self.tab_field if fd else self.tab_pump)

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
        self.h_title.setText(f"{paths.rel(d)}  ·  loading…")
        if switch:
            self.tabs.setCurrentWidget(self.tab_heights)
        run_async(load_heights, d, on_done=lambda r: tok == self._heights_token and self._show_heights(r),
                  on_error=lambda e: tok == self._heights_token and self._load_failed((self.h_title,), d, e))

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
            note.append(f"{len(r['from_npz'])} height(s) not in summary.json, computed from the npz")
        if bad:
            note.append(f"{len(bad)} incomplete npz (h = {', '.join(map(str, bad))} nm)")
        self.h_title.setText(paths.rel(d) + ("  ·  " + ", ".join(note) if note else ""))
        hs, af, ra = self._fractions(r)
        f = lambda v, fmt: fmt.format(v) if np.isfinite(v) else "—"
        n_ok = len(files) - len(bad)
        if hs:
            self.h_n.set(str(len(hs)), f"{hs[0]}–{hs[-1]} nm · {n_ok} npz" + (f", {len(bad)} incomplete" if bad else ""))
            nt, k = norm["notip"], norm["factor"]
            if norm["basis"] == "vacuum":
                hint = f"no tip {nt:.3f}"
                tip = ("Absorbed energy / incident energy of the vacuum reference — directly comparable with “Absorption” "
                       "and the Fresnel value.\nsummary.json divides by the notip run's net incident flux (incident minus "
                       f"part of the reflection); here it is multiplied by launched_notip / launched_vacuum = {k:.3f};\n"
                       f"raw summary.json values: {f(af[0] / k, '{:.3f}')} → {f(af[-1] / k, '{:.3f}')}.")
            else:
                hint = f"no tip {nt:.3f} (÷ notip net flux)" if nt else "÷ notip run's net incident flux"
                tip = ("No vacuum.npz next to the reference, so the values cannot be converted to the vacuum incident "
                       "energy: they equal summary.json (divided by the notip run's net incident flux, i.e. incident "
                       "minus part of the reflection) and are not directly comparable with the absorbed fractions "
                       "under “Absorption”.")
            self.h_abs.set(f"{f(af[0], '{:.3f}')} → {f(af[-1], '{:.3f}')}", hint)
            self.h_abs.setToolTip(tip)
            self.h_apex.set(f"{f(ra[0], '{:.2f}')} → {f(ra[-1], '{:.2f}')}", "z = −20 nm; < 1 means shadow")
        else:
            self.h_n.set("0", f"{n_ok} npz" + (f", {len(bad)} incomplete" if bad else ""))
            self.h_abs.set("—", "")
            self.h_abs.setToolTip("")
            self.h_apex.set("—", "")
        ref = r["ref"]
        self.h_ref.set(ref.parent.name, ref.name + ("" if ref.exists() else "  (missing)"))
        self.h_ref.setToolTip(paths.rel(ref))
        if hs:
            self.h_plot.plot(lambda fig: self._draw_heights(fig, r))
        else:
            self.h_plot.plot(lambda fig: fig.text(0.5, 0.5, "No summary.json and no complete ratio_h*.npz in this folder",
                                                  ha="center", va="center", color=ThemeManager.instance().tokens.muted))
        self.h_map.set_layers(r["layers"], "This folder has no usable ratio_h*.npz (they are .gitignored; the repository "
                                           "only has summary.json). Run a height series to see the ratio map of each "
                                           "height here.")
        self.btn_summary.setEnabled(bool(files))
        self.btn_summary.setToolTip(("Rename the incomplete npz to *.npz.partial first, then rebuild summary.json from the "
                                     "remaining ratio_h*.npz" if bad
                                     else "Rebuild summary.json from every ratio_h*.npz in the folder") if files
                                    else "No ratio_h*.npz — rebuilding would empty summary.json")

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

        header("Pump field (vacuum / notip / tip)")
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
                line = (f"no tip {s.get('absorbed_fraction_notip', float('nan')):.3f} · with tip "
                        f"{s.get('absorbed_fraction_tip', float('nan')):.3f} · θ {s.get('theta_deg', 45):g}° · w {s.get('w_um', 3):g} µm")
            elif npz:
                line = "incomplete: only " + " / ".join(npz)
            else:
                line = "empty folder (run failed or still running)"
            # runs without tip_shape (the repository reference, older GUI runs) used the cone
            line += f" · {shape_text(shape_of(kw))}"
            line += f" · dx {kw.get('dx_nm', 15):g} nm" if d != PUMP_DIR else " · repository reference"
            extra = ", ".join(form_kwarg(k, v) for k, v in kw.items() if k not in ("dx_nm", "tip_shape", "workers"))
            it = QListWidgetItem(f"{paths.rel(d)}\n    {line}" + (f" · {extra}" if extra else "") + f" · {when(d)}")
            it.setData(Qt.ItemDataRole.UserRole, ("pump", str(d)))
            self.hist.addItem(it)
            n += 1
        if not n:
            self.hist.addItem("    No pump-field results yet")

        header("Tip-height series (ratio_h*.npz)")
        n = 0
        for d in [HEIGHTS_DIR] + sorted((p for p in paths.GUI_RUNS.glob("fdtd_heights_*") if p.is_dir()), reverse=True):
            if not d.exists():
                continue
            s = read_dict(d / "summary.json")
            files = heights_in(d)
            hs = sorted({int(float(k)) for k in s} | set(files))
            shape = heights_shape(d)
            line = (f"{len(hs)} heights: {', '.join(map(str, hs))} nm" if hs else "no finished heights yet") + \
                   f" · {'mixed tips' if shape == 'mixed' else shape_text(shape)} · {len(files)} npz" + \
                   (" · repository result (npz not committed)" if d == HEIGHTS_DIR and not files else "")
            it = QListWidgetItem(f"{paths.rel(d)}\n    {line} · {when(d)}")
            it.setData(Qt.ItemDataRole.UserRole, ("heights", str(d)))
            self.hist.addItem(it)
            n += 1
        if not n:
            self.hist.addItem("    No height-series results yet")

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
            elif self.pump is None:
                self.field.set_fields(None, "Run the pump FDTD (or load a result from History) to see the optical field.")
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
