"""Device cross-section graphics: structure diagrams and COMSOL-style field plots.

* `Device` describes the biased SI-GaAs gap (steady-state, single-pulse and tip-position models);
  `draw_device(fig, dev)` draws it to scale with a zoom on the gap region.
* `draw_fdtd_device(fig, cfg)` draws the near-tip optical (FDTD) model: GaAs, air, CPML, source,
  oblique p-polarised pump beam and the PtIr tip to scale, with an apex inset.
* `comsol_field(ax, ...)` draws a field map the way COMSOL's default plots look: rainbow surface,
  black geometry edges, normalised arrows or streamlines, colorbar with max/min markers.
* `FieldView` is a widget around `comsol_field` with quantity / view / overlay / colour controls;
  `load_steady_fields(path)` reads a steady-state solution.npz into its data format.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from .mpl import PlotWidget

C_AIR = "#e6f0fa"
C_GAAS = "#cbc4e4"
C_AU = "#e0ad2f"
C_TIP = "#8e9297"
C_BEAM = "#ff5a4f"
C_INK = "#1c2230"
C_EDGE = "#2b2f38"
COLORMAPS = {"Rainbow (COMSOL)": "jet", "Turbo": "turbo", "Thermal": "inferno", "Viridis": "viridis",
             "Wave (signed)": "RdBu_r"}


# ----------------------------------------------------------------------------- geometry
@dataclass
class Device:
    gap: float = 40.0              # µm, electrode spacing
    x_half: float = 60.0           # µm, half width of the simulated box
    depth: float = 30.0            # µm, GaAs thickness simulated
    air: float = 30.0              # µm, air above the surface
    V_bias: float = 25.0           # V
    anode_side: str = "-x"         # electrode on the + terminal at +V_bias
    spot_x0: float = 0.0           # µm, beam centre on the surface
    spot_w: float = 3.0            # µm, 1/e² intensity radius (projected on the surface)
    alpha_cm: float = 1.2e4        # absorption coefficient
    theta_deg: float = 0.0         # incidence angle from the normal (beam travels toward +x)
    P_mW: float | None = None
    tip_x: float | None = 0.0      # µm; None: no tip
    tip_h_um: float = 0.05         # apex height above the surface
    tip_half_angle: float = 15.0
    tip_shape: str = "rod"         # "rod": straight shank of width 2R; "cone": half-angle cone
    tip_radius_um: float = 0.04    # apex radius R
    V_tip: float | None = None
    marks: tuple = ()              # positions to mark on the surface (scan positions)
    marks_label: str = ""
    model: str = ""                # caption, e.g. "steady-state drift-diffusion + EL2"
    electrode_edges: tuple | None = None   # (left_inner, right_inner) from the actual grid, µm

    @property
    def potentials(self):
        """(V_left, V_right) as in Simulation.electrode_voltages (symmetric split)."""
        h = self.V_bias / 2.0
        return (h, -h) if self.anode_side == "-x" else (-h, h)

    @property
    def inner_edges(self):
        if self.electrode_edges:
            return self.electrode_edges
        return (-self.gap / 2.0, self.gap / 2.0)


def device_from_params(p, **extra) -> Device:
    """From an ltemsim Params object or a dict of Params fields (missing fields take Params defaults)."""
    from ltemsim.params import Params
    d = Params() if p is None else p
    if isinstance(d, dict):
        base = Params()
        g = lambda k: d.get(k, getattr(base, k))
    else:
        g = lambda k: getattr(d, k)
    scan = tuple(-x for x in g("tip_x_scan_um"))          # experimental X = -x_tip
    dev = Device(gap=g("gap"), x_half=g("x_half"), depth=g("depth"), air=g("air_height"), V_bias=g("V_bias"),
                 anode_side=g("anode_side"), spot_x0=g("spot_x0_um"), spot_w=g("spot_w_um"), alpha_cm=g("alpha_cm"),
                 theta_deg=0.0, P_mW=g("P_avg_mW"), tip_x=0.0, tip_h_um=g("tip_height_um"),
                 marks=(min(scan), max(scan)) if scan else (), marks_label="tip scan range",
                 model="steady state: drift-diffusion + EL2, continuous pump")
    for k, v in extra.items():
        setattr(dev, k, v)
    return dev


def device_from_pulse(V_bias=25.0, w_um=3.0, P_avg_mW=2.0, x_list=(), anode_side="-x", theta_deg=45.0,
                      tip_h_um=0.1, V_tip=None, **_) -> Device:
    """Single-pulse / tip-position chains: the tip is fixed at x = 0, the beam centre moves to x = X."""
    xs = tuple(x_list or ())
    return Device(V_bias=V_bias, anode_side=anode_side, spot_w=w_um, P_mW=P_avg_mW, theta_deg=theta_deg,
                  spot_x0=0.0, tip_x=0.0, tip_h_um=tip_h_um, V_tip=V_tip, marks=xs,
                  marks_label="beam positions X", model="single pulse: hydrodynamic carriers + Poisson")


# ----------------------------------------------------------------------------- structure: biased gap
_K = 1.0          # annotation scale of the figure being drawn


def _label(ax, x, y, text, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw["fontsize"] = kw.get("fontsize", 8.5) * _K
    kw.setdefault("color", C_INK)
    ax.text(x, y, text, **kw)


def _dim(ax, x0, x1, y, text, color, vertical=False):
    """Dimension arrow with a label."""
    if vertical:            # label on the inner side, so it stays inside the axes
        ax.annotate("", xy=(y, x0), xytext=(y, x1), arrowprops=dict(arrowstyle="<->", color=color, lw=0.9))
        ax.text(y, 0.5 * (x0 + x1), text + " ", ha="right", va="center", fontsize=8 * _K, color=color)
    else:
        ax.annotate("", xy=(x0, y), xytext=(x1, y), arrowprops=dict(arrowstyle="<->", color=color, lw=0.9))
        ax.text(0.5 * (x0 + x1), y, text, ha="center", va="bottom", fontsize=8 * _K, color=color)


def _electrode_labels(dev: Device):
    vl, vr = dev.potentials
    def lab(v):
        if abs(v) < 1e-12:
            return "Au electrode (0 V)"
        return f"Au anode (+{v:g} V)" if v > 0 else f"Au cathode ({v:+g} V)"
    return lab(vl), lab(vr)


def _beam(ax, dev: Device, z_top, half_width=None):
    w = dev.spot_w if half_width is None else half_width
    t = math.radians(dev.theta_deg)
    L = z_top
    dx = L * math.tan(t)                                  # beam comes from the upper left when theta > 0
    x0 = dev.spot_x0
    poly = [(x0 - w - dx, z_top), (x0 + w - dx, z_top), (x0 + w, 0), (x0 - w, 0)]
    ax.add_patch(__import__("matplotlib").patches.Polygon(poly, closed=True, fc=C_BEAM, ec="none", alpha=0.28, zorder=3))
    ax.annotate("", xy=(x0, 0.08 * z_top), xytext=(x0 - 0.55 * dx, 0.55 * z_top),
                arrowprops=dict(arrowstyle="-|>", color=C_BEAM, lw=1.4), zorder=4)


def _tip(ax, x, h, top, half_angle, shape="cone", half_width=None):
    """Tip from the apex height h to `top`: a cone of the given half-angle, or a straight rod of half width
    `half_width` (drawn with a rounded apex)."""
    import matplotlib.patches as mp
    if shape == "rod":
        hw = half_width
        ax.add_patch(mp.FancyBboxPatch((x - hw, h + hw), 2 * hw, max(top - h - hw, 0), boxstyle="square,pad=0",
                                       fc=C_TIP, ec=C_EDGE, lw=0.6, zorder=5))
        ax.add_patch(mp.Ellipse((x, h + hw), 2 * hw, 2 * hw, fc=C_TIP, ec=C_EDGE, lw=0.6, zorder=5))
        return
    a = math.tan(math.radians(half_angle))
    pts = [(x, h), (x + a * (top - h), top), (x - a * (top - h), top)]
    ax.add_patch(mp.Polygon(pts, closed=True, fc=C_TIP, ec=C_EDGE, lw=0.6, zorder=5))


def draw_device(fig, dev: Device, zoom_half: float | None = None):
    """Two panels: (a) the whole simulated cross-section to scale, (b) the gap region."""
    import matplotlib
    import matplotlib.patches as mp
    fig.clear()
    # annotation sizes follow the figure width (the defaults suit ~1000 px)
    k = float(np.clip(fig.get_figwidth() * fig.dpi / 1000.0, 0.72, 1.0))
    with matplotlib.rc_context({"font.size": 9.5 * k}):
        return _draw_device(fig, dev, zoom_half, k, mp)


def _draw_device(fig, dev, zoom_half, k, mp):
    global _K
    _K = k
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    ax, az = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    xl, xr = dev.inner_edges
    ll, lr = _electrode_labels(dev)
    vl, vr = dev.potentials
    t_au = 0.035 * dev.air                                # electrode thickness exaggerated for visibility

    # ---- (a) whole device
    ax.add_patch(mp.Rectangle((-dev.x_half, 0), 2 * dev.x_half, dev.air, fc=C_AIR, ec=C_EDGE, lw=0.8))
    ax.add_patch(mp.Rectangle((-dev.x_half, -dev.depth), 2 * dev.x_half, dev.depth, fc=C_GAAS, ec=C_EDGE, lw=0.8))
    for x0, x1 in ((-dev.x_half, xl), (xr, dev.x_half)):
        ax.add_patch(mp.Rectangle((x0, 0), x1 - x0, t_au, fc=C_AU, ec=C_EDGE, lw=0.6, zorder=4))
    _beam(ax, dev, dev.air * 0.92)
    if dev.tip_x is not None:
        if dev.tip_shape == "rod":
            _tip(ax, dev.tip_x, max(dev.tip_h_um, 0.25), dev.air * 0.55, 0, "rod", 0.012 * dev.x_half)
        else:
            _tip(ax, dev.tip_x, max(dev.tip_h_um, 0.25), dev.air * 0.55, 8.0)
        _label(ax, dev.tip_x + 4.5, dev.air * 0.45, "AFM tip\n(schematic)", ha="left", fontsize=7.5)
    _label(ax, -dev.x_half * 0.62, dev.air * 0.7, "Air", fontsize=9.5)
    _label(ax, 0, -dev.depth * 0.55, "SI-GaAs (EL2 compensated)", fontsize=9.5)
    _label(ax, 0.5 * (-dev.x_half + xl), t_au + 0.06 * dev.air, ll, fontsize=7.5)
    _label(ax, 0.5 * (xr + dev.x_half), t_au + 0.06 * dev.air, lr, fontsize=7.5)
    _dim(ax, xl, xr, -0.12 * dev.depth, f"gap = {xr - xl:.4g} µm", "#3346c8")
    _dim(ax, -dev.depth, 0, dev.x_half * 0.93, f"{dev.depth:g} µm", "#3346c8", vertical=True)
    beam_txt = f"pump {dev.P_mW:g} mW, w = {dev.spot_w:g} µm" if dev.P_mW is not None else f"pump, w = {dev.spot_w:g} µm"
    if dev.theta_deg:
        beam_txt += f", θ = {dev.theta_deg:g}°"
    _label(ax, dev.spot_x0 - 2 - math.tan(math.radians(dev.theta_deg)) * dev.air * 0.9, dev.air * 0.97, beam_txt,
           ha="left" if dev.theta_deg else "center", va="top", fontsize=7.5, color="#c0392b")
    ax.set_xlim(-dev.x_half, dev.x_half)
    ax.set_ylim(-dev.depth, dev.air)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("z (µm)")

    # ---- (b) gap region
    zh = zoom_half or (max(xr, -xl) + 4.0)
    ztop, zbot = 0.22 * zh, -0.28 * zh
    az.add_patch(mp.Rectangle((-zh, 0), 2 * zh, ztop, fc=C_AIR, ec="none"))
    az.add_patch(mp.Rectangle((-zh, zbot), 2 * zh, -zbot, fc=C_GAAS, ec="none"))
    # generation profile: Gaussian spot × Beer–Lambert decay
    xs = np.linspace(-zh, zh, 400)
    zs = np.linspace(zbot, 0, 160)
    X, Z = np.meshgrid(xs, zs)
    delta = 1e4 / dev.alpha_cm                            # µm
    G = np.exp(-2 * (X - dev.spot_x0) ** 2 / dev.spot_w ** 2) * np.exp(Z / delta)
    az.imshow(np.dstack([np.full_like(G, 1.0), np.full_like(G, 0.55), np.full_like(G, 0.1), 0.85 * G]),
              extent=(-zh, zh, zbot, 0), origin="lower", aspect="auto", zorder=2)
    t_b = 0.05 * ztop
    for x0, x1 in ((-zh, xl), (xr, zh)):
        if x1 > x0:
            az.add_patch(mp.Rectangle((x0, 0), x1 - x0, t_b, fc=C_AU, ec=C_EDGE, lw=0.6, zorder=4))
    az.axhline(0, color=C_EDGE, lw=0.8, zorder=3)
    # incident intensity profile above the surface
    prof = np.exp(-2 * (xs - dev.spot_x0) ** 2 / dev.spot_w ** 2)
    az.fill_between(xs, 0, 0.55 * ztop * prof, color=C_BEAM, alpha=0.25, lw=0, zorder=2)
    az.plot(xs, 0.55 * ztop * prof, color=C_BEAM, lw=1.0, zorder=3)
    if dev.tip_x is not None:
        if dev.tip_shape == "rod":
            hw = max(dev.tip_radius_um, 0.006 * zh)              # true width 2R is below a pixel at this zoom
            _tip(az, dev.tip_x, max(dev.tip_h_um, 0.02 * ztop), ztop * 0.98, 0, "rod", hw)
            note = "" if hw <= dev.tip_radius_um * 1.01 else " (drawn wider)"
            _label(az, dev.tip_x + hw + 0.02 * zh, 0.32 * ztop, f"rod tip, 2R = {2 * dev.tip_radius_um * 1e3:.0f} nm{note}",
                   ha="left", fontsize=7)
        else:
            _tip(az, dev.tip_x, max(dev.tip_h_um, 0.02 * ztop), ztop * 0.98, dev.tip_half_angle)
        if dev.V_tip is not None:
            _label(az, dev.tip_x + 0.06 * zh, 0.55 * ztop, f"V_tip = {dev.V_tip:+g} V", ha="left", fontsize=7.5)
    # applied field direction (from the + to the − electrode)
    if abs(vl - vr) > 0:
        sgn = 1 if vl > vr else -1
        a0, a1 = (xl * 0.6, xr * 0.6) if sgn > 0 else (xr * 0.6, xl * 0.6)
        az.annotate("", xy=(a1, 0.62 * zbot), xytext=(a0, 0.62 * zbot),
                    arrowprops=dict(arrowstyle="-|>", color="#1f5fbf", lw=1.6), zorder=6)
        _label(az, 0, 0.62 * zbot, "applied E (bare gap)", va="bottom", fontsize=7.5, color="#1f5fbf")
        _label(az, 0, 0.62 * zbot - 0.04 * zh, f"V/gap = {abs(dev.V_bias) / (xr - xl) * 1e4 / 1e3:.2g} kV/cm",
               va="top", fontsize=7, color="#1f5fbf")
    if dev.marks:
        m = np.asarray(dev.marks, float)
        y = 0.86 * ztop
        if len(m) == 2 and dev.marks_label.endswith("range"):
            az.annotate("", xy=(m.min(), y), xytext=(m.max(), y), arrowprops=dict(arrowstyle="|-|", color="#2e7d32", lw=1.0))
        else:
            az.plot(m, np.full_like(m, y), "v", color="#2e7d32", ms=4, zorder=6)
        _label(az, m.max() + 0.02 * zh, y, " " + dev.marks_label, ha="left", fontsize=7.5, color="#2e7d32")
    _label(az, -zh * 0.97, zbot * 0.2, f"1/α = {delta:.2g} µm", ha="left", fontsize=7.5)
    _label(az, -zh * 0.97, ztop * 0.75, "Air", ha="left", fontsize=8.5)
    _label(az, -zh * 0.97, zbot * 0.85, "SI-GaAs", ha="left", fontsize=8.5)
    az.set_xlim(-zh, zh)
    az.set_ylim(zbot, ztop)
    az.set_xlabel("x (µm)")
    az.set_ylabel("z (µm)  [vertical stretched]")
    az.set_title("(b) gap region: pump absorption and bias", loc="left", fontsize=9.5 * k)
    ax.set_title("(a) simulated cross-section (to scale)", loc="left", fontsize=9.5 * k)
    if dev.model:
        fig.suptitle(dev.model, fontsize=9 * k, x=0.01, ha="left", color="#687185")
    return ax, az


# ----------------------------------------------------------------------------- structure: FDTD near-tip model
def draw_fdtd_device(fig, cfg: dict | None = None):
    """Near-tip optical model of run_fdtd_pump.py (lengths in metres in FDTDConfig, drawn in µm)."""
    import matplotlib.patches as mp
    from ltemsim.optics_fdtd import FDTDConfig
    c = FDTDConfig().__dict__.copy()
    c.update(dict(x_half=4.0e-6, z_min=-3.0e-6, z_max=2.5e-6, z_source=2.2e-6))      # fixed by the script
    c.update({k: v for k, v in (cfg or {}).items() if k in c})
    if cfg and "dx_nm" in cfg:
        c["dx"] = cfg["dx_nm"] * 1e-9
    global _K
    _K = 1.0
    um = 1e6
    xh, zmin, zmax, zs = c["x_half"] * um, c["z_min"] * um, c["z_max"] * um, c["z_source"] * um
    pml = c["n_pml"] * c["dx"] * um
    R, gap, ha, xt = c["tip_radius"] * um, c["tip_gap"] * um, c["tip_half_angle_deg"], c["tip_x"] * um
    th = math.radians(c["theta_deg"])
    fig.clear()
    ax = fig.add_subplot(111)
    ax.add_patch(mp.Rectangle((-xh - pml, zmin - pml), 2 * (xh + pml), zmax - zmin + 2 * pml, fc="#eceef2",
                              ec=C_EDGE, lw=0.8, hatch="////", zorder=0))
    ax.add_patch(mp.Rectangle((-xh, 0), 2 * xh, zmax, fc=C_AIR, ec="none", zorder=1))
    ax.add_patch(mp.Rectangle((-xh, zmin), 2 * xh, -zmin, fc=C_GAAS, ec="none", zorder=1))
    ax.add_patch(mp.Rectangle((-xh, zmin), 2 * xh, zmax - zmin, fc="none", ec=C_EDGE, lw=0.8, zorder=6))
    ax.axhline(0, xmin=0, xmax=1, color=C_EDGE, lw=0.8, zorder=6)
    # beam: Gaussian band from the source line to the surface, travelling toward +x
    xb, w = c["x_beam_um"], c["w_um"]
    dxs = zs * math.tan(th)
    ex, ez = math.cos(th), math.sin(th)                    # p-polarisation: E in the plane, perpendicular to k
    p = [(xb - dxs - w, zs), (xb - dxs + w, zs), (xb + w, 0), (xb - w, 0)]
    ax.add_patch(mp.Polygon(p, closed=True, fc=C_BEAM, ec="none", alpha=0.22, zorder=2))
    ax.plot([-xh, xh], [zs, zs], ls="--", color=C_BEAM, lw=1.0, zorder=3)
    ax.annotate("", xy=(xb - 0.38 * dxs, 0.38 * zs), xytext=(xb - 0.92 * dxs, 0.92 * zs),
                arrowprops=dict(arrowstyle="-|>", color=C_BEAM, lw=1.6), zorder=4)
    _label(ax, xb - 0.62 * dxs + 0.12, 0.62 * zs + 0.12, "k", fontsize=8, color=C_BEAM, ha="left")
    mx, mz = xb - 0.62 * dxs - 0.35, 0.62 * zs - 0.35
    ax.annotate("", xy=(mx + 0.35 * ex, mz + 0.35 * ez), xytext=(mx - 0.35 * ex, mz - 0.35 * ez),
                arrowprops=dict(arrowstyle="<->", color="#8e1b12", lw=1.0), zorder=4)
    _label(ax, mx - 0.4 * ex - 0.05, mz - 0.4 * ez, "E (p-pol)", ha="right", fontsize=7, color="#8e1b12")
    _label(ax, -xh + 0.1, zs - 0.06, f"source: {c['wavelength'] * 1e9:.0f} nm, {c['fwhm_fs']:g} fs, p-pol\n"
           f"θ = {c['theta_deg']:g}°, w = {w:g} µm", ha="left", va="top", fontsize=7.5, color="#c0392b", zorder=7)
    # tip to scale: sphere + cone
    zc = gap + R
    zt = zc + R * math.sin(math.radians(ha))
    xr0 = R * math.cos(math.radians(ha))
    rod = c.get("tip_shape", "cone") == "rod"

    def shank(top):
        if rod:
            return [(xt - R, zc), (xt + R, zc), (xt + R, top), (xt - R, top)]
        return [(xt - xr0, zt), (xt + xr0, zt), (xt + xr0 + (top - zt) * math.tan(math.radians(ha)), top),
                (xt - xr0 - (top - zt) * math.tan(math.radians(ha)), top)]
    ax.add_patch(mp.Polygon(shank(zmax), closed=True, fc=C_TIP, ec=C_EDGE, lw=0.6, zorder=5))
    ax.add_patch(mp.Circle((xt, zc), R, fc=C_TIP, ec=C_EDGE, lw=0.6, zorder=5))
    lx = xt + R + 0.25 if rod else xt + 0.25 + 1.7 * math.tan(math.radians(ha)) + 0.1
    _label(ax, lx, 1.5, "PtIr tip (Drude)\n" + (f"straight rod, 2R = {2 * R * 1e3:.0f} nm" if rod else f"cone, {ha:g}° half-angle"),
           ha="left", fontsize=8)
    _label(ax, -xh * 0.9, 0.5, "Air", ha="left", fontsize=9)
    _label(ax, -xh * 0.9, zmin * 0.5, f"GaAs  n = {c['n_gaas']:g} + {c['kappa_gaas']:g}i", ha="left", fontsize=9)
    _label(ax, -xh - pml * 0.5, zmax + pml * 0.5, "CPML", ha="left", va="center", fontsize=7, color="#333",
           bbox=dict(fc="white", ec="none", pad=0.5))
    ax.set_xlim(-xh - pml, xh + pml)
    ax.set_ylim(zmin - pml, zmax + pml)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("z (µm)")
    ax.set_title(f"2-D FDTD domain (to scale), dx = {c['dx'] * 1e9:.0f} nm", loc="left", fontsize=9.5)
    # apex inset (to scale)
    ins = ax.inset_axes([0.62, 0.1, 0.34, 0.34])
    s = max(4 * R, 6 * gap, 0.12)
    ins.add_patch(mp.Rectangle((-s, -s), 2 * s, s, fc=C_GAAS, ec="none"))
    ins.add_patch(mp.Rectangle((-s, 0), 2 * s, s + 2 * s, fc=C_AIR, ec="none"))
    ins.add_patch(mp.Polygon(shank(zt + 3 * s), closed=True, fc=C_TIP, ec=C_EDGE, lw=0.6))
    ins.add_patch(mp.Circle((xt, zc), R, fc=C_TIP, ec=C_EDGE, lw=0.6))
    ins.axhline(0, color=C_EDGE, lw=0.8)
    ins.annotate("", xy=(xt + 1.6 * R, 0), xytext=(xt + 1.6 * R, gap), arrowprops=dict(arrowstyle="<->", color="#3346c8", lw=0.8))
    ins.text(xt + 1.8 * R, gap + 0.06 * s, f"{gap * 1e3:.0f} nm", fontsize=6.5, va="bottom", color="#3346c8")
    ins.text(xt + 1.25 * R, zc, f"R = {R * 1e3:.0f} nm", fontsize=6.5, ha="left", va="center", color=C_INK)
    ins.set_xlim(xt - s, xt + s)
    ins.set_ylim(-0.5 * s, 1.5 * s)
    ins.set_aspect("equal")
    ins.tick_params(labelsize=6)
    ins.set_title("apex (to scale, µm)", fontsize=7, color=C_INK, backgroundcolor="white")
    return ax


# ----------------------------------------------------------------------------- COMSOL-style field plot
def _sample(xc, zc, A, xs, zs):
    """Bilinear sample of a cell-centred array A[z, x] on the grid xs × zs (NaN outside)."""
    from scipy.interpolate import RegularGridInterpolator
    it = RegularGridInterpolator((zc, xc), A, bounds_error=False, fill_value=np.nan)
    Z, X = np.meshgrid(zs, xs, indexing="ij")
    return it(np.stack([Z.ravel(), X.ravel()], axis=1)).reshape(Z.shape)


def comsol_field(ax, xe, ze, S, *, label="Electric field norm", unit="kV/cm", Ex=None, Ez=None, cmap="jet",
                 log=False, signed=False, overlay="arrows", xlim=None, zlim=None, outlines=(), electrodes=(),
                 surface=True, contours=None, equal=False, cax=None):
    """Surface plot in COMSOL's style.

    xe, ze: cell edges (µm) of S[z, x]; Ex, Ez (same shape) for arrows / streamlines;
    outlines: [(xc, zc, mask)] boolean masks drawn as black edges (e.g. the tip);
    electrodes: [(x0, x1)] intervals of metal on the surface (z = 0).
    Returns (mesh, colorbar).
    """
    import matplotlib.colors as mcolors
    xc, zc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ze[1:] + ze[:-1])
    xlim = xlim or (xe[0], xe[-1])
    zlim = zlim or (ze[0], ze[-1])
    sel_x = (xc >= xlim[0]) & (xc <= xlim[1])
    sel_z = (zc >= zlim[0]) & (zc <= zlim[1])
    vis = S[np.ix_(sel_z, sel_x)] if sel_x.any() and sel_z.any() else S
    finite = vis[np.isfinite(vis)]
    if log:
        pos = finite[finite > 0]
        vmin = max(pos.min(), pos.max() * 1e-6) if pos.size else 1e-6
        vmax = pos.max() if pos.size else 1.0
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    elif signed:
        m = np.nanmax(np.abs(finite)) if finite.size else 1.0
        norm = mcolors.Normalize(-m, m)
    else:
        norm = mcolors.Normalize(np.nanmin(finite) if finite.size else 0, np.nanmax(finite) if finite.size else 1)
    data = np.where(S > 0, S, np.nan) if log else S
    ax.set_facecolor("white")                             # cells without data show white, as in COMSOL
    mesh = ax.pcolormesh(xe, ze, data, cmap=cmap, norm=norm, shading="flat", rasterized=True)
    if contours:
        ax.contour(xc, zc, S, levels=contours, colors="k", linewidths=0.5, alpha=0.8)
    # vector overlay on a regular grid of the visible window
    if Ex is not None and Ez is not None and overlay in ("arrows", "streamlines"):
        nx = 26 if overlay == "arrows" else 120
        nzz = max(8, int(nx * (zlim[1] - zlim[0]) / max(xlim[1] - xlim[0], 1e-9))) if equal else (14 if overlay == "arrows" else 80)
        xs = np.linspace(xlim[0], xlim[1], nx + 2)[1:-1]
        zs = np.linspace(zlim[0], zlim[1], nzz + 2)[1:-1]
        U, W = _sample(xc, zc, Ex, xs, zs), _sample(xc, zc, Ez, xs, zs)
        if overlay == "arrows":
            mag = np.hypot(U, W)
            with np.errstate(invalid="ignore", divide="ignore"):
                Un, Wn = U / mag, W / mag                  # COMSOL "normalized" arrow length
            X, Z = np.meshgrid(xs, zs)
            # quiver works in display units: equal-length arrows regardless of the axis stretch
            ax.quiver(X, Z, Un, Wn, color="#d4141c", angles="xy", pivot="mid", scale=nx * 1.25,
                      scale_units="width", width=0.0028, headwidth=3.6, headlength=4.2, zorder=4)
        else:
            ok = np.isfinite(U) & np.isfinite(W)
            U, W = np.where(ok, U, 0.0), np.where(ok, W, 0.0)
            ax.streamplot(xs, zs, U, W, color="white", linewidth=0.7, density=1.3, arrowsize=0.7, zorder=4)
    # geometry edges
    if surface:
        ax.axhline(0, color="k", lw=0.9, zorder=5)
    for x0, x1 in electrodes:
        ax.plot([x0, x1], [0, 0], color="#b8860b", lw=4.0, solid_capstyle="butt", zorder=6)
        ax.plot([x0, x1], [0, 0], color="k", lw=0.6, zorder=7)
    for oxc, ozc, mask in outlines:
        ax.contour(oxc, ozc, mask.astype(float), levels=[0.5], colors="k", linewidths=0.9, zorder=6)
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    if equal:
        ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("z (µm)")
    import textwrap
    head = f"Surface: {label} ({unit})" if unit else f"Surface: {label}"
    head = "\n".join(textwrap.wrap(head, 52, subsequent_indent="    ")) or head      # long labels wrap, never clip
    if Ex is not None and overlay == "arrows":
        head += "\nArrow Surface: Electric field"
    elif Ex is not None and overlay == "streamlines":
        head += "\nStreamline: Electric field"
    ax.set_title(head, loc="left", fontsize=8.5, linespacing=1.3)
    cb = ax.figure.colorbar(mesh, ax=None if cax is not None else ax, cax=cax, fraction=0.04, pad=0.02)
    if finite.size:
        fmt = lambda v: f"{v:.3g}"
        cb.ax.set_title(f"▲ {fmt(np.nanmax(finite))}", fontsize=7.5, pad=4)
        cb.ax.set_xlabel(f"▼ {fmt(np.nanmin(finite))}", fontsize=7.5, labelpad=4)
    cb.ax.tick_params(labelsize=7.5)
    return mesh, cb


# ----------------------------------------------------------------------------- data loading
def load_steady_fields(path) -> dict:
    """Fields of a steady-state solution.npz for FieldView (run in a worker thread)."""
    from ltemsim import Simulation, Solution
    sol = Solution.load(str(path))
    sim = Simulation(sol.params, verbose=False)
    g = sim.grid
    Ex, Ez = sol.fields_physical(sim)                      # kV/cm, all cells
    V = sol.potential_V(sim)
    elec = sim.top.electrode_mask
    cols = np.where(elec)[0]
    left = cols[g.xc[cols] < 0]
    right = cols[g.xc[cols] > 0]
    electrodes = []
    if left.size:
        electrodes.append((g.xe[0], g.xe[left.max() + 1]))
    if right.size:
        electrodes.append((g.xe[right.min()], g.xe[-1]))
    edges = (electrodes[0][1] if left.size else -sol.params.gap / 2,
             electrodes[-1][0] if right.size else sol.params.gap / 2)
    p = sol.params
    dev = device_from_params(p, electrode_edges=edges, P_mW=p.P_avg_mW * sol.gen_factor)
    return dict(xe=g.xe, ze=g.ze, Ex=Ex, Ez=Ez, V=V, electrodes=electrodes, device=dev,
                title=f"{p.V_bias:+g} V, {p.P_avg_mW * sol.gen_factor:g} mW, w = {p.spot_w_um:g} µm",
                gap=(edges[0], edges[1]), x_half=p.x_half, depth=p.depth, air=p.air_height)


# ----------------------------------------------------------------------------- widget
class FieldView(QWidget):
    """COMSOL-style field map with controls.

    set_fields(data) with data from load_steady_fields() (vector field) or a dict
    {xe, ze, quantities: {name: (array, unit, signed)}, outlines, electrodes, views: {name: (xlim, zlim)}}
    for scalar maps (e.g. the FDTD optical field).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.qty = QComboBox()
        self.view = QComboBox()
        self.overlay = QComboBox()
        self.overlay.addItems(["Arrows", "Streamlines", "None"])
        self.cmap = QComboBox()
        self.cmap.addItems(list(COLORMAPS))
        self.log = QCheckBox("Log scale")
        self.equal = QCheckBox("1:1 aspect")
        self.labels = []
        for lab, w in (("Quantity", self.qty), ("View", self.view), ("Overlay", self.overlay), ("Colors", self.cmap)):
            l = QLabel(lab)
            l.setObjectName("muted")
            self.labels.append(l)
            w.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            w.setMinimumContentsLength(5)
            w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.bar = QWidget()
        self.grid = QGridLayout(self.bar)            # one row when wide, two rows when narrow (see resizeEvent)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(6)
        self.grid.setVerticalSpacing(4)
        self._wide = None
        self._layout_bar(False)
        lay.addWidget(self.bar)
        self.bar.hide()
        self.plot = PlotWidget(figsize=(8, 4.6))
        lay.addWidget(self.plot, 1)
        self.empty = QLabel("Run a solve (or load a result) to see the field map.", alignment=Qt.AlignmentFlag.AlignCenter)
        self.empty.setObjectName("muted")
        self.empty.setWordWrap(True)
        lay.addWidget(self.empty, 1)
        self.plot.hide()
        for w in (self.qty, self.view, self.overlay, self.cmap):
            w.currentIndexChanged.connect(self._redraw)
        self.log.toggled.connect(self._redraw)
        self.equal.toggled.connect(self._redraw)
        self._busy = False

    def _layout_bar(self, wide: bool):
        if wide == self._wide:
            return
        self._wide = wide
        g = self.grid
        for w in (*self.labels, self.qty, self.view, self.overlay, self.cmap, self.log, self.equal):
            g.removeWidget(w)
        for c in range(11):
            g.setColumnStretch(c, 0)
        (lq, lv, lo, lc) = self.labels
        if wide:
            row = [(lq, 0), (self.qty, 3), (lv, 0), (self.view, 2), (lo, 0), (self.overlay, 1), (lc, 0), (self.cmap, 2),
                   (self.log, 0), (self.equal, 0)]
            for c, (w, st) in enumerate(row):
                g.addWidget(w, 0, c)
                g.setColumnStretch(c, st)
        else:
            for r, items in enumerate(((lq, self.qty, lv, self.view), (lo, self.overlay, lc, self.cmap))):
                for c, w in enumerate(items):
                    g.addWidget(w, r, c)
            g.addWidget(self.log, 0, 4)
            g.addWidget(self.equal, 1, 4)
            g.setColumnStretch(1, 3)
            g.setColumnStretch(3, 2)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._layout_bar(self.width() >= 980)

    # -- data
    def set_fields(self, data: dict | None, message: str | None = None):
        self.data = data
        if not data:
            self.plot.hide()
            self.bar.hide()                          # no controls above an empty state
            self.empty.setText(message or "No field data.")
            self.empty.show()
            return
        self._busy = True
        try:
            self.qty.clear()
            vec = "Ex" in data
            if vec:
                u = data.get("unit", "kV/cm")
                self.qtys = {
                    "Electric field norm |E|": (np.hypot(data["Ex"], data["Ez"]), u, False),
                    "Electric field, x component E_x": (data["Ex"], u, True),
                    "Electric field, z component E_z": (data["Ez"], u, True),
                }
                if data.get("V") is not None:
                    self.qtys["Electric potential V"] = (data["V"], "V", True)
                self.qtys.update(data.get("quantities") or {})     # extra scalar maps on the same grid
            else:
                self.qtys = dict(data["quantities"])
            self.qty.addItems(list(self.qtys))
            self.view.clear()
            if data.get("views"):
                self.views = dict(data["views"])
            elif vec:
                g0, g1 = data.get("gap", (-20.0, 20.0))
                self.views = {
                    "Electrode gap": ((g0 - 4, g1 + 4), (-0.35 * (g1 - g0 + 8) / 2, 0.15 * (g1 - g0 + 8) / 2)),
                    "Spot region (±10 µm)": ((-10, 10), (-3, 1)),
                    "Whole device": ((data["xe"][0], data["xe"][-1]), (data["ze"][0], data["ze"][-1])),
                    "Near surface (−4…1 µm)": ((g0 - 4, g1 + 4), (-4, 1)),
                }
            else:
                self.views = {"Whole domain": (None, None)}
            self.view.addItems(list(self.views))
            self.overlay.setEnabled(vec)
            self.overlay.setVisible(vec)                 # scalar maps have no vector overlay
            self.labels[2].setVisible(vec)
            log0 = data.get("log_default")
            if log0 is None and vec:
                En = np.hypot(data["Ex"], data["Ez"])
                med = np.nanmedian(En[En > 0]) if np.any(En > 0) else 0
                log0 = bool(med and np.nanmax(En) / med > 50)
            self.log.setChecked(bool(log0))
        finally:
            self._busy = False
        self.empty.hide()
        self.bar.show()
        self.plot.show()
        for c in (self.qty, self.view, self.overlay, self.cmap):   # popups as wide as their longest entry
            c.view().setMinimumWidth(c.view().sizeHintForColumn(0) + 28)
        self._redraw()

    def _redraw(self, *_):
        if self._busy or not self.data or not self.qty.currentText():
            return
        d = self.data
        name = self.qty.currentText()
        S, unit, signed = self.qtys[name]
        xlim, zlim = self.views.get(self.view.currentText(), (None, None))
        cmap = COLORMAPS[self.cmap.currentText()]
        vec = "Ex" in d
        overlay = self.overlay.currentText().lower() if vec else "none"
        self.log.setEnabled(not signed)                  # signed quantities are always linear
        log = self.log.isChecked() and not signed
        label = name.split(" E_")[0].split(" |E|")[0] if vec else name

        def draw(fig):
            ax = fig.add_subplot(111)
            comsol_field(ax, d["xe"], d["ze"], S, label=label, unit=unit, Ex=d.get("Ex"), Ez=d.get("Ez"),
                         cmap=cmap, log=log, signed=signed and cmap == "RdBu_r", overlay=overlay, xlim=xlim,
                         zlim=zlim, outlines=d.get("outlines", ()), electrodes=d.get("electrodes", ()),
                         contours=12 if name.startswith("Electric potential") else None, equal=self.equal.isChecked())
            if d.get("title"):
                fig.suptitle(d["title"], fontsize=8.5, x=0.99, ha="right", color="#687185")

            # cursor read-out of the plotted quantity
            xe, ze = d["xe"], d["ze"]

            def fmt(x, z):
                i = np.searchsorted(xe, x) - 1
                k = np.searchsorted(ze, z) - 1
                if 0 <= i < S.shape[1] and 0 <= k < S.shape[0]:
                    return f"x = {x:.3f} µm, z = {z:.3f} µm, value = {S[k, i]:.4g} {unit}"
                return f"x = {x:.3f} µm, z = {z:.3f} µm"
            ax.format_coord = fmt
        self.plot.plot(draw)
