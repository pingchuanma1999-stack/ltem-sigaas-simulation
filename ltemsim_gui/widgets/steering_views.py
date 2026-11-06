"""Drawing helpers of the pump-steering page: scopes, spot maps, peak curves, beam-path schematics, offset plan.

Pure matplotlib (no Qt widgets, no pyplot).  Every function draws into the Axes or Figure it is handed, in the
colours of the theme Tokens `t`, so the page calls them from PlotWidget.plot(...) and they re-run on theme changes.
None inputs draw a muted note instead of raising.

Conventions are those of ltemsim.steering and the vendored module README:
* lab frame on the sample, in um: origin at the tip apex (= aligned focus), +X toward the entrance-port wall
  ("right" seen from the open front of the AFM box), +Y toward the back panel ("front" = -Y), +Z up;
* spot arrays: row index = +y (back), column index = +x (port side), drawn with origin='lower'; grid points
  u = -W + a*2W/(M-1) around the chief-ray hit s['x0'], so the image extent is +-(W + half a pixel);
* contours (steering.half_max_contour) are segments in um relative to s['x0'];
* schematics are in mm.  Steering angles are micro-radians, so the schematics exaggerate them ("gain"):
  the top view traces the chief ray exactly with the travels multiplied by the gain, reduced automatically
  until the beam still passes the port and lands on the fold mirror and the illuminated part of the parabolic
  mirror (the label says which gain is shown); the inside and side views
  magnify only the angle of the beam arriving at the parabolic mirror, pivoting about the aligned chief point
  (as the reference page does), so the beam stays on the mirror while the spot shift grows by the gain.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.ticker import FuncFormatter

from ltemsim import steering

# (light, dark) colours of the reference page (ltem-steering-module/reference/ltem-steering.html)
_COLORS = {
    "lower": ("#a86400", "#c98500"),       # lower Z812, horizontal tilt (core: yaw)
    "upper": ("#088e86", "#199e9e"),       # upper Z812, vertical tilt (core: pitch)
    "beam": ("#cf1a3f", "#ff5272"),        # pump
    "thz": ("#5b4fd0", "#a59bff"),         # THz / collection
    "bias": ("#1f5fbf", "#6ea8ff"),        # simulation +x (bias field), as the applied-E arrow of device.py
    "mirror": ("#7d878d", "#8f9aa1"),      # flat mirrors, mount, walls
    "pm": ("#b08a2e", "#c9a54a"),          # gold parabolic mirror
}
FS = 8.0                                   # text size inside dense plots (pt)
FS_SMALL = 7.5
MAX_DISPLAY_ANGLE = 0.06                   # rad: largest exaggerated mirror tilt drawn (reference page)
_GAIN_LADDER = (400, 300, 200, 150, 100, 75, 50, 40, 30, 20, 15, 10, 7.5, 5, 4, 3, 2, 1.5, 1)
_A_MOUNT = 17.0                            # mm, KS1-Z8 corner offset from the plate centre (reference page)
_PORT_R = 13.0                             # mm, entrance-port opening radius (reference page)
_FOLD_HALF = 5.0                           # mm, half width of the 10 mm fold mirrors (reference page)
_DASH = (0, (5, 4))
_DASH_SHORT = (0, (3.5, 2.5))


# ----------------------------------------------------------------------------- colours and small helpers
@lru_cache(maxsize=8)
def _spot_cmap(surface: str, beam: str) -> ListedColormap:
    """Surface colour -> beam colour (gamma 0.8, as the reference heat map).  The lowest 8 % fade to transparent,
    so grid lines under a spot window stay visible; on the surface background this reads as surface -> beam."""
    v = np.linspace(0.0, 1.0, 256)
    g = v ** 0.8
    a, b = np.array(to_rgb(surface)), np.array(to_rgb(beam))
    rgb = a[None, :] * (1 - g[:, None]) + b[None, :] * g[:, None]
    return ListedColormap(np.column_stack([rgb, np.clip(v / 0.08, 0.0, 1.0)]), name="spot")


def palette(t) -> dict:
    """Theme-aware colours: lower, upper, beam, thz, tip, bias, ink, muted, grid (hex strings) and spot_cmap
    (plus mirror, pm and surface for the schematics)."""
    i = 1 if t.dark else 0
    pal = {k: v[i] for k, v in _COLORS.items()}
    pal.update(tip=t.text, ink=t.text, muted=t.muted, grid=t.border, surface=t.surface)
    pal["spot_cmap"] = _spot_cmap(t.surface, pal["beam"])
    return pal


def _signed(v: float, fmt: str = "{:g}") -> str:
    """'+5', '−10', '0' (typographic minus)."""
    if v == 0 or fmt.format(abs(v)) == fmt.format(0.0):
        return fmt.format(0.0)
    return ("+" if v > 0 else "−") + fmt.format(abs(v))


def _nice(v: float, minimum: float) -> float:
    """Smallest 'nice' full scale >= 1.25 v and >= minimum (mantissas 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8)."""
    x = max(1.25 * v, minimum)
    p = 10.0 ** math.floor(math.log10(x))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * p >= x * (1 - 1e-9):
            return m * p
    return 10 * p


def _frame(ax, t, ticks=False):
    """Thin full frame in the border colour (the app's rc hides the top and right spines)."""
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(t.border)
        s.set_linewidth(0.8)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])


def _note(ax, text: str, t):
    _frame(ax, t)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=t.muted, fontsize=9,
            wrap=True)


def _bg(t, alpha=0.75) -> dict:
    """Translucent surface backing for labels that may sit on a spot or a line."""
    return dict(boxstyle="round,pad=0.12", fc=t.surface, ec="none", alpha=alpha)


def _edge_labels(ax, t, top=None, bottom=None, left=None, right=None):
    """Direction words at the middle of each edge, beside the centre lines (as the reference scopes)."""
    kw = dict(xycoords="axes fraction", textcoords="offset points", fontsize=FS_SMALL, color=t.muted, zorder=9,
              bbox=_bg(t))
    if top:
        ax.annotate(top, (0.5, 1), xytext=(3, -3), ha="left", va="top", **kw)
    if bottom:                                            # left of the centre line: the right half holds the scale
        ax.annotate(bottom, (0.5, 0), xytext=(-3, 3), ha="right", va="bottom", **kw)
    if left:
        ax.annotate(left, (0, 0.5), xytext=(3, 2), ha="left", va="bottom", **kw)
    if right:
        ax.annotate(right, (1, 0.5), xytext=(-3, 2), ha="right", va="bottom", **kw)


def _arrow(ax, x0, y0, x1, y1, color, label=None, lw=1.6, fontsize=FS, zorder=6, head=9):
    """Arrow from (x0, y0) to (x1, y1), with an optional label just beyond the head (equal-aspect axes)."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=zorder,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=head))
    if label:
        a = math.atan2(y1 - y0, x1 - x0)
        c, s = math.cos(a), math.sin(a)
        ha = "left" if c > 0.3 else ("right" if c < -0.3 else "center")
        va = "bottom" if s > 0.3 else ("top" if s < -0.3 else "center")
        ax.annotate(label, (x1, y1), xytext=(5 * c, 5 * s), textcoords="offset points", ha=ha, va=va, color=color,
                    fontsize=fontsize, fontweight="bold", zorder=zorder + 1)


def _actuator_arrows(ax, vecs, rng, pal, t, avoid=()):
    """Direction arrows of both actuators from the origin, ~40 % of the full scale with their true length ratio.
    Each label goes beyond the head or beside the shaft, whichever box stays clearest of `avoid` ((x, y, r) in
    data units: the spot, the trail) and of the other arrow; returns the arrow directions (rad)."""
    tips = {}
    vmax = max(float(np.hypot(*v)) for v in vecs.values()) or 1.0
    for a, v in vecs.items():
        n = float(np.hypot(*v))
        if n > 0:
            tips[a] = np.asarray(v, float) / n * rng * max(0.2, 0.4 * n / vmax)
    upt = 2 * rng / max(_side_pt(ax), 1.0)                # data units per point (equal aspect)
    obst = list(avoid) + [(9 * upt, 8 * upt, 7 * upt)]   # + the "tip" label
    shafts = {a: [(w[0] * f, w[1] * f, 0.015 * rng) for f in (0.2, 0.4, 0.6, 0.8, 1.0)] for a, w in tips.items()}
    for a, label in (("upper", "+vert"), ("lower", "+horiz")):
        if a not in tips:
            continue
        v = tips[a]
        _arrow(ax, 0, 0, v[0], v[1], pal[a])
        u = v / np.hypot(*v)
        perp = np.array([-u[1], u[0]])
        bw, bh = (len(label) * 0.62 * FS + 3) * upt, 1.3 * FS * upt
        others = obst + [q for b, pts in shafts.items() if b != a for q in pts]

        def box(c):
            """Centre of the label box for anchor c[0] and direction c[1] (alignment as below)."""
            (bx, by), d = c
            cx = bx + 5 * d[0] * upt + (bw / 2 if d[0] > 0.3 else (-bw / 2 if d[0] < -0.3 else 0.0))
            cy = by + 5 * d[1] * upt + (bh / 2 if d[1] > 0.3 else (-bh / 2 if d[1] < -0.3 else 0.0))
            return cx, cy

        def clear(c):
            """Smallest gap between the label box and the obstacles or the frame (data units)."""
            cx, cy = box(c)
            gaps = [math.hypot(max(abs(ox - cx) - bw / 2, 0.0), max(abs(oy - cy) - bh / 2, 0.0)) - r
                    for ox, oy, r in others]
            return min(gaps + [rng - abs(cx) - bw / 2, rng - abs(cy) - bh / 2])
        cands = [(v, u)] + [(f * v, sg * perp) for f in (0.55, 0.3, 0.8) for sg in (1, -1)]
        best = cands[0] if clear(cands[0]) >= 0.04 * rng else max(cands, key=clear)
        (bx, by), d = best
        ha = "left" if d[0] > 0.3 else ("right" if d[0] < -0.3 else "center")
        va = "bottom" if d[1] > 0.3 else ("top" if d[1] < -0.3 else "center")
        ax.annotate(label, (bx, by), xytext=(5 * d[0], 5 * d[1]), textcoords="offset points", ha=ha, va=va,
                    color=pal[a], fontsize=FS, fontweight="bold", zorder=8, bbox=_bg(t, 0.7))
        cx, cy = box(best)
        obst.append((cx, cy, 0.5 * max(bw, bh)))
    return [math.atan2(v[1], v[0]) for v in tips.values()]


def _trail_obstacles(pts, rng, n=10):
    """A few trail points as label obstacles."""
    if len(pts) < 2:
        return []
    idx = np.unique(np.linspace(0, len(pts) - 1, min(n, len(pts))).astype(int))
    return [(pts[i, 0], pts[i, 1], 0.03 * rng) for i in idx]


def _side_pt(ax) -> float:
    """Side of a square equal-aspect axes in points (from the figure size, before the layout engine runs)."""
    pos = ax.get_position()
    return 72 * min(pos.width * ax.figure.get_figwidth(), pos.height * ax.figure.get_figheight())


def _scope(ax, rng: float, t):
    """Square scope of half-width rng: quarter grid, crosshair and the full-scale text (short in small scopes)."""
    ax.set_xlim(-rng, rng)
    ax.set_ylim(-rng, rng)
    ax.set_aspect("equal", adjustable="box")
    _frame(ax, t)
    for q in (-0.5, 0.5):
        ax.axvline(q * rng, color=t.border, lw=0.7, zorder=0)
        ax.axhline(q * rng, color=t.border, lw=0.7, zorder=0)
    ax.axvline(0, color=t.muted, lw=0.7, alpha=0.5, zorder=0)
    ax.axhline(0, color=t.muted, lw=0.7, alpha=0.5, zorder=0)
    text = f"full scale ±{rng:g} µm" if _side_pt(ax) >= 190 else f"±{rng:g} µm"
    ax.text(0.985, 0.015, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FS_SMALL, color=t.muted, zorder=9, bbox=_bg(t))


def _chain(segs) -> list:
    """Join marching-squares segments (n, 4) into polylines (lists of (x, y)), so dashed contours show dashes."""
    segs = np.asarray(segs, float).reshape(-1, 4)
    key = lambda x, y: (round(x, 7), round(y, 7))
    ends: dict = {}
    for i, (x1, y1, x2, y2) in enumerate(segs):
        ends.setdefault(key(x1, y1), []).append((i, 0))
        ends.setdefault(key(x2, y2), []).append((i, 1))
    used = np.zeros(len(segs), bool)

    def follow(x, y):
        pts = []
        while True:
            nxt = next(((j, e) for j, e in ends.get(key(x, y), ()) if not used[j]), None)
            if nxt is None:
                return pts
            j, e = nxt
            used[j] = True
            x, y = (segs[j][2], segs[j][3]) if e == 0 else (segs[j][0], segs[j][1])
            pts.append((x, y))

    lines = []
    for i in range(len(segs)):
        if not used[i]:
            used[i] = True
            x1, y1, x2, y2 = segs[i]
            lines.append(follow(x1, y1)[::-1] + [(x1, y1), (x2, y2)] + follow(x2, y2))
    return lines


def _contour(ax, s, cx, cy, color, lw=1.4, ls="-", alpha=1.0, zorder=4):
    """Half-maximum contour of spot s, its window centre drawn at (cx, cy)."""
    for line in _chain(steering.half_max_contour(s)):
        a = np.asarray(line)
        ax.plot(a[:, 0] + cx, a[:, 1] + cy, color=color, lw=lw, ls=ls, alpha=alpha, zorder=zorder,
                solid_capstyle="round")


def _heat(ax, s, cx, cy, pal, zorder=1):
    """Intensity of spot s relative to the aligned peak (I*k; vmin 0, vmax 1), window centre at (cx, cy)."""
    W = float(s["W"])
    half = W + W / (s["M"] - 1)                           # grid points sit on +-W: extend by half a pixel
    ax.imshow(np.asarray(s["I"], float) * float(s["k"]), origin="lower", cmap=pal["spot_cmap"], vmin=0.0, vmax=1.0,
              extent=(cx - half, cx + half, cy - half, cy + half), interpolation="bilinear", zorder=zorder)


def _line_label(ax, direction, dist, text, color, pad=3.0, fontsize=FS_SMALL):
    """Text along the line through the origin with unit `direction`, centred at `dist`, kept upright."""
    ang = math.degrees(math.atan2(direction[1], direction[0]))
    if ang > 90:
        ang -= 180
    elif ang <= -90:
        ang += 180
    ax.annotate(text, (direction[0] * dist, direction[1] * dist), xytext=(-pad * math.sin(math.radians(ang)),
                pad * math.cos(math.radians(ang))), textcoords="offset points", rotation=ang, rotation_mode="anchor",
                ha="center", va="bottom", color=color, fontsize=fontsize, zorder=8)


def _ang_gap(a: float, b: float) -> float:
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


# ----------------------------------------------------------------------------- scopes
def draw_port_scope(ax, st, trail_port, t) -> None:
    """Beam at the entrance port seen from outside facing the port: right = lab +y (back), up = +z."""
    pal = palette(t)
    if st is None:
        _note(ax, "No chief-ray solution\n(the beam misses the parabolic mirror)", t)
        return
    pts = np.array([q for q in (trail_port or []) if q is not None], float).reshape(-1, 2)
    cur = np.asarray(st.port_um, float)
    reach = max([1.0, float(np.hypot(*cur))] + [float(np.hypot(*q)) for q in pts])
    rng = _nice(reach, 10.0)
    _scope(ax, rng, t)
    if len(pts) > 1:
        ax.plot(pts[:, 0], pts[:, 1], color=pal["beam"], alpha=0.35, lw=1.3, zorder=2)
    vecs = {a: np.asarray(st.port_per_um[a], float) for a in ("lower", "upper")}
    _actuator_arrows(ax, vecs, rng, pal, t, [(cur[0], cur[1], 0.1 * rng)] + _trail_obstacles(pts, rng))
    ax.plot(*cur, "o", ms=5.5, color=pal["beam"], zorder=8)
    ax.plot(*cur, "o", ms=13, mfc="none", mec=pal["beam"], mew=1.1, zorder=8)
    _edge_labels(ax, t, top="up", left="front", right="back")


def draw_sample_scope(ax, st, s, ref, trail_spot, sp, t) -> None:
    """Spot on the sample, top view: right = lab +X (port side), up = +Y (back); the tip at the origin."""
    pal = palette(t)
    if st is None:
        _note(ax, "No chief-ray solution\n(the beam misses the parabolic mirror)", t)
        return
    pts = np.array([q for q in (trail_spot or []) if q is not None], float).reshape(-1, 2)
    cur = np.asarray(st.spot_um, float)
    ext = 0.6 * s["fwhmL"] if s is not None else 1.5
    reach = max([0.5, float(np.hypot(*cur)) + ext] + [float(np.hypot(*q)) for q in pts])
    rng = _nice(reach, 3.0)
    _scope(ax, rng, t)
    L = 1.5 * rng

    # lines through the tip: plane of incidence (dashed) and the simulation's bias axis
    ph = np.asarray(st.incidence_plane, float)
    ax.plot([-ph[0] * L, ph[0] * L], [-ph[1] * L, ph[1] * L], ls=_DASH, color=t.muted, lw=1.0, alpha=0.8, zorder=1)
    phi = math.radians(sp.bias_axis_deg) if sp is not None else 0.0
    ub = (math.cos(phi), math.sin(phi))
    ax.plot([-ub[0] * L, ub[0] * L], [-ub[1] * L, ub[1] * L], color=pal["bias"], lw=0.9, alpha=0.75, zorder=1)

    # original spot (dashed, under the tip) and the current one with its intensity
    if ref is not None:
        _contour(ax, ref, ref["x0"][0] * 1e3, ref["x0"][1] * 1e3, t.text, lw=1.1, ls=_DASH_SHORT, alpha=0.6)
    if s is not None and ref is not None:
        ox, oy = steering.spot_offset_um(s, ref)
        cx, cy = ref["x0"][0] * 1e3 + ox, ref["x0"][1] * 1e3 + oy
        _heat(ax, s, cx, cy, pal)
        _contour(ax, s, cx, cy, pal["beam"], lw=1.5)
    if len(pts) > 1:
        ax.plot(pts[:, 0], pts[:, 1], color=pal["beam"], alpha=0.4, lw=1.1, zorder=3)
    ax.plot(*cur, "o", ms=3.5, color=pal["beam"], mec=t.surface, mew=0.8, zorder=8)

    # actuator directions
    vecs = {a: np.asarray(st.per_um[a], float) for a in ("lower", "upper")}
    r_spot = (0.45 * float(s["fwhmL"]) if s is not None else 0.0) + 0.02 * rng
    dirs = _actuator_arrows(ax, vecs, rng, pal, t, [(cur[0], cur[1], r_spot)] + _trail_obstacles(pts, rng))

    # labels of the two lines, each on the end that is farthest from the arrows
    def free_end(u):
        cands = (np.asarray(u, float), -np.asarray(u, float))
        return max(cands, key=lambda c: min([_ang_gap(math.atan2(c[1], c[0]), d) for d in dirs] or [1.0]))
    e = free_end(ph)
    _line_label(ax, e, 0.6 * rng, "plane of incidence", t.muted)
    bx, by = ub[0] * 0.97 * rng, ub[1] * 0.97 * rng
    ax.annotate("+x (bias)", (bx, by), xytext=(-2 * ub[0] + 3 * ub[1], -2 * ub[1] - 3 * ub[0]),
                textcoords="offset points", ha="right" if ub[0] > 0.3 else ("left" if ub[0] < -0.3 else "center"),
                va="top" if ub[0] >= -0.3 else "bottom", color=pal["bias"], fontsize=FS_SMALL, zorder=8, bbox=_bg(t))

    # the tip
    ax.plot(0, 0, marker="v", ms=6.5, color=pal["tip"], mec=t.surface, mew=0.6, zorder=9)
    ax.annotate("tip", (0, 0), xytext=(5, 4), textcoords="offset points", ha="left", va="bottom",
                color=pal["tip"], fontsize=FS_SMALL, zorder=9)
    _edge_labels(ax, t, top="back", bottom="front", left="left", right="right (port)")


def draw_spot_map(ax, s, ref, st, t, half_width_um=None) -> None:
    """Magnified spot in window coordinates (um from its chief-ray hit), intensity relative to the aligned peak."""
    pal = palette(t)
    if s is None:
        _note(ax, "No spot: fewer than 20 rays\nreach the sample", t)
        return
    hw = float(half_width_um) if half_width_um else min(max(3.0, 1.3 * float(s["fwhmL"])), float(s["W"]))
    ax.set_xlim(-hw, hw)
    ax.set_ylim(-hw, hw)
    ax.set_aspect("equal", adjustable="box")
    _frame(ax, t)
    step = 1.0 if hw <= 6 else (2.0 if hw <= 12 else 5.0)
    for u in np.arange(-math.floor(hw / step) * step, hw + 1e-9, step):
        ax.axvline(u, color=t.border, lw=0.6, zorder=0)
        ax.axhline(u, color=t.border, lw=0.6, zorder=0)
    if st is not None:
        ph = np.asarray(st.incidence_plane, float)
        ax.plot([-ph[0] * 2 * hw, ph[0] * 2 * hw], [-ph[1] * 2 * hw, ph[1] * 2 * hw], ls=_DASH, color=t.muted,
                lw=1.0, alpha=0.7, zorder=2)
    _heat(ax, s, 0.0, 0.0, pal)
    if ref is not None:
        _contour(ax, ref, 0.0, 0.0, t.text, lw=1.2, ls=_DASH_SHORT, alpha=0.6)
    _contour(ax, s, 0.0, 0.0, pal["beam"], lw=1.7)
    ax.plot(s["peakAt"][0], s["peakAt"][1], marker="+", ms=9, mew=1.3, color=t.text, zorder=7)
    # 1 um scale bar, bottom left
    x0, y0 = -hw + 0.07 * 2 * hw, -hw + 0.06 * 2 * hw
    ax.plot([x0, x0 + 1.0], [y0, y0], color=t.text, lw=2.0, solid_capstyle="butt", zorder=8)
    ax.annotate("1 µm", (x0, y0), xytext=(0, 3), textcoords="offset points", ha="left", va="bottom",
                fontsize=FS_SMALL, color=t.muted, zorder=8)
    ax.text(0.985, 0.015, f"±{hw:.2g} µm", transform=ax.transAxes, ha="right", va="bottom", fontsize=FS_SMALL,
            color=t.muted, zorder=9, bbox=_bg(t))
    _edge_labels(ax, t, top="back", bottom="front", left="left", right="right")


# ----------------------------------------------------------------------------- spot shape vs travel
def draw_peak_curves(ax, curves, current, t) -> None:
    """Peak intensity (aligned spot = 100 %) vs push-rod travel, each actuator moving alone."""
    pal = palette(t)
    curves = curves or {}
    if not any(curves.get(a) for a in ("lower", "upper")):
        _note(ax, "Peak-intensity curves not computed yet", t)
        return
    ax.grid(True, axis="y", color=t.border, lw=0.6)
    ax.set_axisbelow(True)
    xs_all = []
    for a, ls, name in (("lower", "-", "Lower · horizontal"), ("upper", (0, (6, 3.5)), "Upper · vertical")):
        rows = sorted(curves.get(a) or [], key=lambda r: r["travel_um"])
        if not rows:
            continue
        x = np.array([r["travel_um"] for r in rows], float)
        y = np.array([np.nan if r.get("rel") is None else 100.0 * r["rel"] for r in rows], float)
        xs_all.extend(x)
        ax.plot(x, y, color=pal[a], lw=1.9, ls=ls, label=name, solid_capstyle="round", dash_capstyle="round")
        tc = (current or {}).get(a)
        ok = np.isfinite(y)
        if tc is not None and ok.sum() >= 1 and x[ok].min() <= tc <= x[ok].max():
            yc = float(np.interp(tc, x[ok], y[ok])) if ok.sum() > 1 else float(y[ok][0])
            ax.plot(tc, yc, "o", ms=7.5, color=pal[a], mec=t.surface, mew=1.6, zorder=6)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f} %"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _signed(v)))
    if xs_all:
        lo, hi = min(xs_all), max(xs_all)
        pad = 0.02 * (hi - lo or 1.0)
        ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("push-rod travel (µm)", fontsize=FS + 0.5)
    # the longest y label that fits the axes height (the figure is short when the tab is stacked)
    h_pt = 72 * ax.figure.get_figheight() - 3.5 * FS - 6
    ylab = next((s for s in ("peak intensity (aligned = 100 %)", "peak (aligned = 100 %)")
                 if 0.5 * (FS + 0.5) * len(s) <= h_pt), "aligned = 100 %")
    ax.set_ylabel(ylab, fontsize=FS + 0.5)
    ax.tick_params(labelsize=FS)
    ax.legend(loc="upper left", fontsize=FS, frameon=False, handlelength=2.6, borderaxespad=0.3)


def draw_shape_gallery(fig, gallery, t, box_um=10.0) -> None:
    """2 x n thumbnails of the spot (rows: lower / upper actuator; columns: travel), each +-box_um around its
    chief-ray hit, intensity relative to the aligned peak."""
    pal = palette(t)
    gallery = gallery or {}
    rows = [(a, list(gallery.get(a) or [])) for a in ("lower", "upper")]
    n = max(len(r) for _, r in rows)
    if n == 0:
        _note(fig.add_subplot(), "Spot gallery not computed yet", t)
        return
    same = [tr for tr, _ in rows[0][1]] == [tr for tr, _ in rows[1][1]]
    gs = fig.add_gridspec(2, n, wspace=0.06, hspace=0.12)
    for r, (a, items) in enumerate(rows):
        for c in range(n):
            ax = fig.add_subplot(gs[r, c])
            ax.set_xlim(-box_um, box_um)
            ax.set_ylim(-box_um, box_um)
            ax.set_aspect("equal", adjustable="box")
            ax.set_anchor("S" if r == 0 else "N")         # keep the two rows together when the figure is tall
            _frame(ax, t)
            ax.axvline(0, color=t.border, lw=0.6, zorder=0)
            ax.axhline(0, color=t.border, lw=0.6, zorder=0)
            if c == 0:                                    # horizontal: the thumbnails can be shorter than the word
                ax.set_ylabel("Horiz." if a == "lower" else "Vert.", color=pal[a], fontsize=FS + 0.5,
                              fontweight="bold", labelpad=4, rotation=0, ha="right", va="center")
            if c >= len(items):
                ax.set_axis_off()
                continue
            travel, s = items[c]
            if r == 0 or not same:
                ax.set_title(_signed(float(travel)), fontsize=FS, color=t.muted, pad=2)
            if s is None:
                ax.text(0.5, 0.5, "no spot", transform=ax.transAxes, ha="center", va="center", fontsize=FS_SMALL,
                        color=t.muted)
                continue
            _heat(ax, s, 0.0, 0.0, pal)
            _contour(ax, s, 0.0, 0.0, pal["beam"], lw=1.0)
            ax.text(0.96, 0.03, f"{100 * s['rel']:.0f}%", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color=t.muted)
    sep = " · " if fig.get_figwidth() >= 5.2 else "\n"     # two lines in narrow figures
    fig.supxlabel(f"travel (µm) per column · frames ±{box_um:g} µm{sep}% = peak vs aligned", fontsize=FS_SMALL,
                  color=t.muted, linespacing=1.2)


# ----------------------------------------------------------------------------- schematics: geometry
def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _reflect(d, n):
    return d - 2.0 * float(np.dot(d, n)) * n


def _port_point(c, p):
    """Where the chief ray c crosses the entrance-port plane x = portX (mm)."""
    S, d1 = c["g"]["S"], c["d1"]
    return S + d1 * (p["portX"] - S[0]) / d1[0]


def _aperture(p, g):
    """(half-angle of the illumination cone, unit vector focus -> aligned chief point)."""
    return math.asin(min(0.95, p["NA"])), _unit(g["P"])


def _on_mirror(c, p, g, c0) -> bool:
    """Chief ray c passes the entrance port, hits the 10 mm fold mirror and lands on the illuminated part of the
    parabolic mirror (inside the NA cone, above zLow)."""
    if "Q" not in c or "Q" not in c0:
        return False
    if np.linalg.norm(_port_point(c, p) - _port_point(c0, p)) > _PORT_R:
        return False
    if np.linalg.norm(np.asarray(c["A"]) - g["F1"]) > _FOLD_HALF:
        return False
    apA, p0 = _aperture(p, g)
    P = np.asarray(c["P"], float)
    return P[2] >= p["zLow"] and float(np.dot(_unit(P), p0)) >= math.cos(apA) - 1e-12


def _walk_gain(p, lower_um, upper_um, gain):
    """Largest gain <= `gain` whose exaggerated exact trace stays on the mirrors (see _on_mirror): (gain, trace).
    Falls back to the true trace (gain 1)."""
    gain = max(1.0, float(gain or 1))
    O = steering.optics
    g = O.geometry(p)
    c0 = O.trace_chief(p, 0.0, 0.0)
    for N in [gain] + [v for v in _GAIN_LADDER if v < gain]:
        c = O.trace_chief(p, upper_um * N / 1000, lower_um * N / 1000)
        if _on_mirror(c, p, g, c0):
            return N, c
    return 1.0, O.trace_chief(p, upper_um / 1000, lower_um / 1000)


def _display_gain(p, lower_um, upper_um, gain) -> float:
    """Angle gain of the inside/side views, capped like the reference page (tilt x gain <= 0.06 rad)."""
    gain = max(1.0, float(gain or 1))
    big = max(abs(lower_um), abs(upper_um)) / 1000 / p["L"]
    if big <= 0 or gain * big <= MAX_DISPLAY_ANGLE:
        return gain
    return next((v for v in _GAIN_LADDER if v * big <= MAX_DISPLAY_ANGLE), 1.0)   # a round number


def _magnified(p, lower_um, upper_um, N):
    """(aligned trace c0, true trace c, magnified ray): the magnified ray enters the parabolic mirror with the true
    change of direction multiplied by N, pivoting about the aligned chief point (None when it or c misses).
    Exact trace through the core: the magnified ray is unfolded back through the fold mirror and handed to
    optics.trace_after_steer."""
    O = steering.optics
    c0 = O.trace_chief(p, 0.0, 0.0)
    c = O.trace_chief(p, upper_um / 1000, lower_um / 1000)
    if "Q" not in c0 or "Q" not in c:
        return c0, c, None
    g = c0["g"]
    d = _unit(c0["d2"] + (c["d2"] - c0["d2"]) * N)
    C, nF = np.asarray(c0["P"], float), g["nF1"]
    den = float(np.dot(nF, d))
    if abs(den) < 1e-12:
        return c0, c, None
    A = C - d * float(np.dot(nF, C - g["F1"])) / den      # where the magnified ray crosses the fold plane
    d_in = _reflect(d, nF)                                # direction before the fold (reflection is an involution)
    return c0, c, O.trace_after_steer(A - 20.0 * d_in, d_in, g)


def _pm_outline(g, p, side=+1, n=48):
    """Top-view outline (x, y) of one half of the semi-paraboloid as the reference page builds it: azimuth
    psi = 0..80 deg from +Z toward side*X, radius from zLow/cos(psi) to |chief point| + 6.5 mm."""
    fp, zl = g["fp"], p["zLow"]
    rho1 = math.hypot(g["x0"], g["h"]) + 6.5
    psi = np.linspace(0.0, math.radians(80), n)
    rmin = np.maximum(zl, zl / np.maximum(np.cos(psi), 0.06))
    y = lambda rho: rho ** 2 / (4 * fp) - fp
    inner = np.column_stack([side * rmin * np.sin(psi), y(rmin)])
    outer = np.column_stack([side * rho1 * np.sin(psi[::-1]), np.full(n, y(rho1))])
    rs = np.linspace(rmin[-1], rho1, 12)
    flank = np.column_stack([side * rs * math.sin(psi[-1]), y(rs)])
    return np.vstack([inner, flank, outer]), rho1


def _pm_point(u, fp):
    """Points of the paraboloid seen from the focus along unit directions u (polar form r = 2 fp / (1 - u_y))."""
    u = np.atleast_2d(np.asarray(u, float))
    return u * (2 * fp / (1 - u[:, 1]))[:, None]


def _footprint(g, p, n=72):
    """Illuminated area on the mirror: the NA cone around the aligned chief ray, cut at the lower edge zLow (3-D)."""
    apA, p0 = _aperture(p, g)
    e1 = _unit(np.cross(p0, [0.0, 0.0, 1.0]))
    e2 = np.cross(e1, p0)
    phi = np.linspace(0, 2 * math.pi, n)
    ring = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2
    lo, hi = np.zeros(n), np.full(n, apA)
    pts = _pm_point(math.cos(apA) * p0 + math.sin(apA) * ring, g["fp"])
    low = pts[:, 2] < p["zLow"]
    if low.any():                                         # bisect the cone angle down to the lower edge
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            q = _pm_point(np.cos(mid)[:, None] * p0 + np.sin(mid)[:, None] * ring, g["fp"])
            above = q[:, 2] >= p["zLow"]
            lo = np.where(above, mid, lo)
            hi = np.where(above, hi, mid)
        q = _pm_point(np.cos(lo)[:, None] * p0 + np.sin(lo)[:, None] * ring, g["fp"])
        pts = np.where(low[:, None], q, pts)
    return pts


def _side_name(v) -> str:
    """'front', 'back-left', ... for a horizontal lab direction."""
    parts = []
    if v[1] < -0.38:
        parts.append("front")
    elif v[1] > 0.38:
        parts.append("back")
    if v[0] > 0.38:
        parts.append("right")
    elif v[0] < -0.38:
        parts.append("left")
    return "-".join(parts)


def _fold(ax, F, n, pal, z=4, color=None, alpha=1.0):
    """Vertical flat mirror (top view): face through F along n x Z, back plate behind it."""
    w = np.array([-n[1], n[0]])
    a, b = F[:2] - _FOLD_HALF * w, F[:2] + _FOLD_HALF * w
    back = -np.asarray(n[:2]) * 1.4
    ax.add_patch(Polygon([a, b, b + back, a + back], closed=True, fc=pal["mirror"], ec="none", alpha=0.45 * alpha,
                         zorder=z - 1))
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color or pal["mirror"], lw=2.4, alpha=alpha, solid_capstyle="butt",
            zorder=z)


def _scale_bar(ax, length, label, t, loc=(0.04, 0.05)):
    """Horizontal scale bar in data units at axes fraction `loc`."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x, y = x0 + loc[0] * (x1 - x0), y0 + loc[1] * (y1 - y0)
    ax.plot([x, x + length], [y, y], color=t.text, lw=2.0, solid_capstyle="butt", zorder=9)
    ax.annotate(label, (x, y), xytext=(0, 3), textcoords="offset points", ha="left", va="bottom",
                fontsize=FS_SMALL, color=t.muted, zorder=9)


def _tip_mark(ax, pal, t, x=0.0, y=0.0, label="tip", ms=6.5, offset=(5, 3), ha="left", va="bottom"):
    ax.plot(x, y, marker="v", ms=ms, color=pal["tip"], mec=t.surface, mew=0.6, zorder=9)
    if label:
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", ha=ha, va=va, color=pal["tip"],
                    fontsize=FS_SMALL, zorder=9, bbox=_bg(t))


def _inside_top(ax, p, g, pal, t, aligned, deflected, walk):
    """Top view of the box interior near the tip: parabolic mirror (right half pumps, left half collects THz),
    fold mirrors, illuminated area and the pump chief ray.  aligned / deflected: traces with A, P, Q (or None).
    walk=True: the deflected trace is a different path all along (top view); False: only its direction after
    the parabolic mirror differs (magnified angle), the path up to the mirror is the true one = aligned."""
    out, rho1 = _pm_outline(g, p, +1)
    left, _ = _pm_outline(g, p, -1)
    ax.add_patch(Polygon(out, closed=True, fc=pal["pm"], ec="none", alpha=0.28, zorder=1))
    ax.add_patch(Polygon(out, closed=True, fc="none", ec=pal["pm"], lw=1.0, zorder=2))
    ax.add_patch(Polygon(left, closed=True, fc=pal["thz"], ec="none", alpha=0.10, zorder=1))
    ax.add_patch(Polygon(left, closed=True, fc="none", ec=pal["thz"], lw=0.8, alpha=0.7, zorder=2))
    yb = -g["fp"] - 6.4
    ax.add_patch(Rectangle((-rho1 - 1, yb), 2 * rho1 + 2, 6, fc=pal["mirror"], ec="none", alpha=0.22, zorder=0))
    ax.text(0, yb + 3, "parabolic mirror", ha="center", va="center", fontsize=FS_SMALL, color=pal["pm"],
            fontweight="bold", zorder=9)
    fpt = _footprint(g, p)
    ax.add_patch(Polygon(fpt[:, :2], closed=True, fc=pal["beam"], ec="none", alpha=0.16, zorder=2))
    ax.plot(fpt[:, 0], fpt[:, 1], color=pal["beam"], lw=0.7, alpha=0.55, zorder=2)
    # THz collection: focus -> left half -> collection fold (normal (-1, -1, 0)) -> out of the left wall
    x_lo, x_hi = ax.get_xlim()
    _fold(ax, np.array([-g["x0"], g["yf"], g["h"]]), _unit([-1.0, -1.0, 0.0]), pal, color=pal["thz"], alpha=0.8)
    ax.plot([0, -g["P"][0], -g["x0"], x_lo], [0, g["P"][1], g["yf"], g["yf"]], color=pal["thz"], lw=1.0,
            alpha=0.7, zorder=3)
    ax.annotate("THz out", (x_lo, g["yf"]), xytext=(3, -2), textcoords="offset points", ha="left", va="top",
                fontsize=FS_SMALL, color=pal["thz"], zorder=9)
    _fold(ax, g["F1"], g["nF1"], pal)
    ax.annotate("fold mirror", (g["F1"][0], g["F1"][1] + _FOLD_HALF * 0.75), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=FS_SMALL, color=t.muted, zorder=9,
                bbox=_bg(t))
    # pump chief ray
    if aligned is None:
        return
    A0, P0, Q0 = aligned["A"], aligned["P"], aligned["Q"]
    faint = dict(color=pal["beam"], lw=1.2, alpha=0.3, zorder=4)
    bold = dict(color=pal["beam"], lw=2.0, zorder=5, solid_joinstyle="round", solid_capstyle="round")
    if deflected is None:
        ax.plot([x_hi, A0[0], P0[0], Q0[0]], [g["yf"], A0[1], P0[1], Q0[1]], **bold)
    elif walk:
        A, P, Q = deflected["A"], deflected["P"], deflected["Q"]
        y_in = A[1] + (x_hi - A[0]) * deflected.get("slope_in", 0.0)
        ax.plot([x_hi, A0[0], P0[0], Q0[0]], [g["yf"], A0[1], P0[1], Q0[1]], **faint)
        ax.plot([x_hi, A[0], P[0], Q[0]], [y_in, A[1], P[1], Q[1]], **bold)
    else:
        ax.plot([x_hi, A0[0], P0[0]], [g["yf"], A0[1], P0[1]], color=pal["beam"], lw=1.6, zorder=5)
        ax.plot([P0[0], Q0[0]], [P0[1], Q0[1]], **faint)
        P, Q = deflected["P"], deflected["Q"]
        ax.plot([P[0], Q[0]], [P[1], Q[1]], **bold)


def _top_layout(fig, xl, xr, yf, need_l, need_r, share=0.42):
    """y ranges of the two top-view panels: same pixel height, beam height yf at the same fraction of both,
    widths shared share : 1 - share, and at least need_* = (mm below yf, mm above yf) visible in each."""
    W_in = max(fig.get_figwidth() - 0.4, 1.0)
    H_in = max(fig.get_figheight() - 0.5, 0.5)
    Hl = (xl[1] - xl[0]) * H_in / (share * W_in)          # mm shown over the panel height
    Hr = (xr[1] - xr[0]) * H_in / ((1 - share) * W_in)
    lo = max(need_l[0] / Hl, need_r[0] / Hr)
    hi = max(need_l[1] / Hl, need_r[1] / Hr)
    if lo + hi > 1:                                       # too flat a figure: show more, panels get narrower
        k = lo + hi
        Hl, Hr, lo, hi = Hl * k, Hr * k, lo / k, hi / k
    f = lo + 0.5 * (1 - lo - hi)
    return (yf - f * Hl, yf + (1 - f) * Hl), (yf - f * Hr, yf + (1 - f) * Hr)


def _view_top(fig, p, g, lower_um, upper_um, gain, pal, t):
    """Whole path seen from above on a broken x axis: box interior near the tip | port wall and steering mirror.
    The travels are multiplied by the largest gain <= `gain` that keeps the exact chief ray on the fold mirror,
    the port and the illuminated part of the parabolic mirror."""
    N, c = _walk_gain(p, lower_um, upper_um, gain)
    c0 = steering.optics.trace_chief(p, 0.0, 0.0)
    moved = bool(lower_um or upper_um)
    miss = "Q" not in c
    S, yf, n0, w = g["S"], g["yf"], g["n0"], g["uBack"]
    # outside: the two Z812 bodies stick out 18-130 mm behind the plate, +-17 mm beside the mirror centre.  uBack is the
    # viewer's right seen from the actuator side: the lower actuator sits on the yawSide of the pivot, the upper one
    # level with the pivot (as in the mirror view)
    ys = 1 if p["yawSide"] >= 0 else -1
    bodies = {}
    for name, lat in (("lower", +ys * _A_MOUNT), ("upper", -ys * _A_MOUNT)):
        c_ = S[:2] + lat * w[:2]
        b0, b1 = c_ - 18 * n0[:2], c_ - 130 * n0[:2]
        bodies[name] = np.array([b0 - 8 * w[:2], b0 + 8 * w[:2], b1 + 8 * w[:2], b1 - 8 * w[:2]])
    allb = np.vstack(list(bodies.values()))
    rho1 = math.hypot(g["x0"], g["h"]) + 6.5
    xl = (-(rho1 + 3), g["x0"] + 13)
    xr = (p["portX"] - 24, max(S[0] + 34, allb[:, 0].max() + 6))
    yl, yr = _top_layout(fig, xl, xr, yf, (yf + g["fp"] + 8, 12.0), (yf - allb[:, 1].min() + 6, 60.0))
    ar_l = (xl[1] - xl[0]) / (yl[1] - yl[0])
    ar_r = (xr[1] - xr[0]) / (yr[1] - yr[0])
    gs = fig.add_gridspec(1, 2, width_ratios=[ar_l, ar_r], wspace=0.03)
    axl, axr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for ax, (x, y) in ((axl, (xl, yl)), (axr, (xr, yr))):
        ax.set_xlim(*x)
        ax.set_ylim(*y)
        ax.set_aspect("equal", adjustable="box")
        _frame(ax, t)
    inside = dict(fc=t.border, ec="none", alpha=0.22, zorder=-1)
    axl.add_patch(Rectangle((xl[0], yl[0]), xl[1] - xl[0], yl[1] - yl[0], **inside))
    axr.add_patch(Rectangle((xr[0], yr[0]), p["portX"] - xr[0], yr[1] - yr[0], **inside))
    axl.spines["right"].set_visible(False)               # broken axis
    axr.spines["left"].set_visible(False)
    for ax, x, k in ((axl, 1.0, 1.0), (axr, 0.0, ar_l / ar_r)):
        for y in (0.0, 1.0):
            ax.plot([x - 0.012 * k, x + 0.012 * k], [y - 0.024, y + 0.024], transform=ax.transAxes, color=t.muted,
                    lw=0.9, clip_on=False, zorder=10)

    # ---- left: inside the box near the tip
    aligned = {k: c0[k] for k in ("A", "P", "Q")} if "Q" in c0 else None
    defl = None
    if moved and not miss:
        defl = {k: c[k] for k in ("A", "P", "Q")}
        defl["slope_in"] = c["d1"][1] / c["d1"][0]
    _inside_top(axl, p, g, pal, t, aligned, defl, walk=True)
    _tip_mark(axl, pal, t, label="tip", offset=(-4, 3), ha="right")
    axl.set_title("inside the AFM box", fontsize=FS, color=t.muted, loc="left", pad=3)
    _scale_bar(axl, 10, "10 mm", t, loc=(0.05, 0.9))

    # ---- right: port wall, steering mirror and its mount
    wall = dict(fc=pal["mirror"], ec="none", alpha=0.75, zorder=3)
    big = 1000.0
    axr.add_patch(Rectangle((p["portX"], yf + _PORT_R), 8, big, **wall))
    axr.add_patch(Rectangle((p["portX"], yf - _PORT_R - big), 8, big, **wall))
    for sg in (+1, -1):
        axr.plot([p["portX"] - 2, p["portX"] + 10], [yf + sg * _PORT_R] * 2, color=t.text, lw=1.2, zorder=4)
    axr.annotate(f"entrance-port wall, x = {p['portX']:g} mm", (p["portX"] + 8, yr[0]), xytext=(2, 4),
                 textcoords="offset points", ha="left", va="bottom", rotation=90, fontsize=FS_SMALL, color=t.muted,
                 zorder=9)
    for name, poly in bodies.items():
        axr.add_patch(Polygon(poly, closed=True, fc=pal[name], ec="none", alpha=0.55, zorder=2))
        mid, out = poly.mean(axis=0), (ys * w[:2] if name == "lower" else -ys * w[:2])
        ang = math.degrees(math.atan2(-n0[1], -n0[0]))
        ang = ang - 180 if ang > 90 else (ang + 180 if ang <= -90 else ang)
        axr.annotate(f"{name} Z812", mid, xytext=(out[0] * 11, out[1] * 11), textcoords="offset points",
                     ha="center", va="center", rotation=ang, rotation_mode="anchor", fontsize=FS_SMALL,
                     color=pal[name], fontweight="bold", zorder=9)
    half = _A_MOUNT + 4
    plate = [S[:2] + half * w[:2] - n0[:2], S[:2] - half * w[:2] - n0[:2],
             S[:2] - half * w[:2] - 7 * n0[:2], S[:2] + half * w[:2] - 7 * n0[:2]]
    axr.add_patch(Polygon(plate, closed=True, fc=pal["mirror"], ec="none", alpha=0.9, zorder=3))
    m0, m1 = S[:2] - 12.7 * w[:2], S[:2] + 12.7 * w[:2]
    axr.plot([m0[0], m1[0]], [m0[1], m1[1]], color=t.text, lw=2.2, solid_capstyle="butt", zorder=5)
    axr.annotate("steering mirror\n(KS1-Z8)", (S[0], S[1]), xytext=(-12, -7), textcoords="offset points",
                 ha="right", va="top", fontsize=FS_SMALL, color=t.text, zorder=9, linespacing=1.1)
    # beams: input (travels -Y onto S), aligned and deflected toward the port
    axr.plot([S[0], S[0]], [yr[1], S[1]], color=pal["beam"], lw=2.0, zorder=5)
    ym = yr[1] - 0.18 * (yr[1] - yf)
    _arrow(axr, S[0], ym + 1, S[0], ym, pal["beam"], lw=0, head=10)
    axr.annotate("pump in", (S[0], yr[1]), xytext=(-4, -3), textcoords="offset points", ha="right", va="top",
                 fontsize=FS_SMALL, color=pal["beam"], zorder=9)
    axr.plot([S[0], xr[0]], [yf, yf], color=pal["beam"], lw=1.2 if defl else 2.0, alpha=0.3 if defl else 1.0,
             zorder=4)
    if defl is not None:
        d1 = c["d1"]
        axr.plot([S[0], xr[0]], [S[1], S[1] + d1[1] * (xr[0] - S[0]) / d1[0]], color=pal["beam"], lw=2.0,
                 zorder=5)
    yd = yf + 0.5 * (yr[1] - yf)
    axr.annotate("", xy=(p["portX"] + 8, yd), xytext=(S[0], yd), zorder=6,
                 arrowprops=dict(arrowstyle="<->", color=t.muted, lw=0.9, shrinkA=0, shrinkB=0))
    axr.annotate(f"dist = {p['dist']:g} mm", (0.5 * (p["portX"] + 8 + S[0]), yd), xytext=(0, 2),
                 textcoords="offset points", ha="center", va="bottom", fontsize=FS_SMALL, color=t.muted, zorder=9)
    axr.set_title("port wall and steering mirror", fontsize=FS, color=t.muted, loc="left", pad=3,
                  x=0.04)                                 # clear of the axis-break marks
    bar = 100.0 if (xr[1] - xr[0]) > 250 else 50.0
    _scale_bar(axr, bar, f"{bar:g} mm", t, loc=(0.16, 0.88))
    if moved and miss:
        axr.text(0.98, 0.03, "beam misses the parabolic mirror at this gain", transform=axr.transAxes, ha="right",
                 va="bottom", fontsize=FS_SMALL, color=t.danger, zorder=9)
    elif moved:
        msg = f"beam walk ×{N:g}" + (f"\n(×{float(gain):g} leaves the mirrors)" if N < float(gain or 1) else "")
        axr.text(0.98, 0.03, msg, transform=axr.transAxes, ha="right", va="bottom", fontsize=FS_SMALL,
                 color=t.muted, zorder=9, linespacing=1.15)


def _view_inside(fig, p, g, lower_um, upper_um, gain, pal, t):
    """Top view of the box interior, about +-25 mm around the tip, with the spot shift magnified in an inset."""
    N = _display_gain(p, lower_um, upper_um, gain)
    c0, c, mag = _magnified(p, lower_um, upper_um, N)
    moved = bool(lower_um or upper_um)
    ax = fig.add_subplot()
    ax.set_xlim(-25, 27)
    ax.set_ylim(-27, 29)
    ax.set_aspect("equal", adjustable="box")
    _frame(ax, t)
    aligned = {k: c0[k] for k in ("A", "P", "Q")} if "Q" in c0 else None
    defl = {k: mag[k] for k in ("A", "P", "Q")} if (mag is not None and moved) else None
    _inside_top(ax, p, g, pal, t, aligned, defl, walk=False)
    _tip_mark(ax, pal, t, label="tip / focus", offset=(-5, -4), ha="right", va="top")
    ax.annotate("pump from port", (27, g["yf"]), xytext=(-3, 3), textcoords="offset points", ha="right",
                va="bottom", fontsize=FS_SMALL, color=pal["beam"], zorder=9)
    _edge_labels(ax, t, top="back", bottom="front")
    _scale_bar(ax, 10, "10 mm", t, loc=(0.04, 0.04))
    if moved and mag is None:
        ax.text(0.98, 0.98, "beam misses the parabolic mirror at this gain", transform=ax.transAxes, ha="right",
                va="top", fontsize=FS_SMALL, color=t.danger, zorder=9)
        return
    # inset: the focus region, spot shift magnified like the angles
    ins = ax.inset_axes([0.02, 0.67, 0.31, 0.31])
    Q = np.zeros(2) if defl is None else np.asarray(defl["Q"][:2]) - np.asarray(c0["Q"][:2])
    r = _nice(float(np.hypot(*Q)), 0.2)
    ins.set_xlim(-r, r)
    ins.set_ylim(-r, r)
    ins.set_aspect("equal", adjustable="box")
    _frame(ins, t)
    ins.set_facecolor(t.surface)
    ins.axhline(0, color=t.border, lw=0.6, zorder=0)
    ins.axvline(0, color=t.border, lw=0.6, zorder=0)
    u0 = _unit(np.asarray(c0["P"][:2]))
    ins.plot([u0[0] * 3 * r, 0], [u0[1] * 3 * r, 0], color=pal["beam"], lw=1.0, alpha=0.35 if defl else 1.0,
             zorder=2)
    if defl is not None:
        ud = _unit(np.asarray(defl["P"][:2]) - np.asarray(defl["Q"][:2]))
        ins.plot([Q[0] + ud[0] * 3 * r, Q[0]], [Q[1] + ud[1] * 3 * r, Q[1]], color=pal["beam"], lw=1.6, zorder=3)
        if np.hypot(*Q) > 0.08 * r:
            _arrow(ins, 0, 0, Q[0], Q[1], t.text, lw=1.2, head=7)
    _tip_mark(ins, pal, t, label=None, ms=5)
    ins.set_title(f"focus ±{r:g} mm", fontsize=7, color=t.muted, pad=2)
    if moved:                                             # angles and spot shift both magnified N times
        ins.text(0.97, 0.96, f"×{N:g}" + (" (capped)" if N < float(gain or 1) else ""), transform=ins.transAxes,
                 ha="right", va="top", fontsize=7, color=t.muted, zorder=9, bbox=_bg(t))
    if moved and "Q" in c:
        true = (np.asarray(c["Q"][:2]) - np.asarray(c0["Q"][:2])) * 1e3
        ins.text(0.5, 0.03, f"shift {np.hypot(*true):.2f} µm", transform=ins.transAxes, ha="center", va="bottom",
                 fontsize=7, color=t.text, zorder=9, bbox=_bg(t))


def _view_side(fig, p, g, lower_um, upper_um, gain, pal, t):
    """Side view in the plane of incidence (the vertical plane of the aligned chief ray after the parabolic
    mirror): the chief ray at its true angle theta, the illumination cone, the mirror section and its lower edge."""
    N = _display_gain(p, lower_um, upper_um, gain)
    c0, _, mag = _magnified(p, lower_um, upper_um, N)
    moved = bool(lower_um or upper_um)
    if "Q" not in c0:
        _note(fig.add_subplot(), "The aligned chief ray misses the parabolic mirror", t)
        return
    ph = _unit([c0["r"][0], c0["r"][1], 0.0])            # horizontal direction of the chief ray toward the tip
    U = lambda P: float(P[0] * ph[0] + P[1] * ph[1])     # coordinate along the plane of incidence (mm)
    fp, zl, th = g["fp"], p["zLow"], g["th"]
    rho1 = math.hypot(g["x0"], g["h"]) + 6.5
    apA, _ = _aperture(p, g)
    # mirror section: points seen from the focus at elevation eps toward the mirror (direction -ph)
    eps = np.radians(np.linspace(0.2, 89.8, 500))
    pts = _pm_point(-np.cos(eps)[:, None] * ph + np.sin(eps)[:, None] * np.array([0.0, 0.0, 1.0]), fp)
    psi = np.degrees(np.arctan2(pts[:, 0], pts[:, 2]))
    keep = (pts[:, 2] >= zl) & (np.hypot(pts[:, 0], pts[:, 2]) <= rho1) & (np.abs(psi) <= 80)
    prof = np.column_stack([pts[:, 0] * ph[0] + pts[:, 1] * ph[1], pts[:, 2]])
    ep_c = math.pi / 2 - th                               # elevation of the aligned chief ray
    ax = fig.add_subplot()
    if not keep.any():
        _note(ax, "The plane of incidence misses the mirror", t)
        return
    mp = prof[keep]
    Pu, Pz = U(c0["P"]), float(c0["P"][2])
    u_lo = min(mp[:, 0].min(), Pu) - 9.0
    ax.set_xlim(u_lo, 8.0)
    ax.set_ylim(-2.4, mp[:, 1].max() + 3.0)
    ax.set_aspect("equal", adjustable="box")
    _frame(ax, t)
    # sample, with the two ends of the plane of incidence named in it
    ax.add_patch(Rectangle((u_lo - 5, -5), 20 - u_lo, 5, fc=t.border, ec="none", alpha=0.55, zorder=0))
    ax.axhline(0, color=t.muted, lw=1.0, zorder=1)
    kw = dict(textcoords="offset points", fontsize=FS_SMALL, color=t.muted, zorder=9, va="top")
    ax.annotate("sample (z = 0)", (0.5 * (u_lo + 8), 0), xytext=(0, -3), ha="center", **kw)
    if _side_name(-ph):
        ax.annotate("← " + _side_name(-ph), (u_lo, 0), xytext=(3, -3), ha="left", **kw)
    if _side_name(ph):
        ax.annotate(_side_name(ph) + " →", (8.0, 0), xytext=(-3, -3), ha="right", **kw)
    # mirror: reflecting section with a backing band
    r = np.hypot(*mp.T)
    back = mp * ((r + 1.4) / r)[:, None]
    ax.add_patch(Polygon(np.vstack([mp, back[::-1]]), closed=True, fc=pal["pm"], ec="none", alpha=0.28, zorder=2))
    ax.plot(mp[:, 0], mp[:, 1], color=pal["pm"], lw=2.2, zorder=3, solid_capstyle="butt")
    top = mp[np.argmax(mp[:, 1])]
    ax.annotate("parabolic mirror", (top[0], top[1]), xytext=(5, 0), textcoords="offset points", ha="left",
                va="top", fontsize=FS_SMALL, color=pal["pm"], fontweight="bold", zorder=9)
    low = mp[np.argmin(mp[:, 1])]
    ax.plot([low[0] - 2.5, low[0]], [zl, zl], color=t.muted, lw=0.8, ls=(0, (2, 2)), zorder=3)
    ax.annotate(f"lower edge\nz = {zl:g} mm", (low[0] - 2.5, zl), xytext=(-2, 0), textcoords="offset points",
                ha="right", va="bottom", fontsize=FS_SMALL, color=t.muted, zorder=9, linespacing=1.1)
    # illumination cone (NA) around the aligned chief ray
    sel = keep & (eps >= ep_c - apA) & (eps <= ep_c + apA)
    if sel.any():
        ax.add_patch(Polygon(np.vstack([[0.0, 0.0], prof[sel]]), closed=True, fc=pal["beam"], ec="none",
                             alpha=0.13, zorder=2))
        for q in (prof[sel][0], prof[sel][-1]):
            ax.plot([0, q[0]], [0, q[1]], color=pal["beam"], lw=0.7, alpha=0.5, zorder=3)
        e_l = ep_c - 0.5 * apA                            # label between the chief ray and the lower edge
        rl = 0.62 * float(np.hypot(*prof[sel][0]))
        ax.annotate(f"NA {p['NA']:g} (±{math.degrees(apA):.1f}°)", (-rl * math.cos(e_l), rl * math.sin(e_l)),
                    ha="center", va="center", fontsize=FS_SMALL, color=pal["beam"], zorder=9,
                    rotation=-math.degrees(e_l), rotation_mode="anchor")
    # incoming beam from the fold mirror (it runs along -Y, out of this plane: projected)
    Fu = U(g["F1"])
    ax.plot([Fu, Pu], [g["h"], Pz], color=pal["beam"], lw=1.3, alpha=0.55, zorder=4)
    _arrow(ax, Fu, g["h"], Fu + 0.45 * (Pu - Fu), g["h"], pal["beam"], lw=0, head=8)
    ax.annotate("from fold mirror\n(projected)", (Fu, g["h"]), xytext=(3, 2), textcoords="offset points",
                ha="left", va="bottom", fontsize=FS_SMALL, color=t.muted, zorder=9, linespacing=1.1)
    # chief rays
    shown = moved and mag is not None
    ax.plot([Pu, 0.0], [Pz, 0.0], color=pal["beam"], lw=1.2 if shown else 2.0, alpha=0.3 if shown else 1.0,
            zorder=5)
    if shown:
        ax.plot([U(mag["P"]), U(mag["Q"])], [mag["P"][2], 0.0], color=pal["beam"], lw=2.0, zorder=6)
    # angle of incidence
    ax.plot([0, 0], [0, 0.42 * Pz], color=t.muted, lw=0.8, ls=(0, (1.5, 2)), zorder=4)
    arc = np.radians(np.linspace(90, 90 + math.degrees(th), 40))
    ra = 0.3 * Pz
    ax.plot(ra * np.cos(arc), ra * np.sin(arc), color=t.text, lw=0.9, zorder=6)
    am = math.radians(90 + 0.5 * math.degrees(th))
    ax.annotate(f"θ = {math.degrees(th):g}°", (ra * math.cos(am), ra * math.sin(am)), xytext=(-3, 3),
                textcoords="offset points", ha="right", va="bottom", fontsize=FS, color=t.text, zorder=9)
    _tip_mark(ax, pal, t, 0.0, 0.35, label="tip", offset=(5, 2))
    ax.set_title(f"side view in the plane of incidence (β = {p['beta']:g}°)", fontsize=FS, color=t.muted,
                 loc="left", pad=3)
    _scale_bar(ax, 5, "5 mm", t, loc=(0.78, 0.72))
    if moved and mag is None:
        ax.text(0.98, 0.98, "beam misses the parabolic mirror at this gain", transform=ax.transAxes, ha="right",
                va="top", fontsize=FS_SMALL, color=t.danger, zorder=9)
    elif moved:
        msg = f"angles ×{N:g}" + (" (capped)" if N < float(gain or 1) else "")
        ax.text(0.98, 0.98, msg, transform=ax.transAxes, ha="right", va="top", fontsize=FS_SMALL, color=t.muted,
                zorder=9)


def _view_mirror(fig, p, lower_um, upper_um, gain, pal, t):
    """The KS1-Z8 seen from the actuator side (behind the mount), with the tilts and the KDC101 readings."""
    L, ys, ps = p["L"], (1 if p["yawSide"] >= 0 else -1), (1 if p["pitchSide"] >= 0 else -1)
    wide = fig.get_figwidth() / max(fig.get_figheight(), 1e-6) > 1.45
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.9]) if wide else fig.add_gridspec(2, 1, height_ratios=[2.4, 1])
    ax, tx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    tx.set_axis_off()
    h, R = L / 2, 6.5                                     # actuators L from the pivot; Z812 body radius (mm)
    piv, low, up = np.array([-ys * h, -ps * h]), np.array([ys * h, -ps * h]), np.array([-ys * h, ps * h])
    m = h + 7.0
    ax.set_xlim(-m - 9, m + 9)
    ax.set_ylim(-m - 8, m + 8)
    ax.set_aspect("equal", adjustable="box")
    _frame(ax, t)
    ax.add_patch(Rectangle((-m, -m), 2 * m, 2 * m, fc=pal["mirror"], ec=pal["mirror"], alpha=0.22, lw=1.0, zorder=0))
    ax.add_patch(Circle((0, 0), 12.7, fc="none", ec=t.muted, lw=0.9, ls=_DASH_SHORT, zorder=1))
    ax.annotate("mirror\n(far side)", (12.7 * 0.71 * ys, 12.7 * 0.71 * ps), xytext=(ys, ps),
                textcoords="offset points", ha="left" if ys > 0 else "right", va="bottom" if ps > 0 else "top",
                fontsize=FS_SMALL, color=t.muted, zorder=9, linespacing=1.05)
    # tilt axes: the lower actuator turns the plate about pivot-upper, the upper one about pivot-lower
    dashdot = (0, (6, 2, 1.5, 2))
    ax.plot([piv[0], piv[0]], [-m, m], color=pal["lower"], lw=0.9, ls=dashdot, alpha=0.8, zorder=1)
    ax.plot([-m, m], [piv[1], piv[1]], color=pal["upper"], lw=0.9, ls=dashdot, alpha=0.8, zorder=1)
    ax.add_patch(Circle(piv, 3.2, fc=pal["mirror"], ec=t.text, lw=0.9, zorder=4))
    ax.annotate("pivot (knob)", piv + np.array([ys * 4.0, -ps * 4.0]), xytext=(2 * ys, -2 * ps),
                textcoords="offset points", ha="left" if ys > 0 else "right", va="top" if ps > 0 else "bottom",
                fontsize=FS_SMALL, color=t.muted, zorder=9)
    travel = {"lower": float(lower_um or 0), "upper": float(upper_um or 0)}
    for name, pos in (("lower", low), ("upper", up)):
        col, v = pal[name], travel[name]
        ax.add_patch(Circle(pos, R, fc=col, ec="none", alpha=0.25, zorder=3))
        ax.add_patch(Circle(pos, R, fc="none", ec=col, lw=1.5, zorder=4))
        if v > 0:                                         # rod extends: pushes the plate away from the viewer
            ax.plot(*pos, marker="x", ms=7, mew=1.6, color=col, zorder=5)
        elif v < 0:
            ax.plot(*pos, marker="o", ms=3.5, color=col, zorder=5)
        above = pos[1] > 0
        ax.annotate(f"{name} Z812  {_signed(v, '{:.2f}')} µm", pos + np.array([0.0, (R + 0.8) * (1 if above else -1)]),
                    xytext=(0, 2 if above else -2), textcoords="offset points", ha="center",
                    va="bottom" if above else "top", fontsize=FS_SMALL, color=col, fontweight="bold", zorder=9)
    # the mirror normal tilts away from the pushing actuator; arrows at the plate centre, true length ratio
    dl, du = travel["lower"] / L, travel["upper"] / L
    big = max(abs(dl), abs(du))
    for d, vec, name, lab in ((dl, (-ys, 0.0), "lower", "horiz"), (du, (0.0, -ps), "upper", "vert")):
        if d:
            k = 10.0 * max(0.3, abs(d) / big) * math.copysign(1, d)
            _arrow(ax, 0, 0, vec[0] * k, vec[1] * k, pal[name], ("+" if d > 0 else "−") + lab, lw=1.6,
                   fontsize=FS_SMALL)
    # lever arm, along the side opposite the actuators
    xd = -ys * (m + 4)
    ax.annotate("", xy=(xd, up[1]), xytext=(xd, piv[1]), zorder=6,
                arrowprops=dict(arrowstyle="<->", color=t.muted, lw=0.8, shrinkA=0, shrinkB=0))
    ax.annotate(f"L = {L:g} mm", (xd, 0.0), xytext=(-3 * ys, 0), textcoords="offset points", rotation=90,
                ha="right" if ys > 0 else "left", va="center", fontsize=FS_SMALL, color=t.muted, zorder=9)
    ax.set_title("KS1-Z8 seen from the actuator side", fontsize=FS, color=t.muted, loc="left", pad=3)

    # readings: tilt delta = travel / L and the change of the KDC101 display
    for i, (name, title) in enumerate((("lower", "Lower Z812 · horizontal"), ("upper", "Upper Z812 · vertical"))):
        v = travel[name]
        d = v / L                                         # mrad (um / mm)
        state = "rod extends" if v > 0 else ("rod retracts" if v < 0 else "aligned")
        lines = [f"travel {_signed(v, '{:.3f}')} µm · {state}",
                 f"δ = {_signed(d * 1e3, '{:.1f}')} µrad = {_signed(d * 1e-3 * 206264.806, '{:.1f}')}″",
                 f"KDC101 {_signed(v / 1000, '{:.4f}')} mm",
                 f"   = {_signed(steering.kdc101_counts(v), '{:,.0f}')} encoder counts"]
        x0, y0 = (0.02, 0.95 - 0.43 * i) if wide else (0.01 + 0.5 * i, 0.98)
        tx.text(x0, y0, title, transform=tx.transAxes, ha="left", va="top", fontsize=FS, color=pal[name],
                fontweight="bold")
        tx.text(x0, y0 - (0.07 if wide else 0.19), "\n".join(lines), transform=tx.transAxes, ha="left", va="top",
                fontsize=FS_SMALL, color=t.text, linespacing=1.3)
    tx.text(0.01, 0.0, "δ = travel / L · × rod extends (pushes the plate away), • retracts",
            transform=tx.transAxes, ha="left", va="bottom", fontsize=7, color=t.muted)


def draw_schematic(fig, sp, lower_um, upper_um, view, gain, t) -> None:
    """2-D schematic of the pump path for the actuator travels (um), angles exaggerated by `gain`.
    view: 'top' (whole path from above), 'inside' (box interior near the tip), 'side' (plane of incidence),
    'mirror' (the KS1-Z8 seen from the actuator side)."""
    pal = palette(t)
    if sp is None:
        _note(fig.add_subplot(), "No steering parameters", t)
        return
    p = steering.resolve(sp)
    g = steering.optics.geometry(p)
    lower, upper = float(lower_um or 0.0), float(upper_um or 0.0)
    view = view or "top"
    if view == "top":
        _view_top(fig, p, g, lower, upper, gain, pal, t)
    elif view == "inside":
        _view_inside(fig, p, g, lower, upper, gain, pal, t)
    elif view == "side":
        _view_side(fig, p, g, lower, upper, gain, pal, t)
    elif view == "mirror":
        _view_mirror(fig, p, lower, upper, gain, pal, t)
    else:
        _note(fig.add_subplot(), f"Unknown view {view!r}", t)


# ----------------------------------------------------------------------------- link to the tip-position chain
_MODE_TEXT = {"lower": "lower actuator only", "upper": "upper actuator only", "both": "both actuators, Y = 0"}


def draw_link(fig, rows, sp, mode, t, measured=None) -> None:
    """Planned spot-tip offsets X: (a) landing points in the simulation frame, (b) predicted pump spot vs X,
    optionally with the measured tip-scan RMS (normalised to 100 % at its maximum).  rows: steering.plan_offsets;
    measured: [(bias_V, X_um, rms), ...]."""
    pal = palette(t)
    rows = [r for r in (rows or []) if r is not None]
    if not rows:
        _note(fig.add_subplot(), "No offsets planned yet", t)
        return
    mode = mode if mode in _MODE_TEXT else "lower"
    col = {"lower": pal["lower"], "upper": pal["upper"], "both": pal["beam"]}[mode]
    wide = fig.get_figwidth() / max(fig.get_figheight(), 1e-6) >= 1.5
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25]) if wide else fig.add_gridspec(2, 1, height_ratios=[1, 1.7])
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    X = np.array([r["X_um"] for r in rows], float)
    Xg = np.array([r["X_got_um"] for r in rows], float)
    Yg = np.array([r["Y_got_um"] for r in rows], float)

    # (a) landing points, to scale
    o = np.argsort(Xg)
    a1.axhline(0, color=pal["bias"], lw=0.9, alpha=0.6, zorder=0)
    a1.axvline(0, color=t.border, lw=0.7, zorder=0)
    a1.plot(Xg[o], Yg[o], color=col, lw=1.3, zorder=2)
    h_land, = a1.plot(Xg, Yg, "o", ms=4.5, color=col, mec=t.surface, mew=0.8, zorder=3, label="landing point")
    h_tgt, = a1.plot(X, np.zeros_like(X), "|", ms=10, mew=1.3, color=t.text, zorder=4, label="target X")
    _tip_mark(a1, pal, t, label="tip", offset=(4, 4))
    lo, hi = min(Xg.min(), X.min(), 0.0), max(Xg.max(), X.max(), 0.0)
    span = max(hi - lo, 1.0)
    ymax = max(float(np.abs(Yg).max()) * 1.2, 0.12 * span)
    a1.set_xlim(lo - 0.08 * span, hi + 0.08 * span)
    a1.set_ylim(-ymax, ymax)
    a1.set_aspect("equal", adjustable="box")
    a1.set_xlabel("X along the bias axis (µm)", fontsize=FS + 0.5)
    a1.set_ylabel("Y across (µm)", fontsize=FS + 0.5)
    a1.tick_params(labelsize=FS)
    a1.set_title(f"(a) landing points\n{_MODE_TEXT[mode]}", fontsize=FS + 0.5, loc="left", linespacing=1.3)

    # (b) predicted pump spot vs X
    rel = np.array([np.nan if r.get("rel") is None else 100 * r["rel"] for r in rows], float)
    fL = np.array([np.nan if r.get("fwhmL") is None else r["fwhmL"] for r in rows], float)
    fS = np.array([np.nan if r.get("fwhmS") is None else r["fwhmS"] for r in rows], float)
    o = np.argsort(X)
    a2.grid(True, axis="y", color=t.border, lw=0.6)
    a2.set_axisbelow(True)
    a3 = a2.twinx()
    a3.spines["right"].set_visible(True)
    a3.spines["right"].set_color(t.muted)
    hL, = a3.plot(X[o], fL[o], color=t.muted, lw=1.3, label="FWHM long")
    hS, = a3.plot(X[o], fS[o], color=t.muted, lw=1.3, ls=(0, (4, 2.5)), label="FWHM short")
    a2.set_zorder(a3.get_zorder() + 1)                    # intensity and markers above the FWHM lines
    a2.patch.set_visible(False)
    h_rel, = a2.plot(X[o], rel[o], color=pal["beam"], lw=1.9, marker="o", ms=3.5, label="pump peak", zorder=3)
    handles = [h_land, h_tgt, h_rel, hL, hS]
    if measured:
        m = np.array([(float(b), float(x), float(v)) for b, x, v in measured], float).reshape(-1, 3)
        top = np.nanmax(m[:, 2]) if len(m) else np.nan
        if np.isfinite(top) and top > 0:
            biases = list(dict.fromkeys(m[:, 0]))
            for i, b in enumerate(biases):
                sel = m[:, 0] == b
                lab = "measured RMS (norm.)" + ("" if len(biases) == 1 else f", {_signed(b)} V")
                hm, = a2.plot(m[sel, 1], 100 * m[sel, 2] / top, ls="none", marker="D", ms=4.5, mew=1.1,
                              color=t.text, mfc=t.text if i == 0 else t.surface, zorder=5, label=lab)
                handles.append(hm)
    a2.set_ylim(0, 105)
    a2.set_ylabel("pump peak (%)", fontsize=FS + 0.5, color=pal["beam"])
    a2.set_xlabel("spot–tip offset X (µm)", fontsize=FS + 0.5)
    a2.tick_params(labelsize=FS)
    fin = np.r_[fL, fS][np.isfinite(np.r_[fL, fS])]
    a3.set_ylim(0, 1.15 * (fin.max() if len(fin) else 1.0))
    a3.set_ylabel("FWHM (µm)", fontsize=FS + 0.5, color=t.muted)
    a3.tick_params(labelsize=FS)
    a2.set_title("(b) predicted pump spot\naligned spot = 100 %", fontsize=FS + 0.5, loc="left", linespacing=1.3)
    w_in = fig.get_figwidth()
    fig.legend(handles=handles, loc="outside lower center", ncols=4 if w_in >= 7.5 else (3 if w_in >= 5 else 2),
               fontsize=FS_SMALL, frameon=False, columnspacing=1.2, handlelength=1.8)
