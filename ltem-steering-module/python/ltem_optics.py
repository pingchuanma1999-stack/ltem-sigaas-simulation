"""ltem_optics — pump-spot steering model for the neaspec THz s-SNOM (non-local LTEM).

NumPy port of js/ltem_optics.mjs (the reference). Both are checked against tests/golden.json.

Units: geometry in mm, parameter angles in degrees (radians internally), wavelength in nm,
actuator travel and sample-plane results in µm.

Lab frame (right-handed): origin = focus = tip apex on the sample surface,
  +X toward the entrance-port wall ("right", seen from the open front of the AFM box),
  +Y toward the back panel (−Y = "front", the open side where the user stands),
  +Z up. The sample surface is z = 0.

Optical path: the input beam travels −Y onto the steering mirror (KS1-Z8 mount, face vertical),
reflects toward −X into the port (x = portX), hits the vertical fold mirror behind the tip
(normal ∝ (1,−1,0), turns −X into −Y), travels toward the front onto the right half of a
semi-paraboloid (axis along Y, focus at the origin, x²+z² = 4·fp·(y+fp)) and is focused onto the sample.

Actuators (KS1-Z8 as drawn by Thorlabs, seen from the actuator side): manual knob = pivot at the lower
left; lower Z812 at the lower right = pure horizontal tilt ("yaw"); upper Z812 at the upper left =
pure vertical tilt ("pitch"). Positive travel = push rod extends = KDC101 reading increases.

Public API (mirrors the JS names in snake_case):
    DEFAULTS, QUALITY
    resolve_params(params) -> dict              (resolves D='auto')
    steering_response(params, yaw_um, pitch_um) -> dict
    optimal_beam_diameter(params) -> float      (mm)
    spot_shape(params, yaw_um, pitch_um, quality='fine') -> dict | None
    peak_curve(params, axis, travels_um, quality='trend') -> list[dict]
    contour(spot, level) -> np.ndarray (n, 4) segments [x1, y1, x2, y2] in µm relative to spot['x0']
Lower level: geometry, mirror_normal, trace_chief, trace_after_steer, compute_spot_raw, adaptive_spot_raw.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

__all__ = [
    "DEFAULTS", "QUALITY", "resolve_params", "geometry", "mirror_normal", "trace_chief", "trace_after_steer",
    "steering_response", "compute_spot_raw", "adaptive_spot_raw", "optimal_beam_diameter",
    "reference_spot", "spot_shape", "peak_curve", "contour",
]

# ------------------------------------------------------------------ parameters
DEFAULTS = {
    "R": 16.0,       # mm, parabolic-mirror effective focal length (illuminated chief point to focus)
    "theta": 60.0,   # deg, angle of incidence on the sample (from the normal)
    "beta": 45.0,    # deg, azimuth of the plane of incidence from the front–back (Y) axis, toward the port side
    "alpha": 45.0,   # deg, angle of incidence on the steering mirror
    "dist": 250.0,   # mm, steering mirror to entrance-port plane
    "L": 37.7,       # mm, lever arm: actuator contact to pivot (KS1-Z8: 0.2 µm step <-> 1.1 arcsec)
    "lam": 820.0,    # nm, pump wavelength
    "D": "auto",     # mm (1/e^2 diameter of the input beam) or 'auto'
    "NA": 0.42,      # illumination aperture of the parabolic mirror
    "zLow": 1.0,     # mm, height of the parabolic mirror's lower edge above the sample
    "portX": 260.0,  # mm, x of the entrance-port plane (only for the beam shift at the port)
    "foldY": 6.0,    # mm, y of the fold-mirror centre (behind the tip)
    "yawSide": 1,    # +1: horizontal actuator right of the pivot seen from behind (KS1-Z8 upright); -1: left
    "pitchSide": 1,  # +1: vertical actuator above the pivot (KS1-Z8 upright); -1: below
}
QUALITY = {"fast": {"M": 48, "Nmin": 30}, "fine": {"M": 72, "Nmin": 40}, "trend": {"M": 40, "Nmin": 28}}


def _js_round(x: float) -> float:
    """Math.round semantics (half toward +inf)."""
    return math.floor(x + 0.5)


def _param_key(p):
    """Key over every parameter that affects the optics (used by the two caches in this module)."""
    return tuple(p[k] for k in ("R", "theta", "beta", "alpha", "dist", "L", "lam", "D", "NA", "zLow", "portX", "foldY", "yawSide", "pitchSide"))


_auto_d_cache: dict = {}


def resolve_params(params: Optional[dict] = None) -> dict:
    """Merge with DEFAULTS and resolve D='auto' (computed once per parameter set, then cached)."""
    p = dict(DEFAULTS)
    p.update(params or {})
    if p.get("D") in ("auto", None):
        key = _param_key({**p, "D": "auto"})
        if key not in _auto_d_cache:
            q = dict(p)
            q["D"] = 13.0
            _auto_d_cache[key] = _js_round(optimal_beam_diameter(q) * 10) / 10
        p["D"] = _auto_d_cache[key]
        p["Dauto"] = True
    return p


def _merged(params: Optional[dict]) -> dict:
    p = dict(DEFAULTS)
    p.update(params or {})
    return p


# ------------------------------------------------------------------ vectors
def _norm(a):
    a = np.asarray(a, dtype=float)
    return a / math.hypot(*a)


def _reflect(d, n):
    return d - n * (2.0 * float(np.dot(d, n)))


def _rot_axis(v, k, ang):
    v = np.asarray(v, float)
    k = np.asarray(k, float)
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(k, v) * s + k * (float(np.dot(k, v)) * (1 - c))


# ------------------------------------------------------------------ geometry and ray tracing
def geometry(p: dict) -> dict:
    th = p["theta"] * math.pi / 180
    be = p["beta"] * math.pi / 180
    R = p["R"]
    P = np.array([R * math.sin(th) * math.sin(be), -R * math.sin(th) * math.cos(be), R * math.cos(th)])
    fp = R * (1 + math.sin(th) * math.cos(be)) / 2
    x0, h, yf = P[0], P[2], p["foldY"]
    F1 = np.array([x0, yf, h])
    nF1 = _norm([1.0, -1.0, 0.0])
    uIn = np.array([-1.0, 0.0, 0.0])
    S = np.array([p["portX"] + p["dist"], yf, h])
    defl = math.pi - 2 * p["alpha"] * math.pi / 180
    d0 = _norm(_rot_axis(uIn, [0.0, 0.0, 1.0], defl))
    n0 = _norm(uIn - d0)
    vAx = np.array([0.0, 0.0, 1.0])
    uBack = _norm(np.cross(n0, vAx))
    return dict(th=th, be=be, R=R, P=P, fp=fp, x0=x0, h=h, yf=yf, F1=F1, nF1=nF1, uIn=uIn, S=S, d0=d0, n0=n0, vAx=vAx, uBack=uBack)


def mirror_normal(g: dict, p: dict, dp: float, dy: float):
    """Mirror normal for push-rod extensions dp (upper/vertical) and dy (lower/horizontal), in mm."""
    ap = -p["pitchSide"] * (dp / p["L"])
    ay = p["yawSide"] * (dy / p["L"])
    return _norm(_rot_axis(_rot_axis(g["n0"], g["uBack"], ap), g["vAx"], ay))


def _hit_parab_y(o, d, fp):
    """Vectorised smallest positive root of |o + t d| on x^2 + z^2 = 4 fp (y + fp). o: (n,3), d: (3,)."""
    a = d[0] * d[0] + d[2] * d[2]
    b = 2 * (o[:, 0] * d[0] + o[:, 2] * d[2]) - 4 * fp * d[1]
    c = o[:, 0] ** 2 + o[:, 2] ** 2 - 4 * fp * (o[:, 1] + fp)
    disc = b * b - 4 * a * c
    ok = disc >= 0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    q = -0.5 * (b + np.where(b >= 0, sq, -sq))
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = q / a if abs(a) > 1e-300 else np.full_like(b, np.inf)
        t2 = np.where(np.abs(q) > 1e-300, c / np.where(np.abs(q) > 1e-300, q, 1.0), np.inf)
    t1 = np.where(t1 > 1e-9, t1, np.inf)
    t2 = np.where(t2 > 1e-9, t2, np.inf)
    t = np.minimum(t1, t2)
    t = np.where(ok, t, np.inf)
    return t  # inf = no hit


def _trace_after_steer_many(o, d, g):
    """Trace many parallel rays (origins o (n,3), common direction d (3,)) through fold mirror and PM.
    Returns dict of arrays and a validity mask."""
    n = o.shape[0]
    den = float(np.dot(d, g["nF1"]))
    if abs(den) < 1e-12:
        return None
    t0 = ((g["F1"][None, :] - o) @ g["nF1"]) / den
    ok = t0 > 1e-9
    A = o + d[None, :] * t0[:, None]
    d2 = _norm(_reflect(d, g["nF1"]))
    t1 = _hit_parab_y(A, d2, g["fp"])
    ok &= np.isfinite(t1)
    t1 = np.where(np.isfinite(t1), t1, 0.0)
    P = A + d2[None, :] * t1[:, None]
    nP = np.stack([2 * P[:, 0], np.full(n, -4 * g["fp"]), 2 * P[:, 2]], axis=1)
    nP /= np.linalg.norm(nP, axis=1)[:, None]
    r = d2[None, :] - nP * (2 * (nP @ d2))[:, None]
    r /= np.linalg.norm(r, axis=1)[:, None]
    ok &= r[:, 2] < 0
    return dict(A=A, P=P, r=r, d2=d2, ok=ok)


def trace_after_steer(o, d, g):
    """Single ray (scalar API like the JS one). Returns dict with A, d2, P, r, Q or None."""
    res = _trace_after_steer_many(np.asarray(o, float)[None, :], np.asarray(d, float), g)
    if res is None or not res["ok"][0]:
        return None
    P, r = res["P"][0], res["r"][0]
    Q = P + r * (-P[2] / r[2])
    return dict(A=res["A"][0], d2=res["d2"], P=P, r=r, Q=Q)


def trace_chief(p: dict, dp: float, dy: float) -> dict:
    """Chief ray for push-rod extensions dp (upper/vertical) and dy (lower/horizontal) in mm."""
    g = geometry(p)
    n = mirror_normal(g, p, dp, dy)
    d1 = _norm(_reflect(g["d0"], n))
    res = trace_after_steer(g["S"], d1, g)
    out = dict(g=g, n=n, d1=d1)
    if res:
        out.update(res)
    return out


def _perp_frame(d):
    e1 = np.cross(d, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-9:
        e1 = np.cross(d, [0.0, 1.0, 0.0])
    e1 = _norm(e1)
    return e1, _norm(np.cross(e1, d))


def _port_point(cc, p):
    t = (p["portX"] - cc["g"]["S"][0]) / cc["d1"][0]
    return cc["g"]["S"] + cc["d1"] * t


# ------------------------------------------------------------------ steering response (chief ray, exact)
def steering_response(params: Optional[dict] = None, yaw_um: float = 0.0, pitch_um: float = 0.0) -> dict:
    """Where the pump spot goes for given push-rod travels (µm). See README for the fields."""
    p = _merged(params)
    c0 = trace_chief(p, 0.0, 0.0)
    c = trace_chief(p, pitch_um / 1000, yaw_um / 1000)
    cy1 = trace_chief(p, 0.0, 1e-3)
    cp1 = trace_chief(p, 1e-3, 0.0)
    if not all("Q" in x for x in (c0, c, cy1, cp1)):
        raise ValueError("chief ray misses the parabolic mirror for these parameters")
    jy = (cy1["Q"] - c0["Q"]) * 1000
    jp = (cp1["Q"] - c0["Q"]) * 1000
    jy_port = (_port_point(cy1, p) - _port_point(c0, p)) * 1000
    jp_port = (_port_point(cp1, p) - _port_point(c0, p)) * 1000
    dQ = (c["Q"] - c0["Q"]) * 1000
    d_port = (_port_point(c, p) - _port_point(c0, p)) * 1000
    ph = _norm([c0["r"][0], c0["r"][1], 0.0])
    along = float(np.dot(dQ, ph))
    across = float(np.dot(dQ, [-ph[1], ph[0], 0.0]))
    th = p["theta"] * math.pi / 180
    return {
        "spot_um": [float(dQ[0]), float(dQ[1])],
        "port_um": [float(d_port[1]), float(d_port[2])],
        "perUm": {"yaw": [float(jy[0]), float(jy[1])], "pitch": [float(jp[0]), float(jp[1])]},
        "perUmPort": {"yaw": [float(jy_port[1]), float(jy_port[2])], "pitch": [float(jp_port[1]), float(jp_port[2])]},
        "incidencePlane": [float(ph[0]), float(ph[1])],
        "along_um": along,
        "across_um": across,
        "defocus_um": abs(along) * math.sin(th),
        "arrival_fs": abs(along) * math.sin(th) * 3.33564,
        "yawTurnsBeamToFront": bool(float(np.dot(cy1["d1"] - c0["d1"], [0.0, -1.0, 0.0])) > 0),
    }


# ------------------------------------------------------------------ spot engine (scalar Debye sum on the tilted sample)
def compute_spot_raw(P: dict, dp: float, dy: float, opt: dict) -> Optional[dict]:
    """One spot calculation on a fixed window. dp, dy in mm; opt = {M, Nmin, W (µm), peakOnly?}."""
    g = geometry(P)
    nrm = mirror_normal(g, P, dp, dy)
    d1 = _norm(_reflect(g["d0"], nrm))
    ch = trace_after_steer(g["S"], d1, g)
    if ch is None:
        return None
    x0 = ch["Q"]
    e1, e2 = _perp_frame(d1)
    k = 2 * math.pi / (P["lam"] * 1e-6)  # 1/mm
    apA = math.asin(min(0.95, P["NA"]))
    p0 = _norm(g["P"])
    zLow = P["zLow"]
    w = P["D"] / 2
    rmax = min(1.8 * w, 1.3 * P["R"] * math.tan(apA))
    per = (P["lam"] * 1e-3) * P["R"] / (2 * rmax)
    W = opt["W"]
    N = max(opt["Nmin"], min(120, math.ceil(2 * W / (0.8 * per))))
    # pupil grid in the same order as the JS loops (i outer = a, j inner = b)
    coords = -rmax + 2 * rmax * (np.arange(N) + 0.5) / N
    aa, bb = np.meshgrid(coords, coords, indexing="ij")
    aa, bb = aa.ravel(), bb.ravel()
    rr = aa * aa + bb * bb
    inside = rr <= rmax * rmax
    aa, bb, rr = aa[inside], bb[inside], rr[inside]
    A0 = np.exp(-rr / (w * w))
    a2all = float(np.sum(A0 * A0))
    o = g["S"][None, :] + aa[:, None] * e1[None, :] + bb[:, None] * e2[None, :]
    tr = _trace_after_steer_many(o, d1, g)
    if tr is None:
        return None
    ok = tr["ok"]
    Pm, rv, Am = tr["P"], tr["r"], tr["A"]
    # soft (area-weighted) aperture edges
    cell = 2 * rmax / N
    Lp = np.linalg.norm(Pm, axis=1)
    cosang = np.minimum(1.0, (Pm @ p0) / np.where(Lp > 0, Lp, 1.0))
    mEdge = (apA - np.arccos(cosang)) * Lp
    mLow = Pm[:, 2] - zLow
    wAp = np.clip(0.5 + mEdge / cell, 0, 1) * np.clip(0.5 + mLow / cell, 0, 1)
    ok &= wAp > 0
    if int(ok.sum()) < 20:
        return None
    o, Am, Pm, rv, A0, wAp = o[ok], Am[ok], Pm[ok], rv[ok], A0[ok], wAp[ok]
    opl = np.linalg.norm(Am - o, axis=1) + np.linalg.norm(Pm - Am, axis=1)
    dx = x0[None, :] - Pm
    Lr = np.linalg.norm(dx, axis=1)
    ph = opl + np.sum(rv * dx, axis=1)
    ph0 = ph[0]
    amp = A0 * wAp / Lr
    phi = k * (ph - ph0)
    c = amp * np.exp(1j * phi)
    RX, RY = rv[:, 0], rv[:, 1]
    nr = c.shape[0]
    asum = float(np.sum(amp))
    dA = (2 * rmax / N) ** 2
    lamMM = P["lam"] * 1e-6
    perPow = asum * asum * dA / (lamMM * lamMM * a2all) * 1e-6
    if opt.get("peakOnly"):
        s = np.sum(c)
        return {"IperP": float(abs(s) ** 2) / (asum * asum) * perPow, "nr": nr}
    M = opt["M"]
    Wmm = W * 1e-3
    u = -Wmm + 2 * Wmm * np.arange(M) / (M - 1)
    G = c[:, None] * np.exp(1j * k * RX[:, None] * u[None, :])   # (nr, M): x columns
    H = np.exp(1j * k * RY[:, None] * u[None, :])                # (nr, M): y rows
    E = H.T @ G                                                  # (M rows b, M cols a)
    nrm2 = 1.0 / (asum * asum)
    I64 = (E.real ** 2 + E.imag ** 2) * nrm2
    ia = int(np.argmax(I64))
    imax = float(I64.ravel()[ia])
    I = I64.astype(np.float32)
    du = 2 * Wmm / (M - 1)

    def eval_at(uu, vv):
        s = np.sum(c * np.exp(1j * k * (RX * uu + RY * vv)))
        return float(s.real ** 2 + s.imag ** 2) * nrm2

    pu = -Wmm + (ia % M) * du
    pv = -Wmm + (ia // M) * du
    pk = imax
    h = du / 2
    it = 0
    while it < 16 and h > du / 40:
        best = None
        for su, sv in ((h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h)):
            v = eval_at(pu + su, pv + sv)
            if v > pk:
                pk = v
                best = (pu + su, pv + sv)
        if best:
            pu, pv = best
        else:
            h /= 2
        it += 1
    # half-maximum region (soft edge 0.45–0.55) fitted with the ellipse of equal second moments
    px = 2 * W / (M - 1)
    Iflat = I.ravel().astype(np.float64)
    r = Iflat / pk
    t = np.where(r <= 0.45, 0.0, np.where(r >= 0.55, 1.0, (r - 0.45) / 0.1))
    t32 = t.astype(np.float32).astype(np.float64)  # JS stores the weights in a Float32Array for the 2nd pass
    q = np.arange(M * M)
    acol, brow = q % M, q // M
    X = -W + acol * px
    Y = -W + brow * px
    S = float(np.sum(t))
    cx = float(np.sum(t * X)) / S
    cy = float(np.sum(t * Y)) / S
    Xc, Yc = X - cx, Y - cy
    xx = float(np.sum(t32 * Xc * Xc)) / S
    yy = float(np.sum(t32 * Yc * Yc)) / S
    xy = float(np.sum(t32 * Xc * Yc)) / S
    trc = xx + yy
    det = xx * yy - xy * xy
    disc = math.sqrt(max(0.0, trc * trc / 4 - det))
    l1 = trc / 2 + disc
    l2 = max(0.0, trc / 2 - disc)
    I2 = I.astype(np.float64)
    border = float(max(I2[0, :].max(), I2[M - 1, :].max(), I2[:, 0].max(), I2[:, M - 1].max()))
    return {
        "I": I,                      # float32 (M, M); row b ↔ y (back), column a ↔ x (port side)
        "M": M, "W": W, "x0": [float(v) for v in x0], "N": N, "nr": nr,
        "peak": pk,                  # peak / perfect-focus peak of the transmitted light
        "IperP": pk * perPow,        # 1/µm², peak intensity per unit laser power
        "peakAt": [pu * 1e3, pv * 1e3],
        "cx": cx, "cy": cy,
        "fwhmL": 4 * math.sqrt(l1), "fwhmS": 4 * math.sqrt(l2),
        "ang": 0.5 * math.atan2(2 * xy, xx - yy),
        "area": S * px * px,
        "border": border / pk,
    }


def adaptive_spot_raw(P: dict, dp: float, dy: float, q: dict, W_start: Optional[float] = None) -> Optional[dict]:
    W = max(2.4, min(18.0, W_start or 3.5))
    s = None
    shrunk = False
    for _ in range(5):
        s = compute_spot_raw(P, dp, dy, {"M": q["M"], "Nmin": q["Nmin"], "W": W})
        if s is None:
            return None
        if s["border"] > 0.03 and W < 18:
            W = min(18.0, W * 1.6)
            continue
        want = max(2.4, 1.3 * s["fwhmL"])
        if (not shrunk) and s["border"] < 0.004 and W > 2 * want:
            W = max(2.4, 1.25 * want)
            shrunk = True
            continue
        break
    return s


def optimal_beam_diameter(params: Optional[dict] = None) -> float:
    """Beam diameter (mm, 1/e²) with the highest peak intensity per unit laser power for the aligned beam."""
    p = _merged(params)

    def f(D):
        q = dict(p)
        q["D"] = D
        r = compute_spot_raw(q, 0.0, 0.0, {"M": 3, "Nmin": 44, "W": 0.5, "peakOnly": True})
        return r["IperP"] if r else 0.0

    bD, b = 3.0, -1.0
    D = 3.0
    while D <= 26.001:
        v = f(D)
        if v > b:
            b, bD = v, D
        D += 0.5
    a, c = f(bD - 0.5), f(bD + 0.5)
    den = a - 2 * b + c
    Dopt = bD + 0.25 * (a - c) / den if den < 0 else bD
    return max(3.0, min(26.0, Dopt))


_ref_cache: dict = {}


def reference_spot(p: dict) -> Optional[dict]:
    """Aligned spot (both actuators at 0), fine quality, cached. p must have a numeric D. Treat the result as read-only."""
    key = _param_key(p)
    if key not in _ref_cache:
        _ref_cache[key] = adaptive_spot_raw(p, 0.0, 0.0, QUALITY["fine"], 3.5)
    return _ref_cache[key]


def spot_shape(params: Optional[dict] = None, yaw_um: float = 0.0, pitch_um: float = 0.0, quality="fine", W_start=None) -> Optional[dict]:
    """Spot for push-rod travels (µm). Adds 'rel' (peak relative to the aligned spot, power-normalised)
    and 'k' (factor turning I into intensity relative to the aligned peak)."""
    p = resolve_params(params)
    ref = reference_spot(p)
    q = QUALITY[quality] if isinstance(quality, str) else quality
    if yaw_um == 0 and pitch_um == 0 and q is QUALITY["fine"]:
        s = ref
    else:
        s = adaptive_spot_raw(p, pitch_um / 1000, yaw_um / 1000, q, W_start or (ref["W"] if ref else 3.5))
    if s is None or ref is None:
        return None
    out = dict(s)
    if s is ref:  # never hand out the cached arrays
        out["I"] = ref["I"].copy()
        out["x0"] = list(ref["x0"])
        out["peakAt"] = list(ref["peakAt"])
    out["rel"] = s["IperP"] / ref["IperP"]
    out["k"] = out["rel"] / s["peak"]
    out["D"] = p["D"]
    return out


def peak_curve(params: Optional[dict], axis: str, travels_um, quality="trend") -> list:
    """Peak intensity (relative to the aligned spot) and FWHM along one actuator's travel. axis: 'yaw' | 'pitch'."""
    p = resolve_params(params)
    q = QUALITY[quality] if isinstance(quality, str) else quality
    ref0 = adaptive_spot_raw(p, 0.0, 0.0, q, 3.5)
    W = ref0["W"] if ref0 else 3.5
    out = []
    for t in travels_um:
        s = adaptive_spot_raw(p, t / 1000 if axis == "pitch" else 0.0, t / 1000 if axis == "yaw" else 0.0, q, W)
        if s is None:
            out.append({"travel_um": t, "rel": None})
            continue
        W = s["W"]
        out.append({"travel_um": t, "rel": s["IperP"] / ref0["IperP"], "fwhmL": s["fwhmL"], "fwhmS": s["fwhmS"], "area": s["area"], "ang": s["ang"]})
    return out


# ------------------------------------------------------------------ contour (marching squares, same as JS)
_MS = {1: ((3, 0),), 2: ((0, 1),), 3: ((3, 1),), 4: ((1, 2),), 5: ((3, 0), (1, 2)), 6: ((0, 2),), 7: ((3, 2),), 8: ((2, 3),),
       9: ((0, 2),), 10: ((0, 1), (2, 3)), 11: ((1, 2),), 12: ((1, 3),), 13: ((0, 1),), 14: ((3, 0),)}


def contour(s: dict, level: float) -> np.ndarray:
    """Iso-line I = level as an (n, 4) array of segments [x1, y1, x2, y2] in µm relative to s['x0']."""
    I = np.asarray(s["I"], dtype=np.float64)
    M, W = s["M"], s["W"]
    px = 2 * W / (M - 1)
    segs = []
    for b in range(M - 1):
        for a in range(M - 1):
            v0, v1, v2, v3 = I[b, a], I[b, a + 1], I[b + 1, a + 1], I[b + 1, a]
            cidx = int(v0 > level) | (int(v1 > level) << 1) | (int(v2 > level) << 2) | (int(v3 > level) << 3)
            if cidx == 0 or cidx == 15:
                continue
            pts = ((-W + a * px, -W + b * px), (-W + (a + 1) * px, -W + b * px), (-W + (a + 1) * px, -W + (b + 1) * px), (-W + a * px, -W + (b + 1) * px))
            vals = (v0, v1, v2, v3)

            def edge(e):
                i, j = e, (e + 1) % 4
                t = (level - vals[i]) / (vals[j] - vals[i])
                return (pts[i][0] + (pts[j][0] - pts[i][0]) * t, pts[i][1] + (pts[j][1] - pts[i][1]) * t)

            for e_a, e_b in _MS[int(cidx)]:
                A, B = edge(e_a), edge(e_b)
                segs.append((A[0], A[1], B[0], B[1]))
    return np.array(segs, dtype=float).reshape(-1, 4)
