// ltem_optics.mjs — pump-spot steering model for the neaspec THz s-SNOM (non-local LTEM)
//
// Pure computation, no DOM. Reference implementation; the Python port (python/ltem_optics.py)
// reproduces it and both are checked against tests/golden.json.
//
// Units: geometry in mm, angles in degrees (parameters) / radians (internal), wavelength in nm,
// actuator travel and sample-plane results in µm.
//
// Lab frame (right-handed): origin = focus = tip apex on the sample surface,
//   +X toward the entrance-port wall ("right", seen from the open front of the AFM box),
//   +Y toward the back panel (−Y = "front", the open side where the user stands),
//   +Z up. The sample surface is the plane z = 0.
//
// Optical path: input beam travels −Y and hits the steering mirror S (KS1-Z8 mount, face vertical),
// reflects toward −X into the port in the right wall (x = portX), hits the vertical fold mirror F1
// behind the tip (normal ∝ (1,−1,0), turns −X into −Y), travels toward the front onto the right half
// of a semi-paraboloid (axis along Y, focus at the origin, x²+z² = 4·fp·(y+fp)) and is focused onto
// the sample. Collection (left half → F2 → left wall) is not needed for the pump.
//
// Actuators (KS1-Z8 as drawn by Thorlabs, seen from behind = actuator side): manual knob = pivot at the
// lower left; lower Z812 at the lower right (pure horizontal tilt, "yaw"); upper Z812 at the upper left
// (pure vertical tilt, "pitch"). Positive travel = push rod extends = KDC101 reading increases.
// Extending an actuator pushes that corner of the mirror toward its reflecting face, i.e. the mirror
// normal tilts away from that actuator.

// ---------------------------------------------------------------- vectors
export const V = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  mul: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  len: (a) => Math.hypot(a[0], a[1], a[2]),
  norm: (a) => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; },
};
export const reflect = (d, n) => V.sub(d, V.mul(n, 2 * V.dot(d, n)));
/** Rodrigues rotation of v about unit axis k by ang (rad). */
export function rotAxis(v, k, ang) {
  const c = Math.cos(ang), s = Math.sin(ang);
  return V.add(V.add(V.mul(v, c), V.mul(V.cross(k, v), s)), V.mul(k, V.dot(k, v) * (1 - c)));
}

// ---------------------------------------------------------------- parameters
/**
 * Default parameters. See README.md for meaning and sources.
 * D may be a number (mm) or 'auto' (= diameter giving the smallest spot for the given aperture).
 */
export const DEFAULTS = Object.freeze({
  R: 16,        // mm, parabolic-mirror effective focal length (illuminated chief point to focus)
  theta: 60,    // deg, angle of incidence on the sample (from the normal)
  beta: 45,     // deg, azimuth of the plane of incidence from the front–back (Y) axis, toward the port side
  alpha: 45,    // deg, angle of incidence on the steering mirror
  dist: 250,    // mm, steering mirror to entrance-port plane
  L: 37.7,      // mm, lever arm: actuator contact to pivot (KS1-Z8: 0.2 µm step ↔ 1.1″)
  lam: 820,     // nm, pump wavelength
  D: 'auto',    // mm (1/e² diameter of the input beam) or 'auto'
  NA: 0.42,     // illumination aperture of the parabolic mirror (cone half-angle = asin(NA) around the aligned chief ray)
  zLow: 1,      // mm, height of the parabolic mirror's lower edge above the sample
  portX: 260,   // mm, x of the entrance-port plane (only used for the beam shift at the port)
  foldY: 6,     // mm, y of the fold mirror centre (behind the tip)
  yawSide: 1,   // +1: horizontal actuator is right of the pivot seen from behind (KS1-Z8 upright); −1: left
  pitchSide: 1, // +1: vertical actuator is above the pivot (KS1-Z8 upright); −1: below
});

/** Key over every parameter that affects the optics (used by the two caches in this module). */
const paramKey = (p) => [p.R, p.theta, p.beta, p.alpha, p.dist, p.L, p.lam, p.D, p.NA, p.zLow, p.portX, p.foldY, p.yawSide, p.pitchSide].join('|');
const autoDCache = new Map();

/** Merge with defaults and resolve D:'auto' (computed once per parameter set, then cached). */
export function resolveParams(params) {
  const p = Object.assign({}, DEFAULTS, params || {});
  if (p.D === 'auto' || p.D === undefined || p.D === null) {
    const key = paramKey(Object.assign({}, p, { D: 'auto' }));
    if (!autoDCache.has(key)) autoDCache.set(key, Math.round(optimalBeamDiameter(Object.assign({}, p, { D: 13 })) * 10) / 10);
    p.D = autoDCache.get(key);
    p.Dauto = true;
  }
  return p;
}

// ---------------------------------------------------------------- geometry and ray tracing
export function geometry(p) {
  const th = p.theta * Math.PI / 180, be = p.beta * Math.PI / 180, R = p.R;
  const P = [R * Math.sin(th) * Math.sin(be), -R * Math.sin(th) * Math.cos(be), R * Math.cos(th)]; // PM chief point
  const fp = R * (1 + Math.sin(th) * Math.cos(be)) / 2; // parent focal length (vertex at y = −fp)
  const x0 = P[0], h = P[2], yf = p.foldY;
  const F1 = [x0, yf, h], nF1 = V.norm([1, -1, 0]); // fold behind the tip: −X → −Y
  const uIn = [-1, 0, 0];                             // beam direction after the steering mirror (aligned)
  const S = [p.portX + p.dist, yf, h];                // steering-mirror centre
  const defl = Math.PI - 2 * p.alpha * Math.PI / 180;
  const d0 = V.norm(rotAxis(uIn, [0, 0, 1], defl));   // input beam direction (−Y for alpha = 45°)
  const n0 = V.norm(V.sub(uIn, d0));                  // aligned mirror normal (horizontal)
  const vAx = [0, 0, 1];
  const uBack = V.norm(V.cross(n0, vAx));             // viewer's right when standing behind the mount
  return { th, be, R, P, fp, x0, h, yf, F1, nF1, uIn, S, d0, n0, vAx, uBack };
}

/** Mirror normal for push-rod extensions dp (vertical/upper actuator) and dy (horizontal/lower), both in mm. */
export function mirrorNormal(g, p, dp, dy) {
  const ap = -p.pitchSide * (dp / p.L);   // about uBack; + tilts the normal up
  const ay = p.yawSide * (dy / p.L);      // about +Z; + is counter-clockwise seen from above
  return V.norm(rotAxis(rotAxis(g.n0, g.uBack, ap), g.vAx, ay));
}

export function hitPlane(o, d, c, n) {
  const den = V.dot(d, n); if (Math.abs(den) < 1e-12) return null;
  const t = V.dot(V.sub(c, o), n) / den; return t > 1e-9 ? t : null;
}
/** Smallest positive t with o + t·d on x² + z² = 4·fp·(y + fp) (numerically stable roots). */
export function hitParabY(o, d, fp) {
  const a = d[0] * d[0] + d[2] * d[2];
  const b = 2 * (o[0] * d[0] + o[2] * d[2]) - 4 * fp * d[1];
  const c = o[0] * o[0] + o[2] * o[2] - 4 * fp * (o[1] + fp);
  const disc = b * b - 4 * a * c; if (disc < 0) return null;
  const sq = Math.sqrt(disc), q = -0.5 * (b + (b >= 0 ? sq : -sq));
  const ts = []; if (Math.abs(a) > 1e-300) ts.push(q / a); if (Math.abs(q) > 1e-300) ts.push(c / q);
  const pos = ts.filter((t) => t > 1e-9).sort((x, y) => x - y); return pos.length ? pos[0] : null;
}
export const parabNormalY = (pt, fp) => V.norm([2 * pt[0], -4 * fp, 2 * pt[2]]);

export function traceIntoPM(o, d, g) {
  const t1 = hitParabY(o, d, g.fp); if (t1 === null) return null;
  const P = V.add(o, V.mul(d, t1));
  const r = V.norm(reflect(d, parabNormalY(P, g.fp))); if (r[2] >= 0) return null;
  return { P, r, Q: V.add(P, V.mul(r, -P[2] / r[2])) };
}
/** From a point o with direction d (after the steering mirror): fold mirror → parabolic mirror → sample. */
export function traceAfterSteer(o, d, g) {
  const t0 = hitPlane(o, d, g.F1, g.nF1); if (t0 === null) return null;
  const A = V.add(o, V.mul(d, t0)); const d2 = V.norm(reflect(d, g.nF1));
  const rest = traceIntoPM(A, d2, g); if (!rest) return null;
  return Object.assign({ A, d2 }, rest);
}
/** Chief ray for push-rod extensions dp (upper/vertical) and dy (lower/horizontal) in mm. */
export function traceChief(p, dp, dy) {
  const g = geometry(p); const n = mirrorNormal(g, p, dp, dy);
  const d1 = V.norm(reflect(g.d0, n)); const res = traceAfterSteer(g.S, d1, g);
  return Object.assign({ g, n, d1 }, res || {});
}
export function perpFrame(d) {
  let e1 = V.cross(d, [0, 0, 1]); if (V.len(e1) < 1e-9) e1 = V.cross(d, [0, 1, 0]); e1 = V.norm(e1);
  return [e1, V.norm(V.cross(e1, d))];
}
const portPoint = (cc, p) => { const t = (p.portX - cc.g.S[0]) / cc.d1[0]; return V.add(cc.g.S, V.mul(cc.d1, t)); };

// ---------------------------------------------------------------- steering response (chief ray, exact)
/**
 * Where the pump spot goes for given push-rod travels (µm).
 * @returns {{spot_um:number[], port_um:number[], perUm:{yaw:number[],pitch:number[]}, perUmPort:{yaw:number[],pitch:number[]},
 *   incidencePlane:number[], along_um:number, across_um:number, defocus_um:number, arrival_fs:number, yawTurnsBeamToFront:boolean}}
 *   spot_um = [x, y] chief-ray hit point on the sample relative to the aligned one (x: + port side, y: + back);
 *   port_um = [y, z] beam shift in the port plane (y: + back, z: + up);
 *   perUm = linear coefficients (µm of spot per µm of travel, forward difference over +1 µm).
 */
export function steeringResponse(params, yaw_um = 0, pitch_um = 0) {
  const p = Object.assign({}, DEFAULTS, params || {});
  const c0 = traceChief(p, 0, 0), c = traceChief(p, pitch_um / 1000, yaw_um / 1000);
  const cy1 = traceChief(p, 0, 1e-3), cp1 = traceChief(p, 1e-3, 0);
  if (!c0.Q || !c.Q || !cy1.Q || !cp1.Q) throw new Error('chief ray misses the parabolic mirror for these parameters');
  const jy = V.mul(V.sub(cy1.Q, c0.Q), 1000), jp = V.mul(V.sub(cp1.Q, c0.Q), 1000);
  const jyPort = V.mul(V.sub(portPoint(cy1, p), portPoint(c0, p)), 1000), jpPort = V.mul(V.sub(portPoint(cp1, p), portPoint(c0, p)), 1000);
  const dQ = V.mul(V.sub(c.Q, c0.Q), 1000), dPort = V.mul(V.sub(portPoint(c, p), portPoint(c0, p)), 1000);
  const ph = V.norm([c0.r[0], c0.r[1], 0]); // horizontal direction of the incident chief ray (plane of incidence)
  const along = V.dot(dQ, ph), across = V.dot(dQ, [-ph[1], ph[0], 0]);
  const th = p.theta * Math.PI / 180;
  return {
    spot_um: [dQ[0], dQ[1]],
    port_um: [dPort[1], dPort[2]],
    perUm: { yaw: [jy[0], jy[1]], pitch: [jp[0], jp[1]] },
    perUmPort: { yaw: [jyPort[1], jyPort[2]], pitch: [jpPort[1], jpPort[2]] },
    incidencePlane: [ph[0], ph[1]],
    along_um: along, across_um: across,
    defocus_um: Math.abs(along) * Math.sin(th),           // along the beam, ≈ s∥·sinθ
    arrival_fs: Math.abs(along) * Math.sin(th) * 3.33564, // µm → fs (1/c)
    yawTurnsBeamToFront: V.dot(V.sub(cy1.d1, c0.d1), [0, -1, 0]) > 0,
  };
}

// ---------------------------------------------------------------- spot engine (scalar Debye sum on the tilted sample)
export const QUALITY = Object.freeze({ fast: { M: 48, Nmin: 30 }, fine: { M: 72, Nmin: 40 }, trend: { M: 40, Nmin: 28 } });

/**
 * One spot calculation on a fixed window. dp, dy in mm (push-rod travel of the upper/vertical and
 * lower/horizontal actuator). opt = {M, Nmin, W (µm half-width of the window), peakOnly?}.
 * Returns null if fewer than 20 rays reach the sample.
 */
export function computeSpotRaw(P, dp, dy, opt) {
  const g = geometry(P); const nrm = mirrorNormal(g, P, dp, dy); const d1 = V.norm(reflect(g.d0, nrm));
  const ch = traceAfterSteer(g.S, d1, g); if (!ch) return null;
  const x0 = ch.Q; const fr = perpFrame(d1), e1 = fr[0], e2 = fr[1];
  const k = 2 * Math.PI / (P.lam * 1e-6); // 1/mm
  // clear aperture of the parabolic mirror (fixed in space): a cone of half-angle asin(NA) around the
  // aligned chief ray as seen from the focus, and the mirror's lower edge zLow above the sample
  const apA = Math.asin(Math.min(0.95, P.NA)), p0 = V.norm(g.P), zLow = P.zLow;
  const w = P.D / 2, rmax = Math.min(1.8 * w, 1.3 * P.R * Math.tan(apA));
  const per = (P.lam * 1e-3) * P.R / (2 * rmax); // aliasing period on the sample per pupil sample count (µm)
  const W = opt.W; const N = Math.max(opt.Nmin, Math.min(120, Math.ceil(2 * W / (0.8 * per))));
  const cap = N * N, RX = new Float64Array(cap), RY = new Float64Array(cap), AR = new Float64Array(cap), AI = new Float64Array(cap);
  let nr = 0, ph0 = null, asum = 0, a2all = 0;
  for (let i = 0; i < N; i++) {
    const a = -rmax + 2 * rmax * (i + 0.5) / N;
    for (let j = 0; j < N; j++) {
      const b = -rmax + 2 * rmax * (j + 0.5) / N; const rr = a * a + b * b; if (rr > rmax * rmax) continue;
      const A0 = Math.exp(-rr / (w * w)); a2all += A0 * A0;
      const o = [g.S[0] + a * e1[0] + b * e2[0], g.S[1] + a * e1[1] + b * e2[1], g.S[2] + a * e1[2] + b * e2[2]];
      const tr = traceAfterSteer(o, d1, g); if (!tr || !tr.P) continue;
      // soft (area-weighted) aperture edges so the clipped power does not depend on the sampling grid
      const cell = 2 * rmax / N, Lp = V.len(tr.P);
      const mEdge = (apA - Math.acos(Math.min(1, V.dot(tr.P, p0) / Lp))) * Lp, mLow = tr.P[2] - zLow;
      const wAp = Math.min(1, Math.max(0, 0.5 + mEdge / cell)) * Math.min(1, Math.max(0, 0.5 + mLow / cell)); if (wAp <= 0) continue;
      const opl = V.len(V.sub(tr.A, o)) + V.len(V.sub(tr.P, tr.A));
      const dx = V.sub(x0, tr.P), Lr = V.len(dx), ph = opl + V.dot(tr.r, dx);
      if (ph0 === null) ph0 = ph;
      const amp = A0 * wAp / Lr, phi = k * (ph - ph0);
      RX[nr] = tr.r[0]; RY[nr] = tr.r[1]; AR[nr] = amp * Math.cos(phi); AI[nr] = amp * Math.sin(phi); asum += amp; nr++;
    }
  }
  if (nr < 20) return null;
  // a2all also counts the light the aperture cuts away, so IperP is the peak intensity per unit laser power (1/µm²)
  const dA = (2 * rmax / N) * (2 * rmax / N), lamMM = P.lam * 1e-6, perPow = asum * asum * dA / (lamMM * lamMM * a2all) * 1e-6;
  if (opt.peakOnly) { let sr = 0, si = 0; for (let i = 0; i < nr; i++) { sr += AR[i]; si += AI[i]; } return { IperP: (sr * sr + si * si) / (asum * asum) * perPow, nr }; }
  const M = opt.M, Wmm = W * 1e-3, du = 2 * Wmm / (M - 1);
  const ER = new Float64Array(M * M), EI = new Float64Array(M * M);
  const gR = new Float64Array(M), gI = new Float64Array(M), hR = new Float64Array(M), hI = new Float64Array(M);
  for (let i = 0; i < nr; i++) {
    const kx = k * RX[i], ky = k * RY[i], ar = AR[i], ai = AI[i];
    let cr = Math.cos(-kx * Wmm), ci = Math.sin(-kx * Wmm); const sr = Math.cos(kx * du), si = Math.sin(kx * du);
    for (let a = 0; a < M; a++) { gR[a] = ar * cr - ai * ci; gI[a] = ar * ci + ai * cr; const t = cr * sr - ci * si; ci = cr * si + ci * sr; cr = t; }
    let qr = Math.cos(-ky * Wmm), qi = Math.sin(-ky * Wmm); const tr2 = Math.cos(ky * du), ti2 = Math.sin(ky * du);
    for (let b = 0; b < M; b++) { hR[b] = qr; hI[b] = qi; const t = qr * tr2 - qi * ti2; qi = qr * ti2 + qi * tr2; qr = t; }
    for (let b = 0; b < M; b++) {
      const hr = hR[b], hi = hI[b], row = b * M;
      for (let a = 0; a < M; a++) { ER[row + a] += gR[a] * hr - gI[a] * hi; EI[row + a] += gR[a] * hi + gI[a] * hr; }
    }
  }
  const nrm2 = 1 / (asum * asum); const I = new Float32Array(M * M); let imax = 0, ia = 0;
  for (let q = 0; q < M * M; q++) { const v = (ER[q] * ER[q] + EI[q] * EI[q]) * nrm2; I[q] = v; if (v > imax) { imax = v; ia = q; } }
  // refine the peak off-grid by direct evaluation (pattern search)
  const evalAt = (u, v) => { let sr = 0, si = 0; for (let i = 0; i < nr; i++) { const ph = k * (RX[i] * u + RY[i] * v), c = Math.cos(ph), s = Math.sin(ph); sr += AR[i] * c - AI[i] * s; si += AR[i] * s + AI[i] * c; } return (sr * sr + si * si) * nrm2; };
  let pu = -Wmm + (ia % M) * du, pv = -Wmm + Math.floor(ia / M) * du, pk = imax, h = du / 2;
  for (let it = 0; it < 16 && h > du / 40; it++) {
    let best = null;
    for (const st of [[h, 0], [-h, 0], [0, h], [0, -h]]) { const v = evalAt(pu + st[0], pv + st[1]); if (v > pk) { pk = v; best = [pu + st[0], pv + st[1]]; } }
    if (best) { pu = best[0]; pv = best[1]; } else h /= 2;
  }
  // half-maximum region (soft edge 0.45–0.55), fitted with the ellipse of equal second moments
  const px = 2 * W / (M - 1); let S = 0, sx = 0, sy = 0;
  const f = new Float32Array(M * M);
  for (let q = 0; q < M * M; q++) { const r = I[q] / pk; const t = r <= 0.45 ? 0 : (r >= 0.55 ? 1 : (r - 0.45) / 0.1); f[q] = t; if (t) { const a = q % M, b = (q - a) / M; S += t; sx += t * (-W + a * px); sy += t * (-W + b * px); } }
  const cx = sx / S, cy = sy / S; let xx = 0, yy = 0, xy = 0;
  for (let q = 0; q < M * M; q++) { const t = f[q]; if (!t) continue; const a = q % M, b = (q - a) / M; const X = -W + a * px - cx, Y = -W + b * px - cy; xx += t * X * X; yy += t * Y * Y; xy += t * X * Y; }
  xx /= S; yy /= S; xy /= S; const trc = xx + yy, det = xx * yy - xy * xy, disc = Math.sqrt(Math.max(0, trc * trc / 4 - det));
  const l1 = trc / 2 + disc, l2 = Math.max(0, trc / 2 - disc);
  let border = 0; for (let a = 0; a < M; a++) border = Math.max(border, I[a], I[(M - 1) * M + a], I[a * M], I[a * M + M - 1]);
  return {
    I, M, W, x0, N, nr,
    peak: pk,                       // peak / perfect-focus peak of the transmitted light (Strehl-like)
    IperP: pk * perPow,             // peak intensity per unit laser power, 1/µm² (|E|² on the sample plane)
    peakAt: [pu * 1e3, pv * 1e3],   // µm, brightest point relative to x0
    cx, cy,                         // µm, centroid of the half-max region relative to x0
    fwhmL: 4 * Math.sqrt(l1), fwhmS: 4 * Math.sqrt(l2), // µm, half-max ellipse full axes
    ang: 0.5 * Math.atan2(2 * xy, xx - yy),            // rad, long-axis angle from +X toward +Y
    area: S * px * px,              // µm², half-max area
    border: border / pk,            // max intensity on the window border / peak
  };
}

/** Spot with the window adapted so the spot fits (border < 3 %) and stays resolved. q = QUALITY preset. */
export function adaptiveSpotRaw(P, dp, dy, q, Wstart) {
  let W = Math.max(2.4, Math.min(18, Wstart || 3.5)), s = null, shrunk = false;
  for (let it = 0; it < 5; it++) {
    s = computeSpotRaw(P, dp, dy, { M: q.M, Nmin: q.Nmin, W }); if (!s) return null;
    if (s.border > 0.03 && W < 18) { W = Math.min(18, W * 1.6); continue; }
    const want = Math.max(2.4, 1.3 * s.fwhmL);
    if (!shrunk && s.border < 0.004 && W > 2 * want) { W = Math.max(2.4, 1.25 * want); shrunk = true; continue; }
    break;
  }
  return s;
}

/** Beam diameter (mm, 1/e²) with the highest peak intensity per unit laser power for the aligned beam. */
export function optimalBeamDiameter(params) {
  const p = Object.assign({}, DEFAULTS, params || {});
  const f = (D) => { const r = computeSpotRaw(Object.assign({}, p, { D }), 0, 0, { M: 3, Nmin: 44, W: 0.5, peakOnly: true }); return r ? r.IperP : 0; };
  let bD = 3, b = -1;
  for (let D = 3; D <= 26.001; D += 0.5) { const v = f(D); if (v > b) { b = v; bD = D; } }
  const a = f(bD - 0.5), c = f(bD + 0.5), den = a - 2 * b + c; let D = bD;
  if (den < 0) D = bD + 0.25 * (a - c) / den;
  return Math.max(3, Math.min(26, D));
}

const refCache = new Map();
/** Aligned spot (both actuators at 0), fine quality, cached. p must be resolved (numeric D). Treat the result as read-only. */
export function referenceSpot(p) {
  const k = paramKey(p); if (refCache.has(k)) return refCache.get(k);
  const s = adaptiveSpotRaw(p, 0, 0, QUALITY.fine, 3.5); refCache.set(k, s); return s;
}

/**
 * Spot shape for push-rod travels (µm). quality: 'fine' | 'fast' | 'trend' (or a {M, Nmin} object).
 * Adds rel = peak intensity relative to the aligned spot (power-normalised, includes clipping),
 * and k = factor that converts I (Strehl-normalised) into intensity relative to the aligned peak.
 */
export function spotShape(params, yaw_um = 0, pitch_um = 0, quality = 'fine', Wstart) {
  const p = resolveParams(params);
  const ref = referenceSpot(p);
  const q = typeof quality === 'string' ? QUALITY[quality] : quality;
  let s = (yaw_um === 0 && pitch_um === 0 && q === QUALITY.fine) ? ref : adaptiveSpotRaw(p, pitch_um / 1000, yaw_um / 1000, q, Wstart || (ref ? ref.W : 3.5));
  if (!s || !ref) return null;
  if (s === ref) s = Object.assign({}, ref, { I: ref.I.slice(), x0: ref.x0.slice(), peakAt: ref.peakAt.slice() }); // never hand out the cached arrays
  return Object.assign({}, s, { rel: s.IperP / ref.IperP, k: (s.IperP / ref.IperP) / s.peak, D: p.D });
}

/** Peak intensity (relative to the aligned spot) and FWHM along one actuator's travel. axis: 'yaw' | 'pitch'. */
export function peakCurve(params, axis, travels_um, quality = 'trend') {
  const p = resolveParams(params); const q = typeof quality === 'string' ? QUALITY[quality] : quality;
  // normalise with a reference of the same quality so that travel 0 reads exactly 100 %
  const ref0 = adaptiveSpotRaw(p, 0, 0, q, 3.5);
  let W = ref0 ? ref0.W : 3.5;
  return travels_um.map((t) => {
    const s = adaptiveSpotRaw(p, axis === 'pitch' ? t / 1000 : 0, axis === 'yaw' ? t / 1000 : 0, q, W);
    if (!s) return { travel_um: t, rel: null };
    W = s.W;
    return { travel_um: t, rel: s.IperP / ref0.IperP, fwhmL: s.fwhmL, fwhmS: s.fwhmS, area: s.area, ang: s.ang };
  });
}

// ---------------------------------------------------------------- contour (marching squares)
const MS = { 1: [[3, 0]], 2: [[0, 1]], 3: [[3, 1]], 4: [[1, 2]], 5: [[3, 0], [1, 2]], 6: [[0, 2]], 7: [[3, 2]], 8: [[2, 3]], 9: [[0, 2]], 10: [[0, 1], [2, 3]], 11: [[1, 2]], 12: [[1, 3]], 13: [[0, 1]], 14: [[3, 0]] };
/** Iso-line I = level of a spot as a flat list of segments [x1,y1,x2,y2, ...] in µm relative to x0. */
export function contour(s, level) {
  const { I, M, W } = s, px = 2 * W / (M - 1), out = [];
  const Pt = (a, b) => [-W + a * px, -W + b * px];
  const lerp = (p1, v1, p2, v2) => { const t = (level - v1) / (v2 - v1); return [p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t]; };
  for (let b = 0; b < M - 1; b++) for (let a = 0; a < M - 1; a++) {
    const v0 = I[b * M + a], v1 = I[b * M + a + 1], v2 = I[(b + 1) * M + a + 1], v3 = I[(b + 1) * M + a];
    const c = (v0 > level ? 1 : 0) | (v1 > level ? 2 : 0) | (v2 > level ? 4 : 0) | (v3 > level ? 8 : 0); if (c === 0 || c === 15) continue;
    const p0 = Pt(a, b), p1 = Pt(a + 1, b), p2 = Pt(a + 1, b + 1), p3 = Pt(a, b + 1);
    const e = [() => lerp(p0, v0, p1, v1), () => lerp(p1, v1, p2, v2), () => lerp(p2, v2, p3, v3), () => lerp(p3, v3, p0, v0)];
    for (const sg of MS[c]) { const A = e[sg[0]](), B = e[sg[1]](); out.push(A[0], A[1], B[0], B[1]); }
  }
  return out;
}
