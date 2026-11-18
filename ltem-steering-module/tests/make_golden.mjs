// Regenerates tests/golden.json from the JS reference implementation.
//   node tests/make_golden.mjs
// Only rerun this when the physics is changed on purpose; both test suites compare against this file.
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import * as O from '../js/ltem_optics.mjs';

const here = dirname(fileURLToPath(import.meta.url));

const PARAM_SETS = {
  default: {},
  alt: { R: 12, theta: 55, beta: 30, alpha: 40, dist: 180, L: 35, lam: 780, D: 8, NA: 0.46 },
  flipped: { yawSide: -1, pitchSide: -1, D: 10 },
};
const RESPONSE_POINTS = [[0, 0], [4, 0], [0, 5], [-7.3, 2.1], [20, -20]];
const SPOT_POINTS = [[0, 0, 'fine'], [5, 0, 'fine'], [0, 5, 'fine'], [-10, 3, 'fine'], [12, 0, 'fast']];
const CURVE_TRAVELS = [-10, -5, -2, 0, 2, 5, 10];

function spotSummary(s) {
  if (!s) return null;
  let sum = 0; for (const v of s.I) sum += v;
  const mid = (s.M - 1) / 2, pick = [[0, 0], [Math.floor(mid), Math.floor(mid)], [Math.floor(s.M / 3), Math.floor(2 * s.M / 3)], [s.M - 1, s.M - 1]];
  return {
    M: s.M, W: s.W, N: s.N, nr: s.nr,
    rel: s.rel, peak: s.peak, IperP: s.IperP,
    fwhmL: s.fwhmL, fwhmS: s.fwhmS, ang: s.ang, area: s.area,
    peakAt: s.peakAt, cx: s.cx, cy: s.cy, border: s.border,
    x0_mm: s.x0,
    I_sum: sum,
    I_samples: pick.map(([b, a]) => ({ row: b, col: a, value: s.I[b * s.M + a] })),
    halfmax_segments: O.contour(s, 0.5 * s.peak).length / 4,
  };
}

const out = {
  generated_by: 'tests/make_golden.mjs from js/ltem_optics.mjs',
  units: 'travel in µm; spot/port shifts in µm; W, fwhm, peakAt, cx, cy in µm; area in µm²; x0 in mm; ang in rad; IperP in 1/µm²',
  tolerance: { response_abs_um: 1e-9, spot_rel: 1e-6, spot_abs: 1e-6, curve_rel: 1e-6, D_abs_mm: 1e-9 },
  cases: {},
};

for (const [name, raw] of Object.entries(PARAM_SETS)) {
  const p = O.resolveParams(raw);
  const c = { params_in: raw, params_resolved: p, optimal_D_mm: O.optimalBeamDiameter(p), responses: [], spots: [], curves: {} };
  for (const [y, q] of RESPONSE_POINTS) c.responses.push({ yaw_um: y, pitch_um: q, result: O.steeringResponse(p, y, q) });
  for (const [y, q, qual] of SPOT_POINTS) c.spots.push({ yaw_um: y, pitch_um: q, quality: qual, result: spotSummary(O.spotShape(p, y, q, qual)) });
  for (const ax of ['yaw', 'pitch']) c.curves[ax] = O.peakCurve(p, ax, CURVE_TRAVELS, 'trend');
  out.cases[name] = c;
  console.log(name, 'D =', p.D, 'optimal D =', c.optimal_D_mm.toFixed(3));
}
writeFileSync(join(here, 'golden.json'), JSON.stringify(out, null, 1));
console.log('wrote tests/golden.json');
