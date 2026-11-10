// Regression test of the JS reference against tests/golden.json.
//   node js/test_golden.mjs        (exit code 0 = pass)
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import * as O from './ltem_optics.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const G = JSON.parse(readFileSync(join(here, '..', 'tests', 'golden.json'), 'utf8'));
const tol = G.tolerance;
let fails = 0, checks = 0;

function close(a, b, rel, abs) { return Math.abs(a - b) <= Math.max(abs, rel * Math.max(Math.abs(a), Math.abs(b))); }
function cmp(path, got, exp, rel, abs) {
  if (exp === null || exp === undefined) { if (got !== null && got !== undefined) { fails++; console.log('FAIL', path, 'expected null, got', got); } return; }
  if (typeof exp === 'number') { checks++; if (typeof got !== 'number' || !close(got, exp, rel, abs)) { fails++; console.log('FAIL', path, 'got', got, 'expected', exp); } return; }
  if (typeof exp === 'boolean' || typeof exp === 'string') { checks++; if (got !== exp) { fails++; console.log('FAIL', path, 'got', got, 'expected', exp); } return; }
  if (Array.isArray(exp)) { exp.forEach((e, i) => cmp(path + '[' + i + ']', got ? got[i] : undefined, e, rel, abs)); return; }
  for (const k of Object.keys(exp)) cmp(path + '.' + k, got ? got[k] : undefined, exp[k], rel, abs);
}
function spotSummary(s) {
  if (!s) return null;
  let sum = 0; for (const v of s.I) sum += v;
  return {
    M: s.M, W: s.W, N: s.N, nr: s.nr, rel: s.rel, peak: s.peak, IperP: s.IperP, fwhmL: s.fwhmL, fwhmS: s.fwhmS,
    ang: s.ang, area: s.area, peakAt: s.peakAt, cx: s.cx, cy: s.cy, border: s.border, x0_mm: s.x0, I_sum: sum,
    I_samples: null, halfmax_segments: O.contour(s, 0.5 * s.peak).length / 4,
  };
}

for (const [name, c] of Object.entries(G.cases)) {
  const p = O.resolveParams(c.params_in);
  cmp(name + '.D', p.D, c.params_resolved.D, 0, tol.D_abs_mm);
  cmp(name + '.optimal_D_mm', O.optimalBeamDiameter(p), c.optimal_D_mm, 0, tol.D_abs_mm);
  for (const r of c.responses) cmp(`${name}.response(${r.yaw_um},${r.pitch_um})`, O.steeringResponse(p, r.yaw_um, r.pitch_um), r.result, 1e-9, tol.response_abs_um);
  for (const s of c.spots) {
    const got = O.spotShape(p, s.yaw_um, s.pitch_um, s.quality);
    const sum = spotSummary(got);
    const exp = Object.assign({}, s.result); const samples = exp.I_samples; delete exp.I_samples;
    cmp(`${name}.spot(${s.yaw_um},${s.pitch_um},${s.quality})`, sum, exp, tol.spot_rel, tol.spot_abs);
    for (const smp of samples) cmp(`${name}.spot(${s.yaw_um},${s.pitch_um}).I[${smp.row},${smp.col}]`, got.I[smp.row * got.M + smp.col], smp.value, tol.spot_rel, 1e-9);
  }
  for (const ax of ['yaw', 'pitch']) {
    const travels = c.curves[ax].map((o) => o.travel_um);
    cmp(`${name}.curve.${ax}`, O.peakCurve(p, ax, travels, 'trend'), c.curves[ax], tol.curve_rel, tol.spot_abs);
  }
}
// mutating a returned spot must not change later results (the aligned spot is cached internally)
{
  const p = O.resolveParams({}); const a = O.spotShape(p, 0, 0); a.I.fill(0); a.x0[0] = 1e9;
  const b = O.spotShape(p, 0, 0); let mx = 0; for (const v of b.I) mx = Math.max(mx, v);
  checks++; if (!(mx > 0.9 && Math.abs(b.x0[0]) < 1)) { fails++; console.log('FAIL returned spot aliases the cached reference spot'); }
}
console.log(fails ? `FAILED ${fails} of ${checks} checks` : `OK — ${checks} checks passed`);
process.exit(fails ? 1 : 0);
