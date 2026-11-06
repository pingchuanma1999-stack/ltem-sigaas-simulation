// Minimal usage example.   node js/example.mjs
import * as O from './ltem_optics.mjs';

const params = O.resolveParams({}); // defaults; D:'auto' → beam diameter for the smallest spot
console.log(`beam diameter D = ${params.D.toFixed(1)} mm (auto), NA = ${params.NA}`);

const r1 = O.steeringResponse(params, 1, 0);
console.log('per µm of push-rod travel (spot shift in µm, x = port side, y = back):');
console.log(`  lower / horizontal: ${r1.perUm.yaw.map((v) => v.toFixed(3)).join(', ')}`);
console.log(`  upper / vertical:   ${r1.perUm.pitch.map((v) => v.toFixed(3)).join(', ')}`);

for (const [yaw, pitch] of [[4, 0], [0, 5], [-2, 1]]) {
  const r = O.steeringResponse(params, yaw, pitch);
  const s = O.spotShape(params, yaw, pitch, 'fine');
  console.log(`yaw ${yaw} µm, pitch ${pitch} µm → spot shift (${r.spot_um.map((v) => v.toFixed(2)).join(', ')}) µm, ` +
    `defocus ${r.defocus_um.toFixed(2)} µm, FWHM ${s.fwhmL.toFixed(2)} × ${s.fwhmS.toFixed(2)} µm, peak ${(100 * s.rel).toFixed(0)} %`);
}

const curve = O.peakCurve(params, 'yaw', [-10, -5, -2, 0, 2, 5, 10]);
console.log('horizontal actuator, peak intensity:', curve.map((c) => `${c.travel_um}: ${(100 * c.rel).toFixed(0)}%`).join(', '));
