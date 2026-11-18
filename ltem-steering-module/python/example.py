"""Minimal usage example.  python python/example.py  [--plot]

Prints where the pump spot goes for a few actuator settings and how its shape changes;
with --plot (needs matplotlib) also writes example_spots.png next to this file.
"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ltem_optics as O  # noqa: E402

params = O.resolve_params({})          # defaults; D='auto' → beam diameter for the smallest spot
print(f"beam diameter D = {params['D']:.1f} mm (auto), NA = {params['NA']}")

r = O.steering_response(params, 1.0, 0.0)
print("per µm of push-rod travel (spot shift in µm, x = port side, y = back):")
print(f"  lower / horizontal: {r['perUm']['yaw'][0]:+.3f}, {r['perUm']['yaw'][1]:+.3f}")
print(f"  upper / vertical:   {r['perUm']['pitch'][0]:+.3f}, {r['perUm']['pitch'][1]:+.3f}")

for yaw, pitch in [(4, 0), (0, 5), (-2, 1)]:
    r = O.steering_response(params, yaw, pitch)
    s = O.spot_shape(params, yaw, pitch, "fine")
    print(f"yaw {yaw:+.1f} µm, pitch {pitch:+.1f} µm → spot shift ({r['spot_um'][0]:+.2f}, {r['spot_um'][1]:+.2f}) µm, "
          f"defocus {r['defocus_um']:.2f} µm, FWHM {s['fwhmL']:.2f} × {s['fwhmS']:.2f} µm, peak {100 * s['rel']:.0f} %")

curve = O.peak_curve(params, "yaw", [-10, -5, -2, 0, 2, 5, 10])
print("horizontal actuator, peak intensity:", ", ".join(f"{c['travel_um']:+d}: {100 * c['rel']:.0f}%" for c in curve))

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, (yaw, pitch) in zip(axes, [(0, 0), (5, 0), (0, 5)]):
        s = O.spot_shape(params, yaw, pitch, "fine")
        W = s["W"]
        ax.imshow(s["I"] * s["k"], origin="lower", extent=[-W, W, -W, W], cmap="magma", vmin=0, vmax=1)
        for x1, y1, x2, y2 in O.contour(s, 0.5 * s["peak"]):
            ax.plot([x1, x2], [y1, y2], color="w", lw=1)
        ax.set_title(f"yaw {yaw} µm, pitch {pitch} µm\npeak {100 * s['rel']:.0f} %, FWHM {s['fwhmL']:.2f}×{s['fwhmS']:.2f} µm", fontsize=9)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_facecolor("black")  # outside the computed window the intensity is ~0
        ax.set_xlabel("x (µm, + port side)")
        ax.set_ylabel("y (µm, + back)")
    fig.tight_layout()
    out = pathlib.Path(__file__).resolve().parent / "example_spots.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)
