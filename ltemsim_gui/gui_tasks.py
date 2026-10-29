"""Task functions the GUI runs in a subprocess (through ltemsim_gui.runner) when a script's own
entry point is too rigid.  They reuse the scripts' code instead of copying it.

Run as: runner spec {"script": "ltemsim_gui/gui_tasks.py", "func": "<name>", "kwargs": {...}}
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (os.path.join(ROOT, "scripts"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def custom_sweep(root, key, values, base=None, name=None, verbose=True):
    """Sweep any Params field over `values` on top of `base` overrides (e.g. a coarse grid).

    Writes <root>/<name>/<key>=<value>/... and <root>/<name>/compare.png exactly like scripts/sweep.py.
    With verbose=True the Newton progress of every solve is printed (sweep.py runs them silently).
    """
    import sweep
    from ltemsim import Simulation

    base = dict(base or {})
    name = name or key
    orig_run_one = sweep.run_one

    def run_one(out, **kw):
        print(f"@@SWEEP-START {key}={kw.get(key)}", flush=True)
        return orig_run_one(out, **{**base, **kw})

    class LoudSimulation(Simulation):
        def __init__(self, params, verbose=False):
            super().__init__(params, verbose=loud)

    loud = verbose
    sweep.run_one = run_one
    sweep.Simulation = LoudSimulation
    sweep.SWEEPS[name] = (key, list(values))
    os.makedirs(os.path.join(root, name), exist_ok=True)
    with open(os.path.join(root, name, "sweep_spec.json"), "w") as fh:
        json.dump(dict(key=key, values=list(values), base=base), fh, indent=1)
    sweep.main(name, root, list(values))
    print(f"@@SWEEP-DONE {os.path.join(root, name)}", flush=True)


def fdtd_heights_summary(out="results/fdtd_pump/heights", ref="results/fdtd_pump/notip.npz"):
    """Rebuild heights/summary.json from all ratio_h*.npz present (run_fdtd_heights.py keeps only the last batch)."""
    import glob
    import re
    import numpy as np
    rn = np.load(ref, allow_pickle=True)
    summary = {}
    files = []
    for f in glob.glob(os.path.join(out, "ratio_h*.npz")):
        m = re.fullmatch(r"ratio_h(\d+)\.npz", os.path.basename(f))
        if m:
            files.append((int(m.group(1)), f))
        else:
            print(f"skip {os.path.basename(f)}: unexpected name", flush=True)
    for h, f in sorted(files):
        try:
            d = np.load(f)
            k0 = np.argmin(np.abs(d["z"] + 0.02e-6)); i0 = np.argmin(np.abs(d["x"]))
            row = dict(absorbed_fraction=float(d["absorbed"] / rn["launched"]),
                       surface_ratio_under_apex=float(d["ratio"][i0, k0]))
        except Exception as e:         # incomplete file from a cancelled run
            print(f"skip h = {h} nm: unreadable ({type(e).__name__}: {e})", flush=True)
            continue
        summary[h] = row
        print(f"h = {h} nm: absorbed {row['absorbed_fraction']:.3f}", flush=True)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
