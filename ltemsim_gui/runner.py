"""Subprocess entry point: call a function of one of the project scripts with typed arguments.

    python -u -m ltemsim_gui.runner <spec.json>

spec = {"script": "scripts/run_baseline.py", "func": "main", "args": [...], "kwargs": {...},
        "out": "results/gui_runs/steady_...", "title": "..."}

With "func": null the file is executed as __main__ with sys.argv = [script, *argv]
(for the scripts that have no main() or no __main__ guard).

Going through the script's Python function instead of its command line avoids the
per-script argv parsers (some cannot take an output folder, all turn every value into
a float) and keeps tuples, booleans and strings intact.  Lists in the JSON become tuples.
The working directory is the repository root, as the scripts expect.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _tuplify(v):
    if isinstance(v, list):
        return tuple(_tuplify(x) for x in v)
    return v


def job_tag(spec: dict) -> str:
    return spec.get("func") or Path(spec["script"]).stem


def meta_name(out, spec: dict, base="gui_run", ext=".json") -> str:
    """gui_run.json belongs to the first job that wrote into `out` (re-runs of it overwrite it);
    a different job sharing the folder (e.g. demod after simulate) gets gui_run_<func>.json."""
    p = os.path.join(out, "gui_run.json")
    try:
        with open(p) as fh:
            old = json.load(fh)
    except (OSError, ValueError):
        return base + ext
    same = old.get("script") == spec.get("script") and old.get("func", "main") == spec.get("func", "main")
    return base + ext if same else f"{base}_{job_tag(spec)}{ext}"


def load_script(script: str):
    path = (ROOT / script).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def main(spec_path: str) -> int:
    with open(spec_path) as fh:
        spec = json.load(fh)
    os.chdir(ROOT)
    for p in (str(ROOT / "scripts"), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg")
    # CJK glyphs (some figures have Chinese labels) fall back to a system font instead of empty boxes
    sys.path.insert(0, str(ROOT))
    from ltemsim_gui.fonts import install_cjk_fallback
    install_cjk_fallback()
    import warnings
    warnings.filterwarnings("ignore", message="Glyph .* missing from")

    args = [_tuplify(a) for a in spec.get("args", [])]
    kwargs = {k: _tuplify(v) for k, v in spec.get("kwargs", {}).items()}
    out = spec.get("out")
    t0 = time.time()
    print(f"@@START {spec.get('title', spec['script'])}", flush=True)
    if spec.get("func", "main") is None:
        print(f"@@CALL python {spec['script']} {' '.join(map(str, spec.get('argv', [])))}", flush=True)
    else:
        print(f"@@CALL {spec['script']}:{spec.get('func', 'main')} args={args} kwargs={kwargs}", flush=True)
    meta_path = None
    if out:
        os.makedirs(out, exist_ok=True)
        meta_path = os.path.join(out, meta_name(out, spec))
        meta = dict(spec, started=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.executable)
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=1)
    try:
        if spec.get("func", "main") is None:
            import runpy
            sys.argv = [spec["script"]] + [str(a) for a in spec.get("argv", [])]
            runpy.run_path(str(ROOT / spec["script"]), run_name="__main__")
        else:
            mod = load_script(spec["script"])
            func = getattr(mod, spec.get("func", "main"))
            func(*args, **kwargs)
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f"@@FAILED exit code {e.code}", flush=True)
            return int(e.code) if isinstance(e.code, int) else 1
    except KeyboardInterrupt:
        print("@@CANCELLED", flush=True)
        return 130
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        print(f"@@FAILED after {time.time() - t0:.0f} s", flush=True)
        return 1
    if meta_path:
        try:
            p = meta_path
            meta = json.load(open(p))
            meta.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), wall_s=round(time.time() - t0, 1))
            json.dump(meta, open(p, "w"), indent=1)
        except OSError:
            pass
    print(f"@@DONE in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
