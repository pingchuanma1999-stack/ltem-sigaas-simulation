"""Run the independent simulations of a multi-run job on several cores.

The scripts loop over independent runs (sweep values, beam positions X, (X, h) pairs, the three FDTD
runs, tip heights).  These helpers compute those runs in a process pool and then call the script's
own function with the results injected, so summaries and figures come from the script's code exactly
as in a sequential run.  An injected result is used only if the run's configuration equals the one
the script builds itself; otherwise the script simply computes it.

Progress: one line "@@PAR i/n done: <label>" per finished run (count pattern r"^@@PAR \\d+/\\d+ done").
Each worker uses one BLAS thread, so n workers keep n cores busy without oversubscription.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_THREAD_VARS = ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def default_workers(n_runs: int) -> int:
    """Leave a couple of cores for the desktop; never more workers than runs."""
    return max(1, min(n_runs, (os.cpu_count() or 2) - 2))


def _init_worker():
    for v in _THREAD_VARS:
        os.environ[v] = "1"
    for p in (os.path.join(ROOT, "scripts"), ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    import matplotlib
    matplotlib.use("Agg")
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # the parent handles Ctrl-C / Stop and shuts the pool down


def _run_logged(args):
    """Worker side: run fn(item) with this process's stdout/stderr going to the run's own log file."""
    fn, item, log_path = args
    sys.stdout.flush()
    sys.stderr.flush()
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    saved = os.dup(1), os.dup(2)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    for stream in (sys.stdout, sys.stderr):            # unbuffered, so the parent sees progress while the run goes on
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass
    try:
        return fn(item)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])


class _Monitor:
    """Parent side: reads the runs' logs and prints the combined progress ("@@PAR progress f | text")."""

    def __init__(self, logs, sub_spec, n):
        from .progress import ProgressParser
        self.logs, self.n = logs, n
        self.parsers = [ProgressParser(sub_spec) for _ in logs] if sub_spec else None
        self.scale = 0.9 if (sub_spec or {}).get("type") == "steady" else 1.0     # steady parser tops out at 0.9
        self.pos = [0] * len(logs)
        self.running, self.done = set(), set()
        self.last = -1.0

    def poll(self):
        if self.parsers is None:
            return
        part = 0.0
        for i in list(self.running):
            try:
                with open(self.logs[i], errors="replace") as fh:
                    fh.seek(self.pos[i])
                    chunk = fh.read()
                    self.pos[i] = fh.tell()
            except OSError:
                chunk = ""
            for line in chunk.splitlines():
                self.parsers[i].feed(line)
            part += min(0.99, (self.parsers[i].fraction() or 0.0) / self.scale)
        f = (len(self.done) + part) / self.n
        if f - self.last >= 0.005:
            self.last = f
            print(f"@@PAR progress {f:.4f} | {len(self.done)}/{self.n} done · {len(self.running)} running", flush=True)


def _tail(path, n=3):
    try:
        with open(path, errors="replace") as fh:
            lines = [l.rstrip() for l in fh.read().splitlines() if l.strip()]
    except OSError:
        return []
    return lines[-n:]


def _pool_map(fn, items, labels, workers, sub=None):
    """Run fn(item) for every item on `workers` processes; returns results in item order.

    Each run's own output goes to a log file; with a ProgressParser spec `sub` for one run's output the
    combined progress of the runs in flight is printed every ~2 s. The last lines of every finished run's
    log are copied into this job's log.
    """
    import shutil
    import tempfile
    import threading
    n = len(items)
    workers = max(1, min(int(workers or 1), n))
    print(f"@@PAR start: {n} runs on {workers} cores", flush=True)
    for v in _THREAD_VARS:                            # inherited by the spawned workers before numpy loads
        os.environ[v] = "1"
    logdir = tempfile.mkdtemp(prefix="ltem_par_")
    logs = [os.path.join(logdir, f"run{i}.log") for i in range(n)]
    mon = _Monitor(logs, sub, n)
    stop = threading.Event()

    def watch():
        while not stop.wait(2.0):
            mon.poll()
    ex = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_worker)
    results = [None] * n
    t0 = time.time()
    th = threading.Thread(target=watch, daemon=True)
    try:
        futs = {}
        for i, it in enumerate(items):
            futs[ex.submit(_run_logged, (fn, it, logs[i]))] = i
        mon.running = set(range(min(workers, n)))      # the first `workers` runs start right away
        th.start()
        for f in as_completed(futs):
            i = futs[f]
            try:
                results[i] = f.result()
            except Exception:
                for line in _tail(logs[i], 30):
                    print(f"    | {line}", flush=True)
                raise
            mon.done.add(i)
            mon.running.discard(i)
            nxt = len(mon.done) + len(mon.running)
            if nxt < n:
                mon.running.add(nxt)                   # FIFO pool: the next queued run starts now
            print(f"@@PAR {len(mon.done)}/{n} done: {labels[i]} ({time.time() - t0:.0f} s)", flush=True)
            for line in _tail(logs[i]):
                print(f"    | {line}", flush=True)
    except BaseException:
        for p in list(getattr(ex, "_processes", {}).values()):
            try:
                p.terminate()
            except Exception:
                pass
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        stop.set()
        shutil.rmtree(logdir, ignore_errors=True)
    ex.shutdown()
    return results


def _same(a, b) -> bool:
    import numpy as np
    if hasattr(a, "__dataclass_fields__"):
        return type(a) is type(b) and all(_same(getattr(a, k), getattr(b, k)) for k in a.__dataclass_fields__)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


# ============================================================================= steady-state sweep
def _sweep_one(args):
    out, kw = args
    import sweep
    from ltemsim import Simulation

    class LoudSimulation(Simulation):               # Newton progress into this run's log (sweep.py runs silently)
        def __init__(self, params, verbose=False):
            super().__init__(params, verbose=True)
    sweep.Simulation = LoudSimulation
    return sweep.run_one(out, **kw)


def sweep(root, key, values, base=None, name=None, workers=None):
    """Like gui_tasks.custom_sweep, with the solves of all values in parallel."""
    import sweep as sw
    import json
    base = dict(base or {})
    name = name or key
    values = list(values)
    out_root = os.path.join(root, name)
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "sweep_spec.json"), "w") as fh:
        json.dump(dict(key=key, values=values, base=base, workers=workers), fh, indent=1)
    items, labels = [], []
    for v in values:                                   # the same folder names as sweep.main
        label = f"{key}={v}"
        items.append((os.path.join(out_root, label.replace("/", "_")), {**base, key: v}))
        labels.append(label)
    results = _pool_map(_sweep_one, items, labels, workers or default_workers(len(items)), sub={"type": "steady"})
    done = {out: (v, r) for (out, _), v, r in zip(items, values, results)}
    orig = sw.run_one

    def run_one(out, **kw):
        if out in done and _same(kw, {key: done[out][0]}):
            s = done[out][1]
            print(f"  {out}: I={s['I_mA']['left']:.3f} mA, converged={s['converged']}, {s['wall_s']:.0f}s", flush=True)
            return s
        return orig(out, **{**base, **kw})
    sw.run_one = run_one
    sw.SWEEPS[name] = (key, values)
    sw.main(name, root, values)
    print(f"@@SWEEP-DONE {out_root}", flush=True)


# ============================================================================= single-pulse chain
def _pulse_one(args):
    cfg, fdtd = args
    import run_pulse_chain as rpc
    from ltemsim.pulse_dynamics import PulseSim
    ratio_map, _ = rpc.load_ratio(fdtd)
    sim = PulseSim(cfg, verbose=True)
    N = sim.generation_profile(ratio_map)
    return N, sim.run(N)


def pulse_chain(workers=None, **kw):
    """scripts/run_pulse_chain.py main(**kw) with the beam positions computed in parallel."""
    import run_pulse_chain as rpc
    from ltemsim.pulse_dynamics import PulseConfig, PulseSim
    import inspect
    defaults = {k: v.default for k, v in inspect.signature(rpc.main).parameters.items()}
    a = {**defaults, **kw}
    X_LIST = tuple(a["x_list"]) if a["x_list"] else rpc.X_LIST
    cfgs = []
    for X in X_LIST:                                   # as in main(); checked again below before use
        xf = a["x_fine"] if a["x_fine"] else max(8.0, 2.0 * a["w_um"] + max(abs(x) for x in X_LIST) - 4.0)
        cfgs.append(PulseConfig(P_avg_mW=a["P_avg_mW"], x_beam_um=X, t_end_ps=a["t_end_ps"],
                                degenerate_diffusion=bool(a["degenerate"]), T=a["T"], V_bias=a["V_bias"],
                                w_um=a["w_um"], x_fine=xf))
    res = _pool_map(_pulse_one, [(c, a["fdtd"]) for c in cfgs], [f"X={X:+.1f}" for X in X_LIST],
                    workers or default_workers(len(cfgs)), sub=_ps_sub(a["t_end_ps"]))
    cache = list(zip(cfgs, res))

    class CachedPulseSim(PulseSim):
        def run(self, N, *args, **kwargs):
            for c, (Nc, rec) in cache:
                if _same(c, self.cfg) and _same(Nc, N):
                    return rec
            print(f"  (computing X={self.cfg.x_beam_um:+.1f} here: no matching parallel result)", flush=True)
            return super().run(N, *args, **kwargs)
    rpc.PulseSim = CachedPulseSim
    rpc.main(**kw)


# ============================================================================= tip-position chain
def _ps_sub(t_end_ps):
    """Per-run progress of a PulseSim: it prints t = … ps every 1 ps when verbose."""
    return {"type": "count", "pattern": "(?!)", "total": 1, "sub": r"t=([0-9.]+) ps", "sub_total": float(t_end_ps)}


_STEP_SUB = {"type": "count", "pattern": "(?!)", "total": 1, "sub": r"step (\d+)/(\d+)"}     # FDTD verbose prints


def _tip_one(args):
    kw, tip_shape = args
    import run_tip_positions as rtp
    from ltemsim.pulse_dynamics import PulseSim
    _inject(rtp, "PulseConfig", tip_shape=tip_shape)

    class LoudPulseSim(PulseSim):
        def __init__(self, cfg, verbose=False):
            super().__init__(cfg, verbose=True)
    rtp.PulseSim = LoudPulseSim
    rtp.simulate(**kw)
    return True


def _inject(module, cls_name, **fixed):
    """Make module.<cls_name>(...) build its config with extra fields (e.g. tip_shape) unless given."""
    fixed = {k: v for k, v in fixed.items() if v is not None}
    if not fixed:
        return
    orig = getattr(module, cls_name)
    if getattr(orig, "_ltem_injected", False):
        orig = orig._ltem_orig

    def make(*a, **k):
        return orig(*a, **{**fixed, **k})
    make._ltem_injected = True
    make._ltem_orig = orig
    setattr(module, cls_name, make)


def tip_simulate(workers=None, tip_shape=None, **kw):
    """scripts/run_tip_positions.py simulate(**kw) with the (X, h) runs in parallel (each writes its own npz)."""
    import run_tip_positions as rtp
    import inspect
    defaults = {k: v.default for k, v in inspect.signature(rtp.simulate).parameters.items() if k != "out"}
    a = {**defaults, **kw}
    pairs = [(X, h) for X in a["x_list"] for h in a["heights"]
             if not os.path.exists(os.path.join(kw["out"], f"X{X:+.1f}_h{int(h)}.npz"))]
    items = [({**kw, "x_list": (X,), "heights": (h,)}, tip_shape) for X, h in pairs]
    if items:
        _pool_map(_tip_one, items, [f"X={X:+.1f} h={int(h)} nm" for X, h in pairs], workers or default_workers(len(items)),
                  sub=_ps_sub(a["t_end_ps"]))
    _inject(rtp, "PulseConfig", tip_shape=tip_shape)
    rtp.simulate(**kw)                                 # writes the full meta.json; every run already exists


# ============================================================================= pump FDTD
def _fdtd_one(cfg):
    from ltemsim.optics_fdtd import run
    return run(cfg, verbose=True)


def fdtd_pump(workers=None, **kw):
    """scripts/run_fdtd_pump.py main(**kw) with its three FDTD runs (vacuum, no tip, tip) in parallel."""
    import run_fdtd_pump as rfp
    from ltemsim.optics_fdtd import FDTDConfig
    k = dict(kw)
    k.pop("out", None)
    dx = k.pop("dx_nm", 15.0) * 1e-9
    base = dict(dx=dx, x_half=4.0e-6, z_min=-3.0e-6, z_max=2.5e-6, z_source=2.2e-6, **k)      # as in main()
    cfgs = [FDTDConfig(**base, **extra) for extra in (dict(gaas=False, tip=False), dict(tip=False), dict(tip=True))]
    res = _pool_map(_fdtd_one, cfgs, ["vacuum", "no tip", "tip"], workers or 3, sub=_STEP_SUB)
    orig = rfp.run

    def run(cfg, verbose=True):
        for c, r in zip(cfgs, res):
            if _same(c, cfg):
                return r
        return orig(cfg, verbose=verbose)
    rfp.run = run
    rfp.main(**kw)


def _height_one(args):
    out, h, ref, tip_shape = args
    import run_fdtd_heights as rfh
    _inject(rfh, "FDTDConfig", tip_shape=tip_shape)
    orig_run = rfh.run
    rfh.run = lambda cfg, verbose=False: orig_run(cfg, verbose=True)     # step progress into this run's log
    rfh.main(out=out, heights=(h,), ref=ref)
    return True


def fdtd_heights(out="results/fdtd_pump/heights", heights=(10, 15, 25, 45, 80, 130, 200, 270, 330, 370),
                 ref="results/fdtd_pump/notip.npz", tip_shape=None, workers=None):
    """scripts/run_fdtd_heights.py with one height per core, then the summary rebuilt from all maps."""
    from .gui_tasks import fdtd_heights_summary
    os.makedirs(out, exist_ok=True)
    todo = [h for h in heights if not os.path.exists(os.path.join(out, f"ratio_h{int(h)}.npz"))]
    if todo:
        _pool_map(_height_one, [(out, h, ref, tip_shape) for h in todo], [f"h = {int(h)} nm" for h in todo],
                  workers or default_workers(len(todo)), sub=_STEP_SUB)
    fdtd_heights_summary(out=out, ref=ref)
