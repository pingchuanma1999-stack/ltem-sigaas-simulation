"""Tools & figures: catalogue of the remaining analysis, figure and maintenance scripts.

Every tool is one entry of TOOLS: its script, arguments (FieldSpecs and/or a picker), inputs, outputs and
prerequisites (checked live), how it is submitted to the JobManager and how its outputs are shown (PNG tabs,
tables, plots computed from the JSON, or the captured text).  Scripts that can only write to fixed paths
(results/slides, results/*.png) are flagged, and overwriting existing files asks for confirmation first.
"""
from __future__ import annotations

import ast
import html
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QTextOption
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QFrame, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QScrollArea, QSizePolicy,
                               QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QTabWidget,
                               QTextBrowser, QVBoxLayout, QWidget)

from .. import icons, nav, paths
from ..jobs import FAILED, FINISHED, QUEUED, JobManager
from ..paramspec import FieldSpec
from ..theme import ThemeManager, repolish
from ..widgets.common import Page, button, card, error_box, label, open_external, reveal
from ..widgets.imageview import ImageView
from ..widgets.logview import LogView
from ..widgets.mpl import PlotWidget
from ..widgets.paramform import ParamForm
from ..widgets.runpanel import RunPanel

R = paths.RESULTS
SLIDES = R / "slides"
PC = R / "pulse_chain"
TP = R / "tip_positions"
FBC = R / "field_bending_chain"
HEIGHTS = (10, 15, 25, 45, 80, 130, 200, 270, 330, 370)
X7 = (-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5)
X5 = (5.0, 2.5, 0.0, -2.5, -5.0)
GROUPS = ("Analysis", "Figures", "Maintenance")
PAGE_NAMES = {"tip": "Tip-position chain", "pulse": "Single-pulse chain", "steady": "Steady state",
              "fdtd": "Pump FDTD (optics)", "sweep": "Parameter sweep"}


# ============================================================================ helpers
def _esc(s) -> str:
    return html.escape(str(s))


def _code(s) -> str:
    return f"<span style='font-family:Menlo,Monaco,monospace'>{_esc(s)}</span>"


def _plain(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s))


def _rel(p) -> str:
    return paths.rel(p)


def mtime_str(p) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(Path(p).stat().st_mtime))
    except OSError:
        return ""


_GIT_FILES: set | None = None


def committed(p) -> bool:
    """True when the file is tracked by git (a committed result that should not be overwritten casually)."""
    global _GIT_FILES
    if _GIT_FILES is None:
        _GIT_FILES = set()
        if shutil.which("git") and (paths.ROOT / ".git").exists():
            try:
                r = subprocess.run(["git", "ls-files", "-z", "--", "results", "data"], cwd=paths.ROOT,
                                   capture_output=True, timeout=5)
                _GIT_FILES = {s for s in r.stdout.decode(errors="replace").split("\0") if s}
            except (OSError, subprocess.SubprocessError):
                pass
    return _rel(p) in _GIT_FILES


def gui_dirs(prefix: str) -> list[Path]:
    """results/gui_runs/<prefix>_<timestamp>[_n] folders, newest first (not <prefix>_chain_..., not backups)."""
    rx = re.compile(rf"^{re.escape(prefix)}_\d{{8}}-\d{{6}}(_\d+)?$")
    if not paths.GUI_RUNS.is_dir():
        return []
    return sorted((d for d in paths.GUI_RUNS.iterdir() if d.is_dir() and rx.match(d.name)), key=lambda d: d.name,
                  reverse=True)


def clean_log(log: str) -> str:
    return "\n".join(l for l in (log or "").splitlines() if not l.startswith("@@")).strip("\n")


def heights(folder: str) -> tuple:
    try:
        return tuple(json.loads((TP / folder / "meta.json").read_text())["heights_nm"])
    except (OSError, ValueError, KeyError):
        return HEIGHTS


def tp_npz(folder, Xs):
    return [TP / folder / f"X{X:+.1f}_h{int(h)}.npz" for X in Xs for h in heights(folder)]


def pc_npz(folder, Xs):
    return [PC / folder / f"X{X:+.1f}.npz" for X in Xs]


def x_files(d) -> list[Path]:
    try:
        return sorted(Path(d).glob("X*.npz"))
    except OSError:
        return []


def _xval(p: Path):
    try:
        return float(p.stem[1:])
    except ValueError:
        return None


# ---------------------------------------------------------------------------- prerequisites
PRODUCERS = {
    "tip": ("tip", "Run simulate on the Tip-position chain page with output results/tip_positions/P2mW_sym "
                   "(≈ 20–50 min; these npz files are gitignored, so the repository has none)"),
    "pulse": ("pulse", "Run the matching chain on the Single-pulse chain page (output into "
                       "results/pulse_chain/<run name>/)"),
    "steady": ("steady", "Run the production grid on the Steady state page with the output folder set to "
                         "results/baseline (≈ 6–7 min)"),
    "fdtd": ("fdtd", "Run the Pump FDTD (optics) page with output results/fdtd_pump"),
    "fbc": ("tool:field_bending_chain", "Run “Field-bending contribution” on this page, then copy "
                                        "its summary.json into results/field_bending_chain/"),
    "power": ("tool:power_dependence", "Run “Signal vs pump power” on this page (X = +5 and −5), then copy the "
                                       "JSON files into results/pulse_chain/power/"),
}


def req(lab: str, files, producer=None, optional=False, hard=False) -> dict:
    """A file prerequisite: ok when every file exists."""
    files = [Path(f) for f in files]
    have = [f.exists() for f in files]
    n = sum(have)
    ok = n == len(files)
    if len(files) == 1:
        detail = "" if ok else "missing"
    else:
        detail = f"{n}/{len(files)} files"
        if not ok:
            k = len(files) - n
            detail += f", missing {files[have.index(False)].name}" + (f" and {k - 1} more" if k > 1 else "")
    return dict(label=lab, ok=ok, detail=detail, producer=PRODUCERS.get(producer), optional=optional, hard=hard)


def cond(lab: str, ok: bool, detail="", producer=None, optional=False, hard=True, sel=False) -> dict:
    """A computed prerequisite; sel=True marks "the user still has to pick an input" rather than a missing file."""
    return dict(label=lab, ok=bool(ok), detail=detail, producer=PRODUCERS.get(producer), optional=optional, hard=hard,
                sel=sel)


def measured_req():
    return req(_code("data/measured_tip_scan.csv") + " (measured X scan, overlaid)", [paths.MEASURED_CSV])


def producer_name(target: str) -> str:
    """Display name of a PRODUCERS target: a page id or "tool:<id>"."""
    if target.startswith("tool:"):
        return BY_ID[target[5:]]["title"]
    return PAGE_NAMES.get(target, target)


def curated(folder) -> bool:
    """A hand-curated folder under results/ (not a results/gui_runs/ run folder)."""
    d = Path(folder).resolve()
    res, gui = paths.RESULTS.resolve(), Path(paths.GUI_RUNS).resolve()
    return (d == res or res in d.parents) and d != gui and gui not in d.parents


def new_dir_line(tid: str) -> str:
    return "New folder " + _code(f"results/gui_runs/tool_{tid}_<timestamp>/") + " containing:"


def sub(s: str) -> str:
    """A second-level bullet (rendered indented under the previous one)."""
    return "\x01" + s


# ---------------------------------------------------------------------------- result items
# ("image", title, path) | ("table", title, headers, rows, note) | ("plot", title, func(fig)) | ("text", title, text, fname)
def md_table(title: str, text: str):
    pre, rows, headers = [], [], None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            if headers is None:
                headers = cells
            else:
                rows.append(cells)
        elif s:
            pre.append(s)
    return ("table", title, headers or [], rows, " ".join(pre))


def _vals(d: dict) -> dict:
    """{'-7.5': v, ...} -> {-7.5: v, ...}"""
    out = {}
    for k, v in d.items():
        try:
            out[float(k)] = v
        except ValueError:
            pass
    return out


# ============================================================================ tool: field_bending
def fb_render(target, log):
    d = Path(target)
    items = [("image", "field_bending.png", d / "field_bending.png")]
    if (d / "summary.md").exists():
        items.append(md_table("Summary", (d / "summary.md").read_text(encoding="utf-8")))
    if log:
        items.append(("text", "Log", clean_log(log), "field_bending_log.txt"))
    return items


def dir_history(committed_dir: Path | None, prefix: str, marker: str):
    out = [(f"{_rel(d)}  ·  {mtime_str(d / marker)}", d) for d in gui_dirs(prefix) if (d / marker).exists()]
    if committed_dir is not None and (committed_dir / marker).exists():
        out.append((f"{_rel(committed_dir)} (committed)  ·  {mtime_str(committed_dir / marker)}", committed_dir))
    return out


def out_target(spec):
    return paths.absolute(spec["out"]) if spec.get("out") else None


# ============================================================================ tool: field_bending_chain
def fbc_render(target, log):
    d = Path(target)
    s = json.loads((d / "summary.json").read_text())
    X = np.array(s.get("X", X7), float)
    bent, hor = _vals(s["bent"]), _vals(s["horizontal"])

    def arr(var, k):
        return np.array([var[x][k] for x in X], float)

    def draw(fig):
        ax1, ax2 = fig.subplots(1, 2)
        b, h = arr(bent, "a3"), arr(hor, "a3")
        m = max(b.max(), h.max())
        ax1.plot(X, b / m, "o-", lw=1.8, label="bent: E_x + E_z")
        ax1.plot(X, h / m, "s--", lw=1.5, mfc="none", label="horizontal: E_z removed")
        ax1.axvline(0, color="0.5", lw=0.6)
        ax1.set_xlabel("X = beam − tip (µm)")
        ax1.set_ylabel("a3, 0.3–2 THz band (norm.)")
        ax1.set_title("n = 3: full field vs E_z removed")
        ax1.legend()
        for i, (k, mk) in enumerate((("a1", "o"), ("a2", "s"), ("a3", "^"), ("a4", "D"), ("far", "x"))):
            r = arr(bent, k) / arr(hor, k)
            ax2.plot(X, 100 * (r - 1), marker=mk, lw=1.5, ms=5, color=f"C{i}", label=k if k == "far" else f"n = {k[1]}")
        ax2.axhline(0, color="0.5", lw=0.6)
        ax2.axvline(0, color="0.5", lw=0.6)
        ax2.set_xlabel("X = beam − tip (µm)")
        ax2.set_ylabel("bent / horizontal − 1 (%)")
        ax2.set_title("field-line bending contribution")
        ax2.legend(ncol=3, loc="best")

    rows = []
    for x in X:
        b, h = bent[x], hor[x]
        rows.append([f"{x:+.1f}", f"{b['a3']:.3e}", f"{h['a3']:.3e}"] +
                    [f"{b[k] / h[k]:.4f}" for k in ("a3", "a1", "a2", "a4", "far")])
    items = [("plot", "a3: bent vs horizontal", draw),
             ("table", "Values", ["X (µm)", "a3 bent", "a3 horizontal", "a3 ratio", "a1 ratio", "a2 ratio", "a4 ratio",
                                  "far ratio"], rows,
              "Band-limited amplitudes (0.3–2 THz, d²S/dt²); ratio = bent / horizontal.")]
    tf = d / "tilt.json"
    if tf.exists():
        t = _vals(json.loads(tf.read_text()))
        zs = list(next(iter(t.values())).keys()) if t else []
        trows = [[f"{x:g}"] + [f"{t[x][z]['tilt_deg']:.3g}" for z in zs] for x in sorted(t)]
        items.append(("table", "Applied-field tilt", ["x (µm)"] + [f"{z} tilt (°)" for z in zs], trows,
                      "Tilt arctan(E_z / E_x) of the applied field at several depths (from the Poisson solution of "
                      "the X = 0 bent run)."))
    if log:
        items.append(("text", "Log", clean_log(log), "field_bending_chain_log.txt"))
    return items


# ============================================================================ tool: power_dependence
def power_jsons(target) -> list[Path]:
    p = Path(target)
    return [p] if p.is_file() else sorted(p.glob("power_dependence_*.json"))


# current JSON (band-limited 0.3–2 THz amplitudes of d²S/dt²) and the older one (RMS of dS/dt, dJ_x/dt)
POWER_METRICS = ((("det2_band", "detected, n = 2"), ("det3_band", "detected, n = 3"), ("far_band", "far field")),
                 (("dS2dt_rms", "RMS dS₂/dt, n = 2"), ("dJxdt_rms", "far field, RMS dJ_x/dt")))


def power_render(target, log):
    data = []
    for f in power_jsons(target):
        rows = json.loads(f.read_text())
        name = f.stem.replace("power_dependence_", "").replace("_", " ")
        data.append((name, rows))
    if not data:
        return []
    old = not all(k in data[0][1][0] for k, _ in POWER_METRICS[0])
    metrics = POWER_METRICS[old]
    guide = metrics[-2][0]                          # highest detected harmonic
    # one reference per metric across all files, so X = +5 vs −5 keeps its relative amplitude (one-sidedness)
    ref = {k: max(r[k] for _, rows in data for r in rows) or 1.0 for k, _ in metrics}

    def draw(fig):
        ax1, ax2 = fig.subplots(1, 2)
        from matplotlib.lines import Line2D
        styles = ("-", "--", ":", "-.")
        lo = 1.0
        for i, (name, rows) in enumerate(data):
            P = np.array([r["P_mW"] for r in rows], float)
            for j, (k, _) in enumerate(metrics):
                y = np.array([r[k] for r in rows], float) / ref[k]
                lo = min(lo, y.min())
                ax1.loglog(P, y, ls=styles[i % 4], marker="o", ms=4, lw=1.6, color=f"C{j}")
            N = np.array([r["N_surface"] for r in rows], float)
            ax2.loglog(P, N, ls=styles[i % 4], marker="o", ms=4, lw=1.6, color=f"C{i}", label=name)
        rows0 = data[0][1]
        P = np.array([r["P_mW"] for r in rows0], float)
        y0 = rows0[0][guide] / ref[guide]
        ax1.loglog(P, y0 * P / P[0], color="0.6", lw=1, ls=":")
        ax1.set_ylim(lo / 2, 1.6)
        h1 = [Line2D([], [], color=f"C{j}", marker="o", ms=4, lw=1.6) for j in range(len(metrics))] + \
             [Line2D([], [], color="0.6", ls=":", lw=1)]
        leg = ax1.legend(h1, [m[1] for m in metrics] + ["∝ P (guide)"], loc="lower right", fontsize=7.5)
        if len(data) > 1:
            ax1.add_artist(leg)
            ax1.legend([Line2D([], [], color="0.4", ls=styles[i % 4], lw=1.6) for i in range(len(data))],
                       [n for n, _ in data], loc="lower left", fontsize=7.5)
        ax1.set_xlabel("P_avg (mW)")
        ax1.set_ylabel(("RMS, older metric" if old else "0.3–2 THz band amp.") + " (norm.)")
        ax1.set_title("detected signal vs pump power")
        ax2.set_xlabel("P_avg (mW)")
        ax2.set_ylabel("N_surface (cm⁻³)")
        ax2.set_title("peak surface pair density")
        ax2.legend(fontsize=7.5)

    rows = []
    for name, rr in data:
        for r in rr:
            rows.append([name, f"{r['P_mW']:g}", f"{r['N_surface']:.3e}", f"{r['pairs_per_cm']:.3e}"] +
                        [f"{r[k]:.3e}" for k, _ in metrics])
    note = "Each metric is normalized to its maximum over all files (keeps the relative amplitude of X = +5 vs −5)."
    if old:
        note += (" Older JSON: the metrics are RMS values of dS₂/dt and dJ_x/dt (not 0.3–2 THz band-limited "
                 "amplitudes) and cannot be compared directly with newer values.")
    items = [("plot", "Power dependence", draw),
             ("table", "Values", ["File", "P (mW)", "N_surface (cm⁻³)", "pairs / cm"] + [k for k, _ in metrics],
              rows, note)]
    if log:
        items.append(("text", "Log", clean_log(log), "power_dependence_log.txt"))
    return items


def power_history():
    out = []
    for d in gui_dirs("tool_power_dependence"):
        js = sorted(d.glob("power_dependence_*.json"))
        if js:
            out.append((f"{_rel(d)}  ·  {mtime_str(js[0])}", d))
    if list((PC / "power").glob("power_dependence_*.json")):
        out.append(("results/pulse_chain/power (committed, X = ±5)", PC / "power"))
    for f in sorted(PC.glob("power_dependence_*.json")):
        out.append((f"{_rel(f)} (committed, older format)", f))
    return out


# ============================================================================ tool: redemodulate
def redemod_name(A, h, lf) -> str:
    return f"redemod_A{float(A):g}_h{float(h):g}{'_ltemfactor' if lf else ''}.json"


def redemod_target(spec):
    k = spec.get("kwargs", {})
    if not k.get("run_dir"):
        return None
    return paths.absolute(k["run_dir"]) / redemod_name(k.get("A_nm", 180.0), k.get("h_min_nm", 0.0),
                                                       k.get("ltem_factor", False))


def redemod_render(target, log):
    f = Path(target)
    d = json.loads(f.read_text())
    rows = _vals(d)
    Xs = sorted(rows)
    if not Xs:
        return [("text", "Result", "Empty JSON: the run folder has no X*.npz files.", "redemod.txt")]
    ns = sorted({int(m.group(1)) for k in rows[Xs[0]] for m in [re.match(r"S(\d)_rms", k)] if m})
    keys = [f"dS{n}dt_rms" for n in ns] + [f"S{n}_rms" for n in ns]
    X = np.array(Xs)

    def draw(fig):
        ax1, ax2 = fig.subplots(1, 2, sharey=True)
        for i, n in enumerate(ns):
            for ax, k in ((ax1, f"dS{n}dt_rms"), (ax2, f"S{n}_rms")):
                v = np.array([rows[x][k] for x in Xs], float)
                ax.plot(X, v / v.max(), marker="os^D"[i % 4], lw=1.7, ms=5, color=f"C{i}", label=f"n = {n}")
        for ax, t in ((ax1, "RMS of low-passed dS_n/dt"), (ax2, "RMS of low-passed S_n")):
            ax.axvline(0, color="0.5", lw=0.6)
            ax.set_xlabel("X = beam − tip (µm)")
            ax.set_title(t)
            ax.legend()
        ax1.set_ylabel("norm. per harmonic")

    trows = []
    for k in keys:
        v = np.array([rows[x][k] for x in Xs], float)
        m = v.max() if v.max() > 0 else 1.0
        trows.append([k] + [(f"{x / m:.2f}", f"raw value {x:.4g}") for x in v])
    items = [("plot", "Re-demodulated X scan", draw),
             ("table", "Normalized table", ["Metric"] + [f"X = {x:+.1f}" for x in Xs], trows,
              f"{_rel(f)} — each row normalized to its own maximum (as in the script's printed table); "
              "hover for raw values.")]
    if log:
        items.append(("text", "Log", clean_log(log), "redemodulate_log.txt"))
    return items


def pulse_run_dirs() -> list[Path]:
    out = []
    if PC.is_dir():
        out += [d for d in sorted(PC.iterdir()) if d.is_dir() and x_files(d)]
    if paths.GUI_RUNS.is_dir():
        out += [d for d in sorted(paths.GUI_RUNS.glob("pulse_*"), reverse=True) if d.is_dir() and x_files(d)]
    return out


def redemod_prefer(ctx, entries):
    """Index of the entry that belongs to the selected run folder (exact A/h match first)."""
    d = ctx.get("rundir")
    if d is None:
        return None
    w = _rundir_writes(ctx)
    best = None
    for i, e in enumerate(entries):
        t = e.get("target")
        if t is None:
            continue
        t = Path(t).resolve()
        if w is not None and t == Path(w).resolve():
            return i
        if best is None and t.parent == Path(d).resolve():
            best = i
    return best


def redemod_history():
    files = []
    for d in pulse_run_dirs():
        files += list(d.glob("redemod_*.json"))
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return [(f"{_rel(f)}{' (committed)' if committed(f) else ''}  ·  {mtime_str(f)}", f) for f in files]


# ============================================================================ tool: compare_scenarios
def solution_files() -> list[Path]:
    out = sorted((R / "coarse_scenarios" / "solutions").glob("*.npz"))
    try:
        out += sorted(R.rglob("solution.npz"))
    except OSError:
        pass
    return out


def solution_label(p: Path) -> str:
    p = Path(p)
    if p.name == "solution.npz":
        return p.parent.name
    return p.stem.removeprefix("coarse_")


def compare_target(spec):
    o = spec.get("kwargs", {}).get("out")
    return paths.absolute(o) if o else None


def compare_render(target, log):
    png = Path(target)
    items = [("image", png.name, png)]
    md = png.with_suffix(".md")
    if md.exists():
        items.append(md_table("Values", md.read_text(encoding="utf-8")))
    if log:
        items.append(("text", "Log", clean_log(log), "compare_log.txt"))
    return items


def compare_history():
    out = []
    for d in gui_dirs("tool_compare_scenarios"):
        if (d / "compare.png").exists():
            out.append((f"{_rel(d)}  ·  {mtime_str(d / 'compare.png')}", d / "compare.png"))
    for png in sorted((R / "coarse_scenarios").glob("*.png")):
        if png.with_suffix(".md").exists():
            out.append((f"{_rel(png)} (committed)", png))
    return out


# ============================================================================ text-only tools
def text_render(fname):
    def render(target, log):
        txt = clean_log(log)
        return [("text", "Text output", txt or "(no output)", fname)]
    return render


def parse_waveform(text: str):
    rx = re.compile(r"^(.*?)\s+(-?\d+)\s+(-?\d+\.\d+)\s+(-?\d+)\s+(\{.*?\})\s+/\s+(\{.*?\})\s*$")

    def shift(s):
        try:
            d = ast.literal_eval(s)
            return f"residual {d['residual']:.2f} · shift {d['shift_ps']:+.3f} ps"
        except (ValueError, SyntaxError, KeyError, TypeError):
            return s
    rows = []
    for line in text.splitlines():
        m = rx.match(line)
        if m and not line.startswith("pair"):
            rows.append([m.group(1).strip(), m.group(2), m.group(3), m.group(4), shift(m.group(5)), shift(m.group(6))])
    return rows


def waveform_render(target, log):
    if log:
        txt = clean_log(log)
    elif target and Path(target).exists():
        txt = Path(target).read_text(encoding="utf-8", errors="replace")
    else:
        txt = ""
    items = []
    rows = parse_waveform(txt)
    if rows:
        items.append(("table", "Phase fits", ["Pair (ref vs test)", "φ0 (°)", "τ (ps)", "Residual (°)",
                                             "Best shift, same sign", "Best shift, opposite sign"], rows,
                      "φ0: 0° same shape, 180° inverted, ±90° derivative relation; τ is the delay; the residual is the "
                      "rms of the phase fit over 0.25–1.6 THz."))
    items.append(("text", "Text output", txt or "(no output)", "waveform_phase_table.txt"))
    return items


def parse_dember(text: str):
    blocks = []
    for line in text.splitlines():
        m = re.match(r"\s*T_carrier = (\d+) K \(D_n = (\d+)", line)
        if m:
            blocks.append(dict(T=float(m.group(1)), Dn=float(m.group(2)), V={}))
            continue
        m = re.match(r"\s+V =\s*([-\d.]+) V:(.*)", line)
        if m and blocks:
            pairs = re.findall(r"X=([+-]?\d+\.\d+): ([\d.]+)", m.group(2))
            blocks[-1]["V"][float(m.group(1))] = [(float(a), float(b)) for a, b in pairs]
    return blocks


def dember_render(target, log):
    txt = clean_log(log)
    blocks = parse_dember(txt)
    items = []
    if blocks:
        def draw(fig):
            ax = fig.subplots()
            for i, b in enumerate(blocks):
                for V, pts in sorted(b["V"].items(), reverse=True):
                    x, y = zip(*pts)
                    ax.plot(x, y, marker="o" if V else "s", ls="-" if V else "--", lw=1.8, ms=5, color=f"C{i}",
                            mfc=None if V else "none", label=f"T = {b['T']:.0f} K, V = {V:g} V")
            ax.axvline(0, color="0.5", lw=0.6)
            ax.set_xlabel("X = beam − tip (µm)")
            ax.set_ylabel("RMS of low-passed dS3/dt (norm. per T)")
            ax.set_title("photo-Dember (even) vs drift (odd) interference")
            ax.legend()
        rows = []
        for b in blocks:
            for V, pts in sorted(b["V"].items(), reverse=True):
                rows.append([f"{b['T']:.0f}", f"{b['Dn']:.0f}", f"{V:g}"] + [f"{v:.2f}" for _, v in pts])
        Xs = [x for x, _ in next(iter(blocks[0]["V"].values()))] if blocks[0]["V"] else []
        items.append(("plot", "X scan", draw))
        items.append(("table", "Values", ["T (K)", "D_n (cm²/s)", "V (V)"] + [f"X = {x:+.1f}" for x in Xs], rows,
                      "Normalized within each temperature to the maximum over both biases (older first-derivative "
                      "RMS metric)."))
    items.append(("text", "Text output", txt or "(no output)", "dember_interference.txt"))
    return items


NUMERIC_TESTS = (("poisson", "Bare Poisson field vs analytic solution (error < 12 %, E_z/E_x < 5 %)"),
                 ("equilibrium", "Zero residual at equilibrium (V = 0, G = 0)"),
                 ("jacobian", "Finite-difference check of the coupled Jacobian (relative error < 1e-5)"),
                 ("dark", "10 V dark state: converges, |I| < 1e-3 mA, plausible EL2⁺"))


def numerics_render(target, log):
    txt = clean_log(log)
    lines = {l.strip() for l in txt.splitlines()}
    failed = "Traceback" in txt or "@@FAILED" in (log or "")
    rows, first_bad = [], True
    for name, desc in NUMERIC_TESTS:
        if f"{name} ok" in lines:
            rows.append([name, desc, "✓ Passed"])
        elif failed and first_bad:
            rows.append([name, desc, "✗ Failed"])
            first_bad = False
        else:
            rows.append([name, desc, "— Not run"])
    n_ok = sum(r[2].startswith("✓") for r in rows)
    return [("table", "Test results", ["Test", "Checks", "Result"], rows, f"{n_ok}/{len(rows)} passed"),
            ("text", "Text output", txt or "(no output)", "test_numerics.txt")]


# ============================================================================ fixed-output figure tools
def png_render(target, log):
    files = target if isinstance(target, (list, tuple)) else [target]
    items = [("image", Path(f).name, Path(f)) for f in files if Path(f).suffix == ".png"]
    if log and clean_log(log):
        items.append(("text", "Log", clean_log(log), "log.txt"))
    return items


def fixed_history(files, note="current version"):
    out = []
    for f in files:
        if Path(f).exists():
            tag = ", committed" if committed(f) else ""
            out.append((f"{_rel(f)} ({note}{tag})  ·  {mtime_str(f)}", Path(f)))
    return out


def _slide_reqs_positions():
    return [req(_code("results/tip_positions/P2mW_sym/meta.json"), [TP / "P2mW_sym" / "meta.json"], "tip"),
            req(_code("results/tip_positions/P2mW_sym/X±5/±2.5/0_h*.npz"), tp_npz("P2mW_sym", X5), "tip"),
            req(_code("results/tip_positions/{P2mW_sym, P2mW_sym_Vtip12}/demod_A180_h10.json"),
                [TP / f / "demod_A180_h10.json" for f in ("P2mW_sym", "P2mW_sym_Vtip12")], "tip")]


DEMBER_FOLDERS = ("P2mW_notipmap", "P2mW_notipmap_degen", "P2mW_notipmap_degen_T600", "P2mW_notipmap_degen_T1500",
                  "P2mW_notipmap_degen_0V", "P30mW_notipmap_degen", "P30mW_notipmap_degen_0V",
                  "P30mW_notipmap_degen_T600", "P30mW_notipmap_degen_T600_0V")


def _dember_req():
    have = sum(all(f.exists() for f in pc_npz(d, X7)) for d in DEMBER_FOLDERS)
    return [cond("9 diffusion-model comparison runs in " + _code("results/pulse_chain/")
                 + " (missing ones are skipped)", have == len(DEMBER_FOLDERS), f"{have}/{len(DEMBER_FOLDERS)} folders complete", "pulse", optional=True,
                 hard=False)]


SLIDE_KEYS = [
    dict(key="steady", name="Steady-state overview", files=("sim_steady_state.png",), runtime="≈ 1.5 s",
         reqs=lambda: [req(_code(f"results/baseline/{f}"), [R / "baseline" / f], "steady")
                       for f in ("profiles_depth0.5um.csv", "tip_scan.csv")]),
    dict(key="positions", name="Tip-position chain (route 2)", files=("sim_positions.png",), runtime="< 1 s",
         reqs=_slide_reqs_positions),
    dict(key="bending", name="Field-line bending contribution", files=("sim_field_bending.png",),
         runtime="≈ 1.5 s (first run up to ≈ 15 s)",
         reqs=lambda: [req(_code("results/field_bending_chain/summary.json"), [FBC / "summary.json"], "fbc")]),
    dict(key="pump", name="Pump FDTD", files=("sim_pump_fdtd.png",), runtime="≈ 2 s",
         reqs=lambda: [req(_code("results/fdtd_pump/{tip, notip, ratio}.npz"),
                           [R / "fdtd_pump" / f"{n}.npz" for n in ("tip", "notip", "ratio")], "fdtd")]),
    dict(key="dynamics", name="Carrier dynamics", files=("sim_dynamics.png",),
         runtime="≈ 3–4 s (includes a 0.6 ps PulseSim)",
         reqs=lambda: [req(_code("results/pulse_chain/P2mW_notipmap_degen/X+0.0.npz"),
                           pc_npz("P2mW_notipmap_degen", (0.0,)), "pulse")]),
    dict(key="nearfield", name="Near field and demodulation", files=("sim_nearfield_demod.png",), runtime="≈ 1 s",
         reqs=lambda: [req(_code("results/pulse_chain/P2mW_notipmap_degen/X{±5, ±2.5, 0}.npz"),
                           pc_npz("P2mW_notipmap_degen", X5), "pulse")]),
    dict(key="xscan", name="X-scan comparison", files=("sim_xscan.png", "xscan_A180.json"), runtime="≈ 1 s",
         reqs=lambda: [req(_code("results/pulse_chain/P2mW_notipmap_degen/X*.npz (7 X values)"),
                           pc_npz("P2mW_notipmap_degen", X7), "pulse"),
                       req(_code("results/pulse_chain/P30mW_notipmap_degen/X*.npz"), pc_npz("P30mW_notipmap_degen", X7),
                           "pulse", optional=True),
                       req(_code("results/tip_positions/*/demod_A180_h10.json (4 route-2 runs)"),
                           [TP / f / "demod_A180_h10.json" for f in ("P2mW_sym", "P2mW_tipmap", "P30mW_sym",
                                                                      "P2mW_sym_Vtip12")], "tip", optional=True)]),
    dict(key="dember", name="Diffusion model / Dember", files=("sim_dember.png",), runtime="≈ 2 s", reqs=_dember_req),
    dict(key="power", name="Power dependence", files=("sim_power.png",), runtime="≈ 1 s",
         reqs=lambda: [req(_code("results/pulse_chain/power/power_dependence_X±5.0_nsat2e+19.json"),
                           [PC / "power" / f"power_dependence_X{x}_nsat2e+19.json" for x in ("+5.0", "-5.0")],
                           "power")]),
    dict(key="dipole", name="Dipole geometry (English)", files=("sim_dipole_geometry.png",), runtime="≈ 2 s",
         reqs=lambda: []),
]
CARTOON_KEYS = [
    dict(key="distortions", name="Three field distortions", files=("cartoon_distortions.png",), runtime="< 1 s",
         reqs=lambda: []),
    dict(key="chain", name="Three-stage chain", files=("cartoon_chain.png",), runtime="< 1 s", reqs=lambda: []),
    dict(key="oddeven", name="Odd/even split of the X scan", files=("cartoon_oddeven.png",), runtime="< 1 s",
         reqs=lambda: []),
    dict(key="tippot", name="Tip DC potential", files=("cartoon_tippot.png",), runtime="< 1 s", reqs=lambda: []),
    dict(key="power", name="Spot broadening with power", files=("cartoon_power.png",), runtime="< 1 s",
         reqs=lambda: []),
    dict(key="all", name="All 5 cartoons", files=("cartoon_distortions.png", "cartoon_chain.png", "cartoon_oddeven.png",
                                                 "cartoon_tippot.png", "cartoon_power.png"), runtime="≈ 1.5 s",
         reqs=lambda: []),
]
KEYSETS = {"slide_keys": SLIDE_KEYS, "cartoon_keys": CARTOON_KEYS}


def keyinfo(keys, key):
    return next((k for k in keys if k["key"] == key), keys[0])


def key_of_spec(spec):
    a = spec.get("argv") or []
    return str(a[0]) if a else "all"


def keyed_target(keys):
    def target(spec):
        k = keyinfo(keys, key_of_spec(spec))
        return [SLIDES / f for f in k["files"]]
    return target


def keyed_history(keys):
    def hist():
        out = []
        for k in keys:
            files = [SLIDES / f for f in k["files"]]
            if k["key"] == "all":
                if all(f.exists() for f in files):
                    out.append((f"all · all {len(files)} figures (current version)", files, "all"))
            elif files[0].exists():
                tag = ", committed" if committed(files[0]) else ""
                out.append((f"{k['key']} · {files[0].name} (current version{tag})  ·  {mtime_str(files[0])}", files,
                            k["key"]))
        return out
    return hist


# ============================================================================ registry
def _v(ctx, k, default=None):
    return ctx.get("v", {}).get(k, default)


def _rundir_inputs(ctx):
    d = ctx.get("rundir")
    if not d:
        return ["(pick a single-pulse chain run folder on the right)"]
    return [_code(f"{_rel(d)}/X*.npz") + " — run_pulse_chain output with t, heights_nm, Ez_tip and n_center"]


def _rundir_prereq(ctx):
    d = ctx.get("rundir")
    if not d:
        return [cond("Pick a run folder", False, "none selected", sel=True)]
    xs = x_files(d)
    out = [cond(_code(f"{_rel(d)}/X*.npz"), bool(xs), f"{len(xs)} X values" if xs else "no X*.npz in the folder",
                "pulse")]
    if xs:
        try:
            with np.load(xs[0]) as z:
                miss = [k for k in ("t", "heights_nm", "Ez_tip", "n_center") if k not in z.files]
        except Exception as e:      # unreadable / truncated npz
            miss = [f"cannot read {xs[0].name}: {type(e).__name__}"]
        out.append(cond("npz contains t / heights_nm / Ez_tip / n_center", not miss,
                        "missing " + ", ".join(miss) if miss else ""))
    return out


def _rundir_writes(ctx):
    d = ctx.get("rundir")
    if not d:
        return None
    return Path(d) / redemod_name(_v(ctx, "A_nm", 180.0), _v(ctx, "h_min_nm", 0.0), _v(ctx, "ltem_factor", False))


def _power_runtime(ctx):
    n = len(_v(ctx, "powers", ()) or ())
    if not n:
        return "—"
    return f"≈ {max(1, round(n * 22 / 60))}–{max(1, round(n * 55 / 60))} min ({n} powers × 22–55 s)"


def _power_outputs(ctx):
    X, ns = float(_v(ctx, "X", 5.0)), float(_v(ctx, "n_sat", 2e19))
    return [new_dir_line("power_dependence"),
            sub(_code(f"power_dependence_X{X:+.1f}_nsat{ns:.0e}.json") +
                " — one row per power: P_mW, N_surface, pairs_per_cm, det2_band, det3_band, far_band"),
            sub(_code(f"power_X{X:+.1f}_nsat{ns:.0e}_P*.npz") + " — waveforms for each power"),
            "The slide figure sim_power.png reads the fixed X = ±5 JSON files in results/pulse_chain/power/ and is "
            "not affected by this run."]


def _power_prereq(ctx):
    P = _v(ctx, "powers", ()) or ()
    return [cond("Pump-power list is non-empty and all values > 0", len(P) > 0 and all(p > 0 for p in P),
                 f"{len(P)} powers")]


def _compare_inputs(ctx):
    sols = ctx.get("solutions") or []
    out = [f"{len(sols)} steady-state solutions selected (tick them in the table on the right)"]
    out += [_code(_rel(p)) + f" → “{_esc(l)}”" for l, p in sols[:6]]
    if len(sols) > 6:
        out.append(f"… and {len(sols) - 6} more")
    return out + [_code("data/measured_tip_scan.csv") + " — measured +25 V points"]


def _compare_prereq(ctx):
    sols = ctx.get("solutions") or []
    gone = [p for _, p in sols if not Path(p).exists()]
    out = [cond("Tick at least 1 steady-state solution", bool(sols), f"{len(sols)} selected", sel=True)]
    if gone:
        out.append(cond("All selected solution files exist", False,
                        f"no longer exist: {', '.join(Path(p).name for p in gone[:3])}"))
    return out + [measured_req()]


def _tool(**t):
    """Registry entry with defaults: no inputs / outputs / prerequisites / history; argv tools run without arguments.
    Multi-figure scripts (extra = a KEYSETS name) render one figure per run: everything follows from the key set."""
    keys = KEYSETS.get(t.get("extra"))
    if keys is not None:
        info = lambda c: keyinfo(keys, c.get("key"))                                     # noqa: E731
        t.setdefault("runtime", lambda c: info(c)["runtime"])
        t.setdefault("fixed", lambda c: [SLIDES / f for f in info(c)["files"]])
        t.setdefault("prereq", lambda c: info(c)["reqs"]())
        t.setdefault("progress", lambda c: {"type": "count", "pattern": r"^saved ", "total": len(info(c)["files"])})
        t.setdefault("build", lambda c, out: (dict(argv=[c["key"]]), c["key"]))
        t.setdefault("target", keyed_target(keys))
        t.setdefault("history", keyed_history(keys))
        t.setdefault("render", png_render)
    t.setdefault("prereq", lambda c: [])
    t.setdefault("inputs", lambda c: [])
    t.setdefault("outputs", lambda c: [])
    t.setdefault("target", lambda spec: None)
    t.setdefault("history", lambda: [])
    if t["func"] is None:
        t.setdefault("build", lambda c, out: (dict(argv=[]), ""))
    return t


TOOLS = [
    _tool(id="field_bending", group="Analysis", title="Applied-field bending",
         short="Bare electrostatics: field-line tilt, 5 cases", tag="≈1 s",
         desc="Electrostatics of the applied bias only (no carriers, no EL2): one linear Poisson solve for each of "
              "5 wafer-thickness / back-side cases (30 µm floating reference; 350 µm and 500 µm with a floating or "
              "grounded back side), giving the field-line tilt vs depth, the vertical/horizontal polarization ratio "
              "∫P_z/∫P_x and the tip E_z scanned over X. The cases and the wide grid (CASES / BOX) are hard-coded in "
              "the script; there are no adjustable parameters. Note: “tip peak X” in the table is only a tie-break "
              "between two antisymmetric peaks at ±1.5 µm and has no physical meaning.",
         script="scripts/field_bending.py", func="main", out=True,
         runtime="≈ 1–2 s (5 grids, one linear Poisson solve each + 3 tip-scan solves)",
         inputs=lambda c: [_code("data/measured_tip_scan.csv") + " — measured +25 V tip scan (overlaid)"],
         outputs=lambda c: [new_dir_line("field_bending"),
                            sub(_code("field_bending.png") + " — 2×2: tilt vs depth, surface E_z, signed and "
                                                             "absolute tip E_z"),
                            sub(_code("summary.json") + " / " + _code("summary.md")
                                + " — summary table of the 5 cases")],
         prereq=lambda c: [measured_req()],
         build=lambda c, out: (dict(kwargs=dict(out=_rel(out))), ""),
         target=out_target, render=fb_render,
         history=lambda: dir_history(R / "field_bending", "tool_field_bending", "summary.json")),

    _tool(id="field_bending_chain", group="Analysis", title="Field-bending contribution",
         short="Pulse chain: full applied field vs E_z removed", tag="5–10 min",
         desc="Route 1 (near-field E_z(h,t) without the tip + Pizzuto finite-dipole demodulation, degenerate "
              "diffusion, t_end = 4 ps): two runs at each of 7 spot positions X = −7.5…+7.5 µm — the full applied "
              "field (E_x + E_z, bent) and the field with its vertical component removed (horizontal) — comparing "
              "the 0.3–2 THz band-limited amplitudes of harmonics 1–4 and of the far field. The X = 0 bent run also "
              "writes the applied-field tilt tilt.json. The X list, bias (25 V), temperature and demodulation "
              "settings are fixed in the script.",
         script="scripts/field_bending_chain.py", func="main", out=True,
         runtime="≈ 5–10 min (14 PulseSim runs, ≈ 17–40 s each)",
         specs=lambda: [FieldSpec("P_avg_mW", 2.0, "float", "Pump power P_avg", "mW",
                                  "Average pump power. Everything else (X list, V_bias, T, demodulation) is fixed in "
                                  "the script.", "Parameters", minimum=1e-4)],
         inputs=lambda c: ["None (pure simulation; needs no other results)"],
         outputs=lambda c: [new_dir_line("field_bending_chain"),
                            sub(_code("summary.json") + " — a1…a4 and far for bent / horizontal, plus the a3 ratio"),
                            sub(_code("tilt.json") + ", " + _code("applied_field.npz") + " — applied-field tilt and "
                                                                                       "field map"),
                            sub(_code("bent_X*.npz") + ", " + _code("horizontal_X*.npz") + " — waveforms of each run"),
                            "The slide figure sim_field_bending.png only reads the fixed "
                            "results/field_bending_chain/summary.json and does not use this new result."],
         progress=lambda c: {"type": "count", "pattern": r"^(bent|horizontal)\s+X=", "total": 14},
         build=lambda c, out: (dict(kwargs=dict(P_avg_mW=_v(c, "P_avg_mW", 2.0), out=_rel(out))),
                               f"P = {_v(c, 'P_avg_mW', 2.0):g} mW"),
         target=out_target, render=fbc_render,
         history=lambda: dir_history(FBC, "tool_field_bending_chain", "summary.json")),

    _tool(id="power_dependence", group="Analysis", title="Signal vs pump power",
         short="Pump-power sweep at fixed X (band-limited)", tag="4–9 min",
         desc="Runs the single-pulse chain at a fixed spot offset X for a list of pump powers (route 1: near field "
              "without the tip + FDM, symmetric Gaussian spot, no FDTD tip map, t_end = 4 ps) and records the "
              "0.3–2 THz band-limited spectral amplitude of the detected waveform d²S/dt² (harmonics 2 and 3), the "
              "far-field reference from J_x, the surface density N_surface and the number of pairs per cm. "
              "V_bias = 25 V, T = 300 K, w = 3 µm and A = 180 nm are fixed. The JSON is written only after all "
              "powers have finished; stopping early leaves only the per-power npz files.",
         script="scripts/power_dependence.py", func="main", out=True, runtime=_power_runtime,
         specs=lambda: [FieldSpec("X", 5.0, "float", "Spot position X", "µm",
                                  "Spot position relative to the tip; X > 0: the tip is on the anode side of the spot",
                                  "Sweep"),
                        FieldSpec("n_sat", 2e19, "float", "Saturation n_sat", "cm⁻³",
                                  "Saturation density: soft upper limit of the single-pulse carrier density", "Sweep",
                                  minimum=1e12),
                        FieldSpec("powers", (0.25, 0.5, 1, 2, 4, 8, 16, 30, 60, 120), "tuple", "Pump powers", "mW",
                                  "Comma-separated; one PulseSim per power (22–55 s)", "Sweep"),
                        FieldSpec("degenerate", True, "bool", "Degenerate diffusion", "",
                                  "Fermi–Dirac statistics via the generalized Einstein relation; differs from the "
                                  "run_pulse_chain default (off)",
                                  "Sweep")],
         inputs=lambda c: ["None (pure simulation; needs no other results)"], outputs=_power_outputs,
         prereq=_power_prereq,
         progress=lambda c: {"type": "count", "pattern": r"^P=", "total": max(1, len(_v(c, "powers", ()) or ()))},
         build=lambda c, out: (dict(kwargs=dict(X=float(_v(c, "X", 5.0)), n_sat=float(_v(c, "n_sat", 2e19)),
                                                powers=list(_v(c, "powers", ())), out=_rel(out),
                                                degenerate=1.0 if _v(c, "degenerate", True) else 0.0)),
                               f"X = {float(_v(c, 'X', 5.0)):+g} µm · {len(_v(c, 'powers', ()))} powers"),
         target=out_target, render=power_render, history=power_history),

    _tool(id="redemodulate", group="Analysis", title="Re-demodulate a run",
         short="New A / h_min on a saved pulse-chain run", tag="seconds",
         desc="Post-processing only, no new simulation: repeats the Pizzuto finite-dipole demodulation (harmonics "
              "1–4) of a saved single-pulse chain run (the X*.npz files in its folder) with a new tapping amplitude "
              "A, minimum tip height h_min or the LTEM (1−β) factor, computes the RMS of the low-passed dS_n/dt and "
              "S_n for every X and shows them below as normalized curves and a table. Note: the script's docstring "
              "gives A = 200 nm as the default; the actual default is 180 nm.",
         script="scripts/redemodulate.py", func="main", out=False, extra="rundir",
         runtime="A few seconds (≈ 0.2–0.5 s per X)",
         specs=lambda: [FieldSpec("A_nm", 180.0, "float", "Tapping amplitude A", "nm", "Tip tapping amplitude",
                                  "Demodulation", minimum=1e-3),
                        FieldSpec("h_min_nm", 0.0, "float", "Min. tip height h_min", "nm",
                                  "Lowest tip height in the tapping cycle", "Demodulation", minimum=0.0),
                        FieldSpec("ltem_factor", False, "bool", "LTEM (1−β) factor", "",
                                  "Multiply α by (1 − β) as in Pizzuto's LTEM FDM", "Demodulation")],
         inputs=_rundir_inputs,
         writes_input=_rundir_writes, prereq=_rundir_prereq,
         build=lambda c, out: (dict(kwargs=dict(run_dir=_rel(c["rundir"]), A_nm=float(_v(c, "A_nm", 180.0)),
                                                h_min_nm=float(_v(c, "h_min_nm", 0.0)),
                                                ltem_factor=bool(_v(c, "ltem_factor", False)))),
                               f"{Path(c['rundir']).name} · A = {float(_v(c, 'A_nm', 180.0)):g} nm, "
                               f"h_min = {float(_v(c, 'h_min_nm', 0.0)):g} nm" + (" · LTEM" if _v(c, "ltem_factor") else "")),
         target=redemod_target, render=redemod_render, history=redemod_history, prefer=redemod_prefer),

    _tool(id="compare_scenarios", group="Analysis", title="Compare steady states",
         short="Overlay fields, carriers and tip scans", tag="≈1 s",
         desc="Overlays several saved steady-state solutions (solution.npz) in one 2×2 figure: E_x, n and EL2⁺ at "
              "0.5 µm depth, the emission weight n_pulse·|E| and the normalized |tip signal| vs X (with the measured "
              "+25 V points), and writes a Markdown table of key values per case (P, τ_srh, S, I, n(0), E_x(±5), "
              "peak X). The axis ranges are fixed (E_x −1…6 kV/cm, densities 1e10…1e18 cm⁻³), so extreme cases may "
              "be clipped; the measured points are only meaningful at 25 V bias.",
         script="scripts/compare_scenarios.py", func="main", out=True, extra="solutions",
         runtime="≈ 0.3 s per solution (coarse grid); ≈ 1–2 s each on the production grid",
         inputs=_compare_inputs,
         outputs=lambda c: [new_dir_line("compare_scenarios"),
                            sub(_code("compare.png") + " — 2×2 comparison figure"),
                            sub(_code("compare.md") + " — table of values per case")],
         prereq=_compare_prereq,
         build=lambda c, out: (dict(kwargs=dict(out=_rel(Path(out) / "compare.png"),
                                                items=[[l, _rel(p)] for l, p in c["solutions"]])),
                               f"{len(c['solutions'])} solutions"),
         target=compare_target, render=compare_render, history=compare_history),

    _tool(id="show_solution", group="Analysis", title="Steady-state text report",
         short="Print currents, profiles and tip scan", tag="<1 s",
         desc="Loads a saved steady-state solution (.npz) and prints a text report: bias and electrode potentials, "
              "power, photocurrent and gain, convergence of each stage, lateral profile tables at 4 depths "
              "(0.2 / 0.5 / 1 / 2 µm, nearest grid centre — 0.12 / 0.62 µm etc. on a coarse grid), the tip-scan "
              "table and the far-field dipole. Writes no files. To explore the same solution interactively, use the "
              "Solution explorer page.",
         script="scripts/show_solution.py", func="main", out=False, extra="solution", runtime="< 1 s",
         inputs=lambda c: [_code(_rel(c["solution"])) if c.get("solution") else "(pick a .npz on the right)"],
         outputs=lambda c: ["Text output only (shown below; can be saved as .txt)"],
         prereq=lambda c: [cond("Pick an existing steady-state .npz",
                                bool(c.get("solution")) and Path(c["solution"]).exists(),
                                "" if c.get("solution") else "none selected", sel=not c.get("solution"))],
         build=lambda c, out: (dict(args=[_rel(c["solution"])]), Path(c["solution"]).name),
         render=text_render("show_solution.txt")),

    _tool(id="band_check", group="Analysis", title="Detection-band check",
         short="Is the weak-side emission outside the band?", tag="≈1 s",
         desc="Checks whether the emission on the weak side (X = −5 µm) merely falls outside the detection band: "
              "for route 1 (no tip field + FDM, P2mW_notipmap_degen) and route 2 (tip as an equipotential in "
              "Poisson, P2mW_sym) at X = +5, −5 and 0 it computes the spectrum of d²S₃/dt², band-limited amplitudes "
              "in 6 bands, the 10–90 % rise time and the −5/+5 ratio per band, and plots the normalized spectra "
              "(shaded: the 0.3–2 THz detection band). The figure is saved only after both routes have finished.",
         script="scripts/band_check.py", func=None, out=False, runtime="≈ 1–3 s",
         inputs=lambda c: [_code("results/pulse_chain/P2mW_notipmap_degen/X{+5, −5, 0}.npz") + " — route 1",
                           _code("results/tip_positions/P2mW_sym/") + " meta.json + X{+5, −5, 0}_h*.npz — route 2"],
         outputs=lambda c: ["Terminal tables (shown below)"],
         fixed=lambda c: [SLIDES / "sim_band_check.png"],
         prereq=lambda c: [req(_code("results/pulse_chain/P2mW_notipmap_degen/X{+5, −5, 0}.npz"),
                               pc_npz("P2mW_notipmap_degen", (5.0, -5.0, 0.0)), "pulse"),
                           req(_code("results/tip_positions/P2mW_sym/meta.json"), [TP / "P2mW_sym" / "meta.json"], "tip"),
                           req(_code("results/tip_positions/P2mW_sym/X{+5, −5, 0}_h*.npz"),
                               tp_npz("P2mW_sym", (5.0, -5.0, 0.0)), "tip")],
         target=lambda spec: SLIDES / "sim_band_check.png", render=png_render,
         history=lambda: fixed_history([SLIDES / "sim_band_check.png"])),

    _tool(id="waveform_phase", group="Analysis", title="Waveform phase fit",
         short="FFT phase-difference fit φ0 + 2πfτ", tag="seconds",
         desc="Compares two simulated detected waveforms (d²S₃/dt², 2.2 THz low-pass) the same way as the measured "
              "±25 V analysis: fits the phase difference φ(test) − φ(ref) = φ0 + 2πfτ over 0.25–1.6 THz, giving φ0 "
              "(0° same shape, 180° inverted, ±90° derivative relation), the delay τ and the phase residual, plus a "
              "time-domain best-shift check against same-sign and opposite-sign templates. The pairs come from "
              "route 1 (pulse_chain/P2mW_notipmap_degen) and route 2 (tip_positions/P2mW_sym); 3 more rows are added "
              "when the tip-potential runs (P2mW_sym_Vtip12 / Vtipm12) exist. The table is printed once at the end.",
         script="scripts/waveform_phase.py", func=None, out=False, runtime="A few seconds",
         inputs=lambda c: [_code("results/pulse_chain/P2mW_notipmap_degen/X{±5, +2.5, 0, −2.5}.npz") + " — route 1",
                           _code("results/tip_positions/P2mW_sym/") + " meta.json + X*_h*.npz — route 2",
                           _code("results/tip_positions/P2mW_sym_Vtip12, P2mW_sym_Vtipm12/") + " — optional"],
         outputs=lambda c: ["Stdout only: the phase-fit table (shown below; can be saved as text — the committed "
                            + _code("results/tip_positions/waveform_phase_table.txt") + " is this output)"],
         prereq=lambda c: [req(_code("results/pulse_chain/P2mW_notipmap_degen/X{±5, +2.5, 0, −2.5}.npz"),
                               pc_npz("P2mW_notipmap_degen", X5), "pulse"),
                           req(_code("results/tip_positions/P2mW_sym/meta.json"), [TP / "P2mW_sym" / "meta.json"], "tip"),
                           req(_code("results/tip_positions/P2mW_sym/X{±5, +2.5, 0, −2.5}_h*.npz"),
                               tp_npz("P2mW_sym", X5), "tip"),
                           req(_code("results/tip_positions/P2mW_sym_Vtipm12/X+5.0_h370.npz")
                               + " (extra tip-potential rows)",
                               [TP / "P2mW_sym_Vtipm12" / "X+5.0_h370.npz"], "tip", optional=True)],
         render=waveform_render,
         history=lambda: fixed_history([TP / "waveform_phase_table.txt"], "saved output")),

    _tool(id="dember_interference", group="Analysis", title="Dember interference",
         short="Lateral Dember term vs drift front", tag="5–10 min",
         desc="Tests whether the one-sidedness comes from interference between the lateral photo-Dember term (even "
              "in X, bias-independent) and the drift front (odd in X, ∝ V): scans X = −5…+5 µm at V = 25 V and 0 V "
              "with cold (300 K) and hot (1500 K) carriers (classical diffusion, no tip, 2 mW, t_end = 3 ps; 20 "
              "PulseSim runs in total) and prints the RMS of the low-passed dS₃/dt, normalized per temperature. "
              "Note: this is the older first-derivative RMS metric, so the values cannot be compared directly with "
              "the other tables (d²S/dt² band-limited amplitudes). The script has no parameters and writes no files.",
         script="scripts/dember_interference.py", func=None, out=False,
         runtime="≈ 5–10 min (20 PulseSim runs); 3–5 min without output between the two temperature blocks is normal",
         inputs=lambda c: ["None (pure simulation)"],
         outputs=lambda c: ["Stdout only: one block per temperature (parsed below into curves and a table; can be "
                            "saved as text)"],
         progress=lambda c: {"type": "count", "pattern": r"^\s+V =", "total": 4},
         render=dember_render),

    _tool(id="fig_slides", group="Figures", title="Slide figures", short="Simulation slides sim_*.png",
         tag="1–4 s",
         desc="Renders the English slide figures (results/slides/sim_*.png, ≈ 2550 px wide) from results saved on "
              "disk. One figure per run: the script's all mode aborts at the first figure whose inputs are missing, "
              "so each figure runs on its own here. dynamics runs a 0.6 ps PulseSim in-process and bending does one "
              "bare-field Poisson solve. The “Inputs” column of the list on the right shows live whether each "
              "figure's input files are present.",
         script="scripts/fig_slides.py", func=None, out=False, extra="slide_keys",
         inputs=lambda c: [f"Selected figure: {_code(c.get('key'))} — {keyinfo(SLIDE_KEYS, c.get('key'))['name']}",
                           _code("data/measured_tip_scan.csv") + " (read on import)"]),

    _tool(id="fig_cartoons", group="Figures", title="Physics cartoons", short="Data-free schematics cartoon_*.png",
         tag="≈1 s",
         desc="Draws physics schematics for the slides from analytic shapes only, with no simulation data: three "
              "field distortions, the three-stage chain, the odd/even decomposition of the X scan, the tip DC "
              "potential and spot broadening with power. Render one of them or all 5. Note that cartoon_power.png "
              "and the slide figure sim_power.png are different figures.",
         script="scripts/fig_cartoons.py", func=None, out=False, extra="cartoon_keys",
         inputs=lambda c: ["None (analytic schematics)"]),

    _tool(id="fig_mechanism", group="Figures", title="Mechanism figure (Chinese)",
         short="0.6 ps simulation → mechanism.png", tag="≈3 s",
         desc="Mechanism schematic with Chinese labels, computed from the model itself: runs a 0.6 ps PulseSim "
              "(2 mW, spot at x = 0) and draws (a) the net space charge at 0.15 ps, (b) electron, hole and net sheet "
              "densities along x, (c) the near-field E_z at h = 100 nm vs tip position (odd term, from the model), "
              "together with an illustrative even term (a 4 µm wide Gaussian, not a model result) and their "
              "one-sided sum. The Chinese labels use a system font.",
         script="scripts/fig_mechanism.py", func=None, out=False,
         runtime="≈ 3 s (the first run may take 10–15 s longer while the font cache is built)",
         inputs=lambda c: ["None (runs a 0.6 ps PulseSim in-process)"],
         fixed=lambda c: [R / "mechanism.png"],
         target=lambda spec: R / "mechanism.png", render=png_render,
         history=lambda: fixed_history([R / "mechanism.png"])),

    _tool(id="fig_dipole_geometry", group="Figures", title="Dipole geometry (Chinese)",
         short="Vertical vs lateral dipole → dipole_geometry.png", tag="≈1 s",
         desc="Analytic schematic, with Chinese labels, of 2-D line-charge dipole fields: compares the field lines "
              "of a vertical dipole (classical photo-Dember) and a lateral dipole (drift separation) and plots the "
              "E_z seen by the tip at h = 50, 200 and 500 nm vs x. Uses no simulation data. The English version is "
              "dipole (sim_dipole_geometry.png) under Slide figures.",
         script="scripts/fig_dipole_geometry.py", func=None, out=False, runtime="≈ 1 s",
         inputs=lambda c: ["None (analytic)"],
         fixed=lambda c: [R / "dipole_geometry.png"],
         target=lambda spec: R / "dipole_geometry.png", render=png_render,
         history=lambda: fixed_history([R / "dipole_geometry.png"])),

    _tool(id="numerics", group="Maintenance", title="Numerical self-test",
         short="Poisson / equilibrium / Jacobian / dark current", tag="≈2 s",
         desc="Validation suite of the steady-state solver (small 86×50 grid): (1) bare Poisson field between "
              "coplanar thin electrodes within 12 % of the analytic V/(π√(a²−x²)), E_z/E_x < 5 % at the gap centre; "
              "(2) zero residual at equilibrium (V = 0, G = 0); (3) finite-difference check of the full coupled "
              "Jacobian (inv_dt = 0 and 3, relative error < 1e-5 per block); (4) 10 V dark state: converges, "
              "|I| < 1e-3 mA, mean EL2⁺ between 0.5 and 3 × 10¹⁵ cm⁻³. Confirms that the installation and the "
              "numerical environment work; the remaining tests are skipped after the first failure.",
         script="tests/test_numerics.py", func=None, out=False, runtime="≈ 2 s",
         inputs=lambda c: ["None (the ltemsim library itself)"],
         outputs=lambda c: ["Stdout only: one “… ok” line per passed test (shown below as a checklist)"],
         progress=lambda c: {"type": "count", "pattern": r"^(poisson|equilibrium|jacobian|dark) ok", "total": 4},
         render=numerics_render),
]
BY_ID = {t["id"]: t for t in TOOLS}
SCRIPT_TO_TOOL = {t["script"]: t["id"] for t in TOOLS}


def tool_of(job) -> str | None:
    if job is None or job.kind != "tool":
        return None
    return SCRIPT_TO_TOOL.get(job.spec.get("script"))


# ============================================================================ pickers
class _PathPicker(QWidget):
    """A titled combo of candidate paths plus a browse button; value() is the selected Path (or None)."""
    changed = Signal()

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(4)
        self.lay.addWidget(label(title, "cardTitle", wrap=True))
        row = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(16)
        self.combo.currentIndexChanged.connect(self._on_change)
        row.addWidget(self.combo, 1)
        row.addWidget(button("Browse…", self._browse))
        self.lay.addLayout(row)

    # subclasses: candidates() -> [Path], text(Path) -> str, _browse(); optional default() -> Path | None
    def default(self):
        return None

    def refresh(self):
        cur = self.value() or self.default()
        self.combo.blockSignals(True)
        self.combo.clear()
        for p in self.candidates():
            self.combo.addItem(self.text(p), str(Path(p).resolve()))
        if cur is not None:
            self.set(cur)
        self.combo.blockSignals(False)
        self._on_change()

    def value(self) -> Path | None:
        d = self.combo.currentData()
        return Path(d) if d else None

    def set(self, path):
        p = Path(path).resolve()
        i = self.combo.findData(str(p))
        if i < 0:
            self.combo.addItem(self.text(p), str(p))
            i = self.combo.count() - 1
        self.combo.setCurrentIndex(i)

    def _on_change(self, *_):
        self.changed.emit()


class RunDirPicker(_PathPicker):
    """Single pulse-chain run folder (results/pulse_chain/*, results/gui_runs/pulse_*, or any folder with X*.npz)."""

    def __init__(self, parent=None):
        super().__init__("Run folder (single-pulse chain output with X*.npz)", parent)
        self.info = label("", "muted", wrap=True)
        self.lay.addWidget(self.info)
        self.refresh()

    def candidates(self):
        return pulse_run_dirs()

    @staticmethod
    def text(d: Path) -> str:
        return f"{_rel(d).removeprefix('results/')}  ·  {len(x_files(d))} X"

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Choose a single-pulse chain run folder", str(PC))
        if d:
            self.set(d)

    def _on_change(self, *_):
        d = self.value()
        if d is None:
            self.info.setText("No run folder with X*.npz files found.")
        else:
            xs = [x for x in (_xval(f) for f in x_files(d)) if x is not None]
            done = sorted(f.name for f in d.glob("redemod_*.json"))
            txt = f"X = {', '.join(f'{x:+g}' for x in sorted(xs))} µm" if xs else "No X*.npz in this folder"
            if done:
                txt += "  ·  existing: " + ", ".join(done)
            self.info.setText(txt)
        self.changed.emit()


class SolutionPicker(_PathPicker):
    def __init__(self, parent=None):
        super().__init__("Steady-state solution (.npz)", parent)
        self.lay.addWidget(label("Coarse-grid solutions come from results/coarse_scenarios; solution.npz files exist "
                                 "only after a local run (they are not committed).",
                                 "muted", wrap=True))
        self.lay.addWidget(button("Open in Solution explorer", self._explore, "flat"), 0, Qt.AlignmentFlag.AlignLeft)
        self.refresh()

    def candidates(self):
        return solution_files()

    def default(self):
        p = R / "coarse_scenarios" / "solutions" / "coarse_P2.00.npz"
        return p if p.exists() else None

    @staticmethod
    def text(p: Path) -> str:
        return _rel(p)

    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(self, "Choose a steady-state solution", str(R), "Solution (*.npz)")
        if f:
            self.set(f)

    def _explore(self):
        p = self.value()
        if p is not None and p.exists():
            nav.navigate("explorer", path=str(p))


class SolutionTable(QWidget):
    """Multi-select list of solutions with editable legend labels."""
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(label("Solutions to compare (tick them; double-click a legend label to edit it)", "cardTitle",
                            wrap=True))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["", "Legend label", "File"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                   QAbstractItemView.EditTrigger.EditKeyPressed)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(1, 130)
        self.table.itemChanged.connect(lambda *_: self._changed())
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.count = label("", "muted")
        row.addWidget(self.count, 1)
        for text, slot in (("Add files…", self._browse), ("Uncheck all", self.clear_checks)):
            b = button(text, slot, "flat")
            row.addWidget(b)
        lay.addLayout(row)
        self._paths: list[str] = []
        self.refresh()

    def refresh(self):
        for p in solution_files():
            if str(p.resolve()) not in self._paths:
                self._add(p, False)
        self._changed()

    def _add(self, p: Path, checked: bool):
        p = Path(p).resolve()
        if str(p) in self._paths:
            self.table.item(self._paths.index(str(p)), 0).setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            return
        self.table.blockSignals(True)
        r = self.table.rowCount()
        self.table.insertRow(r)
        c0 = QTableWidgetItem()
        c0.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        c0.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        c1 = QTableWidgetItem(solution_label(p))
        c2 = QTableWidgetItem(_rel(p))
        c2.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        c2.setToolTip(str(p))
        for c, it in enumerate((c0, c1, c2)):
            self.table.setItem(r, c, it)
        self._paths.append(str(p))
        self.table.blockSignals(False)

    def _browse(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Add steady-state solutions", str(R), "Solution (*.npz)")
        for f in files:
            self._add(Path(f), True)
        self._changed()

    def clear_checks(self):
        self.table.blockSignals(True)
        for r in range(self.table.rowCount()):
            self.table.item(r, 0).setCheckState(Qt.CheckState.Unchecked)
        self.table.blockSignals(False)
        self._changed()

    def set_checked(self, items):
        """items: [(label or None, path), ...] — check these rows (adding files as needed)."""
        self.clear_checks()
        for lab, p in items:
            self._add(Path(p), True)
            if lab:
                self.table.item(self._paths.index(str(Path(p).resolve())), 1).setText(lab)
        self._changed()

    def value(self) -> list[tuple[str, Path]]:
        out = []
        for r, p in enumerate(self._paths):
            if self.table.item(r, 0).checkState() == Qt.CheckState.Checked:
                lab = self.table.item(r, 1).text().strip() or solution_label(Path(p))
                out.append((lab, Path(p)))
        return out

    def _changed(self):
        n = len(self.value())
        self.count.setText(f"{n} of {len(self._paths)} selected")
        self.changed.emit()


class KeyPicker(QWidget):
    """Which figure of a multi-figure script to render, with live input status per figure."""
    changed = Signal()

    def __init__(self, keys, title, parent=None):
        super().__init__(parent)
        self.keys = keys
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(label(title, "cardTitle", wrap=True))
        self.table = QTableWidget(len(keys), 3)
        self.table.setHorizontalHeaderLabels(["Figure", "Content", "Inputs"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        for r, k in enumerate(keys):
            a = QTableWidgetItem(k["key"])
            a.setToolTip("Output: " + ", ".join(f"results/slides/{f}" for f in k["files"]))
            self.table.setItem(r, 0, a)
            b = QTableWidgetItem(k["name"])
            b.setToolTip(k["name"])
            self.table.setItem(r, 1, b)
            self.table.setItem(r, 2, QTableWidgetItem(""))
        self.table.resizeRowsToContents()
        self.table.currentCellChanged.connect(lambda *_: self.changed.emit())
        lay.addWidget(self.table, 1)
        self.table.selectRow(0)
        self.refresh()

    def refresh(self):
        t = ThemeManager.instance().tokens
        for r, k in enumerate(self.keys):
            rq = k["reqs"]()
            miss = [q for q in rq if not q["ok"] and not q["optional"]]
            if not rq:
                txt, col = "— none needed", t.muted
            elif miss:
                txt, col = f"✗ {len(miss)} missing", t.danger
            else:
                txt, col = "✓ complete", t.success
            it = self.table.item(r, 2)
            it.setText(txt)
            it.setForeground(QColor(col))
            it.setToolTip("\n".join(_plain(q["label"]) + (f" ({q['detail']})" if q["detail"] else "") for q in miss))

    def value(self) -> str:
        r = self.table.currentRow()
        return self.keys[r if r >= 0 else 0]["key"]

    def set(self, key):
        for r, k in enumerate(self.keys):
            if k["key"] == key:
                self.table.selectRow(r)
                return


# ============================================================================ list items
class _ToolItem(QWidget):
    def __init__(self, tool, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 10, 6)
        lay.setSpacing(8)
        col = QVBoxLayout()
        col.setSpacing(1)
        self.title = _Elided()                          # elide (not clip) when the list is narrow
        self.title.setStyleSheet("font-weight: 600;")
        self.title.setText(tool["title"])
        self.desc = _Elided("muted")
        self.desc.setStyleSheet("font-size: 11px;")
        self.desc.setText(tool["short"])
        col.addWidget(self.title)
        col.addWidget(self.desc)
        lay.addLayout(col, 1)
        self.tag = label(tool["tag"], "muted")
        self.tag.setStyleSheet("font-size: 11px;")
        self.tag.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        lay.addWidget(self.tag, 0, Qt.AlignmentFlag.AlignTop)
        self._tag = tool["tag"]

    def set_state(self, job, accent: str):
        active = job is not None and job.active
        text = ("Queued" if job.status == QUEUED else "Running") if active else self._tag
        style = f"font-size: 11px; color: {accent}; font-weight: 600;" if active else "font-size: 11px;"
        if (text, style) != (self.tag.text(), self.tag.styleSheet()):     # called every second while a job runs
            self.tag.setText(text)
            self.tag.setStyleSheet(style)


class _Elided(QLabel):
    """Single-line label that elides its text to the available width instead of forcing a minimum width."""

    def __init__(self, obj=None):
        super().__init__()
        if obj:
            self.setObjectName(obj)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def setText(self, text):
        self._full = text
        self.setToolTip(text)
        self._elide()
        self.updateGeometry()

    def sizeHint(self):
        self.ensurePolished()
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + 6, super().sizeHint().height())

    def minimumSizeHint(self):
        return QSize(60, super().minimumSizeHint().height())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._elide()

    def _elide(self):
        self.ensurePolished()
        super().setText(self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, max(20, self.width())))


class _VSplitter(QSplitter):
    """Vertical splitter whose preferred height is its minimum: the page then fills the window instead of growing
    to the size hints of the output views (the app's page holder sizes pages by height-for-width)."""

    def sizeHint(self):
        return QSize(super().sizeHint().width(), self.minimumSizeHint().height())


# ============================================================================ page
class ToolsPage(Page):
    title = "Tools & figures"
    subtitle = ("The remaining analysis, figure and maintenance scripts: pick a tool, check its description and live "
                "prerequisites, set the parameters and run; figures, tables and text output appear below.")

    def __init__(self):
        super().__init__()
        self.current: str | None = None
        self.tool_jobs: dict = {}
        self.param_pages: dict = {}
        self.entries: list[dict] = []
        self._entry = None
        self._plots: list[PlotWidget] = []
        self._items: dict = {}

        self.btn_check = button("  Recheck", self.refresh_all, "flat",
                                "Re-check prerequisite files, available solutions and existing outputs")
        self.header.add_action(self.btn_check)
        self.run_btn = button("  Run", self.run, "primary", "Run the selected tool in a background subprocess")
        self.header.add_action(self.run_btn)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._build_list())
        split.addWidget(self._build_right())
        split.setStretchFactor(1, 1)
        split.setSizes([300, 900])
        self.body.addWidget(split, 1)

        jm = JobManager.instance()
        for j in jm.jobs:                      # latest job of each tool (restores the log after a restart)
            tid = tool_of(j)
            if tid:
                self.tool_jobs[tid] = j
        jm.job_added.connect(self._on_job_added)
        jm.job_changed.connect(self._on_job_changed)
        jm.job_finished.connect(self._on_job_finished)
        ThemeManager.instance().changed.connect(self._retheme)
        self._retheme()

    # ------------------------------------------------------------------ layout
    def _build_list(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search tools (name / script / description)…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        lay.addWidget(self.search)
        self.list = QListWidget()
        self.list.setObjectName("toolList")
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list.currentItemChanged.connect(self._on_list)
        self._headers = {}
        for g in GROUPS:
            hdr = QListWidgetItem()
            hdr.setFlags(Qt.ItemFlag.NoItemFlags)
            hdr.setSizeHint(QSize(200, 30))
            self.list.addItem(hdr)
            hl = label(g, "sectionHeader")
            hl.setContentsMargins(10, 0, 0, 0)
            self.list.setItemWidget(hdr, hl)
            self._headers[g] = hdr
            for t in TOOLS:
                if t["group"] != g:
                    continue
                it = QListWidgetItem()
                it.setData(Qt.ItemDataRole.UserRole, t["id"])
                it.setSizeHint(QSize(200, 52))
                it.setToolTip(t["script"])
                self.list.addItem(it)
                iw = _ToolItem(t)
                self.list.setItemWidget(it, iw)
                self._items[t["id"]] = (it, iw)
        lay.addWidget(self.list, 1)
        w.setMinimumWidth(235)
        w.setMaximumWidth(420)
        return w

    def _build_right(self):
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 0, 0, 0)
        rl.setSpacing(8)
        head = QVBoxLayout()                            # title row + meta row (badges and runtime)
        head.setSpacing(2)
        tr = QHBoxLayout()
        tr.setSpacing(8)
        self.t_title = _Elided("cardTitle")             # long titles elide instead of widening the page
        self.t_title.setStyleSheet("font-size: 18px;")
        tr.addWidget(self.t_title, 1)
        self.btn_script = button("View script", self._open_script, "flat", "Open the script source in the default app")
        tr.addWidget(self.btn_script)
        head.addLayout(tr)
        mr = QHBoxLayout()
        mr.setSpacing(10)
        self.t_group = label("", "badge")
        mr.addWidget(self.t_group)
        self.t_prereq = label("", "badgeOk")
        mr.addWidget(self.t_prereq)
        self.t_fixed = label("Fixed output path", "badgeWarn")
        mr.addWidget(self.t_fixed)
        self.t_runtime = _Elided("muted")
        mr.addWidget(self.t_runtime, 1)
        head.addLayout(mr)
        rl.addLayout(head)

        self.warn = QFrame()
        self.warn.setObjectName("warnBox")
        wl = QHBoxLayout(self.warn)
        wl.setContentsMargins(12, 7, 12, 7)
        self.warn_text = label("", wrap=True)
        self.warn_text.setTextFormat(Qt.TextFormat.RichText)
        wl.addWidget(self.warn_text, 1)
        rl.addWidget(self.warn)
        self.note = label("", "muted", wrap=True, selectable=True)
        self.note.hide()
        rl.addWidget(self.note)

        vs = _VSplitter(Qt.Orientation.Vertical)
        vs.setChildrenCollapsible(False)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.setChildrenCollapsible(False)
        ic = card()
        il = QVBoxLayout(ic)
        il.setContentsMargins(14, 10, 6, 10)
        il.setSpacing(4)
        il.addWidget(label("Description", "cardTitle"))
        # a text browser rather than a QLabel: long paths wrap anywhere when the card is narrow (no sideways scroll)
        self.info = QTextBrowser()
        self.info.setObjectName("toolInfo")
        self.info.setStyleSheet("QTextBrowser#toolInfo { border: none; background: transparent; padding: 0; }")
        self.info.setFrameShape(QFrame.Shape.NoFrame)
        self.info.viewport().setAutoFillBackground(False)
        self.info.document().setDocumentMargin(1)
        self.info.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.info.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.info.setOpenLinks(False)
        self.info.anchorClicked.connect(lambda url: self._on_link(url.toString()))
        il.addWidget(self.info, 1)
        ic.setMinimumWidth(190)
        top.addWidget(ic)
        pc = card()
        pl = QVBoxLayout(pc)
        pl.setContentsMargins(12, 10, 12, 10)
        pl.setSpacing(6)
        pl.addWidget(label("Parameters & inputs", "cardTitle"))
        self.param_stack = QStackedWidget()
        pl.addWidget(self.param_stack, 1)
        pc.setMinimumWidth(300)                         # raised per tool in _activate
        self.param_card = pc
        top.addWidget(pc)
        top.setStretchFactor(0, 1)
        top.setSizes([470, 440])
        vs.addWidget(top)
        self._vs = vs

        outw = QWidget()                                # output source selector above the output tabs
        ol = QVBoxLayout(outw)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(4)
        srow = QHBoxLayout()
        srow.setSpacing(6)
        srow.addWidget(label("Output source", "muted"))
        self.src = QComboBox()
        self.src.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.src.setMinimumContentsLength(18)
        self.src.setToolTip("Existing outputs of this tool: jobs run in this app, results under results/gui_runs and "
                            "committed results")
        self.src.view().setMinimumWidth(520)
        self.src.view().setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.src.currentIndexChanged.connect(self._on_src)
        srow.addWidget(self.src, 1)
        self.btn_folder = button("Show in Finder", self._open_folder, "flat",
                                 "Reveal the folder of the current output in Finder")
        srow.addWidget(self.btn_folder)
        ol.addLayout(srow)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(False)
        run_tab = QWidget()
        rtl = QVBoxLayout(run_tab)
        rtl.setContentsMargins(12, 12, 12, 12)
        self.panel = RunPanel()
        rtl.addWidget(self.panel)
        self.tabs.addTab(run_tab, "Run")
        self.tabs.tabBar().setElideMode(Qt.TextElideMode.ElideMiddle)
        ol.addWidget(self.tabs, 1)
        vs.addWidget(outw)
        vs.setStretchFactor(1, 1)
        vs.setSizes([280, 490])
        rl.addWidget(vs, 1)
        return right

    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, "_sized", False):          # first show: give the outputs ~60 % of the height
            self._sized = True
            QTimer.singleShot(0, self._initial_split)

    def _initial_split(self):
        h = sum(self._vs.sizes())
        if h > 0:
            top = max(190, min(300, int(0.4 * h)))
            self._vs.setSizes([top, h - top])

    # ------------------------------------------------------------------ theme
    def _retheme(self, *_):
        t = ThemeManager.instance().tokens
        self.run_btn.setIcon(icons.icon("play", t.on_accent))
        self.btn_check.setIcon(icons.icon("refresh", t.muted))
        self.list.setStyleSheet(
            f"QListWidget#toolList {{ padding: 4px; }}"
            f"QListWidget#toolList::item {{ border-radius: 8px; margin: 1px 2px; }}"
            f"QListWidget#toolList::item:selected {{ background: {t.accent_soft}; }}"
            f"QListWidget#toolList::item:hover:!selected {{ background: {t.surface2}; }}")
        c = QColor(t.warning)
        self.warn.setStyleSheet(
            f"QFrame#warnBox {{ background: rgba({c.red()},{c.green()},{c.blue()},0.10); "
            f"border: 1px solid rgba({c.red()},{c.green()},{c.blue()},0.55); border-radius: 10px; }}")
        for tid in self._items:
            self._set_item_state(tid)
        self._refresh_extras(KeyPicker)                 # status colours
        if self.current:
            self._refresh_info()
            if self._entry is not None:
                self._display(self._entry, switch=False)

    # ------------------------------------------------------------------ list
    def _filter(self, text):
        q = text.strip().lower()
        vis = {g: False for g in GROUPS}
        for t in TOOLS:
            it, _ = self._items[t["id"]]
            hay = f"{t['title']} {t['short']} {t['desc']} {t['script']} {t['id']}".lower()
            show = not q or q in hay
            it.setHidden(not show)
            vis[t["group"]] |= show
        for g, hdr in self._headers.items():
            hdr.setHidden(not vis[g])

    def _on_list(self, cur, _prev):
        if cur is None:
            return
        tid = cur.data(Qt.ItemDataRole.UserRole)
        if tid and tid != self.current:
            self._activate(tid)

    def select_tool(self, tid: str):
        if tid not in BY_ID:
            return
        it, _ = self._items[tid]
        self.list.blockSignals(True)
        self.list.setCurrentItem(it)
        self.list.blockSignals(False)
        self.list.scrollToItem(it)
        if tid != self.current:
            self._activate(tid)

    def _activate(self, tid: str):
        self.current = tid
        tool = BY_ID[tid]
        w, form, extra = self._param_page(tid)
        self.param_stack.setCurrentWidget(w)
        self.param_card.setVisible(form is not None or extra is not None)
        # wide enough for the form / picker (the card scrolls vertically, never horizontally)
        self.param_card.setMinimumWidth(max(300, w.widget().minimumSizeHint().width() + 40))
        self.t_title.setText(tool["title"])
        self.t_group.setText(tool["group"])
        self.note.hide()
        job = self.tool_jobs.get(tid)
        self.panel.attach(job)
        self._update_run_state()
        self._refresh_info()
        self._load_history(key=self._ctx(tid).get("key"))
        if job is not None and (job.active or job.status == FAILED):
            self.tabs.setCurrentIndex(0)
        elif self.tabs.count() > 1 and self.entries:
            self.tabs.setCurrentIndex(1)
        else:
            self.tabs.setCurrentIndex(0)

    # ------------------------------------------------------------------ parameters
    def _param_page(self, tid):
        if tid in self.param_pages:
            return self.param_pages[tid]
        tool = BY_ID[tid]
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(10)
        kind = tool.get("extra")
        pickers = {"rundir": RunDirPicker, "solution": SolutionPicker, "solutions": SolutionTable}
        if kind in KEYSETS:
            extra = KeyPicker(KEYSETS[kind], "Figure to render (output to results/slides/)")
        else:
            extra = pickers[kind]() if kind in pickers else None
        tall = kind in ("solutions", "slide_keys", "cartoon_keys")      # tables that take the card's height
        if extra is not None:
            l.addWidget(extra, 1 if tall else 0)
            extra.changed.connect(lambda t=tid: self._on_inputs_changed(t))
        specs = tool["specs"]() if tool.get("specs") else []
        form = None
        if specs:
            form = ParamForm(specs, toolbar=False, scroll=False)
            for r in form.rows.values():
                if isinstance(r.editor, QLineEdit):
                    r.editor.setCursorPosition(0)        # long tuples: show the start of the list
            form.changed.connect(lambda t=tid: self._on_inputs_changed(t))
            l.addWidget(form, 1)
        elif not tall:
            l.addStretch(1)                             # (no form and no picker: the card is hidden)
        sa = QScrollArea()                              # natural heights; the whole card scrolls when short
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.Shape.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sa.setWidget(w)
        self.param_stack.addWidget(sa)
        self.param_pages[tid] = (sa, form, extra)
        return self.param_pages[tid]

    def _ctx(self, tid) -> dict:
        _, form, extra = self._param_page(tid)
        c = {"v": form.values() if form is not None else {}}
        kind = BY_ID[tid].get("extra")
        if extra is not None:                           # ctx["rundir" | "solution" | "solutions" | "key"]
            c["key" if kind in KEYSETS else kind] = extra.value()
        return c

    def _on_inputs_changed(self, tid):
        if tid != self.current:
            return
        self._refresh_info()
        tool = BY_ID[tid]
        if tool.get("extra") in KEYSETS:
            key = self._ctx(tid).get("key")
            if not self._entry or self._entry.get("key") != key:
                self._select_key_entry(key)
        elif tool.get("prefer") and self.entries:
            idx = tool["prefer"](self._ctx(tid), self.entries)
            if idx is not None and self.entries[idx] is not self._entry:
                self.src.blockSignals(True)
                self.src.setCurrentIndex(idx)
                self.src.blockSignals(False)
                self._display(self.entries[idx], switch=False)

    # ------------------------------------------------------------------ description / prerequisites
    @staticmethod
    def _checks(tool, ctx):
        """(prerequisites, fixed output files, file written into the input folder or None)."""
        try:
            pre = tool["prereq"](ctx)
        except Exception as e:      # never let a check break the page
            pre = [cond(f"Prerequisite check failed: {_esc(e)}", False, hard=False)]
        fixed = tool["fixed"](ctx) if tool.get("fixed") else []
        writes = tool["writes_input"](ctx) if tool.get("writes_input") else None
        return pre, fixed, writes

    def _refresh_info(self):
        tid = self.current
        if not tid:
            return
        tool = BY_ID[tid]
        t = ThemeManager.instance().tokens
        ctx = self._ctx(tid)
        pre, fixed, writes = self._checks(tool, ctx)
        runtime = tool["runtime"](ctx) if callable(tool["runtime"]) else tool["runtime"]
        head = runtime.split(" (")[0].split("; ")[0]
        self.t_runtime.setText("Runtime: " + head[:1].lower() + head[1:])
        self.t_runtime.setToolTip(runtime)

        missing = [p for p in pre if not p["ok"] and not p["optional"]]
        if not pre:
            self.t_prereq.setText("No prerequisites")
            self.t_prereq.setObjectName("badgeOk")
        elif missing and all(p.get("sel") for p in missing):
            self.t_prereq.setText("Pick an input on the right")
            self.t_prereq.setObjectName("badgeWarn")
        elif missing:
            self.t_prereq.setText(f"✗ {len(missing)} prerequisite{'s' if len(missing) > 1 else ''} missing")
            self.t_prereq.setObjectName("badgeErr")
        else:
            self.t_prereq.setText("✓ Prerequisites met")
            self.t_prereq.setObjectName("badgeOk")
        repolish(self.t_prereq)
        self.t_fixed.setVisible(bool(fixed) or writes is not None)
        self.t_fixed.setText("Writes into input folder" if writes is not None else "Fixed output path")

        # warning banner
        warn = ""
        if fixed:
            n_ex = sum(f.exists() for f in fixed)
            n_git = sum(committed(f) for f in fixed if f.exists())
            warn = (f"<b style='color:{t.warning}'>⚠ Fixed output path</b>&nbsp;&nbsp;This script can only write to "
                    + ", ".join(_code(_rel(f)) for f in fixed) + ".")
            if n_ex:
                warn += (f" Running it overwrites {n_ex} existing file{'s' if n_ex > 1 else ''}"
                         + (f" ({n_git} committed to git)" if n_git else "")
                         + "; you will be asked to confirm first and can back them up to results/gui_runs/.")
        elif writes is not None:
            warn = (f"<b style='color:{t.warning}'>⚠ Writes into the input folder</b>&nbsp;&nbsp;The result JSON goes "
                    "straight into the selected run folder (not a new folder): " + _code(_rel(writes)))
            if Path(writes).exists():
                warn += " — a file with this name already exists; you will be asked before it is overwritten."
            elif curated(Path(writes).parent):
                warn += " — this is a committed result folder; you will be asked to confirm first."
        self.warn_text.setText(warn)
        self.warn.setVisible(bool(warn))

        # description card
        H = [f"<p style='margin:0 0 2px 0'>{_esc(tool['desc'])}</p>"]

        def sec(name):
            H.append(f"<p style='margin:12px 0 3px 0'><b style='color:{t.accent}'>{name}</b></p>")

        def bullets(lines):
            if not lines:
                H.append(f"<span style='color:{t.muted}'>None</span>")
                return
            rows = []
            for s in lines:
                if s.startswith("\x01"):
                    rows.append(f"<tr><td></td><td><table cellspacing='0' cellpadding='0'><tr><td valign='top' "
                                f"style='color:{t.muted}'>&nbsp;–&nbsp;</td><td>{s[1:]}</td></tr></table></td></tr>")
                else:
                    rows.append(f"<tr><td valign='top' style='color:{t.muted}'>•&nbsp;</td><td>{s}</td></tr>")
            H.append("<table cellspacing='0' cellpadding='1'>" + "".join(rows) + "</table>")

        sec("Prerequisites")
        if not pre:
            H.append(f"<span style='color:{t.success}'>✓</span> None (does not depend on results of other scripts)")
        else:
            rows = []
            for p in pre:
                if p["ok"]:
                    mark, col = "✓", t.success
                elif p["optional"]:
                    mark, col = "○", t.muted
                else:
                    mark, col = "✗", t.danger
                txt = p["label"]
                if p["optional"]:
                    txt += f" <span style='color:{t.muted}'>(optional)</span>"
                if p["detail"]:
                    txt += f" <span style='color:{t.muted if p['ok'] else col}'>— {_esc(p['detail'])}</span>"
                if not p["ok"] and p.get("producer"):
                    target, how = p["producer"]
                    txt += (f"<br><span style='color:{t.muted}'>How to get it: {_esc(how)}.</span> "
                            f"<a href='go:{target}'>"
                            f"<span style='color:{t.accent}'>Go to {_esc(producer_name(target))}</span></a>")
                rows.append(f"<tr><td valign='top' style='color:{col}; font-weight:700'>{mark}&nbsp;</td>"
                            f"<td style='padding-bottom:3px'>{txt}</td></tr>")
            H.append("<table cellspacing='0' cellpadding='1'>" + "".join(rows) + "</table>")
        sec("Inputs")
        bullets(tool["inputs"](ctx))
        sec("Outputs")
        outs = list(tool["outputs"](ctx))
        for f in fixed:
            notes = ["fixed path"]
            if f.exists():
                notes.append("exists, will be overwritten")
                if committed(f):
                    notes.append("committed to git")
            outs.append(_code(_rel(f)) + f" <span style='color:{t.warning}'>({'; '.join(notes)})</span>")
        if writes is not None:
            outs.append(_code(_rel(writes)) + f" <span style='color:{t.warning}'>(written into the selected run folder"
                        + ("; exists, will be overwritten" if Path(writes).exists() else "") + ")</span>")
        bullets(outs)
        sec("Runtime")
        H.append(_esc(runtime))
        sec("Script")
        H.append(_code(tool["script"]) + f" <span style='color:{t.muted}'>— "
                 + (f"calls {_esc(tool['func'])}() in a subprocess" if tool["func"]
                    else "runs as __main__ in a subprocess")
                 + ", working directory = repository root</span>")
        sb = self.info.verticalScrollBar()
        pos = sb.value()                                # keep the reading position while parameters are edited
        self.info.setHtml("".join(H))
        sb.setValue(pos)

    def _on_link(self, href: str):
        if not href.startswith("go:"):
            return
        target = href[3:]
        if target.startswith("tool:"):
            self.select_tool(target[5:])
        else:
            nav.navigate(target)

    # ------------------------------------------------------------------ run
    @staticmethod
    def _active_job(tid):
        """Newest queued/running job of the tool (also ones started from the jobs page), or None."""
        return next((j for j in reversed(JobManager.instance().jobs) if j.active and tool_of(j) == tid), None)

    def _update_run_state(self):
        j = self._active_job(self.current) if self.current else None
        busy = j is not None
        self.run_btn.setEnabled(bool(self.current) and not busy)
        self.run_btn.setText(("  Queued…" if j.status == QUEUED else "  Running…") if busy else "  Run")

    def run(self):
        tid = self.current
        if not tid:
            return None
        tool = BY_ID[tid]
        _, form, _ = self._param_page(tid)
        if form is not None and not form.is_valid():
            error_box(self, "Invalid parameters", "These values cannot be parsed or are out of range:\n"
                      + ", ".join(form.invalid_fields()))
            return None
        ctx = self._ctx(tid)
        pre, fixed, writes = self._checks(tool, ctx)
        hard = [p for p in pre if p.get("hard") and not p["ok"]]
        if hard:
            error_box(self, "Cannot run", "\n".join("• " + _plain(p["label"])
                                                 + (f" ({p['detail']})" if p["detail"] else "") for p in hard))
            return None
        missing = [p for p in pre if not p["ok"] and not p["optional"]]
        if missing:
            r = self._confirm_missing(tool, missing)
            if r is None:
                return None
            if r.startswith("go:"):
                self._on_link(r)
                return None
        fixed = [f for f in fixed if f.exists()]
        backup = None
        if fixed:
            r = self._confirm_overwrite(tool, fixed)
            if r is None:
                return None
            if r == "backup":
                try:
                    backup = self._backup(tid, fixed)
                except OSError as e:
                    error_box(self, "Backup failed", f"Could not back up the files that would be overwritten: {e}")
                    return None
        if writes is not None and (Path(writes).exists() or curated(Path(writes).parent)) \
                and not self._confirm_write(Path(writes)):
            return None
        out = paths.new_run_dir(f"tool_{tid}") if tool.get("out") else None
        call, suffix = tool["build"](ctx, out)
        prog = tool.get("progress")
        prog = prog(ctx) if callable(prog) else prog
        title = tool["title"] + (f" · {suffix}" if suffix else "")
        job = JobManager.instance().submit(title, "tool", tool["script"], tool["func"], args=call.get("args"),
                                           kwargs=call.get("kwargs"), out=out, progress=prog, argv=call.get("argv"))
        self.tool_jobs[tid] = job
        self.panel.attach(job)
        self.tabs.setCurrentIndex(0)
        if backup is not None:
            self.note.setText(f"Backed up {len(fixed)} file{'s' if len(fixed) > 1 else ''} to {_rel(backup)} before "
                              "overwriting (committed files can also be restored with git checkout).")
            self.note.show()
        else:
            self.note.hide()
        self._update_run_state()
        return job

    # dialogs (separate methods so that headless tests can replace them)
    def _confirm_missing(self, tool, missing):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Missing prerequisite files")
        box.setText(f"Some input files of “{tool['title']}” do not exist; the script will probably fail partway.")
        lines = ["• " + _plain(p["label"]) + (f" ({p['detail']})" if p["detail"] else "") for p in missing]
        prods = []
        for p in missing:
            if p.get("producer") and p["producer"] not in prods:
                prods.append(p["producer"])
        info = "Missing:\n" + "\n".join(lines)
        if prods:
            info += "\n\nHow to get them:\n" + "\n".join("• " + how for _, how in prods)
        box.setInformativeText(info)
        go = None
        if prods:
            go = box.addButton(f"Go to {producer_name(prods[0][0])}", QMessageBox.ButtonRole.ActionRole)
        anyway = box.addButton("Run anyway", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(go or cancel)
        box.exec()
        c = box.clickedButton()
        if go is not None and c is go:
            return "go:" + prods[0][0]
        if c is anyway:
            return "run"
        return None

    def _confirm_overwrite(self, tool, files):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Overwrite fixed-path outputs?")
        box.setText(f"“{tool['title']}” can only write to fixed paths; running it overwrites these files:")
        lines = [f"• {_rel(f)}" + (" (committed to git)" if committed(f) else "") for f in files]
        box.setInformativeText("\n".join(lines) + "\n\nBack them up first (copied into a new folder under "
                               "results/gui_runs/); committed files can also be restored later with git checkout.")
        b_backup = box.addButton("Back up and run", QMessageBox.ButtonRole.AcceptRole)
        b_over = box.addButton("Overwrite", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(b_backup)
        box.exec()
        c = box.clickedButton()
        return "backup" if c is b_backup else ("overwrite" if c is b_over else None)

    def _confirm_write(self, path: Path) -> bool:
        """Writing into the selected input folder: asks before overwriting, and before adding to a curated folder."""
        if path.exists():
            title = "Overwrite existing result?"
            text = (f"{_rel(path)} already exists" + (" (committed to git)" if committed(path) else "")
                    + "; re-demodulating will overwrite it. Continue?")
        else:
            title = "Write into a committed result folder?"
            text = (f"The script can only write its result into the selected run folder itself and will create "
                    f"{_rel(path)} ({_rel(path.parent)} is a committed result folder, not a new folder under "
                    "results/gui_runs/). Continue?")
        return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes

    def _backup(self, tid, files) -> Path:
        dest = paths.new_run_dir(f"tool_{tid}_backup")
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest / f.name)
        return dest

    # ------------------------------------------------------------------ jobs
    def _on_job_added(self, job):
        tid = tool_of(job)
        if not tid:
            return
        self.tool_jobs[tid] = job
        self._set_item_state(tid)
        if tid == self.current:
            self.panel.attach(job)
            self._update_run_state()

    def _on_job_changed(self, job):
        tid = tool_of(job)
        if tid:
            self._set_item_state(tid)
            if tid == self.current:
                self._update_run_state()

    def _on_job_finished(self, job):
        tid = tool_of(job)
        if not tid:
            return
        self._set_item_state(tid)
        self._refresh_extras(KeyPicker, RunDirPicker)  # input status / redemod_*.json already in the folder
        if tid != self.current:
            return
        self._update_run_state()
        self._refresh_info()
        ok = job.status == FINISHED
        self._load_history(select_job=job if ok else None, switch=ok)
        if not ok:
            self.tabs.setCurrentIndex(0)

    def _refresh_extras(self, *kinds):
        """Re-scan the pickers of the given classes (all pickers when none given)."""
        for _, _, extra in self.param_pages.values():
            if extra is not None and (not kinds or isinstance(extra, kinds)):
                extra.refresh()

    def _set_item_state(self, tid):
        if tid in self._items:
            self._items[tid][1].set_state(self._active_job(tid), ThemeManager.instance().tokens.accent)

    # ------------------------------------------------------------------ outputs
    def _history_entries(self, tid) -> list[dict]:
        tool = BY_ID[tid]
        entries, seen = [], set()
        for j in reversed(JobManager.instance().jobs):
            if tool_of(j) != tid or j.status != FINISHED:
                continue
            try:
                target = tool["target"](j.spec)
            except Exception:
                target = None
            first = target[0] if isinstance(target, (list, tuple)) and target else target
            if first is not None and not Path(first).exists():
                continue                        # output deleted since (e.g. a cleaned-up gui_runs folder)
            when = time.strftime("%m-%d %H:%M", time.localtime(j.finished or j.created))
            key = key_of_spec(j.spec) if tool.get("extra") in KEYSETS else None
            entries.append(dict(label=f"Job {when} · {j.title}", target=target, job=j, key=key))
            if target is not None:
                seen.add(self._tkey(target))
        try:
            hist = tool["history"]()
        except Exception:
            hist = []
        for h in hist:
            lab, target = h[0], h[1]
            if self._tkey(target) in seen:
                continue
            entries.append(dict(label=lab, target=target, job=None, key=h[2] if len(h) > 2 else None))
        return entries

    @staticmethod
    def _tkey(target):
        if isinstance(target, (list, tuple)):
            return tuple(str(Path(t).resolve()) for t in target)
        return str(Path(target).resolve())

    def _load_history(self, select_job=None, key=None, switch=False, keep=False):
        tid = self.current
        if not tid:
            return
        entries = self._history_entries(tid)
        prev = self._entry.get("label") if (keep and self._entry) else None
        if keep and [e["label"] for e in entries] == [e["label"] for e in self.entries]:
            self.entries = entries
            return
        self.entries = entries
        self.src.blockSignals(True)
        self.src.clear()
        for i, e in enumerate(entries):
            self.src.addItem(e["label"])
            self.src.setItemData(i, e["label"], Qt.ItemDataRole.ToolTipRole)
        if not entries:
            self.src.addItem("(no outputs yet)")
        self.src.setEnabled(bool(entries))
        idx = 0
        if select_job is not None:
            idx = next((i for i, e in enumerate(entries) if e["job"] is select_job), 0)
        elif key is not None and BY_ID[tid].get("extra") in KEYSETS:
            idx = next((i for i, e in enumerate(entries) if e["key"] == key), -1)
        elif prev is not None:
            idx = next((i for i, e in enumerate(entries) if e["label"] == prev), 0)
        elif BY_ID[tid].get("prefer"):
            idx = BY_ID[tid]["prefer"](self._ctx(tid), entries) or 0
        self.src.setCurrentIndex(idx if (idx >= 0 and entries) else (-1 if entries else 0))
        self.src.blockSignals(False)
        if entries and idx >= 0:
            self._display(entries[idx], switch=switch)
        else:
            self._show_empty(key)

    def _show_empty(self, key=None):
        self._entry = None
        if BY_ID[self.current].get("extra") in KEYSETS and key:
            msg = (f"Figure {key} has not been rendered yet (no matching file in results/slides/). "
                   "Click Run (top right) to render it.")
        else:
            msg = ("No outputs yet. Set the parameters and click Run (top right); the results appear here "
                   "automatically when the job finishes.")
        self._set_items([], empty=msg)
        self.btn_folder.setEnabled(False)

    def _select_key_entry(self, key):
        idx = next((i for i, e in enumerate(self.entries) if e["key"] == key), -1)
        self.src.blockSignals(True)
        self.src.setCurrentIndex(idx)
        self.src.blockSignals(False)
        if idx >= 0:
            self._display(self.entries[idx], switch=True)
        else:
            self._show_empty(key)

    def _on_src(self, i):
        if 0 <= i < len(self.entries):
            self._display(self.entries[i], switch=True)

    def _display(self, e: dict, switch=False):
        tool = BY_ID[self.current]
        log = e["job"].full_log() if e.get("job") is not None else ""
        try:
            items = tool["render"](e["target"], log)
        except Exception as ex:
            items = [("text", "Error", f"Cannot display this output: {type(ex).__name__}: {ex}", "error.txt")]
        self._entry = e
        cur = self.tabs.currentIndex()
        self._set_items(items)
        folder = self._folder_of(e)
        self.btn_folder.setEnabled(folder is not None and folder.exists())
        if switch and self.tabs.count() > 1:
            self.tabs.setCurrentIndex(1)
        elif not switch and cur < self.tabs.count():
            self.tabs.setCurrentIndex(cur)

    def _folder_of(self, e):
        t = e.get("target")
        if isinstance(t, (list, tuple)):
            t = t[0] if t else None
        if t is not None:
            t = Path(t)
            return t if t.is_dir() else t.parent
        j = e.get("job")
        if j is not None and j.out:
            return paths.absolute(j.out)
        return None

    def _open_folder(self):
        if self._entry is not None:
            f = self._folder_of(self._entry)
            if f is not None:
                reveal(f)

    def _set_items(self, items, empty=None):
        while self.tabs.count() > 1:
            w = self.tabs.widget(1)
            self.tabs.removeTab(1)
            if w not in self._plots:
                w.deleteLater()
        n_plot = 0
        for it in items:
            kind, title = it[0], it[1]
            if kind == "image":
                v = ImageView()
                v.load(it[2])
                self.tabs.addTab(v, title)
            elif kind == "plot":
                if n_plot >= len(self._plots):
                    self._plots.append(PlotWidget())
                pw = self._plots[n_plot]
                n_plot += 1
                pw.plot(it[2])
                self.tabs.addTab(pw, title)
            elif kind == "table":
                self.tabs.addTab(self._table_widget(it[2], it[3], it[4] if len(it) > 4 else ""), title)
            elif kind == "text":
                self.tabs.addTab(self._text_widget(it[2], it[3] if len(it) > 3 else "output.txt"), title)
        if not items:
            w = QWidget()
            l = QVBoxLayout(w)
            msg = label(empty or "No output files found.", "muted", wrap=True)
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            l.addStretch(1)
            l.addWidget(msg)
            l.addStretch(1)
            self.tabs.addTab(w, "Output")

    def _table_widget(self, headers, rows, note=""):
        t = ThemeManager.instance().tokens
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(10, 8, 10, 10)
        l.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(label(note or f"{len(rows)} row{'s' if len(rows) != 1 else ''}", "muted", wrap=True), 1)
        tw = QTableWidget(len(rows), len(headers))
        top.addWidget(button("Copy table", lambda: self._copy_table(headers, rows), "flat",
                             "Copy to the clipboard as tab-separated text"), 0,
                      Qt.AlignmentFlag.AlignTop)
        l.addLayout(top)
        tw.setHorizontalHeaderLabels([str(h) for h in headers])
        tw.verticalHeader().setVisible(False)
        tw.setAlternatingRowColors(True)
        tw.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        tw.setWordWrap(False)
        num = re.compile(r"^[\s+\-−]?[\d.]+(e[+\-]?\d+)?\s*%?$", re.I)
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                tip = None
                if isinstance(val, tuple):
                    val, tip = val
                it = QTableWidgetItem(str(val))
                if tip:
                    it.setToolTip(tip)
                s = str(val)
                if num.match(s):
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if s.startswith("✓"):
                    it.setForeground(QColor(t.success))
                elif s.startswith("✗"):
                    it.setForeground(QColor(t.danger))
                tw.setItem(r, c, it)
        tw.resizeColumnsToContents()
        for c in range(tw.columnCount()):
            tw.setColumnWidth(c, min(tw.columnWidth(c) + 12, 520))
        tw.horizontalHeader().setStretchLastSection(True)
        l.addWidget(tw, 1)
        return w

    def _copy_table(self, headers, rows):
        lines = ["\t".join(map(str, headers))]
        lines += ["\t".join(str(v[0] if isinstance(v, tuple) else v) for v in row) for row in rows]
        QGuiApplication.clipboard().setText("\n".join(lines))

    def _text_widget(self, text, fname):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(10, 8, 10, 10)
        l.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(label(f"{len(text.splitlines())} lines", "muted"), 1)
        top.addWidget(button("Copy", lambda: QGuiApplication.clipboard().setText(text), "flat"))
        top.addWidget(button("Save as text…", lambda: self._save_text(text, fname), "flat",
                             "Save this text output as a .txt file"))
        l.addLayout(top)
        lv = LogView()
        lv.set_text(text)
        lv.verticalScrollBar().setValue(0)
        l.addWidget(lv, 1)
        return w

    def _save_text(self, text, fname):
        stem, ext = Path(fname).stem, Path(fname).suffix or ".txt"
        folder = paths.GUI_RUNS if paths.GUI_RUNS.is_dir() else Path.home()
        default = folder / f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}{ext}"
        path, _ = QFileDialog.getSaveFileName(self, "Save text output", str(default), "Text (*.txt);;All files (*)")
        if path:
            try:
                Path(path).write_text(text + "\n", encoding="utf-8")
            except OSError as e:
                error_box(self, "Save failed", str(e))

    # ------------------------------------------------------------------ misc
    def _open_script(self):
        if self.current:
            open_external(paths.ROOT / BY_ID[self.current]["script"])

    def refresh_all(self):
        global _GIT_FILES
        _GIT_FILES = None
        self._refresh_extras()
        if self.current:
            self._refresh_info()
            self._load_history(keep=True)

    def on_show(self):
        if self.current is None:
            self.select_tool(TOOLS[0]["id"])
        else:
            self._refresh_info()
            self._load_history(keep=True)

    def open(self, tool=None, job_id=None, path=None, **_):
        if job_id:
            job = next((j for j in JobManager.instance().jobs if j.id == job_id), None)
            tid = tool_of(job)
            if tid:
                self.tool_jobs[tid] = job
                if tid == self.current:
                    self.panel.attach(job)
                    self._update_run_state()
                else:
                    self.select_tool(tid)
                if job.status == FINISHED:
                    self._load_history(select_job=job, switch=True)
                else:
                    self.tabs.setCurrentIndex(0)
                return
        if path:
            p = paths.absolute(path)
            if p.is_file() and p.suffix == ".npz" and not re.match(r"X[+-]", p.name):
                self.select_tool("show_solution")
                self._param_page("show_solution")[2].set(p)
                return
            if p.is_dir() and x_files(p):
                self.select_tool("redemodulate")
                self._param_page("redemodulate")[2].set(p)
                return
        if tool in BY_ID:
            self.select_tool(tool)
