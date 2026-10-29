"""分析与作图: catalogue of the remaining analysis, figure and maintenance scripts.

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
GROUPS = ("分析", "作图", "维护")
PAGE_ZH = {"tip": "针尖位置链", "pulse": "单脉冲发射链", "steady": "稳态求解", "fdtd": "泵浦 FDTD", "sweep": "参数扫描"}


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
    "tip": ("tip", "在『针尖位置链』页面运行 simulate，输出到 results/tip_positions/P2mW_sym（约 20–50 分钟；"
                   "这些 npz 被 gitignore，仓库中没有）"),
    "pulse": ("pulse", "在『单脉冲发射链』页面运行对应的链（输出放到 results/pulse_chain/<运行名>/）"),
    "steady": ("steady", "在『稳态求解』页面运行生产网格，输出文件夹设为 results/baseline（约 6–7 分钟）"),
    "fdtd": ("fdtd", "在『泵浦 FDTD』页面运行，输出到 results/fdtd_pump"),
    "fbc": ("tool:field_bending_chain", "运行本页的『场线弯曲对近场信号的贡献』，再把 summary.json 放到 "
                                        "results/field_bending_chain/"),
    "power": ("tool:power_dependence", "运行本页的『信号随泵浦功率的变化』（X = +5 与 −5），再把 JSON 放到 "
                                       "results/pulse_chain/power/"),
}


def req(lab: str, files, producer=None, optional=False, hard=False) -> dict:
    """A file prerequisite: ok when every file exists."""
    files = [Path(f) for f in files]
    have = [f.exists() for f in files]
    n = sum(have)
    ok = n == len(files)
    if len(files) == 1:
        detail = "" if ok else "不存在"
    else:
        detail = f"{n}/{len(files)} 个文件"
        if not ok:
            detail += f"，缺 {files[have.index(False)].name} 等"
    return dict(label=lab, ok=ok, detail=detail, producer=PRODUCERS.get(producer), optional=optional, hard=hard)


def cond(lab: str, ok: bool, detail="", producer=None, optional=False, hard=True, sel=False) -> dict:
    """A computed prerequisite; sel=True marks "the user still has to pick an input" rather than a missing file."""
    return dict(label=lab, ok=bool(ok), detail=detail, producer=PRODUCERS.get(producer), optional=optional, hard=hard,
                sel=sel)


def measured_req():
    return req(_code("data/measured_tip_scan.csv") + "（实测 X 扫描，图中叠加）", [paths.MEASURED_CSV])


def producer_name(target: str) -> str:
    """Display name of a PRODUCERS target: a page id or "tool:<id>"."""
    if target.startswith("tool:"):
        return BY_ID[target[5:]]["title"]
    return PAGE_ZH.get(target, target)


def curated(folder) -> bool:
    """A hand-curated folder under results/ (not a results/gui_runs/ run folder)."""
    d = Path(folder).resolve()
    res, gui = paths.RESULTS.resolve(), Path(paths.GUI_RUNS).resolve()
    return (d == res or res in d.parents) and d != gui and gui not in d.parents


def new_dir_line(tid: str) -> str:
    return "新文件夹 " + _code(f"results/gui_runs/tool_{tid}_<时间>/") + "，其中："


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
        items.append(md_table("汇总表", (d / "summary.md").read_text(encoding="utf-8")))
    if log:
        items.append(("text", "日志", clean_log(log), "field_bending_log.txt"))
    return items


def dir_history(committed_dir: Path | None, prefix: str, marker: str):
    out = [(f"{_rel(d)}  ·  {mtime_str(d / marker)}", d) for d in gui_dirs(prefix) if (d / marker).exists()]
    if committed_dir is not None and (committed_dir / marker).exists():
        out.append((f"{_rel(committed_dir)}（仓库）  ·  {mtime_str(committed_dir / marker)}", committed_dir))
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
    items = [("plot", "a3：bent vs horizontal", draw),
             ("table", "数值表", ["X (µm)", "a3 bent", "a3 horizontal", "a3 比值", "a1 比值", "a2 比值", "a4 比值",
                                  "far 比值"], rows, "带限幅值（0.3–2 THz，d²S/dt²），比值 = bent / horizontal。")]
    tf = d / "tilt.json"
    if tf.exists():
        t = _vals(json.loads(tf.read_text()))
        zs = list(next(iter(t.values())).keys()) if t else []
        trows = [[f"{x:g}"] + [f"{t[x][z]['tilt_deg']:.3g}" for z in zs] for x in sorted(t)]
        items.append(("table", "外加场倾角", ["x (µm)"] + [f"{z} 倾角 (°)" for z in zs], trows,
                      "外加场在不同深度的倾角 arctan(E_z / E_x)（来自 X = 0 的 bent 运行的 Poisson 解）。"))
    if log:
        items.append(("text", "日志", clean_log(log), "field_bending_chain_log.txt"))
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
    note = "图中每个指标按所有文件中的最大值归一化（保留 X = +5 与 −5 的相对幅度）。"
    if old:
        note += "旧版 JSON：指标是 dS₂/dt 与 dJ_x/dt 的 RMS（不是 0.3–2 THz 带限幅值），不能与新版数值直接比较。"
    items = [("plot", "功率依赖", draw),
             ("table", "数值表", ["文件", "P (mW)", "N_surface (cm⁻³)", "pairs / cm"] + [k for k, _ in metrics],
              rows, note)]
    if log:
        items.append(("text", "日志", clean_log(log), "power_dependence_log.txt"))
    return items


def power_history():
    out = []
    for d in gui_dirs("tool_power_dependence"):
        js = sorted(d.glob("power_dependence_*.json"))
        if js:
            out.append((f"{_rel(d)}  ·  {mtime_str(js[0])}", d))
    if list((PC / "power").glob("power_dependence_*.json")):
        out.append(("results/pulse_chain/power（仓库，X = ±5）", PC / "power"))
    for f in sorted(PC.glob("power_dependence_*.json")):
        out.append((f"{_rel(f)}（仓库，旧版）", f))
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
        return [("text", "结果", "JSON 为空：运行文件夹中没有 X*.npz。", "redemod.txt")]
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
        trows.append([k] + [(f"{x / m:.2f}", f"原始值 {x:.4g}") for x in v])
    items = [("plot", "重解调 X 扫描", draw),
             ("table", "归一化表", ["指标"] + [f"X = {x:+.1f}" for x in Xs], trows,
              f"{_rel(f)} — 每行按自身最大值归一化（与脚本打印的表相同）；悬停查看原始值。")]
    if log:
        items.append(("text", "日志", clean_log(log), "redemodulate_log.txt"))
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
    return [(f"{_rel(f)}{'（仓库）' if committed(f) else ''}  ·  {mtime_str(f)}", f) for f in files]


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
        items.append(md_table("数值表", md.read_text(encoding="utf-8")))
    if log:
        items.append(("text", "日志", clean_log(log), "compare_log.txt"))
    return items


def compare_history():
    out = []
    for d in gui_dirs("tool_compare_scenarios"):
        if (d / "compare.png").exists():
            out.append((f"{_rel(d)}  ·  {mtime_str(d / 'compare.png')}", d / "compare.png"))
    for png in sorted((R / "coarse_scenarios").glob("*.png")):
        if png.with_suffix(".md").exists():
            out.append((f"{_rel(png)}（仓库）", png))
    return out


# ============================================================================ text-only tools
def text_render(fname):
    def render(target, log):
        txt = clean_log(log)
        return [("text", "文本输出", txt or "（没有输出）", fname)]
    return render


def parse_waveform(text: str):
    rx = re.compile(r"^(.*?)\s+(-?\d+)\s+(-?\d+\.\d+)\s+(-?\d+)\s+(\{.*?\})\s+/\s+(\{.*?\})\s*$")

    def shift(s):
        try:
            d = ast.literal_eval(s)
            return f"残差 {d['residual']:.2f} · 平移 {d['shift_ps']:+.3f} ps"
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
        items.append(("table", "相位拟合表", ["成对比较（ref vs test）", "φ0 (°)", "τ (ps)", "残差 (°)", "同号最佳平移",
                                              "反号最佳平移"], rows,
                      "φ0：0° 同形，180° 反相，±90° 导数关系；τ 为延迟；残差为 0.25–1.6 THz 内相位拟合的 rms。"))
    items.append(("text", "文本输出", txt or "（没有输出）", "waveform_phase_table.txt"))
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
        items.append(("plot", "X 扫描", draw))
        items.append(("table", "数值表", ["T (K)", "D_n (cm²/s)", "V (V)"] + [f"X = {x:+.1f}" for x in Xs], rows,
                      "每个温度内按两种偏压的最大值归一化（旧的一阶导数 RMS 指标）。"))
    items.append(("text", "文本输出", txt or "（没有输出）", "dember_interference.txt"))
    return items


NUMERIC_TESTS = (("poisson", "裸 Poisson 场 vs 解析解（误差 < 12 %，E_z/E_x < 5 %）"),
                 ("equilibrium", "V = 0、G = 0 的平衡态残差为零"),
                 ("jacobian", "耦合 Jacobian 有限差分校验（相对误差 < 1e-5）"),
                 ("dark", "10 V 暗态：收敛、|I| < 1e-3 mA、EL2⁺ 合理"))


def numerics_render(target, log):
    txt = clean_log(log)
    lines = {l.strip() for l in txt.splitlines()}
    failed = "Traceback" in txt or "@@FAILED" in (log or "")
    rows, first_bad = [], True
    for name, desc in NUMERIC_TESTS:
        if f"{name} ok" in lines:
            rows.append([name, desc, "✓ 通过"])
        elif failed and first_bad:
            rows.append([name, desc, "✗ 失败"])
            first_bad = False
        else:
            rows.append([name, desc, "— 未运行"])
    n_ok = sum(r[2].startswith("✓") for r in rows)
    return [("table", "测试结果", ["测试", "检查内容", "结果"], rows, f"{n_ok}/{len(rows)} 项通过"),
            ("text", "文本输出", txt or "（没有输出）", "test_numerics.txt")]


# ============================================================================ fixed-output figure tools
def png_render(target, log):
    files = target if isinstance(target, (list, tuple)) else [target]
    items = [("image", Path(f).name, Path(f)) for f in files if Path(f).suffix == ".png"]
    if log and clean_log(log):
        items.append(("text", "日志", clean_log(log), "log.txt"))
    return items


def fixed_history(files, note="当前版本"):
    out = []
    for f in files:
        if Path(f).exists():
            tag = "，仓库" if committed(f) else ""
            out.append((f"{_rel(f)}（{note}{tag}）  ·  {mtime_str(f)}", Path(f)))
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
    return [cond(_code("results/pulse_chain/") + " 下 9 个扩散模型对比运行（缺失的自动跳过）", have == len(DEMBER_FOLDERS),
                 f"{have}/{len(DEMBER_FOLDERS)} 个文件夹齐全", "pulse", optional=True, hard=False)]


SLIDE_KEYS = [
    dict(key="steady", zh="稳态结果总览", files=("sim_steady_state.png",), runtime="约 1.5 s",
         reqs=lambda: [req(_code(f"results/baseline/{f}"), [R / "baseline" / f], "steady")
                       for f in ("profiles_depth0.5um.csv", "tip_scan.csv")]),
    dict(key="positions", zh="针尖位置链（路线 2）", files=("sim_positions.png",), runtime="< 1 s",
         reqs=_slide_reqs_positions),
    dict(key="bending", zh="场线弯曲贡献", files=("sim_field_bending.png",), runtime="约 1.5 s（首次最多约 15 s）",
         reqs=lambda: [req(_code("results/field_bending_chain/summary.json"), [FBC / "summary.json"], "fbc")]),
    dict(key="pump", zh="泵浦 FDTD", files=("sim_pump_fdtd.png",), runtime="约 2 s",
         reqs=lambda: [req(_code("results/fdtd_pump/{tip, notip, ratio}.npz"),
                           [R / "fdtd_pump" / f"{n}.npz" for n in ("tip", "notip", "ratio")], "fdtd")]),
    dict(key="dynamics", zh="载流子动力学", files=("sim_dynamics.png",), runtime="约 3–4 s（含 0.6 ps PulseSim）",
         reqs=lambda: [req(_code("results/pulse_chain/P2mW_notipmap_degen/X+0.0.npz"),
                           pc_npz("P2mW_notipmap_degen", (0.0,)), "pulse")]),
    dict(key="nearfield", zh="近场与解调", files=("sim_nearfield_demod.png",), runtime="约 1 s",
         reqs=lambda: [req(_code("results/pulse_chain/P2mW_notipmap_degen/X{±5, ±2.5, 0}.npz"),
                           pc_npz("P2mW_notipmap_degen", X5), "pulse")]),
    dict(key="xscan", zh="X 扫描对比", files=("sim_xscan.png", "xscan_A180.json"), runtime="约 1 s",
         reqs=lambda: [req(_code("results/pulse_chain/P2mW_notipmap_degen/X*.npz（7 个 X）"),
                           pc_npz("P2mW_notipmap_degen", X7), "pulse"),
                       req(_code("results/pulse_chain/P30mW_notipmap_degen/X*.npz"), pc_npz("P30mW_notipmap_degen", X7),
                           "pulse", optional=True),
                       req(_code("results/tip_positions/*/demod_A180_h10.json（4 个路线 2 运行）"),
                           [TP / f / "demod_A180_h10.json" for f in ("P2mW_sym", "P2mW_tipmap", "P30mW_sym",
                                                                      "P2mW_sym_Vtip12")], "tip", optional=True)]),
    dict(key="dember", zh="扩散模型 / Dember", files=("sim_dember.png",), runtime="约 2 s", reqs=_dember_req),
    dict(key="power", zh="功率依赖", files=("sim_power.png",), runtime="约 1 s",
         reqs=lambda: [req(_code("results/pulse_chain/power/power_dependence_X±5.0_nsat2e+19.json"),
                           [PC / "power" / f"power_dependence_X{x}_nsat2e+19.json" for x in ("+5.0", "-5.0")],
                           "power")]),
    dict(key="dipole", zh="偶极几何（英文）", files=("sim_dipole_geometry.png",), runtime="约 2 s", reqs=lambda: []),
]
CARTOON_KEYS = [
    dict(key="distortions", zh="三种场畸变", files=("cartoon_distortions.png",), runtime="< 1 s", reqs=lambda: []),
    dict(key="chain", zh="三段链路示意", files=("cartoon_chain.png",), runtime="< 1 s", reqs=lambda: []),
    dict(key="oddeven", zh="X 扫描的奇偶分解", files=("cartoon_oddeven.png",), runtime="< 1 s", reqs=lambda: []),
    dict(key="tippot", zh="针尖直流电势", files=("cartoon_tippot.png",), runtime="< 1 s", reqs=lambda: []),
    dict(key="power", zh="光斑随功率展宽", files=("cartoon_power.png",), runtime="< 1 s", reqs=lambda: []),
    dict(key="all", zh="全部 5 张", files=("cartoon_distortions.png", "cartoon_chain.png", "cartoon_oddeven.png",
                                          "cartoon_tippot.png", "cartoon_power.png"), runtime="约 1.5 s", reqs=lambda: []),
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
                    out.append((f"all · 全部 {len(files)} 张（当前版本）", files, "all"))
            elif files[0].exists():
                tag = "，仓库" if committed(files[0]) else ""
                out.append((f"{k['key']} · {files[0].name}（当前版本{tag}）  ·  {mtime_str(files[0])}", files, k["key"]))
        return out
    return hist


# ============================================================================ registry
def _v(ctx, k, default=None):
    return ctx.get("v", {}).get(k, default)


def _rundir_inputs(ctx):
    d = ctx.get("rundir")
    if not d:
        return ["（请在右侧选择一个单脉冲链运行文件夹）"]
    return [_code(f"{_rel(d)}/X*.npz") + " — run_pulse_chain 的输出，需含 t、heights_nm、Ez_tip、n_center"]


def _rundir_prereq(ctx):
    d = ctx.get("rundir")
    if not d:
        return [cond("选择一个运行文件夹", False, "尚未选择", sel=True)]
    xs = x_files(d)
    out = [cond(_code(f"{_rel(d)}/X*.npz"), bool(xs), f"{len(xs)} 个 X" if xs else "文件夹中没有 X*.npz", "pulse")]
    if xs:
        try:
            with np.load(xs[0]) as z:
                miss = [k for k in ("t", "heights_nm", "Ez_tip", "n_center") if k not in z.files]
        except Exception as e:      # unreadable / truncated npz
            miss = [f"无法读取 {xs[0].name}: {type(e).__name__}"]
        out.append(cond("npz 含 t / heights_nm / Ez_tip / n_center", not miss, "缺 " + ", ".join(miss) if miss else ""))
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
    return f"约 {max(1, round(n * 22 / 60))}–{max(1, round(n * 55 / 60))} 分钟（{n} 个功率 × 22–55 s）"


def _power_outputs(ctx):
    X, ns = float(_v(ctx, "X", 5.0)), float(_v(ctx, "n_sat", 2e19))
    return [new_dir_line("power_dependence"),
            sub(_code(f"power_dependence_X{X:+.1f}_nsat{ns:.0e}.json") +
                " — 每个功率一行：P_mW、N_surface、pairs_per_cm、det2_band、det3_band、far_band"),
            sub(_code(f"power_X{X:+.1f}_nsat{ns:.0e}_P*.npz") + " — 每个功率的波形"),
            "幻灯片图 sim_power.png 读取的是固定的 results/pulse_chain/power/ 下 X = ±5 的 JSON，不受这里影响。"]


def _power_prereq(ctx):
    P = _v(ctx, "powers", ()) or ()
    return [cond("泵浦功率列表非空且全部 > 0", len(P) > 0 and all(p > 0 for p in P), f"{len(P)} 个功率")]


def _compare_inputs(ctx):
    sols = ctx.get("solutions") or []
    out = [f"已选 {len(sols)} 个稳态解（右侧表格勾选）"]
    out += [_code(_rel(p)) + f" → “{_esc(l)}”" for l, p in sols[:6]]
    if len(sols) > 6:
        out.append(f"… 另有 {len(sols) - 6} 个")
    return out + [_code("data/measured_tip_scan.csv") + " — 实测 +25 V 点"]


def _compare_prereq(ctx):
    sols = ctx.get("solutions") or []
    gone = [p for _, p in sols if not Path(p).exists()]
    out = [cond("至少勾选 1 个稳态解", bool(sols), f"已选 {len(sols)} 个", sel=True)]
    if gone:
        out.append(cond("所选的解文件都存在", False, f"已不存在：{', '.join(Path(p).name for p in gone[:3])}"))
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
    _tool(id="field_bending", group="分析", title="外加场弯曲（裸场静电）", short="5 种晶片/背面边界的场线倾斜", tag="≈1 s",
         desc="只计算外加偏压的静电场（无载流子、无 EL2）：对 5 种晶片厚度 / 背面边界（30 µm 浮空参考；350 µm 与 "
              "500 µm，背面浮空或接地）各做一次线性 Poisson 求解，给出场线随深度的倾角、竖直/水平极化比 ∫P_z/∫P_x "
              "以及针尖 E_z 随 X 的扫描。情形与宽网格（CASES / BOX）写死在脚本中，没有可调参数。"
              "注意：表中 tip peak X 只是 ±1.5 µm 两个反对称峰之间的平局判定，没有物理意义。",
         script="scripts/field_bending.py", func="main", out=True,
         runtime="约 1–2 秒（5 个网格，各一次线性 Poisson 求解 + 3 次针尖扫描求解）",
         inputs=lambda c: [_code("data/measured_tip_scan.csv") + " — 实测 +25 V 针尖扫描（图中叠加）"],
         outputs=lambda c: [new_dir_line("field_bending"),
                            sub(_code("field_bending.png") + " — 2×2：深度方向倾角、表面 E_z、带符号与绝对值的针尖 E_z"),
                            sub(_code("summary.json") + " / " + _code("summary.md") + " — 5 种情形的汇总表")],
         prereq=lambda c: [measured_req()],
         build=lambda c, out: (dict(kwargs=dict(out=_rel(out))), ""),
         target=out_target, render=fb_render,
         history=lambda: dir_history(R / "field_bending", "tool_field_bending", "summary.json")),

    _tool(id="field_bending_chain", group="分析", title="场线弯曲对近场信号的贡献", short="单脉冲链：完整外加场 vs 去掉 E_z",
         tag="5–10 min",
         desc="路线 1（无针尖近场 E_z(h,t) + Pizzuto 有限偶极模型解调，简并扩散，t_end = 4 ps）：7 个光斑位置 "
              "X = −7.5…+7.5 µm 各运行两次——完整外加场（E_x + E_z，bent）与去掉竖直分量（horizontal），"
              "比较 1–4 次谐波与远场在 0.3–2 THz 的带限幅值。X = 0 的 bent 运行同时写出外加场倾角 tilt.json。"
              "X 列表、偏压（25 V）、温度与解调设置固定在脚本中。",
         script="scripts/field_bending_chain.py", func="main", out=True,
         runtime="约 5–10 分钟（14 次 PulseSim，每次约 17–40 s）",
         specs=lambda: [FieldSpec("P_avg_mW", 2.0, "float", "平均泵浦功率 P_avg", "mW",
                                  "泵浦平均功率。其余设置（X 列表、V_bias、T、解调）固定在脚本中。", "参数", minimum=1e-4)],
         inputs=lambda c: ["无（纯模拟，不依赖其它结果）"],
         outputs=lambda c: [new_dir_line("field_bending_chain"),
                            sub(_code("summary.json") + " — bent / horizontal 的 a1…a4、far，以及 a3 比值"),
                            sub(_code("tilt.json") + "、" + _code("applied_field.npz") + " — 外加场倾角与场分布"),
                            sub(_code("bent_X*.npz") + "、" + _code("horizontal_X*.npz") + " — 每次运行的波形"),
                            "幻灯片图 sim_field_bending.png 只读取固定的 results/field_bending_chain/summary.json，"
                            "不会用到这里的新结果。"],
         progress=lambda c: {"type": "count", "pattern": r"^(bent|horizontal)\s+X=", "total": 14},
         build=lambda c, out: (dict(kwargs=dict(P_avg_mW=_v(c, "P_avg_mW", 2.0), out=_rel(out))),
                               f"P = {_v(c, 'P_avg_mW', 2.0):g} mW"),
         target=out_target, render=fbc_render,
         history=lambda: dir_history(FBC, "tool_field_bending_chain", "summary.json")),

    _tool(id="power_dependence", group="分析", title="信号随泵浦功率的变化", short="固定 X 扫描泵浦功率（带限幅值）",
         tag="4–9 min",
         desc="在固定光斑偏移 X 下对一组泵浦功率运行单脉冲链（路线 1：无针尖近场 + FDM，对称高斯光斑，无 FDTD 针尖图，"
              "t_end = 4 ps），记录探测波形 d²S/dt² 在 0.3–2 THz 的带限谱幅（2、3 次谐波）、由 J_x 得到的远场参考、"
              "表面密度 N_surface 与片载流子对数。V_bias = 25 V、T = 300 K、w = 3 µm、A = 180 nm 固定。"
              "JSON 在全部功率算完后才写出，中途停止只留下各功率的 npz。",
         script="scripts/power_dependence.py", func="main", out=True, runtime=_power_runtime,
         specs=lambda: [FieldSpec("X", 5.0, "float", "光斑位置 X", "µm", "光斑相对针尖的位置；X > 0：针尖在光斑的阳极侧",
                                  "扫描参数"),
                        FieldSpec("n_sat", 2e19, "float", "饱和密度 n_sat", "cm⁻³", "单脉冲载流子密度的软饱和上限",
                                  "扫描参数", minimum=1e12),
                        FieldSpec("powers", (0.25, 0.5, 1, 2, 4, 8, 16, 30, 60, 120), "tuple", "泵浦功率列表", "mW",
                                  "逗号分隔；每个功率一次 PulseSim（22–55 s）", "扫描参数"),
                        FieldSpec("degenerate", True, "bool", "简并扩散（Fermi-Dirac）", "",
                                  "广义 Einstein 关系；与 run_pulse_chain 的默认值（关）不同", "扫描参数")],
         inputs=lambda c: ["无（纯模拟，不依赖其它结果）"], outputs=_power_outputs, prereq=_power_prereq,
         progress=lambda c: {"type": "count", "pattern": r"^P=", "total": max(1, len(_v(c, "powers", ()) or ()))},
         build=lambda c, out: (dict(kwargs=dict(X=float(_v(c, "X", 5.0)), n_sat=float(_v(c, "n_sat", 2e19)),
                                                powers=list(_v(c, "powers", ())), out=_rel(out),
                                                degenerate=1.0 if _v(c, "degenerate", True) else 0.0)),
                               f"X = {float(_v(c, 'X', 5.0)):+g} µm · {len(_v(c, 'powers', ()))} 个功率"),
         target=out_target, render=power_render, history=power_history),

    _tool(id="redemodulate", group="分析", title="重新解调脉冲链结果", short="用新的 A / h_min 重解调已有运行", tag="数秒",
         desc="纯后处理，不重新模拟：对一次已保存的单脉冲链运行（文件夹中的 X*.npz）用新的敲击振幅 A、最小针尖高度 h_min "
              "或 LTEM (1−β) 因子重新做 Pizzuto 有限偶极解调（谐波 1–4），计算每个 X 的低通 dS_n/dt 与 S_n 的 RMS，"
              "下方以归一化曲线和表格显示。注意：脚本文档写的默认 A = 200 nm，实际默认是 180 nm。",
         script="scripts/redemodulate.py", func="main", out=False, extra="rundir",
         runtime="数秒（每个 X 约 0.2–0.5 s）",
         specs=lambda: [FieldSpec("A_nm", 180.0, "float", "敲击振幅 A", "nm", "针尖敲击振幅", "解调参数", minimum=1e-3),
                        FieldSpec("h_min_nm", 0.0, "float", "最小针尖高度 h_min", "nm", "敲击周期中针尖的最低高度",
                                  "解调参数", minimum=0.0),
                        FieldSpec("ltem_factor", False, "bool", "LTEM (1−β) 因子", "",
                                  "按 Pizzuto 的 LTEM FDM 把 α 乘以 (1 − β)", "解调参数")],
         inputs=_rundir_inputs,
         writes_input=_rundir_writes, prereq=_rundir_prereq,
         build=lambda c, out: (dict(kwargs=dict(run_dir=_rel(c["rundir"]), A_nm=float(_v(c, "A_nm", 180.0)),
                                                h_min_nm=float(_v(c, "h_min_nm", 0.0)),
                                                ltem_factor=bool(_v(c, "ltem_factor", False)))),
                               f"{Path(c['rundir']).name} · A = {float(_v(c, 'A_nm', 180.0)):g} nm, "
                               f"h_min = {float(_v(c, 'h_min_nm', 0.0)):g} nm" + (" · LTEM" if _v(c, "ltem_factor") else "")),
         target=redemod_target, render=redemod_render, history=redemod_history, prefer=redemod_prefer),

    _tool(id="compare_scenarios", group="分析", title="对比多个稳态解", short="叠加多个解的场、载流子与针尖扫描", tag="≈1 s",
         desc="把若干已保存的稳态解（solution.npz）叠加到一张 2×2 图：0.5 µm 深处的 E_x、n 与 EL2⁺、发射权重 "
              "n_pulse·|E|、归一化 |针尖信号| vs X（叠加实测 +25 V），并写出各情形关键数值（P、τ_srh、S、I、n(0)、"
              "E_x(±5)、峰值 X）的 Markdown 表。坐标范围固定（E_x −1…6 kV/cm，密度 1e10…1e18 cm⁻³），极端情形"
              "可能被裁剪；实测点只对 25 V 偏压有意义。",
         script="scripts/compare_scenarios.py", func="main", out=True, extra="solutions",
         runtime="每个解约 0.3 s（粗网格）；生产网格每个约 1–2 s",
         inputs=_compare_inputs,
         outputs=lambda c: [new_dir_line("compare_scenarios"),
                            sub(_code("compare.png") + " — 2×2 对比图"),
                            sub(_code("compare.md") + " — 各情形的数值表")],
         prereq=_compare_prereq,
         build=lambda c, out: (dict(kwargs=dict(out=_rel(Path(out) / "compare.png"),
                                                items=[[l, _rel(p)] for l, p in c["solutions"]])),
                               f"{len(c['solutions'])} 个解"),
         target=compare_target, render=compare_render, history=compare_history),

    _tool(id="show_solution", group="分析", title="稳态解文本报告", short="打印解的电流、剖面与针尖扫描", tag="<1 s",
         desc="载入一个已保存的稳态解（.npz），以文本形式打印：偏压与电极电势、功率、光电流与增益、各阶段收敛情况、"
              "4 个深度（0.2 / 0.5 / 1 / 2 µm，取最近的网格中心，粗网格上会变成 0.12 / 0.62 µm 等）的横向剖面表，"
              "以及针尖扫描表和远场偶极。不写任何文件。交互式查看同一个解请用『解查看器』。",
         script="scripts/show_solution.py", func="main", out=False, extra="solution", runtime="< 1 秒",
         inputs=lambda c: [_code(_rel(c["solution"])) if c.get("solution") else "（请在右侧选择一个 .npz）"],
         outputs=lambda c: ["仅文本输出（显示在下方，可另存为 .txt）"],
         prereq=lambda c: [cond("选择一个存在的 .npz 稳态解", bool(c.get("solution")) and Path(c["solution"]).exists(),
                                "" if c.get("solution") else "尚未选择", sel=not c.get("solution"))],
         build=lambda c, out: (dict(args=[_rel(c["solution"])]), Path(c["solution"]).name),
         render=text_render("show_solution.txt")),

    _tool(id="band_check", group="分析", title="探测带宽检验", short="弱侧发射是否落在探测带外", tag="≈1 s",
         desc="检查弱侧（X = −5 µm）的发射是否只是落在探测带宽之外：对路线 1（无针尖场 + FDM，P2mW_notipmap_degen）"
              "与路线 2（针尖作为等势体参与 Poisson，P2mW_sym），在 X = +5、−5、0 计算 d²S₃/dt² 的频谱、6 个频带的"
              "带限幅值、10–90 % 上升时间以及各频带的 −5/+5 比值，并绘制归一化频谱（阴影为 0.3–2 THz 探测带）。"
              "图在两条路线都算完后才保存。",
         script="scripts/band_check.py", func=None, out=False, runtime="约 1–3 秒",
         inputs=lambda c: [_code("results/pulse_chain/P2mW_notipmap_degen/X{+5, −5, 0}.npz") + " — 路线 1",
                           _code("results/tip_positions/P2mW_sym/") + " meta.json + X{+5, −5, 0}_h*.npz — 路线 2"],
         outputs=lambda c: ["终端表格（显示在下方）"],
         fixed=lambda c: [SLIDES / "sim_band_check.png"],
         prereq=lambda c: [req(_code("results/pulse_chain/P2mW_notipmap_degen/X{+5, −5, 0}.npz"),
                               pc_npz("P2mW_notipmap_degen", (5.0, -5.0, 0.0)), "pulse"),
                           req(_code("results/tip_positions/P2mW_sym/meta.json"), [TP / "P2mW_sym" / "meta.json"], "tip"),
                           req(_code("results/tip_positions/P2mW_sym/X{+5, −5, 0}_h*.npz"),
                               tp_npz("P2mW_sym", (5.0, -5.0, 0.0)), "tip")],
         target=lambda spec: SLIDES / "sim_band_check.png", render=png_render,
         history=lambda: fixed_history([SLIDES / "sim_band_check.png"])),

    _tool(id="waveform_phase", group="分析", title="波形相位比较", short="FFT 相位差拟合 φ0 + 2πfτ", tag="数秒",
         desc="按实测 ±25 V 分析的同样方式比较两条模拟探测波形（d²S₃/dt²，2.2 THz 低通）：在 0.25–1.6 THz 内拟合相位差 "
              "φ(test) − φ(ref) = φ0 + 2πfτ，给出 φ0（0° 同形，180° 反相，±90° 导数关系）、延迟 τ 与相位残差，并做"
              "同号/反号模板的时域最佳平移检查。成对比较来自路线 1（pulse_chain/P2mW_notipmap_degen）与路线 2"
              "（tip_positions/P2mW_sym）；若存在针尖电势运行（P2mW_sym_Vtip12 / Vtipm12）再加 3 行。"
              "表格在全部算完后一次性打印。",
         script="scripts/waveform_phase.py", func=None, out=False, runtime="数秒",
         inputs=lambda c: [_code("results/pulse_chain/P2mW_notipmap_degen/X{±5, +2.5, 0, −2.5}.npz") + " — 路线 1",
                           _code("results/tip_positions/P2mW_sym/") + " meta.json + X*_h*.npz — 路线 2",
                           _code("results/tip_positions/P2mW_sym_Vtip12, P2mW_sym_Vtipm12/") + " — 可选"],
         outputs=lambda c: ["仅标准输出：相位拟合表（显示在下方，可另存为文本；仓库中的 "
                            + _code("results/tip_positions/waveform_phase_table.txt") + " 就是这份输出）"],
         prereq=lambda c: [req(_code("results/pulse_chain/P2mW_notipmap_degen/X{±5, +2.5, 0, −2.5}.npz"),
                               pc_npz("P2mW_notipmap_degen", X5), "pulse"),
                           req(_code("results/tip_positions/P2mW_sym/meta.json"), [TP / "P2mW_sym" / "meta.json"], "tip"),
                           req(_code("results/tip_positions/P2mW_sym/X{±5, +2.5, 0, −2.5}_h*.npz"),
                               tp_npz("P2mW_sym", X5), "tip"),
                           req(_code("results/tip_positions/P2mW_sym_Vtipm12/X+5.0_h370.npz") + "（针尖电势附加行）",
                               [TP / "P2mW_sym_Vtipm12" / "X+5.0_h370.npz"], "tip", optional=True)],
         render=waveform_render,
         history=lambda: fixed_history([TP / "waveform_phase_table.txt"], "仓库中保存的输出")),

    _tool(id="dember_interference", group="分析", title="Dember 项干涉检验", short="横向 Dember 项与漂移前沿的干涉",
         tag="5–10 min",
         desc="检验单侧性是否来自横向光致 Dember 项（对 X 偶、与偏压无关）与漂移前沿（对 X 奇、∝ V）的干涉：在 "
              "V = 25 V 与 0 V、冷（300 K）与热（1500 K）载流子下扫描 X = −5…+5 µm（经典扩散、无针尖、2 mW、"
              "t_end = 3 ps，共 20 次 PulseSim），输出低通 dS₃/dt 的 RMS（每个温度内归一化）。注意：这里用的是旧的"
              "一阶导数 RMS 指标，数值不能与其它表格（d²S/dt² 带限幅值）直接比较。脚本没有可调参数，也不写文件。",
         script="scripts/dember_interference.py", func=None, out=False,
         runtime="约 5–10 分钟（20 次 PulseSim）；两个温度块之间约 3–5 分钟没有输出，属正常",
         inputs=lambda c: ["无（纯模拟）"],
         outputs=lambda c: ["仅标准输出：每个温度一块（下方解析为曲线与表格，可另存为文本）"],
         progress=lambda c: {"type": "count", "pattern": r"^\s+V =", "total": 4},
         render=dember_render),

    _tool(id="fig_slides", group="作图", title="幻灯片图（模拟结果）", short="渲染英文幻灯片图 sim_*.png", tag="1–4 s",
         desc="从磁盘上已保存的结果渲染英文幻灯片图（results/slides/sim_*.png，约 2550 px 宽）。每次只渲染一张——脚本的 "
              "all 模式会在第一张缺少输入的图处中止，所以这里按图单独运行。dynamics 会在进程内跑一个 0.6 ps 的 "
              "PulseSim，bending 会做一次裸场 Poisson 求解。右侧列表的“输入”列实时显示每张图的前置文件是否齐全。",
         script="scripts/fig_slides.py", func=None, out=False, extra="slide_keys",
         inputs=lambda c: [f"所选的图：{_code(c.get('key'))} — {keyinfo(SLIDE_KEYS, c.get('key'))['zh']}",
                           _code("data/measured_tip_scan.csv") + "（导入时读取）"]),

    _tool(id="fig_cartoons", group="作图", title="物理示意图（卡通）", short="无需数据的示意图 cartoon_*.png", tag="≈1 s",
         desc="为幻灯片绘制物理示意图，只用解析形状，不需要任何模拟数据：三种场畸变、三段链路、X 扫描的奇偶分解、"
              "针尖直流电势、光斑随功率展宽。可单独渲染一张或全部 5 张。注意 cartoon_power.png 与幻灯片图 "
              "sim_power.png 是两张不同的图。",
         script="scripts/fig_cartoons.py", func=None, out=False, extra="cartoon_keys",
         inputs=lambda c: ["无（解析示意图）"]),

    _tool(id="fig_mechanism", group="作图", title="单脉冲机制图（中文）", short="0.6 ps 模拟 → mechanism.png", tag="≈3 s",
         desc="由模型本身计算的中文机制示意：运行一个 0.6 ps 的 PulseSim（2 mW，光斑在 x = 0），绘制 (a) 0.15 ps 时的"
              "净空间电荷分布，(b) 沿 x 的电子、空穴与净片密度，(c) h = 100 nm 处近场 E_z 随针尖位置的变化（奇项，来自"
              "模型），以及示意性的偶项（宽 4 µm 的高斯，不是模型结果）与二者的单侧叠加。中文标签使用系统字体。",
         script="scripts/fig_mechanism.py", func=None, out=False,
         runtime="约 3 秒（首次运行可能多 10–15 秒用于建立字体缓存）",
         inputs=lambda c: ["无（进程内运行一个 0.6 ps 的 PulseSim）"],
         fixed=lambda c: [R / "mechanism.png"],
         target=lambda spec: R / "mechanism.png", render=png_render,
         history=lambda: fixed_history([R / "mechanism.png"])),

    _tool(id="fig_dipole_geometry", group="作图", title="偶极几何示意（中文）", short="竖直 vs 横向偶极 → dipole_geometry.png",
         tag="≈1 s",
         desc="二维线电荷偶极场的中文解析示意：比较竖直偶极（经典光致 Dember）与横向偶极（漂移分离）的场线，并绘制针尖"
              "在 h = 50、200、500 nm 处感受到的 E_z 随 x 的变化。不使用模拟数据。英文版是『幻灯片图』中的 dipole"
              "（sim_dipole_geometry.png）。",
         script="scripts/fig_dipole_geometry.py", func=None, out=False, runtime="约 1 秒",
         inputs=lambda c: ["无（解析计算）"],
         fixed=lambda c: [R / "dipole_geometry.png"],
         target=lambda spec: R / "dipole_geometry.png", render=png_render,
         history=lambda: fixed_history([R / "dipole_geometry.png"])),

    _tool(id="numerics", group="维护", title="数值自检", short="Poisson / 平衡态 / Jacobian / 暗电流", tag="≈2 s",
         desc="稳态求解器的验证套件（86×50 小网格）：① 共面薄电极间的裸 Poisson 场与解析解 V/(π√(a²−x²)) 相差 < 12 %，"
              "间隙中心 E_z/E_x < 5 %；② V = 0、G = 0 的平衡态残差为零；③ 完整耦合 Jacobian 的有限差分校验"
              "（inv_dt = 0 与 3，各块相对误差 < 1e-5）；④ 10 V 暗态：收敛、|I| < 1e-3 mA、平均 EL2⁺ 在 "
              "0.5–3 × 10¹⁵ cm⁻³ 之间。用于确认安装与数值环境正常；第一项失败后其余项不再运行。",
         script="tests/test_numerics.py", func=None, out=False, runtime="约 2 秒",
         inputs=lambda c: ["无（ltemsim 库本身）"],
         outputs=lambda c: ["仅标准输出：每通过一项打印一行 “… ok”（下方显示为检查表）"],
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
        self.lay.addWidget(label(title, "cardTitle"))
        row = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(16)
        self.combo.currentIndexChanged.connect(self._on_change)
        row.addWidget(self.combo, 1)
        row.addWidget(button("浏览…", self._browse))
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
        super().__init__("运行文件夹（单脉冲链输出，含 X*.npz）", parent)
        self.info = label("", "muted", wrap=True)
        self.lay.addWidget(self.info)
        self.refresh()

    def candidates(self):
        return pulse_run_dirs()

    @staticmethod
    def text(d: Path) -> str:
        return f"{_rel(d).removeprefix('results/')}  ·  {len(x_files(d))} 个 X"

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "选择单脉冲链运行文件夹", str(PC))
        if d:
            self.set(d)

    def _on_change(self, *_):
        d = self.value()
        if d is None:
            self.info.setText("没有找到含 X*.npz 的运行文件夹。")
        else:
            xs = [x for x in (_xval(f) for f in x_files(d)) if x is not None]
            done = sorted(f.name for f in d.glob("redemod_*.json"))
            txt = f"X = {', '.join(f'{x:+g}' for x in sorted(xs))} µm" if xs else "该文件夹中没有 X*.npz"
            if done:
                txt += "  ·  已有：" + ", ".join(done)
            self.info.setText(txt)
        self.changed.emit()


class SolutionPicker(_PathPicker):
    def __init__(self, parent=None):
        super().__init__("稳态解 (.npz)", parent)
        self.lay.addWidget(label("粗网格解来自 results/coarse_scenarios；solution.npz 只在本机运行后才有（未提交）。",
                                 "muted", wrap=True))
        self.lay.addWidget(button("在解查看器中打开", self._explore, "flat"), 0, Qt.AlignmentFlag.AlignLeft)
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
        f, _ = QFileDialog.getOpenFileName(self, "选择稳态解", str(R), "Solution (*.npz)")
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
        lay.addWidget(label("勾选要对比的稳态解；“图例标签”列可双击编辑", "cardTitle"))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["", "图例标签", "文件"])
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
        for text, slot in (("添加文件…", self._browse), ("全部取消", self.clear_checks)):
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
        files, _ = QFileDialog.getOpenFileNames(self, "添加稳态解", str(R), "Solution (*.npz)")
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
        self.count.setText(f"已选 {n} 个 / 共 {len(self._paths)} 个")
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
        lay.addWidget(label(title, "cardTitle"))
        self.table = QTableWidget(len(keys), 3)
        self.table.setHorizontalHeaderLabels(["图", "内容", "输入"])
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
            a.setToolTip("输出：" + ", ".join(f"results/slides/{f}" for f in k["files"]))
            self.table.setItem(r, 0, a)
            self.table.setItem(r, 1, QTableWidgetItem(k["zh"]))
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
                txt, col = "— 无需输入", t.muted
            elif miss:
                txt, col = f"✗ 缺 {len(miss)} 项", t.danger
            else:
                txt, col = "✓ 齐全", t.success
            it = self.table.item(r, 2)
            it.setText(txt)
            it.setForeground(QColor(col))
            it.setToolTip("\n".join(_plain(q["label"]) + (f"（{q['detail']}）" if q["detail"] else "") for q in miss))

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
        text = ("排队中" if job.status == QUEUED else "运行中") if active else self._tag
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


# ============================================================================ page
class ToolsPage(Page):
    title = "分析与作图"
    subtitle = "其余的分析、作图与维护脚本：选择工具，查看说明与前置条件（实时检查），设置参数后运行；输出的图、表与文本显示在下方。"

    def __init__(self):
        super().__init__()
        self.current: str | None = None
        self.tool_jobs: dict = {}
        self.param_pages: dict = {}
        self.entries: list[dict] = []
        self._entry = None
        self._plots: list[PlotWidget] = []
        self._items: dict = {}

        self.btn_check = button("  刷新检查", self.refresh_all, "flat", "重新检查前置文件、可选的解和已有输出")
        self.header.add_action(self.btn_check)
        self.run_btn = button("  运行", self.run, "primary", "在后台子进程中运行所选工具")
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
        self.search.setPlaceholderText("搜索工具（名称 / 脚本 / 说明）…")
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
        tr = QHBoxLayout()
        tr.setSpacing(8)
        self.t_title = _Elided("cardTitle")             # long titles elide instead of widening the page
        self.t_title.setStyleSheet("font-size: 18px;")
        tr.addWidget(self.t_title)
        self.t_group = label("", "badge")
        tr.addWidget(self.t_group)
        self.t_prereq = label("", "badgeOk")
        tr.addWidget(self.t_prereq)
        self.t_fixed = label("固定输出路径", "badgeWarn")
        tr.addWidget(self.t_fixed)
        tr.addStretch(1)
        self.t_runtime = _Elided("muted")
        tr.addWidget(self.t_runtime)
        self.btn_script = button("查看脚本", self._open_script, "flat", "用默认程序打开脚本源码")
        tr.addWidget(self.btn_script)
        rl.addLayout(tr)

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

        vs = QSplitter(Qt.Orientation.Vertical)
        vs.setChildrenCollapsible(False)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.setChildrenCollapsible(False)
        ic = card()
        il = QVBoxLayout(ic)
        il.setContentsMargins(14, 10, 6, 10)
        il.setSpacing(4)
        il.addWidget(label("说明", "cardTitle"))
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
        pl.addWidget(label("参数与输入", "cardTitle"))
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
        srow.addWidget(label("输出来源", "muted"))
        self.src = QComboBox()
        self.src.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.src.setMinimumContentsLength(18)
        self.src.setToolTip("这个工具已有的输出：本程序运行过的任务、results/gui_runs 下的结果以及仓库中提交的结果")
        self.src.view().setMinimumWidth(520)
        self.src.view().setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.src.currentIndexChanged.connect(self._on_src)
        srow.addWidget(self.src, 1)
        self.btn_folder = button("打开文件夹", self._open_folder, "flat", "在访达中显示当前输出所在的文件夹")
        srow.addWidget(self.btn_folder)
        ol.addLayout(srow)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(False)
        run_tab = QWidget()
        rtl = QVBoxLayout(run_tab)
        rtl.setContentsMargins(12, 12, 12, 12)
        self.panel = RunPanel()
        rtl.addWidget(self.panel)
        self.tabs.addTab(run_tab, "运行")
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
            extra = KeyPicker(KEYSETS[kind], "选择要渲染的图（输出到 results/slides/）")
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
            pre = [cond(f"前置检查出错：{_esc(e)}", False, hard=False)]
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
        self.t_runtime.setText("预计 " + runtime.split("（")[0].split("；")[0].removeprefix("约 "))
        self.t_runtime.setToolTip(runtime)

        missing = [p for p in pre if not p["ok"] and not p["optional"]]
        if not pre:
            self.t_prereq.setText("无前置条件")
            self.t_prereq.setObjectName("badgeOk")
        elif missing and all(p.get("sel") for p in missing):
            self.t_prereq.setText("请先在右侧选择输入")
            self.t_prereq.setObjectName("badgeWarn")
        elif missing:
            self.t_prereq.setText(f"✗ 缺少 {len(missing)} 项前置")
            self.t_prereq.setObjectName("badgeErr")
        else:
            self.t_prereq.setText("✓ 前置条件满足")
            self.t_prereq.setObjectName("badgeOk")
        repolish(self.t_prereq)
        self.t_fixed.setVisible(bool(fixed) or writes is not None)
        self.t_fixed.setText("写入所选文件夹" if writes is not None else "固定输出路径")

        # warning banner
        warn = ""
        if fixed:
            n_ex = sum(f.exists() for f in fixed)
            n_git = sum(committed(f) for f in fixed if f.exists())
            warn = (f"<b style='color:{t.warning}'>⚠ 固定输出路径</b>　此脚本只能写到 "
                    + "、".join(_code(_rel(f)) for f in fixed) + "。")
            if n_ex:
                warn += (f"运行会覆盖已有的 {n_ex} 个文件" + (f"（其中 {n_git} 个已提交到 git）" if n_git else "")
                         + "；运行前会请你确认，并可先自动备份到 results/gui_runs/。")
        elif writes is not None:
            warn = (f"<b style='color:{t.warning}'>⚠ 写入输入文件夹</b>　结果 JSON 直接写入所选运行文件夹（不是新文件夹）："
                    + _code(_rel(writes)))
            if Path(writes).exists():
                warn += "　——同名文件已存在，运行前会请你确认覆盖。"
            elif curated(Path(writes).parent):
                warn += "　——这是仓库中的结果文件夹，运行前会请你确认。"
        self.warn_text.setText(warn)
        self.warn.setVisible(bool(warn))

        # description card
        H = [f"<p style='margin:0 0 2px 0'>{_esc(tool['desc'])}</p>"]

        def sec(name):
            H.append(f"<p style='margin:12px 0 3px 0'><b style='color:{t.accent}'>{name}</b></p>")

        def bullets(lines):
            if not lines:
                H.append(f"<span style='color:{t.muted}'>无</span>")
                return
            rows = []
            for s in lines:
                if s.startswith("\x01"):
                    rows.append(f"<tr><td></td><td><table cellspacing='0' cellpadding='0'><tr><td valign='top' "
                                f"style='color:{t.muted}'>&nbsp;–&nbsp;</td><td>{s[1:]}</td></tr></table></td></tr>")
                else:
                    rows.append(f"<tr><td valign='top' style='color:{t.muted}'>•&nbsp;</td><td>{s}</td></tr>")
            H.append("<table cellspacing='0' cellpadding='1'>" + "".join(rows) + "</table>")

        sec("前置条件")
        if not pre:
            H.append(f"<span style='color:{t.success}'>✓</span> 无（不依赖其它脚本的结果）")
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
                    txt += f" <span style='color:{t.muted}'>（可选）</span>"
                if p["detail"]:
                    txt += f" <span style='color:{t.muted if p['ok'] else col}'>— {_esc(p['detail'])}</span>"
                if not p["ok"] and p.get("producer"):
                    target, how = p["producer"]
                    txt += (f"<br><span style='color:{t.muted}'>生成方法：{_esc(how)}</span> <a href='go:{target}'>"
                            f"<span style='color:{t.accent}'>前往『{_esc(producer_name(target))}』</span></a>")
                rows.append(f"<tr><td valign='top' style='color:{col}; font-weight:700'>{mark}&nbsp;</td>"
                            f"<td style='padding-bottom:3px'>{txt}</td></tr>")
            H.append("<table cellspacing='0' cellpadding='1'>" + "".join(rows) + "</table>")
        sec("输入")
        bullets(tool["inputs"](ctx))
        sec("输出")
        outs = list(tool["outputs"](ctx))
        for f in fixed:
            notes = ["固定路径"]
            if f.exists():
                notes.append("已存在，运行会覆盖")
                if committed(f):
                    notes.append("已提交到 git")
            outs.append(_code(_rel(f)) + f" <span style='color:{t.warning}'>（{'，'.join(notes)}）</span>")
        if writes is not None:
            outs.append(_code(_rel(writes)) + f" <span style='color:{t.warning}'>（写入所选运行文件夹"
                        + ("，已存在，将覆盖" if Path(writes).exists() else "") + "）</span>")
        bullets(outs)
        sec("运行时间")
        H.append(_esc(runtime))
        sec("脚本")
        H.append(_code(tool["script"]) + f" <span style='color:{t.muted}'>— "
                 + (f"在子进程中调用 {_esc(tool['func'])}()" if tool["func"] else "在子进程中作为 __main__ 运行")
                 + "，工作目录为仓库根目录</span>")
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
        self.run_btn.setText(("  排队中…" if j.status == QUEUED else "  运行中…") if busy else "  运行")

    def run(self):
        tid = self.current
        if not tid:
            return None
        tool = BY_ID[tid]
        _, form, _ = self._param_page(tid)
        if form is not None and not form.is_valid():
            error_box(self, "参数无效", "以下参数的值无法解析或超出范围：\n" + ", ".join(form.invalid_fields()))
            return None
        ctx = self._ctx(tid)
        pre, fixed, writes = self._checks(tool, ctx)
        hard = [p for p in pre if p.get("hard") and not p["ok"]]
        if hard:
            error_box(self, "无法运行", "\n".join("• " + _plain(p["label"]) + (f"（{p['detail']}）" if p["detail"] else "")
                                               for p in hard))
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
                    error_box(self, "备份失败", f"无法备份将被覆盖的文件：{e}")
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
            self.note.setText(f"已把将被覆盖的 {len(fixed)} 个文件备份到 {_rel(backup)}（已提交的文件也可用 git checkout 恢复）。")
            self.note.show()
        else:
            self.note.hide()
        self._update_run_state()
        return job

    # dialogs (separate methods so that headless tests can replace them)
    def _confirm_missing(self, tool, missing):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("缺少前置文件")
        box.setText(f"「{tool['title']}」需要的部分输入文件不存在，脚本很可能会中途报错。")
        lines = ["• " + _plain(p["label"]) + (f"（{p['detail']}）" if p["detail"] else "") for p in missing]
        prods = []
        for p in missing:
            if p.get("producer") and p["producer"] not in prods:
                prods.append(p["producer"])
        info = "缺少：\n" + "\n".join(lines)
        if prods:
            info += "\n\n生成方法：\n" + "\n".join("• " + how for _, how in prods)
        box.setInformativeText(info)
        go = None
        if prods:
            go = box.addButton(f"前往『{producer_name(prods[0][0])}』", QMessageBox.ButtonRole.ActionRole)
        anyway = box.addButton("仍然运行", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
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
        box.setWindowTitle("覆盖固定路径的输出？")
        box.setText(f"「{tool['title']}」只能写到固定路径，运行会覆盖以下文件：")
        lines = [f"• {_rel(f)}" + ("（已提交到 git）" if committed(f) else "") for f in files]
        box.setInformativeText("\n".join(lines) + "\n\n建议先备份（复制到 results/gui_runs/ 下的新文件夹）；"
                               "已提交的文件也可以之后用 git checkout 恢复。")
        b_backup = box.addButton("备份后运行", QMessageBox.ButtonRole.AcceptRole)
        b_over = box.addButton("直接覆盖", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(b_backup)
        box.exec()
        c = box.clickedButton()
        return "backup" if c is b_backup else ("overwrite" if c is b_over else None)

    def _confirm_write(self, path: Path) -> bool:
        """Writing into the selected input folder: asks before overwriting, and before adding to a curated folder."""
        if path.exists():
            title = "覆盖已有结果？"
            text = (f"{_rel(path)} 已存在" + ("（已提交到 git）" if committed(path) else "")
                    + "，重新解调会覆盖它。继续吗？")
        else:
            title = "写入仓库中的结果文件夹？"
            text = (f"脚本只能把结果写到所选运行文件夹本身，将新建 {_rel(path)}"
                    f"（{_rel(path.parent)} 是仓库中的结果文件夹，不是 results/gui_runs/ 下的新文件夹）。继续吗？")
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
            entries.append(dict(label=f"任务 {when} · {j.title}", target=target, job=j, key=key))
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
            self.src.addItem("（暂无输出）")
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
            msg = f"图 {key} 还没有生成（results/slides/ 中没有对应文件）。点击右上角『运行』渲染它。"
        else:
            msg = "还没有输出。设置好参数后点击右上角『运行』，完成后结果会自动显示在这里。"
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
            items = [("text", "错误", f"无法显示这个输出：{type(ex).__name__}: {ex}", "error.txt")]
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
            msg = label(empty or "没有找到输出文件。", "muted", wrap=True)
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            l.addStretch(1)
            l.addWidget(msg)
            l.addStretch(1)
            self.tabs.addTab(w, "输出")

    def _table_widget(self, headers, rows, note=""):
        t = ThemeManager.instance().tokens
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(10, 8, 10, 10)
        l.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(label(note or f"{len(rows)} 行", "muted", wrap=True), 1)
        tw = QTableWidget(len(rows), len(headers))
        top.addWidget(button("复制表格", lambda: self._copy_table(headers, rows), "flat", "以制表符分隔复制到剪贴板"), 0,
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
        top.addWidget(label(f"{len(text.splitlines())} 行", "muted"), 1)
        top.addWidget(button("复制", lambda: QGuiApplication.clipboard().setText(text), "flat"))
        top.addWidget(button("保存为文本…", lambda: self._save_text(text, fname), "flat",
                             "把这份文本输出保存为 .txt 文件"))
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
        path, _ = QFileDialog.getSaveFileName(self, "保存文本输出", str(default), "Text (*.txt);;所有文件 (*)")
        if path:
            try:
                Path(path).write_text(text + "\n", encoding="utf-8")
            except OSError as e:
                error_box(self, "保存失败", str(e))

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
