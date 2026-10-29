"""Steady-state drift-diffusion + EL2 solver page (scripts/run_baseline.py)."""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QSizePolicy, QSplitter, QTabWidget,
                               QVBoxLayout, QWidget)

from .. import icons, nav, paths
from ..jobs import FINISHED, STATUS_ZH, JobManager
from ..paramspec import params_specs
from ..theme import ThemeManager, repolish
from ..widgets.common import MetricTile, Page, button, error_box, label, reveal
from ..widgets.imageview import ImageView
from ..widgets.paramform import ParamForm
from ..widgets.runpanel import RunPanel

# Params fields that no code reads (verified with grep) or that only the pulse model uses
HIDDEN = {"gummel_max_iter", "gummel_tol", "poisson_max_iter", "poisson_tol", "generation_ramp", "bias_ramp_steps",
          "rep_rate"}
QUICK = {"dx_fine": 0.5, "x_fine": 8.0, "dz_fine": 0.25}      # the coarse grid of results/coarse_scenarios (30-60 s)
FIGURES = (("maps.png", "场与载流子分布"), ("profiles.png", "0.5 µm 深度剖面"), ("tip_scan.png", "针尖扫描 vs 实测"),
           ("polarisation.png", "极化 P = n·E"))
RUNTIME = "生产网格约 6–7 分钟，粗网格约 30–60 秒"
# Allowed ranges (inclusive). Out-of-range values make the solver fail late or hang: a negative dx_fine never
# finishes building the grid, T = 0 or spot_w_um = 0 divide by zero, P_avg_mW < 0 or reflectivity > 1 give G < 0.
_POS = 1e-30
BOUNDS = {
    "T": (1.0, 2000.0), "gap": (0.1, None), "x_half": (1.0, None), "depth": (0.5, None), "air_height": (0.5, None),
    "dx_fine": (0.005, None), "x_fine": (0.05, None), "dx_coarse": (0.01, None), "dz_fine": (0.005, None),
    "z_fine": (0.05, None), "dz_coarse_semi": (0.01, None), "dz_coarse_air": (0.01, None),
    "eps_r": (1.0, None), "Eg": (0.01, None), "Nc": (1.0, None), "Nv": (1.0, None), "mu_n": (_POS, None),
    "mu_p": (_POS, None), "v_sat_n": (1.0, None), "v_sat_p": (1.0, None), "B_rad": (0.0, None), "S_surface": (0.0, None),
    "contact_density": (0.0, None), "tau_srh_ns": (0.0, None), "N_A": (0.0, None), "N_D": (0.0, None),
    "N_T": (0.0, None), "E_T_below_Ec": (0.0, None), "sigma_n": (0.0, None), "sigma_p": (0.0, None),
    "sigma_n_opt": (0.0, None), "sigma_p_opt": (0.0, None), "m_eff_n": (1e-3, None), "m_eff_p": (1e-3, None),
    "wavelength_nm": (1.0, None), "P_avg_mW": (0.0, None), "spot_w_um": (0.01, None), "reflectivity": (0.0, 1.0),
    "alpha_cm": (0.0, None), "tip_height_um": (1e-4, None), "n_floor": (_POS, None),
}


def steady_specs():
    out = []
    for s in params_specs():
        if s.name in HIDDEN:
            continue
        lo, hi = BOUNDS.get(s.name, (None, None))
        out.append(dataclasses.replace(s, minimum=lo, maximum=hi) if s.type in ("float", "int") else s)
    return out


def check_params(v: dict) -> list[str]:
    """Cross-field checks the per-field ranges cannot express (Chinese messages; empty = ok)."""
    bad = []
    if v["x_fine"] >= v["x_half"]:
        bad.append(f"细网格范围 x_fine = {v['x_fine']:g} µm 必须小于计算区半宽 x_half = {v['x_half']:g} µm")
    if v["gap"] / 2 >= v["x_half"]:
        bad.append(f"电极间隙 gap = {v['gap']:g} µm 的一半必须小于 x_half = {v['x_half']:g} µm（电极要在计算区内）")
    if v["z_fine"] >= min(v["depth"], v["air_height"]):
        bad.append(f"细网格 z 范围 z_fine = {v['z_fine']:g} µm 必须小于 GaAs 厚度和空气层高度")
    if v["tip_height_um"] >= v["air_height"]:
        bad.append(f"针尖采样高度 {v['tip_height_um']:g} µm 必须小于空气层高度 {v['air_height']:g} µm")
    if v["E_T_below_Ec"] >= v["Eg"]:
        bad.append("EL2 能级 E_c−E_T 必须小于带隙 E_g")
    xs = v.get("tip_x_scan_um") or ()
    if len(xs) < 2:
        bad.append("针尖扫描位置至少需要 2 个点（空列表会在求解完成后才报错）")
    elif max(abs(x) for x in xs) >= v["x_half"]:
        bad.append(f"针尖扫描位置必须在计算区内（|X| < x_half = {v['x_half']:g} µm）")
    return bad


def steady_metrics(summary: dict) -> dict:
    """Headline numbers from a run_baseline / sweep summary.json."""
    out = {}
    I = summary.get("I_mA", {})
    if I:
        out["I"] = abs(I.get("left", float("nan")))
        ug = I.get("unity_gain")
        out["gain"] = out["I"] / ug if ug else float("nan")
        out["unity"] = ug
    ts = summary.get("tip_scan")
    if ts and ts.get("X"):
        X = np.asarray(ts["X"], float)
        S = np.asarray(ts["S"], float)
        o = np.argsort(X)
        X, S = X[o], S[o]
        A = np.abs(S)
        out["peak_X"] = float(X[np.argmax(A)])
        out["S0"] = float(np.interp(0.0, X, A) / A.max()) if A.max() > 0 else float("nan")
        a5, am5 = np.interp(5.0, X, A), np.interp(-5.0, X, A)
        out["ratio5"] = float(a5 / am5) if am5 > 0 else float("nan")
    log = summary.get("log")
    if log:
        out["converged"] = all(bool(l.get("converged")) for l in log if l.get("converged") is not None)
        out["wall"] = max((l.get("t_wall", 0) for l in log), default=0)      # t_wall is cumulative per stage
    elif isinstance(summary.get("converged"), list):          # sweep run summaries
        out["converged"] = all(bool(c) for c in summary["converged"] if c is not None)
        out["wall"] = summary.get("wall_s")
    return out


def _fmt(v, f):
    return f.format(v + 0.0) if v is not None and np.isfinite(v) else "—"      # + 0.0: no "-0.0"


def result_params(d: Path, summary: dict | None = None) -> dict:
    """Params overrides stored with a result: params.json (run_baseline) or summary["params"] (sweep runs)."""
    try:
        p = json.loads((d / "params.json").read_text())
        if isinstance(p, dict):
            return p
    except (OSError, ValueError):
        pass
    if summary is None:
        try:
            summary = json.loads((d / "summary.json").read_text())
        except (OSError, ValueError):
            summary = {}
    p = summary.get("params") if isinstance(summary, dict) else None
    return p if isinstance(p, dict) else {}


def is_quick(pr: dict) -> bool:
    try:
        return all(abs(float(pr.get(k, -1)) - v) < 1e-12 for k, v in QUICK.items())
    except (TypeError, ValueError):
        return False


class _Elided(QLabel):
    """One-line label that elides in the middle instead of forcing the layout to its full text width."""

    def __init__(self, text="", obj=None):
        super().__init__()
        if obj:
            self.setObjectName(obj)
        self._full = ""
        self.setText(text)

    def setText(self, s):
        self._full = s
        self.setToolTip(s)
        self._elide()
        self.updateGeometry()

    def text(self):
        return self._full

    def sizeHint(self):
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + 4, super().sizeHint().height())

    def minimumSizeHint(self):
        return QSize(80, super().minimumSizeHint().height())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._elide()

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._elide()
            self.updateGeometry()

    def _elide(self):
        super().setText(self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideMiddle, max(20, self.width())))


class SteadyPage(Page):
    title = "稳态求解"
    subtitle = ("连续泵浦、横向偏压的 SI-GaAs 间隙：2-D 漂移-扩散 + EL2 深能级自洽稳态（耦合 Newton + 伪瞬态延拓），"
                f"并计算针尖 E_z 随光斑-针尖位置 X 的变化。{RUNTIME}。")
    WIDE = 640          # results tab width above which the metric tiles fit in one row

    def __init__(self):
        super().__init__()
        self.job = None
        self.result_dir = None
        self.quick = QCheckBox("粗网格快速模式")
        self.quick.setToolTip("dx_fine = 0.5 µm, x_fine = 8 µm, dz_fine = 0.25 µm（与 results/coarse_scenarios 相同，"
                              "约 30–60 秒；数值比生产网格粗，用于快速试参数）")
        self.quick.toggled.connect(self._toggle_quick)
        self.header.add_action(self.quick)
        self.run_btn = button("  运行求解", self.run, "primary")
        self.header.add_action(self.run_btn)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        # ---------------- left: parameters
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)
        self.form = ParamForm(steady_specs())
        self.form.changed.connect(self._sync_quick)
        ll.addWidget(self.form, 1)
        orow = QHBoxLayout()
        orow.addWidget(QLabel("输出文件夹"))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("自动：results/gui_runs/steady_<时间>")
        self.out_edit.setToolTip("留空 = 每次运行新建 results/gui_runs/steady_<时间>（不会覆盖已有结果）")
        orow.addWidget(self.out_edit, 1)
        b = QPushButton("选择…")
        b.clicked.connect(self._pick_out)
        orow.addWidget(b)
        ll.addLayout(orow)
        left.setMinimumWidth(330)     # the form needs ~376 px without a horizontal scroll bar; 330 keeps 1000 px usable
        split.addWidget(left)

        # ---------------- right: run / results / history
        self.tabs = QTabWidget()
        run_tab = QWidget()
        rl = QVBoxLayout(run_tab)
        rl.setContentsMargins(12, 12, 12, 12)
        self.panel = RunPanel()
        # long job titles must not widen the page (RunPanel's title label does not elide)
        self.panel.title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.panel.job_finished.connect(self._on_finished)
        rl.addWidget(self.panel)
        self.tabs.addTab(run_tab, "运行")

        self.res_tab = res_tab = QWidget()
        xl = QVBoxLayout(res_tab)
        xl.setContentsMargins(10, 10, 10, 10)
        xl.setSpacing(8)
        self.res_head = QGridLayout()
        self.res_head.setHorizontalSpacing(6)
        self.res_head.setVerticalSpacing(4)
        self.res_bar = QWidget()            # folder name (elided) + status badges
        ml = QHBoxLayout(self.res_bar)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(6)
        self.res_title = _Elided("尚未载入结果", "cardTitle")
        self.res_state = label("", "badge")
        self.res_grid = label("粗网格", "badge")
        self.res_grid.setToolTip("dx_fine = 0.5 µm, x_fine = 8 µm, dz_fine = 0.25 µm：比生产网格粗，数值仅供快速比较")
        ml.addWidget(self.res_title)
        ml.addWidget(self.res_state)
        ml.addWidget(self.res_grid)
        ml.addStretch(1)
        self.btn_explore = button("在解查看器中打开", self._open_explorer)
        self.btn_folder = button("打开文件夹", lambda: self.result_dir and reveal(self.result_dir))
        self.btn_params = button("载入此结果的参数", self._load_result_params)
        for b in (self.btn_params, self.btn_explore, self.btn_folder):
            b.setObjectName("flat")
        xl.addLayout(self.res_head)
        self.tile_grid = QGridLayout()
        self.tile_grid.setSpacing(6)
        self.m_I = MetricTile("光电流 |I|", "—")
        self.m_gain = MetricTile("增益 I / I_unity", "—")
        self.m_peak = MetricTile("|S| 峰值位置 X", "—")
        self.m_ratio = MetricTile("S(+5) / S(−5)", "—")
        self.m_s0 = MetricTile("S(0) / max", "—")
        self.tiles = (self.m_I, self.m_gain, self.m_peak, self.m_ratio, self.m_s0)
        for m in self.tiles:        # MetricTile's Maximum height policy clips a hint that wraps to two lines
            m.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        xl.addLayout(self.tile_grid)
        self._wide = None
        self._relayout(True)
        self.fig_tabs = QTabWidget()
        self.views = {}
        for fn, title in FIGURES:
            v = ImageView()
            # the "name · W×H" caption must not add its width to the toolbar's minimum (keeps 1000 px windows usable)
            v.caption.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            self.views[fn] = v
            self.fig_tabs.addTab(v, title)
        xl.addWidget(self.fig_tabs, 1)
        self.tabs.addTab(res_tab, "结果")

        hist_tab = QWidget()
        hl = QVBoxLayout(hist_tab)
        hl.setContentsMargins(12, 12, 12, 12)
        hrow = QHBoxLayout()
        hrow.addWidget(label("已有的稳态结果（双击载入）", "cardTitle"), 1)
        rb = button("刷新", self.refresh_history)
        rb.setObjectName("flat")
        hrow.addWidget(rb)
        hl.addLayout(hrow)
        self.hist = QListWidget()
        self.hist.itemDoubleClicked.connect(self._on_hist)
        hl.addWidget(self.hist, 1)
        self.tabs.addTab(hist_tab, "历史结果")
        split.addWidget(self.tabs)
        # follow the tab widget's width: a hidden tab page is not resized, and a stale wide layout there
        # would keep the whole page wider than the window
        self.tabs.installEventFilter(self)
        split.setStretchFactor(0, 0)          # extra width goes to the results; the form keeps its size
        split.setStretchFactor(1, 1)
        split.setSizes([430, 900])
        self.body.addWidget(split, 1)

        self.refresh_history()
        default = paths.RESULTS / "baseline"
        if (default / "summary.json").exists():
            self.show_result(default, switch=False)
        self._sync_quick()
        self._retheme()
        ThemeManager.instance().changed.connect(self._retheme)
        JobManager.instance().job_finished.connect(self._on_any_finished)

    def _retheme(self, *_):
        t = ThemeManager.instance().tokens
        self.run_btn.setIcon(icons.icon("play", t.on_accent))

    # ------------------------------------------------------------------ layout
    def eventFilter(self, obj, e):
        if obj is self.tabs and e.type() == QEvent.Type.Resize:
            self._relayout(obj.width() - 4 >= self.WIDE)
        return super().eventFilter(obj, e)

    def _relayout(self, wide: bool):
        """One row of tiles / buttons beside the title when wide; 3 + 2 tiles and buttons below the title when narrow."""
        if wide == self._wide:
            return
        self._wide = wide
        h, g = self.res_head, self.tile_grid
        head = (self.res_bar, self.btn_params, self.btn_explore, self.btn_folder)
        for w in (*head, *self.tiles):
            (h if w in head else g).removeWidget(w)
        for c in range(6):
            h.setColumnStretch(c, 0)
            g.setColumnStretch(c, 0)
        if wide:
            h.addWidget(self.res_bar, 0, 0)
            for i, b in enumerate((self.btn_params, self.btn_explore, self.btn_folder)):
                h.addWidget(b, 0, i + 1)
            h.setColumnStretch(0, 1)
        else:
            h.addWidget(self.res_bar, 0, 0, 1, 4)
            for i, b in enumerate((self.btn_params, self.btn_explore, self.btn_folder)):
                h.addWidget(b, 1, i)
            h.setColumnStretch(3, 1)
        cols = 5 if wide else 3
        for i, m in enumerate(self.tiles):
            g.addWidget(m, i // cols, i % cols)
        for c in range(cols):
            g.setColumnStretch(c, 1)
        self._tile_hints()
        self.res_tab.layout().invalidate()       # the tab may be hidden: drop its cached minimum size now

    def _tile_hints(self):
        """Narrow layout: two rows of tiles already take the height, so the hints live only in the tooltips."""
        for m in self.tiles:
            m.setToolTip(m.property("tip") or m.hint.text())
            m.hint.setVisible(bool(self._wide) and bool(m.hint.text()))

    # ------------------------------------------------------------------ run
    def _toggle_quick(self, on):
        if on:
            self.form.set_values(QUICK)
        else:
            self.form.set_values({k: self.form.rows[k].spec.default for k in QUICK})

    def _sync_quick(self):
        """The checkbox mirrors the grid fields (editing dx_fine etc. by hand un-ticks it)."""
        q = is_quick(self.form.values())
        if self.quick.isChecked() != q:
            self.quick.blockSignals(True)
            self.quick.setChecked(q)
            self.quick.blockSignals(False)
        self.run_btn.setToolTip(f"提交到任务队列（{'粗网格：约 30–60 秒' if q else '生产网格：约 6–7 分钟'}）")

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出文件夹", str(paths.GUI_RUNS if paths.GUI_RUNS.exists()
                                                                       else paths.RESULTS))
        if d:
            self.out_edit.setText(paths.rel(d))

    def _output_dir(self, quick: bool) -> Path | None:
        """Validated output folder (asks before overwriting); None = cancelled."""
        txt = self.out_edit.text().strip()
        if not txt:
            return paths.new_run_dir("steady_quick" if quick else "steady")
        out = paths.absolute(Path(txt).expanduser())
        r = out.resolve()
        if r in (paths.ROOT.resolve(), paths.RESULTS.resolve(), paths.GUI_RUNS.resolve()) or r in paths.ROOT.resolve().parents:
            error_box(self, "输出文件夹不合适", f"{txt} 是项目或 results 的根目录。请选择（或输入）一个专用的子文件夹，"
                                              "或留空让程序自动新建 results/gui_runs/steady_<时间>。")
            return None
        if out.exists() and not out.is_dir():
            error_box(self, "输出路径不是文件夹", f"{txt} 是一个文件。请输入文件夹路径，或留空自动新建。")
            return None
        busy = [j for j in JobManager.instance().active() if j.out and paths.absolute(j.out).resolve() == r]
        if busy:
            error_box(self, "输出文件夹正被占用", f"任务「{busy[0].title}」（{STATUS_ZH.get(busy[0].status)}）"
                                               f"正在写入 {paths.rel(out)}。请换一个文件夹，或留空自动新建。")
            return None
        if out.is_dir() and any(out.iterdir()):
            committed = r.is_relative_to(paths.RESULTS.resolve()) and not r.is_relative_to(paths.GUI_RUNS.resolve())
            msg = f"{paths.rel(out)} 中已有文件，运行会覆盖其中的图、CSV、summary.json 和 params.json。"
            if committed:
                msg += "\n\n注意：这是仓库中已提交的结果文件夹，覆盖后只能用 git 恢复。"
            if (out / "summary.json").exists():
                msg += "\n\n该文件夹中已有一个完整的稳态结果。"
            r2 = QMessageBox.question(self, "覆盖已有结果？", msg + "\n\n继续吗？",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
            if r2 != QMessageBox.StandardButton.Yes:
                return None
        return out

    def run(self):
        if not self.form.is_valid():
            rows = self.form.rows
            lines = []
            for n in self.form.invalid_fields():
                s = rows[n].spec
                rng = ("" if s.minimum is None and s.maximum is None else
                       f"（允许范围 {'' if s.minimum is None else f'≥ {s.minimum:g}'}"
                       f"{'' if s.minimum is None or s.maximum is None else '，'}"
                       f"{'' if s.maximum is None else f'≤ {s.maximum:g}'}）")
                lines.append(f"• {s.label}（{n}）= {rows[n].editor.text() if hasattr(rows[n].editor, 'text') else '?'}{rng}")
            error_box(self, "参数无效", "以下参数无法解析或超出允许范围：\n" + "\n".join(lines))
            return
        v = self.form.values()
        bad = check_params(v)
        if bad:
            error_box(self, "参数组合无效", "请先修正：\n" + "\n".join(f"• {b}" for b in bad))
            return
        if self.job is not None and self.job.active:
            r = QMessageBox.question(self, "已有稳态任务", f"「{self.job.title}」仍在{STATUS_ZH.get(self.job.status)}。\n"
                                     "再提交一个新任务吗？（新任务会排队，等前面的任务完成后运行）",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes:
                return
        quick = is_quick(v)
        out = self._output_dir(quick)
        if out is None:
            return
        ov = self.form.overrides()
        desc = ", ".join(f"{k}={x}" for k, x in ov.items() if not (quick and k in QUICK)) or "默认参数"
        if len(desc) > 60:
            desc = desc[:58].rstrip(", ") + " …"
        title = ("稳态(粗网格) · " if quick else "稳态 · ") + desc
        self.job = JobManager.instance().submit(
            title, "steady", "scripts/run_baseline.py", "main", kwargs=dict(out=paths.rel(out), **ov), out=out,
            progress={"type": "steady"})
        self.panel.attach(self.job)
        self.panel.title.setToolTip(", ".join(f"{k}={x}" for k, x in ov.items()) or "默认参数")
        self.tabs.setCurrentIndex(0)

    def _on_finished(self, job):
        if job is self.job and job.status == FINISHED and job.out:
            self.show_result(paths.absolute(job.out))
            self.refresh_history()

    def _on_any_finished(self, job):
        # runs not attached to the panel (queued behind another one, re-run from the job center) still show up
        if job.kind == "steady" and job is not self.job and job.status == FINISHED:
            self.refresh_history()

    # ------------------------------------------------------------------ results
    def show_result(self, d, switch=True):
        d = Path(d)
        if d.is_file():
            d = d.parent
        self.result_dir = d
        try:
            summ = json.loads((d / "summary.json").read_text())
            if not isinstance(summ, dict):
                summ = {}
        except (OSError, ValueError):
            summ = {}
        pr = result_params(d, summ)
        m = steady_metrics(summ)
        try:
            V = float(pr.get("V_bias", 25.0))
        except (TypeError, ValueError):
            V = float("nan")
        side = pr.get("anode_side", "-x")
        # X = -x_tip, so X > 0 puts the tip on the -x side; that is the anode side when the -x electrode is positive
        anode_at_pos_X = None if not np.isfinite(V) or V == 0 else ((side == "-x") == (V > 0))
        ug = m.get("unity")
        same = anode_at_pos_X in (None, True)          # same polarity as the measured +25 V scan
        hints = {   # tile: (short hint under the value, tooltip)
            self.m_I: ("实测 ≈ 0.6 mA", "实测光电流约 0.6 mA（+25 V）"),
            self.m_gain: (f"I_unity {ug:.2f} mA" if ug else "",
                          "增益 = |I| / I_unity；I_unity 为每个吸收光子贡献一个电荷时的电流" if ug else ""),
            self.m_peak: ("实测 +5 µm" if same else "实测(镜像) −5 µm",
                          "实测 +25 V 扫描的峰值在 X = +5 µm（针尖在阳极侧）" +
                          ("" if same else "；本结果极性相反，对应 X = −5 µm")),
            self.m_ratio: ({None: "无偏压", True: ">1 阳极侧强", False: ">1 阴极侧强"}[anode_at_pos_X],
                           "无偏压，无阳极/阴极之分" if anode_at_pos_X is None else
                           f"S(+5)/S(−5) > 1：针尖在{'阳极' if anode_at_pos_X else '阴极'}侧时信号更强"
                           f"（X = 光斑 − 针尖 > 0 是{'阳极' if anode_at_pos_X else '阴极'}侧）"),
            self.m_s0: ("实测 0.24", "实测 +25 V：S(0) / max = 0.24"),
        }
        for tile, key, f in ((self.m_I, "I", "{:.3f} mA"), (self.m_gain, "gain", "{:.2f}"), (self.m_peak, "peak_X", "{:+.1f} µm"),
                             (self.m_ratio, "ratio5", "{:.2f}"), (self.m_s0, "S0", "{:.2f}")):
            tile.set(_fmt(m.get(key), f), hints[tile][0])
            tile.setProperty("tip", hints[tile][1])
        self._tile_hints()
        conv = m.get("converged")
        if not d.is_dir():
            state, obj, tip = "文件夹不存在", "badgeErr", str(d)
        elif not summ:
            state, obj, tip = "未完成：无 summary.json", "badgeWarn", "运行未完成、已取消或失败（文件夹中没有 summary.json）"
        elif conv is None:
            state, obj, tip = "", "badge", ""
        else:
            state, obj, tip = (("已收敛", "badgeOk", "暗态与光照两个阶段的稳态 Newton 均已收敛") if conv else
                               ("未完全收敛", "badgeWarn", "至少一个阶段到达 t_end 或步数上限时稳态 Newton 仍未收敛"))
        self.res_title.setText(paths.rel(d))
        self.res_state.setText(state)
        self.res_state.setToolTip(tip)
        self.res_state.setObjectName(obj)
        self.res_state.setVisible(bool(state))
        repolish(self.res_state)
        self.res_grid.setVisible(bool(summ) and is_quick(pr))
        for fn, _ in FIGURES:
            if (d / fn).exists():
                self.views[fn].load(d / fn)
            else:
                self.views[fn].clear(f"此结果没有 {fn}")
        has_sol = (d / "solution.npz").exists()
        self.btn_explore.setEnabled(has_sol)
        self.btn_explore.setToolTip("在「解查看器」中交互查看 ψ、E、n、p、EL2⁺" if has_sol else
                                    "该结果没有保存 solution.npz（仓库中未提交）")
        self.btn_params.setEnabled(bool(pr))
        if switch:
            self.tabs.setCurrentIndex(1)

    def _load_result_params(self):
        if not self.result_dir:
            return
        d = result_params(self.result_dir)
        if not d:
            error_box(self, "无法载入参数", f"{paths.rel(self.result_dir)} 中没有可读的 params.json / summary.json。")
            return
        self.form.set_values({k: v for k, v in d.items() if k in self.form.rows}, reset_others=True)

    def _open_explorer(self):
        if self.result_dir and (self.result_dir / "solution.npz").exists():
            nav.navigate("explorer", path=str(self.result_dir / "solution.npz"))

    def _on_hist(self, it):
        d = it.data(Qt.ItemDataRole.UserRole)
        if d:
            self.show_result(d)

    def refresh_history(self):
        cur = self.hist.currentItem()
        keep = cur.data(Qt.ItemDataRole.UserRole) if cur is not None else None
        self.hist.clear()

        def mtime(p):
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0

        gui = sorted((d for d in paths.GUI_RUNS.glob("steady*") if (d / "summary.json").exists()),
                     key=lambda d: mtime(d / "summary.json"), reverse=True)
        cands = [d for d in [paths.RESULTS / "baseline"] if (d / "summary.json").exists()] + gui
        cands += [s.parent for s in sorted(paths.RESULTS.glob("sweeps/*/*/summary.json"))]
        for d in cands:
            try:
                summ = json.loads((d / "summary.json").read_text())
                m = steady_metrics(summ)
            except (OSError, ValueError, AttributeError, TypeError):
                summ, m = {}, {}
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime(d / "summary.json")))
            tags = ("  ·  粗网格" if is_quick(result_params(d, summ)) else "") + \
                   ("" if m.get("converged", True) else "  ·  ⚠ 未完全收敛")
            txt = (f"{paths.rel(d)}{tags}\n    I = {_fmt(m.get('I'), '{:.3f} mA')} · 峰值 X = "
                   f"{_fmt(m.get('peak_X'), '{:+.1f} µm')} · {when}")
            it = QListWidgetItem(txt)
            it.setData(Qt.ItemDataRole.UserRole, str(d))
            it.setToolTip(str(d))
            self.hist.addItem(it)
            if keep == str(d):
                self.hist.setCurrentItem(it)
        if not cands:
            it = QListWidgetItem("还没有稳态结果。运行一次求解后，结果会出现在这里。")
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.hist.addItem(it)

    def on_show(self):
        self.refresh_history()          # cheap; picks up runs finished from the job center or the command line

    def open(self, path=None, **_):
        if path:
            self.show_result(path)
