"""Home page: project overview, quick start, recent jobs, key result figures and the measured scan."""
from __future__ import annotations

import csv
import html
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QImageReader, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QSizePolicy, QVBoxLayout,
                               QWidget)

from .. import icons, nav, paths
from ..jobs import CANCELLED, FAILED, FINISHED, QUEUED, RUNNING, STATUS_ZH, JobManager
from ..theme import ThemeManager
from ..widgets.common import ClickableCard, Page, button, label, reveal
from ..widgets.runpanel import fmt_elapsed
from ..workers import run_async

KIND_ZH = {"steady": "稳态求解", "sweep": "参数扫描", "pulse": "单脉冲链", "redemod": "重新解调", "fdtd": "泵浦 FDTD",
           "fdtd_heights": "FDTD 多高度", "tip_sim": "针尖位置模拟", "tip_demod": "针尖位置解调", "tool": "分析工具"}
KIND_ICON = {"steady": "grid", "sweep": "sliders", "pulse": "zap", "redemod": "zap", "fdtd": "waves",
             "fdtd_heights": "waves", "tip_sim": "target", "tip_demod": "target", "tool": "chart"}

# (page id, icon, title, description, typical runtime)
QUICK = (
    ("steady", "grid", "稳态求解", "2-D 漂移-扩散 + EL2 自洽稳态：场与载流子分布、针尖扫描，并与实测对比。",
     "粗网格 ~1 min · 生产网格 ~7 min"),
    ("sweep", "sliders", "参数扫描", "对泵浦功率、EL2、σ_p、光斑或任意 Params 字段做扫描，汇总对比。", "每个取值 ~7 min"),
    ("pulse", "zap", "单脉冲发射链", "泵浦 → 载流子动力学 → 针尖近场 → tapping 解调，逐个光斑位置 X 计算。",
     "~3–6 min（7 个 X）"),
    ("fdtd", "waves", "泵浦 FDTD", "820 nm 泵浦光在针尖下的 2-D FDTD，得到近场增强图（有/无针尖）。", "~5 min"),
    ("tip", "target", "针尖位置链", "针尖作为等势体进入 Poisson，逐 (X, h) 模拟后联合 cos nθ 解调。", "25–60 min"),
    ("explorer", "layers", "解查看器", "打开 solution.npz，交互查看 ψ、E、n、p、EL2⁺ 分布与剖面。", "即时"),
    ("results", "folder", "结果浏览", "浏览 results/ 下全部图像、表格与数据文件。", "即时"),
    ("tools", "chart", "分析与作图", "场弯曲、重新解调、幻灯片作图等分析工具。", "多数数秒 · 部分 5–10 min"),
)

# key result figures shown as thumbnails: (repo-relative path, caption)
GALLERY = (
    ("results/slides/sim_xscan.png", "X 扫描：各谐波 vs 实测"),
    ("results/slides/sim_positions.png", "逐针尖位置链"),
    ("results/slides/sim_steady_state.png", "稳态漂移-扩散"),
    ("results/slides/sim_pump_fdtd.png", "泵浦 FDTD 近场增强"),
    ("results/slides/sim_dynamics.png", "单脉冲载流子动力学"),
    ("results/slides/sim_nearfield_demod.png", "近场与 tapping 解调"),
    ("results/slides/sim_dember.png", "漂移 + photo-Dember 干涉"),
    ("results/slides/sim_power.png", "功率依赖"),
    ("results/slides/sim_field_bending.png", "偏压场弯曲"),
    ("results/slides/sim_band_check.png", "频段检查"),
    ("results/slides/sim_dipole_geometry.png", "偶极几何"),
    ("results/baseline/tip_scan.png", "基准稳态：针尖扫描 vs 实测"),
    ("results/baseline/maps.png", "基准稳态：场与载流子分布"),
)

FALLBACK_SUMMARY = ("对连续泵浦、横向加偏压的 SI-GaAs 间隙做 2-D 自洽漂移-扩散 + EL2 深能级稳态模拟，再计算针尖处 E_z "
                    "随光斑-针尖相对位置 X 的变化，并与 ±25 V 的实测 RMS 幅度对比。")


# ----------------------------------------------------------------------------- helpers
def status_color(status: str, t) -> str:
    return {RUNNING: t.accent, QUEUED: t.muted, FINISHED: t.success, FAILED: t.danger, CANCELLED: t.warning}.get(status, t.muted)


def ago(ts: float | None) -> str:
    if not ts:
        return ""
    d = max(0, time.time() - ts)
    if d < 60:
        return "刚刚"
    if d < 3600:
        return f"{int(d // 60)} 分钟前"
    if d < 86400:
        return f"{int(d // 3600)} 小时前"
    if d < 30 * 86400:
        return f"{int(d // 86400)} 天前"
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def readme_summary() -> str:
    """The 中文摘要 paragraph of README.md as rich text (inline `code` kept)."""
    try:
        text = (paths.ROOT / "README.md").read_text(encoding="utf-8")
    except OSError:
        return html.escape(FALLBACK_SUMMARY)
    m = re.search(r"中文摘要[：:]\s*(.+?)(?:\n\s*\n|\Z)", text, re.S)
    if not m:
        return html.escape(FALLBACK_SUMMARY)
    s = ""
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        # CJK lines are joined without a space; keep one next to ASCII words
        if s and (s[-1].isascii() and s[-1] not in "，。；：" or line[0].isascii()):
            s += " "
        s += line
    s = html.escape(s)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", s)


def style_code(rich: str, color: str) -> str:
    """QLabel renders <code> in the proportional font and breaks paths after '/'; use a styled span instead,
    with a word joiner (U+2060) after each '/' (QLabel ignores white-space: nowrap/pre on spans)."""
    return re.sub(r"<code>(.*?)</code>", lambda m: f"<span style='font-family: Menlo, \"SF Mono\", monospace; "
                  f"color: {color}'>{m.group(1).replace('/', '/&#8288;')}</span>", rich)


def git_info() -> dict:
    """Branch, last commit and number of modified tracked files; empty dict when git is unavailable."""
    def run(*a):
        r = subprocess.run(["git", "-C", str(paths.ROOT), *a], capture_output=True, text=True, timeout=4)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip())
        return r.stdout.strip()
    try:
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        h, ct, subject = run("log", "-1", "--format=%h%x1f%ct%x1f%s").split("\x1f", 2)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        return {}
    try:
        dirty = len([l for l in run("status", "--porcelain", "-uno").splitlines() if l.strip()])
    except (OSError, RuntimeError, subprocess.SubprocessError):
        dirty = None
    return dict(branch=branch, hash=h, time=float(ct), subject=subject, dirty=dirty)


def env_info() -> dict:
    out = dict(python=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
               cpus=os.cpu_count() or 0, exe=sys.executable)
    for mod in ("numpy", "scipy", "matplotlib"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def read_measured(path=None) -> dict:
    """{bias_V: [(X, rms, err), ...]} sorted by X."""
    data: dict[float, list] = {}
    with open(path or paths.MEASURED_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                data.setdefault(float(r["bias_V"]), []).append((float(r["X_um"]), float(r["rms"]), float(r["rms_err"])))
            except (KeyError, ValueError):
                continue
    return {v: sorted(pts) for v, pts in data.items()}


def load_thumbs(files, width=720):
    """Decode images scaled to at most `width` px (or a per-file width for (path, width) items).

    Runs in a worker thread, so it only touches QImage.
    """
    out = []
    for f in files:
        if isinstance(f, tuple):
            f, width = f
        r = QImageReader(str(f))
        s = r.size()
        if s.isValid() and s.width() > width:
            r.setScaledSize(s.scaled(width, 100000, Qt.AspectRatioMode.KeepAspectRatio))
        img = r.read()
        out.append((str(f), None if img.isNull() else img))
    return out


# ----------------------------------------------------------------------------- widgets
class FlowGrid(QWidget):
    """Equal-width columns: as many as fit (each at least `min_w` wide), at most `max_cols`.

    Items are packed densely in reading order; an item may span n x n cells (e.g. a featured figure).
    """

    def __init__(self, min_w=240, max_cols=4, spacing=12, parent=None):
        super().__init__(parent)
        self.min_w, self.max_cols, self.spacing = min_w, max_cols, spacing
        self.items: list[tuple[QWidget, int]] = []
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(spacing)
        self._cols = 0

    def add(self, w: QWidget, span: int = 1):
        w.setMinimumWidth(min(w.minimumSizeHint().width(), self.min_w - 60))
        self.items.append((w, span))
        self._relayout()

    def widgets(self):
        return [w for w, _ in self.items]

    def clear(self):
        for w, _ in self.items:
            self.grid.removeWidget(w)
            w.hide()
            w.deleteLater()
        self.items = []

    def _columns(self):
        w = self.width() if self.width() > 50 else self.min_w * self.max_cols
        return max(1, min(self.max_cols, (w + self.spacing) // (self.min_w + self.spacing)))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._columns() != self._cols:
            self._relayout()

    def _relayout(self):
        cols = self._columns()
        self._cols = cols
        for w, _ in self.items:
            self.grid.removeWidget(w)
        occ = set()

        def free(r, c, n):
            return c + n <= cols and all((r + i, c + j) not in occ for i in range(n) for j in range(n))

        for w, span in self.items:
            n = max(1, min(span, cols - 1 if cols > 2 else 1))
            r = c = 0
            while not free(r, c, n):
                c += 1
                if c >= cols:
                    c, r = 0, r + 1
            occ.update((r + i, c + j) for i in range(n) for j in range(n))
            self.grid.addWidget(w, r, c, n, n)
        for c in range(self.max_cols):
            self.grid.setColumnStretch(c, 1 if c < cols else 0)


class Elided(QLabel):
    """Single-line label that elides its text to the available width instead of forcing a minimum width."""

    def __init__(self, text="", obj=None, mode=Qt.TextElideMode.ElideMiddle):
        super().__init__()
        self._full, self._mode = text, mode
        if obj:
            self.setObjectName(obj)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(40)
        self.setText(text)

    def setText(self, text):
        self._full = text
        self._elide()

    def full_text(self):
        return self._full

    def sizeHint(self):                 # the full text's width (only used when the size policy is not Ignored)
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + 4, super().sizeHint().height())

    def minimumSizeHint(self):
        return QSize(40, super().minimumSizeHint().height())

    def _elide(self):
        super().setText(self.fontMetrics().elidedText(self._full, self._mode, max(40, self.width())))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._elide()


class _Image(QWidget):
    """Paints a pixmap scaled to fit, on a white rounded plate (the figures have white backgrounds)."""

    def __init__(self, height=118, parent=None):
        super().__init__(parent)
        self.pm: QPixmap | None = None
        self.msg = "载入中…"
        self.setFixedHeight(height)
        self.setMinimumWidth(120)

    def set_image(self, img: QImage | None, msg="无法读取图像"):
        self.pm = QPixmap.fromImage(img) if img is not None else None
        self.msg = msg
        self.update()

    def paintEvent(self, e):
        t = ThemeManager.instance().tokens
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(r, 8, 8)
        p.fillPath(path, QColor("#ffffff") if self.pm is not None else QColor(t.surface2))
        p.setPen(QColor(t.border))
        p.drawPath(path)
        if self.pm is None:
            p.setPen(QColor(t.muted))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.msg)
            return
        box = r.adjusted(6, 6, -6, -6)
        s = QSize(self.pm.width(), self.pm.height()).scaled(int(box.width()), int(box.height()),
                                                             Qt.AspectRatioMode.KeepAspectRatio)
        x = box.x() + (box.width() - s.width()) / 2
        y = box.y() + (box.height() - s.height()) / 2
        p.drawPixmap(QRectF(x, y, s.width(), s.height()), self.pm, QRectF(self.pm.rect()))


class QuickCard(ClickableCard):
    def __init__(self, pid, icon, title, desc, runtime):
        super().__init__()
        self.pid, self.icon_name = pid, icon
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(10)
        self.ic = QLabel()
        self.ic.setObjectName("iconBox")
        self.ic.setFixedSize(34, 34)
        self.ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(self.ic)
        top.addWidget(label(title, "cardTitle"), 1)
        lay.addLayout(top)
        d = label(desc, "muted", wrap=True)
        d.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        lay.addWidget(d, 1)
        row = QHBoxLayout()
        self.rt = label("⏱ " + runtime, "badge")
        row.addWidget(self.rt)
        row.addStretch(1)
        lay.addLayout(row)
        self.setMinimumHeight(132)
        self.setToolTip(f"打开「{title}」")
        self.clicked.connect(lambda: nav.navigate(self.pid))

    def retheme(self, t):
        self.ic.setPixmap(icons.pixmap(self.icon_name, t.accent, 18))


class ThumbCard(ClickableCard):
    def __init__(self, path: Path, caption: str, featured=False):
        super().__init__()
        self.path = path
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)
        self.img = _Image()
        if featured:
            self.img.setMinimumHeight(118)
            self.img.setMaximumHeight(16777215)
            self.img.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay.addWidget(self.img, 1)
        crow = QHBoxLayout()
        crow.setSpacing(8)
        cap = Elided(caption, "cardTitle", Qt.TextElideMode.ElideRight)
        if not featured:
            cap.setStyleSheet("font-size: 13px;")
        crow.addWidget(cap, 1)
        if featured:
            crow.addWidget(label("核心结果", "badge"))
        lay.addLayout(crow)
        mt = _mtime(path)
        when = time.strftime("%Y-%m-%d", time.localtime(mt)) if mt else ""
        lay.addWidget(Elided(f"{path.name} · {when}", "muted", Qt.TextElideMode.ElideMiddle))
        self.setToolTip(f"{paths.rel(path)}\n点击在「结果浏览」中打开")
        self.clicked.connect(lambda: nav.navigate("results", path=str(self.path)))


class JobRow(QFrame):
    clicked = Signal(str)

    def __init__(self, job):
        super().__init__()
        self.job_id = job.id
        self.setObjectName("jobRow")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 7, 10, 7)
        lay.setSpacing(10)
        self.ic = QLabel()
        self.ic.setFixedSize(18, 18)
        lay.addWidget(self.ic)
        col = QVBoxLayout()
        col.setSpacing(1)
        self.title = QLabel()
        self.title.setTextFormat(Qt.TextFormat.PlainText)
        self.title.setMinimumWidth(80)
        self.title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.meta = label("", "muted")
        self.meta.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        col.addWidget(self.title)
        col.addWidget(self.meta)
        lay.addLayout(col, 1)
        right = QVBoxLayout()
        right.setSpacing(3)
        self.status = QLabel()
        self.status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        right.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedSize(96, 5)
        right.addWidget(self.bar, 0, Qt.AlignmentFlag.AlignRight)
        lay.addLayout(right)
        self.update_job(job)

    def update_job(self, job):
        t = ThemeManager.instance().tokens
        self.ic.setPixmap(icons.pixmap(KIND_ICON.get(job.kind, "file"), t.muted, 18))
        fm = self.title.fontMetrics()
        w = max(120, self.title.width())
        self.title.setText(fm.elidedText(job.title, Qt.TextElideMode.ElideRight, w))
        self.title.setToolTip(job.title)
        kind = KIND_ZH.get(job.kind, job.kind)
        if job.status == RUNNING:
            when = f"已运行 {fmt_elapsed(job.elapsed())}"
        elif job.status == QUEUED:
            when = f"排队于 {time.strftime('%H:%M', time.localtime(job.created))}"
        else:
            when = (f"用时 {fmt_elapsed(job.elapsed())} · " if job.started else "") + ago(job.finished or job.created)
        self.meta.setText(f"{kind} · {when}")
        col = status_color(job.status, t)
        txt = STATUS_ZH.get(job.status, job.status)
        if job.status == RUNNING and job.progress is not None:
            txt += f" {100 * job.progress:.0f}%"
        self.status.setText(f"<span style='color:{col}; font-weight:600'>● {txt}</span>")
        running = job.status == RUNNING
        self.bar.setVisible(running)
        if running:
            if job.progress is None:
                self.bar.setRange(0, 0)
            else:
                self.bar.setRange(0, 1000)
                self.bar.setValue(int(1000 * job.progress))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        j = next((j for j in JobManager.instance().jobs if j.id == self.job_id), None)
        if j is not None:
            self.title.setText(self.title.fontMetrics().elidedText(j.title, Qt.TextElideMode.ElideRight, self.title.width()))

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit(self.job_id)
        super().mouseReleaseEvent(e)


# ----------------------------------------------------------------------------- page
class HomePage(Page):
    title = "首页"
    subtitle = ""

    def __init__(self):
        super().__init__(scroll=True)
        self.header.hide()
        self.mgr = JobManager.instance()
        self._started = False
        self._gallery_sig = None
        self._gallery_gen = 0
        self._rows: dict[str, JobRow] = {}
        self.plot = None
        self.measured = None

        # ---------------- hero
        self.hero = QFrame()
        self.hero.setObjectName("hero")
        hl = QHBoxLayout(self.hero)
        hl.setContentsMargins(26, 22, 26, 22)
        hl.setSpacing(22)
        self.logo = QLabel()
        self.logo.setFixedSize(76, 76)
        icon_path = paths.ASSETS / "icon.png"
        if icon_path.exists():
            pm = QPixmap(str(icon_path))
            if not pm.isNull():
                pm = pm.scaled(152, 152, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                pm.setDevicePixelRatio(2.0)
                self.logo.setPixmap(pm)
        hl.addWidget(self.logo, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(6)
        trow = QHBoxLayout()
        trow.setSpacing(10)
        name = label("LTEM Sim", "heroTitle")
        trow.addWidget(name)
        trow.addWidget(label("v1.0", "badge"), 0, Qt.AlignmentFlag.AlignVCenter)
        trow.addStretch(1)
        self.btn_docs = button("阅读文档", lambda: nav.navigate("docs"), "flat")
        self.btn_folder = button("打开项目文件夹", lambda: reveal(paths.ROOT), "flat")
        trow.addWidget(self.btn_docs)
        trow.addWidget(self.btn_folder)
        col.addLayout(trow)
        col.addWidget(label("SI-GaAs 光导间隙的近场太赫兹发射（LTEM）模拟 · 桌面版", "heroSubtitle"))
        self._summary_html = readme_summary()
        self.summary = label("", "heroText", wrap=True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        col.addWidget(self.summary)
        col.addSpacing(4)
        info = QGridLayout()
        info.setHorizontalSpacing(14)
        info.setVerticalSpacing(5)
        self.info_vals = {}
        for i, (key, text) in enumerate((("repo", "仓库"), ("git", "Git"), ("env", "环境"))):
            k = label(text, "infoKey")
            info.addWidget(k, i, 0, Qt.AlignmentFlag.AlignTop)
            v = label("…", None, wrap=True, selectable=True)
            v.setTextFormat(Qt.TextFormat.RichText)
            info.addWidget(v, i, 1)
            self.info_vals[key] = v
        info.setColumnStretch(1, 1)
        self.info_vals["repo"].setText(html.escape(str(paths.ROOT)))
        col.addLayout(info)
        hl.addLayout(col, 1)
        self.body.addWidget(self.hero)

        # ---------------- quick start
        self.body.addWidget(self._section("快速开始", "点击进入对应模块；时间为本机上的典型运行时长"))
        self.quick = FlowGrid(min_w=230, max_cols=4)
        self.quick_cards = [QuickCard(*q) for q in QUICK]
        for c in self.quick_cards:
            self.quick.add(c)
        self.body.addWidget(self.quick)

        # ---------------- recent jobs | measured
        row = QHBoxLayout()
        row.setSpacing(12)
        jobs_card = QFrame()
        jobs_card.setObjectName("card")
        jl = QVBoxLayout(jobs_card)
        jl.setContentsMargins(14, 12, 14, 12)
        jl.setSpacing(6)
        jh = QHBoxLayout()
        jh.addWidget(label("最近任务", "cardTitle"), 1)
        self.jobs_count = label("", "muted")
        jh.addWidget(self.jobs_count)
        jh.addWidget(button("任务中心 →", lambda: nav.navigate("jobs"), "flat"))
        jl.addLayout(jh)
        self.rows_box = QVBoxLayout()
        self.rows_box.setSpacing(2)
        jl.addLayout(self.rows_box)
        self.jobs_empty = label("还没有运行过任务。从上方的「快速开始」选择一个模块，开始第一次模拟。", "muted", wrap=True)
        self.jobs_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.jobs_empty.setMinimumHeight(120)
        jl.addWidget(self.jobs_empty)
        jl.addStretch(1)
        row.addWidget(jobs_card, 3)

        meas_card = QFrame()
        meas_card.setObjectName("card")
        ml = QVBoxLayout(meas_card)
        ml.setContentsMargins(14, 12, 14, 8)
        ml.setSpacing(4)
        mh = QHBoxLayout()
        mh.addWidget(label("实测数据", "cardTitle"), 1)
        mh.addWidget(button("查看 →", lambda: nav.navigate("measured"), "flat"))
        ml.addLayout(mh)
        self.meas_note = label("±25 V 近场 RMS 幅度 vs 光斑-针尖位置 X（data/measured_tip_scan.csv）", "muted", wrap=True)
        ml.addWidget(self.meas_note)
        self.plot_box = QVBoxLayout()
        self.plot_placeholder = label("载入中…", "muted")
        self.plot_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_placeholder.setFixedHeight(250)
        self.plot_box.addWidget(self.plot_placeholder)
        ml.addLayout(self.plot_box)
        row.addWidget(meas_card, 2)
        self.body.addLayout(row)

        # ---------------- gallery
        gh = self._section("关键结果", "仓库中已提交的结果图；点击在「结果浏览」中打开")
        self.body.addWidget(gh)
        self.gallery = FlowGrid(min_w=250, max_cols=4)
        self.body.addWidget(self.gallery)
        self.gallery_empty = label("results/ 中还没有结果图。运行模拟或在「分析与作图」中生成幻灯片图后会显示在这里。", "muted", wrap=True)
        self.gallery_empty.hide()
        self.body.addWidget(self.gallery_empty)
        self.body.addStretch(1)

        self._rebuild_jobs()
        self.mgr.job_added.connect(lambda *_: self._rebuild_jobs())
        self.mgr.jobs_reset.connect(self._rebuild_jobs)
        self.mgr.job_changed.connect(self._on_job_changed)
        self._retheme()
        ThemeManager.instance().changed.connect(self._retheme)

    # ------------------------------------------------------------------ building blocks
    def _section(self, title, hint):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(2, 10, 2, 0)
        l.setSpacing(10)
        l.addWidget(label(title, "homeSection"))
        h = label(hint, "muted")
        l.addWidget(h, 1, Qt.AlignmentFlag.AlignBottom)
        return w

    def _retheme(self, *_):
        t = ThemeManager.instance().tokens
        self.setStyleSheet(f"""
            QFrame#hero {{ border: 1px solid {t.border}; border-radius: 14px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {t.accent_soft}, stop:0.65 {t.surface}, stop:1 {t.surface}); }}
            QLabel#heroTitle {{ font-size: 28px; font-weight: 800; color: {t.text}; }}
            QLabel#heroSubtitle {{ font-size: 14px; font-weight: 600; color: {t.accent}; }}
            QLabel#heroText {{ font-size: 13px; color: {t.text}; }}
            QLabel#infoKey {{ color: {t.muted}; font-weight: 600; }}
            QLabel#homeSection {{ font-size: 16px; font-weight: 700; }}
            QLabel#iconBox {{ background: {t.accent_soft}; border-radius: 9px; }}
            QFrame#jobRow {{ border-radius: 8px; background: transparent; }}
            QFrame#jobRow:hover {{ background: {t.surface2}; }}
        """)
        self.summary.setText(style_code(self._summary_html, t.accent))
        for c in self.quick_cards:
            c.retheme(t)
        for r in self._rows.values():
            j = self._job(r.job_id)
            if j is not None:
                r.update_job(j)
        if self._started:
            self._fill_env()
            if hasattr(self, "_git") and self._git is not None:
                self._fill_git(self._git)

    # ------------------------------------------------------------------ lifecycle
    def on_show(self):
        if not self._started:
            self._started = True
            self._fill_env()
            run_async(git_info, on_done=self._fill_git, on_error=lambda *_: self._fill_git({}))
            QTimer.singleShot(30, self._build_plot)
        self.refresh_gallery()
        for r in self._rows.values():
            j = self._job(r.job_id)
            if j is not None:
                r.update_job(j)

    def _fill_env(self):
        t = ThemeManager.instance().tokens
        e = env_info()
        parts = [f"Python {e['python']}"]
        np_v = e.get("numpy")
        try:
            ok = np_v is not None and int(np_v.split(".")[0]) >= 2
        except ValueError:
            ok = False
        if ok:
            parts.append(f"numpy {np_v} <span style='color:{t.success}; font-weight:600'>✓</span>")
        else:
            parts.append(f"<span style='color:{t.danger}; font-weight:600'>numpy {np_v or '未安装'} ✗ 需要 ≥ 2.0</span>")
        for mod in ("scipy", "matplotlib"):
            v = e.get(mod)
            parts.append(f"{mod} {v}" if v else f"<span style='color:{t.danger}'>{mod} 未安装</span>")
        parts.append(f"{e['cpus']} 核 CPU")
        self.info_vals["env"].setText(" &nbsp;·&nbsp; ".join(parts))
        self.info_vals["env"].setToolTip(f"解释器：{e['exe']}")

    def _fill_git(self, g):
        self._git = g
        t = ThemeManager.instance().tokens
        v = self.info_vals["git"]
        if not g:
            v.setText(f"<span style='color:{t.muted}'>不可用（不是 git 仓库或未安装 git）</span>")
            return
        s = (f"<b>{html.escape(g['branch'])}</b> &nbsp;·&nbsp; <span style='font-family: Menlo, monospace'>{g['hash']}</span> "
             f"{html.escape(g['subject'])} &nbsp;·&nbsp; <span style='color:{t.muted}'>{ago(g['time'])}</span>")
        if g.get("dirty"):
            s += f" &nbsp;·&nbsp; <span style='color:{t.warning}; white-space:nowrap'>{g['dirty']} 个文件未提交</span>"
        v.setText(s)

    # ------------------------------------------------------------------ measured plot
    def _build_plot(self):
        try:
            self.measured = read_measured()
        except OSError:
            self.measured = None
        if not self.measured:
            self.plot_placeholder.setText("未找到实测数据 data/measured_tip_scan.csv")
            return
        from ..widgets.mpl import PlotWidget
        self.plot = PlotWidget(toolbar=False, figsize=(5, 2.8))
        self.plot.setFixedHeight(250)
        self.plot_box.replaceWidget(self.plot_placeholder, self.plot)
        self.plot_placeholder.deleteLater()
        self.plot.plot(self._draw_measured)

    def _draw_measured(self, fig):
        t = ThemeManager.instance().tokens
        ax = fig.add_subplot(111)
        for V, pts in sorted(self.measured.items(), reverse=True):
            X, y, e = zip(*pts)
            ax.errorbar(X, y, yerr=e, marker="o", ms=4.5, lw=1.5, capsize=2.5, elinewidth=1, label=f"{V:+.0f} V")
        ax.axvline(0, color=t.muted, lw=0.8, ls=":")
        ax.set_xlabel("X = spot − tip (µm)")
        ax.set_ylabel("RMS (norm.)")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.5, lw=0.6)
        ax.legend(frameon=False, loc="upper left")

    # ------------------------------------------------------------------ recent jobs
    def _job(self, jid):
        return next((j for j in self.mgr.jobs if j.id == jid), None)

    def _recent(self):
        return sorted(self.mgr.jobs, key=lambda j: j.created or 0, reverse=True)[:6]

    def _rebuild_jobs(self):
        for r in self._rows.values():
            self.rows_box.removeWidget(r)
            r.hide()
            r.deleteLater()
        self._rows = {}
        for j in self._recent():
            r = JobRow(j)
            r.clicked.connect(lambda jid: nav.navigate("jobs", job_id=jid))
            self.rows_box.addWidget(r)
            self._rows[j.id] = r
        self.jobs_empty.setVisible(not self._rows)
        self._update_count()

    def _on_job_changed(self, job):
        r = self._rows.get(job.id)
        if r is not None:
            r.update_job(job)
            self._update_count()

    def _update_count(self):
        run = len(self.mgr.running())
        q = len(self.mgr.active()) - run
        parts = []
        if run:
            parts.append(f"{run} 个运行中")
        if q:
            parts.append(f"{q} 个排队")
        self.jobs_count.setText(" · ".join(parts) if parts else (f"共 {len(self.mgr.jobs)} 个" if self.mgr.jobs else ""))

    # ------------------------------------------------------------------ gallery
    def _gallery_files(self):
        seen, out = set(), []
        for rel, cap in GALLERY:
            p = paths.ROOT / rel
            if p.exists():
                out.append((p, cap))
                seen.add(p.name)
        for p in sorted((paths.RESULTS / "slides").glob("sim_*.png")):
            if p.name not in seen:
                out.append((p, p.stem.replace("sim_", "").replace("_", " ")))
        return out

    def refresh_gallery(self):
        files = self._gallery_files()
        sig = tuple((str(p), _mtime(p)) for p, _ in files)
        if sig == self._gallery_sig:
            return
        self._gallery_sig = sig
        self._gallery_gen += 1
        self.gallery.clear()
        self._thumbs = {}
        for i, (p, cap) in enumerate(files):
            c = ThumbCard(p, cap, featured=(i == 0))
            self.gallery.add(c, 2 if i == 0 else 1)
            self._thumbs[str(p)] = c
        self.gallery.setVisible(bool(files))
        self.gallery_empty.setVisible(not files)
        if files:
            # the featured (first) figure spans 2x2 cells: decode it at a higher resolution
            gen = self._gallery_gen
            run_async(load_thumbs, [(p, 1500 if i == 0 else 720) for i, (p, _) in enumerate(files)],
                      on_done=lambda res: self._set_thumbs(res, gen))

    def _set_thumbs(self, res, gen=None):
        if gen is not None and gen != self._gallery_gen:
            return                          # a newer refresh is loading (or has loaded) fresher pixels
        for path, img in res:
            c = self._thumbs.get(path)
            if c is not None:
                try:
                    c.img.set_image(img)
                except RuntimeError:        # card deleted by a later refresh
                    pass

    def open(self, **_):
        pass
