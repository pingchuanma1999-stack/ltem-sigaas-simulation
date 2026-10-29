"""Per-tip-position chain, route 2 (scripts/run_tip_positions.py: simulate → demod).

For every beam position X and every apex height h one PulseSim with the tip as an equipotential body in Poisson
(optionally with the per-height FDTD pump map) records Q_tip(t); demod projects Q_tip(h(θ), t) on cos nθ.
"""
from __future__ import annotations

import csv
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetrics, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QBoxLayout, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                               QScrollArea, QSizePolicy, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)

from .. import icons, paths
from ..jobs import CANCELLED, FAILED, FINISHED, PREREQ_FAILED, RUNNING, STATUS_TEXT, JobManager
from ..paramspec import FieldSpec
from ..theme import ThemeManager, repolish
from ..widgets.common import MetricTile, Page, button, card, error_box, label, reveal
from ..widgets.device import C_INK, device_from_pulse, draw_device
from ..widgets.fileviews import FlowLayout
from ..widgets.imageview import ImageView
from ..widgets.mpl import PlotWidget
from ..widgets.paramform import ParamForm
from ..widgets.runpanel import RunPanel, fmt_elapsed
from ..workers import run_async

SCRIPT = "scripts/run_tip_positions.py"
# the script's defaults (not imported: it calls matplotlib.use("Agg") and imports pyplot at import time)
HEIGHTS_NM = (10, 15, 25, 45, 80, 130, 200, 270, 330, 370)
X_LIST = (-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5)
COMMITTED = paths.RESULTS / "tip_positions"
FDTD_HEIGHTS = paths.RESULTS / "fdtd_pump" / "heights"
FDTD_REF = paths.RESULTS / "fdtd_pump" / "notip.npz"
RUN_RE = re.compile(r"^X=([+-]\d+\.\d) h=\s*(\d+) nm:")
HARMONICS = (1, 2, 3, 4)
FDTD_SECONDS = (75.0, 90.0)          # one run_fdtd_heights height on this machine
DEFAULT_SERIES = {"det1", "det2", "det3", "det4", "det3_d1", "far", "meas+", "meas-"}
NARROW_PX = 620                      # plots narrower than this get a compact layout (legend below, short titles)
DEV_NARROW_PX = 680                  # device diagram narrower than this: panels stacked, short caption
DEV_MIN_H = 520                      # below this the Device tab scrolls instead of shrinking the diagram


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def run_seconds(t_end_ps: float):
    """(low, high) wall time of one PulseSim incl. setup: measured 2 s per 0.5 ps run and 17–20 s per 4 ps run here,
    32–40 s per 4 ps run in the original logs."""
    return 0.5 + 4.2 * t_end_ps, 2.0 + 12.0 * t_end_ps


def _dur(s: float) -> str:
    if s < 90:
        return f"{max(1, round(s))} s"
    if s < 5400:
        return f"{round(s / 60)} min"
    return f"{s / 3600:.1f} h"


def fmt_range(lo: float, hi: float) -> str:
    a, b = _dur(lo), _dur(hi)
    if a.split()[1] == b.split()[1]:
        return a if a == b else f"{a.split()[0]}–{b}"
    return f"{a} – {b}"


def npz_name(X, h) -> str:
    return f"X{float(X):+.1f}_h{int(h)}.npz"


def fx(X) -> str:
    """Signed beam position for display ('0' rather than '+0')."""
    return "0" if float(X) == 0 else f"{float(X):+g}"


def xkey(X) -> str:
    return f"{float(X):+.1f}"


def read_json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def demod_name(A, h_min) -> str:
    return f"demod_A{float(A):g}_h{float(h_min):g}.json"


def demod_files(d: Path):
    def key(p):
        m = re.match(r"demod_A([\d.]+)_h([\d.]+)\.json", p.name)
        return (float(m.group(1)), float(m.group(2))) if m else (1e9, 1e9)
    return sorted(Path(d).glob("demod_A*_h*.json"), key=key)


def npz_count(d: Path, xs, hs) -> int:
    return sum((Path(d) / npz_name(X, h)).exists() for X in xs for h in hs)


def _inside(d, root: Path) -> bool:
    try:
        Path(d).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def is_committed(d: Path) -> bool:
    """A hand-curated (committed) result folder: anywhere under results/ except results/gui_runs/."""
    return _inside(d, paths.RESULTS) and not _inside(d, paths.GUI_RUNS)


def fdtd_ref_for(folder: Path) -> Path:
    """The no-tip reference the ratio maps in `folder` were computed with (its gui_run.json), else notip.npz."""
    for name in ("gui_run.json", "gui_run_fdtd_heights_summary.json"):
        ref = ((read_json(Path(folder) / name) or {}).get("kwargs") or {}).get("ref")
        if ref:
            return paths.absolute(ref)
    return FDTD_REF


def quarantine_ratio(folder: Path):
    """A run_fdtd_heights height killed while being written leaves a broken ratio_h*.npz that both scripts would
    treat as done: rename the newest one to *.npz.partial when its zip is damaged. Returns the bad name or None."""
    files = sorted(Path(folder).glob("ratio_h*.npz"), key=lambda p: p.stat().st_mtime) if Path(folder).is_dir() else []
    if not files:
        return None
    f = files[-1]
    try:
        with zipfile.ZipFile(f) as z:
            if z.testzip() is None:
                return None
    except (zipfile.BadZipFile, OSError):
        pass
    f.rename(f.with_suffix(".npz.partial"))
    return f.name


class _HomeOnBlur(QObject):
    """Show the start of long tuple fields (X list, heights) whenever they are not being edited."""

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.FocusOut and isinstance(obj, QLineEdit):
            obj.setCursorPosition(0)
        return False


_MEASURED = None


def measured():
    """data/measured_tip_scan.csv → {bias: (X, rms, err)} (parsed here: importing run_baseline pulls in pyplot)."""
    global _MEASURED
    if _MEASURED is None:
        rows = {}
        try:
            with open(paths.MEASURED_CSV, newline="") as fh:
                for r in csv.DictReader(fh):
                    rows.setdefault(round(float(r["bias_V"])), []).append(
                        (float(r["X_um"]), float(r["rms"]), float(r["rms_err"])))
        except (OSError, ValueError, KeyError):
            rows = {}
        _MEASURED = {b: tuple(np.array(c) for c in zip(*v)) for b, v in rows.items()}
    return _MEASURED


def sim_specs():
    S, G = "Pump and bias (simulate)", "Scan grid (simulate)"
    return [
        FieldSpec("P_avg_mW", 2.0, "float", "Average pump power", "mW",
                  "80 MHz repetition rate, 3 µm spot, 820 nm. The repository has results for 2 mW and 30 mW", S, minimum=0.0),
        FieldSpec("V_bias", 25.0, "float", "Gap bias V_bias", "V", "The anode is on the −x side", S),
        FieldSpec("V_tip", 0.0, "float", "Tip potential V_tip", "V",
                  "Relative to the gap midpoint; 0 = grounded tip. P2mW_sym_Vtip12 / P2mW_sym_Vtipm12 use +12.5 / −12.5 V", S),
        FieldSpec("T", 300.0, "float", "Temperature T", "K", "Carrier and lattice temperature (sets kT/q diffusion and N_c)", S,
                  minimum=1.0),
        FieldSpec("degenerate", True, "bool", "Degenerate diffusion (Fermi–Dirac)", "",
                  "Generalised Einstein relation; off = classical Boltzmann diffusion", S),
        FieldSpec("x_list", X_LIST, "tuple", "Beam positions X", "µm",
                  "Beam centre relative to the tip, comma-separated; file names use X{X:+.1f}", G),
        FieldSpec("heights", HEIGHTS_NM, "tuple", "Tip heights h", "nm",
                  "Tip–sample distances over the tapping cycle, one full run per height; file names use int(h)", G),
        FieldSpec("t_end_ps", 4.0, "float", "Simulated time t_end", "ps",
                  "dt = 0.5 fs, recorded every 5 fs; the run time scales with t_end", G, minimum=0.05),
    ]


def demod_specs():
    D = "Demodulation (demod)"
    return [
        FieldSpec("A_nm", 180.0, "float", "Tapping amplitude A", "nm",
                  "h(θ) = h_min + A(1 + cos θ); h_min + 2A should not exceed the largest simulated height", D, minimum=0.0),
        FieldSpec("h_min_nm", 10.0, "float", "Lowest height h_min", "nm", "Bottom of the tapping cycle", D, minimum=0.0),
        FieldSpec("f_cut_thz", 2.2, "float", "Detector cut-off", "THz", "Steep 8th-order Butterworth low-pass", D, minimum=0.01),
        FieldSpec("deriv", "2", "choice", "Detected derivative order", "",
                  "2 = radiated field ∝ d²S/dt² (current convention); 1 = the old first-order convention", D,
                  choices=("2", "1")),
        FieldSpec("band", (0.3, 2.0), "tuple", "Amplitude band", "THz", "Frequency band (f_lo, f_hi) of band_metric", D,
                  advanced=True),
    ]


# ----------------------------------------------------------------------------- background work
def _load_wave_data(d, xs, hs):
    """Q_tip(t) of every X*_h*.npz present → {"t": {xk: t}, "Q": {xk: {int h: Q}}}."""
    t, Q = {}, {}
    for X in xs:
        row = {}
        for h in hs:
            f = Path(d) / npz_name(X, h)
            if not f.exists():
                continue
            try:
                with np.load(f) as z:
                    row[int(h)] = np.asarray(z["Q_tip"], float)
                    t.setdefault(xkey(X), np.asarray(z["t"], float))
            except Exception:        # a file truncated by a cancelled run
                continue
        if row:
            Q[xkey(X)] = row
    return {"t": t, "Q": Q}


def _wave_work(d, xs, hs, dem, data=None):
    from ltemsim.tip_demod import demodulate_positions, detected_waveform
    if data is None:
        data = _load_wave_data(d, xs, hs)
    det = {}
    for xk, row in data["Q"].items():
        if not all(int(h) in row for h in hs):
            continue
        t = data["t"][xk]
        Q = np.array([row[int(h)] for h in hs])
        S = demodulate_positions(list(hs), Q, A_nm=dem["A_nm"], h_min_nm=dem["h_min_nm"], harmonics=HARMONICS)
        det[xk] = {n: detected_waveform(t, S[n], order=int(dem["deriv"]), f_cut_thz=dem["f_cut_thz"]) for n in HARMONICS}
    return data, det


# ----------------------------------------------------------------------------- small widgets
class _TileRow(QWidget):
    """Metric tiles in one row when there is room, otherwise in rows of three."""

    def __init__(self, tiles, parent=None):
        super().__init__(parent)
        self._tiles = list(tiles)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(8)
        self._cols = 0
        self._arrange(len(self._tiles))

    def _arrange(self, cols):
        if cols == self._cols:
            return
        for c in range(max(self._cols, cols)):
            self._grid.setColumnStretch(c, 1 if c < cols else 0)
        self._cols = cols
        for i, t in enumerate(self._tiles):
            self._grid.addWidget(t, i // cols, i % cols)      # re-adding moves the widget to its new cell

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._arrange(len(self._tiles) if e.size().width() >= 118 * len(self._tiles) else 3)


class _Split(QSplitter):
    """Prefers its minimum height: the page holder (a scroll area that asks for heightForWidth) should only scroll when
    the window really is too small, not because the plots and the form would like to be taller."""

    def sizeHint(self):
        return QSize(super().sizeHint().width(), self.minimumSizeHint().height())


class _FigTabs(QTabWidget):
    """Plot tabs that take whatever height is left: neither the canvas' 7x4.5 in size hint nor the wrapping series
    checkboxes (height-for-width) may make the results scroll area taller than the window when everything fits."""

    def hasHeightForWidth(self):
        return False

    def sizeHint(self):
        return QSize(super().sizeHint().width(), self.minimumHeight())


class RunGrid(QWidget):
    """X × h matrix of the simulate runs: finished / running / pending."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.xs, self.hs, self.done, self.running = [], [], set(), None
        self.setMinimumSize(250, 150)

    def set_state(self, xs, hs, done, running=None):
        self.xs, self.hs, self.done, self.running = list(xs), list(hs), set(done), running
        self.updateGeometry()
        self.update()

    def _legend(self, t):
        return ((f"Done {len(self.done)}/{len(self.xs) * len(self.hs)}", t.accent, None),
                ("Running", t.accent_soft, t.warning), ("Pending", t.surface2, t.muted))

    def sizeHint(self):
        f = self.font()
        f.setPixelSize(10)
        fm = QFontMetrics(f)
        legend = 46 + sum(fm.horizontalAdvance(txt) + 30 for txt, *_ in self._legend(ThemeManager.instance().tokens))
        return QSize(max(58 + 26 * max(1, len(self.hs)), legend), 44 + 20 * max(1, len(self.xs)))

    def paintEvent(self, _e):
        t = ThemeManager.instance().tokens
        nx, nh = len(self.xs), len(self.hs)
        if not nx or not nh:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        f = self.font()
        f.setPixelSize(10)
        p.setFont(f)
        lw, top, bottom = 46.0, 16.0, 22.0
        cw = min(26.0, (self.width() - lw) / nh)
        ch = min(20.0, (self.height() - top - bottom) / nx)
        muted, text = QColor(t.muted), QColor(t.text)
        p.setPen(muted)
        p.drawText(QRectF(0, 0, lw - 4, top), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "X \\ h")
        for j, h in enumerate(self.hs):
            p.drawText(QRectF(lw + j * cw, 0, cw, top), Qt.AlignmentFlag.AlignCenter, f"{float(h):g}")
        for i, X in enumerate(self.xs):
            p.setPen(text)
            p.drawText(QRectF(0, top + i * ch, lw - 6, ch), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, fx(X))
            for j in range(nh):
                r = QRectF(lw + j * cw + 2, top + i * ch + 2, cw - 4, ch - 4)
                if (i, j) in self.done:
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QColor(t.accent))
                elif (i, j) == self.running:
                    p.setPen(QPen(QColor(t.warning), 1.5))
                    p.setBrush(QColor(t.accent_soft))
                else:
                    edge = QColor(t.muted)
                    edge.setAlpha(110)
                    p.setPen(QPen(edge, 1))
                    p.setBrush(QColor(t.surface2))
                p.drawRoundedRect(r, 3, 3)
        y = top + nx * ch + 6
        x = lw
        for txt, fill, pen in self._legend(t):
            p.setPen(QPen(QColor(pen), 1.5) if pen else Qt.PenStyle.NoPen)
            p.setBrush(QColor(fill))
            p.drawRoundedRect(QRectF(x, y + 3, 10, 10), 2, 2)
            p.setPen(muted)
            w = p.fontMetrics().horizontalAdvance(txt)
            p.drawText(QRectF(x + 14, y, w + 4, 16), Qt.AlignmentFlag.AlignVCenter, txt)
            x += w + 30
        p.end()


class _Step(QFrame):
    """One step of the job chain: status badge, name, elapsed / progress; click → show its log."""
    clicked = Signal()

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click to show this step's log")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 3, 2, 3)
        lay.setSpacing(8)
        self.badge = label("—", "badge")
        self.badge.setMinimumWidth(52)
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name = label(title)
        self.info = label("", "muted")
        lay.addWidget(self.badge)
        lay.addWidget(self.name, 1)
        lay.addWidget(self.info)

    def set_job(self, job, focused):
        self.badge.setText(STATUS_TEXT.get(job.status, job.status))
        self.badge.setObjectName({FINISHED: "badgeOk", FAILED: "badgeErr", CANCELLED: "badgeWarn"}.get(job.status, "badge"))
        repolish(self.badge)
        self.name.setText(f"<b>{self.title}</b>" if focused else self.title)
        info = fmt_elapsed(job.elapsed()) if job.started else ""
        if job.status == RUNNING and job.progress_text:
            info = f"{job.progress_text} · {info}"
        elif job.status == CANCELLED and job.progress_text == PREREQ_FAILED:
            info = PREREQ_FAILED
        self.info.setText(info)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(e)


# ----------------------------------------------------------------------------- page
class TipPositionsPage(Page):
    title = "Tip-position chain"
    subtitle = ("Route 2: one PulseSim per beam position X and tip height h (tip as an equipotential body, optional FDTD "
                "pump shadow) records the tip charge Q_tip(t); demodulation along h(θ) = h_min + A(1 + cos θ) gives S_n(t) "
                "for comparison with the measured ±25 V scans. Default: 70 runs, ≈ 25–60 min.")

    def __init__(self):
        super().__init__()
        self.chain: dict = {}
        self.result_dir: Path | None = None
        self.meta: dict = {}
        self.demod: dict | None = None
        self.n_expected = self.n_present = 0
        self._wave = None              # (key, data, det)
        self._wave_gen = 0
        self._fdtd_extra: list[str] = []
        self._narrow = {"amp": None, "wave": None, "dev": None}
        self._dev_dirty = True         # the device diagram is drawn when it is first shown
        self._dev_timer = QTimer(self)                 # debounced redraw of the device diagram on form edits
        self._dev_timer.setSingleShot(True)
        self._dev_timer.setInterval(300)
        self._dev_timer.timeout.connect(self._redraw_device)
        jm = JobManager.instance()

        self.demod_btn = button("Demodulate only", self.redemod)
        self.header.add_action(self.demod_btn)
        self.run_btn = button("  Simulate + demodulate", self.run, "primary")
        self.header.add_action(self.run_btn)

        split = self.split = _Split(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._build_left())
        self.tabs = QTabWidget()
        self.dev_tab = self._build_device_tab()
        self.run_tab = self._build_run_tab()
        self.res_tab = self._build_results_tab()
        self.tabs.addTab(self.dev_tab, "Device")
        self.tabs.addTab(self.run_tab, "Run")
        self.tabs.addTab(self.res_tab, "Results")
        self.tabs.addTab(self._build_history_tab(), "History")
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([430, 900])
        self._user_split = False             # until the user drags the handle, the form column follows the window width
        split.splitterMoved.connect(lambda *_: setattr(self, "_user_split", True))
        split.installEventFilter(self)
        self.body.addWidget(split, 1)

        jm.job_changed.connect(self._on_job_changed)
        jm.job_finished.connect(self._on_job_finished)
        self.run_btn.setIcon(icons.icon("play", ThemeManager.instance().tokens.on_accent))
        self._refresh_fdtd_choices()
        self.refresh_history()
        default = COMMITTED / "P2mW_sym"
        if default.is_dir():
            self.show_result(default, switch=False)
        else:
            self._update_demod_btn()
        ThemeManager.instance().changed.connect(self._retheme)

    # ------------------------------------------------------------------ layout
    def _build_left(self):
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)
        self.form = ParamForm(sim_specs() + demod_specs())
        self.form.changed.connect(self._update_plan)
        self.form.changed.connect(self._home_tuples)
        self.form.changed.connect(self._schedule_device)
        self._blur = _HomeOnBlur(self)
        for k in ("x_list", "heights"):
            self.form.rows[k].editor.installEventFilter(self._blur)
        self._home_tuples()
        ll.addWidget(self.form, 1)

        fd = card()
        fl = QVBoxLayout(fd)
        fl.setContentsMargins(12, 10, 12, 10)
        fl.setSpacing(6)
        fl.addWidget(label("Pump shadow maps from FDTD (fdtd)", "cardTitle"))
        row = QHBoxLayout()
        self.fdtd_combo = QComboBox()
        self.fdtd_combo.setToolTip("none = symmetric Gaussian pump; or a folder with ratio_h<h>.npz maps "
                                   "(output of scripts/run_fdtd_heights.py)")
        self.fdtd_combo.currentIndexChanged.connect(self._update_plan)
        self.fdtd_combo.currentIndexChanged.connect(self._schedule_device)
        row.addWidget(self.fdtd_combo, 1)
        row.addWidget(button("Choose…", self._pick_fdtd, tooltip="Choose a folder with ratio_h*.npz maps"))
        fl.addLayout(row)
        self.fdtd_status = label("", wrap=True)
        self.fdtd_status.setTextFormat(Qt.TextFormat.RichText)
        fl.addWidget(self.fdtd_status)
        ll.addWidget(fd)

        orow = QHBoxLayout()
        orow.addWidget(QLabel("Output folder"))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Automatic: results/gui_runs/tip_<time>")
        self.out_edit.setToolTip("Leave empty to create a new folder for every run. Pick an existing folder to resume: "
                                 "the script skips X*_h*.npz files that already exist")
        self.out_edit.textChanged.connect(self._update_plan)
        orow.addWidget(self.out_edit, 1)
        b = QPushButton("Choose…")
        b.clicked.connect(self._pick_out)
        orow.addWidget(b)
        ll.addLayout(orow)
        left.setMinimumWidth(370)
        return left

    def _build_device_tab(self):
        w = QWidget()
        dl = QVBoxLayout(w)
        dl.setContentsMargins(8, 8, 8, 8)
        dl.setSpacing(6)
        self.dev_plot = PlotWidget(figsize=(8, 4.6))
        self.dev_plot.setMinimumHeight(DEV_MIN_H)
        self.dev_plot.installEventFilter(self)
        self.dev_plot.canvas.installEventFilter(self)       # the wheel over the canvas scrolls the tab (small windows)
        self.dev_scroll = sc = QScrollArea()                # small windows: scroll rather than shrink the diagram
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(self.dev_plot)
        dl.addWidget(sc, 1)
        self.dev_note = label("", "muted", wrap=True)
        self.dev_note.setTextFormat(Qt.TextFormat.RichText)
        dl.addWidget(self.dev_note)
        return w

    def _build_run_tab(self):
        w = QWidget()
        rl = QVBoxLayout(w)
        rl.setContentsMargins(12, 12, 12, 12)
        rl.setSpacing(10)
        plan = card()
        pg = QGridLayout(plan)
        pg.setContentsMargins(16, 12, 16, 12)
        pg.setHorizontalSpacing(14)
        pg.setVerticalSpacing(6)
        pg.addWidget(label("Run plan (what “Simulate + demodulate” will do)", "cardTitle"), 0, 0, 1, 2)
        self.plan = {}
        for i, (k, text) in enumerate((("size", "Size"), ("time", "Estimated time"), ("pump", "Pump"), ("out", "Output"))):
            pg.addWidget(label(text, "muted"), i + 1, 0, Qt.AlignmentFlag.AlignTop)
            v = label("", wrap=True, selectable=True)
            v.setTextFormat(Qt.TextFormat.RichText)
            pg.addWidget(v, i + 1, 1)
            self.plan[k] = v
        self.plan_warn = label("", wrap=True)
        self.plan_warn.setTextFormat(Qt.TextFormat.RichText)
        pg.addWidget(self.plan_warn, 5, 0, 1, 2)
        pg.setColumnStretch(1, 1)
        rl.addWidget(plan)

        self.chain_card = card()
        cl = self._chain_lay = QHBoxLayout(self.chain_card)     # steps above the run grid when narrow (eventFilter)
        cl.setContentsMargins(16, 12, 16, 12)
        cl.setSpacing(18)
        col = QVBoxLayout()
        col.setSpacing(4)
        col.addWidget(label("Current job chain", "cardTitle"))
        self.steps = {}
        for k, title in (("fdtd", "① FDTD height maps · run_fdtd_heights"), ("sim", "② Simulation · simulate"),
                         ("demod", "③ Demodulation · demod")):
            st = _Step(title)
            st.clicked.connect(lambda k=k: self._focus_step(k))
            col.addWidget(st)
            self.steps[k] = st
        self.last_line = label("", "muted", wrap=True, selectable=True)
        col.addWidget(self.last_line)
        col.addStretch(1)
        cl.addLayout(col, 1)
        self.grid_w = RunGrid()
        cl.addWidget(self.grid_w)
        self.chain_card.setVisible(False)
        self.chain_card.installEventFilter(self)
        rl.addWidget(self.chain_card)

        self.panel = RunPanel()
        self.panel.title.setWordWrap(True)       # long job titles must not widen the whole right-hand pane
        self.panel.setMinimumHeight(240)
        rl.addWidget(self.panel, 1)
        sc = QScrollArea()                       # small windows: scroll instead of clipping the plan / chain cards
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        return sc

    def _build_results_tab(self):
        w = QWidget()
        xl = QVBoxLayout(w)
        xl.setContentsMargins(12, 12, 12, 12)
        xl.setSpacing(8)
        sc = QScrollArea()                 # small windows: scroll instead of squeezing the plots to nothing
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setWidget(w)
        top = QHBoxLayout()
        self.res_title = label("No result loaded", "cardTitle", wrap=True)
        self.res_title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        top.addWidget(self.res_title, 1)
        self.btn_params = button("Load into form", self._load_result_params, "flat",
                                 "Fill the form with this result's meta.json and the selected demodulation settings "
                                 "(the output folder is unchanged)")
        self.btn_resume = button("Resume this folder", self._resume, "flat",
                                 "Load the parameters and set the output folder to this folder; existing npz files are skipped")
        self.btn_redemod = button("Re-demodulate with form settings", self.redemod, "flat")
        self.btn_folder = button("Show in Finder", lambda: self.result_dir and reveal(self.result_dir), "flat")
        for b in (self.btn_params, self.btn_folder):
            top.addWidget(b)
        xl.addLayout(top)
        self.res_note = label("", "muted", wrap=True)
        self.res_note.setVisible(False)
        xl.addWidget(self.res_note)

        self.tiles = {k: MetricTile(text, "—") for k, text in (
            ("P", "Pump power"), ("Vb", "Bias V_bias"), ("Vt", "Tip V_tip"), ("pump", "Pump map"),
            ("h", "Heights h"), ("X", "Positions X"))}
        for tile in self.tiles.values():         # long values (folder names) must not widen the results pane
            for w_ in (tile.lab, tile.val, tile.hint):
                w_.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        xl.addWidget(_TileRow(self.tiles.values()))

        drow = QWidget()                     # wraps the buttons below the selector in narrow windows
        dflow = FlowLayout(drow, spacing=8)
        sel = QWidget()
        sl = QHBoxLayout(sel)
        sl.setContentsMargins(0, 0, 8, 0)
        sl.addWidget(label("Demodulation", "muted"))
        self.demod_combo = QComboBox()
        self.demod_combo.setMinimumWidth(210)
        self.demod_combo.currentIndexChanged.connect(self._on_demod_selected)
        sl.addWidget(self.demod_combo)
        dflow.addWidget(sel)
        dflow.addWidget(self.btn_resume)
        dflow.addWidget(self.btn_redemod)
        xl.addWidget(drow)
        self.demod_desc = label("", "muted", wrap=True)
        self.demod_desc.setTextFormat(Qt.TextFormat.RichText)
        xl.addWidget(self.demod_desc)

        self.fig_tabs = _FigTabs()
        self.fig_tabs.addTab(self._build_amp_tab(), "Amplitude vs X")
        png = QWidget()
        pl = QVBoxLayout(png)
        pl.setContentsMargins(6, 6, 6, 6)
        self.png_view = ImageView()
        if self.png_view.caption is not None:      # a long file-name caption must not widen the whole results pane
            self.png_view.caption.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        pl.addWidget(self.png_view, 1)
        note = label("The file name tip_positions_A{A}.png does not contain h_min: demodulations with the same A and a "
                     "different h_min overwrite this figure (the latest one is shown).", "muted", wrap=True)
        note.setTextFormat(Qt.TextFormat.PlainText)
        pl.addWidget(note)
        self.fig_tabs.addTab(png, "Script PNG")
        self.fig_tabs.addTab(self._build_wave_tab(), "Waveforms")
        self.fig_tabs.addTab(self._build_table_tab(), "Table")
        self.fig_tabs.setMinimumHeight(450)
        self.fig_tabs.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        xl.addWidget(self.fig_tabs, 1)
        return sc

    def _build_amp_tab(self):
        w = QWidget()
        al = QVBoxLayout(w)
        al.setContentsMargins(8, 8, 8, 4)
        al.setSpacing(4)
        self.series = {}
        groups = (
            ("Detected d^kS_n/dt^k  n =", [(f"det{n}", str(n), f"det{n}: band amplitude of the detected waveform "
                                            f"d^kS_{n}/dt^k (k = detected derivative order)") for n in HARMONICS]),
            ("First-order dS_n/dt  n =", [(f"det{n}_d1", str(n), f"det{n}_d1: band amplitude with the first-derivative "
                                           f"convention dS_{n}/dt") for n in HARMONICS]),
            ("", [("far", "Far field dJ/dt", "far: band amplitude of the far-field reference dJ/dt")]),
            ("Peak |S_n|  n =", [(f"S{n}", str(n), f"S{n}: peak of the demodulated signal S_{n}(t)") for n in HARMONICS]),
            ("", [("meas+", "Measured +25 V", "data/measured_tip_scan.csv, bias = +25 V"),
                  ("meas-", "Measured −25 V (X→−X)", "bias = −25 V, mirrored X → −X")]),
        )
        self.series_box = QWidget()
        flow = FlowLayout(self.series_box, spacing=6)        # groups wrap as a whole in narrow windows
        flow.setContentsMargins(0, 0, 0, 0)

        def group():
            g = QWidget()
            gl = QHBoxLayout(g)
            gl.setContentsMargins(0, 0, 18, 0)
            gl.setSpacing(6)
            flow.addWidget(g)
            return gl
        for gname, items in groups:
            gl = group()
            if gname:
                gl.addWidget(label(gname, "muted"))
            for key, text, tip in items:
                c = QCheckBox(text)
                c.setToolTip(tip)
                c.setChecked(key in DEFAULT_SERIES)
                c.toggled.connect(self._redraw_amp)
                gl.addWidget(c)
                self.series[key] = c
        self.abs_box = QCheckBox("Raw rms · log axis")
        self.abs_box.setToolTip("Plot the raw rms (the quantities have different units and compare only within a group); "
                                "measured points are shown only in normalised mode")
        self.abs_box.toggled.connect(self._redraw_amp)
        group().addWidget(self.abs_box)
        al.addWidget(self.series_box)
        self.amp_stack = QStackedWidget()
        self.amp_empty = label("", "muted", wrap=True)
        self.amp_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.amp_stack.addWidget(self.amp_empty)
        self.amp_plot = PlotWidget()
        self.amp_plot.installEventFilter(self)
        self.amp_stack.addWidget(self.amp_plot)
        al.addWidget(self.amp_stack, 1)
        return w

    def _build_wave_tab(self):
        w = QWidget()
        wl = QVBoxLayout(w)
        wl.setContentsMargins(8, 8, 8, 4)
        wl.setSpacing(4)
        ctl = QWidget()
        flow = FlowLayout(ctl, spacing=16)
        self.wave_x = QComboBox()
        self.wave_x.currentIndexChanged.connect(lambda *_: self._redraw_wave())
        self.wave_n = QComboBox()
        self.wave_n.addItems([str(n) for n in HARMONICS])
        self.wave_n.setCurrentText("3")
        self.wave_n.currentIndexChanged.connect(lambda *_: self._redraw_wave())
        for text, combo in (("(a) Beam position X", self.wave_x), ("(b) Harmonic n", self.wave_n)):
            g = QWidget()
            gl = QHBoxLayout(g)
            gl.setContentsMargins(0, 0, 0, 0)
            gl.setSpacing(6)
            gl.addWidget(label(text, "muted"))
            gl.addWidget(combo)
            flow.addWidget(g)
        self.wave_status = label("", "muted")
        self.wave_status.setMinimumHeight(self.wave_x.sizeHint().height())
        flow.addWidget(self.wave_status)
        wl.addWidget(ctl)
        self.wave_stack = QStackedWidget()
        self.wave_empty = label("", "muted", wrap=True)
        self.wave_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wave_stack.addWidget(self.wave_empty)
        self.wave_plot = PlotWidget()
        self.wave_plot.installEventFilter(self)
        self.wave_stack.addWidget(self.wave_plot)
        wl.addWidget(self.wave_stack, 1)
        return w

    def _build_table_tab(self):
        w = QWidget()
        tl = QVBoxLayout(w)
        tl.setContentsMargins(8, 8, 8, 8)
        row = QHBoxLayout()
        self.table_note = label("Each column normalised to its maximum (rms_norm)", "muted", wrap=True)
        row.addWidget(self.table_note, 1)
        row.addWidget(button("Copy as TSV", self._copy_table, "flat", "Copy to the clipboard, ready to paste into Excel / Origin"))
        tl.addLayout(row)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        tl.addWidget(self.table, 1)
        return w

    def _build_history_tab(self):
        w = QWidget()
        hl = QVBoxLayout(w)
        hl.setContentsMargins(12, 12, 12, 12)
        hrow = QHBoxLayout()
        hrow.addWidget(label("Tip-position chain results (double-click to load)", "cardTitle", wrap=True), 1)
        hrow.addWidget(button("Open another folder…", self._open_other, "flat"))
        hrow.addWidget(button("Refresh", self.refresh_history, "flat"))
        hl.addLayout(hrow)
        hl.addWidget(label("The repository's results/tip_positions/* hold only meta.json, the demodulation JSON and PNGs "
                           "(the raw X*_h*.npz are git-ignored); GUI runs are in results/gui_runs/tip_*.", "muted", wrap=True))
        self.hist = QListWidget()
        self.hist.setSpacing(2)
        self.hist.itemDoubleClicked.connect(self._hist_open)
        hl.addWidget(self.hist, 1)
        return w

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.Resize:
            if obj is self.chain_card:
                d = QBoxLayout.Direction.TopToBottom if obj.width() < 560 else QBoxLayout.Direction.LeftToRight
                if self._chain_lay.direction() != d:
                    self._chain_lay.setDirection(d)
            elif obj is self.split and not self._user_split:
                QTimer.singleShot(0, self._fit_split)
            elif obj is self.amp_plot and self._narrow["amp"] not in (None, obj.width() < NARROW_PX):
                self._narrow["amp"] = None
                QTimer.singleShot(0, self._redraw_amp)
            elif obj is self.wave_plot and self._narrow["wave"] not in (None, obj.width() < NARROW_PX):
                self._narrow["wave"] = None
                QTimer.singleShot(0, self._redraw_wave)
            elif obj is self.dev_plot and self._narrow["dev"] not in (None, self._dev_layout()):
                self._narrow["dev"] = None
                QTimer.singleShot(0, self._redraw_device)
        elif ev.type() == QEvent.Type.Show and obj is self.dev_plot and self._dev_dirty:
            QTimer.singleShot(0, self._redraw_device)
        elif ev.type() == QEvent.Type.Wheel and obj is self.dev_plot.canvas:
            sb = self.dev_scroll.verticalScrollBar()
            if sb.maximum() > 0:
                d = ev.pixelDelta().y() or round(ev.angleDelta().y() / 120 * 3 * sb.singleStep())
                sb.setValue(sb.value() - d)
                return True
        return super().eventFilter(obj, ev)

    def _fit_split(self):
        """Form column: 390 px in wide windows, 380 px (the narrowest without a horizontal scroll bar) in small ones."""
        tot = sum(self.split.sizes())
        if tot > 0 and not self._user_split:
            left = int(min(390, max(380, tot - 520)))
            if self.split.sizes()[0] != left:
                self.split.setSizes([left, tot - left])

    def _retheme(self, *_):
        t = ThemeManager.instance().tokens
        self.run_btn.setIcon(icons.icon("play", t.on_accent))
        self._update_plan()
        self._update_note_colors()
        self.grid_w.update()

    # ------------------------------------------------------------------ device diagram
    def _schedule_device(self, *_):
        self._dev_timer.start()

    def _dev_layout(self):
        """(stacked, narrow): the two panels go one above the other in tall or narrow plot areas (side by side the
        to-scale cross-section would be a thin strip), and narrow plots get a short caption."""
        w, h = self.dev_plot.width(), self.dev_plot.height()
        return (w < DEV_NARROW_PX or h > 0.5 * w), w < DEV_NARROW_PX

    def _redraw_device(self):
        """Cross-section for the current form values; while hidden it is only marked stale and drawn on show."""
        self._dev_timer.stop()
        if not self.dev_plot.isVisible():
            self._dev_dirty = True
            return
        self._dev_dirty = False
        v = self.form.values()                      # unparseable fields fall back to their defaults
        xs = [float(x) for x in v["x_list"]]
        hs = sorted(float(h) for h in v["heights"])
        A, hm, Vt = float(v["A_nm"]), float(v["h_min_nm"]), float(v["V_tip"])
        dev = device_from_pulse(float(v["V_bias"]), P_avg_mW=float(v["P_avg_mW"]), x_list=xs, V_tip=Vt,
                                tip_h_um=max(hs[0] if hs else hm, 0.0) / 1000)
        stacked, narrow = self._narrow["dev"] = self._dev_layout()
        pump = "FDTD pump maps (tip shadow)" if self._fdtd_value() != "none" else "symmetric Gaussian pump"
        top = hm + 2 * A
        tap = [f"tapping h = {hm:g} … {top:g} nm", f"(clipped at {hs[-1]:g} nm)" if hs and top > hs[-1] + 1e-9 else "",
               f"A = {A:g} nm · V_tip = {Vt:g} V"]
        dev.model = (f"tip-position chain · {pump}" if narrow else
                     f"tip-position chain: hydrodynamic carriers + Poisson, tip as an equipotential body · {pump}")

        def draw(fig):
            ax, az = draw_device(fig, dev)
            if stacked:
                gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15])
                ax.set_subplotspec(gs[0])
                az.set_subplotspec(gs[1])
            x1, z1 = az.get_xlim()[1], az.get_ylim()[1]
            az.text(0.07 * x1, 0.58 * z1, "\n".join(s for s in tap if s), ha="left", va="center", fontsize=7.5,
                    color=C_INK, zorder=7)
        self.dev_plot.plot(draw)

    # ------------------------------------------------------------------ form values
    def _values(self):
        """(simulate kwargs without out/fdtd, demod kwargs without out); raises ValueError with a readable message."""
        if not self.form.is_valid():
            raise ValueError("These values cannot be parsed: " + ", ".join(self.form.invalid_fields()))
        v = self.form.values()
        sim = {k: v[k] for k in ("P_avg_mW", "degenerate", "T", "V_bias", "V_tip", "t_end_ps")}
        sim["degenerate"] = bool(sim["degenerate"])
        sim["heights"] = list(v["heights"])
        sim["x_list"] = [float(x) for x in v["x_list"]]
        dem = dict(A_nm=float(v["A_nm"]), h_min_nm=float(v["h_min_nm"]), f_cut_thz=float(v["f_cut_thz"]),
                   deriv=int(v["deriv"]), band=[float(b) for b in v["band"]])
        return sim, dem

    def _problems(self, sim, dem):
        """(blocking errors, warnings) for the grid / tapping range."""
        errs, warns = [], []
        xs, hs = sim["x_list"], sim["heights"]
        if not xs:
            errs.append("The list of beam positions X is empty")
        if not hs:
            errs.append("The list of tip heights is empty")
        if any(h <= 0 for h in hs):
            errs.append("Tip heights must be positive")
        if len({int(h) for h in hs}) != len(hs):
            errs.append("Two heights have the same int(h), e.g. 12.5 and 12 (file names X…_h<int(h)>.npz would collide)")
        if len({xkey(X) for X in xs}) != len(xs):
            errs.append("Two X values are equal to one decimal place (file names X{X:+.1f} would collide)")
        errs += self._demod_errors(dem)
        if hs and dem["h_min_nm"] + 2 * dem["A_nm"] > max(hs) + 1e-9:
            warns.append(f"The tapping trajectory reaches h_min + 2A = {dem['h_min_nm'] + 2 * dem['A_nm']:g} nm, above the "
                         f"largest simulated height {max(hs):g} nm: higher points take that height's value (clipped)")
        if hs and dem["h_min_nm"] < min(hs) - 1e-9:
            warns.append(f"h_min = {dem['h_min_nm']:g} nm is below the smallest simulated height {min(hs):g} nm: "
                         "lower points take that height's value")
        return errs, warns

    @staticmethod
    def _demod_errors(dem):
        errs = []
        if dem["A_nm"] <= 0:
            errs.append("The tapping amplitude A must be positive (with A = 0, S_n is numerical noise and the normalised "
                        "curves are meaningless)")
        if len(dem["band"]) != 2 or dem["band"][0] >= dem["band"][1]:
            errs.append("The amplitude band must be two increasing numbers (f_lo, f_hi)")
        return errs

    def _home_tuples(self):
        for k in ("x_list", "heights"):
            e = self.form.rows[k].editor
            if not e.hasFocus():
                e.setCursorPosition(0)

    def _out_dir(self):
        txt = self.out_edit.text().strip()
        return paths.absolute(txt) if txt else None

    def _new_out(self) -> Path:
        """A fresh results/gui_runs/tip_<time> (a chain queued in the same second has not created its folder yet)."""
        base = out = paths.new_run_dir("tip")
        i = 2
        while self._busy(out) or out.exists():
            out = base.with_name(f"{base.name}_{i}")
            i += 1
        return out

    def _fdtd_value(self) -> str:
        return self.fdtd_combo.currentData() or "none"

    @staticmethod
    def _missing_ratio(folder: Path, hs):
        return [h for h in hs if not (folder / f"ratio_h{int(h)}.npz").exists()]

    @staticmethod
    def _active_fdtd_jobs(folder: Path):
        out = []
        for j in JobManager.instance().active():
            if j.kind == "fdtd_heights" and j.spec.get("script", "").endswith("run_fdtd_heights.py"):
                o = j.spec.get("kwargs", {}).get("out", "results/fdtd_pump/heights")
                if paths.absolute(o).resolve() == folder.resolve():
                    out.append(j)
        return out

    def _covering_fdtd_job(self, folder: Path, missing):
        """The queued/running run_fdtd_heights job into `folder` that will produce every missing map, else None."""
        running = self._active_fdtd_jobs(folder)
        covered = {int(h) for j in running for h in j.spec.get("kwargs", {}).get("heights", HEIGHTS_NM)}
        return running[-1] if running and all(int(h) in covered for h in missing) else None

    @staticmethod
    def _busy(folder: Path):
        """Active tip jobs writing to this folder."""
        r = folder.resolve()
        return [j for j in JobManager.instance().active()
                if j.kind in ("tip_sim", "tip_demod") and j.out and paths.absolute(j.out).resolve() == r]

    # ------------------------------------------------------------------ plan
    def _update_plan(self, *_):
        t = ThemeManager.instance().tokens
        warn = lambda s: f"<span style='color:{t.warning}'>{s}</span>"
        ok = lambda s: f"<span style='color:{t.success}'>{s}</span>"
        try:
            sim, dem = self._values()
        except ValueError as e:
            for k in self.plan:
                self.plan[k].setText("—")
            self.plan_warn.setText(f"<span style='color:{t.danger}'>{e}</span>")
            self.plan_warn.setVisible(True)
            self.fdtd_status.setText("")
            self.dev_note.setText(warn(f"⚠ {e}. The diagram shows their default values."))
            return
        xs, hs = sim["x_list"], sim["heights"]
        total = len(xs) * len(hs)
        out = self._out_dir()
        present = npz_count(out, xs, hs) if out and out.is_dir() else 0
        remaining = total - present
        xr = f"{fx(min(xs))} … {fx(max(xs))} µm" if xs else "—"
        hr = f"{min(hs):g}–{max(hs):g} nm" if hs else "—"
        size = (f"{plural(len(xs), 'beam position')} ({xr}) × {plural(len(hs), 'height')} ({hr}) = "
                f"<b>{total}</b> PulseSim runs, t_end = {sim['t_end_ps']:g} ps each")
        if present:
            size += f"; {present} already in the output folder are skipped, <b>{remaining} remaining</b>"
        self.plan["size"].setText(size)
        lo, hi = run_seconds(sim["t_end_ps"])
        tl, th = remaining * lo + 3, remaining * hi + 6

        fdtd = self._fdtd_value()
        extra = ""
        if fdtd == "none":
            self.plan["pump"].setText("none — symmetric Gaussian pump (no shadowing of the pump by the tip)")
            self.fdtd_status.setText("<span>Symmetric Gaussian pump, no FDTD tip maps.</span>")
        else:
            folder = paths.absolute(fdtd)
            missing = self._missing_ratio(folder, hs)
            running = self._covering_fdtd_job(folder, missing) is not None if missing else False
            if not missing:
                s = short = ok(f"✓ ratio_h*.npz ready for all {plural(len(hs), 'selected height')}")
            else:
                hl = ", ".join(f"{h:g}" for h in missing[:6]) + (" …" if len(missing) > 6 else "")
                where = (" This is a repository folder: the maps are computed into a new "
                         "results/gui_runs/fdtd_heights_&lt;time&gt; and it stays untouched." if is_committed(folder) else "")
                s = warn(f"⚠ ratio_h*.npz missing for {len(missing)}/{len(hs)} heights (h = {hl}). "
                         + ("An FDTD heights job is already running or queued; the simulation will wait for it." if running
                            else "The script would wait forever for missing files; on run you are asked to queue the "
                                 "FDTD maps first or to switch to none." + where))
                short = warn(f"⚠ {len(missing)}/{len(hs)} ratio maps missing, " + (
                    "queued behind the running FDTD job" if running else
                    "you will be asked before running: queue the FDTD maps, or use none"))
                if not running:
                    n_fdtd = len(hs) if is_committed(folder) else len(missing)
                    fl, fh = n_fdtd * FDTD_SECONDS[0], n_fdtd * FDTD_SECONDS[1]
                    tl, th = tl + fl, th + fh
                    extra = f", incl. FDTD height maps ≈ {fmt_range(fl, fh)}"
            self.fdtd_status.setText(s)
            self.plan["pump"].setText(f"FDTD tip maps {paths.rel(folder)} — " + short)
        per = f"≈ {lo:.0f}–{hi:.0f} s per run" if remaining else "nothing to simulate"
        self.plan["time"].setText(f"<b>≈ {fmt_range(tl, th)}</b> ({per}{extra}) + demodulation ≈ 1–5 s")
        if out is None:
            self.plan["out"].setText("results/gui_runs/tip_&lt;time&gt; (a new folder for every run)")
        else:
            o = paths.rel(out)
            if is_committed(out):
                o += " — " + warn("repository result folder: meta.json will be overwritten")
            elif present:
                o += f" — resume ({present} npz present)"
            self.plan["out"].setText(o)
        errs, warns = self._problems(sim, dem)
        lines = [f"<span style='color:{t.danger}'>✕ {e}</span>" for e in errs] + [warn("⚠ " + w) for w in warns]
        self.plan_warn.setText("<br>".join(lines))
        self.plan_warn.setVisible(bool(lines))
        self.dev_note.setText(
            f"<b>{total} PulseSim runs</b> ({len(xs)} X × {len(hs)} h), ≈ {fmt_range(tl, th)}. The tip stays at x = 0; the "
            "beam centre moves to each X (green markers). The nm-scale apex gap is exaggerated in (b)."
            + ("" if not errs else "<br>" + "<br>".join(f"<span style='color:{t.danger}'>✕ {e}</span>" for e in errs)))

    def _refresh_fdtd_choices(self):
        cur = self.fdtd_combo.currentData() or "none"
        self.fdtd_combo.blockSignals(True)
        self.fdtd_combo.clear()
        self.fdtd_combo.addItem("none — symmetric Gaussian pump", "none")
        cands = [FDTD_HEIGHTS] + sorted((p for p in paths.GUI_RUNS.glob("fdtd_heights_*") if p.is_dir()), reverse=True)
        cands += [paths.absolute(p) for p in self._fdtd_extra]
        seen = set()
        for p in cands:
            r = paths.rel(p)
            if r in seen:
                continue
            seen.add(r)
            n = len(list(p.glob("ratio_h*.npz"))) if p.is_dir() else 0
            self.fdtd_combo.addItem(f"{r} · {plural(n, 'map')}", r)
            self.fdtd_combo.setItemData(self.fdtd_combo.count() - 1, f"{r}\n{n} ratio_h*.npz", Qt.ItemDataRole.ToolTipRole)
        i = self.fdtd_combo.findData(cur)
        self.fdtd_combo.setCurrentIndex(max(0, i))
        self.fdtd_combo.blockSignals(False)
        self._update_plan()

    def _set_fdtd(self, value):
        value = value or "none"
        if value != "none" and self.fdtd_combo.findData(value) < 0:
            self._fdtd_extra.append(value)
            self._refresh_fdtd_choices()
        self.fdtd_combo.setCurrentIndex(max(0, self.fdtd_combo.findData(value)))

    def _pick_fdtd(self):
        d = QFileDialog.getExistingDirectory(self, "Choose a folder with ratio_h*.npz maps", str(FDTD_HEIGHTS.parent))
        if d:
            self._set_fdtd(paths.rel(d))

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Choose the output folder",
                                             str(paths.GUI_RUNS if paths.GUI_RUNS.exists() else paths.RESULTS))
        if d:
            self.out_edit.setText(paths.rel(d))

    # ------------------------------------------------------------------ dialogs (overridable in tests)
    def _confirm(self, title, text) -> bool:
        return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes

    def _error(self, title, text):
        error_box(self, title, text)

    def _ask_existing(self, out: Path, diffs, present, remaining) -> str:
        """'go' | 'new' | 'cancel' for an output folder that already holds npz files."""
        m = QMessageBox(self)
        m.setWindowTitle("Output folder already has results")
        if diffs:
            m.setIcon(QMessageBox.Icon.Warning)
            m.setText(f"{paths.rel(out)} already holds {present} npz files computed with different parameters:")
            m.setInformativeText("\n".join(diffs) + "\n\nThe script skips every X…_h….npz that exists, so old and new results "
                                 "would be mixed. A new output folder is recommended.")
            b_new = m.addButton("Use a new folder", QMessageBox.ButtonRole.AcceptRole)
            b_go = m.addButton("Resume anyway", QMessageBox.ButtonRole.DestructiveRole)
        else:
            m.setIcon(QMessageBox.Icon.Question)
            m.setText(f"{paths.rel(out)} already holds {present} npz files (same parameters).")
            m.setInformativeText(f"The script skips them and computes only the remaining {remaining} (resume).")
            b_go = m.addButton("Resume", QMessageBox.ButtonRole.AcceptRole)
            b_new = m.addButton("Use a new folder", QMessageBox.ButtonRole.ActionRole)
        m.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        m.exec()
        c = m.clickedButton()
        return "go" if c is b_go else "new" if c is b_new else "cancel"

    def _ask_fdtd(self, folder: Path, missing, sim_time: str, target: Path, todo, ref: Path) -> str:
        """'fdtd' | 'none' | 'cancel' when ratio maps are missing (`todo` heights would be computed into `target`)."""
        hl = ", ".join(f"{h:g}" for h in missing)
        m = QMessageBox(self)
        m.setIcon(QMessageBox.Icon.Warning)
        m.setWindowTitle("FDTD tip maps missing")
        m.setText(f"{paths.rel(folder)} has no ratio_h*.npz for {plural(len(missing), 'height')} (h = {hl} nm).")
        info = ("In fdtd mode run_tip_positions.py waits for these files indefinitely and prints nothing, so they must be "
                "computed first.\n\n")
        b_fdtd = None
        n = len(todo)
        if ref.exists():
            where = paths.rel(target)
            if target != folder:
                where = (f"the new folder {paths.rel(target)} ({paths.rel(folder)} is a repository result folder whose "
                         f"committed summary.json run_fdtd_heights would overwrite, so all {n} heights are computed in the "
                         f"new folder instead)")
            info += (f"• Compute the FDTD maps first: queue scripts/run_fdtd_heights.py ({n} heights × 75–90 s ≈ "
                     f"{fmt_range(n * FDTD_SECONDS[0], n * FDTD_SECONDS[1])}) writing to {where}; then summary.json is "
                     f"rebuilt from all ratio maps and simulation + demodulation start (simulation ≈ {sim_time}).\n")
        else:
            info += (f"• The FDTD maps cannot be computed first: the reference field {paths.rel(ref)} is missing. Run the "
                     f"no-tip FDTD on the Pump FDTD (optics) page first.\n")
        info += "• Use none: symmetric Gaussian pump (no shadowing of the pump by the tip)."
        m.setInformativeText(info)
        if ref.exists():
            b_fdtd = m.addButton("Compute FDTD maps first", QMessageBox.ButtonRole.AcceptRole)
        b_none = m.addButton("Use none", QMessageBox.ButtonRole.ActionRole)
        m.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        m.exec()
        c = m.clickedButton()
        return "fdtd" if (b_fdtd is not None and c is b_fdtd) else "none" if c is b_none else "cancel"

    # ------------------------------------------------------------------ run
    @staticmethod
    def _meta_diff(meta, sim, fdtd):
        if not meta:
            return []
        out = []
        for k in ("P_avg_mW", "V_bias", "V_tip", "T", "t_end_ps"):
            a, b = meta.get(k), sim[k]
            if a is not None and abs(float(a) - float(b)) > 1e-9:
                out.append(f"{k}: {float(a):g} → {float(b):g}")
        if "degenerate" in meta and bool(meta["degenerate"]) != sim["degenerate"]:
            out.append(f"degenerate: {bool(meta['degenerate'])} → {sim['degenerate']}")
        if (meta.get("fdtd") or "none") != fdtd:
            out.append(f"fdtd: {meta.get('fdtd') or 'none'} → {fdtd}")
        return out

    def run(self):
        try:
            sim, dem = self._values()
        except ValueError as e:
            self._error("Invalid parameters", str(e))
            return
        errs, _ = self._problems(sim, dem)
        if errs:
            self._error("Cannot run", "\n".join(errs))
            return
        xs, hs = sim["x_list"], sim["heights"]
        total = len(xs) * len(hs)
        fdtd = self._fdtd_value()
        out = self._out_dir()
        if out is None:
            out = self._new_out()
        if self._busy(out):
            self._error("Folder in use", f"{paths.rel(out)} already has a running or queued tip-position job. Wait for it "
                                         "to finish or choose another output folder.")
            return
        present = npz_count(out, xs, hs) if out.is_dir() else 0
        if present:
            diffs = self._meta_diff(read_json(out / "meta.json"), sim, fdtd)
            c = self._ask_existing(out, diffs, present, total - present)
            if c == "cancel":
                return
            if c == "new":
                out = self._new_out()
                self.out_edit.clear()
                present = 0
        if is_committed(out) and not self._confirm(
                "Write into a repository result?", f"{paths.rel(out)} is a committed result folder: the simulation "
                                                   "overwrites its meta.json and the demodulation its JSON and PNG. Continue?"):
            return
        remaining = total - present
        lo, hi = run_seconds(sim["t_end_ps"])
        sim_time = fmt_range(remaining * lo + 3, remaining * hi + 6)

        jm = JobManager.instance()
        fdtd_job = None
        asked = False
        if fdtd != "none":
            folder = paths.absolute(fdtd)
            missing = self._missing_ratio(folder, hs)
            running = self._active_fdtd_jobs(folder)
            covering = self._covering_fdtd_job(folder, missing) if missing else None
            if covering is not None:
                fdtd_job = covering              # wait for the FDTD job that is already producing these maps
            elif missing:
                asked = True
                # never let run_fdtd_heights rewrite a committed summary.json: compute into a fresh GUI folder instead
                target = paths.new_run_dir("fdtd_heights") if is_committed(folder) else folder
                todo = list(hs) if target != folder else list(missing)
                ref = fdtd_ref_for(folder)
                c = self._ask_fdtd(folder, missing, sim_time, target, todo, ref)
                if c == "cancel":
                    return
                if c == "none":
                    fdtd = "none"
                    self._set_fdtd("none")
                else:
                    fdtd_job = jm.submit(
                        f"FDTD height maps · {plural(len(todo), 'height')} · {target.name}", "fdtd_heights",
                        "scripts/run_fdtd_heights.py",
                        "main", kwargs=dict(out=paths.rel(target), heights=todo, ref=paths.rel(ref)),
                        out=target, progress={"type": "count", "pattern": r"^h = \S+ nm: ", "total": len(todo)},
                        after=running[-1] if (running and target == folder) else None)
                    jm.submit(f"FDTD height maps · rebuild summary.json · {target.name}", "fdtd_heights", "ltemsim_gui/gui_tasks.py",
                              "fdtd_heights_summary", kwargs=dict(out=paths.rel(target), ref=paths.rel(ref)),
                              out=target, after=fdtd_job)
                    if target != folder:
                        fdtd = paths.rel(target)
                        self._set_fdtd(fdtd)
        if not asked and hi * remaining > 15 * 60 and not self._confirm(
                "Start a long run?", f"This runs {remaining} PulseSim simulations, ≈ {sim_time}.\nResults go to "
                                     f"{paths.rel(out)}. You can stop at any time in the Job center; finished npz files are "
                                     "kept and the run can be resumed later.\n\nContinue?"):
            return

        rel = paths.rel(out)
        pump = "FDTD maps" if fdtd != "none" else "symmetric pump"
        sim_job = jm.submit(
            f"Tip-position chain · simulate · {sim['P_avg_mW']:g} mW · V_tip {sim['V_tip']:g} V · {pump} · "
            f"{len(xs)}×{len(hs)}", "tip_sim", SCRIPT, "simulate", kwargs=dict(out=rel, fdtd=fdtd, **sim), out=out,
            progress={"type": "count", "pattern": r"^X=[+-]\d+\.\d h=", "total": max(1, remaining)}, after=fdtd_job)
        demod_job = jm.submit(
            f"Tip-position chain · demodulate · A {dem['A_nm']:g} / h_min {dem['h_min_nm']:g} nm · {out.name}", "tip_demod",
            SCRIPT, "demod", kwargs=dict(out=rel, **dem), out=out, after=sim_job)
        self._set_chain(dict(fdtd=fdtd_job, sim=sim_job, demod=demod_job))
        self.tabs.setCurrentWidget(self.run_tab)

    def redemod(self):
        d = self.result_dir
        if d is None or not self._can_redemod():
            self._update_demod_btn()
            self._error("Cannot re-demodulate", self.demod_btn.toolTip())
            return
        try:
            _, dem = self._values()
        except ValueError as e:
            self._error("Invalid parameters", str(e))
            return
        errs = self._demod_errors(dem)
        if errs:
            self._error("Invalid parameters", "\n".join(errs))
            return
        name = demod_name(dem["A_nm"], dem["h_min_nm"])
        if is_committed(d) and not self._confirm(
                "Write into a repository result?", f"{paths.rel(d)} is a committed result folder: the demodulation writes or "
                                                   f"overwrites {name} and the PNG there. Continue?"):
            return
        if (d / name).exists() and not is_committed(d) and not self._confirm(
                "Overwrite the existing demodulation?", f"{name} and tip_positions_A{dem['A_nm']:g}.png already exist and "
                                                        "will be replaced by the new demodulation. Continue?"):
            return
        job = JobManager.instance().submit(
            f"Tip-position chain · demodulate · A {dem['A_nm']:g} / h_min {dem['h_min_nm']:g} nm · {d.name}", "tip_demod",
            SCRIPT, "demod", kwargs=dict(out=paths.rel(d), **dem), out=d)
        self._set_chain(dict(demod=job))
        self.tabs.setCurrentWidget(self.run_tab)

    # ------------------------------------------------------------------ chain tracking
    def _set_chain(self, chain):
        old = self.chain.get("sim")
        if old is not None:
            try:
                old.output.disconnect(self._on_sim_output)
            except (RuntimeError, TypeError):
                pass
        self.chain = {k: j for k, j in chain.items() if j is not None}
        sim = self.chain.get("sim")
        if sim is not None:
            sim.output.connect(self._on_sim_output)
        first = next((j for j in self.chain.values() if j.active), None) or list(self.chain.values())[-1]
        self.panel.attach(first)
        self.last_line.setText("")
        self._update_chain()

    def _chain_jobs(self):
        return [j for j in self.chain.values()]

    def _focus_step(self, k):
        j = self.chain.get(k)
        if j is not None:
            self.panel.attach(j)
            self._update_chain()

    def _update_chain(self):
        self._update_demod_btn()             # the shown folder may just have become busy / free
        self.chain_card.setVisible(bool(self.chain))
        for k, st in self.steps.items():
            j = self.chain.get(k)
            st.setVisible(j is not None)
            if j is not None:
                st.set_job(j, j is self.panel.job)
        self._update_grid()

    def _update_grid(self):
        j = self.chain.get("sim")
        self.grid_w.setVisible(j is not None)
        if j is None:
            return
        kw = j.spec.get("kwargs", {})
        out = paths.absolute(kw.get("out", ""))
        xs, hs = kw.get("x_list", X_LIST), kw.get("heights", HEIGHTS_NM)
        done = {(i, k) for i, X in enumerate(xs) for k, h in enumerate(hs) if (out / npz_name(X, h)).exists()}
        running = None
        if j.status == RUNNING:
            running = next(((i, k) for i in range(len(xs)) for k in range(len(hs)) if (i, k) not in done), None)
        self.grid_w.set_state(xs, hs, done, running)

    def _on_sim_output(self, text):
        hits = [l for l in text.split("\n") if RUN_RE.match(l)]
        if hits:
            self.last_line.setText("Last finished: " + hits[-1])
            self._update_grid()

    def _on_job_changed(self, job):
        if job not in self._chain_jobs():
            return
        p = self.panel.job
        if job.status == RUNNING and p is not job and (p is None or p.status != RUNNING):
            self.panel.attach(job)
        self._update_chain()

    def _on_job_finished(self, job):
        if job.kind in ("tip_sim", "tip_demod"):
            self.refresh_history()
        if job.kind == "fdtd_heights":
            self._refresh_fdtd_choices()
        shown = (job.kind in ("tip_sim", "tip_demod") and job.out and self.result_dir is not None
                 and paths.absolute(job.out).resolve() == self.result_dir.resolve())
        if job not in self._chain_jobs():
            if shown:                       # e.g. re-run from the job center into the folder on display
                self.show_result(self.result_dir, switch=False, prefer=self._selected_demod_name())
            return
        if job is self.chain.get("sim") and job.status in (CANCELLED, FAILED) and job.started:
            self._quarantine_partial(job)
        if job is self.chain.get("fdtd") and job.status in (CANCELLED, FAILED) and job.started:
            folder = paths.absolute(job.spec.get("kwargs", {}).get("out", ""))
            run_async(quarantine_ratio, folder, on_done=self._ratio_quarantined)
        self._update_chain()
        if job is self.chain.get("demod") and job.status == FINISHED and job.out:
            kw = job.spec.get("kwargs", {})
            self.show_result(paths.absolute(job.out), prefer=demod_name(kw.get("A_nm", 180), kw.get("h_min_nm", 10)))
        elif shown:
            self.show_result(self.result_dir, switch=False, prefer=self._selected_demod_name())
        self._update_plan()

    def _ratio_quarantined(self, bad):
        if bad:
            self.last_line.setText(f"⚠ The FDTD map {bad} was not completely written when the job stopped and was renamed to "
                                   "*.npz.partial; this height is recomputed next time.")
        self._refresh_fdtd_choices()

    def _quarantine_partial(self, job):
        """A run stopped inside np.savez_compressed leaves a truncated npz that simulate would skip forever on resume:
        rename the newest one to *.broken if it cannot be read."""
        out = paths.absolute(job.spec.get("kwargs", {}).get("out", ""))
        files = sorted(out.glob("X*_h*.npz"), key=lambda p: p.stat().st_mtime) if out.is_dir() else []
        if not files:
            return None
        f = files[-1]
        try:
            with np.load(f) as z:
                z["Q_tip"], z["t"]
            return None
        except Exception:
            bad = f.with_name(f.name + ".broken")
            f.rename(bad)
            self.last_line.setText(f"⚠ {f.name} was not completely written when the job stopped and was renamed to "
                                   f"{bad.name}; it is recomputed when you resume.")
            return bad

    def _adopt_active_chain(self):
        """Re-attach to a tip chain started earlier (e.g. before this page existed, or re-run from the job center)."""
        act = [j for j in JobManager.instance().active() if j.kind in ("tip_sim", "tip_demod")]
        if not act:
            return
        j = act[-1]
        if j.kind == "tip_demod" and j.after is not None and j.after.kind == "tip_sim":
            sim, demod = j.after, j
        elif j.kind == "tip_sim":
            sim = j
            demod = next((d for d in JobManager.instance().jobs if d.kind == "tip_demod" and d.after is sim), None)
        else:
            sim, demod = None, j
        fdtd = sim.after if sim is not None and sim.after is not None and sim.after.kind == "fdtd_heights" else None
        self._set_chain(dict(fdtd=fdtd, sim=sim, demod=demod))

    # ------------------------------------------------------------------ results
    def show_result(self, d, switch=True, prefer=None):
        d = Path(d)
        if d.is_file():
            prefer = d.name if d.name.startswith("demod_") else prefer
            d = d.parent
        self.result_dir = d
        demods = demod_files(d)
        meta = read_json(d / "meta.json") or (read_json(demods[0]) if demods else None) or {}
        self.meta = meta
        xs, hs = meta.get("X") or [], meta.get("heights_nm") or []
        self.n_expected = len(xs) * len(hs)
        self.n_present = npz_count(d, xs, hs) if self.n_expected else len(list(d.glob("X*_h*.npz")))
        complete = self.n_expected > 0 and self.n_present == self.n_expected

        busy = bool(self._busy(d))
        state = (f"raw npz {self.n_present}/{self.n_expected}" if self.n_expected else "no meta.json")
        tag = "repository" if is_committed(d) else "GUI run" if d.parent == paths.GUI_RUNS else ""
        shown = paths.rel(d).replace("/", "/​")     # zero-width spaces: the path may wrap in narrow windows
        self.res_title.setText("  ·  ".join(s for s in (shown, tag, state, "running" if busy else "") if s))
        self.res_title.setToolTip(str(d))

        P = meta.get("P_avg_mW")
        self.tiles["P"].set(f"{P:g} mW" if P is not None else "—", "3 µm spot")
        Vb, T = meta.get("V_bias"), meta.get("T")
        hint = f"{T:g} K · " + ("degenerate" if meta.get("degenerate") else "classical") if T is not None else ""
        self.tiles["Vb"].set(f"{Vb:g} V" if Vb is not None else "—", hint)
        self.tiles["Vb"].setToolTip("Anode at −x; T = carrier temperature; degenerate = Fermi–Dirac diffusion")
        Vt = meta.get("V_tip")
        self.tiles["Vt"].set((f"{Vt:+g} V" if Vt else "0 V") if Vt is not None else "—", "vs. gap midpoint")
        fd = meta.get("fdtd")
        sym = fd in (None, "none")
        self.tiles["pump"].set("—" if not meta else ("none" if sym else "FDTD"),
                               "symmetric Gaussian" if sym else Path(str(fd)).name)
        self.tiles["pump"].setToolTip("Symmetric Gaussian pump (no tip shadow)" if sym else f"Per-height FDTD pump ratio maps: {fd}")
        self.tiles["h"].set(f"{len(hs)}" if hs else "—", f"{min(hs):g}–{max(hs):g} nm" if hs else "")
        self.tiles["h"].setToolTip(", ".join(f"{h:g}" for h in hs) + " nm" if hs else "")
        te = meta.get("t_end_ps")
        self.tiles["X"].set(f"{len(xs)}" if xs else "—", f"{fx(min(xs))}…{fx(max(xs))} µm" if xs else "")
        self.tiles["X"].setToolTip((", ".join(fx(x) for x in xs) + " µm" + (f"; t_end = {te:g} ps" if te else ""))
                                   if xs else "")

        self.demod_combo.blockSignals(True)
        self.demod_combo.clear()
        for p in demods:
            js = read_json(p) or {}
            self.demod_combo.addItem(f"A = {js.get('A_nm', '?'):g} nm · h_min = {js.get('h_min_nm', '?'):g} nm" if js else p.name,
                                     p.name)
        idx = 0
        for want in (prefer, "demod_A180_h10.json"):
            if want and self.demod_combo.findData(want) >= 0:
                idx = self.demod_combo.findData(want)
                break
        if demods:
            self.demod_combo.setCurrentIndex(idx)
        self.demod_combo.setEnabled(bool(demods))
        self.demod_combo.blockSignals(False)

        if not meta and not demods:
            note = "This folder has no meta.json and no demodulation results (demod_A*_h*.json)."
        elif not demods and not self.n_present:
            note = ("This folder only has meta.json: there are no demodulation results, and the raw X*_h*.npz are not in "
                    "the repository (git-ignored). To see results, use “Load into form” and run “Simulate + demodulate”.")
        elif self.n_present == 0:
            note = ("The raw X*_h*.npz are not in this folder (git-ignored in the repository), so it cannot be "
                    "re-demodulated with a new A / h_min and the Q_tip(t) waveforms are unavailable. To recompute: "
                    "“Load into form” → “Simulate + demodulate” (the default 70 runs take ≈ 25–60 min).")
        elif not complete:
            note = (f"Raw data incomplete ({self.n_present}/{self.n_expected}): the simulation is still running or was "
                    "interrupted." + ("" if busy else " “Resume this folder” computes the rest with the same parameters "
                                                     "(existing npz files are skipped)."))
        elif not demods:
            note = "The simulation is complete but not demodulated yet: click “Re-demodulate with form settings”."
        else:
            note = ""
        self.res_note.setText(note)
        self.res_note.setVisible(bool(note))
        self.btn_params.setEnabled(bool(meta))
        self.btn_resume.setVisible(bool(meta) and not complete and self.n_expected > 0 and not is_committed(d) and not busy
                                   and self.n_present > 0)
        self.btn_folder.setEnabled(d.exists())
        self._wave = None if (self._wave and self._wave[0][0] != str(d)) else self._wave
        self._on_demod_selected()
        self._update_demod_btn()
        if switch:
            self.tabs.setCurrentWidget(self.res_tab)

    def _selected_demod_name(self):
        return self.demod_combo.currentData() if self.demod_combo.count() else None

    def _on_demod_selected(self, *_):
        d = self.result_dir
        name = self._selected_demod_name()
        self.demod = read_json(d / name) if (d is not None and name) else None
        js = self.demod
        if js:
            A, hm = float(js.get("A_nm", 180)), float(js.get("h_min_nm", 10))
            k = int(js.get("deriv", 2))
            b = js.get("band_thz", [0.3, 2.0])
            der = "d²S/dt²" if k == 2 else "dS/dt" if k == 1 else f"d^{k}S/dt^{k}"
            hs = self.meta.get("heights_nm") or []
            rng = f"tapping range {hm:g}–{hm + 2 * A:g} nm"
            if hs and hm + 2 * A > max(hs) + 1e-9:
                t = ThemeManager.instance().tokens
                rng = f"<span style='color:{t.warning}'>{rng} (above the largest height {max(hs):g} nm, clipped)</span>"
            self.demod_desc.setText(f"f_cut = {float(js.get('f_cut_thz', 2.2)):g} THz · detected ∝ {der} · "
                                    f"band {b[0]:g}–{b[1]:g} THz · {rng}")
            png = d / f"tip_positions_A{A:g}.png"
        else:
            self.demod_desc.setText("No demodulation results" if d is not None else "")
            pngs = sorted(d.glob("tip_positions_A*.png")) if d is not None else []
            png = pngs[0] if pngs else None
        if png is not None and png.exists():
            self.png_view.load(png)
        else:
            self.png_view.clear("This result has no script figure tip_positions_A*.png" if d is not None else "No image")
        self._redraw_amp()
        self._fill_table()
        self._start_wave_load()

    def _redraw_amp(self, *_):
        js = self.demod
        if not js or not js.get("X"):
            self.amp_empty.setText("This folder has no demodulation results (demod_A*_h*.json), so there is no amplitude "
                                   "vs X to plot." if self.result_dir is not None else "No result loaded")
            self.amp_stack.setCurrentIndex(0)
            self.series_box.setVisible(False)
            return
        self.amp_stack.setCurrentIndex(1)
        self.series_box.setVisible(True)
        sel = [k for k, c in self.series.items() if c.isChecked()]
        absolute = self.abs_box.isChecked()
        meta = self.meta or js
        narrow = self._narrow["amp"] = self.amp_plot.width() < NARROW_PX

        def draw(fig):
            t = ThemeManager.instance().tokens
            ax = fig.add_subplot(111)
            X = np.asarray(js["X"], float)
            o = np.argsort(X)
            src = js.get("rms" if absolute else "rms_norm", {})
            cyc = [c["color"] for c in __import__("matplotlib").rcParams["axes.prop_cycle"]]
            for key in sel:
                if key in src:
                    y = np.asarray(src[key], float)[o]
                    if key == "far":
                        ax.plot(X[o], y, "x:", color=t.muted, lw=1.4, ms=7, label="far field dJ/dt")
                        continue
                    n = int(key[-1] if key[0] == "S" else key[3])
                    col = cyc[(n - 1) % len(cyc)]
                    if key.startswith("S"):
                        ax.plot(X[o], y, "s:", color=col, alpha=0.7, lw=1.2, ms=5, label=f"peak |S{n}|")
                    elif key.endswith("_d1"):
                        ax.plot(X[o], y, "^--", color=col, alpha=0.8, lw=1.2, ms=5, label=f"n = {n}, dS/dt")
                    else:
                        ax.plot(X[o], y, "o-", color=col, lw=1.8, ms=5.5, label=f"n = {n}")
            if not absolute:
                m = measured()
                for b, key, mk, col in ((25, "meas+", "o", t.text), (-25, "meas-", "s", t.muted)):
                    if key in sel and b in m:
                        xm, r, e = m[b]
                        ax.errorbar(-xm if b < 0 else xm, r, yerr=e, fmt=mk, color=col, mfc="none" if b < 0 else col,
                                    capsize=3, ms=6, lw=1.0, label=f"measured {b:+d} V" + (" (X→−X)" if b < 0 else ""))
            ax.axvline(0, color=t.border, lw=0.8, zorder=0)
            ax.set_xlabel("X (µm)" if narrow else "X (µm), beam position relative to the tip")
            b = js.get("band_thz", [0.3, 2.0])
            if absolute:
                ax.set_yscale("log")
                ax.set_ylabel("band amplitude (arb. u.)" if narrow else f"amplitude in {b[0]:g}–{b[1]:g} THz (arb. u.)")
            else:
                ax.set_ylabel("band amplitude (norm.)" if narrow else f"amplitude in {b[0]:g}–{b[1]:g} THz, normalised")
                ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.35)
            pump = "FDTD tip map" if meta.get("fdtd") not in (None, "none") else "symmetric pump"
            fig.suptitle(f"{meta.get('P_avg_mW', 0):g} mW, V_bias = {meta.get('V_bias', 0):g} V, "
                         f"V_tip = {meta.get('V_tip', 0):g} V, {pump}" + ("\n" if narrow else ";  ")
                         + f"A = {js.get('A_nm', 0):g} nm, h_min = {js.get('h_min_nm', 0):g} nm", fontsize=8.5 if narrow else 10)
            h, l = ax.get_legend_handles_labels()
            if h and narrow:                 # a legend at the side would leave no room for the axes
                fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=7)
            elif h:
                fig.legend(h, l, loc="outside right center", fontsize=8)
        self.amp_plot.plot(draw)

    def _fill_table(self):
        js = self.demod
        self.table.clear()
        if not js or not js.get("X"):
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        src = js.get("rms_norm", {})
        keys = [k for k in [*(f"det{n}" for n in HARMONICS), *(f"det{n}_d1" for n in HARMONICS),
                            *(f"S{n}" for n in HARMONICS), "far"] if k in src]
        raw = js.get("rms", {})
        X = js["X"]
        cols = ["X (µm)"] + keys + [f"{k} (rms)" for k in keys if k in raw]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(X))
        for i, x in enumerate(X):
            vals = [fx(x)] + [f"{src[k][i]:.3f}" for k in keys] + [f"{raw[k][i]:.4g}" for k in keys if k in raw]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, c, it)
        self.table.resizeColumnsToContents()
        self.table_note.setText(f"{self.demod_combo.currentData()} · det/S/far columns are normalised to their own maximum "
                                "(rms_norm); columns marked (rms) are raw values")

    def _copy_table(self):
        t = self.table
        rows = ["\t".join(t.horizontalHeaderItem(c).text() for c in range(t.columnCount()))]
        for r in range(t.rowCount()):
            rows.append("\t".join(t.item(r, c).text() if t.item(r, c) else "" for c in range(t.columnCount())))
        QGuiApplication.clipboard().setText("\n".join(rows))

    # -- waveforms (only when the npz are present)
    def _wave_params(self):
        js = self.demod or {}
        return dict(A_nm=float(js.get("A_nm", 180.0)), h_min_nm=float(js.get("h_min_nm", 10.0)),
                    f_cut_thz=float(js.get("f_cut_thz", 2.2)), deriv=int(js.get("deriv", 2)))

    def _start_wave_load(self):
        d = self.result_dir
        self._wave_gen += 1
        if d is None or not self.n_present:
            self.wave_empty.setText("This result folder has no raw X*_h*.npz (git-ignored in the repository), so the "
                                    "Q_tip(t) waveforms cannot be shown.\nRun “Simulate + demodulate” again to see Q_tip(t) "
                                    "for every height and the detected waveform for each X here."
                                    if d is not None else "No result loaded")
            self.wave_stack.setCurrentIndex(0)
            self.wave_status.setText("")
            self.wave_x.clear()
            return
        xs, hs = self.meta.get("X") or [], self.meta.get("heights_nm") or []
        dem = self._wave_params()
        key = (str(d), self.n_present)
        cached = self._wave[1] if (self._wave and self._wave[0] == key) else None
        gen = self._wave_gen
        self.wave_status.setText("Reading npz files and demodulating…")
        if cached is None:                 # another folder: do not leave its plot (or an old message) on display
            self.wave_empty.setText("Reading npz files and demodulating…")
            self.wave_stack.setCurrentIndex(0)
        run_async(_wave_work, str(d), xs, hs, dem, cached,
                  on_done=lambda r, g=gen, k=key: self._wave_done(g, k, r),
                  on_error=lambda e, g=gen: g == self._wave_gen and self.wave_status.setText("Could not read: "
                                                                                             + e.splitlines()[0]))

    def _wave_done(self, gen, key, res):
        if gen != self._wave_gen:
            return
        data, det = res
        self._wave = (key, data, det)
        cur = self.wave_x.currentData()
        xs = [xkey(X) for X in (self.meta.get("X") or []) if xkey(X) in data["Q"]]
        self.wave_x.blockSignals(True)
        self.wave_x.clear()
        for k in xs:
            self.wave_x.addItem(f"{fx(k)} µm", k)
        i = self.wave_x.findData(cur if cur in xs else ("+5.0" if "+5.0" in xs else (xs[0] if xs else None)))
        self.wave_x.setCurrentIndex(max(0, i))
        self.wave_x.blockSignals(False)
        nx = len(self.meta.get("X") or [])
        self.wave_status.setText(f"{self.n_present} npz · {len(det)}/{nx} X with all heights")
        self.wave_stack.setCurrentIndex(1)
        self._redraw_wave()

    def _redraw_wave(self):
        if not self._wave or self.wave_stack.currentIndex() != 1:
            return
        _, data, det = self._wave
        xk = self.wave_x.currentData()
        n = int(self.wave_n.currentText())
        dem = self._wave_params()
        narrow = self._narrow["wave"] = self.wave_plot.width() < NARROW_PX

        def draw(fig):
            import matplotlib
            t = ThemeManager.instance().tokens
            ax1, ax2 = fig.subplots(1, 2)
            fs = 8.5 if narrow else 9.5
            row = data["Q"].get(xk, {})
            tt = data["t"].get(xk)
            hs = sorted(row)
            cmap = matplotlib.colormaps["viridis"]
            c0, c1 = (0.3, 1.0) if t.dark else (0.0, 0.85)      # keep both ends of viridis readable on the background
            for k, h in enumerate(hs):
                ax1.plot(tt * 1e12, row[h] * 1e12, color=cmap(c0 + (c1 - c0) * k / max(1, len(hs) - 1)), lw=1.3,
                         label=f"h = {h} nm")
            ax1.set_xlabel("t (ps)")
            ax1.set_ylabel("Q_tip (pC/cm)" if narrow else "induced tip charge Q_tip (pC/cm)")
            ax1.set_title(f"(a) Q_tip(t) per tip height\nX = {fx(xk)} µm" if xk else "(a) Q_tip(t)", fontsize=fs)
            if hs:
                ax1.legend(fontsize=7, ncol=2)
            ax1.grid(True, alpha=0.3)
            cyc = [c["color"] for c in matplotlib.rcParams["axes.prop_cycle"]]
            keys = [k for k in sorted(det, key=float)]
            peak = max((np.abs(det[k][n]).max() for k in keys), default=1.0) or 1.0
            for i, k in enumerate(keys):
                ax2.plot(data["t"][k] * 1e12, det[k][n] / peak, color=cyc[i % len(cyc)], lw=2.2 if k == xk else 1.2,
                         label=f"X = {fx(k)}")
            if keys:
                ax2.legend(fontsize=8)
            else:
                ax2.text(0.5, 0.5, "no X with all heights yet", transform=ax2.transAxes, ha="center", color=t.muted)
            ax2.axhline(0, color=t.border, lw=0.8)
            der = "d²" if dem["deriv"] == 2 else "d" if dem["deriv"] == 1 else f"d^{dem['deriv']}"
            sq = '²' if dem['deriv'] == 2 else ''
            ax2.set_title(f"(b) {der}S{n}/dt{sq}, all X\nA = {dem['A_nm']:g}, h_min = {dem['h_min_nm']:g} nm" if narrow else
                          f"(b) detected ∝ {der}S{n}/dt{sq}, all X\n"
                          f"{dem['f_cut_thz']:g} THz LP, A = {dem['A_nm']:g} nm, h_min = {dem['h_min_nm']:g} nm", fontsize=fs)
            ax2.set_xlabel("t (ps)")
            ax2.set_ylabel("normalised" if narrow else "normalised (largest |trace| = 1)")
            ax2.grid(True, alpha=0.3)
            tmax = max((float(v[-1]) for v in data["t"].values()), default=3e-12) * 1e12
            ax1.set_xlim(0, min(3.0, tmax))
            ax2.set_xlim(0, min(2.5, tmax))
        self.wave_plot.plot(draw)

    def _can_redemod(self):
        d = self.result_dir
        return (d is not None and self.n_expected > 0 and self.n_present == self.n_expected and not self._busy(d))

    def _update_demod_btn(self):
        d = self.result_dir
        ok = self._can_redemod()
        if d is None:
            tip = "First load a result folder with raw X*_h*.npz from History"
        elif ok:
            tip = (f"Re-demodulate {paths.rel(d)} with the Demodulation settings in the form (≈ 1–5 s); writes "
                   "demod_A<A>_h<h_min>.json and the PNG")
        elif self._busy(d):
            tip = "A simulation or demodulation job for this folder is still running"
        elif self.n_present == 0:
            tip = (f"{paths.rel(d)} has no raw X*_h*.npz (git-ignored in the repository) and cannot be re-demodulated. "
                   "Simulate again first (Load into form → Simulate + demodulate).")
        else:
            tip = (f"The raw data in {paths.rel(d)} is incomplete ({self.n_present}/{self.n_expected}); finish the "
                   "remaining simulations first")
        for b in (self.demod_btn, self.btn_redemod):
            b.setEnabled(ok)
            b.setToolTip(tip)

    def _update_note_colors(self):
        """Rich-text colours follow the theme (the plots re-render themselves)."""
        if self.result_dir is not None and self.demod:
            js, hs = self.demod, self.meta.get("heights_nm") or []
            if hs and float(js.get("h_min_nm", 10)) + 2 * float(js.get("A_nm", 180)) > max(hs) + 1e-9:
                self._on_demod_selected()

    def _load_result_params(self):
        m = self.meta
        if not m:
            return
        vals = {k: m[k] for k in ("P_avg_mW", "V_bias", "V_tip", "T", "degenerate", "t_end_ps") if k in m}
        if "heights_nm" in m:
            vals["heights"] = tuple(m["heights_nm"])
        if "X" in m:
            vals["x_list"] = tuple(float(x) for x in m["X"])
        js = self.demod
        if js:
            vals.update({k: js[k] for k in ("A_nm", "h_min_nm", "f_cut_thz") if k in js})
            if "deriv" in js:
                vals["deriv"] = str(int(js["deriv"]))
            if "band_thz" in js:
                vals["band"] = tuple(js["band_thz"])
        self.form.set_values(vals, reset_others=True)
        self._home_tuples()
        self._set_fdtd(m.get("fdtd") or "none")
        self._update_plan()

    def _resume(self):
        if self.result_dir is None:
            return
        self._load_result_params()
        self.out_edit.setText(paths.rel(self.result_dir))
        self.tabs.setCurrentWidget(self.run_tab)

    # ------------------------------------------------------------------ history
    def refresh_history(self):
        self.hist.clear()
        dirs = sorted(p for p in COMMITTED.iterdir() if p.is_dir()) if COMMITTED.is_dir() else []
        dirs += sorted((p for p in paths.GUI_RUNS.glob("tip_*") if p.is_dir()), reverse=True)
        for d in dirs:
            demods = demod_files(d)
            meta = read_json(d / "meta.json") or (read_json(demods[0]) if demods else None) or {}
            xs, hs = meta.get("X") or [], meta.get("heights_nm") or []
            n_exp = len(xs) * len(hs)
            n = npz_count(d, xs, hs) if n_exp else 0
            bits = []
            if meta:
                bits.append(f"{meta.get('P_avg_mW', 0):g} mW")
                bits.append(f"V_tip {meta.get('V_tip', 0):g} V")
                bits.append("symmetric pump" if meta.get("fdtd") in (None, "none") else "FDTD maps")
                bits.append(f"{len(xs)} X × {len(hs)} h")
            bits.append("demod: " + (", ".join(p.stem.replace("demod_", "") for p in demods) if demods else "none"))
            bits.append(f"raw npz {n}/{n_exp}" if n_exp else "no meta.json")
            if self._busy(d):
                bits.append("running")
            if not is_committed(d):         # checkout time of committed folders says nothing
                stamp = (d / "meta.json") if (d / "meta.json").exists() else d
                bits.append(time.strftime("%Y-%m-%d %H:%M", time.localtime(stamp.stat().st_mtime)))
            tag = "repository" if is_committed(d) else "GUI"
            it = QListWidgetItem(f"{paths.rel(d)}   [{tag}]\n    " + " · ".join(bits))
            it.setData(Qt.ItemDataRole.UserRole, str(d))
            it.setToolTip("Double-click to load into Results" + ("" if n == n_exp and n_exp else
                                                                 "; raw npz incomplete, cannot re-demodulate"))
            self.hist.addItem(it)
        if not dirs:
            self.hist.addItem("No tip-position chain results yet")

    def _hist_open(self, it):
        p = it.data(Qt.ItemDataRole.UserRole)
        if p:
            self.show_result(p)

    def _open_other(self):
        d = QFileDialog.getExistingDirectory(self, "Choose a tip-position chain result folder", str(COMMITTED))
        if d:
            self.show_result(d)

    # ------------------------------------------------------------------ page API
    def on_show(self):
        if not self.chain:
            self._adopt_active_chain()
            if self.chain:                  # a chain started elsewhere (e.g. re-run from the Job center): show its progress
                self.tabs.setCurrentWidget(self.run_tab)
        self._refresh_fdtd_choices()
        self.refresh_history()
        self._update_chain()

    def open(self, path=None, **_):
        if path:
            self.show_result(paths.absolute(path))
