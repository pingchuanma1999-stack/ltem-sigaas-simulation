"""Embedded matplotlib figure with toolbar, following the app theme."""
from __future__ import annotations

import warnings

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QVBoxLayout, QWidget

from ..theme import ThemeManager, mpl_rc

from ..fonts import install_cjk_fallback

install_cjk_fallback()        # Chinese text in embedded plots (DejaVu Sans has no CJK glyphs)
# plots drawn into not-yet-visible widgets have no size yet; matplotlib's warning about it is noise
warnings.filterwarnings("ignore", message="constrained_layout not applied")


class _Toolbar(NavigationToolbar2QT):
    def __init__(self, canvas, owner):
        super().__init__(canvas, owner)
        self._owner = owner

    def save_figure(self, *args):
        self._owner.save_dialog()


class PlotWidget(QWidget):
    """A Figure + canvas + navigation toolbar.

    Use `fig = w.new_figure()` (themed, cleared) then draw on it and call `w.draw()`.
    `w.adopt(fig)` shows an existing Figure (e.g. one returned by ltemsim.plots).
    `redraw` (optional callable taking the figure) is re-run on theme changes.
    """

    def __init__(self, parent=None, toolbar=True, figsize=(7, 4.5)):
        super().__init__(parent)
        self.redraw = None
        self._themed = True
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        with matplotlib.rc_context(mpl_rc(ThemeManager.instance().tokens)):
            self.figure = Figure(figsize=figsize, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = _Toolbar(self.canvas, self) if toolbar else None
        self._bar = QHBoxLayout()
        self._bar.setContentsMargins(4, 0, 4, 0)
        if self.toolbar:
            self._bar.addWidget(self.toolbar)
            self._bar.addStretch(1)
            lay.addLayout(self._bar)
        lay.addWidget(self.canvas, 1)
        self._style_toolbar()
        ThemeManager.instance().changed.connect(self._on_theme)

    def sizeHint(self):
        return QSize(480, 320)

    def minimumSizeHint(self):
        return QSize(220, 160)

    def _rebuild_toolbar(self):
        """New toolbar for the current canvas (matplotlib picks light/dark icons when it is created)."""
        if self.toolbar is None:
            return
        new_tb = _Toolbar(self.canvas, self)
        self._bar.replaceWidget(self.toolbar, new_tb)
        self.toolbar.setParent(None)
        self.toolbar.deleteLater()
        self.toolbar = new_tb
        self._style_toolbar()

    def _style_toolbar(self):
        """Compact, theme-matched toolbar; its opaque background lets matplotlib pick light/dark icons."""
        tb = self.toolbar
        if tb is None:
            return
        t = ThemeManager.instance().tokens
        self._tb_dark = t.dark
        tb.setIconSize(QSize(18, 18))
        tb.setStyleSheet(f"""
QToolBar {{ background: {t.surface}; border: none; spacing: 1px; padding: 0; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 3px; margin: 0; }}
QToolButton:hover {{ background: {t.surface2}; border-color: {t.border}; }}
QToolButton:checked {{ background: {t.accent_soft}; border-color: {t.accent}; }}
QLabel {{ color: {t.muted}; font-size: 11px; }}""")

    def rc(self):
        return matplotlib.rc_context(mpl_rc(ThemeManager.instance().tokens))

    def new_figure(self):
        """Clear and re-theme the figure; returns it. Draw inside `with w.rc():` for themed defaults."""
        self.figure.clear()
        t = ThemeManager.instance().tokens
        self.figure.set_facecolor(t.surface)
        self._themed = True
        return self.figure

    def plot(self, func):
        """Set `func(fig)` as the drawing routine, run it inside the theme context and draw."""
        self.redraw = func
        self._render()

    def _render(self):
        self._stale = False
        if self.redraw is None:
            return
        fig = self.new_figure()
        with self.rc():
            try:
                self.redraw(fig)
            except Exception as e:      # show the error in the figure instead of crashing the GUI
                fig.clear()
                fig.text(0.5, 0.5, f"Plot failed:\n{type(e).__name__}: {e}", ha="center", va="center", wrap=True,
                         color=ThemeManager.instance().tokens.danger)
        self.canvas.draw_idle()

    def adopt(self, fig: Figure):
        """Display an externally created figure (pyplot figures are detached from pyplot first)."""
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass
        old = self.canvas
        self.figure = fig
        self.canvas = FigureCanvasQTAgg(fig)
        lay = self.layout()
        lay.replaceWidget(old, self.canvas)
        old.setParent(None)
        old.deleteLater()
        self._rebuild_toolbar()
        self.redraw = None
        self._themed = False
        self.canvas.draw_idle()

    def draw(self):
        self.canvas.draw_idle()

    def save_dialog(self, default_name="figure.png"):
        path, _ = QFileDialog.getSaveFileName(self, "Save figure", default_name, "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.export(path)

    def export(self, path, dpi=200):
        """Save the figure; themed plots are re-drawn in the light theme so dark-mode exports stay readable."""
        tm = ThemeManager.instance()
        if self._themed and self.redraw is not None and tm.tokens.dark:
            from ..theme import LIGHT
            saved, tm.tokens = tm.tokens, LIGHT          # silently, no `changed` signal
            try:
                with matplotlib.rc_context(mpl_rc(LIGHT)):
                    fig = Figure(figsize=self.figure.get_size_inches(), layout="constrained")
                    fig.set_facecolor("white")
                    self.redraw(fig)
            finally:
                tm.tokens = saved
            fig.savefig(path, dpi=dpi, facecolor="white")
        else:
            self.figure.savefig(path, dpi=dpi, facecolor="white")

    def _on_theme(self, t):
        if self.toolbar is not None and getattr(self, "_tb_dark", None) != t.dark:
            self._rebuild_toolbar()
        self._style_toolbar()
        if self._themed and self.redraw is not None:
            if self.isVisible():
                self._render()
            else:
                self._stale = True

    def showEvent(self, e):
        super().showEvent(e)
        if getattr(self, "_stale", False):
            self._stale = False
            if self._themed and self.redraw is not None:
                self._render()
