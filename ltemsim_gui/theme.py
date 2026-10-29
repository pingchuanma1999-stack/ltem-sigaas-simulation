"""Light / dark theme: colour tokens, the Qt style sheet and matching matplotlib styles."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Tokens:
    dark: bool
    bg: str
    surface: str
    surface2: str
    sidebar: str
    border: str
    text: str
    muted: str
    accent: str
    accent_hover: str
    accent_soft: str
    on_accent: str
    success: str
    warning: str
    danger: str
    selection: str


LIGHT = Tokens(
    dark=False, bg="#f4f5f8", surface="#ffffff", surface2="#f8f9fb", sidebar="#eceef3", border="#dde1e8",
    text="#1c2230", muted="#687185", accent="#4357e8", accent_hover="#3446cf", accent_soft="#e6e9fd",
    on_accent="#ffffff", success="#14894a", warning="#b86e00", danger="#d23a3a", selection="#d6dcfb")

DARK = Tokens(
    dark=True, bg="#15171c", surface="#1d2027", surface2="#23262e", sidebar="#191b21", border="#2e323c",
    text="#e5e8ef", muted="#98a1b3", accent="#7b8cff", accent_hover="#909fff", accent_soft="#262c4d",
    on_accent="#0f1222", success="#3ccf7f", warning="#f0a63a", danger="#ff6b6b", selection="#33396a")


ASSETS = Path(__file__).resolve().parent / "assets"
CHEVRON = (ASSETS / "chevron-down.svg").as_posix()


def stylesheet(t: Tokens) -> str:
    return f"""
* {{ outline: 0; }}
QWidget {{ color: {t.text}; font-size: 13px; }}
QMainWindow, QDialog {{ background: {t.bg}; }}
QWidget#page {{ background: {t.bg}; }}
QToolTip {{ background: {t.surface}; color: {t.text}; border: 1px solid {t.border}; padding: 6px; border-radius: 6px; }}

/* ---------- sidebar ---------- */
QFrame#sidebar {{ background: {t.sidebar}; border-right: 1px solid {t.border}; }}
QLabel#appTitle {{ font-size: 16px; font-weight: 700; }}
QLabel#appSubtitle {{ color: {t.muted}; font-size: 11px; }}
QListWidget#nav {{ background: transparent; border: none; padding: 4px 6px; }}
QListWidget#nav::item {{ padding: 6px 10px; margin: 1px 0; border-radius: 8px; color: {t.text}; }}
QListWidget#nav::item:hover {{ background: {t.surface2}; }}
QListWidget#nav::item:selected {{ background: {t.accent_soft}; color: {t.accent}; font-weight: 600; }}
QLabel#navSection {{ color: {t.muted}; font-size: 11px; font-weight: 600; padding: 4px 16px 0 16px; }}

/* ---------- cards & headers ---------- */
QFrame#card {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 12px; }}
QFrame#card:hover[clickable="true"] {{ border: 1px solid {t.accent}; }}
QLabel#pageTitle {{ font-size: 22px; font-weight: 700; }}
QLabel#pageSubtitle {{ color: {t.muted}; font-size: 13px; }}
QLabel#cardTitle {{ font-size: 14px; font-weight: 650; }}
QLabel#muted {{ color: {t.muted}; }}
QLabel#metricValue {{ font-size: 22px; font-weight: 700; }}
QLabel#metricLabel {{ color: {t.muted}; font-size: 11px; }}
QLabel#badge {{ background: {t.accent_soft}; color: {t.accent}; border-radius: 9px; padding: 2px 8px; font-size: 11px; font-weight: 600; }}
QLabel#badgeOk {{ background: transparent; color: {t.success}; font-weight: 600; }}
QLabel#badgeWarn {{ background: transparent; color: {t.warning}; font-weight: 600; }}
QLabel#badgeErr {{ background: transparent; color: {t.danger}; font-weight: 600; }}
QLabel#sectionHeader {{ font-size: 12px; font-weight: 700; color: {t.accent}; padding-top: 8px; }}

/* ---------- buttons ---------- */
QPushButton, QToolButton {{
    background: {t.surface}; border: 1px solid {t.border}; border-radius: 8px; padding: 6px 14px; }}
QPushButton:hover, QToolButton:hover {{ border-color: {t.accent}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {t.surface2}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {t.muted}; background: {t.surface2}; }}
QPushButton#primary {{ background: {t.accent}; color: {t.on_accent}; border: 1px solid {t.accent}; font-weight: 600; padding: 7px 18px; }}
QPushButton#primary:hover {{ background: {t.accent_hover}; }}
QPushButton#primary:disabled {{ background: {t.border}; border-color: {t.border}; color: {t.muted}; }}
QPushButton#danger {{ color: {t.danger}; }}
QPushButton#danger:hover {{ border-color: {t.danger}; }}
QPushButton#danger:disabled {{ color: {t.muted}; border-color: {t.border}; }}
QPushButton#flat, QToolButton#flat {{ background: transparent; border: none; padding: 4px 8px; color: {t.muted}; }}
QPushButton#flat:hover, QToolButton#flat:hover {{ color: {t.accent}; }}
QToolButton::menu-indicator {{ image: none; }}

/* ---------- inputs ---------- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    background: {t.surface}; border: 1px solid {t.border}; border-radius: 7px; padding: 5px 8px;
    selection-background-color: {t.selection}; selection-color: {t.text}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{ border: 1px solid {t.accent}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QPlainTextEdit:disabled {{
    color: {t.muted}; background: {t.surface2}; border-color: {t.border}; }}
QLabel:disabled, QCheckBox:disabled, QRadioButton:disabled {{ color: {t.muted}; }}
QLineEdit[modified="true"] {{ border: 1px solid {t.accent}; background: {t.accent_soft}; }}
QLineEdit[invalid="true"] {{ border: 1px solid {t.danger}; }}
QComboBox[modified="true"] {{ border: 1px solid {t.accent}; background: {t.accent_soft}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url("{CHEVRON}"); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ background: {t.surface}; border: 1px solid {t.border}; selection-background-color: {t.accent_soft}; selection-color: {t.text}; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator, QGroupBox::indicator, QTreeView::indicator, QListView::indicator, QTableView::indicator {{
    width: 14px; height: 14px; border: 1px solid {t.muted}; border-radius: 4px; background: {t.surface}; }}
QCheckBox::indicator:hover {{ border-color: {t.accent}; }}
QCheckBox::indicator:checked, QGroupBox::indicator:checked, QTreeView::indicator:checked, QListView::indicator:checked,
QTableView::indicator:checked {{ background: {t.accent}; border-color: {t.accent}; image: url("{(ASSETS / ("check-dark.svg" if t.dark else "check-light.svg")).as_posix()}"); }}
QCheckBox::indicator:disabled {{ background: {t.surface2}; border-color: {t.border}; }}
QRadioButton::indicator {{ width: 14px; height: 14px; border: 1px solid {t.muted}; border-radius: 8px; background: {t.surface}; }}
QRadioButton::indicator:checked {{ width: 6px; height: 6px; background: {t.surface}; border: 5px solid {t.accent}; border-radius: 8px; }}
QCheckBox[modified="true"] {{ color: {t.accent}; font-weight: 600; }}
QLabel[modified="true"] {{ color: {t.accent}; font-weight: 600; }}

/* ---------- tabs ---------- */
QTabWidget::pane {{ border: 1px solid {t.border}; border-radius: 10px; background: {t.surface}; top: -1px; }}
QTabBar::tab {{ background: transparent; padding: 7px 14px; margin-right: 2px; border: none; color: {t.muted}; }}
QTabBar::tab:selected {{ color: {t.accent}; border-bottom: 2px solid {t.accent}; font-weight: 600; }}
QTabBar::tab:hover {{ color: {t.text}; }}
QTabBar QToolButton {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 5px; padding: 0; margin: 2px 1px; }}
QTabBar QToolButton::left-arrow {{ image: url("{(ASSETS / "chevron-left.svg").as_posix()}"); width: 11px; height: 11px; }}
QTabBar QToolButton::right-arrow {{ image: url("{(ASSETS / "chevron-right.svg").as_posix()}"); width: 11px; height: 11px; }}

/* ---------- views ---------- */
QTreeView, QTableView, QListView, QTreeWidget, QTableWidget {{
    background: {t.surface}; border: 1px solid {t.border}; border-radius: 8px; alternate-background-color: {t.surface2};
    selection-background-color: {t.accent_soft}; selection-color: {t.text}; gridline-color: {t.border}; }}
QHeaderView::section {{ background: {t.surface2}; color: {t.muted}; border: none; border-bottom: 1px solid {t.border}; padding: 5px 8px; font-weight: 600; }}
QPlainTextEdit#log {{ font-family: "SF Mono", Menlo, Monaco, monospace; font-size: 12px; background: {t.surface2}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QGroupBox {{ border: 1px solid {t.border}; border-radius: 10px; margin-top: 14px; padding: 10px 8px 8px 8px; background: {t.surface}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {t.accent}; font-weight: 700; }}

/* ---------- progress ---------- */
QProgressBar {{ background: {t.surface2}; border: 1px solid {t.border}; border-radius: 6px; height: 12px; text-align: center; font-size: 10px; color: {t.muted}; }}
QProgressBar::chunk {{ background: {t.accent}; border-radius: 5px; }}

/* ---------- splitters & scrollbars ---------- */
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: 6px; }}
QSplitter::handle:vertical {{ height: 6px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: {t.border}; border-radius: 4px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {t.muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QStatusBar {{ background: {t.sidebar}; border-top: 1px solid {t.border}; color: {t.muted}; }}
QMenu {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 5px; }}
QMenu::item:selected {{ background: {t.accent_soft}; color: {t.text}; }}
"""


def mpl_rc(t: Tokens) -> dict:
    """matplotlib rcParams that blend embedded figures into the current theme."""
    return {
        "figure.facecolor": t.surface, "axes.facecolor": t.surface, "savefig.facecolor": "white",
        "axes.edgecolor": t.muted, "axes.labelcolor": t.text, "text.color": t.text,
        "xtick.color": t.muted, "ytick.color": t.muted, "grid.color": t.border, "grid.alpha": 0.8,
        "legend.facecolor": t.surface, "legend.edgecolor": t.border, "legend.framealpha": 0.9,
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "legend.fontsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.prop_cycle": __import__("matplotlib").cycler(color=[
            "#4357e8", "#e8590c", "#12b886", "#d6336c", "#7048e8", "#f59f00", "#1098ad", "#868e96"]
            if not t.dark else ["#7b8cff", "#ff922b", "#38d9a9", "#f783ac", "#b197fc", "#ffd43b", "#66d9e8", "#ced4da"]),
    }


class ThemeManager(QObject):
    changed = Signal(object)          # Tokens

    _instance = None

    @classmethod
    def instance(cls) -> "ThemeManager":
        if cls._instance is None:
            cls._instance = ThemeManager()
        return cls._instance

    def __init__(self):
        super().__init__()
        self.mode = "system"
        self.tokens = LIGHT
        hints = QGuiApplication.styleHints()
        if hasattr(hints, "colorSchemeChanged"):
            hints.colorSchemeChanged.connect(lambda *_: self.mode == "system" and self.apply(self.mode))

    def _system_dark(self) -> bool:
        hints = QGuiApplication.styleHints()
        try:
            return hints.colorScheme() == Qt.ColorScheme.Dark
        except AttributeError:
            return QGuiApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128

    def apply(self, mode: str = "system"):
        self.mode = mode
        dark = self._system_dark() if mode == "system" else (mode == "dark")
        t = DARK if dark else LIGHT
        self.tokens = t
        app = QApplication.instance()
        app.setOverrideCursor(Qt.CursorShape.WaitCursor)     # re-polishing every widget takes a moment
        try:
            self._apply(app, t)
            self.changed.emit(t)
        finally:
            app.restoreOverrideCursor()

    def _apply(self, app, t: Tokens):
        if not getattr(self, "_fusion", False):        # once: re-creating the style is expensive
            app.setStyle("Fusion")
            self._fusion = True
        pal = QPalette()
        for role, col in ((QPalette.ColorRole.Window, t.bg), (QPalette.ColorRole.Base, t.surface),
                          (QPalette.ColorRole.AlternateBase, t.surface2), (QPalette.ColorRole.Text, t.text),
                          (QPalette.ColorRole.WindowText, t.text), (QPalette.ColorRole.Button, t.surface),
                          (QPalette.ColorRole.ButtonText, t.text), (QPalette.ColorRole.Highlight, t.accent),
                          (QPalette.ColorRole.HighlightedText, t.on_accent), (QPalette.ColorRole.ToolTipBase, t.surface),
                          (QPalette.ColorRole.ToolTipText, t.text), (QPalette.ColorRole.PlaceholderText, t.muted),
                          (QPalette.ColorRole.Link, t.accent)):
            pal.setColor(role, QColor(col))
        app.setPalette(pal)
        app.setStyleSheet(stylesheet(t))


def tokens() -> Tokens:
    return ThemeManager.instance().tokens


def repolish(w):
    """Re-apply the style sheet after a dynamic property change."""
    s = w.style()
    s.unpolish(w)
    s.polish(w)
    w.update()
