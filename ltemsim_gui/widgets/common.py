"""Small building blocks shared by all pages."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout,
                               QWidget, QMessageBox)


def card(parent=None, clickable=False) -> QFrame:
    f = QFrame(parent)
    f.setObjectName("card")
    if clickable:
        f.setProperty("clickable", True)
        f.setCursor(Qt.CursorShape.PointingHandCursor)
    return f


class ClickableCard(QFrame):
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setProperty("clickable", True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(e)


def label(text="", obj=None, wrap=False, selectable=False) -> QLabel:
    l = QLabel(text)
    if obj:
        l.setObjectName(obj)
    if wrap:
        l.setWordWrap(True)
    if selectable:
        l.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return l


def button(text, slot=None, obj=None, tooltip=None) -> QPushButton:
    b = QPushButton(text)
    if obj:
        b.setObjectName(obj)
    if slot:
        b.clicked.connect(slot)
    if tooltip:
        b.setToolTip(tooltip)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


class PageHeader(QWidget):
    """Big title + subtitle + right-aligned action area."""

    def __init__(self, title, subtitle="", parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 6)
        col = QVBoxLayout()
        col.setSpacing(2)
        self.title = label(title, "pageTitle")
        self.subtitle = label(subtitle, "pageSubtitle", wrap=True)
        col.addWidget(self.title)
        if subtitle:
            col.addWidget(self.subtitle)
        lay.addLayout(col, 1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(8)
        lay.addLayout(self.actions)

    def add_action(self, w):
        self.actions.addWidget(w)
        return w


class MetricTile(QFrame):
    def __init__(self, label_text, value="—", hint="", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        self.lab = label(label_text, "metricLabel")
        self.val = label(value, "metricValue", selectable=True)
        self.hint = label(hint, "muted", wrap=True)
        self.hint.setVisible(bool(hint))
        lay.addWidget(self.lab)
        lay.addWidget(self.val)
        lay.addWidget(self.hint)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    def set(self, value, hint=None):
        self.val.setText(value)
        if hint is not None:
            self.hint.setText(hint)
            self.hint.setVisible(bool(hint))


class Page(QWidget):
    """Base page: scrollable column with a header; subclasses fill self.body."""
    title = ""
    subtitle = ""

    def __init__(self, parent=None, scroll=False):
        super().__init__(parent)
        self.setObjectName("page")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 16)
        outer.setSpacing(10)
        self.header = PageHeader(self.title, self.subtitle)
        outer.addWidget(self.header)
        if scroll:
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setFrameShape(QFrame.Shape.NoFrame)
            inner = QWidget()
            inner.setObjectName("page")
            self.body = QVBoxLayout(inner)
            self.body.setContentsMargins(0, 0, 4, 0)
            self.body.setSpacing(12)
            area.setWidget(inner)
            outer.addWidget(area, 1)
        else:
            self.body = QVBoxLayout()
            self.body.setSpacing(10)
            outer.addLayout(self.body, 1)

    def on_show(self):
        """Called each time the page becomes visible."""


def reveal(path):
    """Show a file or folder in Finder / Explorer."""
    p = Path(path)
    if not p.exists():
        p = p.parent
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(p)] if p.is_file() else ["open", str(p)])
    elif sys.platform.startswith("win"):
        subprocess.Popen(["explorer", "/select,", str(p)] if p.is_file() else ["explorer", str(p)])
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p if p.is_dir() else p.parent)))


def open_external(path):
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def error_box(parent, title, text, details=None):
    m = QMessageBox(parent)
    m.setIcon(QMessageBox.Icon.Warning)
    m.setWindowTitle(title)
    m.setText(text)
    if details:
        m.setDetailedText(details)
    m.exec()


def hline():
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("color: rgba(127,127,127,0.25);")
    return f
