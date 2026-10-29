"""Read-only, auto-scrolling log with light syntax colouring."""
from __future__ import annotations

import re

from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import QPlainTextEdit

from ..theme import ThemeManager


class _Highlighter(QSyntaxHighlighter):
    RULES = (
        (re.compile(r"Traceback|Error|error|FAILED|Exception|not yet converging|rejected|collapsed"), "danger"),
        (re.compile(r"@@\w+.*|converged=True|converged after|saved|DONE"), "success"),
        (re.compile(r"Warning|warning"), "warning"),
        (re.compile(r"\[(dark|light[^\]]*)\]|X=[+-]?\d+\.\d|h =\s*\d+ nm|P=\s*[\d.]+ mW"), "accent"),
    )

    def __init__(self, doc):
        super().__init__(doc)
        self._formats = {}
        self.retheme()

    def retheme(self):
        t = ThemeManager.instance().tokens
        for key in ("danger", "success", "warning", "accent"):
            f = QTextCharFormat()
            f.setForeground(QColor(getattr(t, key)))
            if key in ("danger", "success"):
                f.setFontWeight(QFont.Weight.DemiBold)
            self._formats[key] = f
        self.rehighlight()

    def highlightBlock(self, text):
        for rx, key in self.RULES:
            for m in rx.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), self._formats[key])


class LogView(QPlainTextEdit):
    def __init__(self, parent=None, max_lines=20000):
        super().__init__(parent)
        self.setObjectName("log")
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setMaximumBlockCount(max_lines)
        self.setPlaceholderText("运行日志会显示在这里…")
        self._hl = _Highlighter(self.document())
        ThemeManager.instance().changed.connect(lambda *_: self._hl.retheme())

    def append_text(self, text: str):
        sb = self.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        self.appendPlainText(text)
        if at_bottom:
            sb.setValue(sb.maximum())

    def set_text(self, text: str):
        self.setPlainText(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
