"""Zoomable image viewer (wheel / pinch to zoom, drag to pan, double-click to fit)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QEvent, QRectF
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from .common import open_external, reveal


class _View(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self._fit = True

    def wheelEvent(self, e):
        # mouse wheel zooms; a trackpad scroll (pixel deltas, no modifier) pans
        if e.pixelDelta().isNull() or e.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            f = 1.0015 ** e.angleDelta().y()
            self._fit = False
            self.scale(f, f)
        else:
            super().wheelEvent(e)

    def viewportEvent(self, e):
        if e.type() == QEvent.Type.NativeGesture and e.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
            f = 1.0 + e.value()
            self._fit = False
            self.scale(f, f)
            return True
        return super().viewportEvent(e)

    def mouseDoubleClickEvent(self, e):
        self.fit()

    def fit(self):
        if self.scene() and not self.scene().itemsBoundingRect().isEmpty():
            self.fitInView(self.scene().itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._fit = True

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._fit:
            self.fit()


class ImageView(QWidget):
    def __init__(self, parent=None, toolbar=True):
        super().__init__(parent)
        self.path = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        if toolbar:
            bar = QHBoxLayout()
            bar.setSpacing(2)
            self.caption = QLabel("")
            self.caption.setObjectName("muted")
            self.caption.setMinimumWidth(40)
            self.caption.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)   # elided, never widens the pane
            bar.addWidget(self.caption, 1)
            for text, tip, slot in (("Fit", "Fit to window (or double-click the image)", lambda: self.view.fit()),
                                    ("1:1", "Actual size", self.actual_size),
                                    ("Finder", "Show in Finder", lambda: self.path and reveal(self.path)),
                                    ("Open", "Open with the default app", lambda: self.path and open_external(self.path))):
                b = QPushButton(text)
                b.setObjectName("flat")
                b.setToolTip(tip)
                b.clicked.connect(slot)
                bar.addWidget(b)
            lay.addLayout(bar)
        else:
            self.caption = None
        self.scene = QGraphicsScene(self)
        self.view = _View()
        self.view.setScene(self.scene)
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene.addItem(self.item)
        lay.addWidget(self.view, 1)
        self.empty = QLabel("No image", alignment=Qt.AlignmentFlag.AlignCenter)
        self.empty.setObjectName("muted")
        self.empty.setWordWrap(True)
        self.empty.setMinimumWidth(80)
        lay.addWidget(self.empty, 1)
        self.view.hide()

    def load(self, path):
        p = Path(path) if path else None
        if p is None or not p.exists():
            self.clear(f"File not found: {path}" if path else "No image")
            return False
        pm = QPixmap(str(p))
        if pm.isNull():
            self.clear(f"Cannot read image: {p.name}")
            return False
        self.path = p
        self.item.setPixmap(pm)
        self.scene.setSceneRect(QRectF(pm.rect()))
        self.empty.hide()
        self.view.show()
        if self.caption is not None:
            self.caption.setText(f"{p.name}  ·  {pm.width()}×{pm.height()}")
            self.caption.setToolTip(str(p))
        self.view.resetTransform()
        self.view._fit = True
        self.view.fit()
        return True

    def clear(self, msg="No image"):
        self.path = None
        self.item.setPixmap(QPixmap())
        self.view.hide()
        self.empty.setText(msg)
        self.empty.show()
        if self.caption is not None:
            self.caption.setText("")

    def actual_size(self):
        self.view.resetTransform()
        self.view._fit = False
