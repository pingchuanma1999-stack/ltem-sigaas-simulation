"""Render the app icon (ltemsim_gui/assets/icon.png, 1024 px) and the macOS AppIcon.icns.

    .venv/bin/python tools/make_icon.py
"""
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "ltemsim_gui" / "assets"


def render(size=1024) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 1024.0
    # macOS icon grid: 824 px body, ~185 px corner radius
    body = QRectF(100 * s, 100 * s, 824 * s, 824 * s)
    g = QLinearGradient(body.topLeft(), body.bottomRight())
    g.setColorAt(0.0, QColor("#2b3bd1"))
    g.setColorAt(0.55, QColor("#4b3fe0"))
    g.setColorAt(1.0, QColor("#8a4dff"))
    path = QPainterPath()
    path.addRoundedRect(body, 185 * s, 185 * s)
    p.fillPath(path, QBrush(g))
    p.setClipPath(path)
    # soft glow where the pump hits the surface
    glow = QRadialGradient(QPointF(512 * s, 640 * s), 330 * s)
    glow.setColorAt(0.0, QColor(255, 196, 120, 170))
    glow.setColorAt(0.35, QColor(255, 140, 90, 70))
    glow.setColorAt(1.0, QColor(255, 140, 90, 0))
    p.fillRect(body, QBrush(glow))
    # semiconductor surface and the two electrodes
    surf_y = 668 * s
    p.fillRect(QRectF(100 * s, surf_y, 824 * s, 260 * s), QColor(10, 14, 60, 90))
    pen = QPen(QColor(255, 255, 255, 235), 14 * s)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.drawLine(QPointF(150 * s, surf_y), QPointF(874 * s, surf_y))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#ffd27a"))
    p.drawRoundedRect(QRectF(150 * s, surf_y - 30 * s, 170 * s, 30 * s), 8 * s, 8 * s)
    p.drawRoundedRect(QRectF(704 * s, surf_y - 30 * s, 170 * s, 30 * s), 8 * s, 8 * s)
    # single-cycle THz waveform radiated from the apex
    wave = QPainterPath()
    x0, x1, yc = 170 * s, 854 * s, 452 * s
    n = 240
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        u = (x - 512 * s) / (150 * s)
        y = yc - 150 * s * (-2 * u) * math.exp(-u * u) * 1.1
        if i == 0:
            wave.moveTo(QPointF(x, y))
        else:
            wave.lineTo(QPointF(x, y))
    pen = QPen(QColor("#7ff2d6"), 26 * s)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(wave)
    # AFM tip (cantilever + cone) above the gap, in front of the waveform
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(255, 255, 255))
    cone = QPainterPath()
    cone.moveTo(QPointF(452 * s, 300 * s))
    cone.lineTo(QPointF(572 * s, 300 * s))
    cone.lineTo(QPointF(518 * s, 586 * s))
    cone.quadTo(QPointF(512 * s, 600 * s), QPointF(506 * s, 586 * s))
    cone.closeSubpath()
    p.drawPath(cone)
    p.drawRoundedRect(QRectF(300 * s, 262 * s, 420 * s, 46 * s), 14 * s, 14 * s)
    p.end()
    return img


def main():
    if QApplication.instance() is None:
        main.app = QApplication(sys.argv[:1])      # QPainter needs a QApplication
    ASSETS.mkdir(parents=True, exist_ok=True)
    render(1024).save(str(ASSETS / "icon.png"))
    iconset = ASSETS / "AppIcon.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir()
    for base in (16, 32, 128, 256, 512):
        render(base).save(str(iconset / f"icon_{base}x{base}.png"))
        render(base * 2).save(str(iconset / f"icon_{base}x{base}@2x.png"))
    if sys.platform == "darwin" and shutil.which("iconutil"):
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(ASSETS / "AppIcon.icns")], check=True)
    shutil.rmtree(iconset)
    print("wrote", ASSETS / "icon.png", ASSETS / "AppIcon.icns")


if __name__ == "__main__":
    main()
