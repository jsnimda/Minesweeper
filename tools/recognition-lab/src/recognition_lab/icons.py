"""Palette-aware Lucide icons rendered as vectors by Qt."""
from functools import lru_cache

from lucide import lucide_icon
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QIcon, QIconEngine, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer


class _SvgIconEngine(QIconEngine):
    def __init__(self, normal_svg, disabled_svg):
        super().__init__()
        self.normal_svg = normal_svg
        self.disabled_svg = disabled_svg
        self.normal = QSvgRenderer(normal_svg)
        self.disabled = QSvgRenderer(disabled_svg)

    def clone(self):
        return _SvgIconEngine(self.normal_svg, self.disabled_svg)

    def paint(self, painter, rect, mode, state):
        renderer = self.disabled if mode == QIcon.Mode.Disabled else self.normal
        renderer.render(painter, QRectF(rect))

    def pixmap(self, size, mode, state):
        return self.scaledPixmap(size, mode, state, 1.0)

    def scaledPixmap(self, size, mode, state, scale):
        pixmap = QPixmap(max(1, round(size.width() * scale)), max(1, round(size.height() * scale)))
        pixmap.setDevicePixelRatio(scale)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        self.paint(painter, QRectF(0, 0, size.width(), size.height()), mode, state)
        painter.end()
        return pixmap


@lru_cache(maxsize=128)
def _icon(name, foreground, disabled):
    normal_svg = lucide_icon(name, stroke=foreground).encode('utf-8')
    disabled_svg = lucide_icon(name, stroke=disabled).encode('utf-8')
    return QIcon(_SvgIconEngine(normal_svg, disabled_svg))


def icon(name, palette):
    """Return a scalable icon colored for the supplied application palette."""
    foreground = palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.ButtonText).name()
    disabled = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText).name()
    return _icon(name, foreground, disabled)
