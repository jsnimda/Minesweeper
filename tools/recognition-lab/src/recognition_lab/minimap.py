"""Source image minimap and its logical coordinate calculations."""
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget
from PIL import Image

from .viewport import ViewportTransform


def thumbnail_transform(image_size, widget_size):
    return ViewportTransform.fit(*image_size, *widget_size)


def minimap_size(image_size):
    """Fit the image into the maximum interior with a usable minimum dimension."""
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError('image dimensions must be positive')
    scale = min(176 / width, 136 / height)
    return max(24, round(width * scale)) + 4, max(24, round(height * scale)) + 4


def visible_region(transform, image_size, viewport_size):
    w, h = image_size
    vw, vh = viewport_size
    if min(w, h, vw, vh) <= 0:
        return None, False
    left, top = transform.to_image(0, 0)
    right, bottom = transform.to_image(vw, vh)
    fully_visible = left <= 0 and top <= 0 and right >= w and bottom >= h
    clipped = (max(0., left), max(0., top), min(float(w), right), min(float(h), bottom))
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        return None, fully_visible
    return clipped, fully_visible


def centered_transform(transform, viewport_size, point):
    return ViewportTransform(transform.scale, viewport_size[0] / 2 - point[0] * transform.scale,
                             viewport_size[1] / 2 - point[1] * transform.scale)


class Minimap(QWidget):
    def __init__(self, viewer):
        super().__init__(viewer)
        self.viewer = viewer
        self.enabled = True
        self.thumbnail = None
        self.drag_start = None
        self.drag_center = None
        self.setFixedSize(180, 140)
        self.hide()
        viewer.transform_changed.connect(self.refresh)

    def set_source(self, image):
        self.drag_start = None
        self.thumbnail = None
        if image is not None:
            self.setFixedSize(*minimap_size(image.size))
            copy = image.convert('RGBA')
            copy.thumbnail((176, 136), Image.Resampling.LANCZOS)
            data = copy.tobytes('raw', 'RGBA')
            self.thumbnail = QImage(data, copy.width, copy.height, copy.width * 4, QImage.Format.Format_RGBA8888).copy()
        self.refresh()

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        self.refresh()

    def refresh(self, *args):
        v = self.viewer
        self.move(max(0, v.width() - self.width() - 8), max(0, v.height() - self.height() - 8))
        if v.image_size is None:
            self.hide(); return
        _, entire = visible_region(v.transform, v.image_size, (v.width(), v.height()))
        self.setVisible(self.enabled and not entire and self.thumbnail is not None and v.width() >= self.width() and v.height() >= self.height())
        self.update()

    def _map(self):
        return thumbnail_transform(self.viewer.image_size, (self.width() - 4, self.height() - 4)).panned(2, 2)

    def paintEvent(self, event):
        if self.thumbnail is None: return
        v = self.viewer
        tr = self._map()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(35, 35, 35, 230))
        painter.setPen(QPen(QColor(160, 160, 160, 180), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        w, h = v.image_size
        painter.drawImage(QRectF(tr.offset_x, tr.offset_y, w * tr.scale, h * tr.scale), self.thumbnail)
        bounds, _ = visible_region(v.transform, v.image_size, (v.width(), v.height()))
        if bounds:
            x0, y0 = tr.to_viewport(bounds[0], bounds[1])
            x1, y1 = tr.to_viewport(bounds[2], bounds[3])
            painter.setPen(QPen(QColor('white'), 1))
            painter.setBrush(QColor(40, 170, 255, 65))
            painter.drawRect(QRectF(x0, y0, x1-x0, y1-y0))

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton: return
        tr = self._map()
        x, y = tr.to_image(event.position().x(), event.position().y())
        w, h = self.viewer.image_size
        if not (0 <= x <= w and 0 <= y <= h): return
        bounds, _ = visible_region(self.viewer.transform, self.viewer.image_size, (self.viewer.width(), self.viewer.height()))
        self.drag_start = (x, y)
        if bounds and bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]:
            self.drag_center = self.viewer.transform.to_image(self.viewer.width()/2, self.viewer.height()/2)
        else:
            self.drag_center = (x, y)
            self.viewer.center_on_image(x, y)
        event.accept()

    def mouseMoveEvent(self, event):
        if self.drag_start is None: return
        x, y = self._map().to_image(event.position().x(), event.position().y())
        w, h = self.viewer.image_size
        x, y = max(0., min(w, x)), max(0., min(h, y))
        self.viewer.center_on_image(self.drag_center[0] + x - self.drag_start[0],
                                    self.drag_center[1] + y - self.drag_start[1])
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start = self.drag_center = None
            event.accept()
