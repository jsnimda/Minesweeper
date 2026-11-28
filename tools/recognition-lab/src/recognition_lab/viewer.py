"""Reusable Skia image viewer in logical widget coordinates."""
import math

import numpy as np
import skia
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QCursor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

from .inspection import inspect_pixel
from .minimap import Minimap, centered_transform
from .viewport import (ViewportTransform, WHEEL_STEP_FACTOR, clamp_magnification,
                       logical_scale, physical_scale, reanchor_viewport,
                       wheel_factor, zoom_by_factor, visible_pixel_grid)


def surface_pixmap(surface):
    """Copy premultiplied RGBA pixels so the pixmap owns its data."""
    snapshot = surface.makeImageSnapshot()
    pixels = snapshot.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType, alphaType=skia.AlphaType.kPremul_AlphaType)
    h, w = pixels.shape[:2]
    return QPixmap.fromImage(QImage(pixels.data, w, h, pixels.strides[0], QImage.Format.Format_RGBA8888_Premultiplied).copy())


def image_sampling(mode):
    """Select Skia sampling for the source image only."""
    filters = {'nearest': skia.FilterMode.kNearest, 'smooth': skia.FilterMode.kLinear}
    if mode not in filters:
        raise ValueError('unknown image sampling mode')
    return skia.SamplingOptions(filters[mode])


class ImageViewer(QWidget):
    transform_changed = Signal(object)
    magnification_changed = Signal(float)  # physical pixels per source pixel
    cursor_changed = Signal(object)
    cursor_left = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.transform = ViewportTransform()
        self.image = None
        self.image_size = None
        self.source_image = None
        self.visualization = None
        self.sampling_mode = 'nearest'
        self.pixel_grid_visible = False
        self.mode = 'fit'
        self.drag = None
        self.drag_button = None
        self.pointer = None
        self._pinch_anchor = None
        self._dpr = self.devicePixelRatioF()
        self._viewport_size = (self.width(), self.height())
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(200, 200)
        self.minimap = Minimap(self)

    def set_image(self, image):
        """Replace the image, clear old visualization and enter fit mode."""
        self.source_image = None if image is None else image.convert('RGBA').copy()
        self.image = None if image is None else skia.Image.fromarray(np.asarray(self.source_image).copy())
        self.image_size = None if image is None else image.size
        self.visualization = None
        self._clear_interaction()
        self.minimap.set_source(self.source_image)
        self.mode = 'fit'
        self.fit()
        self.update()

    def set_sampling(self, mode):
        image_sampling(mode)  # Validate before changing state.
        self.sampling_mode = mode
        self.update()

    def set_pixel_grid_visible(self, visible):
        self.pixel_grid_visible = bool(visible)
        self.update()

    def set_visualization(self, render):
        """Set a callback receiving the canvas, physical magnification and device pixel ratio."""
        self.visualization = render
        self.update()

    def _apply_transform(self, transform):
        self.transform = transform
        self.update()
        self.transform_changed.emit(transform)
        self.magnification_changed.emit(self.magnification())
        self._refresh_cursor()

    def set_transform(self, transform):
        """Set an explicit logical transform and enter manual navigation mode."""
        if not isinstance(transform, ViewportTransform):
            raise TypeError('expected a ViewportTransform')
        self.mode = 'manual'
        self._apply_transform(transform)

    def magnification(self):
        """Return physical rendering pixels per source image pixel."""
        return physical_scale(self.transform.scale, self._dpr)

    def _center(self):
        return self.width() / 2, self.height() / 2

    def set_magnification(self, magnification, anchor=None):
        """Set bounded physical magnification around a logical viewport anchor."""
        if self.image is None:
            return
        current = self.magnification()
        target = clamp_magnification(current, magnification)
        if target == current:
            return
        anchor = self._center() if anchor is None else anchor
        self.set_transform(self.transform.scaled_at(logical_scale(target, self._dpr), *anchor))

    def zoom_by(self, factor, anchor=None):
        """Apply a bounded multiplicative zoom, centered unless an anchor is supplied."""
        if self.image is None:
            return
        anchor = self._center() if anchor is None else anchor
        transform = zoom_by_factor(self.transform, self._dpr, factor, anchor)
        if transform != self.transform:
            self.set_transform(transform)

    def center_on_image(self, x, y):
        if self.image is not None:
            self.set_transform(centered_transform(self.transform, (self.width(), self.height()), (x, y)))

    def set_minimap_enabled(self, enabled):
        self.minimap.set_enabled(enabled)

    def fit(self):
        self.mode = 'fit'
        if self.image_size and self.width() > 0 and self.height() > 0:
            self._apply_transform(ViewportTransform.fit(*self.image_size, self.width(), self.height()))

    def actual_size(self):
        if self.image is not None:
            self.set_transform(self.transform.scaled_at(logical_scale(1.0, self._dpr), *self._center()))

    def _clear_interaction(self):
        self.drag = None
        self.drag_button = None
        self.pointer = None
        self._pinch_anchor = None
        self.unsetCursor()
        self.cursor_left.emit()

    def _cursor_position(self):
        """Return the actual cursor in viewer-local logical coordinates, if inside."""
        p = self.mapFromGlobal(QCursor.pos())
        return (float(p.x()), float(p.y())) if self.rect().contains(p) else None

    def _gesture_anchor(self, event, cursor):
        # Cursor-centering is our interaction policy; native event positions can
        # differ by platform. Both candidates are viewer-local logical points.
        if cursor is not None:
            return cursor
        p = event.position()
        x, y = p.x(), p.y()
        if math.isfinite(x) and math.isfinite(y) and 0 <= x < self.width() and 0 <= y < self.height():
            return (x, y)
        return None

    def inspect_at(self, x, y):
        """Inspect a viewer-local logical position against the original source."""
        if self.source_image is None:
            return None
        return inspect_pixel(self.source_image, *self.transform.to_image(x, y))

    def _refresh_cursor(self):
        if self.pointer is not None and self.image is not None:
            self.cursor_changed.emit(self.inspect_at(*self.pointer))

    def _sync_display(self, new_size=None):
        """Handle resize and DPR changes together, irrespective of event order."""
        new_size = new_size or (self.width(), self.height())
        if min(new_size) <= 0:
            return  # Keep the last usable center and DPR until the viewport returns.
        old_size, old_dpr = self._viewport_size, self._dpr
        new_dpr = self.devicePixelRatioF()
        if new_size == old_size and new_dpr == old_dpr:
            return
        self._viewport_size, self._dpr = new_size, new_dpr
        if self.pointer is not None:
            p = self.mapFromGlobal(QCursor.pos())
            if self.rect().contains(p):
                self.pointer = (float(p.x()), float(p.y()))
            else:
                self.pointer = None
                self.cursor_left.emit()
        if self.image_size and min(new_size) > 0:
            if self.mode == 'fit':
                self.fit()
            elif min(old_size) > 0:
                self._apply_transform(reanchor_viewport(self.transform, old_size, new_size, old_dpr, new_dpr))
            else:
                self.magnification_changed.emit(self.magnification())
                self._refresh_cursor()
        self.update()

    def event(self, event):
        if not hasattr(self, '_dpr'):
            return super().event(event)
        kind = event.type()
        if kind == QEvent.Type.NativeGesture and self.image is not None:
            gesture = event.gestureType()
            if gesture == Qt.NativeGestureType.BeginNativeGesture:
                self._pinch_anchor = self._gesture_anchor(event, self._cursor_position())
                event.accept()
                return True
            if gesture == Qt.NativeGestureType.ZoomNativeGesture:
                cursor = self._cursor_position()
                self.pointer = cursor
                if cursor is None:
                    self.cursor_left.emit()
                if self._pinch_anchor is None:
                    self._pinch_anchor = self._gesture_anchor(event, cursor)
                # Qt reports an incremental fractional change: 0.1 means +10%.
                factor = 1.0 + event.value()
                if self._pinch_anchor is not None and math.isfinite(factor) and factor > 0:
                    self.zoom_by(factor, self._pinch_anchor)
                self._refresh_cursor()
                event.accept()
                return True
            if gesture == Qt.NativeGestureType.EndNativeGesture:
                self._pinch_anchor = None
                event.accept()
                return True
        if kind in (QEvent.Type.FocusOut, QEvent.Type.WindowDeactivate, QEvent.Type.UngrabMouse, QEvent.Type.Hide):
            self._clear_interaction()
        result = super().event(event)
        if kind in (QEvent.Type.DevicePixelRatioChange, QEvent.Type.ScreenChangeInternal, QEvent.Type.Show):
            self._sync_display()
        return result

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_dpr'):
            self._sync_display((event.size().width(), event.size().height()))
            if hasattr(self, 'minimap'):
                self.minimap.refresh()

    def paintEvent(self, event):
        if self.width() <= 0 or self.height() <= 0: return
        dpr = self.devicePixelRatioF()
        surface = skia.Surface(max(1, round(self.width()*dpr)), max(1, round(self.height()*dpr)))
        canvas = surface.getCanvas()
        canvas.clear(0xff272727)
        # The surface is physical pixels; all viewer and image transforms are logical.
        canvas.scale(dpr, dpr)
        if self.image is not None:
            tr = self.transform
            canvas.translate(tr.offset_x, tr.offset_y)
            canvas.scale(tr.scale, tr.scale)
            canvas.drawImage(self.image, 0, 0, image_sampling(self.sampling_mode))
            # The pixel grid is above the source image, below heatmaps and vectors.
            if self.pixel_grid_visible:
                self._draw_pixel_grid(canvas, dpr)
            if self.visualization is not None:
                self.visualization(canvas, physical_scale(tr.scale, dpr), dpr)
        painter = QPainter(self)
        pixmap = surface_pixmap(surface); pixmap.setDevicePixelRatio(dpr)
        painter.drawPixmap(0, 0, pixmap)

    def _draw_pixel_grid(self, canvas, dpr):
        visible = visible_pixel_grid(self.transform, self.image_size,
                                     (self.width(), self.height()), dpr)
        if visible is None:
            return
        (left, top, right, bottom), (x0, x1), (y0, y1) = visible
        # A physical-pixel stroke; positions remain integer source boundaries.
        width = 1.0 / physical_scale(self.transform.scale, dpr)
        paint = skia.Paint(Color=0x99808080, StrokeWidth=width, AntiAlias=False)
        canvas.save()
        canvas.clipRect(skia.Rect.MakeXYWH(0, 0, *self.image_size))
        for x in range(x0, x1 + 1):
            canvas.drawLine(x, top, x, bottom, paint)
        for y in range(y0, y1 + 1):
            canvas.drawLine(left, y, right, y, paint)
        canvas.restore()

    def wheelEvent(self, event):
        if self.image is None:
            event.ignore()
            return
        pixel = event.pixelDelta()
        # A pixel-delta event takes precedence even when it is horizontal-only.
        pixel_present = not pixel.isNull()
        factor = wheel_factor(pixel.y(), 0 if pixel_present else event.angleDelta().y())
        if factor != 1:
            p = event.position()
            self.pointer = (p.x(), p.y())
            self.zoom_by(factor, self.pointer)
            self._refresh_cursor()
        event.accept()

    def keyPressEvent(self, event):
        if self.image is not None and not (event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier)):
            if event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
                self.zoom_by(WHEEL_STEP_FACTOR)
                event.accept(); return
            if event.key() == Qt.Key.Key_Minus:
                self.zoom_by(1 / WHEEL_STEP_FACTOR)
                event.accept(); return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if self.image is not None and self.drag is None and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self.drag = event.position()
            self.drag_button = event.button()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == self.drag_button:
            self.drag = None
            self.drag_button = None
            self.unsetCursor()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        p = event.position()
        self.pointer = (p.x(), p.y())
        if self.drag is not None:
            if not (event.buttons() & self.drag_button):
                self.drag = None
                self.drag_button = None
                self.unsetCursor()
            else:
                dx, dy = p.x()-self.drag.x(), p.y()-self.drag.y()
                if dx or dy:
                    self.set_transform(self.transform.panned(dx, dy))
                self.drag = p
        self._refresh_cursor()

    def leaveEvent(self, event):
        self.pointer = None
        self.cursor_left.emit()
        super().leaveEvent(event)
