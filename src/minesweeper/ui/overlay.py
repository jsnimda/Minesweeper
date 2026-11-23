
from collections.abc import Iterable

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from minesweeper.ui.annotations import BoxAnnotation


class DetectionOverlay(QWidget):
    def __init__(self):
        super().__init__()

        self.annotations: list[BoxAnnotation] = []

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.Tool
        )

        self.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground
        )

    def set_annotations(
        self,
        annotations: Iterable[BoxAnnotation],
    ):
        self.annotations = list(annotations)
        self.update()

    def show_on_screen(self):
        screen = QApplication.primaryScreen()

        if screen is None:
            raise RuntimeError("No screen available")

        self.setGeometry(screen.geometry())
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for annotation in self.annotations:
            detection = annotation.bbox
            color = QColor(annotation.color)

            top_left = self.mapFromGlobal(
                QPoint(detection.x, detection.y)
            )

            rect = QRect(
                top_left.x(),
                top_left.y(),
                detection.width,
                detection.height,
            )

            painter.setPen(QPen(color, 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)

            label = annotation.label

            padding = 6
            metrics = painter.fontMetrics()
            label_width = metrics.horizontalAdvance(label) + padding * 2
            label_height = metrics.height() + padding * 2

            label_y = rect.top() - label_height
            if label_y < 0:
                label_y = rect.top()

            label_rect = QRect(
                rect.left(),
                label_y,
                label_width,
                label_height,
            )

            painter.fillRect(label_rect, color)
            painter.setPen(QColor("#111111"))
            painter.drawText(
                label_rect.adjusted(padding, 0, -padding, 0),
                Qt.AlignmentFlag.AlignVCenter,
                label,
            )
