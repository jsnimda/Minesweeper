
import sys

from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from minesweeper.devtools.mock_detector import MockGridDetector
from minesweeper.devtools.mock_game import MockGameWindow
from minesweeper.manager import GameManager
from minesweeper.ui.annotations import BoxAnnotation
from minesweeper.ui.overlay import DetectionOverlay


class DevControlPanel(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Minesweeper Development Panel")

        self.next_game_number = 1
        self.games: list[MockGameWindow] = []

        self.detector = MockGridDetector()
        self.manager = GameManager()
        self.overlay = DetectionOverlay()

        layout = QVBoxLayout(self)

        self.game_count_label = QLabel("Games created: 0")
        layout.addWidget(self.game_count_label)

        self.create_button = QPushButton("Create Game")
        self.create_button.clicked.connect(self.create_game)
        layout.addWidget(self.create_button)

        self.scan_button = QPushButton("Scan Games")
        self.scan_button.clicked.connect(self.scan_games)
        layout.addWidget(self.scan_button)

        self.detection_label = QLabel("Detections: 0")
        layout.addWidget(self.detection_label)

        self.overlay_button = QPushButton("Show Overlay")
        self.overlay_button.clicked.connect(self.toggle_overlay)
        layout.addWidget(self.overlay_button)

    def create_game(self):
        game = MockGameWindow(self.next_game_number)
        self.next_game_number += 1

        self.games.append(game)
        game.show()

        self.game_count_label.setText(
            f"Games created: {len(self.games)}"
        )

    def scan_games(self):
        detections = self.detector.detect(self.games)

        self.manager.update(detections)

        annotations = [
            BoxAnnotation(
                bbox=session.bbox,
                label=session.label,
            )
            for session in self.manager.sessions
            if session.missed_scans == 0
        ]

        self.overlay.set_annotations(annotations)

        self.detection_label.setText(
            f"Detections: {len(detections)}"
        )

    def toggle_overlay(self):
        if self.overlay.isVisible():
            self.overlay.hide()
        else:
            self.overlay.show_on_screen()

        self.overlay_button.setText(
            "Hide Overlay"
            if self.overlay.isVisible()
            else "Show Overlay"
        )

    def closeEvent(self, event):
        self.overlay.close()

        for game in self.games:
            game.close()

        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)

    panel = DevControlPanel()
    panel.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
