
from collections.abc import Iterable

from PySide6.QtCore import QPoint

from minesweeper.devtools.mock_game import MockGameWindow
from minesweeper.recognition.detection import GridDetection


class MockGridDetector:
    def detect(
        self,
        games: Iterable[MockGameWindow],
    ) -> list[GridDetection]:
        detections = []

        for game in games:
            if not game.isVisible() or not game.grid.isVisible():
                continue

            # Convert grid-local (0, 0) to desktop-global position.
            top_left = game.grid.mapToGlobal(QPoint(0, 0))

            detections.append(
                GridDetection(
                    x=top_left.x(),
                    y=top_left.y(),
                    width=game.grid.width(),
                    height=game.grid.height(),
                )
            )

        return detections
