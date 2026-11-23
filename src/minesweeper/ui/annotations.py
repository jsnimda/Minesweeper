
from dataclasses import dataclass

from minesweeper.recognition.detection import GridDetection


@dataclass(frozen=True)
class BoxAnnotation:
    bbox: GridDetection
    label: str
    color: str = "#00d4ff"
