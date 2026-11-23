
from dataclasses import dataclass, field
from uuid import uuid4

from minesweeper.recognition.detection import GridDetection


@dataclass
class GameSession:
    number: int
    bbox: GridDetection

    id: str = field(default_factory=lambda: uuid4().hex)
    missed_scans: int = 0

    @property
    def label(self) -> str:
        return f"Minesweeper Game {self.number} · {self.id[:4]}"
