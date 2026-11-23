
from dataclasses import dataclass


@dataclass(frozen=True)
class GridDetection:
    """Grid rectangle in desktop-global Qt logical coordinates."""

    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
