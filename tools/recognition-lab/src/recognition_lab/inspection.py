"""Source-pixel inspection in continuous image coordinates."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PixelInspection:
    x: float
    y: float
    pixel: tuple[int, int] | None
    rgba: tuple[int, int, int, int] | None


def inspect_pixel(image, x, y):
    """Inspect an RGBA PIL image; integer pixel boundaries are half-open."""
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError('coordinates must be finite')
    if 0 <= x < image.width and 0 <= y < image.height:
        index = (math.floor(x), math.floor(y))
        return PixelInspection(x, y, index, image.getpixel(index))
    return PixelInspection(x, y, None, None)
