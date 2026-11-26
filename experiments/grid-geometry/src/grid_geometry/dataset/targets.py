"""Dense phase and soft occupancy targets in continuous image coordinates."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PhaseGeometry:
    """Unclipped board geometry in the coordinate system of the sample.

    The visible box describes the original board extent, not the crop. Coordinates
    may lie outside the image. Lattice origin and visible sides can differ.
    """
    origin_x: float
    origin_y: float
    pitch_x: float
    pitch_y: float
    visible_box: tuple[float, float, float, float]

    def __post_init__(self):
        values = (self.origin_x, self.origin_y, self.pitch_x, self.pitch_y, *self.visible_box)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("geometry must be finite")
        left, top, right, bottom = self.visible_box
        if self.pitch_x <= 0 or self.pitch_y <= 0 or left >= right or top >= bottom:
            raise ValueError("invalid pitches or visible box")

    def transformed(self, scale_x: float, scale_y: float, offset_x: float, offset_y: float):
        if not all(math.isfinite(v) for v in (scale_x, scale_y, offset_x, offset_y)) or min(scale_x, scale_y) <= 0:
            raise ValueError("transform must have positive finite scales")
        left, top, right, bottom = self.visible_box
        return PhaseGeometry(self.origin_x * scale_x + offset_x, self.origin_y * scale_y + offset_y,
                             self.pitch_x * scale_x, self.pitch_y * scale_y,
                             (left * scale_x + offset_x, top * scale_y + offset_y,
                              right * scale_x + offset_x, bottom * scale_y + offset_y))


def phase_targets(size: tuple[int, int], geometry: PhaseGeometry | None, *,
                  transition_ratio: float = .25, known_mask=None, visible_mask=None):
    """Return (5,H,W) targets, phase mask and occupancy mask.

    known_mask excludes padding and unknown/occluded pixels from both losses.
    visible_mask further restricts phase supervision to visible structure. A
    negative sample has no geometry and receives known occupancy-zero labels.
    Callers must supply masks for occlusion; no visibility is inferred there.
    """
    width, height = size
    if not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0:
        raise ValueError("size must contain positive integers")
    if not math.isfinite(transition_ratio) or transition_ratio <= 0:
        raise ValueError("transition_ratio must be finite and positive")
    shape = (height, width)
    def mask(value, name):
        result = np.ones(shape, dtype=bool) if value is None else np.asarray(value, dtype=bool)
        if result.shape != shape:
            raise ValueError(f"{name} shape must match image")
        return result
    known = mask(known_mask, "known_mask")
    visible = mask(visible_mask, "visible_mask")
    targets = np.zeros((5, height, width), dtype=np.float32)
    if geometry is None:
        return targets, np.zeros(shape, dtype=bool), known
    x = np.arange(width, dtype=np.float32) + .5
    y = np.arange(height, dtype=np.float32) + .5
    theta_x = (2 * np.pi / geometry.pitch_x) * (x - geometry.origin_x)
    theta_y = (2 * np.pi / geometry.pitch_y) * (y - geometry.origin_y)
    targets[0] = np.cos(theta_x)[None, :]
    targets[1] = np.sin(theta_x)[None, :]
    targets[2] = np.cos(theta_y)[:, None]
    targets[3] = np.sin(theta_y)[:, None]
    left, top, right, bottom = geometry.visible_box
    dx = np.minimum(x - left, right - x) / geometry.pitch_x
    dy = np.minimum(y - top, bottom - y) / geometry.pitch_y
    targets[4] = np.clip(.5 + np.minimum(dx[None, :], dy[:, None]) / transition_ratio, 0, 1)
    inside = (dx[None, :] >= 0) & (dy[:, None] >= 0)
    return targets, known & visible & inside, known
