"""Spatial sampling in continuous image coordinates."""

import numpy as np

from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def sample_bilinear(field: FloatArray, x: FloatArray, y: FloatArray) -> FloatArray:
    """Bilinear samples on a separable grid, with zero beyond the canvas."""
    height, width = field.shape
    valid_x, valid_y = (x >= 0) & (x <= width), (y >= 0) & (y <= height)
    x_clamped, y_clamped = (x - 0.5).clip(0, width - 1), (y - 0.5).clip(0, height - 1)
    left, top = x_clamped.astype(int), y_clamped.astype(int)
    right, bottom = np.minimum(left + 1, width - 1), np.minimum(top + 1, height - 1)
    fx, fy = x_clamped - left, (y_clamped - top)[:, None]
    upper = field[top[:, None], left] * (1 - fx) + field[top[:, None], right] * fx
    lower = field[bottom[:, None], left] * (1 - fx) + field[bottom[:, None], right] * fx
    return (upper * (1 - fy) + lower * fy) * valid_y[:, None] * valid_x
