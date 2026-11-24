"""Rasterize transformed grid geometry into three Gaussian target maps."""

import math
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from .geometry import FloatArray, Geometry


DEFAULT_SIGMA = 2.0


def validate_sigma(sigma: float) -> float:
    if not isinstance(sigma, Real) or isinstance(sigma, (bool, np.bool_)):
        raise ValueError("sigma must be finite and positive")
    try:
        sigma = float(sigma)
    except OverflowError as exc:
        raise ValueError("sigma must be finite and positive") from exc
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    return sigma


def _nearest_squared_distance(pixels: FloatArray, positions: FloatArray) -> FloatArray:
    return np.min((pixels[:, None] - positions[None, :]) ** 2, axis=1)


def _gaussian(squared_distance: FloatArray, sigma: float) -> FloatArray:
    # Divide twice so that very small or very large finite sigma values work.
    with np.errstate(over="ignore", under="ignore"):
        normalized_distance = squared_distance / sigma / sigma
        return np.exp(-0.5 * normalized_distance)


def geometry_heatmaps(
    geometry: Geometry, *, sigma: float = DEFAULT_SIGMA, include_outer: bool = True,
) -> NDArray[np.float32]:
    """Return [3, H, W] targets ordered as centers, vertical and horizontal edges."""
    sigma = validate_sigma(sigma)
    width, height = geometry.size
    x = np.arange(width, dtype=np.float64) + 0.5
    y = np.arange(height, dtype=np.float64) + 0.5

    # On a rectangular lattice, the nearest center can be found one axis at a time.
    dx = _nearest_squared_distance(x, geometry.centers[0, :, 0])
    dy = _nearest_squared_distance(y, geometry.centers[:, 0, 1])
    centers = _gaussian(dx[None, :] + dy[:, None], sigma)

    x_edges = geometry.x_edges if include_outer else geometry.x_edges[1:-1]
    y_edges = geometry.y_edges if include_outer else geometry.y_edges[1:-1]
    vertical_profile = (
        _gaussian(_nearest_squared_distance(x, x_edges), sigma) if x_edges.size else np.zeros(width)
    )
    horizontal_profile = (
        _gaussian(_nearest_squared_distance(y, y_edges), sigma) if y_edges.size else np.zeros(height)
    )
    left, top, right, bottom = geometry.grid_box
    inside_y = (y >= top) & (y < bottom)
    inside_x = (x >= left) & (x < right)
    vertical = inside_y[:, None] * vertical_profile[None, :]
    horizontal = horizontal_profile[:, None] * inside_x[None, :]
    return np.stack((centers, vertical, horizontal)).astype(np.float32)
