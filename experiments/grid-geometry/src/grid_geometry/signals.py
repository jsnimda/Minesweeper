"""Project geometry probabilities onto the image's x and y axes."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
ProjectionMethod = Literal["mean", "max", "rms", "top_k"]


@dataclass(frozen=True, eq=False)
class AxisSignals:
    """Raw center and boundary evidence at half-integer pixel coordinates."""

    centers: FloatArray
    boundaries: FloatArray


@dataclass(frozen=True, eq=False)
class HeatmapSignals:
    x: AxisSignals
    y: AxisSignals


def project_heatmaps(
    heatmaps: ArrayLike,
    *,
    method: ProjectionMethod = "mean",
    top_k_fraction: float = 0.1,
) -> HeatmapSignals:
    """Project [centers, vertical, horizontal] heatmaps without renormalizing.

    Top-k averages the strongest fraction of pixels along the reduced axis
    independently at each retained coordinate.
    """
    values = probability_array(heatmaps)
    if values.ndim != 3 or values.shape[0] != 3 or min(values.shape[1:]) == 0:
        raise ValueError("heatmaps must have shape [3, H, W] with positive H and W")
    if method not in ("mean", "max", "rms", "top_k"):
        raise ValueError("method must be 'mean', 'max', 'rms', or 'top_k'")
    if (
        isinstance(top_k_fraction, (bool, np.bool_))
        or not isinstance(top_k_fraction, Real)
        or not 0 < top_k_fraction <= 1
    ):
        raise ValueError("top_k_fraction must be in (0, 1]")

    def reduce(channel: FloatArray, axis: int) -> FloatArray:
        if method == "mean":
            return channel.mean(axis=axis)
        if method == "max":
            return channel.max(axis=axis)
        if method == "rms":
            return np.sqrt(np.mean(channel**2, axis=axis))
        along_last_axis = np.moveaxis(channel, axis, -1)
        count = math.ceil(along_last_axis.shape[-1] * top_k_fraction)
        start = along_last_axis.shape[-1] - count
        strongest = np.partition(along_last_axis, start, axis=-1)[..., start:]
        return strongest.mean(axis=-1)

    centers, vertical, horizontal = values
    return HeatmapSignals(
        x=AxisSignals(reduce(centers, axis=0), reduce(vertical, axis=0)),
        y=AxisSignals(reduce(centers, axis=1), reduce(horizontal, axis=1)),
    )


def probability_array(values: ArrayLike) -> FloatArray:
    """Convert real numeric probabilities to float64 and validate their range."""
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise ValueError("probabilities must be real numbers")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return array
