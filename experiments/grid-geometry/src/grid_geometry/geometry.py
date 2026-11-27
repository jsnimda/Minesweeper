"""Validated rectangular lattice annotations in continuous image coordinates."""

from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


Size = tuple[int, int]
FloatArray = NDArray[np.float64]


def validate_size(size: object, *, name: str = "size") -> Size:
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError(f"{name} must contain two positive integers")
    if any(
        not isinstance(value, Integral) or isinstance(value, bool) or value <= 0
        for value in size
    ):
        raise ValueError(f"{name} must contain two positive integers")
    return int(size[0]), int(size[1])


def coordinate_array(values: ArrayLike, *, name: str) -> FloatArray:
    """Convert numeric coordinates while rejecting malformed and nonfinite input."""
    try:
        if isinstance(values, np.ndarray):
            if values.dtype.kind not in "iuf":
                raise ValueError
        else:
            elements = np.asarray(values, dtype=object)
            if any(
                not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
                for value in elements.flat
            ):
                raise ValueError
        coordinates = np.asarray(values, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{name} must contain finite numbers")
    return coordinates
