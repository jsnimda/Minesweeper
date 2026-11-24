"""Validated rectangular lattice annotations in continuous image coordinates."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


Size = tuple[int, int]
FloatArray = NDArray[np.float64]
_COORDINATE_TOLERANCE = 1e-7


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


@dataclass(frozen=True, eq=False)
class Geometry:
    """Grid extent, lattice edges, and visible tile geometry for one image."""

    size: Size
    rows: int
    columns: int
    grid_box: FloatArray
    x_edges: FloatArray
    y_edges: FloatArray
    cell_boxes: FloatArray
    centers: FloatArray

    def __post_init__(self) -> None:
        width, height = validate_size(self.size)
        rows, columns = validate_size((self.rows, self.columns), name="rows and columns")
        object.__setattr__(self, "size", (width, height))
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)

        for name, shape in (
            ("grid_box", (4,)),
            ("x_edges", (columns + 1,)),
            ("y_edges", (rows + 1,)),
            ("cell_boxes", (rows, columns, 4)),
            ("centers", (rows, columns, 2)),
        ):
            coordinates = coordinate_array(getattr(self, name), name=name)
            if coordinates.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            coordinates = coordinates.copy()
            coordinates.setflags(write=False)
            object.__setattr__(self, name, coordinates)

        self._validate_lattice(width, height)
        self._validate_tiles()

    def _validate_lattice(self, width: int, height: int) -> None:
        left, top, right, bottom = self.grid_box
        tolerance = _COORDINATE_TOLERANCE
        if not (
            -tolerance <= left < right <= width + tolerance
            and -tolerance <= top < bottom <= height + tolerance
        ):
            raise ValueError("grid_box must be inside the image and have a positive extent")

        for name, edges, bounds in (
            ("x_edges", self.x_edges, (left, right)),
            ("y_edges", self.y_edges, (top, bottom)),
        ):
            if np.any(np.diff(edges) <= 0):
                raise ValueError(f"{name} must be strictly increasing")
            if not np.allclose(edges[[0, -1]], bounds, rtol=0, atol=tolerance):
                raise ValueError(f"{name} endpoints must coincide with grid_box")

    def _validate_tiles(self) -> None:
        left, top, right, bottom = self.cell_boxes.transpose(2, 0, 1)
        tolerance = _COORDINATE_TOLERANCE
        if np.any(right <= left) or np.any(bottom <= top):
            raise ValueError("cell_boxes must have positive extents")
        if (
            np.any(left < self.x_edges[:-1] - tolerance)
            or np.any(right > self.x_edges[1:] + tolerance)
            or np.any(top < self.y_edges[:-1, None] - tolerance)
            or np.any(bottom > self.y_edges[1:, None] + tolerance)
        ):
            raise ValueError("cell_boxes must be inside their lattice slots")

        tile_centers = (self.cell_boxes[..., :2] + self.cell_boxes[..., 2:]) / 2
        if not np.allclose(self.centers, tile_centers, rtol=0, atol=tolerance):
            raise ValueError("centers must be the midpoints of the visible cell_boxes")
        center_x = self.centers[..., 0]
        center_y = self.centers[..., 1]
        if (
            not np.allclose(center_x, center_x[:1], rtol=0, atol=tolerance)
            or not np.allclose(center_y, center_y[:, :1], rtol=0, atol=tolerance)
        ):
            raise ValueError("centers must form an axis-aligned rectangular grid")
