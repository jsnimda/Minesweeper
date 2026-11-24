"""Deterministic resize and padding with invertible coordinate transforms."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from PIL import Image

from .geometry import FloatArray, Geometry, Size, coordinate_array, validate_size


DEFAULT_CANVAS_SIZE = (384, 384)


def validate_canvas_size(size: object) -> Size:
    width, height = validate_size(size, name="canvas size")
    if width % 8 or height % 8:
        raise ValueError("canvas width and height must be divisible by eight")
    return width, height


def load_rgb_image(path: str | Path) -> Image.Image:
    """Read an exported geometry PNG and return a detached RGB image."""
    with Image.open(path, formats=["PNG"]) as image:
        return image.convert("RGB")


@dataclass(frozen=True)
class ImageTransform:
    """Map coordinates between an exported crop and its padded model canvas."""

    source_size: Size
    resized_size: Size
    canvas_size: Size
    offset: tuple[int, int]

    @property
    def scale_x(self) -> float:
        return self.resized_size[0] / self.source_size[0]

    @property
    def scale_y(self) -> float:
        return self.resized_size[1] / self.source_size[1]

    def to_canvas(self, coordinates: ArrayLike) -> FloatArray:
        """Map (..., 2) points or (..., 4) xyxy boxes to canvas coordinates."""
        values = _points_or_boxes(coordinates)
        point_count = values.shape[-1] // 2
        scales = (self.scale_x, self.scale_y) * point_count
        return values * scales + self.offset * point_count

    def to_source(self, coordinates: ArrayLike) -> FloatArray:
        """Map (..., 2) points or (..., 4) xyxy boxes back to crop pixels."""
        values = _points_or_boxes(coordinates)
        point_count = values.shape[-1] // 2
        scales = (self.scale_x, self.scale_y) * point_count
        return (values - self.offset * point_count) / scales


def _points_or_boxes(coordinates: ArrayLike) -> FloatArray:
    values = coordinate_array(coordinates, name="coordinates")
    if not values.ndim or values.shape[-1] not in (2, 4):
        raise ValueError("coordinates must end with two point or four xyxy box values")
    return values


def preprocess_image(
    image: Image.Image, canvas_size: Size = DEFAULT_CANVAS_SIZE,
) -> tuple[Image.Image, ImageTransform]:
    """Fit an image within a (width, height) canvas and fill the margins with black."""
    canvas_width, canvas_height = validate_canvas_size(canvas_size)
    source_width, source_height = validate_size(image.size, name="image size")
    fit = min(canvas_width / source_width, canvas_height / source_height)
    resized_width = max(1, round(source_width * fit))
    resized_height = max(1, round(source_height * fit))
    resized_size = (resized_width, resized_height)

    left = (canvas_width - resized_width) // 2
    top = (canvas_height - resized_height) // 2
    offset = (left, top)
    transform = ImageTransform(image.size, resized_size, (canvas_width, canvas_height), offset)

    if image.mode != "RGB":
        image = image.convert("RGB")
    resized = image.resize(resized_size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", transform.canvas_size)
    canvas.paste(resized, offset)
    return canvas, transform


def transform_geometry(geometry: Geometry, transform: ImageTransform) -> Geometry:
    """Apply the same resize and padding to all crop-local annotations."""
    if geometry.size != transform.source_size:
        raise ValueError("geometry size does not match the transform's source image")
    left, top = transform.offset
    return Geometry(
        size=transform.canvas_size,
        rows=geometry.rows,
        columns=geometry.columns,
        grid_box=transform.to_canvas(geometry.grid_box),
        x_edges=geometry.x_edges * transform.scale_x + left,
        y_edges=geometry.y_edges * transform.scale_y + top,
        cell_boxes=transform.to_canvas(geometry.cell_boxes),
        centers=transform.to_canvas(geometry.centers),
    )
