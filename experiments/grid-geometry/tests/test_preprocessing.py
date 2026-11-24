import numpy as np
from PIL import Image
import pytest

from grid_geometry.geometry import Geometry
from grid_geometry.preprocessing import load_rgb_image, preprocess_image, transform_geometry


def test_rgb_resize_preserves_aspect_and_centers_padding(tmp_path):
    path = tmp_path / "grayscale.png"
    Image.new("L", (101, 49), 180).save(path)

    image = load_rgb_image(path)
    canvas, transform = preprocess_image(image, (80, 64))

    assert image.mode == canvas.mode == "RGB"
    assert transform.source_size == (101, 49)
    assert transform.resized_size == (80, 39)
    assert transform.offset == (0, 12)
    assert transform.scale_x == 80 / 101
    assert transform.scale_y == 39 / 49
    assert transform.scale_x != transform.scale_y
    assert canvas.size == (80, 64)
    pixels = np.asarray(canvas)
    assert np.all(pixels[12:51] == 180)
    assert not pixels[:12].any()
    assert not pixels[51:].any()


def test_tall_source_is_centered_on_a_wide_canvas():
    image = Image.new("RGB", (15, 33), (11, 91, 190))

    canvas, transform = preprocess_image(image, (96, 64))

    assert transform.resized_size == (29, 64)
    assert transform.offset == (33, 0)
    assert canvas.getpixel((32, 0)) == (0, 0, 0)
    assert canvas.getpixel((33, 0)) == (11, 91, 190)
    assert canvas.getpixel((61, 63)) == (11, 91, 190)
    assert canvas.getpixel((62, 63)) == (0, 0, 0)


def test_geometry_resizes_and_round_trips_fractional_points_and_boxes(annotation):
    source = Geometry(**{
        key: annotation[key]
        for key in ("size", "rows", "columns", "grid_box", "x_edges", "y_edges", "cell_boxes", "centers")
    })
    _, transform = preprocess_image(Image.new("RGB", source.size), (80, 72))
    transformed = transform_geometry(source, transform)

    assert transform.resized_size == (80, 49)
    assert transform.offset == (0, 11)
    assert transformed.size == (80, 72)
    assert transformed.rows == source.rows
    assert transformed.columns == source.columns
    np.testing.assert_allclose(
        transformed.centers[0, 0],
        (5.75 * 80 / 31, 11 + 4.5 * 49 / 19),
    )
    for field in ("grid_box", "cell_boxes", "centers"):
        np.testing.assert_allclose(transform.to_source(getattr(transformed, field)), getattr(source, field))
    np.testing.assert_allclose(transformed.x_edges, source.x_edges * 80 / 31)
    np.testing.assert_allclose(transformed.y_edges, source.y_edges * 49 / 19 + 11)
    assert transformed.cell_boxes[0, 0, 2] < transformed.x_edges[1]
    assert transformed.x_edges[1] < transformed.cell_boxes[0, 1, 0]


@pytest.mark.parametrize("size", [(0, 32), (-8, 32), (8, 9), (32.0, 32), (32, float("inf")), (True, 8)])
def test_rejects_invalid_canvas_sizes(size):
    with pytest.raises(ValueError, match="canvas"):
        preprocess_image(Image.new("RGB", (4, 3)), size)


def test_rejects_invalid_coordinates():
    _, transform = preprocess_image(Image.new("RGB", (8, 8)))

    with pytest.raises(ValueError, match="finite"):
        transform.to_source([float("nan"), 1.5])
    with pytest.raises(ValueError, match="xyxy"):
        transform.to_canvas([1, 2, 3])
