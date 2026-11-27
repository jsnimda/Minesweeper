import numpy as np
from PIL import Image
import pytest

from grid_geometry.preprocessing import preprocess_image


def test_rgb_resize_preserves_aspect_and_centers_padding():
    image = Image.new("L", (101, 49), 180)
    canvas, transform = preprocess_image(image, (80, 64))

    assert canvas.mode == "RGB"
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


def test_fractional_points_and_boxes_round_trip():
    _, transform = preprocess_image(Image.new("RGB", (31, 19)), (80, 72))
    assert transform.resized_size == (80, 49)
    assert transform.offset == (0, 11)
    point = [5.75, 4.5]
    box = [1.25, 2.5, 15.75, 10.25]
    np.testing.assert_allclose(transform.to_canvas(point), (5.75 * 80 / 31, 11 + 4.5 * 49 / 19))
    np.testing.assert_allclose(transform.to_canvas(box), (1.25 * 80 / 31, 11 + 2.5 * 49 / 19, 15.75 * 80 / 31, 11 + 10.25 * 49 / 19))
    for coordinates in (point, box):
        np.testing.assert_allclose(transform.to_source(transform.to_canvas(coordinates)), coordinates)


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
