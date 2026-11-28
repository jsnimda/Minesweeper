import pytest
from PIL import Image

from recognition_lab.inspection import inspect_pixel


@pytest.mark.parametrize(('point', 'index', 'rgba'), [
    ((0, 0), (0, 0), (255, 0, 0, 255)),
    ((0.5, 0.5), (0, 0), (255, 0, 0, 255)),
    ((0.999, 0.5), (0, 0), (255, 0, 0, 255)),
    ((1, 0.5), (1, 0), (0, 255, 0, 128)),
    ((-0.001, 0), None, None),
    ((2, 0), None, None),
    ((0, 1), None, None),
])
def test_source_pixel_boundaries(point, index, rgba):
    image = Image.new('RGBA', (2, 1))
    image.putpixel((0, 0), (255, 0, 0, 255))
    image.putpixel((1, 0), (0, 255, 0, 128))
    result = inspect_pixel(image, *point)
    assert (result.x, result.y) == point
    assert result.pixel == index
    assert result.rgba == rgba
