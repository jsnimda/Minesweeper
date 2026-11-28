import numpy as np
import pytest
import skia
from PIL import Image

from recognition_lab.inspection import inspect_pixel
from recognition_lab.viewer import image_sampling


def test_sampling_modes_render_without_changing_source():
    source = Image.new('RGBA', (2, 1))
    source.putpixel((0, 0), (255, 0, 0, 255))
    source.putpixel((1, 0), (0, 0, 255, 255))
    image = skia.Image.fromarray(np.asarray(source).copy())
    samples = []
    for mode in ('nearest', 'smooth'):
        surface = skia.Surface(16, 8)
        canvas = surface.getCanvas()
        canvas.scale(8, 8)
        canvas.drawImage(image, 0, 0, image_sampling(mode))
        samples.append(surface.makeImageSnapshot().toarray(colorType=skia.ColorType.kRGBA_8888_ColorType))
    nearest, smooth = samples
    assert tuple(nearest[4, 7, :3]) == (255, 0, 0)
    assert 0 < smooth[4, 7, 0] < 255
    assert 0 < smooth[4, 7, 2] < 255
    assert inspect_pixel(source, .5, .5).rgba == (255, 0, 0, 255)
    assert inspect_pixel(source, 1.5, .5).rgba == (0, 0, 255, 255)
    with pytest.raises(ValueError):
        image_sampling('unknown')
