import numpy as np
from grid_geometry.preprocessing import ImageTransform
from recognition_lab.native import pad_native, unpad_native
from recognition_lab.results import Crop


def test_native_centered_padding_and_output_removal():
    rgb = np.arange(11 * 15 * 3, dtype=np.uint8).reshape(11, 15, 3)
    padded, padding = pad_native(rgb)
    assert padded.shape == (16, 16, 3)
    assert padding == (0, 2, 1, 3)
    assert np.array_equal(padded[2:13, :15], rgb)
    assert not padded[:2].any() and not padded[13:].any()
    maps = padded.transpose(2, 0, 1)
    assert np.array_equal(unpad_native(maps, padding), rgb.transpose(2, 0, 1))


def test_native_aligned_and_coordinates():
    rgb = np.ones((16, 24, 3), dtype=np.uint8)
    padded, padding = pad_native(rgb)
    assert padding == (0, 0, 0, 0)
    assert np.array_equal(padded, rgb)
    assert np.array_equal(unpad_native(padded.transpose(2, 0, 1), padding), rgb.transpose(2, 0, 1))
    transform = ImageTransform((24, 16), (24, 16), (24, 16), (0, 0))
    points = np.array([[0.5, 1.5], [23.5, 15.5]])
    assert np.allclose(Crop((31, 47, 55, 63), False).to_image(transform.to_source(points)), points + [31, 47])


def test_large_crop_minimal_padding():
    padded, padding = pad_native(np.zeros((1240, 2047, 3), dtype=np.uint8))
    assert padded.shape == (1240, 2048, 3)
    assert padding == (0, 0, 1, 0)
