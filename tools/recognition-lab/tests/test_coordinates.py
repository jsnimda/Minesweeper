import numpy as np
import pytest
from recognition_lab.results import Crop, DesktopMapping, crop_box


def test_outward_rounding_and_clip():
    crop = crop_box((1.2, 2.8, 90.1, 99.9), (100, 100), 3)
    assert crop.box == (0, 0, 94, 100)
    assert crop.clipped
    assert crop_box((5.2, 6.2, 7.1, 8.1), (100, 100), 0).box == (5, 6, 8, 9)
    with pytest.raises(ValueError): crop_box((110, 1, 120, 2), (100, 100), 0)


def test_crop_and_retina_coordinates():
    assert np.allclose(Crop((5, 8, 20, 40), False).to_image([[1.5, 2.5]]), [[6.5, 10.5]])
    mapping = DesktopMapping(-1200, 100, 1200, 800, 2400, 1600)
    assert mapping.to_desktop(1200, 800) == (-600, 500)
    mapping = DesktopMapping(100, -800, 900, 600, 1350, 1200)
    assert mapping.to_desktop(675, 600) == (550, -500)


def test_geometry_inverse_transforms():
    from PIL import Image
    from grid_geometry.preprocessing import preprocess_image, ImageTransform
    crop = Crop((31, 47, 231, 147), False)
    _, transform = preprocess_image(Image.new('RGB', (200, 100)), (384, 384))
    points = np.array([[0., 0.], [100.5, 50.5], [200., 100.]])
    assert np.allclose(crop.to_image(transform.to_source(transform.to_canvas(points))), points + [31, 47])
    native = ImageTransform((200, 100), (200, 100), (200, 100), (0, 0))
    assert np.allclose(crop.to_image(native.to_source(points)), points + [31, 47])


def test_capture_region_bounds():
    from recognition_lab.results import validate_region
    assert validate_region(None, 100, 80) == (0, 0, 100, 80)
    assert validate_region((10, 20, 90, 60), 100, 80) == (10, 20, 90, 60)
    for region in [(0, 0, 0, 1), (-1, 0, 1, 1), (99, 0, 2, 1), (0, 79, 1, 2)]:
        with pytest.raises(ValueError): validate_region(region, 100, 80)
