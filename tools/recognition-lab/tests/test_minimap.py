import pytest

from recognition_lab.minimap import thumbnail_transform, visible_region, centered_transform, minimap_size
from recognition_lab.viewport import ViewportTransform


def test_thumbnail_letterbox_and_round_trip():
    tr = thumbnail_transform((400, 200), (180, 140))
    assert (tr.scale, tr.offset_x, tr.offset_y) == pytest.approx((.45, 0, 25))
    point = (123.125, 45.75)
    assert tr.to_image(*tr.to_viewport(*point)) == pytest.approx(point)


def test_visible_region_clipping_and_complete_visibility():
    assert visible_region(ViewportTransform(2, 10, 10), (20, 20), (60, 60)) == ((0, 0, 20, 20), True)
    region, entire = visible_region(ViewportTransform(2, -5, -8), (20, 20), (30, 24))
    assert region == pytest.approx((2.5, 4, 17.5, 16))
    assert not entire
    assert visible_region(ViewportTransform(2, 100, 100), (20, 20), (30, 24)) == (None, False)


def test_centering_preserves_scale_and_drag_offset():
    transform = ViewportTransform(3.5, -12, 9)
    viewport = (500, 300)
    original_center = transform.to_image(250, 150)
    pointer = (10.25, 20.5)
    moved = (13.75, 23.25)
    center = (original_center[0] + moved[0] - pointer[0], original_center[1] + moved[1] - pointer[1])
    updated = centered_transform(transform, viewport, center)
    assert updated.scale == transform.scale
    assert updated.to_image(250, 150) == pytest.approx(center)
    assert centered_transform(transform, viewport, pointer).to_image(250, 150) == pytest.approx(pointer)


@pytest.mark.parametrize(('image', 'expected'), [
    ((400, 200), (180, 92)), ((200, 400), (72, 140)),
    ((10000, 1), (180, 28)), ((1, 10000), (28, 140)),
    ((1, 1), (140, 140)),
])
def test_adaptive_minimap_size(image, expected):
    size = minimap_size(image)
    assert size == expected
    transform = thumbnail_transform(image, (size[0]-4, size[1]-4))
    point = (image[0] / 3, image[1] / 3)
    assert transform.to_image(*transform.to_viewport(*point)) == pytest.approx(point)
