import pytest

from recognition_lab.viewport import ViewportTransform, visible_pixel_grid


def test_fractional_visible_region_and_boundaries():
    transform = ViewportTransform(10, -24.5, -15.25)
    bounds, xs, ys = visible_pixel_grid(transform, (100, 100), (30, 25), 1)
    assert bounds == pytest.approx((2.45, 1.525, 5.45, 4.025))
    assert xs == (3, 5)
    assert ys == (2, 4)


def test_clips_to_source_and_limits_iteration():
    bounds, xs, ys = visible_pixel_grid(ViewportTransform(8, 4, 12), (3, 2), (100, 100), 1)
    assert bounds == pytest.approx((0, 0, 3, 2))
    assert xs == (0, 3) and ys == (0, 2)
    _, xs, ys = visible_pixel_grid(ViewportTransform(8, -800000, -800000), (1000000, 1000000), (80, 64), 1)
    assert xs[1] - xs[0] <= 10 and ys[1] - ys[0] <= 8


def test_threshold_dpr_and_empty_intersection():
    assert visible_pixel_grid(ViewportTransform(7.99), (10, 10), (10, 10), 1) is None
    assert visible_pixel_grid(ViewportTransform(4), (10, 10), (10, 10), 2) is not None
    assert visible_pixel_grid(ViewportTransform(8, 100, 100), (10, 10), (10, 10), 1) is None
    assert visible_pixel_grid(ViewportTransform(8), (10, 10), (0, 10), 1) is None
