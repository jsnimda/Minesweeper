import math

import pytest

from recognition_lab.viewport import ViewportTransform


def test_identity_and_fractional_round_trip():
    identity = ViewportTransform()
    assert identity.to_viewport(0.5, 1.25) == (0.5, 1.25)
    transform = ViewportTransform(1.75, -23.5, 8.125)
    point = (128.35, -19.82)
    mapped = transform.to_viewport(*point)
    assert mapped == pytest.approx((201.1125, -26.56))
    assert transform.to_image(*mapped) == pytest.approx(point)
    assert transform.to_viewport(*transform.to_image(-7.5, 42.25)) == pytest.approx((-7.5, 42.25))


def test_pan_and_anchored_scaling():
    transform = ViewportTransform(2.5, -15.25, 6.75)
    panned = transform.panned(3.125, -8.5)
    assert (panned.scale, panned.offset_x, panned.offset_y) == pytest.approx((2.5, -12.125, -1.75))
    anchor = (50.25, 100.75)
    source = panned.to_image(*anchor)
    for scale in (0.3, 8.25, 1.0, 2.5):
        panned = panned.scaled_at(scale, *anchor)
        assert panned.to_image(*anchor) == pytest.approx(source)
        assert panned.to_viewport(*source) == pytest.approx(anchor)


@pytest.mark.parametrize(('image', 'viewport', 'expected'), [
    ((100, 100), (200, 200), (2, 0, 0)),
    ((200, 100), (300, 300), (1.5, 0, 75)),
    ((100, 200), (300, 200), (1, 100, 0)),
    ((300, 200), (125, 70), (0.35, 10, 0)),
])
def test_fit(image, viewport, expected):
    transform = ViewportTransform.fit(*image, *viewport)
    assert (transform.scale, transform.offset_x, transform.offset_y) == pytest.approx(expected)


@pytest.mark.parametrize('values', [(0, 0, 0), (-1, 0, 0), (math.inf, 0, 0),
                                   (1, math.nan, 0), (1, 0, math.inf)])
def test_invalid_transform(values):
    with pytest.raises(ValueError):
        ViewportTransform(*values)


@pytest.mark.parametrize('dimensions', [(0, 10, 20, 20), (10, -1, 20, 20),
                                        (10, 10, 0, 20), (10, 10, 20, math.inf)])
def test_invalid_fit(dimensions):
    with pytest.raises(ValueError):
        ViewportTransform.fit(*dimensions)


def test_invalid_operations():
    transform = ViewportTransform()
    with pytest.raises(ValueError):
        transform.scaled_at(0, 5, 5)
    with pytest.raises(ValueError):
        transform.panned(math.inf, 0)


from recognition_lab.viewport import (MAX_MAGNIFICATION, MIN_MAGNIFICATION,
                                      clamp_magnification, logical_scale,
                                      physical_scale, reanchor_viewport,
                                      wheel_factor, zoom_by_factor)


@pytest.mark.parametrize(('scale', 'dpr', 'physical'), [
    (1, 1, 1), (0.5, 2, 1), (1, 2, 2), (2, 1.5, 3), (0.0125, 1.5, 0.01875),
])
def test_physical_magnification_conversions(scale, dpr, physical):
    assert physical_scale(scale, dpr) == pytest.approx(physical)
    assert logical_scale(physical, dpr) == pytest.approx(scale)
    assert physical_scale(logical_scale(1, dpr), dpr) == pytest.approx(1)


@pytest.mark.parametrize('value', [0, -1, math.inf, math.nan])
def test_invalid_magnification_conversions(value):
    for call in (lambda: physical_scale(value, 1), lambda: physical_scale(1, value),
                 lambda: logical_scale(value, 1), lambda: logical_scale(1, value)):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize('dpr', [1, 1.5, 2])
def test_clamped_zoom_preserves_anchor_and_limit_has_no_drift(dpr):
    transform = ViewportTransform(logical_scale(1, dpr), -15.25, 20.5)
    anchor = (85.5, 37.25)
    source = transform.to_image(*anchor)
    maximum = zoom_by_factor(transform, dpr, 1e300, anchor)
    assert physical_scale(maximum.scale, dpr) == pytest.approx(MAX_MAGNIFICATION)
    assert maximum.to_image(*anchor) == pytest.approx(source)
    assert zoom_by_factor(maximum, dpr, 2, anchor) == maximum
    minimum = zoom_by_factor(transform, dpr, 1e-300, anchor)
    assert physical_scale(minimum.scale, dpr) == pytest.approx(MIN_MAGNIFICATION)
    assert minimum.to_image(*anchor) == pytest.approx(source)
    assert zoom_by_factor(minimum, dpr, 0.5, anchor) == minimum


def test_exceptional_fit_zoom_moves_only_toward_range():
    assert clamp_magnification(0.001, 0.0005) == 0.001
    assert clamp_magnification(0.001, 0.0012) == 0.0012
    assert clamp_magnification(0.001, 100) == MAX_MAGNIFICATION
    assert clamp_magnification(100, 120) == 100
    assert clamp_magnification(100, 80) == 80
    assert clamp_magnification(100, 0.001) == MIN_MAGNIFICATION
    transform = ViewportTransform(0.001, 3, 4)
    assert zoom_by_factor(transform, 1, 0.5, (10, 20)) == transform
    assert zoom_by_factor(transform, 1, 1.2, (10, 20)).scale == pytest.approx(0.0012)


@pytest.mark.parametrize('factor', [0, -1, math.inf, math.nan])
def test_invalid_zoom_factor(factor):
    with pytest.raises(ValueError):
        zoom_by_factor(ViewportTransform(), 1, factor, (20, 30))


@pytest.mark.parametrize(('pixel', 'angle', 'expected'), [
    (0, 0, 1), (0, 120, 1.2), (0, -120, 1 / 1.2),
    (0, 60, math.sqrt(1.2)), (0, 240, 1.2 ** 2),
    (120, -120, 1.2), (-60, 0, 1 / math.sqrt(1.2)),
])
def test_wheel_factor(pixel, angle, expected):
    assert wheel_factor(pixel, angle) == pytest.approx(expected)
    assert wheel_factor(-pixel, -angle) * wheel_factor(pixel, angle) == pytest.approx(1)


def test_extreme_and_invalid_wheel_deltas():
    assert math.isfinite(wheel_factor(1e308, 0))
    assert wheel_factor(-1e308, 0) > 0
    with pytest.raises(ValueError):
        wheel_factor(math.nan, 0)
    with pytest.raises(ValueError):
        wheel_factor(0, math.inf)


def test_resize_and_dpr_change_preserve_physical_scale_and_center():
    transform = ViewportTransform(1.25, -31.75, 12.5)
    source = transform.to_image(400, 300)
    resized = reanchor_viewport(transform, (800, 600), (1000, 400), 2, 1.5)
    assert resized.to_image(500, 200) == pytest.approx(source)
    assert physical_scale(resized.scale, 1.5) == pytest.approx(physical_scale(transform.scale, 2))
    # Separate DPR and resize notifications produce the same result in either order.
    first = reanchor_viewport(transform, (800, 600), (800, 600), 2, 1.5)
    first = reanchor_viewport(first, (800, 600), (1000, 400), 1.5, 1.5)
    second = reanchor_viewport(transform, (800, 600), (1000, 400), 2, 2)
    second = reanchor_viewport(second, (1000, 400), (1000, 400), 2, 1.5)
    assert (first.scale, first.offset_x, first.offset_y) == pytest.approx((resized.scale, resized.offset_x, resized.offset_y))
    assert (second.scale, second.offset_x, second.offset_y) == pytest.approx((resized.scale, resized.offset_x, resized.offset_y))


def test_viewport_center_zoom_and_refit():
    transform = ViewportTransform.fit(300, 100, 600, 400)
    center = (300, 200)
    source = transform.to_image(*center)
    zoomed = zoom_by_factor(transform, 2, 1.2, center)
    assert zoomed.to_image(*center) == pytest.approx(source)
    fitted = ViewportTransform.fit(300, 100, 400, 600)
    assert fitted.to_viewport(150, 50) == pytest.approx((200, 300))
