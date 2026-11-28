"""Deterministic ground-truth lattice extents; no generated data required."""
import pytest

from grid_geometry.dataset.equivalent_boundaries import equivalent_boundaries
from grid_geometry.dataset.targets import PhaseGeometry


def boundaries(box, known=(0, 0, 100, 100), *, origin=(0, 0), pitch=(10, 10), dimensions=(20, 20), tolerance=.05):
    geometry = PhaseGeometry(*origin, *pitch, box)
    return equivalent_boundaries(geometry, dimensions, known, midpoint_tolerance=tolerance)


@pytest.mark.parametrize(('known', 'expected'), [
    ((0, 0, 100, 100), ((10,), (10,), (90,), (90,))),
    ((20, 20, 80, 80), ((20,), (20,), (80,), (80,))),
    ((0, 20, 80, 80), ((10,), (20,), (80,), (80,))),
    ((0, 0, 80, 80), ((10,), (10,), (80,), (80,))),
    ((0, 20, 100, 80), ((10,), (20,), (90,), (80,))),
    ((0, 0, 100, 80), ((10,), (10,), (90,), (80,))),
])
def test_all_side_arrangements(known, expected):
    result = boundaries((10, 10, 90, 90), known, origin=(10, 10), dimensions=(8, 8))
    assert (result.left, result.top, result.right, result.bottom) == expected


@pytest.mark.parametrize(('edge', 'expected'), [
    (368, (360,)), (384, (392,)), (376, (360, 392)),
    (375, (360, 392)), (373, (360,)), (379, (392,)),
    (360, (360,)), (374.4, (360, 392)), (377.6, (360, 392)),
    (374.4 - 1e-7, (360,)), (374.4 + 1e-7, (360, 392)),
    (377.6 - 1e-7, (360, 392)), (377.6 + 1e-7, (392,)),
])
def test_continuous_edges_and_midpoint_thresholds(edge, expected):
    geometry = PhaseGeometry(40, 40, 32, 32, (40, 40, 680, 680))
    result = equivalent_boundaries(geometry, (20, 20), (edge, edge, edge + 100, edge + 100))
    assert result.left == expected
    assert result.top == expected
    result = equivalent_boundaries(geometry, (20, 20), (edge - 40, edge - 40, edge, edge))
    assert result.right == expected
    assert result.bottom == expected


def test_physical_edges_near_known_edges_are_single_ideal_lines():
    geometry = PhaseGeometry(4.8, 4.8, 10, 10, (5, 5, 24.6, 24.6))
    result = equivalent_boundaries(geometry, (2, 2), (0, 0, 25, 25))
    assert (result.left, result.top, result.right, result.bottom) == ((4.8,), (4.8,), (24.8,), (24.8,))
    equal = equivalent_boundaries(geometry, (2, 2), (5, 5, 24.6, 24.6))
    assert equal == result


def test_fractional_anisotropic_geometry_and_non_full_known_region():
    geometry = PhaseGeometry(-20.25, -30.5, 10.5, 12.5, (-20, -30, 84.5, 94.25))
    result = equivalent_boundaries(geometry, (10, 10), (2, 3, 72, 76))
    assert result.left == pytest.approx((.75,))
    assert result.top == pytest.approx((7.,))
    assert result.right == pytest.approx((74.25,))
    # 76 is 0.52 pitch from 69.5, so both adjacent lines are acceptable.
    assert result.bottom == pytest.approx((69.5, 82.))


def test_negative_and_outside_image_coordinates():
    assert equivalent_boundaries(None, None, (0, 0, 10, 10)) is None
    geometry = PhaseGeometry(-2, -2, 10, 10, (-2, -2, 108, 108))
    result = equivalent_boundaries(geometry, (11, 11), (0, 0, 100, 100))
    assert (result.left, result.top, result.right, result.bottom) == ((-2,), (-2,), (98,), (98,))
    geometry = PhaseGeometry(2, 2, 10, 10, (-1, -1, 111, 111))
    result = equivalent_boundaries(geometry, (11, 11), (0, 0, 100, 100))
    assert (result.left, result.top, result.right, result.bottom) == ((2,), (2,), (102,), (102,))


def test_narrow_region_excludes_invalid_combinations():
    geometry = PhaseGeometry(0, 0, 10, 10, (-1, -1, 101, 101))
    result = equivalent_boundaries(geometry, (10, 10), (4.5, 4.5, 5.5, 5.5))
    assert result.left == result.right == result.top == result.bottom == (0, 10)
    assert result.valid_boxes() == ((0, 0, 10, 10),)


def test_tolerance_is_configurable_and_validated():
    geometry = PhaseGeometry(0, 0, 10, 10, (-1, -1, 101, 101))
    assert equivalent_boundaries(geometry, (10, 10), (4, 4, 84, 84), midpoint_tolerance=.1).left == (0, 10)
    with pytest.raises(ValueError):
        equivalent_boundaries(geometry, (10, 10), (0, 0, 10, 10), midpoint_tolerance=.5)
