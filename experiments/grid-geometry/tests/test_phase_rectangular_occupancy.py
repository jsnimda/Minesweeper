"""Deterministic mathematical fixtures; no model or dataset is required."""
import itertools
import math

import numpy as np
import pytest

from grid_geometry.phase.rectangular_occupancy import (
    integrate_cells, maximum_sum_rectangle, measure_rectangle,
)


def cells(values, mask=None, **kwargs):
    values = np.asarray(values, dtype=float)
    if mask is None:
        mask = np.ones(values.shape, dtype=bool)
    options = dict(pitch_x=1., pitch_y=1., origin_x=0., origin_y=0.,
                   domain=(0., 0., float(values.shape[1]), float(values.shape[0])))
    options.update(kwargs)
    return integrate_cells(values, mask, **options)


def bounds(rect):
    return rect.left, rect.top, rect.right, rect.bottom


def solve(values, mask=None, **kwargs):
    c = cells(values, mask, **kwargs)
    return c, measure_rectangle(c, maximum_sum_rectangle(c.signed_evidence))


def exhaustive(a):
    h, w = a.shape
    candidates = []
    for t in range(h):
        for l in range(w):
            for b in range(t+1, h+1):
                for r in range(l+1, w+1):
                    candidates.append((float(a[t:b, l:r].sum()), (t, l, b, r)))
    best = max(score for score, _ in candidates)
    matches = [box for score, box in candidates if score == best]
    return best, min(matches), len(matches)


def test_exhaustive_random_and_transposition():
    rng = np.random.default_rng(641)
    for h, w in itertools.product(range(1, 5), repeat=2):
        for _ in range(20):
            a = rng.integers(-2, 3, (h, w)).astype(float)
            result = maximum_sum_rectangle(a)
            expected, box, count = exhaustive(a)
            assert (result.score, (result.top, result.left, result.bottom, result.right), result.tie_count) == (expected, box, count)
            transposed = maximum_sum_rectangle(a.T)
            assert transposed.score == result.score and transposed.tie_count == count


def test_empty_negative_zero_positive_and_ties():
    assert maximum_sum_rectangle(np.empty((0, 3))) is None
    r = maximum_sum_rectangle([[-2, -1], [-1, -3]])
    assert bounds(r) == (1, 0, 2, 1) and r.score == -1 and r.tie_count == 2
    assert not r.has_positive_evidence
    r = maximum_sum_rectangle(np.zeros((2, 3)))
    assert bounds(r) == (0, 0, 1, 1) and r.tie_count == 18 and not r.has_positive_evidence
    r = maximum_sum_rectangle([[-1, -1, -1], [-1, 2, -1]])
    assert bounds(r) == (1, 1, 2, 2) and r.has_positive_evidence
    with pytest.raises(ValueError): maximum_sum_rectangle([1, 2])
    with pytest.raises(ValueError): maximum_sum_rectangle([[float('nan')]])


def test_perfect_rectangle_false_positive_hole_and_extra_rows_columns():
    image = np.zeros((8, 10))
    image[2:6, 2:7] = 1
    c, m = solve(image)
    assert bounds(m.rectangle) == (2, 2, 7, 6)
    assert (m.rows, m.columns, m.rectangle.score, m.interior_coverage) == (4, 5, 20, 1)
    assert m.mean_interior_occupancy == m.explained_occupancy_fraction == 1
    assert m.unexplained_exterior_occupancy == 0
    image[1, 8] = 1  # localized exterior prediction does not force an extension
    image[3, 3] = 0  # an interior hole does not necessarily split the optimum
    _, m = solve(image)
    assert bounds(m.rectangle) == (2, 2, 7, 6)
    assert m.unexplained_exterior_occupancy == 1
    image[2:7, 7] = 1
    image[6, 2:7] = 1
    _, m = solve(image)
    assert bounds(m.rectangle) == (2, 2, 8, 7)
    # The selected boundary beats truncating a genuinely occupied column.
    alternative = m.alternatives['right']['inward']
    assert alternative.score_difference == -5
    assert alternative.changed_observed_area == 5
    assert alternative.normalized_margin == 1


def test_unknown_censoring_and_boundary_alternatives():
    image = np.ones((2, 4))
    mask = np.array([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=bool)
    image[~mask] = np.nan
    c, m = solve(image, mask)
    assert np.all(c.signed_evidence[:, [0, 3]] == 0)
    assert np.isnan(c.mean_occupancy[:, [0, 3]]).all()
    assert m.rectangle.tie_count == 4
    assert m.alternatives['left']['outward'].status == 'outside_domain'
    assert m.alternatives['right']['outward'].status == 'unobserved'
    assert m.alternatives['right']['outward'].normalized_margin is None
    missing, m = solve(np.full((2, 2), np.nan), np.zeros((2, 2), bool))
    assert np.all(missing.coverage == 0) and m.rectangle.score == 0
    assert m.mean_interior_occupancy is None and m.explained_occupancy_fraction is None


def test_disconnected_regions_are_only_a_mathematical_candidate():
    a = np.zeros((3, 9)); a[1, 1:3] = 1; a[1, 6:8] = 1
    _, m = solve(a)
    assert m.rectangle.tie_count == 2
    assert m.unexplained_exterior_occupancy == 2


def test_partial_domain_preserves_lattice_indices_and_coordinates():
    c, m = solve(np.ones((3, 4)), pitch_x=2., pitch_y=2., origin_x=1., origin_y=1.,
                 domain=(0., 0., 4., 3.))
    assert (c.x_start, c.y_start) == (-1, -1)
    assert bounds(m.rectangle) == (-1, -1, 2, 1)
    assert m.coordinates == (-1., -1., 5., 3.)
    assert m.interior_coverage == pytest.approx(.5)
    assert m.alternatives['right']['outward'].status == 'outside_domain'


def test_fractional_integration_against_pixel_intersections():
    rng = np.random.default_rng(724)
    for _ in range(20):
        values = rng.random((5, 7))
        mask = rng.random((5, 7)) > .3
        c = cells(values, mask, pitch_x=1.37, pitch_y=1.83, origin_x=.31, origin_y=-.27,
                  domain=(.2, .4, 6.4, 4.8))
        for j, i in itertools.product(range(len(c.y_edges)-1), range(len(c.x_edges)-1)):
            x0, x1 = max(c.x_edges[i], .2), min(c.x_edges[i+1], 6.4)
            y0, y1 = max(c.y_edges[j], .4), min(c.y_edges[j+1], 4.8)
            area = occupancy = 0.
            for y, x in itertools.product(range(5), range(7)):
                intersection = max(0., min(x1, x+1)-max(x0, x)) * max(0., min(y1, y+1)-max(y0, y))
                if mask[y, x]:
                    area += intersection
                    occupancy += intersection * values[y, x]
            assert c.observed_area[j, i] == pytest.approx(area, abs=1e-12)
            assert c.integrated_occupancy[j, i] == pytest.approx(occupancy, abs=1e-12)
            assert c.signed_evidence[j, i] == pytest.approx((2*occupancy-area)/(1.37*1.83), abs=1e-12)
            assert c.coverage[j, i] == pytest.approx(area/(1.37*1.83), abs=1e-12)
            if area:
                assert c.mean_occupancy[j, i] == pytest.approx(occupancy/area, abs=1e-12)
            else:
                assert math.isnan(c.mean_occupancy[j, i])
    assert np.any((c.coverage > 0) & (c.coverage < 1))


def test_invalid_inputs_and_retained_diagnostics():
    with pytest.raises(ValueError): cells([[np.nan]])
    with pytest.raises(ValueError): cells([[1.1]])
    with pytest.raises(ValueError): cells([[0]], np.ones((1, 1)))
    with pytest.raises(ValueError): cells([[0]], pitch_x=0)
    with pytest.raises(ValueError): cells([[0]], origin_y=float('inf'))
    with pytest.raises(ValueError): cells([[0]], domain=(-1, 0, 1, 1))
    diagnostics = {'observation_count': np.ones((1, 1))}
    assert cells([[1]], diagnostics=diagnostics).diagnostics is diagnostics


def test_fractional_integrals_remain_in_probability_bounds():
    values = np.ones((73, 89), dtype=float)
    mask = np.ones_like(values, dtype=bool)
    mask[20:40, 30:60] = False
    c = cells(values, mask, pitch_x=3.17, pitch_y=2.83, origin_x=.19, origin_y=.41)
    assert np.all((c.coverage >= 0) & (c.coverage <= 1))
    known = c.observed_area > 0
    assert np.all((c.mean_occupancy[known] >= 0) & (c.mean_occupancy[known] <= 1))
    assert np.all((c.observed_area >= 0) & (c.observed_area <= c.full_area))
    assert np.all(c.integrated_occupancy >= 0)
    assert np.all(c.integrated_occupancy <= c.observed_area)
    assert np.all(c.signed_evidence >= -c.coverage)
    assert np.all(c.signed_evidence <= c.coverage)
    assert np.all(c.signed_evidence[~known] == 0)
    assert np.isnan(c.mean_occupancy[~known]).all()
