from dataclasses import replace
import math

import numpy as np
import pytest

from grid_geometry.geometry import Geometry
from grid_geometry.heatmaps import geometry_heatmaps


@pytest.fixture
def gapped_geometry(annotation):
    fields = ("size", "rows", "columns", "grid_box", "x_edges", "y_edges", "cell_boxes", "centers")
    return Geometry(**{key: annotation[key] for key in fields})


def test_channel_order_gaussians_and_half_integer_mask(gapped_geometry):
    maps = geometry_heatmaps(gapped_geometry, sigma=0.5)

    assert maps.shape == (3, 19, 31)
    assert maps.dtype == np.float32
    assert np.isfinite(maps).all()
    assert maps.min() >= 0
    assert maps.max() <= 1
    assert maps[0, 4, 5] == pytest.approx(math.exp(-0.5 * (0.25 / 0.5) ** 2))
    assert maps[0, 11, 13] > maps[1, 11, 13]
    assert maps[1, 4, 9] == pytest.approx(math.exp(-0.5 * (0.25 / 0.5) ** 2))
    assert maps[1, 4, 9] > maps[1, 4, 10]
    assert maps[2, 7, 5] == pytest.approx(math.exp(-0.5))
    assert maps[1, 1, 9] > 0
    assert np.count_nonzero(maps[1, [0, 14, 15]]) == 0
    assert np.count_nonzero(maps[2, :, [0, 1, 25, 26]]) == 0


def test_max_combines_centers_and_optionally_excludes_outer_edges(gapped_geometry):
    full = geometry_heatmaps(gapped_geometry, sigma=6)
    internal = geometry_heatmaps(gapped_geometry, sigma=6, include_outer=False)

    np.testing.assert_array_equal(full[0], internal[0])
    assert full[0, 4, 9] == pytest.approx(math.exp(-0.5 * (3.75 / 6) ** 2))
    assert full[1, 4, 2] > internal[1, 4, 2]
    assert full[2, 1, 5] > internal[2, 1, 5]
    assert full[1, 4, 9] == internal[1, 4, 9]
    assert full[2, 7, 5] == internal[2, 7, 5]


def test_single_cell_has_empty_internal_boundary_channels():
    geometry = Geometry(
        size=(4, 4),
        rows=1,
        columns=1,
        grid_box=[0.5, 0.5, 2.5, 2.5],
        x_edges=[0.5, 2.5],
        y_edges=[0.5, 2.5],
        cell_boxes=[[[0.5, 0.5, 2.5, 2.5]]],
        centers=[[[1.5, 1.5]]],
    )
    internal = geometry_heatmaps(geometry, include_outer=False)

    assert internal[0, 1, 1] == 1
    assert not internal[1:].any()
    assert geometry_heatmaps(geometry)[1, 0, 0] == 1


@pytest.mark.parametrize("sigma", [0, -1, True, "2", float("inf"), float("nan")])
def test_rejects_invalid_sigma(gapped_geometry, sigma):
    with pytest.raises(ValueError, match="sigma"):
        geometry_heatmaps(gapped_geometry, sigma=sigma)


@pytest.mark.parametrize("sigma", [5e-324, 1e308])
def test_targets_remain_finite_for_extreme_finite_sigma(gapped_geometry, sigma):
    assert np.isfinite(geometry_heatmaps(gapped_geometry, sigma=sigma)).all()


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"rows": 0}, "rows and columns"),
        ({"x_edges": [2.25, 17.75, 9.75, 25.25]}, "strictly increasing"),
        ({"x_edges": [2.0, 9.75, 17.75, 25.25]}, "coincide"),
        ({"x_edges": [True, 9.75, 17.75, 25.25]}, "finite numbers"),
        ({"y_edges": [1.5, float("nan"), 14.5]}, "finite"),
        ({"grid_box": [2.25, -1, 25.25, 14.5]}, "inside the image"),
        ({"centers": [[[1, 2]]]}, "shape"),
    ],
)
def test_rejects_invalid_geometry(gapped_geometry, change, error):
    with pytest.raises(ValueError, match=error):
        replace(gapped_geometry, **change)
