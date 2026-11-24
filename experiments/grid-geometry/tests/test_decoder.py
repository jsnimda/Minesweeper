from dataclasses import replace

import numpy as np
import pytest

from grid_geometry.decoder import IndependentFits, PitchRatioPrior, decode_geometry, fit_independent


def _targets(width, height, x0, y0, px, py, columns, rows, *, gap=(0, 0), alternating=False):
    x = np.arange(width) + 0.5
    y = np.arange(height) + 0.5
    x_edges = x0 + np.arange(columns + 1) * px
    y_edges = y0 + np.arange(rows + 1) * py
    x_centers, y_centers = x_edges[:-1] + px / 2, y_edges[:-1] + py / 2
    vx, hy = x_edges.copy(), y_edges.copy()
    vx[[0, -1]] += (gap[0] / 2, -gap[0] / 2)
    hy[[0, -1]] += (gap[1] / 2, -gap[1] / 2)

    def distance(samples, peaks):
        return ((samples[:, None] - peaks)**2).min(axis=1)

    x_distance, y_distance = distance(x, x_centers), distance(y, y_centers)
    centers = np.exp(-(y_distance[:, None] + x_distance) / 8)
    if alternating:
        centers.fill(0)
        available_x = vx[::2]
    else:
        available_x = vx
    vertical = np.exp(-distance(x, available_x) / 8) * ((y >= hy[0]) & (y < hy[-1]))[:, None]
    horizontal = np.exp(-distance(y, hy) / 8)[:, None] * ((x >= vx[0]) & (x < vx[-1]))
    return np.stack((centers, vertical, horizontal)), x_edges, y_edges


def test_rectangular_fractional_lattice_and_spatial_diagnostic_shapes():
    maps, xe, ye = _targets(183, 167, 17.4, 21.6, 16.7, 22.1, 8, 5)
    decoded = decode_geometry(maps)
    best = decoded.hypotheses[0]
    assert decoded.status == "fitted"
    assert (best.rows, best.columns) == (5, 8)
    assert best.x.pitch == pytest.approx(16.7, abs=0.1)
    assert best.y.pitch == pytest.approx(22.1, abs=0.1)
    np.testing.assert_allclose(best.x_edges, xe, atol=0.15)
    np.testing.assert_allclose(best.y_edges, ye, atol=0.15)
    assert best.centers.shape == (5, 8, 2)
    assert best.diagnostics.cell_support.shape == (5, 8)
    assert best.diagnostics.vertical_support.shape == (5, 7)
    assert best.diagnostics.horizontal_support.shape == (4, 8)
    assert best.diagnostics.prior_agreement is None
    assert 1 <= decoded.evaluated <= 160


def test_gap_offsets_stay_separate_from_ideal_lattice_and_rim_is_censored():
    maps, xe, ye = _targets(171, 151, -1.05, 19.4, 17.3, 18.2, 9, 6, gap=(2.4, 1.2))
    best = decode_geometry(maps).hypotheses[0]
    np.testing.assert_allclose(best.x_edges, xe, atol=0.18)
    np.testing.assert_allclose(best.y_edges, ye, atol=0.18)
    assert best.outer_edges[0] is not None and best.outer_edges[0].censored
    assert best.visible_extent[0] is None
    np.testing.assert_allclose(best.visible_extent[1:], (ye[0] + 0.6, xe[-1] - 1.2, ye[-1] - 0.6), atol=0.2)


def test_perpendicular_lines_allow_terminal_extent_recovery_beyond_shortlist():
    maps, xe, _ = _targets(205, 159, 17.4, 19.3, 18.15, 17.2, 9, 6)
    x = np.arange(maps.shape[2]) + 0.5
    # Erase the last column's centers and outer vertical edge. The horizontal
    # ridges still extend to the true right edge.
    maps[0, :, x > xe[-2]] = 0
    maps[1, :, x > xe[-1] - 5] = 0
    fits = fit_independent(maps)
    x_best = replace(fits.x.hypotheses[0], cell_count=8)
    y_best = fits.y.hypotheses[0]
    restricted = IndependentFits(replace(fits.x, hypotheses=(x_best,)), replace(fits.y, hypotheses=(y_best,)))

    fixed = decode_geometry(maps, independent=restricted, refine_extent=False).hypotheses[0]
    recovered = decode_geometry(maps, independent=restricted).hypotheses[0]

    assert fixed.columns == 8
    assert recovered.columns == 9
    assert recovered.x.edited
    assert recovered.diagnostics.cell_support[:, -1].mean() < 0.05
    assert recovered.diagnostics.horizontal_support[:, -1].mean() > 0.75
    assert recovered.visible_extent[2] is None


def test_isolated_spurious_center_does_not_extend_the_board():
    maps, xe, ye = _targets(230, 160, 20.75, 14.2, 18.75, 16.75, 8, 7)
    x = np.arange(maps.shape[2]) + 0.5
    y = np.arange(maps.shape[1]) + 0.5
    fake = np.exp(-((x - xe[-1] - 9.375)**2 + (y[:, None] - (ye[2] + 8.375))**2) / 4)
    maps[0] = np.maximum(maps[0], fake)
    decoded = decode_geometry(maps)

    assert (decoded.hypotheses[0].rows, decoded.hypotheses[0].columns) == (7, 8)
    assert not any(candidate.columns == 9 and candidate.x.edited for candidate in decoded.hypotheses)
    assert decoded.hypotheses[0].diagnostics.center_energy_explained < 1.0


def test_boundary_only_alias_is_ambiguous_with_optional_ratio():
    maps, _, _ = _targets(253, 188, 18.1, 21.7, 15.6, 15.6, 12, 8, alternating=True)
    independent = fit_independent(maps)
    decoded = decode_geometry(maps, independent=independent)
    prior = decode_geometry(maps, independent=independent, ratio_prior=PitchRatioPrior(1, weight=0.28))

    assert decoded.status == "ambiguous"
    assert decoded.hypotheses[0].columns == 6
    original_rank = next(i for i, h in enumerate(decoded.hypotheses) if (h.rows, h.columns) == (8, 12))
    prior_rank = next(i for i, h in enumerate(prior.hypotheses) if (h.rows, h.columns) == (8, 12))
    assert prior_rank < original_rank
    assert prior.hypotheses[prior_rank].diagnostics.prior_agreement > 0.98
    assert prior.status == "ambiguous"


def test_insufficient_and_invalid_heatmaps():
    maps = np.zeros((3, 40, 60))
    assert decode_geometry(maps).status == "insufficient_evidence"
    assert decode_geometry(maps).hypotheses == ()
    with pytest.raises(ValueError, match="shape"):
        decode_geometry(np.zeros((2, 40, 60)))
    with pytest.raises(ValueError, match="probabilities"):
        decode_geometry(np.full((3, 40, 60), np.nan))
    with pytest.raises(ValueError, match="enable center or boundary"):
        decode_geometry(maps, use_centers=False, use_boundaries=False)
    with pytest.raises(ValueError, match="positive"):
        PitchRatioPrior(-1)


@pytest.mark.parametrize("resolution", [256, 384, 512])
def test_clean_exports_at_all_resolutions(diagnostic_root, resolution):
    from grid_geometry.dataset import GeometryDataset

    dataset = GeometryDataset(diagnostic_root, "train", canvas_size=(resolution, resolution))
    for index in range(len(dataset)):
        item = dataset.load_sample(index)
        best = decode_geometry(item.heatmaps.numpy()).hypotheses[0]
        assert (best.rows, best.columns) == (item.geometry.rows, item.geometry.columns)
        np.testing.assert_allclose(best.centers, item.geometry.centers, atol=0.25)
