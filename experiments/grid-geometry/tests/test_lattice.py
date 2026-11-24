from dataclasses import replace

import numpy as np
import pytest

from grid_geometry.lattice import fit_axis, rank_axis_pairs
from grid_geometry.pitch import estimate_pitch
from grid_geometry.signals import AxisSignals, project_heatmaps


def _profile(length, positions, strengths=0.75, sigma=1.4):
    pixels = np.arange(length) + 0.5
    distances = pixels[:, None] - np.asarray(positions)
    return np.max(strengths * np.exp(-0.5 * (distances / sigma)**2), axis=1)


def _fixture(origin, pitch, cells, length, gap=0):
    boundaries = origin + np.arange(cells + 1) * pitch
    visible = boundaries.copy()
    visible[0] += gap / 2
    visible[-1] -= gap / 2
    centers = boundaries[:-1] + pitch / 2
    return AxisSignals(_profile(length, centers), _profile(length, visible)), centers, visible


@pytest.mark.parametrize(
    ("origin", "pitch", "count", "length"),
    [(-1.25, 13.71, 8, 125), (11.25, 7.68, 29, 250), (64.93, 21.87, 7, 294)],
)
def test_recovers_fractional_lattice_and_count_across_padding_and_clipping(origin, pitch, count, length):
    signals, centers, _ = _fixture(origin, pitch, count, length)
    result = fit_axis(signals)

    best = result.hypotheses[0]
    assert best.cell_count == count
    assert best.pitch == pytest.approx(pitch, abs=0.07)
    assert best.origin == pytest.approx(origin, abs=0.13)
    np.testing.assert_allclose(best.centers, centers, atol=0.14)
    assert best.diagnostics.observed_agreement > 0.95
    assert best.diagnostics.expected_support > 0.95


def test_visible_gap_edges_do_not_pull_the_internal_pitch_or_origin():
    origin, pitch, count = 17.3, 16.7, 12
    signals, centers, visible = _fixture(origin, pitch, count, 233, gap=3.2)
    best = fit_axis(signals).hypotheses[0]

    assert best.cell_count == count
    np.testing.assert_allclose(best.centers, centers, atol=0.06)
    np.testing.assert_allclose(best.ideal_extent, (origin, origin + count * pitch), atol=0.06)
    np.testing.assert_allclose(best.visible_extent, visible[[0, -1]], atol=0.11)
    assert len(best.diagnostics.internal_boundary_matches) == count - 1


def test_censored_canvas_endpoint_and_unobserved_other_side():
    origin, pitch, count = -0.6, 15.32, 9
    _, centers, visible = _fixture(origin, pitch, count, 150)
    signals = AxisSignals(_profile(150, centers), _profile(150, visible[:-1]))
    best = fit_axis(signals).hypotheses[0]

    assert best.cell_count == count
    assert best.origin == pytest.approx(origin, abs=0.13)
    assert best.left_edge is not None and best.left_edge.censored
    assert best.right_edge is None
    assert best.visible_extent == (None, None)


def test_near_rim_center_with_short_outer_flank_is_retained():
    pitch = 7.5
    origin = -2.52
    count = 28
    expected_boundaries = origin + np.arange(count + 1) * pitch
    centers = expected_boundaries[:-1] + pitch / 2
    signals = AxisSignals(
        _profile(233, centers, 0.41, sigma=1.7),
        _profile(233, expected_boundaries, 0.79, sigma=1.7),
    )
    best = fit_axis(signals).hypotheses[0]

    assert best.cell_count == count
    assert best.diagnostics.center_matches[0] == pytest.approx(centers[0], abs=0.12)
    assert best.visible_extent[0] is None


def test_erased_terminal_cell_leaves_an_unverified_extent():
    origin, pitch, count = 38.4, 21.75, 10
    _, centers, visible = _fixture(origin, pitch, count, 290, gap=1.9)
    signals = AxisSignals(_profile(290, centers[:-1]), _profile(290, visible[:-2]))
    result = fit_axis(signals)

    assert result.status == "ambiguous"
    assert result.hypotheses[0].cell_count == count - 1
    assert result.hypotheses[0].visible_extent[1] is None


def test_refines_coarse_seed_and_reports_missing_and_unexplained_peaks():
    origin, pitch, count, length = 23.3, 18.4, 11, 256
    _, centers, edges = _fixture(origin, pitch, count, length, gap=1.6)
    missing = {2, 3, 7}
    rng = np.random.default_rng(20221)
    center_peaks = np.array([x for index, x in enumerate(centers) if index not in missing])
    center_peaks += rng.normal(0, 0.12, len(center_peaks))
    signals = AxisSignals(
        _profile(length, np.r_[center_peaks, 8.9, 247.1], np.r_[np.full(8, 0.7), 0.61, 0.6]),
        _profile(length, np.r_[np.delete(edges, [3, 6]), 241.3]),
    )
    analysis = estimate_pitch(signals)
    coarse = replace(analysis.candidates[0], pitch=pitch - 1.1)
    best = fit_axis(signals, replace(analysis, candidates=(coarse,))).hypotheses[0]

    assert best.cell_count == count
    assert best.initial_pitch == coarse.pitch
    assert best.pitch == pytest.approx(pitch, abs=0.12)
    assert best.origin == pytest.approx(origin, abs=0.35)
    assert all(best.diagnostics.center_matches[i] is None for i in missing)
    assert any(abs(x - 247.1) < 0.25 for x in best.diagnostics.unexplained_centers)
    assert any(abs(x - 241.3) < 0.25 for x in best.diagnostics.unexplained_boundaries)


def test_boundary_only_alias_is_ambiguous_and_soft_prior_can_rerank():
    origin, pitch, count = 13.3, 14.55, 14
    _, _, edges = _fixture(origin, pitch, count, 247)
    sparse = AxisSignals(np.zeros(247), _profile(247, edges[::2]))
    horizontal = fit_axis(sparse)
    vertical = fit_axis(_fixture(31.3, pitch, 10, 225)[0])

    assert horizontal.status == "ambiguous"
    assert horizontal.hypotheses[0].pitch == pytest.approx(2 * pitch, abs=0.15)
    assert any(h.cell_count == count and abs(h.pitch - pitch) < 0.12 for h in horizontal.hypotheses)
    assert rank_axis_pairs(horizontal, vertical)[0].x == horizontal.hypotheses[0]
    prior = rank_axis_pairs(horizontal, vertical, square_prior_weight=0.24)[0]
    assert prior.x.pitch == pytest.approx(pitch, abs=0.12)
    assert prior.x.cell_count == count


def test_independent_fits_and_ratio_prior_support_rectangular_cells():
    x = fit_axis(_fixture(11.23, 13.7, 9, 180)[0])
    y = fit_axis(_fixture(9.32, 24.5, 7, 219)[0])
    independent = rank_axis_pairs(x, y)[0]
    rectangular = rank_axis_pairs(x, y, square_prior_weight=0.2, expected_pitch_ratio=13.7 / 24.5)[0]

    assert independent.x.pitch == pytest.approx(13.7, abs=0.05)
    assert independent.y.pitch == pytest.approx(24.5, abs=0.05)
    assert rectangular.prior_agreement > 0.99
    assert (rectangular.x.cell_count, rectangular.y.cell_count) == (9, 7)


def test_insufficient_and_invalid_evidence():
    assert fit_axis(AxisSignals(np.zeros(64), np.zeros(64))).status == "insufficient_evidence"
    with pytest.raises(ValueError, match="equal length"):
        fit_axis(AxisSignals(np.zeros(16), np.zeros(12)))
    with pytest.raises(ValueError, match="probabilities"):
        fit_axis(AxisSignals(np.full(12, np.nan), np.zeros(12)))
    with pytest.raises(ValueError, match="max_cells"):
        fit_axis(AxisSignals(np.zeros(12), np.zeros(12)), max_cells=1)


@pytest.mark.parametrize("resolution", [256, 384, 512])
def test_real_exports_recover_both_axes_for_all_styles(diagnostic_root, resolution):
    from grid_geometry.dataset import GeometryDataset

    dataset = GeometryDataset(diagnostic_root, "train", canvas_size=(resolution, resolution))
    for index in range(len(dataset)):
        sample = dataset.load_sample(index)
        projections = project_heatmaps(sample.heatmaps.numpy())
        for signals, expected in (
            (projections.x, sample.geometry.centers[0, :, 0]),
            (projections.y, sample.geometry.centers[:, 0, 1]),
        ):
            best = fit_axis(signals).hypotheses[0]
            assert best.cell_count == len(expected)
            np.testing.assert_allclose(best.centers, expected, atol=0.25)
