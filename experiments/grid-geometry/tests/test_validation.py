from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from grid_geometry.evaluate import decode_predictions
from grid_geometry.evaluation_metrics import BoardEstimate
from grid_geometry.lattice import EdgeObservation
from grid_geometry.validation import ValidationSettings, load_validation_settings, validate_geometry
from grid_geometry.validation_metrics import summarize_validation, validate_decoding, validation_record

from test_decoder import _targets


SETTINGS = ValidationSettings()


@pytest.fixture
def board():
    maps, x, y = _targets(182, 213, 12.4, 14.2, 18.1, 21.7, 8, 8, gap=(2.2, 1.4))
    visible = (x[0] + 1.1, y[0] + .7, x[-1] - 1.1, y[-1] - .7)
    edges = tuple(EdgeObservation(pos, 1.0, False) for pos in visible)
    return maps, BoardEstimate(x, y, edges)


def erase(maps, estimate, rows, columns, *, channel=slice(None)):
    x = np.arange(maps.shape[2]) + 0.5
    y = np.arange(maps.shape[1]) + 0.5
    x0, x1 = estimate.x_edges[list(columns)]
    y0, y1 = estimate.y_edges[list(rows)]
    maps[channel, (y >= y0) & (y < y1), :] *= (x < x0) | (x >= x1)


def test_clean_fractional_rectangular_grid_and_non_mutation(board):
    maps, estimate = board
    original = maps.copy()
    x, y = estimate.x_edges.copy(), estimate.y_edges.copy()
    maps.setflags(write=False)
    estimate.x_edges.setflags(write=False)
    estimate.y_edges.setflags(write=False)
    result = validate_geometry(maps, estimate, SETTINGS)
    assert result.accepted and result.reasons == ()
    assert result.centers.supported.shape == (8, 8)
    assert result.vertical.supported.shape == (8, 7)
    assert result.horizontal.supported.shape == (7, 8)
    assert result.measurements["center_coverage"] == 1
    assert result.measurements["confirmed_outer_edges"] == 4
    assert result.measurements["outer_edges"]["left"]["position"] != x[0]
    np.testing.assert_array_equal(maps, original)
    np.testing.assert_array_equal(estimate.x_edges, x)
    np.testing.assert_array_equal(estimate.y_edges, y)


def test_critical_corner_both_decoders_can_fit_but_must_reject(board):
    maps, estimate = board
    x = np.arange(maps.shape[2]) + 0.5
    y = np.arange(maps.shape[1]) + 0.5
    # Erase through the canvas edge so the visible outer lines also break.
    maps[:, (y[:, None] >= estimate.y_edges[4]) & (x >= estimate.x_edges[4])] = 0
    decodings, _ = decode_predictions(maps, "both")
    for decoding in decodings.values():
        best = decoding.hypotheses[0]
        assert decoding.status == "fitted"
        np.testing.assert_allclose(best.centers, estimate.centers, atol=0.3)
        strict, _ = validate_decoding(maps, decoding, SETTINGS)
        assert not strict.accepted
        assert strict.centers.supported.sum() == 48
        assert not strict.centers.supported[4:, 4:].any()
        assert strict.measurements["largest_missing_center_cluster"] == 16
        assert strict.measurements["confirmed_outer_edges"] == 2
        assert "missing_center_cluster" in strict.reasons
        assert strict.measurements["vertical_longest_unsupported_gap"] >= 3
        assert strict.measurements["horizontal_longest_unsupported_gap"] >= 3


@pytest.mark.parametrize(
    ("rows", "columns", "reason"),
    [
        ((2, 5), (3, 6), "missing_center_cluster"),
        ((3, 4), (0, 8), "unsupported_center_rows_or_columns"),
        ((0, 8), (2, 3), "unsupported_center_rows_or_columns"),
        ((7, 8), (0, 8), "unsupported_center_rows_or_columns"),
        ((0, 8), (7, 8), "unsupported_center_rows_or_columns"),
    ],
)
def test_central_strips_and_terminal_damage(board, rows, columns, reason):
    maps, estimate = board
    erase(maps, estimate, rows, columns)
    strict = validate_geometry(maps, estimate, SETTINGS)
    assert not strict.accepted
    assert reason in strict.reasons


def test_small_missing_groups_and_isolated_missing_centers_are_distinguished(board):
    maps, estimate = board
    clean = maps.copy()
    for row, col in ((1, 1), (1, 6), (4, 1), (4, 6), (6, 3), (6, 6)):
        erase(maps, estimate, (row, row + 1), (col, col + 1), channel=0)
    isolated = validate_geometry(maps, estimate, SETTINGS)
    assert isolated.accepted
    assert isolated.measurements["largest_missing_center_cluster"] == 1
    erase(clean, estimate, (1, 2), (1, 4), channel=0)
    erase(clean, estimate, (5, 6), (4, 7), channel=0)
    grouped = validate_geometry(clean, estimate, SETTINGS)
    assert grouped.reasons == ("missing_center_cluster",)
    assert grouped.measurements["center_coverage"] == isolated.measurements["center_coverage"]
    assert grouped.measurements["largest_missing_center_cluster"] == 3


def test_two_missing_boundary_segments_fail_even_with_full_center_support(board):
    maps, estimate = board
    y = np.arange(maps.shape[1]) + 0.5
    x = np.arange(maps.shape[2]) + 0.5
    mask = ((y[:, None] >= estimate.y_edges[2]) & (y[:, None] < estimate.y_edges[4])
            & (abs(x - estimate.x_edges[3]) < 7))
    maps[1, mask] = 0
    result = validate_geometry(maps, estimate, SETTINGS)
    assert result.measurements["center_coverage"] == 1
    assert result.measurements["vertical_unsupported_segments"] == 2
    assert result.measurements["vertical_coverage"] > SETTINGS.min_boundary_coverage
    assert result.reasons == ("vertical_boundary_gap",)


def test_cross_channel_consistency_when_only_centers_are_present(board):
    maps, estimate = board
    maps[1:] = 0
    strict = validate_geometry(maps, estimate, SETTINGS)
    assert strict.measurements["centers_without_boundary_support"] == 1
    assert "center_boundary_inconsistency" in strict.reasons


def test_wrong_hypothesis_shifts_peaks_or_leaves_exterior_evidence(board):
    maps, estimate = board
    shifted = replace(estimate, x_edges=estimate.x_edges + estimate.pitch_x / 2)
    assert "center_coverage" in validate_geometry(maps, shifted, SETTINGS).reasons
    shortened = replace(estimate, x_edges=estimate.x_edges[:-1], outer_edges=(
        *estimate.outer_edges[:2], EdgeObservation(estimate.x_edges[-2], 1, False), estimate.outer_edges[3],
    ))
    result = validate_geometry(maps, shortened, SETTINGS)
    assert result.measurements["center_coverage"] == 1
    assert "exterior_evidence" in result.reasons
    assert "outer_bottom_unsupported" not in result.reasons


def test_perpendicular_continuation_exposes_shortened_fit_before_global_exterior_threshold(board):
    maps, estimate = board
    short_edge = estimate.x_edges[-2]
    x = np.arange(maps.shape[2]) + 0.5
    maps[0, :, x > short_edge] = 0
    maps[1, :, x > short_edge + 5] = 0
    maps = maps[:, :, :int(short_edge + .46 * estimate.pitch_x)]
    shortened = replace(estimate, x_edges=estimate.x_edges[:-1], outer_edges=(
        *estimate.outer_edges[:2], EdgeObservation(short_edge, 1, False), estimate.outer_edges[3],
    ))
    result = validate_geometry(maps, shortened, SETTINGS)
    assert max(result.measurements["exterior_energy_fraction"].values()) < SETTINGS.max_exterior_energy
    assert result.measurements["exterior_ridge_coverage"]["right"] == 1
    assert result.reasons == ("outer_right_continuation",)


def test_outer_unknown_censored_and_unsupported_are_distinct(board):
    maps, estimate = board
    left, _, right, _ = estimate.outer_edges
    wrong_top = EdgeObservation(estimate.y_edges[0] + estimate.pitch_y, 1, False)
    estimate = replace(estimate, outer_edges=(None, wrong_top, right, replace(left, censored=True)))
    strict = validate_geometry(maps, estimate, SETTINGS)
    states = strict.measurements["outer_edges"]
    assert [states[side]["state"] for side in ("left", "top", "right", "bottom")] == [
        "unknown", "unsupported", "confirmed", "censored",
    ]
    assert "outer_top_unsupported" in strict.reasons
    assert "outer_left_unknown" in strict.reasons and "outer_bottom_censored" in strict.reasons
    assert set(strict.outer) == {"top", "right"}


@pytest.mark.parametrize("status", ["ambiguous", "insufficient_evidence"])
def test_decoder_status_cannot_pass_with_ideal_support(board, status):
    maps, estimate = board
    strict = validate_geometry(maps, estimate, SETTINGS, fitting_status=status)
    assert strict.reasons == (f"decoder_{status}",)
    assert strict.measurements["center_coverage"] == 1


def test_no_hypothesis_and_flat_probabilities(board):
    maps, estimate = board
    strict = validate_geometry(maps, None, SETTINGS, fitting_status="insufficient_evidence")
    assert strict.reasons == ("decoder_insufficient_evidence", "no_hypothesis")
    flat = validate_geometry(np.full_like(maps, 0.9), estimate, SETTINGS)
    assert flat.measurements["center_coverage"] == 0
    assert "center_coverage" in flat.reasons


def test_single_cell_has_no_internal_segments():
    maps, x, y = _targets(70, 65, 15.3, 12.7, 22.4, 25.2, 1, 1)
    estimate = BoardEstimate(x, y, tuple(EdgeObservation(pos, 1, False) for pos in (x[0], y[0], x[-1], y[-1])))
    result = validate_geometry(maps, estimate, SETTINGS)
    assert result.accepted
    assert result.measurements["vertical_coverage"] is None
    assert result.measurements["horizontal_coverage"] is None


def test_invalid_inputs_and_explicit_validation_settings(tmp_path, board):
    maps, estimate = board
    with pytest.raises(ValueError, match="shape"):
        validate_geometry(maps[0], estimate, SETTINGS)
    with pytest.raises(ValueError, match="probabilities"):
        validate_geometry(np.full_like(maps, 2), estimate, SETTINGS)
    assert validate_geometry(maps, replace(estimate, x_edges=np.array([10, 30, 41])), SETTINGS).reasons == ("invalid_lattice",)
    config = tmp_path / "settings.toml"
    config.write_text("[validation]\nmax_boundary_gap = 2\n")
    assert load_validation_settings(config) == replace(SETTINGS, max_boundary_gap=2)
    example = Path(__file__).resolve().parents[1] / "configs" / "strict_validation.toml"
    assert load_validation_settings(example) == SETTINGS
    for field in ("max_boundary_gap = 1.3", "min_center_response = nan", "min_boundary_coverage = true", "typo = 0.1"):
        config.write_text(f"[validation]\n{field}\n")
        with pytest.raises(ValueError):
            load_validation_settings(config)


def test_metrics_do_not_substitute_geometry_correctness_for_labelled_completeness(board):
    maps, estimate = board
    accepted = validate_geometry(maps, estimate, SETTINGS)
    rejected = validate_geometry(maps, estimate, SETTINGS, fitting_status="ambiguous")
    records = [
        validation_record(accepted, True, False, 1), validation_record(accepted, False, None, 2),
        validation_record(rejected, True, True, 3), validation_record(rejected, False, None, 4),
    ]
    stats = summarize_validation(records)
    for key in ("correct_acceptance", "incorrect_acceptance", "rejected_correct", "correct_rejection"):
        assert stats[key] == {"count": 1, "total": 4, "rate": 0.25}
    assert stats["geometry_accuracy_among_accepted"]["rate"] == 0.5
    assert stats["labelled_completeness"]["incomplete_board_acceptance"]["rate"] == 1
    assert stats["labelled_completeness"]["complete_board_rejection"]["rate"] == 1
    assert stats["rejection_reasons"]["decoder_ambiguous"]["count"] == 2
    assert summarize_validation(records[2:])["geometry_accuracy_among_accepted"]["rate"] is None
