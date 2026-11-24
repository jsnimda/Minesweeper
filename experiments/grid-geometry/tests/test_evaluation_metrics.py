from dataclasses import replace

import numpy as np
import pytest

from grid_geometry.evaluate import decode_predictions
from grid_geometry.evaluation_metrics import (
    BoardDecoding, BoardEstimate, evaluate_geometry, summarize_geometry, truth_from_geometry,
)
from grid_geometry.geometry import Geometry
from grid_geometry.heatmaps import geometry_heatmaps
from grid_geometry.lattice import EdgeObservation


@pytest.fixture
def gapped_truth(annotation):
    fields = ("size", "rows", "columns", "grid_box", "x_edges", "y_edges", "cell_boxes", "centers")
    return truth_from_geometry(Geometry(**{field: annotation[field] for field in fields}))


def _estimate(truth, *, dx=0, dy=0, step_x=0, step_y=0, rows=None, columns=None, **kwargs):
    count_x = truth.columns if columns is None else columns
    count_y = truth.rows if rows is None else rows
    return BoardEstimate(
        truth.origin_x + dx + (truth.pitch_x + step_x) * np.arange(count_x + 1),
        truth.origin_y + dy + (truth.pitch_y + step_y) * np.arange(count_y + 1),
        **kwargs,
    )


def test_ideal_truth_origin_is_different_from_microsoft_visible_extent(gapped_truth):
    assert gapped_truth.visible_extent == (2.25, 1.5, 25.25, 14.5)
    assert (gapped_truth.origin_x, gapped_truth.origin_y) == (1.75, 1.0)
    assert (gapped_truth.pitch_x, gapped_truth.pitch_y) == (8.0, 7.0)
    record = evaluate_geometry(gapped_truth, BoardDecoding("fitted", (_estimate(gapped_truth),)))
    assert record["dimensions_correct"] and record["complete_px"]
    assert record["center_max_px"] == 0
    assert record["correct_acceptance"]
    assert record["rank_complete_norm_005"] == 1
    assert record["left_state"] == "unknown"


def test_fractional_pitch_origin_and_separate_axis_normalization(gapped_truth):
    truth = gapped_truth
    shifted = _estimate(truth, dx=0.8, dy=0.35, step_x=0.21, step_y=-0.13)
    result = evaluate_geometry(truth, BoardDecoding("fitted", (shifted,)))
    assert result["pitch_x_error_px"] == pytest.approx(0.21)
    assert result["pitch_x_error_relative"] == pytest.approx(0.21 / 8)
    assert result["pitch_y_error_px"] == pytest.approx(0.13)
    assert result["pitch_y_error_relative"] == pytest.approx(0.13 / 7)
    assert result["origin_x_error_px"] == pytest.approx(0.8)
    assert result["origin_y_error_px"] == pytest.approx(0.35)

    dx = np.array([0.905, 1.115, 1.325])
    dy = np.array([0.285, 0.155])
    euclidean = np.hypot(dx[None, :], dy[:, None])
    normalized = np.hypot(dx[None, :] / 8, dy[:, None] / 7)
    assert result["center_mean_px"] == pytest.approx(euclidean.mean())
    assert result["center_max_px"] == pytest.approx(euclidean.max())
    assert result["center_mean_normalized"] == pytest.approx(normalized.mean())
    assert result["center_max_normalized"] == pytest.approx(normalized.max())
    assert result["complete_px"] and result["complete_norm_020"]
    assert not result["complete_norm_005"] and not result["complete_norm_010"]


def test_historical_and_normalized_complete_recovery_can_disagree(gapped_truth):
    on_limit = evaluate_geometry(gapped_truth, BoardDecoding("fitted", (_estimate(gapped_truth, dx=1.50),)))
    beyond = evaluate_geometry(gapped_truth, BoardDecoding("fitted", (_estimate(gapped_truth, dx=1.52),)))
    assert on_limit["complete_px"]
    assert not beyond["complete_px"]
    assert beyond["complete_norm_020"]
    assert beyond["incorrect_acceptance"]


def test_wrong_dimensions_omit_localization_but_keep_alternative_recall(gapped_truth):
    wrong = _estimate(gapped_truth, columns=2)
    correct = _estimate(gapped_truth)
    result = evaluate_geometry(gapped_truth, BoardDecoding("ambiguous", (wrong, correct)))
    assert result["rows_correct"] and not result["columns_correct"]
    assert result["center_mean_px"] is None and result["center_max_normalized"] is None
    assert result["pitch_x_error_px"] == 0
    assert result["correct_rejection"] and not result["complete_px"]
    assert result["rank_complete_px"] == result["rank_complete_norm_005"] == 2
    summary = summarize_geometry([result])
    assert summary["center_mean_px"]["count"] == 0
    assert summary["complete_px_recall"]["count"] == 1


def test_unknown_and_censored_endpoints_stay_in_coverage_denominator(gapped_truth):
    edges = (None, EdgeObservation(0.5, 0.8, True), EdgeObservation(25.5, 0.9, False), None)
    one = evaluate_geometry(gapped_truth, BoardDecoding("fitted", (_estimate(gapped_truth, outer_edges=edges),)))
    empty = evaluate_geometry(gapped_truth, BoardDecoding("insufficient_evidence", ()))
    assert one["top_state"] == "censored"
    assert one["top_position"] is None and one["top_error_px"] is None
    assert one["right_error_px"] == 0.25
    summary = summarize_geometry([one, empty])
    assert summary["visible_endpoints"]["coverage"] == {"count": 1, "total": 8, "rate": 0.125}
    assert summary["visible_endpoints"]["error_px"] == {"count": 1, "mean": 0.25, "median": 0.25, "max": 0.25}
    assert summary["visible_endpoints"]["sides"]["top"]["censored"] == 1
    assert summary["visible_endpoints"]["sides"]["top"]["unknown"] == 1


def test_status_and_acceptance_aggregation_sums_all_boards(gapped_truth):
    correct = _estimate(gapped_truth)
    wrong = _estimate(gapped_truth, rows=3)
    decodings = [
        BoardDecoding("fitted", (correct,)), BoardDecoding("fitted", (wrong,)),
        BoardDecoding("ambiguous", (correct,)), BoardDecoding("insufficient_evidence", ()),
    ]
    totals = summarize_geometry([evaluate_geometry(gapped_truth, value) for value in decodings])
    assert totals["boards"] == 4
    for field in ("correct_acceptance", "incorrect_acceptance", "correct_rejection", "rejected_correct"):
        assert totals[field]["count"] == 1
    assert totals["rejected"]["rate"] == 0.5
    assert totals["statuses"]["ambiguous"]["count"] == 1
    assert totals["center_mean_px"]["count"] == totals["complete_px"]["count"] == 2


def test_axis_with_no_annotated_pitch_has_undefined_normalized_metrics(gapped_truth):
    truth = replace(gapped_truth, centers=gapped_truth.centers[:, :1], pitch_x=None, origin_x=None)
    gold = _estimate(gapped_truth, columns=1)
    record = evaluate_geometry(truth, BoardDecoding("fitted", (gold,)))
    assert record["complete_px"]
    assert record["pitch_x_error_px"] is None
    assert record["complete_norm_020"] is None
    assert summarize_geometry([record])["complete_norm_020"]["total"] == 0


def _rectangular_geometry():
    nx, ny = 9, 6
    x = 17.4 + 18.15 * np.arange(nx + 1)
    y = 19.3 + 17.2 * np.arange(ny + 1)
    boxes = np.stack(np.broadcast_arrays(x[:-1], y[:-1, None], x[1:], y[1:, None]), axis=-1)
    centers = (boxes[..., :2] + boxes[..., 2:]) / 2
    return Geometry((205, 159), ny, nx, [x[0], y[0], x[-1], y[-1]], x, y, boxes, centers)


def test_compare_decoders_on_perfect_and_corrupted_target_heatmaps():
    geometry = _rectangular_geometry()
    truth = truth_from_geometry(geometry)
    maps = geometry_heatmaps(geometry)
    results, timing = decode_predictions(maps, "both")
    assert set(results) == {"independent", "joint"}
    for decoding in results.values():
        assert evaluate_geometry(truth, decoding)["complete_px"]
    assert timing["joint_total_ms"] == timing["axis_fit_ms"] + timing["joint_additional_ms"]
    assert timing["independent_ms"] >= timing["axis_fit_ms"] > 0

    x = np.arange(geometry.size[0]) + 0.5
    maps[0, :, x > geometry.x_edges[-2]] = 0
    maps[1, :, x > geometry.x_edges[-1] - 5] = 0
    results, _ = decode_predictions(maps, "both")
    independent = evaluate_geometry(truth, results["independent"])
    joint = evaluate_geometry(truth, results["joint"])
    assert not independent["complete_px"]
    assert joint["complete_px"]
    assert joint["right_state"] == "unknown"
    empty, _ = decode_predictions(np.zeros_like(maps), "both")
    assert all(item.status == "insufficient_evidence" and not item.hypotheses for item in empty.values())
