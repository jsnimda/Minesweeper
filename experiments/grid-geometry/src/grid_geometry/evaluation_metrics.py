"""Board-level geometry metrics in model-input coordinates."""

from dataclasses import dataclass

import numpy as np

from .geometry import FloatArray, Geometry
from .lattice import EdgeObservation


PIXEL_TOLERANCE = 1.50
NORMALIZED_THRESHOLDS = (("005", 0.05), ("010", 0.10), ("020", 0.20))
SIDES = ("left", "top", "right", "bottom")
STATUSES = ("fitted", "ambiguous", "insufficient_evidence")


@dataclass(frozen=True, eq=False)
class BoardTruth:
    centers: FloatArray
    pitch_x: float | None
    pitch_y: float | None
    origin_x: float | None
    origin_y: float | None
    visible_extent: tuple[float | None, ...]

    @property
    def rows(self) -> int:
        return self.centers.shape[0]

    @property
    def columns(self) -> int:
        return self.centers.shape[1]


def truth_from_geometry(geometry: Geometry) -> BoardTruth:
    def axis(centers: FloatArray) -> tuple[float | None, float | None]:
        if len(centers) < 2:
            return None, None
        pitch = float(np.median(np.diff(centers)))
        return pitch, float(centers[0] - pitch / 2)

    pitch_x, origin_x = axis(geometry.centers[0, :, 0])
    pitch_y, origin_y = axis(geometry.centers[:, 0, 1])
    return BoardTruth(
        geometry.centers, pitch_x, pitch_y, origin_x, origin_y,
        tuple(map(float, geometry.grid_box)),
    )


@dataclass(frozen=True, eq=False)
class BoardEstimate:
    """Ideal edges and separately observed visible endpoints of one hypothesis."""

    x_edges: FloatArray
    y_edges: FloatArray
    outer_edges: tuple[EdgeObservation | None, ...] = (None, None, None, None)
    score: float | None = None

    @property
    def rows(self) -> int:
        return len(self.y_edges) - 1

    @property
    def columns(self) -> int:
        return len(self.x_edges) - 1

    @property
    def pitch_x(self) -> float:
        return float((self.x_edges[-1] - self.x_edges[0]) / self.columns)

    @property
    def pitch_y(self) -> float:
        return float((self.y_edges[-1] - self.y_edges[0]) / self.rows)

    @property
    def centers(self) -> FloatArray:
        x = (self.x_edges[:-1] + self.x_edges[1:]) / 2
        y = (self.y_edges[:-1] + self.y_edges[1:]) / 2
        return np.stack(np.meshgrid(x, y), axis=-1)


@dataclass(frozen=True)
class BoardDecoding:
    status: str
    hypotheses: tuple[BoardEstimate, ...]
    axis_status: tuple[str, str] | None = None


CENTER_FIELDS = ("center_mean_px", "center_max_px", "center_mean_normalized", "center_max_normalized")
AXIS_ERROR_FIELDS = (
    "pitch_x_error_px", "pitch_y_error_px", "pitch_x_error_relative", "pitch_y_error_relative",
    "origin_x_error_px", "origin_y_error_px",
)
RANK_FIELDS = ("rank_complete_px", *(f"rank_complete_norm_{name}" for name, _ in NORMALIZED_THRESHOLDS))
RECOVERY_FIELDS = ("complete_px", *(f"complete_norm_{name}" for name, _ in NORMALIZED_THRESHOLDS))
GEOMETRY_FIELDS = (
    "status", "axis_x_status", "axis_y_status", "hypotheses", "score", "failure",
    "rows", "columns", "rows_correct", "columns_correct", "dimensions_correct",
    "pitch_x", "pitch_y", "origin_x", "origin_y", *AXIS_ERROR_FIELDS, *CENTER_FIELDS,
    *RECOVERY_FIELDS, *RANK_FIELDS, "accepted", "correct_acceptance", "incorrect_acceptance",
    "correct_rejection", "rejected_correct",
    *(f"{side}_{suffix}" for side in SIDES for suffix in ("state", "position", "error_px", "annotated")),
)


def _center_errors(truth: BoardTruth, prediction: BoardEstimate) -> tuple[FloatArray | None, FloatArray | None]:
    if (truth.rows, truth.columns) != (prediction.rows, prediction.columns):
        return None, None
    offset = prediction.centers - truth.centers
    pixels = np.hypot(offset[..., 0], offset[..., 1])
    if truth.pitch_x is None or truth.pitch_y is None:
        return pixels, None
    return pixels, np.hypot(offset[..., 0] / truth.pitch_x, offset[..., 1] / truth.pitch_y)


def evaluate_geometry(truth: BoardTruth, decoding: BoardDecoding) -> dict:
    """Evaluate the top hypothesis and search the shortlist for correct geometry."""
    if decoding.status not in STATUSES:
        raise ValueError(f"unknown decoder status {decoding.status!r}")
    result = dict.fromkeys(GEOMETRY_FIELDS)
    result.update(status=decoding.status, hypotheses=len(decoding.hypotheses))
    if decoding.axis_status is not None:
        result.update(axis_x_status=decoding.axis_status[0], axis_y_status=decoding.axis_status[1])
    top = decoding.hypotheses[0] if decoding.hypotheses else None
    rows_correct = top is not None and top.rows == truth.rows
    columns_correct = top is not None and top.columns == truth.columns
    result.update(rows_correct=rows_correct, columns_correct=columns_correct, dimensions_correct=rows_correct and columns_correct)

    if top is not None:
        result.update(rows=top.rows, columns=top.columns, score=top.score)
        for axis in ("x", "y"):
            pitch = getattr(top, f"pitch_{axis}")
            origin = float(getattr(top, f"{axis}_edges")[0])
            gold_pitch, gold_origin = getattr(truth, f"pitch_{axis}"), getattr(truth, f"origin_{axis}")
            result[f"pitch_{axis}"] = pitch
            result[f"origin_{axis}"] = origin
            if gold_pitch is not None:
                result[f"pitch_{axis}_error_px"] = abs(pitch - gold_pitch)
                result[f"pitch_{axis}_error_relative"] = abs(pitch - gold_pitch) / gold_pitch
            if gold_origin is not None:
                result[f"origin_{axis}_error_px"] = abs(origin - gold_origin)

    for rank, prediction in enumerate(decoding.hypotheses, 1):
        pixels, normalized = _center_errors(truth, prediction)
        if pixels is None:
            continue
        pixel_max = float(pixels.max())
        if rank == 1:
            result.update(center_mean_px=float(pixels.mean()), center_max_px=pixel_max)
        if result["rank_complete_px"] is None and pixel_max <= PIXEL_TOLERANCE:
            result["rank_complete_px"] = rank
        if normalized is None:
            continue
        normalized_max = float(normalized.max())
        if rank == 1:
            result.update(center_mean_normalized=float(normalized.mean()), center_max_normalized=normalized_max)
        for name, tolerance in NORMALIZED_THRESHOLDS:
            field = f"rank_complete_norm_{name}"
            if result[field] is None and normalized_max <= tolerance:
                result[field] = rank

    complete = result["rank_complete_px"] == 1
    result["complete_px"] = complete
    if truth.pitch_x is not None and truth.pitch_y is not None:
        for name, _ in NORMALIZED_THRESHOLDS:
            result[f"complete_norm_{name}"] = result[f"rank_complete_norm_{name}"] == 1

    edges = top.outer_edges if top is not None else (None,) * 4
    for side, reference, edge in zip(SIDES, truth.visible_extent, edges, strict=True):
        result[f"{side}_annotated"] = reference is not None
        result[f"{side}_state"] = "unknown" if edge is None else "censored" if edge.censored else "estimated"
        if edge is not None and not edge.censored:
            result[f"{side}_position"] = edge.position
            if reference is not None:
                result[f"{side}_error_px"] = abs(edge.position - reference)

    accepted = decoding.status == "fitted" and top is not None
    result.update(
        accepted=accepted, correct_acceptance=accepted and complete,
        incorrect_acceptance=accepted and not complete,
        correct_rejection=not accepted and not complete, rejected_correct=not accepted and complete,
    )
    if top is None:
        result["failure"] = "no_hypothesis"
    elif not result["dimensions_correct"]:
        result["failure"] = "wrong_dimensions"
    elif not complete:
        result["failure"] = "center_error_over_1.50px"
    elif not accepted:
        result["failure"] = decoding.status
    return result


def _rate(count: int, total: int) -> dict:
    return {"count": count, "total": total, "rate": count / total if total else None}


def _stats(numbers: list[float]) -> dict:
    return {
        "count": len(numbers),
        "mean": float(np.mean(numbers)) if numbers else None,
        "median": float(np.median(numbers)) if numbers else None,
        "max": max(numbers) if numbers else None,
    }


def summarize_geometry(boards: list[dict]) -> dict:
    total = len(boards)
    summary = {"boards": total}
    for field in (
        "rows_correct", "columns_correct", "dimensions_correct", "accepted", "correct_acceptance",
        "incorrect_acceptance", "correct_rejection", "rejected_correct",
    ):
        summary[field] = _rate(sum(board[field] for board in boards), total)
    summary["rejected"] = _rate(sum(not board["accepted"] for board in boards), total)
    summary["statuses"] = {status: _rate(sum(b["status"] == status for b in boards), total) for status in STATUSES}
    for complete, rank in zip(RECOVERY_FIELDS, RANK_FIELDS, strict=True):
        eligible = [board for board in boards if board[complete] is not None]
        summary[complete] = _rate(sum(board[complete] for board in eligible), len(eligible))
        summary[f"{complete}_recall"] = _rate(sum(board[rank] is not None for board in eligible), len(eligible))
    for field in (*AXIS_ERROR_FIELDS, *CENTER_FIELDS):
        summary[field] = _stats([board[field] for board in boards if board[field] is not None])

    visible_errors = []
    endpoints = {}
    annotated = 0
    for side in SIDES:
        eligible = [board for board in boards if board[f"{side}_annotated"]]
        errors = [board[f"{side}_error_px"] for board in eligible if board[f"{side}_error_px"] is not None]
        endpoints[side] = {
            "coverage": _rate(len(errors), len(eligible)),
            "censored": sum(board[f"{side}_state"] == "censored" for board in eligible),
            "unknown": sum(board[f"{side}_state"] == "unknown" for board in eligible),
            "error_px": _stats(errors),
        }
        annotated += len(eligible)
        visible_errors.extend(errors)
    summary["visible_endpoints"] = {
        "coverage": _rate(len(visible_errors), annotated), "error_px": _stats(visible_errors), "sides": endpoints,
    }
    return summary
