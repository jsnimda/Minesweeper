"""Check spatial evidence for a fixed lattice against original probability maps."""

from dataclasses import dataclass, fields
from pathlib import Path
import tomllib

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .evaluation_metrics import BoardEstimate, SIDES
from .signals import FloatArray, probability_array
from .spatial import sample_bilinear


@dataclass(frozen=True)
class ValidationSettings:
    """Provisional thresholds for controlled numerical tests."""

    min_center_response: float = 0.40
    min_boundary_response: float = 0.38
    min_center_contrast: float = 0.12
    min_boundary_contrast: float = 0.14
    min_relative_response: float = 0.45
    min_center_coverage: float = 0.90
    min_boundary_coverage: float = 0.88
    min_segment_continuity: float = 0.65
    min_line_continuity: float = 0.75
    max_boundary_gap: int = 1
    max_missing_center_cluster: int = 2
    max_centers_without_boundaries: float = 0.07
    max_exterior_energy: float = 0.08
    max_exterior_ridge_coverage: float = 0.50
    max_outer_offset_cells: float = 0.25

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in ("max_boundary_gap", "max_missing_center_cluster"):
                if type(value) is not int or value < 0:
                    raise ValueError(f"{field.name} must be a nonnegative integer")
            elif type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError(f"{field.name} must be a number in [0, 1]")


def load_validation_settings(path: Path) -> ValidationSettings:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if set(config) != {"validation"} or not isinstance(config["validation"], dict):
        raise ValueError("expected a [validation] table with validation thresholds")
    try:
        return ValidationSettings(**config["validation"])
    except TypeError as exc:
        raise ValueError(f"invalid validation thresholds: {exc}") from exc


@dataclass(frozen=True, eq=False)
class SpatialSupport:
    response: FloatArray
    contrast: FloatArray
    continuity: FloatArray
    supported: NDArray[np.bool_]

    def to_dict(self) -> dict:
        return {field.name: getattr(self, field.name).tolist() for field in fields(self)}


@dataclass(frozen=True, eq=False)
class ValidationResult:
    status: str
    reasons: tuple[str, ...]
    measurements: dict
    centers: SpatialSupport | None = None
    vertical: SpatialSupport | None = None
    horizontal: SpatialSupport | None = None
    outer: dict[str, SpatialSupport] | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    def to_dict(self, *, include_support: bool = False) -> dict:
        result = {"status": self.status, "reasons": self.reasons, "measurements": self.measurements}
        if include_support and self.centers is not None:
            result["support"] = {
                "centers": self.centers.to_dict(),
                "vertical": self.vertical.to_dict(),
                "horizontal": self.horizontal.to_dict(),
                "outer": {side: support.to_dict() for side, support in self.outer.items()},
            }
        return result


def _axis_pitch(edges: FloatArray) -> float | None:
    if edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all():
        return None
    spacing = np.diff(edges)
    pitch = float(spacing.mean())
    if pitch <= 0 or not np.allclose(spacing, pitch, atol=1e-6, rtol=1e-6):
        return None
    return pitch


def _center_support(field, x, y, pitch_x, pitch_y, settings) -> SpatialSupport:
    response = np.maximum.reduce([
        sample_bilinear(field, x + dx * pitch_x, y + dy * pitch_y)
        for dx in (-0.09, 0, 0.09) for dy in (-0.09, 0, 0.09)
    ])
    background = np.mean([
        sample_bilinear(field, x - 0.28 * pitch_x, y), sample_bilinear(field, x + 0.28 * pitch_x, y),
        sample_bilinear(field, x, y - 0.28 * pitch_y), sample_bilinear(field, x, y + 0.28 * pitch_y),
    ], axis=0)
    contrast = response - background
    floor = max(settings.min_center_response, settings.min_relative_response * np.quantile(response, 0.90))
    supported = (response >= floor) & (contrast >= settings.min_center_contrast)
    return SpatialSupport(response, contrast, supported.astype(float), supported)


def _ridge_samples(field, positions, span, pitch) -> tuple[FloatArray, FloatArray]:
    """Sample each segment in longitudinal order with a narrow transverse tolerance."""
    within = np.array([0.23, 0.50, 0.77])
    along = span[:-1, None] + np.diff(span)[:, None] * within
    response = np.maximum.reduce([
        sample_bilinear(field, positions + offset * pitch, along.ravel()) for offset in (-0.08, 0, 0.08)
    ])
    background = (
        sample_bilinear(field, positions - 0.26 * pitch, along.ravel())
        + sample_bilinear(field, positions + 0.26 * pitch, along.ravel())
    ) / 2
    shape = (len(span) - 1, len(within), len(positions))
    return response.reshape(shape), (response - background).reshape(shape)


def _ridge_support(response, contrast, floor, settings, *, transpose=False) -> SpatialSupport:
    continuity = ((response >= floor) & (contrast >= settings.min_boundary_contrast)).mean(axis=1)
    values = (response.mean(axis=1), contrast.mean(axis=1), continuity, continuity >= settings.min_segment_continuity)
    return SpatialSupport(*(value.T if transpose else value for value in values))


def _coverage(supported: NDArray[np.bool_]) -> float | None:
    return float(supported.mean()) if supported.size else None


def _largest_cluster(missing: NDArray[np.bool_]) -> int:
    unseen = set(zip(*missing.nonzero(), strict=True))
    largest = 0
    while unseen:
        pending = [unseen.pop()]
        size = 0
        while pending:
            row, col = pending.pop()
            size += 1
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    pending.append(neighbor)
        largest = max(largest, size)
    return largest


def _line_evidence(segments: SpatialSupport, *, along_rows: bool) -> tuple[float | None, int]:
    masks = segments.supported.T if along_rows else segments.supported
    rates = segments.continuity.T if along_rows else segments.continuity
    weakest = float(rates.mean(axis=1).min()) if rates.size else None
    longest = 0
    for mask in masks:
        supported_indices = np.flatnonzero(np.r_[True, mask, True])
        longest = max(longest, int(np.diff(supported_indices).max() - 1))
    return weakest, longest


def _exterior_energy(maps, x, y, pitch_x, pitch_y, settings) -> dict:
    # Leave a margin for Gaussian tails at confirmed edges.
    columns = np.arange(maps.shape[2]) + 0.5
    rows = np.arange(maps.shape[1]) + 0.5
    inside_x = (columns >= x[0] - 0.5 * pitch_x) & (columns <= x[-1] + 0.5 * pitch_x)
    inside_y = (rows >= y[0] - 0.5 * pitch_y) & (rows <= y[-1] + 0.5 * pitch_y)
    inside = inside_y[:, None] & inside_x
    result = {}
    for name, field, floor in zip(
        ("centers", "vertical", "horizontal"), maps,
        (settings.min_center_response, settings.min_boundary_response, settings.min_boundary_response), strict=True,
    ):
        excess = np.maximum(field - floor, 0) ** 2
        total = excess.sum()
        result[name] = float(max(0, 1 - excess[inside].sum() / total)) if total else 0.0
    return result


def _boundary_evidence(field, edges, span, pitch, observations, settings, *, transpose):
    # Derive the relative response floor at the ideal lattice, independent of the outer observations.
    expected_response, expected_contrast = _ridge_samples(field, edges, span, pitch)
    floor = max(settings.min_boundary_response, settings.min_relative_response * np.quantile(expected_response, 0.90))
    internal = _ridge_support(
        expected_response[:, :, 1:-1], expected_contrast[:, :, 1:-1], floor, settings, transpose=transpose,
    )
    details = {}
    support = {}
    normal_size = field.shape[1]
    for side, endpoint, observation in observations:
        if observation is None or observation.censored:
            details[side] = {"state": "unknown" if observation is None else "censored", "confirmed": False}
            continue
        position = observation.position
        if not np.isfinite(position) or not 0 <= position <= normal_size:
            details[side] = {"state": "unsupported", "confirmed": False}
            continue
        response, contrast = _ridge_samples(field, np.array([position]), span, pitch)
        segments = _ridge_support(response, contrast, floor, settings)
        continuity, gap = _line_evidence(segments, along_rows=True)
        offset = abs(position - endpoint) / pitch
        confirmed = (
            offset <= settings.max_outer_offset_cells and continuity >= settings.min_line_continuity
            and gap <= settings.max_boundary_gap
        )
        details[side] = {
            "state": "confirmed" if confirmed else "unsupported", "confirmed": bool(confirmed),
            "position": float(position), "offset_cells": float(offset), "continuity": continuity,
            "longest_unsupported_gap": gap, "segment_coverage": _coverage(segments.supported),
        }
        support[side] = SpatialSupport(*(getattr(segments, field.name)[:, 0] for field in fields(SpatialSupport)))
    return internal, details, support, floor


def _exterior_continuation(
    maps, x, y, pitch_x, pitch_y, outer_details, vertical_floor, horizontal_floor, settings,
) -> dict[str, float | None]:
    """Check for coherent perpendicular ridges crossing an observed endpoint."""
    coverage = {}
    for side, field, ridges, ridge_pitch, span_pitch, floor, direction in (
        ("left", maps[2].T, y[1:-1], pitch_y, pitch_x, horizontal_floor, -1),
        ("right", maps[2].T, y[1:-1], pitch_y, pitch_x, horizontal_floor, 1),
        ("top", maps[1], x[1:-1], pitch_x, pitch_y, vertical_floor, -1),
        ("bottom", maps[1], x[1:-1], pitch_x, pitch_y, vertical_floor, 1),
    ):
        position = outer_details[side].get("position")
        if position is None or not ridges.size:
            coverage[side] = None
            continue
        span = position + np.array([0.0, 0.48]) * direction * span_pitch
        response, contrast = _ridge_samples(field, ridges, span, ridge_pitch)
        coverage[side] = _coverage(_ridge_support(response, contrast, floor, settings).supported)
    return coverage


def validate_geometry(
    heatmaps: ArrayLike, hypothesis: BoardEstimate | None, settings: ValidationSettings, *,
    fitting_status: str = "fitted",
) -> ValidationResult:
    """Assess the supplied hypothesis using original probabilities and no geometry edits."""
    maps = probability_array(heatmaps)
    if maps.ndim != 3 or maps.shape[0] != 3 or not all(maps.shape[1:]):
        raise ValueError("expected three nonempty heatmaps with shape [channels, height, width]")
    reasons = []
    if fitting_status != "fitted":
        reasons.append(f"decoder_{fitting_status}")
    if hypothesis is None:
        return ValidationResult("rejected", (*reasons, "no_hypothesis"), {})
    x, y = np.asarray(hypothesis.x_edges, dtype=float), np.asarray(hypothesis.y_edges, dtype=float)
    pitch_x, pitch_y = _axis_pitch(x), _axis_pitch(y)
    if pitch_x is None or pitch_y is None or len(hypothesis.outer_edges) != len(SIDES):
        return ValidationResult("rejected", (*reasons, "invalid_lattice"), {})
    centers_x = (x[:-1] + x[1:]) / 2
    centers_y = (y[:-1] + y[1:]) / 2
    if not (0 <= centers_x[0] <= centers_x[-1] <= maps.shape[2]
            and 0 <= centers_y[0] <= centers_y[-1] <= maps.shape[1]):
        return ValidationResult("rejected", (*reasons, "centers_outside_canvas"), {})

    centers = _center_support(maps[0], centers_x, centers_y, pitch_x, pitch_y, settings)
    left, top, right, bottom = hypothesis.outer_edges
    vertical, vertical_outer, outer_v, vertical_floor = _boundary_evidence(
        maps[1], x, y, pitch_x, (("left", x[0], left), ("right", x[-1], right)), settings, transpose=False,
    )
    horizontal, horizontal_outer, outer_h, horizontal_floor = _boundary_evidence(
        maps[2].T, y, x, pitch_y, (("top", y[0], top), ("bottom", y[-1], bottom)), settings, transpose=True,
    )
    outer = outer_v | outer_h
    outer_details = vertical_outer | horizontal_outer
    center_coverage = _coverage(centers.supported)
    missing_cluster = _largest_cluster(~centers.supported)
    empty_rows = np.flatnonzero(~centers.supported.any(axis=1)).tolist()
    empty_cols = np.flatnonzero(~centers.supported.any(axis=0)).tolist()
    if center_coverage < settings.min_center_coverage:
        reasons.append("center_coverage")
    if empty_rows or empty_cols:
        reasons.append("unsupported_center_rows_or_columns")
    if missing_cluster > settings.max_missing_center_cluster:
        reasons.append("missing_center_cluster")

    measurements = {
        "center_coverage": center_coverage, "unsupported_center_rows": empty_rows,
        "unsupported_center_columns": empty_cols, "largest_missing_center_cluster": missing_cluster,
        "minimum_center_response": float(centers.response.min()), "outer_edges": outer_details,
        "confirmed_outer_edges": sum(detail["confirmed"] for detail in outer_details.values()),
    }
    for name, segments, along_rows in (("vertical", vertical, True), ("horizontal", horizontal, False)):
        coverage = _coverage(segments.supported)
        weakest, gap = _line_evidence(segments, along_rows=along_rows)
        measurements.update({
            f"{name}_coverage": coverage, f"{name}_weakest_line_continuity": weakest,
            f"{name}_longest_unsupported_gap": gap,
            f"{name}_unsupported_segments": int(segments.supported.size - np.count_nonzero(segments.supported)),
        })
        if coverage is not None and coverage < settings.min_boundary_coverage:
            reasons.append(f"{name}_boundary_coverage")
        if weakest is not None and weakest < settings.min_line_continuity:
            reasons.append(f"{name}_boundary_continuity")
        if gap > settings.max_boundary_gap:
            reasons.append(f"{name}_boundary_gap")

    def outer_mask(side, length):
        if side in outer:
            return outer[side].supported
        return np.zeros(length, dtype=bool)

    rows, cols = centers.supported.shape
    v = np.column_stack((outer_mask("left", rows), vertical.supported, outer_mask("right", rows)))
    h = np.vstack((outer_mask("top", cols), horizontal.supported, outer_mask("bottom", cols)))
    boundary_support = (v[:, :-1] | v[:, 1:]) & (h[:-1] | h[1:])
    supported_centers = int(np.count_nonzero(centers.supported))
    mismatch = int(np.count_nonzero(centers.supported & ~boundary_support))
    fraction = mismatch / supported_centers if supported_centers else None
    measurements["centers_without_boundary_support"] = fraction
    if fraction is not None and fraction > settings.max_centers_without_boundaries:
        reasons.append("center_boundary_inconsistency")

    exterior = _exterior_energy(maps, x, y, pitch_x, pitch_y, settings)
    measurements["exterior_energy_fraction"] = exterior
    if max(exterior.values()) > settings.max_exterior_energy:
        reasons.append("exterior_evidence")
    continuation = _exterior_continuation(
        maps, x, y, pitch_x, pitch_y, outer_details, vertical_floor, horizontal_floor, settings,
    )
    measurements["exterior_ridge_coverage"] = continuation
    for side in SIDES:
        state = outer_details[side]["state"]
        if state != "confirmed":
            reasons.append(f"outer_{side}_{state}")
        coverage = continuation[side]
        if coverage is not None and coverage > settings.max_exterior_ridge_coverage:
            reasons.append(f"outer_{side}_continuation")
    return ValidationResult(
        "rejected" if reasons else "accepted", tuple(reasons), measurements, centers, vertical, horizontal, outer,
    )
