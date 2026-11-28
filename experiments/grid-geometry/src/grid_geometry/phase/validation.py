"""Rejection-first evidence checks for a fixed phase-grid hypothesis.

Thresholds are provisional heuristics, not calibrated recognition probabilities.
The validator never changes the supplied geometry.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from ..spatial import sample_bilinear
from .decoder import GridHypothesis, _fields


@dataclass(frozen=True)
class ValidationSettings:
    min_occupancy: float = .5
    min_magnitude: float = .25
    min_phase_alignment: float = .65
    min_global_coherence: float = .8
    min_cell_coverage: float = .95
    min_row_column_coverage: float = .75
    max_missing_block: int = 2
    min_side_support: float = .8
    max_exterior_occupancy: float = .25
    max_visible_offset_cells: float = .3

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == 'max_missing_block':
                if type(value) is not int or value < 1:
                    raise ValueError("max_missing_block must be positive")
            elif not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reasons: tuple[str, ...]
    measurements: dict
    cell_occupancy: np.ndarray
    cell_magnitude_x: np.ndarray
    cell_magnitude_y: np.ndarray
    cell_alignment_x: np.ndarray
    cell_alignment_y: np.ndarray
    supported: np.ndarray
    row_support: np.ndarray
    column_support: np.ndarray
    side_evidence: dict = field(default_factory=dict)
    local_accepted: bool = False
    local_reasons: tuple[str, ...] = ()
    observable: np.ndarray | None = None

    @property
    def status(self):
        return 'accepted' if self.accepted else 'rejected'


def _sample(field, x, y):
    return sample_bilinear(field, np.asarray(x, dtype=float), np.asarray(y, dtype=float))


def _block_missing(missing, width):
    """Detect a fully missing width-by-width block without a cell loop."""
    if min(missing.shape) < width:
        return False
    integral = np.pad(missing.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = integral[width:, width:] - integral[:-width, width:] - integral[width:, :-width] + integral[:-width, :-width]
    return bool(np.any(sums == width * width))


def validate_phase(fields, hypothesis: GridHypothesis, *, decoder_status="fitted",
                   settings: ValidationSettings = ValidationSettings()):
    """Independently inspect a fixed proposal using the original activated fields.

    ``decoder_status`` should be the corresponding DecodeResult.status. Passing
    an ambiguous or insufficient result rejects it even if local samples agree.
    """
    values = _fields(fields)
    height, width = values.shape[1:]
    x, y = hypothesis.x, hypothesis.y
    if (not all(math.isfinite(v) for v in (x.origin, x.pitch, y.origin, y.pitch))
            or x.pitch <= 0 or y.pitch <= 0 or type(x.count) is not int or type(y.count) is not int
            or min(x.count, y.count) < 1 or x.count * y.count > 1_000_000):
        raise ValueError("invalid or excessive lattice geometry")
    xs = x.origin + (np.arange(x.count) + .5) * x.pitch
    ys = y.origin + (np.arange(y.count) + .5) * y.pitch
    observable = (ys[:, None] >= .5) & (ys[:, None] <= height - .5) & (xs[None, :] >= .5) & (xs[None, :] <= width - .5)
    occupancy = _sample(values[4], xs, ys)
    xc, xsin = _sample(values[0], xs, ys), _sample(values[1], xs, ys)
    yc, ysin = _sample(values[2], xs, ys), _sample(values[3], xs, ys)
    mx, my = np.hypot(xc, xsin), np.hypot(yc, ysin)
    # Every ideal cell center has phase pi relative to its corresponding axis.
    ax = -xc / np.maximum(mx, 1e-12)
    ay = -yc / np.maximum(my, 1e-12)
    supported = (observable & (occupancy >= settings.min_occupancy) & (mx >= settings.min_magnitude)
                 & (my >= settings.min_magnitude) & (ax >= settings.min_phase_alignment)
                 & (ay >= settings.min_phase_alignment))
    row_count, column_count = observable.sum(axis=1), observable.sum(axis=0)
    rows = np.divide(supported.sum(axis=1), row_count, out=np.zeros(len(ys)), where=row_count > 0)
    columns = np.divide(supported.sum(axis=0), column_count, out=np.zeros(len(xs)), where=column_count > 0)
    weighted = observable * (occupancy >= settings.min_occupancy)
    denominator = max(int(weighted.sum()), 1)
    coherence_x = float(np.hypot((xc * weighted).sum(), (xsin * weighted).sum()) / max((mx * weighted).sum(), 1e-12))
    coherence_y = float(np.hypot((yc * weighted).sum(), (ysin * weighted).sum()) / max((my * weighted).sum(), 1e-12))
    reasons = []
    if decoder_status != "fitted":
        reasons.append(f"decoder status: {decoder_status}")
    if not observable.any():
        reasons.append("no observable cell centers")
    if np.count_nonzero(row_count) < 2 or np.count_nonzero(column_count) < 2:
        reasons.append("fewer than two observable rows or columns")
    if min(coherence_x, coherence_y) < settings.min_global_coherence:
        reasons.append("weak directional phase coherence")
    coverage = float(supported.sum() / max(observable.sum(), 1))
    if coverage < settings.min_cell_coverage:
        reasons.append("insufficient local cell support")
    if np.any(rows[row_count > 0] < settings.min_row_column_coverage):
        reasons.append("unsupported row")
    if np.any(columns[column_count > 0] < settings.min_row_column_coverage):
        reasons.append("unsupported column")
    if _block_missing(observable & ~supported, settings.max_missing_block):
        reasons.append("cluster of unsupported cells")
    local_reasons = list(reasons)
    if not observable.all():
        reasons.append("cell centers outside observable image")
    side_evidence = {}
    # Side probes stay a fraction of a pitch from the fitted ideal boundary.
    # Side samples use cell-center positions along the side and therefore do
    # not average away an entirely unsupported terminal strip.
    sides = (
        ("left", x.origin, x.pitch, True, True, x.start_censored, x.visible_start),
        ("right", x.end, x.pitch, True, False, x.end_censored, x.visible_end),
        ("top", y.origin, y.pitch, False, True, y.start_censored, y.visible_start),
        ("bottom", y.end, y.pitch, False, False, y.end_censored, y.visible_end),
    )
    for name, edge, pitch, horizontal, leading, censored, observed in sides:
        along = ys[row_count > 0] if horizontal else xs[column_count > 0]
        inward = 1 if leading else -1
        inner = edge + inward * .35 * pitch
        outer = edge - inward * .35 * pitch
        limit = width if horizontal else height
        has_exterior = .5 <= outer <= limit - .5
        has_interior = .5 <= inner <= limit - .5
        if horizontal:
            inside = _sample(values[4], [inner], along)[:, 0]
            outside = _sample(values[4], [outer], along)[:, 0]
            outside_cos = _sample(values[0], [outer], along)[:, 0]
            outside_sin = _sample(values[1], [outer], along)[:, 0]
        else:
            inside = _sample(values[4], along, [inner])[0]
            outside = _sample(values[4], along, [outer])[0]
            outside_cos = _sample(values[2], along, [outer])[0]
            outside_sin = _sample(values[3], along, [outer])[0]
        expected_angle = 2 * np.pi * (outer - edge) / pitch
        exterior_alignment = outside_cos * np.cos(expected_angle) + outside_sin * np.sin(expected_angle)
        exterior_phase = float(np.mean((outside >= settings.min_occupancy) &
                                       (exterior_alignment >= settings.min_magnitude))) if has_exterior and along.size else None
        inner_support = float(np.mean(inside >= settings.min_occupancy)) if has_interior and along.size else None
        exterior = float(np.mean(outside)) if has_exterior and along.size else None
        offset = None if observed is None else abs(observed - edge) / pitch
        side_evidence[name] = {"inner_support": inner_support, "exterior_occupancy": exterior,
                               "visible_offset_cells": offset, "exterior_phase_support": exterior_phase,
                               "censored": bool(censored or not has_exterior)}
        if censored or not has_exterior or observed is None:
            reasons.append(f"{name} termination unobservable")
        if inner_support is None or inner_support < settings.min_side_support:
            reasons.append(f"{name} lacks terminal support")
        if exterior is not None and exterior > settings.max_exterior_occupancy:
            reasons.append(f"{name} exterior occupancy indicates continuation")
            if not censored and exterior_phase is not None and exterior_phase >= settings.min_side_support:
                local_reasons.append(f"{name} extent omits supported lattice")
        if offset is not None and offset > settings.max_visible_offset_cells:
            reasons.append(f"{name} visible boundary inconsistent with lattice")
            if not censored:
                local_reasons.append(f"{name} visible boundary inconsistent with lattice")
    measurements = {"cell_coverage": coverage, "observable_cells": int(observable.sum()), "coherence_x": coherence_x,
                    "coherence_y": coherence_y, "mean_magnitude_x": float((mx * weighted).sum() / denominator),
                    "mean_magnitude_y": float((my * weighted).sum() / denominator),
                    "minimum_row_support": float(rows.min()), "minimum_column_support": float(columns.min())}
    return ValidationResult(not reasons, tuple(reasons), measurements, occupancy, mx, my, ax, ay,
                            supported, rows, columns, side_evidence, not local_reasons, tuple(local_reasons), observable)
