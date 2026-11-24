"""Jointly rank finite grid hypotheses against center and boundary heatmaps."""

from dataclasses import dataclass, replace
import math
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .lattice import AxisLatticeFit, EdgeObservation, LatticeHypothesis, fit_axis
from .pitch import estimate_pitch
from .signals import FloatArray, probability_array, project_heatmaps
from .spatial import sample_bilinear as _sample


@dataclass(frozen=True)
class IndependentFits:
    x: AxisLatticeFit
    y: AxisLatticeFit


@dataclass(frozen=True)
class PitchRatioPrior:
    """An explicitly supplied soft prior on pitch_x / pitch_y."""

    ratio: float
    weight: float = 0.12
    log_sigma: float = 0.12

    def __post_init__(self) -> None:
        if any(not isinstance(value, Real) or not math.isfinite(value)
               for value in (self.ratio, self.weight, self.log_sigma)):
            raise ValueError("pitch-ratio prior must use finite numbers")
        if self.ratio <= 0 or self.log_sigma <= 0 or not 0 <= self.weight <= 1:
            raise ValueError("pitch ratio and log_sigma must be positive and weight must be in [0, 1]")


@dataclass(frozen=True)
class AxisSpan:
    """An axis fit with optional integer edits to its two terminal boundaries."""

    source: LatticeHypothesis
    start: int = 0
    stop: int | None = None

    @property
    def end(self) -> int:
        return self.source.cell_count if self.stop is None else self.stop

    @property
    def count(self) -> int:
        return self.end - self.start

    @property
    def pitch(self) -> float:
        return self.source.pitch

    @property
    def origin(self) -> float:
        return self.source.origin + self.start * self.pitch

    @property
    def edges(self) -> FloatArray:
        return self.origin + self.pitch * np.arange(self.count + 1)

    @property
    def centers(self) -> FloatArray:
        return self.origin + self.pitch * (np.arange(self.count) + 0.5)

    @property
    def edited(self) -> bool:
        return self.start != 0 or self.end != self.source.cell_count


@dataclass(frozen=True, eq=False)
class GeometryDiagnostics:
    axis_score: float
    center_score: float | None
    boundary_score: float | None
    center_energy_explained: float | None
    boundary_energy_inside: float | None
    boundary_energy_aligned: float | None
    vertical_continuity: float | None
    horizontal_continuity: float | None
    terminal_contrast: float
    prior_agreement: float | None
    cell_support: FloatArray
    vertical_support: FloatArray
    horizontal_support: FloatArray
    row_support: FloatArray
    column_support: FloatArray


@dataclass(frozen=True, eq=False)
class GeometryHypothesis:
    x: AxisSpan
    y: AxisSpan
    score: float
    diagnostics: GeometryDiagnostics
    # left, top, right, bottom
    outer_edges: tuple[EdgeObservation | None, ...] = (None, None, None, None)

    @property
    def columns(self) -> int:
        return self.x.count

    @property
    def rows(self) -> int:
        return self.y.count

    @property
    def x_edges(self) -> FloatArray:
        return self.x.edges

    @property
    def y_edges(self) -> FloatArray:
        return self.y.edges

    @property
    def centers(self) -> FloatArray:
        x, y = np.meshgrid(self.x.centers, self.y.centers)
        return np.stack((x, y), axis=-1)

    @property
    def visible_extent(self) -> tuple[float | None, ...]:
        return tuple(edge.position if edge is not None and not edge.censored else None for edge in self.outer_edges)


@dataclass(frozen=True, eq=False)
class GeometryDecode:
    status: Literal["fitted", "ambiguous", "insufficient_evidence"]
    hypotheses: tuple[GeometryHypothesis, ...]
    independent: IndependentFits
    evaluated: int


def _heatmap_array(heatmaps: ArrayLike) -> FloatArray:
    values = probability_array(heatmaps)
    if values.ndim != 3 or values.shape[0] != 3 or min(values.shape[1:]) == 0:
        raise ValueError("heatmaps must have shape [3, H, W] with positive H and W")
    return values


def fit_independent(heatmaps: ArrayLike, *, pitch_budget: int = 8) -> IndependentFits:
    """Run Phase 3B on the mean-projected heatmaps."""
    if not isinstance(pitch_budget, Integral) or isinstance(pitch_budget, bool) or not 1 <= pitch_budget <= 16:
        raise ValueError("pitch_budget must be an integer in [1, 16]")
    signals = project_heatmaps(heatmaps)

    def fit_axis_with_budget(axis):
        analysis = estimate_pitch(axis, max_candidates=pitch_budget)
        return fit_axis(axis, analysis, max_pitch_hypotheses=pitch_budget)

    return IndependentFits(fit_axis_with_budget(signals.x), fit_axis_with_budget(signals.y))


@dataclass(frozen=True, eq=False)
class _Field:
    raw: FloatArray
    signal: FloatArray
    integral: FloatArray

    @property
    def energy(self) -> float:
        return float(self.integral[-1, -1])


def _prepare_field(values: FloatArray) -> _Field:
    low, middle, high, strong = np.quantile(values, (0.35, 0.50, 0.65, 0.995))
    # Remove a robust background and retain weak but coherent responses.
    floor = min(middle + max(0.035, 1.9 * (high - low)), (middle + strong) / 2)
    floor = max(0.035, floor)
    positive = np.maximum(values - floor, 0)
    signal = np.minimum(positive / max(0.16, 0.77 * (strong - floor)), 1)
    energy = np.pad(positive**2, ((1, 0), (1, 0)))
    integral = energy.cumsum(axis=0).cumsum(axis=1)
    return _Field(values, signal, integral)


def _rect_energy(
    field: _Field, left: ArrayLike, top: ArrayLike, right: ArrayLike, bottom: ArrayLike,
) -> FloatArray:
    height, width = field.raw.shape

    def limits(values, length):
        return np.ceil(np.asarray(values) - 0.5).clip(0, length).astype(int)

    x0, x1 = limits(left, width), limits(right, width)
    y0, y1 = limits(top, height), limits(bottom, height)
    integral = field.integral
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def _cell_samples(field: _Field, x: FloatArray, y: FloatArray, px: float, py: float) -> FloatArray:
    x_offsets = np.array([-1, 0, 1]) * min(1.35, 0.18 * px)
    y_offsets = np.array([-1, 0, 1]) * min(1.35, 0.18 * py)
    columns = (x[:, None] + x_offsets).ravel()
    rows = (y[:, None] + y_offsets).ravel()
    return _sample(field.signal, columns, rows).reshape(y.size, 3, x.size, 3).max(axis=(1, 3))


def _line_samples(
    field: FloatArray, edges: FloatArray, segments: FloatArray, across_pitch: float, along_pitch: float,
) -> FloatArray:
    """Sample line contrast across each ridge and at three locations per segment."""
    offsets = np.array([-1, 0, 1]) * min(1.35, 0.16 * across_pitch)
    columns = (edges[:, None] + offsets).ravel()
    rows = (segments[:, None] + along_pitch * np.array([-0.30, 0, 0.30])).ravel()
    ridge = _sample(field, columns, rows).reshape(segments.size, 3, edges.size, 3).max(axis=-1)
    side_offsets = np.array([-0.35, 0.35]) * across_pitch
    sides = (edges[:, None] + side_offsets).ravel()
    off_ridge = _sample(field, sides, rows).reshape(segments.size, 3, edges.size, 2).mean(axis=-1)
    return np.maximum(ridge - 0.68 * off_ridge, 0)


def _balance(observed: float, expected: float) -> float:
    return 2 * observed * expected / (observed + expected) if observed + expected > 0 else 0.0


def _segment_support(values: FloatArray) -> FloatArray:
    # Several coherent lines can establish extent even when other ridges are missing.
    keep = max(1, math.ceil(values.shape[1] / 2))
    strongest = np.sort(values, axis=1)[:, -keep:]
    return 0.30 * values.mean(axis=1) + 0.70 * strongest.mean(axis=1)


def _score(
    x: AxisSpan, y: AxisSpan, fields: tuple[_Field, _Field, _Field],
    use_centers: bool, use_boundaries: bool, ratio_prior: PitchRatioPrior | None,
) -> GeometryHypothesis:
    center_field, vertical_field, horizontal_field = fields
    xe, ye = x.edges, y.edges
    cx = x.origin + (np.arange(-1, x.count + 1) + 0.5) * x.pitch
    cy = y.origin + (np.arange(-1, y.count + 1) + 0.5) * y.pitch
    centers = _cell_samples(center_field, cx, cy, x.pitch, y.pitch)
    cells = centers[1:-1, 1:-1]
    vertical_samples = _line_samples(vertical_field.signal, xe[1:-1], cy, x.pitch, y.pitch)
    horizontal_samples = _line_samples(horizontal_field.signal.T, ye[1:-1], cx, y.pitch, x.pitch)
    vertical = vertical_samples[1:-1].mean(axis=1)
    horizontal = horizontal_samples[1:-1].mean(axis=1).T
    row_centers = _segment_support(centers[:, 1:-1])
    column_centers = _segment_support(centers[1:-1, :].T)
    row_boundaries = _segment_support(vertical_samples.mean(axis=1))
    column_boundaries = _segment_support(horizontal_samples.mean(axis=1))
    rows = np.maximum(row_centers * use_centers, row_boundaries * use_boundaries)
    columns = np.maximum(column_centers * use_centers, column_boundaries * use_boundaries)
    terminal = np.array([rows[1], rows[-2], columns[1], columns[-2]])
    beyond = np.array([rows[0], rows[-1], columns[0], columns[-1]])
    terminal_contrast = float((0.4 * terminal + 0.6 * np.maximum(terminal - beyond, 0)).mean())

    center_explained = center_score = None
    if center_field.energy > 0:
        radius_x = min(3.0, 0.32 * x.pitch)
        radius_y = min(3.0, 0.32 * y.pitch)
        x0, x1 = cx[1:-1] - radius_x, cx[1:-1] + radius_x
        y0, y1 = cy[1:-1, None] - radius_y, cy[1:-1, None] + radius_y
        inside = float(_rect_energy(center_field, x0, y0, x1, y1).sum())
        center_explained = float(np.clip(inside / center_field.energy, 0, 1))
        center_score = _balance(center_explained, float(cells.mean()))

    edge_scores = []
    coverage = []
    alignment = []
    continuity: list[float | None] = []
    for field, samples, across, pitch, transpose in (
        (vertical_field, vertical_samples, xe, x.pitch, False),
        (horizontal_field, horizontal_samples, ye, y.pitch, True),
    ):
        if field.energy <= 0:
            continuity.append(None)
            continue
        inside = float(_rect_energy(field, xe[0] - 1.75, ye[0] - 1.75, xe[-1] + 1.75, ye[-1] + 1.75))
        inside_fraction = float(np.clip(inside / field.energy, 0, 1))
        radius = min(3.6, 0.36 * pitch)
        if transpose:
            near_lines = _rect_energy(field, xe[0] - 1.75, across - radius, xe[-1] + 1.75, across + radius)
        else:
            near_lines = _rect_energy(field, across - radius, ye[0] - 1.75, across + radius, ye[-1] + 1.75)
        aligned_fraction = float(np.clip(near_lines.sum() / field.energy, 0, 1))
        observed = 0.18 * inside_fraction + 0.82 * aligned_fraction
        covered = float(np.mean(samples[1:-1] >= 0.28))
        expected = 0.72 * float(samples[1:-1].mean()) + 0.28 * covered
        coverage.append(inside_fraction)
        alignment.append(aligned_fraction)
        continuity.append(covered)
        # Explain the visible ridges as well as supporting predicted lines.
        # Favor the observed direction because true lines can be missing.
        edge_scores.append(
            observed * expected / (0.70 * expected + 0.30 * observed) if observed + expected > 0 else 0.0
        )
    boundary_score = float(np.mean(edge_scores)) if edge_scores else None
    boundary_energy_inside = float(np.mean(coverage)) if coverage else None
    boundary_energy_aligned = float(np.mean(alignment)) if alignment else None

    axis_score = (x.source.score + y.source.score) / 2
    evidence = [(0.18, axis_score), (0.18, terminal_contrast)]
    if use_centers and center_score is not None:
        evidence.append((0.25, center_score))
    if use_boundaries and boundary_score is not None:
        evidence.append((0.39, boundary_score))
    score = sum(weight * value for weight, value in evidence) / sum(weight for weight, _ in evidence)
    prior_agreement = None
    if ratio_prior is not None and ratio_prior.weight > 0:
        log_error = (math.log(x.pitch / y.pitch) - math.log(ratio_prior.ratio)) / ratio_prior.log_sigma
        prior_agreement = math.exp(-0.5 * log_error**2)
        score = (1 - ratio_prior.weight) * score + ratio_prior.weight * prior_agreement
    diagnostics = GeometryDiagnostics(
        axis_score, center_score, boundary_score, center_explained, boundary_energy_inside, boundary_energy_aligned,
        continuity[0], continuity[1], terminal_contrast, prior_agreement,
        cells, vertical, horizontal, rows[1:-1], columns[1:-1],
    )
    return GeometryHypothesis(x, y, score, diagnostics)


def _span_key(span: AxisSpan) -> tuple[float, float, float, int, int, int]:
    # The source axis contributes to the score, even if two edited spans happen
    # to have the same geometry. An edit is relative to its own source.
    source = span.source
    return source.origin, source.pitch, source.score, source.cell_count, span.start, span.end


def _edits(span: AxisSpan, length: int):
    # At most two cells from the original start/stop on either side.
    for start_step, end_step in ((-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0),
                                 (0, -2), (0, 2), (-1, -1), (1, 1)):
        start, end = span.start + start_step, span.end + end_step
        if not (-2 <= start <= 2 and abs(end - span.source.cell_count) <= 2 and 2 <= end - start <= 128):
            continue
        variant = AxisSpan(span.source, start, end)
        if 0 <= variant.origin + 0.5 * variant.pitch and variant.edges[-1] - 0.5 * variant.pitch <= length:
            yield variant


def _permitted_edit(
    parent: GeometryHypothesis, candidate: GeometryHypothesis, axis: Literal["x", "y"],
    use_centers: bool, use_boundaries: bool,
) -> bool:
    previous, new = getattr(parent, axis), getattr(candidate, axis)
    old_info, new_info = parent.diagnostics, candidate.diagnostics
    if axis == "x":
        old_strip, new_strip = old_info.column_support, new_info.column_support
        center_sites, crossing_lines = new_info.cell_support.T, new_info.horizontal_support.T
    else:
        old_strip, new_strip = old_info.row_support, new_info.row_support
        center_sites, crossing_lines = new_info.cell_support, new_info.vertical_support
    old_positions = np.arange(previous.start, previous.end)
    removed = (old_positions < new.start) | (old_positions >= new.end)
    relative_floor = max(0.19, 0.46 * float(np.median(old_strip)))
    if np.any(old_strip[removed] > relative_floor):
        return False

    new_positions = np.arange(new.start, new.end)
    added = (new_positions < previous.start) | (new_positions >= previous.end)
    corroboration = (
        (center_sites[added] >= 0.28).sum(axis=1) * use_centers
        + (crossing_lines[added] >= 0.28).sum(axis=1) * use_boundaries
    )
    return bool(np.all(new_strip[added] >= 0.20) and np.all(corroboration >= 2))


def _outer_edge(
    field: _Field, ideal: float, inward: int, across_pitch: float, perpendicular: AxisSpan,
    *, transpose: bool = False,
) -> EdgeObservation | None:
    raw = field.raw.T if transpose else field.raw
    signal = field.signal.T if transpose else field.signal
    length = raw.shape[1]
    inside = min(0.27 * across_pitch, max(1.6, 0.15 * across_pitch))
    outside = min(0.21 * across_pitch, max(1.2, 0.075 * across_pitch))
    pixels = np.arange(length) + 0.5
    delta = inward * (pixels - ideal)
    options = np.flatnonzero((delta >= -outside) & (delta <= inside))
    if not options.size:
        return None
    first, last = max(0, options[0] - 1), min(length - 1, options[-1] + 1)
    sites = np.arange(first, last + 1)
    perpendicular_samples = (perpendicular.centers[:, None] + perpendicular.pitch * np.array([-0.25, 0, 0.25])).ravel()
    coordinates = pixels[sites]
    along = _sample(signal, coordinates, perpendicular_samples)
    averaged = along.mean(axis=0)
    raw_profile = _sample(raw, coordinates, perpendicular_samples).mean(axis=0)
    local = options - first
    index = int(local[np.argmax(raw_profile[local])])
    position = float(coordinates[index])
    global_index = int(sites[index])
    censored = global_index in (0, length - 1)
    if not censored:
        if not 0 < index < raw_profile.size - 1 or raw_profile[index] < max(raw_profile[index - 1], raw_profile[index + 1]):
            return None
        previous, peak, following = raw_profile[index - 1:index + 2]
        curvature = previous + following - 2 * peak
        if curvature < -1e-9:
            position += float(np.clip(0.5 * (previous - following) / curvature, -0.5, 0.5))
    off = min(0.33 * across_pitch, 3.0)
    side_positions = np.array([position - off, position + off])
    shoulder = _sample(signal, side_positions, perpendicular_samples).mean()
    if averaged[index] < 0.12 or averaged[index] - shoulder < 0.055 or np.mean(along[:, index] >= 0.20) < 0.30:
        return None
    strength = float(_sample(raw, np.array([position]), perpendicular_samples).mean())
    return EdgeObservation(position, strength, censored)


def _describe_edges(candidate: GeometryHypothesis, fields: tuple[_Field, _Field, _Field]) -> GeometryHypothesis:
    _, vertical, horizontal = fields
    x, y = candidate.x, candidate.y
    return replace(candidate, outer_edges=(
        _outer_edge(vertical, x.origin, 1, x.pitch, y),
        _outer_edge(horizontal, y.origin, 1, y.pitch, x, transpose=True),
        _outer_edge(vertical, x.edges[-1], -1, x.pitch, y),
        _outer_edge(horizontal, y.edges[-1], -1, y.pitch, x, transpose=True),
    ))


def _same_grid(a: GeometryHypothesis, b: GeometryHypothesis) -> bool:
    return (
        a.rows == b.rows and a.columns == b.columns
        and max(abs(a.x.origin - b.x.origin), abs(a.y.origin - b.y.origin),
                abs(a.x_edges[-1] - b.x_edges[-1]), abs(a.y_edges[-1] - b.y_edges[-1])) < 0.85
    )


def _harmonic_alternative(best: GeometryHypothesis, candidate: GeometryHypothesis) -> bool:
    """Find an otherwise equal board with a finer explanation of one boundary train."""
    for a, b, other_a, other_b in (
        (best.x, candidate.x, best.y, candidate.y), (best.y, candidate.y, best.x, candidate.x),
    ):
        ratio = a.pitch / b.pitch
        if (
            ratio >= 1.85 and abs(ratio - round(ratio)) <= 0.08
            and abs(a.origin - b.origin) < 0.20 * a.pitch
            and abs(a.edges[-1] - b.edges[-1]) < 0.20 * a.pitch
            and other_a.count == other_b.count
            and abs(other_a.origin - other_b.origin) < 0.85
            and abs(other_a.edges[-1] - other_b.edges[-1]) < 0.85
        ):
            return True
    return False


def decode_geometry(
    heatmaps: ArrayLike, *, independent: IndependentFits | None = None, pitch_budget: int = 8,
    use_centers: bool = True, use_boundaries: bool = True, refine_extent: bool = True,
    ratio_prior: PitchRatioPrior | None = None, max_hypotheses: int = 8,
) -> GeometryDecode:
    """Decode axis-aligned boards from [centers, vertical, horizontal] probabilities.

    Pass independent fits from these same maps to reuse the Phase 3B step in
    ablations. Scores rank hypotheses. An explicit ratio prior is optional.
    """
    values = _heatmap_array(heatmaps)
    if not use_centers and not use_boundaries:
        raise ValueError("enable center or boundary evidence")
    if not isinstance(max_hypotheses, Integral) or isinstance(max_hypotheses, bool) or not 1 <= max_hypotheses <= 16:
        raise ValueError("max_hypotheses must be an integer in [1, 16]")
    if independent is None:
        independent = fit_independent(values, pitch_budget=pitch_budget)
    if not independent.x.hypotheses or not independent.y.hypotheses:
        return GeometryDecode("insufficient_evidence", (), independent, 0)
    fields = tuple(_prepare_field(channel) for channel in values)
    cache = {}
    allowed = {}

    def score_pair(x, y):
        key = (_span_key(x), _span_key(y))
        if key not in cache:
            cache[key] = _score(x, y, fields, use_centers, use_boundaries, ratio_prior)
        return key, cache[key]

    for x in independent.x.hypotheses[:8]:
        for y in independent.y.hypotheses[:8]:
            key, candidate = score_pair(AxisSpan(x), AxisSpan(y))
            allowed[key] = candidate
    if refine_extent:
        height, width = values.shape[1:]
        for _ in range(2):
            leaders = sorted(allowed.values(), key=lambda h: -h.score)[:3]
            for leader in leaders:
                for axis, span, length in (("x", leader.x, width), ("y", leader.y, height)):
                    for edit in _edits(span, length):
                        if len(cache) >= 160:
                            break
                        axes = (edit, leader.y) if axis == "x" else (leader.x, edit)
                        key, candidate = score_pair(*axes)
                        if key not in allowed and _permitted_edit(leader, candidate, axis, use_centers, use_boundaries):
                            allowed[key] = candidate

    ordered = sorted(allowed.values(), key=lambda h: -h.score)
    ranked: list[GeometryHypothesis] = []
    for candidate in ordered:
        if not any(_same_grid(candidate, existing) for existing in ranked):
            ranked.append(candidate)
        if len(ranked) == max(2, max_hypotheses):
            break

    best = ranked[0]
    center_score = best.diagnostics.center_score if use_centers else None
    boundary_score = best.diagnostics.boundary_score if use_boundaries else None
    if max(center_score or 0, boundary_score or 0) < 0.10:
        return GeometryDecode("insufficient_evidence", (), independent, len(cache))
    boundary_only_alias = (best.diagnostics.center_score or 0) < 0.12 and (
        independent.x.status == "ambiguous" or independent.y.status == "ambiguous"
    )
    if boundary_only_alias and max_hypotheses > 1:
        # Preserve a coherent alternative for the same observed extent. It
        # may rank behind unrelated coarser pairs on expected-slot coverage.
        if not any(_harmonic_alternative(best, candidate) for candidate in ranked[1:]):
            alternative = next((candidate for candidate in ordered if _harmonic_alternative(best, candidate)), None)
            if alternative is not None:
                ranked = ranked[:max_hypotheses - 1] + [alternative]
    described = tuple(_describe_edges(candidate, fields) for candidate in ranked[:max_hypotheses])
    best = described[0]
    near_tie = len(ranked) > 1 and best.score - ranked[1].score < 0.022
    # With a missing center channel, a repeated boundary pattern can be a
    # coarser lattice or alternating loss. The boundary score alone cannot decide.
    unknown_end = any(
        edge is None and 0 <= another <= length
        for edge, another, length in zip(best.outer_edges, (
            best.x.origin - best.x.pitch / 2, best.y.origin - best.y.pitch / 2,
            best.x_edges[-1] + best.x.pitch / 2, best.y_edges[-1] + best.y.pitch / 2,
        ), (values.shape[2], values.shape[1], values.shape[2], values.shape[1]), strict=True)
    )
    weak = (center_score or 0) < 0.28 and (boundary_score or 0) < 0.34
    status = "ambiguous" if near_tie or boundary_only_alias or unknown_end or weak else "fitted"
    return GeometryDecode(status, described, independent, len(cache))
