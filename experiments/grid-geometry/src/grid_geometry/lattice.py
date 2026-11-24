"""Fit finite one-dimensional lattices to center and boundary evidence."""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .pitch import ChannelEvidence, PitchAnalysis, estimate_pitch, phase_alignment
from .signals import AxisSignals, FloatArray, probability_array


@dataclass(frozen=True)
class EdgeObservation:
    """A boundary peak near an ideal endpoint, in canvas coordinates."""

    position: float
    strength: float
    censored: bool


@dataclass(frozen=True)
class LatticeDiagnostics:
    observed_agreement: float
    expected_support: float
    center_support: float | None
    internal_boundary_support: float | None
    extent_support: float
    outside_continuation: float
    phase_agreement: float
    matched_strength: float
    unexplained_peak_weight: float
    localization_rmse: float
    center_matches: tuple[float | None, ...]
    internal_boundary_matches: tuple[float | None, ...]
    unexplained_centers: tuple[float, ...]
    unexplained_boundaries: tuple[float, ...]


@dataclass(frozen=True)
class LatticeHypothesis:
    pitch: float
    origin: float
    cell_count: int
    initial_pitch: float
    initial_rank: int
    score: float
    diagnostics: LatticeDiagnostics
    left_edge: EdgeObservation | None
    right_edge: EdgeObservation | None

    @property
    def ideal_extent(self) -> tuple[float, float]:
        return self.origin, self.origin + self.cell_count * self.pitch

    @property
    def visible_extent(self) -> tuple[float | None, float | None]:
        """Visible endpoints with uncensored evidence. Unknown sides are None."""
        left = self.left_edge
        right = self.right_edge
        return (
            left.position if left is not None and not left.censored else None,
            right.position if right is not None and not right.censored else None,
        )

    @property
    def centers(self) -> FloatArray:
        return self.origin + (np.arange(self.cell_count) + 0.5) * self.pitch

    @property
    def boundaries(self) -> FloatArray:
        return self.origin + np.arange(self.cell_count + 1) * self.pitch


@dataclass(frozen=True)
class AxisLatticeFit:
    status: Literal["fitted", "ambiguous", "insufficient_evidence"]
    hypotheses: tuple[LatticeHypothesis, ...]


@dataclass(frozen=True)
class PairedLattice:
    x: LatticeHypothesis
    y: LatticeHypothesis
    independent_score: float
    prior_agreement: float
    score: float


@dataclass(frozen=True)
class _Peaks:
    positions: FloatArray
    heights: FloatArray
    weights: FloatArray
    censored: NDArray[np.bool_]


@dataclass(frozen=True)
class _Matches:
    indices: NDArray[np.intp]
    quality: FloatArray
    explained: FloatArray


def _axis_peaks(signal: FloatArray, evidence: ChannelEvidence) -> _Peaks:
    positions = list(evidence.peak_positions)
    heights = list(evidence.peak_heights)
    prominences = list(evidence.peak_prominences)
    censored = [False] * len(positions)
    # Phase 3A's two-sided peak detector omits endpoints. Add one-sided
    # evidence for span scoring, with its location marked as censored.
    if signal.size >= 3:
        radius = min(signal.size, 6)
        threshold = 0.075 * np.ptp(signal)
        for side, edge in enumerate((signal[:radius], signal[::-1][:radius])):
            peak = int(np.argmax(edge[:min(3, radius - 1)]))
            flank = edge[peak:]
            drop = flank[0] - flank.min()
            if drop <= threshold or np.any(np.diff(flank[:np.argmin(flank) + 1]) > 0):
                continue
            position = peak + 0.5
            if peak > 0:
                left = edge[peak] - edge[peak - 1]
                right = edge[peak + 1] - edge[peak]
                if right >= 0 or left <= 0:
                    continue
                position += 0.5 * (left + right) / (left - right)
            if side == 1:
                position = signal.size - position
            if any(abs(existing - position) <= 1.0 for existing in positions):
                continue
            positions.append(position)
            heights.append(float(edge[peak]))
            prominences.append(float(drop))
            censored.append(peak == 0)

    strengths = np.asarray(prominences)
    weights = np.sqrt(strengths / strengths.max()) if strengths.size else strengths
    return _Peaks(np.asarray(positions), np.asarray(heights), weights, np.asarray(censored, dtype=bool))


def _regular_matches(expected: FloatArray, peaks: _Peaks, pitch: float, *, loose: bool = False) -> _Matches:
    if not peaks.positions.size or not expected.size:
        return _Matches(np.full(expected.size, -1), np.zeros(expected.size), np.zeros(expected.size))
    scale = min(1.65, 0.33 + 0.055 * pitch)
    tolerance = min(0.28 * pitch, max(1.3, 2.3 * scale))
    if loose:
        tolerance = min(0.33 * pitch, max(tolerance, 0.14 * pitch + 0.7))
        scale = max(scale, tolerance / 1.8)
    distance = np.abs(expected[:, None] - peaks.positions)
    localization = np.exp(-0.5 * (distance / scale)**2)
    strength = 0.55 + 0.45 * peaks.weights
    support = localization * strength
    support[distance > tolerance] = 0
    indices = np.argmax(support, axis=1)
    quality = np.take_along_axis(support, indices[:, None], axis=1)[:, 0]
    agreement = np.take_along_axis(localization, indices[:, None], axis=1)[:, 0]
    explained = np.where(quality > 0, agreement * peaks.weights[indices], 0)
    return _Matches(np.where(quality > 0, indices, -1), quality, explained)


def _edge_matches(
    expected: FloatArray, peaks: _Peaks, pitch: float, *, inward_sign: int,
) -> _Matches:
    if not peaks.positions.size:
        zeros = np.zeros(expected.size)
        return _Matches(np.full(expected.size, -1), zeros, zeros.copy())

    shift = (peaks.positions - expected[:, None]) * inward_sign
    # The near side of a gap is inward, while localization noise can have
    # either sign. Keep the window well clear of an adjacent internal peak.
    inside = min(0.28 * pitch, max(1.35, 0.15 * pitch))
    outside = min(0.18 * pitch, max(0.9, 0.06 * pitch))
    location = 1 - 0.35 * np.abs(shift) / np.where(shift >= 0, inside, outside)
    location[(shift < -outside) | (shift > inside)] = 0
    strength = 0.25 + 0.75 * peaks.weights
    support = location * strength
    indices = np.argmax(support, axis=1)
    quality = np.take_along_axis(support, indices[:, None], axis=1)[:, 0]
    explained = np.where(quality > 0, peaks.weights[indices], 0)
    return _Matches(np.where(quality > 0, indices, -1), quality, explained)


def _phase_origins(centers: _Peaks, boundaries: _Peaks, pitch: float) -> list[float]:
    positions = np.concatenate((centers.positions - pitch / 2, boundaries.positions))
    weights = np.concatenate((centers.weights, boundaries.weights))
    phases = positions % pitch
    distance = (phases[:, None] - phases + pitch / 2) % pitch - pitch / 2
    scale = min(0.18 * pitch, max(0.8, 0.09 * pitch))
    kernel = np.exp(-0.5 * (distance / scale)**2)
    support = kernel @ weights
    origins: list[float] = []
    for index in np.argsort(-support, kind="stable"):
        move = np.average(distance[index], weights=kernel[index] * weights)
        origin = float((phases[index] - move) % pitch)
        if all(abs((origin - other + pitch / 2) % pitch - pitch / 2) > 0.19 * pitch for other in origins):
            origins.append(origin)
        if len(origins) == 2:
            break
    return origins


def _center_index_range(origin: float, pitch: float, length: int) -> NDArray[np.intp]:
    first = math.ceil(-origin / pitch - 0.5)
    last = math.floor((length - origin) / pitch - 0.5)
    return np.arange(first, last + 1)


def _refine(
    origin: float, pitch: float, center_indices: NDArray[np.intp], centers: _Peaks, boundaries: _Peaks,
    bounds: tuple[float, float], *, finite: bool,
) -> tuple[float, float, float]:
    """Alternate correspondence assignment and robust weighted line fitting."""
    internal = center_indices[1:]
    strength = 0.0
    for _ in range(3):
        center_matches = _regular_matches(origin + (center_indices + 0.5) * pitch, centers, pitch, loose=True)
        boundary_matches = _regular_matches(origin + internal * pitch, boundaries, pitch, loose=True)
        # In the provisional fit, set aside the extreme candidate boundaries.
        # In the finite fit, internal indices already exclude both endpoints.
        keep_boundaries = boundary_matches.indices >= 0
        if not finite:
            matched_centers = np.flatnonzero(center_matches.indices >= 0)
            if matched_centers.size >= 2:
                keep_boundaries &= (internal > center_indices[matched_centers[0]])
                keep_boundaries &= (internal <= center_indices[matched_centers[-1]])
            else:
                matched_boundaries = np.flatnonzero(keep_boundaries)
                if matched_boundaries.size >= 4:
                    keep_boundaries[matched_boundaries[[0, -1]]] = False

        design, observed, weights = [], [], []
        for indices, peaks, matches, keep in (
            (center_indices + 0.5, centers, center_matches, center_matches.indices >= 0),
            (internal, boundaries, boundary_matches, keep_boundaries),
        ):
            valid = np.flatnonzero(keep)
            picked = matches.indices[valid]
            uncensored = ~peaks.censored[picked]
            design.extend(indices[valid[uncensored]])
            observed.extend(peaks.positions[picked[uncensored]])
            weights.extend(matches.explained[valid[uncensored]])
        if len(design) < 2 or np.ptp(design) < 1:
            break

        x, y, weight = np.asarray(design), np.asarray(observed), np.asarray(weights)
        mean_x, mean_y = np.average(x, weights=weight), np.average(y, weights=weight)
        dx = x - mean_x
        slope = np.sum(weight * dx * (y - mean_y)) / np.sum(weight * dx**2)
        revised_pitch = float(np.clip(slope, *bounds))
        revised_origin = float(mean_y - mean_x * revised_pitch)
        strength = float(weight.sum())
        settled = abs(revised_pitch - pitch) < 1e-4 and abs(revised_origin - origin) < 1e-3
        origin, pitch = revised_origin, revised_pitch
        if settled:
            break
    return origin, pitch, strength


def _cumulative(values: FloatArray | NDArray[np.bool_]) -> FloatArray:
    return np.concatenate(([0.0], np.cumsum(values)))


@dataclass(frozen=True)
class _GridMatches:
    centers: _Matches
    internals: _Matches
    left: _Matches
    right: _Matches


def _grid_matches(
    origin: float, pitch: float, indices: NDArray[np.intp], centers: _Peaks, boundaries: _Peaks,
) -> _GridMatches:
    left = origin + pitch * indices
    return _GridMatches(
        _regular_matches(left + 0.5 * pitch, centers, pitch),
        _regular_matches(left, boundaries, pitch),
        _edge_matches(left, boundaries, pitch, inward_sign=1),
        _edge_matches(left + pitch, boundaries, pitch, inward_sign=-1),
    )


def _span_scores(
    starts: NDArray[np.intp], ends: NDArray[np.intp], grid: _GridMatches,
    centers: _Peaks, boundaries: _Peaks,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    count = ends - starts + 1
    center_quality = _cumulative(grid.centers.quality)
    internal_quality = _cumulative(grid.internals.quality)
    center_support = (center_quality[ends + 1] - center_quality[starts]) / count
    internal_support = (internal_quality[ends + 1] - internal_quality[starts + 1]) / (count - 1)
    available = int(centers.weights.size > 0) + int(boundaries.weights.size > 0)
    expected = (center_support + internal_support) / available

    center_weights = _cumulative(grid.centers.explained)
    boundary_weights = _cumulative(grid.internals.explained)
    explained_centers = center_weights[ends + 1] - center_weights[starts]
    explained_boundaries = boundary_weights[ends + 1] - boundary_weights[starts + 1]
    explained_boundaries += grid.left.explained[starts] + grid.right.explained[ends]
    observed = (explained_centers + explained_boundaries) / (centers.weights.sum() + boundaries.weights.sum())
    left = np.maximum(grid.left.quality[starts], 0.92 * grid.centers.quality[starts])
    right = np.maximum(grid.right.quality[ends], 0.92 * grid.centers.quality[ends])
    # A matched center and another boundary just outside the span are evidence
    # for a neighboring cell. A lone outlying peak has no continuation vote.
    prev = np.clip(starts - 1, 0, None)
    next_ = np.minimum(ends + 1, grid.centers.quality.size - 1)
    before = (starts > 0) * np.sqrt(grid.centers.quality[prev] * grid.left.quality[prev])
    after = (ends < grid.centers.quality.size - 1) * np.sqrt(grid.centers.quality[next_] * grid.right.quality[next_])
    continuation = np.maximum(before, after)
    return observed.clip(0, 1), expected, (left + right) / 2, center_support, internal_support, continuation


def _span_score(
    observed: FloatArray, expected: FloatArray, extent: FloatArray, continuation: FloatArray,
) -> FloatArray:
    total = observed + expected
    balance = np.divide(2 * observed * expected, total, out=np.zeros_like(total), where=total > 0)
    return 0.82 * balance + 0.085 * extent + 0.045 * observed - 0.075 * continuation


def _rank_spans(
    grid: _GridMatches, centers: _Peaks, boundaries: _Peaks, max_cells: int,
) -> list[tuple[int, int]]:
    # Every span competes against all observed evidence, including peaks
    # outside the span. Normalize expected support by its own finite count.
    size = grid.centers.indices.size
    if size < 2:
        return []
    widths = np.arange(1, min(size, max_cells))
    starts = np.repeat(np.arange(size - 1), widths.size)
    ends = starts + np.tile(widths, size - 1)
    allowed = ends < size
    starts, ends = starts[allowed], ends[allowed]
    if not starts.size:
        return []
    center_hits = _cumulative(grid.centers.indices >= 0)
    internal_hits = _cumulative(grid.internals.indices >= 0)
    matches = center_hits[ends + 1] - center_hits[starts]
    matches += internal_hits[ends + 1] - internal_hits[starts + 1]
    matches += grid.left.indices[starts] >= 0
    matches += grid.right.indices[ends] >= 0
    starts, ends = starts[matches >= 3], ends[matches >= 3]
    if not starts.size:
        return []
    observed, expected, extent, _, _, continuation = _span_scores(starts, ends, grid, centers, boundaries)
    scores = _span_score(observed, expected, extent, continuation)
    top = np.argsort(-scores, kind="stable")[:3]
    return list(zip(starts[top].tolist(), ends[top].tolist(), strict=True))


def _edge_observation(peaks: _Peaks, index: int) -> EdgeObservation | None:
    if index < 0:
        return None
    return EdgeObservation(float(peaks.positions[index]), float(peaks.heights[index]), bool(peaks.censored[index]))


def _describe(
    origin: float, pitch: float, count: int, initial_pitch: float, initial_rank: int,
    centers: _Peaks, boundaries: _Peaks, phase: float,
) -> LatticeHypothesis | None:
    grid = _grid_matches(origin, pitch, np.arange(-1, count + 1), centers, boundaries)
    left, right = grid.left.indices[1], grid.right.indices[count]
    used_center_indices = grid.centers.indices[1:count + 1]
    used_internal_indices = grid.internals.indices[2:count + 1]
    actual_centers = used_center_indices[used_center_indices >= 0]
    actual_boundaries = np.r_[left, used_internal_indices, right]
    actual_boundaries = actual_boundaries[actual_boundaries >= 0]
    if actual_centers.size + actual_boundaries.size < 3:
        return None
    observed, expected, extent, center_support, internal_support, continuation = _span_scores(
        np.array([1]), np.array([count]), grid, centers, boundaries,
    )
    strengths = np.r_[centers.heights[actual_centers], boundaries.heights[actual_boundaries]]
    raw = float(strengths.mean())
    total_weight = centers.weights.sum() + boundaries.weights.sum()
    explained_weight = centers.weights[actual_centers].sum() + boundaries.weights[actual_boundaries].sum()
    unexplained_weight = float(np.clip(1 - explained_weight / total_weight, 0, 1))
    base_score = _span_score(observed, expected, extent, continuation)
    score = float(base_score[0] + 0.035 * phase + 0.015 * raw)

    def positions(peaks: _Peaks, matches: NDArray[np.intp]) -> tuple[float | None, ...]:
        return tuple(float(peaks.positions[index]) if index >= 0 else None for index in matches)

    def unexplained(peaks: _Peaks, used: NDArray[np.intp]) -> tuple[float, ...]:
        unused = np.setdiff1d(np.arange(peaks.positions.size), used)
        return tuple(map(float, peaks.positions[unused]))

    center_used = used_center_indices >= 0
    boundary_used = used_internal_indices >= 0
    residuals = np.r_[
        centers.positions[actual_centers] - (origin + (np.flatnonzero(center_used) + 0.5) * pitch),
        boundaries.positions[used_internal_indices[boundary_used]] - (origin + (np.flatnonzero(boundary_used) + 1) * pitch),
    ]
    rmse = float(np.sqrt(np.mean(residuals**2))) if residuals.size else 0.0
    diagnostics = LatticeDiagnostics(
        float(observed[0]), float(expected[0]),
        float(center_support[0]) if centers.weights.size else None,
        float(internal_support[0]) if boundaries.weights.size else None,
        float(extent[0]), float(continuation[0]), phase, raw, unexplained_weight, rmse,
        positions(centers, used_center_indices), positions(boundaries, used_internal_indices),
        unexplained(centers, actual_centers), unexplained(boundaries, actual_boundaries),
    )
    return LatticeHypothesis(
        float(pitch), float(origin), count, initial_pitch, initial_rank, score, diagnostics,
        _edge_observation(boundaries, left), _edge_observation(boundaries, right),
    )


def _provisional_fits(
    seed: float, length: int, centers: _Peaks, boundaries: _Peaks,
) -> tuple[list[tuple[float, float, float]], tuple[float, float]]:
    reach = min(4.0, max(1.4, 0.075 * seed))
    bounds = (max(2.0, seed - reach), min(length, seed + reach))
    provisional = []
    for pitch in np.linspace(*bounds, 7):
        for origin in _phase_origins(centers, boundaries, pitch):
            indices = _center_index_range(origin, pitch, length)
            if indices.size >= 2:
                provisional.append(_refine(origin, pitch, indices, centers, boundaries, bounds, finite=False))
    provisional.sort(key=lambda candidate: -candidate[2])
    selected: list[tuple[float, float, float]] = []
    for origin, pitch, strength in provisional:
        if any(
            abs(pitch - other_pitch) <= 0.13
            and abs((origin - other_origin + pitch / 2) % pitch - pitch / 2) <= 0.5
            for other_origin, other_pitch, _ in selected
        ):
            continue
        selected.append((origin, pitch, strength))
        if len(selected) == 3:
            break
    return selected, bounds


def fit_axis(
    signals: AxisSignals, analysis: PitchAnalysis | None = None, *,
    max_hypotheses: int = 8, max_pitch_hypotheses: int = 8, max_cells: int = 128,
) -> AxisLatticeFit:
    """Fit and rank finite spans using the supplied Phase 3A pitch hypotheses.

    `analysis` must describe the supplied signals. Axes are fit independently
    and origins describe the first ideal boundary. Scores are for ranking.
    """
    center_signal = probability_array(signals.centers)
    boundary_signal = probability_array(signals.boundaries)
    if center_signal.ndim != 1 or center_signal.shape != boundary_signal.shape or not center_signal.size:
        raise ValueError("center and boundary signals must be nonempty 1D arrays of equal length")
    for name, value, minimum in (
        ("max_hypotheses", max_hypotheses, 1),
        ("max_pitch_hypotheses", max_pitch_hypotheses, 1),
        ("max_cells", max_cells, 2),
    ):
        if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if analysis is None:
        analysis = estimate_pitch(AxisSignals(center_signal, boundary_signal))
    centers = _axis_peaks(center_signal, analysis.centers)
    boundaries = _axis_peaks(boundary_signal, analysis.boundaries)
    length = center_signal.size
    hypotheses = []
    for rank, seed in enumerate(analysis.candidates[:max_pitch_hypotheses], 1):
        if not math.isfinite(seed.pitch) or not 2 <= seed.pitch <= length:
            raise ValueError("pitch hypotheses must be finite and within the sampled axis")
        if centers.positions.size + boundaries.positions.size < 3:
            break
        templates, bounds = _provisional_fits(seed.pitch, length, centers, boundaries)
        for origin, pitch, _ in templates:
            indices = _center_index_range(origin, pitch, length)
            grid = _grid_matches(origin, pitch, indices, centers, boundaries)
            for start, end in _rank_spans(grid, centers, boundaries, max_cells):
                count = end - start + 1
                first_boundary = origin + indices[start] * pitch
                refined_origin, refined_pitch, _ = _refine(
                    first_boundary, pitch, np.arange(count), centers, boundaries, bounds, finite=True,
                )
                if not (
                    0 <= refined_origin + 0.5 * refined_pitch
                    and refined_origin + (count - 0.5) * refined_pitch <= length
                ):
                    continue
                phase = phase_alignment(analysis.centers, analysis.boundaries, refined_pitch).score
                candidate = _describe(
                    refined_origin, refined_pitch, count, seed.pitch, rank, centers, boundaries, phase,
                )
                if candidate is not None:
                    hypotheses.append(candidate)
    hypotheses.sort(key=lambda hypothesis: -hypothesis.score)
    distinct: list[LatticeHypothesis] = []
    # Keep two different spans per pitch neighborhood so a single harmonic
    # cannot fill the shortlist with neighboring counts.
    for hypothesis in hypotheses:
        same_period = [item for item in distinct if abs(item.pitch - hypothesis.pitch) < 0.06 * item.pitch]
        duplicate = any(
            item.cell_count == hypothesis.cell_count
            and max(abs(a - b) for a, b in zip(item.ideal_extent, hypothesis.ideal_extent, strict=True)) < 1.0
            for item in same_period
        )
        if not duplicate and len(same_period) < 2:
            distinct.append(hypothesis)
    if not distinct:
        return AxisLatticeFit("insufficient_evidence", ())

    best = distinct[0]
    unverified_extension = (
        (best.left_edge is None and best.origin - best.pitch / 2 >= 0)
        or (best.right_edge is None and best.ideal_extent[1] + best.pitch / 2 <= length)
    )
    ambiguous = unverified_extension or best.diagnostics.observed_agreement < 0.48 or any(
        best.score - item.score <= 0.035 for item in distinct[1:]
    )
    if analysis.centers.peak_positions.size < 2:
        ambiguous = ambiguous or any(
            abs(best.pitch / item.pitch - round(best.pitch / item.pitch)) <= 0.06
            and best.pitch / item.pitch > 1.85
            and item.diagnostics.observed_agreement >= best.diagnostics.observed_agreement - 0.12
            for item in distinct[1:]
        )
    return AxisLatticeFit("ambiguous" if ambiguous else "fitted", tuple(distinct[:max_hypotheses]))


def rank_axis_pairs(
    x: AxisLatticeFit, y: AxisLatticeFit, *, square_prior_weight: float = 0.0,
    expected_pitch_ratio: float = 1.0, log_ratio_sigma: float = 0.12,
) -> tuple[PairedLattice, ...]:
    """Optionally rerank independent fits with a soft x-to-y pitch ratio prior."""
    for name, value in (
        ("square_prior_weight", square_prior_weight),
        ("expected_pitch_ratio", expected_pitch_ratio),
        ("log_ratio_sigma", log_ratio_sigma),
    ):
        if not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0 <= square_prior_weight <= 1 or expected_pitch_ratio <= 0 or log_ratio_sigma <= 0:
        raise ValueError("prior weight must be in [0, 1] and ratio and sigma must be positive")
    ranked = []
    for horizontal in x.hypotheses:
        for vertical in y.hypotheses:
            independent = (horizontal.score + vertical.score) / 2
            log_error = math.log(horizontal.pitch / vertical.pitch / expected_pitch_ratio)
            agreement = math.exp(-0.5 * (log_error / log_ratio_sigma)**2)
            score = (1 - square_prior_weight) * independent + square_prior_weight * agreement
            ranked.append(PairedLattice(horizontal, vertical, independent, agreement, score))
    return tuple(sorted(ranked, key=lambda pair: -pair.score))
