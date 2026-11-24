"""Generate and rank candidate pitches for one axis of grid evidence."""

from dataclasses import dataclass, replace
import math
from numbers import Integral, Real

import numpy as np

from .signals import AxisSignals, FloatArray, probability_array


@dataclass(frozen=True, eq=False)
class ChannelEvidence:
    peak_positions: FloatArray
    peak_heights: FloatArray
    peak_prominences: FloatArray
    lags: FloatArray
    autocorrelation: FloatArray
    support: tuple[int, int]


@dataclass(frozen=True)
class PhaseAlignment:
    center_concentration: float
    boundary_concentration: float
    offset_cycles: float | None
    score: float


@dataclass(frozen=True)
class PitchCandidate:
    pitch: float
    score: float
    pair_alignment: float
    one_step_support: float
    autocorrelation: float
    phase: PhaseAlignment
    sources: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class PitchAnalysis:
    centers: ChannelEvidence
    boundaries: ChannelEvidence
    candidates: tuple[PitchCandidate, ...]


def _local_maxima(values: FloatArray) -> tuple[FloatArray, np.ndarray, np.ndarray]:
    """Return fractional sample indices and the inclusive bounds of each maximum."""
    slopes = np.diff(values)
    changes = np.flatnonzero(slopes != 0)
    ascending = changes[:-1]
    descending = changes[1:]
    turns = (slopes[ascending] > 0) & (slopes[descending] < 0)
    first = ascending[turns] + 1
    last = descending[turns]
    positions = (first + last) / 2

    # Fit a parabola to the neighbors of a single-sample maximum.
    single = first == last
    left_slopes = slopes[first[single] - 1]
    right_slopes = slopes[first[single]]
    positions[single] += 0.5 * (left_slopes + right_slopes) / (left_slopes - right_slopes)
    return positions, first, last


def _detect_peaks(
    signal: FloatArray, min_pitch: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    positions, first, last = _local_maxima(signal)
    if not first.size:
        return np.empty(0), np.empty(0), np.empty(0)

    radius = math.ceil(min_pitch)
    heights = signal[first]
    prominences = np.empty(first.size)
    for index, (start, stop) in enumerate(zip(first, last, strict=True)):
        left = signal[max(0, start - radius):start + 1]
        right = signal[stop:stop + radius + 1]
        prominences[index] = heights[index] - max(left.min(), right.min())

    curvature = np.diff(signal, n=2)
    mad = np.median(np.abs(curvature - np.median(curvature)))
    noise = mad / (0.67449 * math.sqrt(6))
    eligible = np.flatnonzero(prominences >= max(0.05 * prominences.max(), 2.8 * noise))
    strongest = eligible[np.argsort(-prominences[eligible], kind="stable")]
    selected = []
    for index in strongest:
        if all(abs(positions[index] - positions[other]) >= 0.45 * min_pitch for other in selected):
            selected.append(index)
    ordered = np.sort(np.asarray(selected, dtype=int))
    return positions[ordered] + 0.5, heights[ordered], prominences[ordered]


def _autocorrelation(signal: FloatArray, max_pitch: float) -> tuple[FloatArray, FloatArray]:
    """Pearson correlation over the overlapping samples at each lag."""
    length = signal.size
    max_lag = min(math.ceil(max_pitch) + 1, length - max(5, math.ceil(0.4 * length)))
    if max_lag < 1 or np.ptp(signal) == 0:
        return np.empty(0), np.empty(0)

    values = (signal - signal.min()) / np.ptp(signal)
    lags = np.arange(max_lag + 1)
    overlap = length - lags
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    energy = np.concatenate(([0.0], np.cumsum(values**2)))
    left_sum, right_sum = cumulative[overlap], cumulative[-1] - cumulative[lags]
    left_variance = energy[overlap] - left_sum**2 / overlap
    right_variance = energy[-1] - energy[lags] - right_sum**2 / overlap
    covariance = np.correlate(values, values, mode="full")[length - 1:length + max_lag]
    covariance -= left_sum * right_sum / overlap
    denominator = np.sqrt(np.maximum(left_variance, 0) * np.maximum(right_variance, 0))
    correlation = np.divide(
        covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-12,
    )
    return lags.astype(np.float64), correlation.clip(-1, 1)


def _channel_evidence(
    signal: FloatArray, min_pitch: float, max_pitch: float,
) -> ChannelEvidence:
    positions, heights, prominences = _detect_peaks(signal, min_pitch)
    if positions.size < 2:
        return ChannelEvidence(positions, heights, prominences, np.empty(0), np.empty(0), (0, 0))
    margin = 0.55 * min_pitch
    start = max(0, math.floor(positions[0] - margin))
    stop = min(signal.size, math.ceil(positions[-1] + margin))
    lags, correlation = _autocorrelation(signal[start:stop], max_pitch)
    return ChannelEvidence(positions, heights, prominences, lags, correlation, (start, stop))


def _peak_weights(evidence: ChannelEvidence) -> FloatArray:
    return np.sqrt(evidence.peak_prominences / evidence.peak_prominences.max())


def _peak_pairs(evidence: ChannelEvidence) -> tuple[FloatArray, FloatArray]:
    positions = evidence.peak_positions
    left, right = np.triu_indices(positions.size, k=1)
    weights = _peak_weights(evidence)
    return positions[right] - positions[left], weights[left] * weights[right]


def _pair_proposals(
    pairs: list[tuple[FloatArray, FloatArray]],
    min_pitch: float,
    max_pitch: float,
    max_intervals: int,
) -> FloatArray:
    """Bin weighted distance/divisor votes so nearly identical pairs share a seed."""
    spacing = 0.5
    votes = np.zeros(math.ceil(max_pitch / spacing) + 3)
    for distances, weights in pairs:
        for intervals in range(1, max_intervals + 1):
            pitches = distances / intervals
            in_bounds = (pitches >= min_pitch) & (pitches <= max_pitch)
            bins = pitches[in_bounds] / spacing
            lower = bins.astype(int)
            fraction = bins - lower
            vote_weights = weights[in_bounds] / intervals
            np.add.at(votes, lower, vote_weights * (1 - fraction))
            np.add.at(votes, lower + 1, vote_weights * fraction)

    maxima, _, _ = _local_maxima(votes)
    return np.clip(maxima * spacing, min_pitch, max_pitch)


def phase_alignment(
    centers: ChannelEvidence, boundaries: ChannelEvidence, pitch: float,
) -> PhaseAlignment:
    """Measure per-channel phase coherence and center-minus-boundary phase in cycles."""
    if not math.isfinite(pitch) or pitch <= 0:
        raise ValueError("pitch must be finite and positive")

    def phase_vector(evidence: ChannelEvidence) -> complex:
        if evidence.peak_positions.size < 2:
            return 0j
        directions = np.exp(2j * np.pi * evidence.peak_positions / pitch)
        return complex(np.average(directions, weights=_peak_weights(evidence)))

    center_phase = phase_vector(centers)
    boundary_phase = phase_vector(boundaries)
    center_strength = min(abs(center_phase), 1.0)
    boundary_strength = min(abs(boundary_phase), 1.0)
    if min(center_strength, boundary_strength) <= 1e-8:
        return PhaseAlignment(center_strength, boundary_strength, None, 0.0)

    offset = float((np.angle(center_phase * boundary_phase.conjugate()) / (2 * np.pi)) % 1)
    error = abs(offset - 0.5)
    agreement = math.exp(-0.5 * (error / 0.12)**2)
    return PhaseAlignment(center_strength, boundary_strength, offset, center_strength * boundary_strength * agreement)


def _refine_pitch(
    proposal: float, pairs: list[tuple[FloatArray, FloatArray]], min_pitch: float, max_pitch: float,
) -> float:
    total_weight = 0.0
    weighted_pitch = 0.0
    tolerance = min(1.25, 0.25 + 0.08 * proposal)
    for distances, weights in pairs:
        intervals = np.rint(distances / proposal)
        close = (intervals >= 1) & (np.abs(distances - intervals * proposal) <= tolerance)
        total_weight += float(np.sum(weights[close] * intervals[close]))
        weighted_pitch += float(np.sum(weights[close] * distances[close]))
    if total_weight == 0:
        return proposal
    return float(np.clip(weighted_pitch / total_weight, min_pitch, max_pitch))


def _score_candidate(
    pitch: float,
    channels: list[ChannelEvidence],
    pairs: list[tuple[FloatArray, FloatArray]],
    phase: PhaseAlignment,
    source: str,
) -> PitchCandidate:
    tolerance = min(1.3, 0.30 + 0.045 * pitch)
    pair_scores = []
    step_scores = []
    correlations = []
    for evidence, (distances, weights) in zip(channels, pairs, strict=True):
        intervals = np.rint(distances / pitch)
        residuals = distances - intervals * pitch
        agreement = np.exp(-0.5 * (residuals / tolerance)**2) * (intervals >= 1)
        pair_scores.append(float(np.average(agreement, weights=weights)))

        positions = evidence.peak_positions
        expected = positions + pitch
        eligible = expected <= positions[-1] + tolerance
        if eligible.any():
            errors = np.abs(expected[eligible, None] - positions[None, :]).min(axis=1)
            agreement = np.exp(-0.5 * (errors / tolerance)**2)
            step_scores.append(float(np.average(agreement, weights=_peak_weights(evidence)[eligible])))
        else:
            step_scores.append(0.0)

        if evidence.lags.size:
            correlation = np.interp(pitch, evidence.lags, evidence.autocorrelation, right=0)
            length = evidence.support[1] - evidence.support[0]
            overlap = max(0.0, 1 - pitch / length)
            correlations.append(max(0.0, float(correlation)) * math.sqrt(overlap))
        else:
            correlations.append(0.0)

    pair = float(np.mean(pair_scores))
    step = float(np.mean(step_scores))
    correlation = float(np.mean(correlations))
    score = 0.40 * pair + 0.23 * step + 0.20 * correlation + 0.17 * phase.score
    return PitchCandidate(pitch, score, pair, step, correlation, phase, (source,))


def estimate_pitch(
    signals: AxisSignals,
    *,
    min_pitch: float = 4.0,
    max_pitch: float = 128.0,
    max_candidates: int = 8,
    max_pair_intervals: int = 6,
) -> PitchAnalysis:
    """Return ranked pitch hypotheses and diagnostic evidence for one axis.

    Each channel proposes periods independently. The phase term tests whether
    their peaks are half a cycle apart at a proposed pitch.
    """
    centers = probability_array(signals.centers)
    boundaries = probability_array(signals.boundaries)
    if centers.ndim != 1 or centers.shape != boundaries.shape or not centers.size:
        raise ValueError("center and boundary signals must be nonempty 1D arrays of equal length")
    if (
        not isinstance(min_pitch, Real) or not isinstance(max_pitch, Real)
        or not math.isfinite(min_pitch) or not math.isfinite(max_pitch)
        or not 2 <= min_pitch < max_pitch
    ):
        raise ValueError("pitch bounds must be finite and satisfy 2 <= min_pitch < max_pitch")
    for name, value in (("max_candidates", max_candidates), ("max_pair_intervals", max_pair_intervals)):
        if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    max_pitch = min(max_pitch, centers.size - 1)
    center_evidence = _channel_evidence(centers, min_pitch, max_pitch)
    boundary_evidence = _channel_evidence(boundaries, min_pitch, max_pitch)
    channels = [channel for channel in (center_evidence, boundary_evidence) if channel.peak_positions.size >= 2]
    if not channels or max_pitch < min_pitch:
        return PitchAnalysis(center_evidence, boundary_evidence, ())

    pairs = [_peak_pairs(evidence) for evidence in channels]
    proposals = []
    for evidence in channels:
        peaks, _, _ = _local_maxima(evidence.autocorrelation)
        for peak in peaks:
            if min_pitch <= peak <= max_pitch and np.interp(peak, evidence.lags, evidence.autocorrelation) > 0:
                proposals.append((float(peak), "autocorrelation"))
    proposals.extend(
        (float(pitch), "peak_pairs")
        for pitch in _pair_proposals(pairs, min_pitch, max_pitch, max_pair_intervals)
    )

    scored = []
    for proposal, source in proposals:
        pitch = _refine_pitch(proposal, pairs, min_pitch, max_pitch)
        phase = phase_alignment(center_evidence, boundary_evidence, pitch)
        scored.append(_score_candidate(pitch, channels, pairs, phase, source))
    scored.sort(key=lambda candidate: (-candidate.score, candidate.pitch))

    distinct: list[PitchCandidate] = []
    for candidate in scored:
        for index, existing in enumerate(distinct):
            if abs(candidate.pitch - existing.pitch) <= min(1.0, 0.07 * existing.pitch):
                sources = tuple(sorted(set(existing.sources + candidate.sources)))
                distinct[index] = replace(existing, sources=sources)
                break
        else:
            distinct.append(candidate)
    return PitchAnalysis(center_evidence, boundary_evidence, tuple(distinct[:max_candidates]))
