"""Decode one axis-aligned board from activated phase and occupancy fields.

All coordinates are continuous image coordinates; pixel centers are half integer.
The estimates and thresholds are experimental and are not calibrated probabilities.
"""
from dataclasses import dataclass, field
import math
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class FrequencyCandidate:
    pitch: float
    origin_mod_pitch: float
    coherence: float
    magnitude: float


@dataclass(frozen=True)
class DecoderSettings:
    min_pitch: float = 6.0
    max_pitch: float = 64.0
    frequency_limit: int = 4
    joint_limit: int = 4
    occupancy_threshold: float = .5
    min_magnitude: float = .15
    min_coherence: float = .5
    square_tau: float = .15
    square_weight: float = .08
    scale_x: float = 1.0
    scale_y: float = 1.0
    ambiguity_margin: float = .025

    def __post_init__(self):
        numbers = (self.min_pitch, self.max_pitch, self.occupancy_threshold,
                   self.min_magnitude, self.min_coherence, self.square_tau,
                   self.square_weight, self.scale_x, self.scale_y, self.ambiguity_margin)
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError("settings must be finite")
        if not (2 <= self.min_pitch < self.max_pitch and 0 < self.occupancy_threshold < 1
                and 0 <= self.min_magnitude <= 2 and 0 <= self.min_coherence <= 1
                and self.square_tau > 0 and 0 <= self.square_weight <= 1
                and self.scale_x > 0 and self.scale_y > 0 and self.ambiguity_margin >= 0
                and type(self.frequency_limit) is int and 1 <= self.frequency_limit <= 16
                and type(self.joint_limit) is int and 1 <= self.joint_limit <= 16):
            raise ValueError("invalid decoder settings")


@dataclass(frozen=True)
class AxisExtent:
    origin: float
    pitch: float
    count: int
    visible_start: float | None
    visible_end: float | None
    start_censored: bool
    end_censored: bool

    @property
    def end(self):
        return self.origin + self.count * self.pitch


@dataclass(frozen=True)
class GridHypothesis:
    x: AxisExtent
    y: AxisExtent
    x_evidence: FrequencyCandidate
    y_evidence: FrequencyCandidate
    square_preference: float
    prior_contribution: float
    score: float

    @property
    def columns(self):
        return self.x.count

    @property
    def rows(self):
        return self.y.count


@dataclass(frozen=True)
class DecodeResult:
    status: str
    hypotheses: tuple[GridHypothesis, ...]
    signals: dict = field(default_factory=dict)
    frequency_candidates: dict = field(default_factory=dict)
    occupancy_profiles: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)
    reasons: tuple[str, ...] = ()


def _fields(fields):
    values = np.asarray(fields, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 5 or min(values.shape[1:]) < 2 or not np.isfinite(values).all():
        raise ValueError("fields must be finite (5, height, width) predictions")
    if np.any(values[:4] < -1.00001) or np.any(values[:4] > 1.00001) or np.any(values[4] < 0) or np.any(values[4] > 1):
        raise ValueError("expected activated phase fields in [-1,1] and occupancy in [0,1]")
    return values


def directional_signal(fields: np.ndarray, axis: str, occupancy_threshold: float = .5):
    """Return the magnitude-weighted complex projection and its support."""
    values = _fields(fields)
    if axis not in ("x", "y") or not 0 <= occupancy_threshold <= 1:
        raise ValueError("invalid axis or occupancy threshold")
    first = 0 if axis == "x" else 2
    complex_field = values[first] + 1j * values[first + 1]
    magnitude = np.abs(complex_field)
    weights = np.where(values[4] >= occupancy_threshold, values[4], 0) * magnitude
    reduce_axis = 0 if axis == "x" else 1
    support = weights.sum(axis=reduce_axis)
    signal = (complex_field * weights).sum(axis=reduce_axis) / np.maximum(support, 1e-12)
    return signal, support


def _proposals(signal, support, min_pitch, max_pitch, limit):
    n = len(signal)
    fft_size = 1 << (max(256, n * 4) - 1).bit_length()
    frequencies = np.fft.fftfreq(fft_size)
    spectrum = np.abs(np.fft.fft(signal * support, n=fft_size))
    eligible = (frequencies >= 1 / max_pitch) & (frequencies <= 1 / min_pitch)
    peaks = eligible & (spectrum >= np.roll(spectrum, 1)) & (spectrum >= np.roll(spectrum, -1))
    # Include range endpoints: the maximum may be just outside the allowed interval.
    valid = np.flatnonzero(eligible)
    if valid.size:
        peaks[valid[0]] = peaks[valid[-1]] = True
    indices = np.flatnonzero(peaks)
    indices = sorted(indices, key=lambda i: spectrum[i], reverse=True)[:limit]
    return frequencies[indices], 1 / fft_size


def _refine(signal, support, centers, step, min_pitch, max_pitch):
    positions = np.arange(len(signal), dtype=np.float64) + .5
    weighted = signal * support
    denom = np.sum(support * np.abs(signal))
    candidates = []
    for center in centers:
        options = np.linspace(max(1 / max_pitch, center - step), min(1 / min_pitch, center + step), 33)
        correlations = np.array([np.sum(weighted * np.exp(-2j * np.pi * frequency * positions)) for frequency in options])
        best = int(np.argmax(np.abs(correlations)))
        frequency = float(options[best])
        amplitude = correlations[best] / support.sum()
        pitch = 1 / frequency
        candidates.append(FrequencyCandidate(pitch, float((-np.angle(amplitude) / (2 * np.pi * frequency)) % pitch),
                                             float(abs(correlations[best]) / max(denom, 1e-12)),
                                             float(denom / support.sum())))
    return candidates


def frequency_candidates(signal, support, *, min_pitch=6., max_pitch=64., limit=4):
    """Estimate frequencies using FFT proposals and local complex correlation."""
    signal = np.asarray(signal, dtype=np.complex128)
    support = np.asarray(support, dtype=np.float64)
    if signal.ndim != 1 or signal.shape != support.shape or not np.isfinite(signal).all() or not np.isfinite(support).all() or np.any(support < 0):
        raise ValueError("signal and nonnegative support must be finite matching vectors")
    if not 2 <= min_pitch < max_pitch or type(limit) is not int or limit < 1:
        raise ValueError("invalid pitch range or limit")
    if not np.any(support > 0) or not np.any(np.abs(signal) > 0):
        return []
    centers, step = _proposals(signal, support, min_pitch, max_pitch, limit)
    return _refine(signal, support, centers, step, min_pitch, max_pitch)


def _occupancy_profiles(occupancy, threshold):
    # Restrict the orthogonal mean to rows/columns containing evidence, avoiding
    # dilution by large surrounding margins. Multiple components remain ambiguous.
    active_rows = np.max(occupancy, axis=1) >= threshold
    active_columns = np.max(occupancy, axis=0) >= threshold
    if not active_rows.any() or not active_columns.any():
        return None
    return occupancy[active_rows].mean(axis=0), occupancy[:, active_columns].mean(axis=1)


def _crossing(a, b, threshold):
    return 0.5 if a == b else float(np.clip((threshold - a) / (b - a), 0, 1))


def _profile_extent(profile, threshold):
    active = np.flatnonzero(profile >= threshold)
    if not active.size:
        return None
    first, last = int(active[0]), int(active[-1])
    start_censored, end_censored = first == 0, last == len(profile) - 1
    start = None if start_censored else first - .5 + _crossing(profile[first - 1], profile[first], threshold)
    end = None if end_censored else last + .5 + _crossing(profile[last], profile[last + 1], threshold)
    gaps = bool(np.any(profile[first:last + 1] < threshold))
    return start, end, start_censored, end_censored, gaps


def _axis_extent(candidate, profile_extent, length):
    start, end, start_censored, end_censored, _ = profile_extent
    pitch, phase = candidate.pitch, candidate.origin_mod_pitch
    # A censored edge indicates only the lattice cells intersecting the image;
    # it is never represented as an observed board termination.
    if start_censored:
        first = math.floor((0 - phase) / pitch)
    else:
        first = round((start - phase) / pitch)
    if end_censored:
        last = math.ceil((length - phase) / pitch)
    else:
        last = round((end - phase) / pitch)
    if last <= first:
        return None
    return AxisExtent(phase + first * pitch, pitch, last - first, start, end, start_censored, end_censored)


def decode_phase(fields, settings: DecoderSettings = DecoderSettings()):
    """Return ranked finite proposals and diagnostics without validation or repair."""
    started = perf_counter()
    values = _fields(fields)
    timings = {}
    x_signal, x_support = directional_signal(values, "x", settings.occupancy_threshold)
    y_signal, y_support = directional_signal(values, "y", settings.occupancy_threshold)
    profiles = _occupancy_profiles(values[4], settings.occupancy_threshold)
    now = perf_counter(); timings['signal_extraction'] = (now - started) * 1000
    signals = {"x": x_signal, "y": y_signal, "x_support": x_support, "y_support": y_support}
    profile_map = {} if profiles is None else {"x": profiles[0], "y": profiles[1]}
    if profiles is None or not x_support.any() or not y_support.any():
        timings['total'] = (perf_counter() - started) * 1000
        return DecodeResult("insufficient_evidence", (), signals, {}, profile_map, timings, ("no occupancy or phase support",))
    proposal_data = [_proposals(signal, support, settings.min_pitch, settings.max_pitch, settings.frequency_limit)
                     for signal, support in ((x_signal, x_support), (y_signal, y_support))]
    next_time = perf_counter(); timings['frequency_proposals'] = (next_time - now) * 1000; now = next_time
    candidates = [_refine(signal, support, centers, step, settings.min_pitch, settings.max_pitch)
                  for (signal, support), (centers, step) in zip(((x_signal, x_support), (y_signal, y_support)), proposal_data)]
    next_time = perf_counter(); timings['pitch_refinement'] = (next_time - now) * 1000; now = next_time
    candidate_map = {"x": tuple(candidates[0]), "y": tuple(candidates[1])}
    valid = [[c for c in axis if c.magnitude >= settings.min_magnitude and c.coherence >= settings.min_coherence] for axis in candidates]
    joint = []
    for x in valid[0]:
        for y in valid[1]:
            ratio = math.log((x.pitch / settings.scale_x) / (y.pitch / settings.scale_y))
            preference = math.exp(-(ratio * ratio) / (2 * settings.square_tau ** 2))
            prior = settings.square_weight * preference
            score = (x.coherence + y.coherence) / 2 + prior
            joint.append((score, x, y, preference, prior))
    joint.sort(key=lambda item: item[0], reverse=True)
    joint = joint[:settings.joint_limit]
    next_time = perf_counter(); timings['joint_candidates'] = (next_time - now) * 1000; now = next_time
    extents = [_profile_extent(profile, settings.occupancy_threshold) for profile in profiles]
    hypotheses = []
    if all(extent is not None for extent in extents):
        for score, x, y, preference, prior in joint:
            x_extent = _axis_extent(x, extents[0], values.shape[2])
            y_extent = _axis_extent(y, extents[1], values.shape[1])
            if x_extent is not None and y_extent is not None:
                hypotheses.append(GridHypothesis(x_extent, y_extent, x, y, preference, prior, score))
    timings['extent_estimation'] = (perf_counter() - now) * 1000
    reasons = []
    if any(extent is not None and extent[4] for extent in extents):
        reasons.append("disconnected occupancy profile")
    if any(extent is not None and (extent[2] or extent[3]) for extent in extents):
        reasons.append("image boundary censors termination")
    if len(hypotheses) > 1 and hypotheses[0].score - hypotheses[1].score < settings.ambiguity_margin:
        reasons.append("competing frequency hypotheses")
    status = "insufficient_evidence" if not hypotheses else ("ambiguous" if reasons else "fitted")
    timings['total'] = (perf_counter() - started) * 1000
    return DecodeResult(status, tuple(hypotheses), signals, candidate_map, profile_map, timings, tuple(reasons))
