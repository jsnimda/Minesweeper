"""Read retained decoder evidence without repeating frequency estimation."""
from dataclasses import dataclass
import math

import numpy as np

from .local_geometry import resolve_local
from .model_visualization import resolve_observation


@dataclass(frozen=True)
class FrequencyEvidence:
    observation: object
    observation_index: int
    detection_index: int | None
    mode: str | None
    fallback: bool
    decoder: object | None
    hypothesis: object | None
    validation: str | None


def resolve_frequency(selection, tile):
    snapshot = selection.snapshot
    if snapshot is None or snapshot.node(selection.selected) is None:
        return None
    grid = selection.grid()
    observation = resolve_observation(selection, tile)
    if grid is None or observation is None:
        return None
    observations = grid.phase_data.get('observations', ()) if isinstance(grid.phase_data, dict) else ()
    index = next((i for i, value in enumerate(observations) if value is observation), None)
    if index is None: return None
    local = resolve_local(selection, tile)
    return FrequencyEvidence(observation, index, snapshot.detection_index(selection.selected), grid.diagnostics.get('mode'),
                             selection.selected.kind not in ('observation', 'hypothesis'),
                             getattr(observation, 'decoding', None),
                             local.hypothesis if local else None, local.validation if local else None)


def candidate_rows(evidence, axis):
    """Preserve decoder ordering; association uses retained object identity."""
    if evidence is None or evidence.decoder is None or axis not in ('x', 'y'):
        return ()
    candidates = getattr(evidence.decoder, 'frequency_candidates', {}).get(axis, ())
    selected = getattr(evidence.hypothesis, f'{axis}_evidence', None)
    rows = []
    for index, candidate in enumerate(candidates):
        pitch = candidate.pitch
        try:
            frequency = 1. / pitch if np.isrealobj(pitch) and math.isfinite(pitch) and pitch > 0 else None
        except (TypeError, ValueError, OverflowError):
            frequency = None
        rows.append((index, pitch, frequency, candidate.origin_mod_pitch,
                     candidate.coherence, candidate.magnitude, candidate is selected))
    return tuple(rows)


PROFILE_CHOICES = ('Signal real', 'Signal imaginary', 'Signal support', 'Occupancy profile')


def profile(evidence, axis, choice):
    """Return original array/view at observation pixel centers, never interpolated."""
    if evidence is None or evidence.decoder is None or axis not in ('x', 'y'):
        return None
    decoded = evidence.decoder
    if choice == 'Occupancy profile':
        values = getattr(decoded, 'occupancy_profiles', {}).get(axis)
    elif choice == 'Signal support':
        values = getattr(decoded, 'signals', {}).get(axis + '_support')
    else:
        signal = getattr(decoded, 'signals', {}).get(axis)
        if signal is None: return None
        values = np.asarray(signal).real if choice == 'Signal real' else np.asarray(signal).imag if choice == 'Signal imaginary' else None
    if values is None: return None
    values = np.asarray(values)
    return values if values.ndim == 1 and np.issubdtype(values.dtype, np.number) and not np.iscomplexobj(values) else None


def profile_markers(evidence, axis, length, limit=1000):
    """Visible reference markers; actual coordinates remain unchanged."""
    if evidence is None or evidence.hypothesis is None or length <= 0 or axis not in ('x', 'y'):
        return ()
    extent = getattr(evidence.hypothesis, axis, None)
    try:
        if (extent is None or not math.isfinite(extent.origin) or not math.isfinite(extent.pitch)
                or extent.pitch <= 0 or type(extent.count) is not int or not 0 < extent.count <= 1_000_000
                or not math.isfinite(extent.end)):
            return ()
        lo, hi = .5, length-.5
        result = []
        first = max(0, math.ceil((lo-extent.origin)/extent.pitch))
        last = min(extent.count, math.floor((hi-extent.origin)/extent.pitch))
        for index in range(first, min(last+1, first+limit)):
            result.append(('lattice', extent.origin + index*extent.pitch))
        for value in (extent.origin, extent.end):
            if lo <= value <= hi: result.append(('equivalent boundary', value))
        for value, censored in ((extent.visible_start, extent.start_censored), (extent.visible_end, extent.end_censored)):
            if not censored and value is not None and math.isfinite(value) and lo <= value <= hi:
                result.append(('visible crossing', value))
        return tuple(result)
    except (TypeError, ValueError, OverflowError):
        return ()


def sample_value(values, coordinate):
    """Nearest discrete pixel-center sample; returns the stored value."""
    if values is None or not math.isfinite(coordinate): return None
    index = math.floor(coordinate)
    return (index, values[index]) if 0 <= index < len(values) else None
