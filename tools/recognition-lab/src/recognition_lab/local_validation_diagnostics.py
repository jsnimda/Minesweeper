"""Presentation helpers for the retained, evaluated local validation result."""
from dataclasses import dataclass
import math

import numpy as np

from .frequency_diagnostics import resolve_frequency
from .local_geometry import resolve_local
from .model_visualization import colors


METRICS = {
    'Occupancy': ('cell_occupancy', (0., 1., 'sequential')),
    'X phase magnitude': ('cell_magnitude_x', (0., math.sqrt(2), 'sequential')),
    'Y phase magnitude': ('cell_magnitude_y', (0., math.sqrt(2), 'sequential')),
    'X phase alignment': ('cell_alignment_x', (-1., 1., 'diverging')),
    'Y phase alignment': ('cell_alignment_y', (-1., 1., 'diverging')),
    'Supported mask': ('supported', None),
    'Observable mask': ('observable', None),
}


@dataclass(frozen=True)
class LocalValidationEvidence:
    frequency: object
    local: object | None
    validation: object | None
    status: str


def resolve_validation(selection, tile):
    frequency = resolve_frequency(selection, tile)
    if frequency is None:
        return None
    local = resolve_local(selection, tile)
    status = local.validation if local is not None else 'Unavailable'
    validation = getattr(local.observation, 'local_validation', None) if local is not None and status.startswith('Evaluated ') else None
    return LocalValidationEvidence(frequency, local, validation, status)


def cell_array(evidence, name):
    """Return the retained array only if it matches the evaluated lattice."""
    if evidence is None or evidence.validation is None or evidence.local is None:
        return None
    values = getattr(evidence.validation, name, None)
    if not isinstance(values, np.ndarray) or values.ndim != 2:
        return None
    if name in ('supported', 'observable') and values.dtype != np.bool_:
        return None
    if name not in ('supported', 'observable') and (not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values)):
        return None
    h = evidence.local.hypothesis
    if values.shape != (h.y.count, h.x.count):
        return None
    return values


def overlay_colors(evidence, metric):
    if metric not in METRICS: return None
    name, scale = METRICS[metric]
    values = cell_array(evidence, name)
    if values is None: return None
    observable = cell_array(evidence, 'observable')
    if metric == 'Observable mask':
        rgb = np.where(values[..., None], np.array([50, 180, 235], np.uint8), np.array([105, 105, 105], np.uint8))
    elif scale is None:
        rgb = np.where(values[..., None], np.array([46, 210, 126], np.uint8), np.array([245, 78, 86], np.uint8))
    else:
        rgb = colors(values, scale)
    if metric != 'Observable mask':
        # Values at unobservable centers may be clipped samples; do not present
        # them as observed evidence. Missing observability is also explicit.
        if observable is None:
            rgb[...] = (105, 105, 105)
        else:
            rgb[~observable.astype(bool)] = (105, 105, 105)
    alpha = np.full((*values.shape, 1), 255, dtype=np.uint8)
    return np.concatenate((rgb, alpha), axis=2)


def cell_at(evidence, transform, x, y):
    """Inspect the actual cell under viewer coordinates, retaining raw values."""
    if evidence is None or evidence.validation is None or evidence.local is None or transform is None:
        return None
    try:
        sx, sy, tx, ty = transform
        h = evidence.local.hypothesis
        numbers = (sx, sy, tx, ty, x, y, h.x.origin, h.y.origin, h.x.pitch, h.y.pitch)
        if not all(math.isfinite(v) for v in numbers) or min(sx, sy, h.x.pitch, h.y.pitch) <= 0: return None
        col = math.floor(((x-tx)/sx-h.x.origin)/h.x.pitch)
        row = math.floor(((y-ty)/sy-h.y.origin)/h.y.pitch)
        if not (0 <= row < h.y.count and 0 <= col < h.x.count): return None
        result = {}
        for name, _ in METRICS.values():
            values = cell_array(evidence, name)
            if values is not None:
                value = values[row, col]
                result[name] = value.item() if hasattr(value, 'item') else value
        return row, col, result
    except (AttributeError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None


def side_rows(evidence):
    """Retained side summaries plus the evaluated hypothesis's axis evidence."""
    if evidence is None or evidence.validation is None or evidence.local is None: return ()
    h = evidence.local.hypothesis
    sides = (('left', h.x.origin, h.x.visible_start, h.x.start_censored),
             ('right', h.x.end, h.x.visible_end, h.x.end_censored),
             ('top', h.y.origin, h.y.visible_start, h.y.start_censored),
             ('bottom', h.y.end, h.y.visible_end, h.y.end_censored))
    recorded = getattr(evidence.validation, 'side_evidence', {}) or {}
    return tuple((name, boundary, crossing, censored, recorded.get(name, {}))
                 for name, boundary, crossing, censored in sides)


class SupportCache:
    """Cache at most one display array; clear obsolete selection references."""
    def __init__(self):
        self.key = None
        self.rgba = None

    def clear(self):
        self.key = None
        self.rgba = None

    def get(self, snapshot, evidence, metric):
        if evidence is None or evidence.validation is None or metric not in METRICS:
            self.clear(); return None
        key = (snapshot, evidence.local.observation, evidence.local.hypothesis, evidence.validation, metric)
        if self.key is None or any(a is not b for a, b in zip(self.key[:4], key[:4])) or self.key[4] != metric:
            self.clear()
            self.key = key
            self.rgba = overlay_colors(evidence, metric)
        return self.rgba
