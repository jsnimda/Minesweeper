"""Read retained global consensus and fusion artifacts for inspection."""
from dataclasses import dataclass
import math

import numpy as np

from .viewer_selection import observation_source_bounds


@dataclass(frozen=True)
class GlobalEvidence:
    grid: object
    result: object | None
    observations: object
    detection_index: int | None
    mode: str | None


def resolve_global(selection):
    snapshot = selection.snapshot
    if snapshot is None or snapshot.node(selection.selected) is None: return None
    grid = selection.grid()
    if grid is None: return None
    phase = grid.phase_data if isinstance(grid.phase_data, dict) else {}
    observations = phase.get('observations')
    return GlobalEvidence(grid, phase.get('global'), observations,
                          snapshot.detection_index(selection.selected), grid.diagnostics.get('mode'))


def participation(lattice, index):
    """Read membership by stored indices; missing or contradictory data stays unknown."""
    if lattice is None: return 'Unavailable'
    inside, excluded = getattr(lattice, 'inliers', None), getattr(lattice, 'rejected', None)
    if inside is None or excluded is None: return 'Unavailable'
    a, b = index in inside, index in excluded
    return 'Inlier' if a and not b else 'Excluded' if b and not a else 'Unavailable'


def observation_row(evidence, index):
    if evidence is None or evidence.observations is None or not 0 <= index < len(evidence.observations): return None
    obs = evidence.observations[index]
    local = getattr(obs, 'local_validation', None)
    accepted = getattr(local, 'local_accepted', None)
    local_status = 'Accepted' if accepted is True else 'Rejected' if accepted is False else 'Unavailable'
    eligible = getattr(obs, 'local_valid', None)
    eligibility = 'Eligible' if eligible is True else 'Not eligible' if eligible is False else 'Unavailable'
    hypothesis = getattr(obs, 'hypothesis', None)
    x, y = getattr(hypothesis, 'x', None), getattr(hypothesis, 'y', None)
    xe, ye = getattr(hypothesis, 'x_evidence', None), getattr(hypothesis, 'y_evidence', None)
    lattice = getattr(evidence.result, 'lattice', None)
    return (index, getattr(getattr(obs, 'decoding', None), 'status', None), local_status, eligibility,
            participation(lattice, index), getattr(x, 'pitch', None), getattr(y, 'pitch', None),
            getattr(xe, 'origin_mod_pitch', None), getattr(ye, 'origin_mod_pitch', None), None, None)


def global_point(grid, x, y):
    """Map screenshot continuous coordinates to retained global array coordinates."""
    try:
        tr = grid.heatmap_transform
        left, top = grid.crop.box[:2]
        px = (x-left)*tr.scale_x + tr.offset[0]
        py = (y-top)*tr.scale_y + tr.offset[1]
        if not math.isfinite(px) or not math.isfinite(py): return None
        return px, py
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def inspect_global(evidence, x, y):
    if evidence is None or evidence.result is None: return None
    position = global_point(evidence.grid, x, y)
    if position is None: return None
    px, py = position
    col, row = math.floor(px), math.floor(py)
    values = {}
    for name in ('occupancy', 'weights'):
        array = getattr(evidence.result, name, None)
        if isinstance(array, np.ndarray) and array.ndim == 2 and 0 <= row < array.shape[0] and 0 <= col < array.shape[1]:
            value = array[row, col]
            values[name] = value.item() if hasattr(value, 'item') else value
    if not values: return None
    weight = values.get('weights')
    values['coverage'] = (bool(weight > 0) if weight is not None and math.isfinite(weight) else None)
    return (px, py), (col, row), values


def participation_rectangles(evidence):
    """Source rectangles with independent local eligibility and actual membership."""
    if evidence is None or evidence.observations is None: return ()
    result = []
    lattice = getattr(evidence.result, 'lattice', None)
    for index, observation in enumerate(evidence.observations):
        box = observation_source_bounds(evidence.grid, observation)
        if box is not None:
            checked = getattr(observation, 'local_validation', None)
            accepted = getattr(checked, 'local_accepted', None)
            local_status = 'accepted' if accepted is True else 'rejected' if accepted is False else 'unavailable'
            result.append((index, box, local_status, getattr(observation, 'local_valid', None), participation(lattice, index)))
    return tuple(result)


class GlobalHeatCache:
    """One selected snapshot/grid display; raw arrays remain authoritative."""
    def __init__(self):
        self.key = None
        self.rgba = None
        self.legend = None

    def clear(self):
        self.key = self.rgba = self.legend = None

    def get(self, snapshot, evidence, mode):
        if evidence is None or evidence.result is None or mode not in ('Phase occupancy', 'Phase coverage', 'Phase fusion weights'):
            self.clear(); return None, None
        key = (snapshot, evidence.grid, evidence.result, mode)
        if self.key is not None and all(a is b for a, b in zip(self.key[:3], key[:3])) and self.key[3] == mode:
            return self.rgba, self.legend
        self.clear(); self.key = key
        attr = 'occupancy' if mode == 'Phase occupancy' else 'weights'
        values = getattr(evidence.result, attr, None)
        if not isinstance(values, np.ndarray) or values.ndim != 2 or not values.size or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            return None, None
        valid = np.isfinite(values)
        if mode == 'Phase occupancy':
            intensity = np.clip(np.where(valid, values, 0), 0, 1)
            self.legend = 'Fused occupancy: 0–1; red alpha 0–50%; nonfinite transparent'
        elif mode == 'Phase coverage':
            intensity = (valid & (values > 0)).astype(float)
            self.legend = 'Coverage derived from fusion weight > 0: uncovered transparent; covered red at 50%'
        else:
            max_weight = float(np.max(values[valid])) if valid.any() else 0.
            max_weight = max(0., max_weight)
            intensity = np.clip(np.where(valid, values, 0) / max_weight, 0, 1) if max_weight > 0 else np.zeros(values.shape)
            self.legend = f'Fusion weight sum: 0–{max_weight!r}, linear red alpha 0–50%; not an observation count or probability'
        rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
        rgba[..., 0] = 255
        rgba[..., 3] = np.where(valid, np.floor(intensity*255).astype(np.uint8) // 2, 0).astype(np.uint8)
        self.rgba = rgba
        return rgba, self.legend
