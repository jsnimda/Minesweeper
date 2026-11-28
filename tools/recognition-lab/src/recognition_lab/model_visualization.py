"""Presentation of retained model artifacts; numerical arrays remain authoritative."""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image


CHANNELS = ('X sine', 'X cosine', 'Y sine', 'Y cosine', 'Occupancy',
            'X wrapped phase', 'Y wrapped phase', 'X phase magnitude', 'Y phase magnitude')
REPRESENTATION = 'RGB CHW float32 divided by 255'


def resolve_observation(selection, tile_index):
    snapshot = selection.snapshot
    if snapshot is None or snapshot.node(selection.selected) is None:
        return None
    selected = selection.selected
    if selected.kind in ('observation', 'hypothesis'):
        return snapshot.observation(selected)
    grid = selection.grid()
    phase = grid.phase_data if grid is not None and isinstance(grid.phase_data, dict) else {}
    observations = phase.get('observations') or ()
    index = tile_index if grid is not None and grid.diagnostics.get('mode') == 'tiled' else 0
    return observations[index] if 0 <= index < len(observations) else None


def scalar_field(fields, channel):
    indices = {'X cosine': 0, 'X sine': 1, 'Y cosine': 2, 'Y sine': 3, 'Occupancy': 4}
    if channel in indices:
        return fields[indices[channel]]
    if channel in ('X wrapped phase', 'X phase magnitude'):
        cosine, sine = fields[0], fields[1]
    elif channel in ('Y wrapped phase', 'Y phase magnitude'):
        cosine, sine = fields[2], fields[3]
    else:
        raise ValueError('unknown output channel')
    value = np.arctan2(sine, cosine) if channel.endswith('wrapped phase') else np.hypot(cosine, sine)
    return np.where(np.isfinite(cosine) & np.isfinite(sine), value, np.nan)


def scale_for(channel):
    if channel.endswith('wrapped phase'): return (-math.pi, math.pi, 'cyclic')
    if channel.endswith('phase magnitude'): return (0., math.sqrt(2), 'sequential')
    if channel == 'Occupancy': return (0., 1., 'sequential')
    if channel in CHANNELS: return (-1., 1., 'diverging')
    raise ValueError('unknown output channel')


def colors(values, scale):
    """Fixed scales; nonfinite samples are opaque magenta, never written back."""
    low, high, kind = scale
    values = np.asarray(values)
    finite = np.isfinite(values)
    t = np.clip(np.where(finite, (values-low)/(high-low), 0), 0, 1)
    if kind == 'cyclic':
        # Three equally spaced cosine waves meet continuously at both ends.
        rgb = np.stack([.5 + .5*np.cos(2*np.pi*t + phase)
                        for phase in (0, -2*np.pi/3, 2*np.pi/3)], axis=-1)
    elif kind == 'diverging':
        rgb = np.stack([np.minimum(2*t, 1), 1-np.abs(2*t-1), np.minimum(2*(1-t), 1)], axis=-1)
    else:
        rgb = np.stack([t, t, t], axis=-1)
    result = np.rint(rgb*255).astype(np.uint8)
    result[~finite] = (255, 0, 255)
    return result


@dataclass(frozen=True)
class DiagnosticView:
    image: Image.Image
    values: np.ndarray
    canvas_offset: tuple[int, int]
    coordinate_label: str
    channel: str | None
    scale: tuple | None

    def inspect(self, x, y):
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < self.image.width and 0 <= y < self.image.height):
            return None
        ix, iy = math.floor(x), math.floor(y)
        value = self.values[:, iy, ix] if self.channel is None else self.values[iy, ix]
        return (ix, iy), (x + self.canvas_offset[0], y + self.canvas_offset[1]), value


class DiagnosticCache:
    """Cache only one current observation, releasing it on selection/snapshot change."""
    def __init__(self):
        self.snapshot = None
        self.observation = None
        self.views = {}

    def clear(self):
        self.snapshot = self.observation = None
        self.views.clear()

    def get(self, snapshot, observation, mode, region, channel):
        if snapshot is not self.snapshot or observation is not self.observation:
            self.clear()
            self.snapshot, self.observation = snapshot, observation
        if observation is None:
            return None
        key = (mode, region, channel if mode == 'Model output' else None)
        if key not in self.views:
            self.views[key] = self._create(observation, *key)
        return self.views[key]

    @staticmethod
    def _create(observation, mode, region, channel):
        array = getattr(observation, 'input_array', None) if mode == 'Model input' else getattr(observation, 'full_fields', None)
        if array is None or array.ndim != 3 or array.shape[0] != (3 if mode == 'Model input' else 5):
            return None
        if mode == 'Model input' and getattr(observation, 'input_representation', None) != REPRESENTATION:
            return None
        h, w = array.shape[1:]
        if region == 'Known region':
            box = getattr(observation, 'known_box', None)
            if box is None: return None
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h): return None
        else:
            x0, y0, x1, y1 = 0, 0, w, h
        part = array[:, y0:y1, x0:x1]
        label = 'Known-region pixels' if region == 'Known region' else 'Model-canvas pixels'
        if mode == 'Model input':
            finite = np.isfinite(part).all(axis=0)
            rgb = np.moveaxis(np.clip(np.where(np.isfinite(part), part, 0), 0, 1), 0, -1)
            rgb = np.rint(rgb*255).astype(np.uint8)
            rgb[~finite] = (255, 0, 255)
            return DiagnosticView(Image.fromarray(rgb), part, (x0, y0), label, None, None)
        field = scalar_field(part, channel)
        scale = scale_for(channel)
        return DiagnosticView(Image.fromarray(colors(field, scale)), field, (x0, y0), label, channel, scale)
