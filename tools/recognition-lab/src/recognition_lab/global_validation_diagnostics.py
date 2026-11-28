"""Read retained global validation evidence without revalidating a board."""
import math
import numpy as np

from .global_diagnostics import global_point
from .model_visualization import colors


GLOBAL_CELL_METRICS = {'Cell occupancy': 'cell_occupancy', 'Cell coverage': 'cell_coverage'}


def cell_artifact(evidence, metric):
    if evidence is None or evidence.result is None or metric not in GLOBAL_CELL_METRICS: return None
    result = evidence.result
    geometry = getattr(result, 'geometry', None)
    if geometry is None: return None
    x, y = getattr(geometry, 'x_edges', None), getattr(geometry, 'y_edges', None)
    values = getattr(result, GLOBAL_CELL_METRICS[metric], None)
    if not all(isinstance(v, np.ndarray) for v in (x, y, values)): return None
    if any(not np.issubdtype(v.dtype, np.number) or np.iscomplexobj(v) for v in (x, y, values)): return None
    if (x.ndim != 1 or y.ndim != 1 or values.ndim != 2 or len(x) < 2 or len(y) < 2
            or values.shape != (len(y)-1, len(x)-1) or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.all(np.diff(x) > 0) or not np.all(np.diff(y) > 0)):
        return None
    return x, y, values


def inspect_cell(evidence, x, y):
    """Return original global cell values at screenshot coordinates."""
    if evidence is None: return None
    position = global_point(evidence.grid, x, y)
    if position is None: return None
    artifact = next((v for metric in GLOBAL_CELL_METRICS if (v := cell_artifact(evidence, metric)) is not None), None)
    if artifact is None: return None
    xx, yy, _ = artifact
    col = int(np.searchsorted(xx, position[0], side='right')-1)
    row = int(np.searchsorted(yy, position[1], side='right')-1)
    if not (0 <= row < len(yy)-1 and 0 <= col < len(xx)-1): return None
    result = {}
    for metric in GLOBAL_CELL_METRICS:
        data = cell_artifact(evidence, metric)
        if data is not None:
            value = data[2][row, col]
            result[metric] = value.item() if hasattr(value, 'item') else value
    return row, col, result


def termination_lines(evidence):
    """Project retained axis estimates across the crop; these are not sample locations."""
    if evidence is None or evidence.result is None: return ()
    sides = getattr(evidence.result, 'boundaries', None)
    if not isinstance(sides, dict): return ()
    try:
        grid = evidence.grid; tr = grid.heatmap_transform
        left, top, right, bottom = grid.crop.box
        if not (math.isfinite(tr.scale_x) and math.isfinite(tr.scale_y) and tr.scale_x > 0 and tr.scale_y > 0): return ()
        lines = []
        for name in ('left', 'right', 'top', 'bottom'):
            side = sides.get(name)
            value = side.get('estimate') if isinstance(side, dict) else None
            if value is None or not math.isfinite(value): continue
            if name in ('left', 'right'):
                x = left+(value-tr.offset[0])/tr.scale_x
                line = (x, top, x, bottom)
            else:
                y = top+(value-tr.offset[1])/tr.scale_y
                line = (left, y, right, y)
            if all(math.isfinite(v) for v in line): lines.append((name, line))
        return tuple(lines)
    except (AttributeError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return ()


class GlobalCellCache:
    def __init__(self): self.key = self.rgb = self.artifact = None
    def clear(self): self.key = self.rgb = self.artifact = None

    def get(self, snapshot, evidence, metric):
        if evidence is None or evidence.result is None or metric not in GLOBAL_CELL_METRICS:
            self.clear(); return None
        key = (snapshot, evidence.grid, evidence.result, metric)
        if self.key is not None and all(a is b for a, b in zip(self.key[:3], key[:3])) and self.key[3] == metric:
            return self.rgb
        self.clear(); self.key = key
        self.artifact = cell_artifact(evidence, metric)
        if self.artifact is not None:
            self.rgb = colors(self.artifact[2], (0., 1., 'sequential'))
        return self.rgb


def visible_cells(evidence, rgb, viewport, limit=20000, artifact=None):
    """Yield exact source rectangles for visible cells, bounded for rendering."""
    if evidence is None or rgb is None or viewport is None or limit <= 0: return ()
    artifact = artifact if artifact is not None else (cell_artifact(evidence, 'Cell occupancy') or cell_artifact(evidence, 'Cell coverage'))
    if artifact is None: return ()
    xx, yy, _ = artifact
    if rgb.shape != (len(yy)-1, len(xx)-1, 3): return ()
    try:
        grid = evidence.grid; tr = grid.heatmap_transform
        left, top = grid.crop.box[:2]
        if min(tr.scale_x, tr.scale_y) <= 0: return ()
        x0, y0, x1, y1 = viewport
        a = global_point(grid, x0, y0); b = global_point(grid, x1, y1)
        if a is None or b is None: return ()
        c0 = max(0, int(np.searchsorted(xx, a[0], side='right')-1))
        c1 = min(len(xx)-1, int(np.searchsorted(xx, b[0], side='left')))
        r0 = max(0, int(np.searchsorted(yy, a[1], side='right')-1))
        r1 = min(len(yy)-1, int(np.searchsorted(yy, b[1], side='left')))
        result = []
        for row in range(r0, r1):
            for col in range(c0, c1):
                if len(result) >= limit: return tuple(result)
                bounds = (left+(xx[col]-tr.offset[0])/tr.scale_x, top+(yy[row]-tr.offset[1])/tr.scale_y,
                          left+(xx[col+1]-tr.offset[0])/tr.scale_x, top+(yy[row+1]-tr.offset[1])/tr.scale_y)
                if all(math.isfinite(v) for v in bounds):
                    result.append((row, col, bounds, tuple(int(v) for v in rgb[row, col])))
        return tuple(result)
    except (AttributeError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return ()
