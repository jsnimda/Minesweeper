"""Resolve and project retained local hypotheses without altering recognition."""
from dataclasses import dataclass
import math

from .model_visualization import resolve_observation
from .recognition_state import hypothesis_validation
from .viewer_selection import observation_to_source


@dataclass(frozen=True)
class LocalSelection:
    grid: object
    observation: object
    hypothesis: object
    validation: str
    decoder_status: str | None
    fallback: bool


def resolve_local(selection, tile_index):
    snapshot, identity = selection.snapshot, selection.selected
    if snapshot is None or snapshot.node(identity) is None:
        return None
    grid = selection.grid()
    observation = resolve_observation(selection, tile_index)
    if grid is None or observation is None:
        return None
    if identity.kind == 'hypothesis':
        hypothesis = snapshot.resolve(identity)
        index = identity.hypothesis_index
    else:
        hypothesis = getattr(observation, 'hypothesis', None)
        index = 0
    if hypothesis is None:
        return None
    decoded = getattr(observation, 'decoding', None)
    return LocalSelection(grid, observation, hypothesis, hypothesis_validation(observation, index),
                          getattr(decoded, 'status', None), identity.kind not in ('observation', 'hypothesis'))


def mapping(local, mode, region):
    """Return axis-aligned decoder-local to viewer mapping (sx, sy, tx, ty)."""
    try:
        if mode == 'Source image':
            x0, y0 = observation_to_source(local.grid, local.observation, 0, 0)
            tr = local.grid.heatmap_transform
            result = (1/tr.scale_x, 1/tr.scale_y, x0, y0)
        elif region == 'Full canvas':
            box = getattr(local.observation, 'known_box', None)
            if box is None: return None
            result = (1., 1., box[0], box[1])
        else:
            result = (1., 1., 0., 0.)
        return result if all(math.isfinite(v) for v in result) and result[0] > 0 and result[1] > 0 else None
    except (AttributeError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None


def primitives(local, transform, viewport, layers, limit=10000):
    """Generate bounded, visible primitives in viewer coordinates.

    Equivalent boundaries are never clamped. Viewport bounds only limit line and
    center iteration; crossings represent one axis-level sample each.
    """
    if transform is None or viewport is None or limit <= 0:
        return ()
    try:
        sx, sy, tx, ty = transform
        vx0, vy0, vx1, vy1 = viewport
        if (not all(math.isfinite(v) for v in (*transform, *viewport)) or sx <= 0 or sy <= 0
                or vx1 <= vx0 or vy1 <= vy0): return ()
        x, y = local.hypothesis.x, local.hypothesis.y
        axes = (x, y)
        if any(not math.isfinite(a.origin) or not math.isfinite(a.pitch) or a.pitch <= 0
               or type(a.count) is not int or not 0 < a.count <= 1_000_000
               or not math.isfinite(a.end) for a in axes): return ()
        box = local.observation.box
        w, h = box[2]-box[0], box[3]-box[1]
        if not all(math.isfinite(v) and v > 0 for v in (w, h)): return ()
        lx0, ly0, lx1, ly1 = (vx0-tx)/sx, (vy0-ty)/sy, (vx1-tx)/sx, (vy1-ty)/sy
        if not all(math.isfinite(v) for v in (lx0, ly0, lx1, ly1)): return ()
        def point(a, b): return (tx + a*sx, ty + b*sy)
        def indices(axis, lo, hi, center=False):
            shift = .5 if center else 0.
            lower = max(0, math.ceil((lo-axis.origin)/axis.pitch-shift))
            upper = min(axis.count-1 if center else axis.count, math.floor((hi-axis.origin)/axis.pitch-shift))
            return range(lower, max(lower, upper+1))
        result = []
        regular_limit = max(0, (limit - 16)//2)
        def add(kind, *coords):
            if len(result) >= limit or not all(math.isfinite(v) for v in coords):
                return
            if kind in ('lattice', 'boundary'):
                a, b, c, d = coords
                if a == c:
                    if not vx0 <= a <= vx1 or max(b, d) < vy0 or min(b, d) > vy1: return
                    coords = (a, max(b, vy0), c, min(d, vy1))
                else:
                    if not vy0 <= b <= vy1 or max(a, c) < vx0 or min(a, c) > vx1: return
                    coords = (max(a, vx0), b, min(c, vx1), d)
            elif not (vx0 <= coords[0] <= vx1 and vy0 <= coords[1] <= vy1):
                return
            result.append((kind, coords))
        if layers.get('local_lattice'):
            if y.end >= ly0 and y.origin <= ly1:
                for i in indices(x, lx0, lx1):
                    if len(result) >= regular_limit: break
                    px = x.origin+i*x.pitch
                    add('lattice', *point(px, y.origin), *point(px, y.end))
            if x.end >= lx0 and x.origin <= lx1:
                for i in indices(y, ly0, ly1):
                    if len(result) >= regular_limit: break
                    py = y.origin+i*y.pitch
                    add('lattice', *point(x.origin, py), *point(x.end, py))
        if layers.get('local_centers'):
            columns = indices(x, lx0, lx1, True)
            rows = indices(y, ly0, ly1, True) if columns else ()
            center_start = len(result)
            for row in rows:
                if len(result) - center_start >= regular_limit: break
                for col in columns:
                    if len(result) - center_start >= regular_limit: break
                    add('center', *point(x.origin+(col+.5)*x.pitch, y.origin+(row+.5)*y.pitch))
        if layers.get('local_boundaries'):
            for a, b, c, d in ((x.origin,y.origin,x.origin,y.end), (x.end,y.origin,x.end,y.end),
                               (x.origin,y.origin,x.end,y.origin), (x.origin,y.end,x.end,y.end)):
                add('boundary', *point(a,b), *point(c,d))
        if layers.get('terminations'):
            for value, censored, horizontal in ((x.visible_start,x.start_censored,True), (x.visible_end,x.end_censored,True),
                                                 (y.visible_start,y.start_censored,False), (y.visible_end,y.end_censored,False)):
                if not censored and value is not None and math.isfinite(value):
                    add('crossing', *point(value, h/2) if horizontal else point(w/2, value))
        if layers.get('local_censoring'):
            for censored, a, b in ((x.start_censored,0,h/2), (x.end_censored,w,h/2),
                                    (y.start_censored,w/2,0), (y.end_censored,w/2,h)):
                if censored: add('censored', *point(a,b))
        return tuple(result)
    except (AttributeError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return ()
