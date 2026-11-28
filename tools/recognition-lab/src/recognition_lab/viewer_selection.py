"""Source-space selection and logical-pointer policies, independent of Qt."""
import math


def valid_bounds(bounds):
    try:
        x1, y1, x2, y2 = bounds
        if all(math.isfinite(v) for v in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1:
            return (x1, y1, x2, y2)
    except (TypeError, ValueError):
        pass
    return None


def image_point(transform, point, image_size):
    if image_size is None:
        return None
    x, y = transform.to_image(*point)
    if math.isfinite(x) and math.isfinite(y) and 0 <= x < image_size[0] and 0 <= y < image_size[1]:
        return x, y
    return None


def hit_detection(snapshot, transform, point, image_size, detections_visible):
    """Smallest containing visible box wins; original detection order breaks ties."""
    source = image_point(transform, point, image_size)
    if snapshot is None or source is None or not detections_visible:
        return None
    x, y = source
    candidates = []
    for index, detection in enumerate(snapshot.result.detections):
        box = valid_bounds(detection.box)
        if box is not None and box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            candidates.append(((box[2]-box[0]) * (box[3]-box[1]), index))
    return snapshot.detection_id(min(candidates)[1]) if candidates else None


def geometry_extent(grid):
    """Existing screenshot-space hypothesis edges; never derive extent from lattice."""
    if grid.hypotheses:
        xs, ys = grid.hypotheses[0][:2]
        if len(xs) and len(ys):
            return valid_bounds((xs[0], ys[0], xs[-1], ys[-1]))
    return None


def observation_source_bounds(grid, observation):
    """Map the known observation rectangle from crop space to screenshot pixels."""
    box = getattr(observation, 'box', None)
    transform = grid.heatmap_transform
    if box is None or transform is None:
        return None
    try:
        x0, y0, x1, y1 = box
        a = observation_to_source(grid, observation, 0, 0)
        b = observation_to_source(grid, observation, x1-x0, y1-y0)
        return valid_bounds((*a, *b))
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return None


def observation_to_source(grid, observation, x, y):
    """Map decoder-local coordinates through the actual crop transform."""
    tr = grid.heatmap_transform
    left, top = grid.crop.box[:2]
    ox, oy = observation.box[:2]
    return (left + (ox + x - tr.offset[0])/tr.scale_x,
            top + (oy + y - tr.offset[1])/tr.scale_y)


def selection_region(snapshot, identity, layers=None):
    """Return screenshot bounds, optionally requiring a corresponding visible layer."""
    if snapshot is None or snapshot.node(identity) is None:
        return None
    if identity.kind == 'detection':
        if layers is not None and not layers.get('detections'):
            return None
        return valid_bounds(snapshot.resolve(identity).box)
    grid = snapshot.grid(identity)
    if grid is None:
        return None
    if identity.kind in ('observation', 'hypothesis'):
        if layers is not None and not layers.get('tiles'):
            return None
        return observation_source_bounds(grid, snapshot.observation(identity))
    extent = geometry_extent(grid)
    phase = grid.phase_data if isinstance(grid.phase_data, dict) else None
    global_phase = phase.get('global') if phase else None
    board_visible = (layers is None or layers.get('boundaries') or layers.get('centers')
                     or (layers.get('cell_support') and getattr(global_phase, 'cell_occupancy', None) is not None))
    if extent is not None and board_visible:
        return extent
    if layers is None or layers.get('crops'):
        return valid_bounds(grid.crop.box)
    return None


def is_drag(start, point, threshold=4.0):
    """Threshold and coordinates are logical pixels; crossing is irreversible per gesture."""
    return math.hypot(point[0] - start[0], point[1] - start[1]) >= threshold


def bracket_dimensions(magnification, dpr):
    """10, 3 and 1.5 logical pixels expressed in source coordinates."""
    if not all(math.isfinite(v) and v > 0 for v in (magnification, dpr)):
        raise ValueError('magnification and DPR must be positive and finite')
    factor = dpr / magnification
    return 10 * factor, 3 * factor, 1.5 * factor
