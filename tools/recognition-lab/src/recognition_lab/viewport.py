"""Continuous source-image to logical viewport coordinates (no device scaling).

The image starts at its top-left boundary: pixel (0, 0) has center (.5, .5).
Model input and grid crop transforms belong to preprocessing/recognition; logical
desktop mapping belongs to capture, and physical surface scaling to the viewer.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ViewportTransform:
    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.scale, self.offset_x, self.offset_y)) or self.scale <= 0:
            raise ValueError('viewport transform must have a positive finite scale and finite offsets')

    def to_viewport(self, x, y):
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y

    def to_image(self, x, y):
        return (x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale

    def panned(self, dx, dy):
        return ViewportTransform(self.scale, self.offset_x + dx, self.offset_y + dy)

    def scaled_at(self, scale, x, y):
        ix, iy = self.to_image(x, y)
        return ViewportTransform(scale, x - ix * scale, y - iy * scale)

    @classmethod
    def fit(cls, image_width, image_height, viewport_width, viewport_height):
        dimensions = (image_width, image_height, viewport_width, viewport_height)
        if not all(math.isfinite(v) and v > 0 for v in dimensions):
            raise ValueError('image and viewport dimensions must be positive and finite')
        scale = min(viewport_width / image_width, viewport_height / image_height)
        return cls(scale, (viewport_width - image_width * scale) / 2,
                   (viewport_height - image_height * scale) / 2)


# Physical rendering pixels per source pixel; fit is allowed outside this range.
MIN_MAGNIFICATION = 0.01
MAX_MAGNIFICATION = 64.0
WHEEL_STEP_FACTOR = 1.2
# A 120-pixel vertical scroll has the same effect as one 120-unit wheel step.
PIXELS_PER_ZOOM_STEP = 120.0


def _positive_finite(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return value


def physical_scale(logical_scale, dpr):
    result = _positive_finite(logical_scale, 'logical scale') * _positive_finite(dpr, 'DPR')
    return _positive_finite(result, 'physical scale')


def logical_scale(magnification, dpr):
    result = _positive_finite(magnification, 'magnification') / _positive_finite(dpr, 'DPR')
    return _positive_finite(result, 'logical scale')


def clamp_magnification(current, requested):
    """Allow an exceptional fit scale to move only toward the ordinary range.

    Outward zoom is a no-op; inward zoom advances normally until it enters the
    range. A large step may pass into the range, but cannot cross its far limit.
    """
    _positive_finite(current, 'current magnification')
    _positive_finite(requested, 'requested magnification')
    if current < MIN_MAGNIFICATION:
        return max(current, min(requested, MAX_MAGNIFICATION))
    if current > MAX_MAGNIFICATION:
        return min(current, max(requested, MIN_MAGNIFICATION))
    return min(MAX_MAGNIFICATION, max(MIN_MAGNIFICATION, requested))


def zoom_by_factor(transform, dpr, factor, anchor):
    """Bound a multiplicative zoom before converting back to logical scale."""
    current = physical_scale(transform.scale, dpr)
    _positive_finite(factor, 'zoom factor')
    # Compare in log space so extreme but finite event values cannot overflow.
    target_log = math.log(current) + math.log(factor)
    low = min(math.log(current), math.log(MIN_MAGNIFICATION))
    high = max(math.log(current), math.log(MAX_MAGNIFICATION))
    requested = math.exp(max(low, min(high, target_log)))
    target = clamp_magnification(current, requested)
    if target == current or math.isclose(target, current, rel_tol=1e-15):
        return transform
    return transform.scaled_at(logical_scale(target, dpr), *anchor)


def wheel_factor(pixel_y, angle_y):
    """Use vertical pixel deltas when present; otherwise 120 angle units per step."""
    if not math.isfinite(pixel_y) or not math.isfinite(angle_y):
        raise ValueError('wheel deltas must be finite')
    steps = pixel_y / PIXELS_PER_ZOOM_STEP if pixel_y else angle_y / 120.0
    # Keep factor finite and positive even for implausibly large input.
    return math.exp(max(-700.0, min(700.0, steps * math.log(WHEEL_STEP_FACTOR))))


def reanchor_viewport(transform, old_size, new_size, old_dpr, new_dpr):
    """Preserve the old center's source coordinate and physical magnification."""
    if not all(math.isfinite(v) and v > 0 for v in (*old_size, *new_size)):
        raise ValueError('viewport dimensions must be positive and finite')
    center = transform.to_image(old_size[0] / 2, old_size[1] / 2)
    scale = logical_scale(physical_scale(transform.scale, old_dpr), new_dpr)
    return ViewportTransform(scale, new_size[0] / 2 - center[0] * scale,
                             new_size[1] / 2 - center[1] * scale)


PIXEL_GRID_MIN_MAGNIFICATION = 8.0


def visible_pixel_grid(transform, image_size, viewport_size, dpr):
    """Return clipped visible bounds and inclusive integer grid-line ranges, or None."""
    if physical_scale(transform.scale, dpr) < PIXEL_GRID_MIN_MAGNIFICATION:
        return None
    iw, ih = image_size
    vw, vh = viewport_size
    if min(iw, ih, vw, vh) <= 0:
        return None
    left, top = transform.to_image(0, 0)
    right, bottom = transform.to_image(vw, vh)
    left, top = max(0.0, left), max(0.0, top)
    right, bottom = min(float(iw), right), min(float(ih), bottom)
    if left >= right or top >= bottom:
        return None
    return ((left, top, right, bottom),
            (math.ceil(left), math.floor(right)),
            (math.ceil(top), math.floor(bottom)))
