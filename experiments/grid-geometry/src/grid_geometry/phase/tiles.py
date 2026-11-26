"""Conservative consensus of tile-local lattices in continuous image coordinates.

This module does not infer physical board terminations from tile edges.
The agreement tolerances are provisional and have not been calibrated.
"""
from dataclasses import dataclass
import numpy as np

from .decoder import decode_phase
from .validation import validate_phase


@dataclass(frozen=True)
class TileObservation:
    box: tuple[int, int, int, int]
    hypothesis: object
    local_valid: bool
    reasons: tuple[str, ...] = ()
    fields: np.ndarray | None = None


@dataclass(frozen=True)
class LatticeConsensus:
    pitch_x: float
    pitch_y: float
    origin_x: float
    origin_y: float
    inliers: tuple[int, ...]
    rejected: tuple[int, ...]
    ambiguous: bool
    residual_x: float = 0.
    residual_y: float = 0.


def periodic_distance(a, b, pitch):
    return abs((a - b + pitch / 2) % pitch - pitch / 2)


def tile_starts(length, tile, overlap):
    if length <= 0 or tile <= 0 or not 0 <= overlap < tile:
        raise ValueError('invalid tile layout')
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile - overlap))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def observe_tiles(rgb, tile, overlap, predict):
    """Decode each tile independently; ``predict`` returns activated fields.

    Right/bottom padding is discarded before decoding, so it cannot supply
    evidence for a physical termination.
    """
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
        raise ValueError('expected an RGB image at least two pixels in each dimension')
    height, width = image.shape[:2]
    xs, ys = tile_starts(width, tile, overlap), tile_starts(height, tile, overlap)
    observations = []
    for y in ys:
        for x in xs:
            part = image[y:min(y+tile, height), x:min(x+tile, width)]
            th, tw = part.shape[:2]
            padded = np.zeros((tile, tile, 3), dtype=image.dtype)
            padded[:th, :tw] = part
            fields = predict(padded)[:, :th, :tw]
            decoded = decode_phase(fields)
            checked = validate_phase(fields, decoded.hypotheses[0], decoder_status=decoded.status) if decoded.hypotheses else None
            observations.append(TileObservation((x, y, x+tw, y+th), decoded.hypotheses[0] if decoded.hypotheses else None,
                                                bool(checked and checked.local_accepted),
                                                checked.local_reasons if checked else decoded.reasons, fields))
    return observations, (len(xs), len(ys))


def consensus(observations, *, pitch_tolerance=.05, phase_tolerance=.20):
    """Select the largest pairwise-compatible seed cluster, retaining conflicts.

    Phase origins are global boundary coordinates modulo pitch; they cannot be
    averaged until unwrapped against a common reference. No fields are averaged.
    """
    valid = [(i, o) for i, o in enumerate(observations) if o.local_valid]
    if not valid:
        return None
    def axes(o):
        h = o.hypothesis
        result = []
        for extent, start, end in ((h.x, o.box[0], o.box[2]), (h.y, o.box[1], o.box[3])):
            origin = extent.origin + start
            anchor = origin + round(((start + end) / 2 - origin) / extent.pitch) * extent.pitch
            result.append((extent.pitch, anchor))
        return result
    def compatible(a, b):
        for (p, x), (q, y) in zip(axes(a), axes(b)):
            # Compare phases at the midpoint; extrapolating each fitted origin
            # from its own tile to coordinate zero amplifies pitch error.
            middle = (x + y) / 2
            xa = x + round((middle-x)/p)*p
            yb = y + round((middle-y)/q)*q
            if abs(p - q) / min(p, q) > pitch_tolerance or periodic_distance(xa, yb, (p + q) / 2) > phase_tolerance * min(p, q):
                return False
        return True
    clusters = [[(j, b) for j, b in valid if compatible(a, b)] for _, a in valid]
    clusters.sort(key=len, reverse=True)
    chosen = clusters[0]
    # A conflicting cluster with the same support is not a unique solution.
    chosen_ids = {i for i, _ in chosen}
    ambiguous = any(len(c) == len(chosen) and {i for i, _ in c} != chosen_ids for c in clusters[1:])
    result, residuals = [], []
    for axis in range(2):
        pairs = [axes(o)[axis] for _, o in chosen]
        pitch = float(np.median([p for p, _ in pairs]))
        reference = pairs[0][1]
        positions = np.array([x for _, x in pairs])
        indices = np.rint((positions-reference)/pitch)
        # Long-baseline fit is useful only when the span constrains pitch;
        # local phase observations alone cannot resolve an integer alias.
        if len(pairs) >= 3 and np.ptp(indices) >= 4:
            confidence = [max(.01, (o.hypothesis.x_evidence if axis == 0 else o.hypothesis.y_evidence).coherence *
                              (o.hypothesis.x_evidence if axis == 0 else o.hypothesis.y_evidence).magnitude)
                          for _, o in chosen]
            fitted_pitch, fitted_origin = np.polyfit(indices, positions, 1, w=np.sqrt(confidence))
            if abs(fitted_pitch-pitch) <= pitch_tolerance*pitch:
                pitch = float(fitted_pitch)
                reference = float(fitted_origin)
        aligned = positions - np.rint((positions-reference)/pitch)*pitch
        origin = float(np.median(aligned))
        residuals.append(float(np.max(np.abs(aligned-origin))))
        result.extend((pitch, origin % pitch))
    indices = tuple(i for i, _ in chosen)
    return LatticeConsensus(result[0], result[2], result[1], result[3], indices,
                            tuple(i for i in range(len(observations)) if i not in indices), ambiguous,
                            residuals[0], residuals[1])
