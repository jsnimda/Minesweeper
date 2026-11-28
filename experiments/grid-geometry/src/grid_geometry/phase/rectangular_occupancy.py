"""Independent lattice rectangle evidence, without physical-board acceptance.

Pixels occupy [x,x+1) x [y,y+1); array coordinates start at (0, 0).
The explicit domain is a continuous rectangle inside that array. Coverage is a
boolean per-pixel mask; uncovered occupancy is ignored (including NaNs), while
covered values must be finite probabilities. Repeated tile observations must be
fused before calling this module. Indices refer to the supplied lattice origins,
not to the first cell in the returned arrays.
"""
from dataclasses import dataclass, field
import math

import numpy as np


@dataclass(frozen=True)
class CellEvidence:
    x_start: int
    y_start: int
    x_edges: np.ndarray
    y_edges: np.ndarray
    full_area: float
    observed_area: np.ndarray
    integrated_occupancy: np.ndarray
    coverage: np.ndarray
    mean_occupancy: np.ndarray
    signed_evidence: np.ndarray
    # Reserved for caller-provided, aligned diagnostics such as observation
    # counts or disagreement. These do not change geometric evidence.
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Rectangle:
    left: int
    top: int
    right: int  # exclusive
    bottom: int  # exclusive
    score: float
    tie_count: int

    @property
    def has_positive_evidence(self):
        return self.score > 0


@dataclass(frozen=True)
class Alternative:
    status: str
    score: float | None = None
    score_difference: float | None = None  # alternative minus selected
    changed_observed_area: float | None = None
    changed_cell_coverage: float | None = None
    normalized_margin: float | None = None  # (selected - alternative) / coverage


@dataclass(frozen=True)
class RectangleMeasurements:
    rectangle: Rectangle
    coordinates: tuple[float, float, float, float]
    rows: int
    columns: int
    interior_coverage: float
    mean_interior_occupancy: float | None
    explained_occupancy_fraction: float | None
    unexplained_exterior_occupancy: float
    alternatives: dict[str, dict[str, Alternative]]


def _prefix(values):
    return np.pad(values, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)


def _at(prefix, xs, ys):
    """Integral at all Cartesian pairs, bilinear within each pixel."""
    h, w = prefix.shape[0] - 1, prefix.shape[1] - 1
    x = np.clip(xs, 0, w)
    y = np.clip(ys, 0, h)
    ix = np.minimum(np.floor(x).astype(int), w - 1)
    iy = np.minimum(np.floor(y).astype(int), h - 1)
    fx, fy = x - ix, y - iy
    a = prefix[iy[:, None], ix[None, :]]
    b = prefix[iy[:, None], (ix + 1)[None, :]]
    c = prefix[(iy + 1)[:, None], ix[None, :]]
    d = prefix[(iy + 1)[:, None], (ix + 1)[None, :]]
    return (a * (1 - fy[:, None]) * (1 - fx[None, :])
            + b * (1 - fy[:, None]) * fx[None, :]
            + c * fy[:, None] * (1 - fx[None, :])
            + d * fy[:, None] * fx[None, :])


def integrate_cells(occupancy, observed, *, pitch_x, pitch_y, origin_x, origin_y, domain, diagnostics=None):
    """Integrate every lattice cell intersecting domain, including partial cells.

    Domain is (left, top, right, bottom), in pixel-boundary coordinates.
    Uncovered values are ignored; covered invalid values raise ValueError.
    Diagnostics are retained untouched and do not participate in calculations.
    """
    values = np.asarray(occupancy, dtype=float)
    mask = np.asarray(observed)
    if values.ndim != 2 or not values.size or mask.shape != values.shape or mask.dtype != np.bool_:
        raise ValueError('occupancy and boolean coverage must be nonempty matching 2D arrays')
    if not np.isfinite(values[mask]).all() or np.any((values[mask] < 0) | (values[mask] > 1)):
        raise ValueError('observed occupancy must contain finite probabilities')
    if not all(math.isfinite(v) for v in (pitch_x, pitch_y, origin_x, origin_y)) or pitch_x <= 0 or pitch_y <= 0:
        raise ValueError('lattice origins must be finite and pitches finite and positive')
    if len(domain) != 4 or not all(math.isfinite(v) for v in domain):
        raise ValueError('domain must have four finite coordinates')
    left, top, right, bottom = domain
    h, w = values.shape
    if not (0 <= left < right <= w and 0 <= top < bottom <= h):
        raise ValueError('domain must be nonempty and inside the occupancy array')
    x0, x1 = math.floor((left-origin_x)/pitch_x), math.ceil((right-origin_x)/pitch_x)
    y0, y1 = math.floor((top-origin_y)/pitch_y), math.ceil((bottom-origin_y)/pitch_y)
    xe = origin_x + np.arange(x0, x1 + 1) * pitch_x
    ye = origin_y + np.arange(y0, y1 + 1) * pitch_y
    if not np.isfinite(xe).all() or not np.isfinite(ye).all() or np.any(np.diff(xe) <= 0) or np.any(np.diff(ye) <= 0):
        raise ValueError('lattice boundaries are not representable')
    xc, yc = np.clip(xe, left, right), np.clip(ye, top, bottom)
    def cells(data):
        integral = _at(_prefix(data), xc, yc)
        return integral[1:, 1:] - integral[:-1, 1:] - integral[1:, :-1] + integral[:-1, :-1]
    area = pitch_x * pitch_y
    if not math.isfinite(area) or area <= 0:
        raise ValueError('cell area must be finite and positive')
    observed_area = cells(mask.astype(float))
    occupied = cells(np.where(mask, values, 0.))
    # Prefix subtraction can leave roundoff in a truly empty region. Detect
    # that state with integer pixel counts, without rounding lattice edges.
    counts = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    xa, xb = np.floor(xc[:-1]).astype(int), np.ceil(xc[1:]).astype(int)
    ya, yb = np.floor(yc[:-1]).astype(int), np.ceil(yc[1:]).astype(int)
    covered_pixels = (counts[yb[:, None], xb[None, :]] - counts[ya[:, None], xb[None, :]]
                      - counts[yb[:, None], xa[None, :]] + counts[ya[:, None], xa[None, :]])
    observed_area[covered_pixels == 0] = 0.
    occupied[covered_pixels == 0] = 0.
    coverage = observed_area / area
    mean = np.full(observed_area.shape, np.nan)
    np.divide(occupied, observed_area, out=mean, where=observed_area > 0)
    return CellEvidence(x0, y0, xe, ye, area, observed_area, occupied, coverage, mean,
                        (2 * occupied - observed_area) / area, {} if diagnostics is None else diagnostics)


def maximum_sum_rectangle(evidence):
    """Exact nonempty maximum sum; lexicographically smallest (top,left,bottom,right).

    Equal scores use exact floating-point equality, with no tolerance. tie_count
    counts all equal maxima. Empty 2D matrices return None. The best rectangle
    remains available when its score is nonpositive; this is not acceptance.
    """
    a = np.asarray(evidence, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('evidence must be a finite 2D matrix')
    if not a.size:
        return None
    transposed = a.shape[0] > a.shape[1]
    work = a.T if transposed else a
    h, w = work.shape
    best = -math.inf
    chosen = None
    ties = 0
    for start in range(h):
        compressed = np.zeros(w)
        for end in range(start + 1, h + 1):
            compressed += work[end - 1]
            prefix = 0.
            minimum = 0.
            min_count = 1
            min_first = 0
            for j, value in enumerate(compressed, 1):
                prefix += float(value)
                score = prefix - minimum
                # The earliest minimum is lexicographically first in either orientation.
                bounds = ((min_first, start, j, end) if transposed
                          else (start, min_first, end, j))
                if score > best:
                    best, chosen, ties = score, bounds, min_count
                elif score == best:
                    ties += min_count
                    chosen = min(chosen, bounds)
                if prefix < minimum:
                    minimum, min_count, min_first = prefix, 1, j
                elif prefix == minimum:
                    min_count += 1
    top, left, bottom, right = chosen
    return Rectangle(left, top, right, bottom, float(best), ties)


def measure_rectangle(cells, rectangle):
    """Measure a matrix-relative rectangle and neighboring one-cell boundaries.

    Returned lattice indices are relative to the original supplied origins.
    Scores are sums of full-cell-normalized signed evidence. Missing coverage
    gives no termination evidence; undefined ratios are None.
    """
    if rectangle is None:
        return None
    l, t, r, b = rectangle.left, rectangle.top, rectangle.right, rectangle.bottom
    h, w = cells.signed_evidence.shape
    if not (0 <= l < r <= w and 0 <= t < b <= h):
        raise ValueError('rectangle is outside the cell domain')
    def total(a, box):
        x0, y0, x1, y1 = box
        return float(a[y0:y1, x0:x1].sum())
    box = (l, t, r, b)
    score = total(cells.signed_evidence, box)
    area = total(cells.observed_area, box)
    occupied = total(cells.integrated_occupancy, box)
    all_occupied = float(cells.integrated_occupancy.sum())
    alternatives = {}
    for side, inward, outward in (
        ('left', (l+1,t,r,b), (l-1,t,r,b)),
        ('right', (l,t,r-1,b), (l,t,r+1,b)),
        ('top', (l,t+1,r,b), (l,t-1,r,b)),
        ('bottom', (l,t,r,b-1), (l,t,r,b+1))):
        alternatives[side] = {}
        for direction, alt in (('inward', inward), ('outward', outward)):
            x0, y0, x1, y1 = alt
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                status = 'outside_domain' if direction == 'outward' else 'empty_rectangle'
                alternatives[side][direction] = Alternative(status)
                continue
            alt_score = total(cells.signed_evidence, alt)
            changed_area = abs(total(cells.observed_area, alt) - area)
            changed_coverage = changed_area / cells.full_area
            difference = alt_score - score
            alternatives[side][direction] = Alternative(
                'available' if changed_area > 0 else 'unobserved', alt_score, difference,
                changed_area, changed_coverage, -difference / changed_coverage if changed_coverage > 0 else None)
    absolute = Rectangle(l + cells.x_start, t + cells.y_start, r + cells.x_start,
                         b + cells.y_start, score, rectangle.tie_count)
    return RectangleMeasurements(absolute, (float(cells.x_edges[l]), float(cells.y_edges[t]),
                                 float(cells.x_edges[r]), float(cells.y_edges[b])), b-t, r-l,
                                 area / ((r-l)*(b-t)*cells.full_area), occupied / area if area > 0 else None,
                                 occupied / all_occupied if all_occupied > 0 else None,
                                 all_occupied - occupied, alternatives)
