"""Global phase-lattice and occupancy geometry for whole and tiled inputs.

Coordinates are continuous, with pixel centers at (i + .5, j + .5).
All thresholds are provisional geometric heuristics, not probabilities.
"""
from dataclasses import dataclass, field
import math
import numpy as np

from .tiles import consensus


@dataclass(frozen=True)
class GlobalSettings:
    # Require the termination to lie within .35 cell of a lattice boundary;
    # similarly reject disagreement among independent tile observations.
    max_termination_residual: float = .35
    min_coverage: float = .90
    min_cell_occupancy: float = .50
    min_cell_fraction: float = .95
    min_strip_fraction: float = .75
    min_inner_occupancy: float = .70
    max_outer_occupancy: float = .30
    max_lattice_residual: float = .20
    strategy: str = 'd7.5'
    rectangular: 'RectangularSettings' = field(default_factory=lambda: RectangularSettings())

    def __post_init__(self):
        if self.strategy not in ('d7.5', 'd7.6'):
            raise ValueError('unknown global recognition strategy')


@dataclass(frozen=True)
class RectangularSettings:
    """Provisional, uncalibrated experimental safeguards.

    Runs are limited only when both an absolute and a side-relative limit are
    exceeded, allowing isolated errors while rejecting structured continuation.
    """
    min_coverage: float = .90
    min_cell_fraction: float = .95
    min_strip_fraction: float = .75
    min_interior_occupancy: float = .70
    min_explained_fraction: float = .90
    min_transition: float = .40
    min_boundary_margin: float = .10
    max_exterior_fraction: float = .10
    max_conflict_run: int = 2
    max_conflict_run_fraction: float = .10
    max_termination_residual: float = .35
    max_cells: int = 200_000
    max_search_work: int = 20_000_000
    retain_fusion_diagnostics: bool = False

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == 'retain_fusion_diagnostics':
                if type(value) is not bool:
                    raise ValueError(f'{name} must be boolean')
            elif name in ('max_conflict_run', 'max_cells', 'max_search_work'):
                if type(value) is not int or value < 1:
                    raise ValueError(f'{name} must be a positive integer')
            elif not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be finite and in [0, 1]')


@dataclass(frozen=True)
class GlobalGeometry:
    x_edges: np.ndarray
    y_edges: np.ndarray
    x_indices: tuple[int, int]
    y_indices: tuple[int, int]

    @property
    def columns(self): return len(self.x_edges) - 1

    @property
    def rows(self): return len(self.y_edges) - 1


@dataclass(frozen=True)
class GlobalResult:
    lattice: object
    geometry: GlobalGeometry | None
    accepted: bool
    reasons: tuple[str, ...]
    occupancy: np.ndarray
    weights: np.ndarray
    cell_occupancy: np.ndarray | None = None
    cell_coverage: np.ndarray | None = None
    boundaries: dict = field(default_factory=dict)
    measurements: dict = field(default_factory=dict)
    strategy: str = 'd7.5'
    experimental: dict = field(default_factory=dict)
    incomplete: bool = False

    @property
    def status(self): return 'accepted' if self.accepted else ('incomplete' if self.incomplete else 'rejected')


def fuse_occupancy(observations, shape, *, retain_diagnostics=False):
    """Weighted mean of sigmoid probabilities; uncovered pixels remain NaN.

    Only each tile's real (unpadded) rectangle contributes. A positive taper
    downweights tile margins without losing coverage at the outer crop edge.
    """
    height, width = shape
    total = np.zeros(shape, dtype=np.float64)
    weights = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.uint32) if retain_diagnostics else None
    second = np.zeros(shape, dtype=np.float64) if retain_diagnostics else None
    for observation in observations:
        if observation.fields is None:
            continue
        x0, y0, x1, y1 = observation.box
        values = np.asarray(observation.fields)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height and values.shape == (5, y1-y0, x1-x0)):
            raise ValueError('tile fields and valid rectangle differ')
        occ = values[4]
        if not np.isfinite(occ).all() or np.any((occ < 0) | (occ > 1)):
            raise ValueError('expected finite occupancy probabilities')
        h, w = occ.shape
        def taper(n):
            d = np.minimum(np.arange(n) + 1, np.arange(n, 0, -1))
            return np.minimum(1., d / 32.)
        weight = taper(h)[:, None] * taper(w)[None, :]
        total[y0:y1, x0:x1] += occ * weight
        weights[y0:y1, x0:x1] += weight
        if retain_diagnostics:
            count[y0:y1, x0:x1] += 1
            second[y0:y1, x0:x1] += occ.astype(np.float64)**2 * weight
    occupancy = np.full(shape, np.nan, dtype=np.float32)
    np.divide(total, weights, out=occupancy, where=weights > 0)
    if not retain_diagnostics:
        return occupancy, weights
    mean = np.zeros(shape, dtype=np.float64)
    np.divide(total, weights, out=mean, where=weights > 0)
    variance = np.full(shape, np.nan, dtype=np.float64)
    np.divide(second, weights, out=variance, where=weights > 0)
    variance[weights > 0] = np.maximum(0., variance[weights > 0] - mean[weights > 0]**2)
    return occupancy, weights, {'observation_count': count, 'weighted_variance': variance,
                                'agreement_observable': count >= 2}


def _integrals(occupancy, weights, *, dtype=None):
    valid = weights > 0
    def integral(a): return np.pad(a, ((1, 0), (1, 0))).cumsum(0, dtype=dtype).cumsum(1, dtype=dtype)
    return integral(np.where(valid, occupancy, 0.)), integral(valid.astype(float))


def _region(integrals, box):
    """Mean probability and coverage in a continuous rectangle of pixel centers."""
    sums, counts = integrals
    h, w = sums.shape[0]-1, sums.shape[1]-1
    x0, y0, x1, y1 = box
    # Pixel i belongs to the region when its center i+.5 lies inside.
    a, b, c, d = (math.ceil(x0-.5), math.ceil(y0-.5), math.ceil(x1-.5), math.ceil(y1-.5))
    expected = max(c-a, 0) * max(d-b, 0)
    a, c = np.clip([a, c], 0, w); b, d = np.clip([b, d], 0, h)
    if not expected or c <= a or d <= b:
        return None, 0.
    def rect(v): return float(v[d, c]-v[b, c]-v[d, a]+v[b, a])
    count = rect(counts)
    return (rect(sums) / count if count else None), count / expected


def _termination(observations, indices, axis, leading, pitch):
    samples, confidence = [], []
    for i in indices:
        o = observations[i]
        h = o.hypothesis
        extent = h.x if axis == 'x' else h.y
        value = extent.visible_start if leading else extent.visible_end
        censored = extent.start_censored if leading else extent.end_censored
        if censored or value is None:
            continue
        samples.append(float(value + (o.box[0] if axis == 'x' else o.box[1])))
        evidence = h.x_evidence if axis == 'x' else h.y_evidence
        confidence.append(max(.01, float(evidence.coherence * evidence.magnitude)))
    if not samples:
        return {'count': 0, 'estimate': None, 'spread_cells': None, 'samples': ()}
    # Median resists a single anomalous tile; retain all values and reject
    # conflicting observations rather than silently deleting them.
    estimate = float(np.median(samples))
    spread = max(abs(v-estimate) for v in samples) / pitch
    return {'count': len(samples), 'estimate': estimate, 'spread_cells': spread,
            'samples': tuple(samples), 'confidence': tuple(confidence)}


def _recognize_baseline(observations, shape, settings, occupancy, weights):
    """Fit a lattice and validate a complete rectangular board from evidence."""
    lattice = consensus(observations)
    reasons, boundaries = [], {}
    if lattice is None:
        return GlobalResult(None, None, False, ('no valid local lattice',), occupancy, weights)
    if lattice.ambiguous:
        reasons.append('competing lattice hypotheses')
    if any(observations[i].local_valid for i in lattice.rejected):
        reasons.append('incompatible valid tile lattice')
    if lattice.residual_x > settings.max_lattice_residual*lattice.pitch_x or lattice.residual_y > settings.max_lattice_residual*lattice.pitch_y:
        reasons.append('global lattice residual exceeds tolerance')
    for name, axis, leading, pitch, origin in (
        ('left', 'x', True, lattice.pitch_x, lattice.origin_x),
        ('right', 'x', False, lattice.pitch_x, lattice.origin_x),
        ('top', 'y', True, lattice.pitch_y, lattice.origin_y),
        ('bottom', 'y', False, lattice.pitch_y, lattice.origin_y)):
        term = _termination(observations, lattice.inliers, axis, leading, pitch)
        boundaries[name] = term
        if term['estimate'] is None:
            reasons.append(f'{name} termination unobservable')
            continue
        if term['spread_cells'] > settings.max_termination_residual:
            reasons.append(f'{name} termination observations conflict')
        index = round((term['estimate']-origin)/pitch)
        snapped = origin + index*pitch
        residual = abs(term['estimate']-snapped)/pitch
        term.update(index=index, snapped=snapped, residual_cells=residual)
        if residual > settings.max_termination_residual:
            reasons.append(f'{name} termination inconsistent with lattice')
    if any('index' not in boundaries[k] for k in ('left', 'right', 'top', 'bottom')):
        return GlobalResult(lattice, None, False, tuple(reasons), occupancy, weights, boundaries=boundaries)
    l, r, t, b = (boundaries[k]['index'] for k in ('left', 'right', 'top', 'bottom'))
    if r <= l or b <= t or (r-l)*(b-t) > 1_000_000:
        reasons.append('invalid or excessive board dimensions')
        return GlobalResult(lattice, None, False, tuple(reasons), occupancy, weights, boundaries=boundaries)
    xx = lattice.origin_x + np.arange(l, r+1)*lattice.pitch_x
    yy = lattice.origin_y + np.arange(t, b+1)*lattice.pitch_y
    geometry = GlobalGeometry(xx, yy, (l, r), (t, b))
    integral = _integrals(occupancy, weights)
    phase_coverage = np.zeros(shape, dtype=bool)
    for i in lattice.inliers:
        x0, y0, x1, y1 = observations[i].box
        phase_coverage[y0:y1, x0:x1] = True
    phase_integral = _integrals(phase_coverage.astype(float), phase_coverage.astype(float))
    cell = np.full((b-t, r-l), np.nan)
    coverage = np.zeros_like(cell)
    phase_known = np.zeros_like(cell, dtype=bool)
    for row in range(b-t):
        for col in range(r-l):
            box = (xx[col]+.2*lattice.pitch_x, yy[row]+.2*lattice.pitch_y,
                   xx[col]+.8*lattice.pitch_x, yy[row]+.8*lattice.pitch_y)
            value, cov = _region(integral, box)
            coverage[row, col] = cov
            _, phase_cov = _region(phase_integral, box)
            phase_known[row, col] = phase_cov >= settings.min_coverage
            if value is not None: cell[row, col] = value
    observed = coverage >= settings.min_coverage
    supported = observed & (cell >= settings.min_cell_occupancy)
    if not observed.all(): reasons.append('incomplete cell occupancy coverage')
    if not phase_known.all(): reasons.append('incomplete valid phase tile coverage')
    if float(supported.mean()) < settings.min_cell_fraction: reasons.append('insufficient rectangular cell support')
    if np.any(supported.mean(axis=0) < settings.min_strip_fraction): reasons.append('unsupported board column')
    if np.any(supported.mean(axis=1) < settings.min_strip_fraction): reasons.append('unsupported board row')
    if min(supported.shape) >= 2 and np.any(~supported[:-1, :-1] & ~supported[1:, :-1] & ~supported[:-1, 1:] & ~supported[1:, 1:]):
        reasons.append('cluster of unsupported cells')
    # Strips on both sides are measured over the whole proposed side, excluding
    # corners. Outside coverage is required; an image edge is not background.
    px, py = lattice.pitch_x, lattice.pitch_y
    xlo, xhi = xx[0]+.2*px, xx[-1]-.2*px
    ylo, yhi = yy[0]+.2*py, yy[-1]-.2*py
    strips = {
        'left': ((xx[0]+.15*px,ylo,xx[0]+.45*px,yhi),(xx[0]-.45*px,ylo,xx[0]-.15*px,yhi)),
        'right': ((xx[-1]-.45*px,ylo,xx[-1]-.15*px,yhi),(xx[-1]+.15*px,ylo,xx[-1]+.45*px,yhi)),
        'top': ((xlo,yy[0]+.15*py,xhi,yy[0]+.45*py),(xlo,yy[0]-.45*py,xhi,yy[0]-.15*py)),
        'bottom': ((xlo,yy[-1]-.45*py,xhi,yy[-1]-.15*py),(xlo,yy[-1]+.15*py,xhi,yy[-1]+.45*py))}
    for name, (inside, outside) in strips.items():
        iv, ic = _region(integral, inside); ov, oc = _region(integral, outside)
        boundaries[name].update(inside_occupancy=iv, outside_occupancy=ov, inside_coverage=ic, outside_coverage=oc)
        # Inspect neighboring integer boundaries too. A neighbor cannot replace
        # the selected one without lying within the termination tolerance.
        is_x = name in ('left', 'right')
        pitch = px if is_x else py
        estimate = boundaries[name]['estimate']
        candidates = []
        for delta in (-1, 0, 1):
            def shift(box):
                return tuple(v + (delta*pitch if (j % 2 == 0) == is_x else 0) for j, v in enumerate(box))
            candidate_inner, inner_cov = _region(integral, shift(inside))
            candidate_outer, outer_cov = _region(integral, shift(outside))
            coordinate = boundaries[name]['snapped'] + delta*pitch
            residual = abs(coordinate-estimate)/pitch
            admissible = (residual <= settings.max_termination_residual and
                          inner_cov >= settings.min_coverage and outer_cov >= settings.min_coverage and
                          candidate_inner is not None and candidate_inner >= settings.min_inner_occupancy and
                          candidate_outer is not None and candidate_outer <= settings.max_outer_occupancy)
            candidates.append({'index': boundaries[name]['index']+delta, 'coordinate': coordinate,
                               'termination_residual_cells': residual, 'admissible': admissible,
                               'inside_occupancy': candidate_inner, 'outside_occupancy': candidate_outer,
                               'inside_coverage': inner_cov, 'outside_coverage': outer_cov})
        boundaries[name]['candidates'] = candidates
        if sum(c['admissible'] for c in candidates) != 1:
            reasons.append(f'{name} has no unique supported boundary candidate')
        # Also inspect each exterior cell strip, so a localized continuation is
        # not hidden by averaging over a long side.
        exterior_cells = []
        along_edges = yy if is_x else xx
        along_pitch = py if is_x else px
        for first in along_edges[:-1]:
            if is_x:
                segment = (outside[0], first+.2*along_pitch, outside[2], first+.8*along_pitch)
            else:
                segment = (first+.2*along_pitch, outside[1], first+.8*along_pitch, outside[3])
            value, cov = _region(integral, segment)
            if cov >= settings.min_coverage and value is not None:
                exterior_cells.append(value)
        exterior_supported = sum(v >= settings.min_cell_occupancy for v in exterior_cells)
        boundaries[name]['exterior_supported_cells'] = exterior_supported
        if exterior_supported:
            reasons.append(f'{name} exterior contains supported cells')
        if ic < settings.min_coverage or iv is None or iv < settings.min_inner_occupancy:
            reasons.append(f'{name} lacks interior occupancy support')
        if oc < settings.min_coverage or ov is None:
            reasons.append(f'{name} exterior is unobservable')
        elif ov > settings.max_outer_occupancy:
            reasons.append(f'{name} exterior occupancy indicates continuation')
    measurements = {'observed_cell_fraction': float(observed.mean()), 'supported_cell_fraction': float(supported.mean()),
                    'phase_covered_cell_fraction': float(phase_known.mean()),
                    'min_row_support': float(supported.mean(axis=1).min()), 'min_column_support': float(supported.mean(axis=0).min()),
                    'crop_coverage': float((weights > 0).mean())}
    return GlobalResult(lattice, geometry, not reasons, tuple(reasons), occupancy, weights, cell, coverage, boundaries, measurements)


def recognize_global(observations, shape, settings=GlobalSettings()):
    """Run the unchanged baseline or the explicit experimental strategy."""
    if settings.strategy == 'd7.5':
        occupancy, weights = fuse_occupancy(observations, shape)
        return _recognize_baseline(observations, shape, settings, occupancy, weights)
    if settings.strategy != 'd7.6':
        raise ValueError('unknown global recognition strategy')
    if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('experimental fusion shape must contain two positive integer dimensions')
    fused = fuse_occupancy(observations, shape, retain_diagnostics=settings.rectangular.retain_fusion_diagnostics)
    occupancy, weights = fused[:2]
    try:
        baseline = _recognize_baseline(observations, shape, settings, occupancy, weights)
    except (ValueError, OverflowError, FloatingPointError) as exc:
        return GlobalResult(None, None, False, (f'invalid lattice evidence: {exc}',), occupancy, weights,
                            strategy='d7.6', experimental={'fusion': fused[2] if len(fused) == 3 else None})
    from .rectangular_global import recognize_rectangular
    return recognize_rectangular(observations, baseline, settings, fused[2] if len(fused) == 3 else None)
