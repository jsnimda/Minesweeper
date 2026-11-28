"""Experimental global rectangle interpretation; no model inference occurs here.

All coordinates are in the fused array / observation-box space, which may be
resized crop space. The caller owns any transform to native or screen pixels.
"""
import math

import numpy as np

from .global_geometry import GlobalGeometry, GlobalResult, _integrals, _region
from .rectangular_occupancy import integrate_cells, maximum_sum_rectangle, measure_rectangle


def _longest(flags):
    longest = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return longest


def _segments(integral, geometry, side, settings, baseline_settings):
    """Cell-sized strips; unknown gaps join possible conflict runs, never break them.

    A supported exterior segment has outer mean >= min_cell_occupancy. A
    conflicting segment has a supported exterior or transition below the
    configured minimum. Unknown segments are counted in the conservative
    possible conflict run but excluded from observed statistics.
    """
    x, y = geometry.x_edges, geometry.y_edges
    px, py = x[1]-x[0], y[1]-y[0]
    vertical = side in ('left', 'right')
    edge = {'left': x[0], 'right': x[-1], 'top': y[0], 'bottom': y[-1]}[side]
    sign = 1 if side in ('left', 'top') else -1
    along = y if vertical else x
    pitch = px if vertical else py
    segments = []
    for lo, hi in zip(along[:-1], along[1:]):
        a, b = lo + .2*(hi-lo), hi - .2*(hi-lo)
        def box(d0, d1):
            low, high = sorted((edge + sign*d0*pitch, edge + sign*d1*pitch))
            return (low, a, high, b) if vertical else (a, low, b, high)
        inside, ic = _region(integral, box(.15, .45))
        outside, oc = _region(integral, box(-.45, -.15))
        observable = inside is not None and outside is not None and min(ic, oc) >= settings.min_coverage
        transition = inside - outside if observable else None
        supported = outside >= baseline_settings.min_cell_occupancy if observable else None
        conflict = (supported or transition < settings.min_transition) if observable else None
        segments.append({'range': (float(lo), float(hi)), 'interior_occupancy': inside,
                         'exterior_occupancy': outside, 'interior_coverage': ic,
                         'exterior_coverage': oc, 'observable': observable,
                         'transition': transition, 'exterior_supported': supported, 'conflict': conflict})
    known = [s for s in segments if s['observable']]
    transitions = [s['transition'] for s in known]
    support = [s['exterior_supported'] for s in known]
    return {'segments': segments, 'observable_count': len(known),
            'observable_fraction': len(known)/len(segments),
            'mean_transition': float(np.mean(transitions)) if transitions else None,
            'median_transition': float(np.median(transitions)) if transitions else None,
            'lower_transition_quantile_10': float(np.quantile(transitions, .10)) if transitions else None,
            'exterior_supported_count': sum(support),
            'exterior_supported_fraction': sum(support)/len(known) if known else None,
            'longest_supported_run': _longest([s['exterior_supported'] is True for s in segments]),
            'longest_conflict_run': _longest([s['conflict'] is True for s in segments]),
            'longest_possible_conflict_run': _longest([s['conflict'] is not False for s in segments])}


def _termination_agreement(observations, lattice, side, coordinate, pitch, span, limit):
    vertical = side in ('left', 'right')
    leading = side in ('left', 'top')
    samples = []
    for i in lattice.inliers:
        o = observations[i]
        extent = o.hypothesis.x if vertical else o.hypothesis.y
        censored = extent.start_censored if leading else extent.end_censored
        value = extent.visible_start if leading else extent.visible_end
        along = (o.box[1], o.box[3]) if vertical else (o.box[0], o.box[2])
        overlap = (max(span[0], along[0]), min(span[1], along[1]))
        if censored or value is None or overlap[0] >= overlap[1]:
            continue
        coordinate_sample = float(value + (o.box[0] if vertical else o.box[1]))
        residual = (coordinate_sample-coordinate)/pitch
        samples.append({'observation_index': i, 'coordinate': coordinate_sample,
                        'residual_cells': residual, 'side_interval': overlap,
                        'outlier': abs(residual) > limit})
    residuals = [s['residual_cells'] for s in samples]
    median = float(np.median(residuals)) if residuals else None
    intervals = sorted(s['side_interval'] for s in samples)
    covered = 0.
    if intervals:
        lo, hi = intervals[0]
        for a, b in intervals[1:]:
            if a > hi:
                covered += hi-lo
                lo, hi = a, b
            else:
                hi = max(hi, b)
        covered += hi-lo
    return {'samples': samples, 'count': len(samples), 'median_residual_cells': median,
            'max_abs_residual_cells': max((abs(v) for v in residuals), default=None),
            'median_absolute_deviation_cells': float(np.median(np.abs(np.asarray(residuals)-median))) if residuals else None,
            'outlier_count': sum(s['outlier'] for s in samples),
            'side_interval_union_fraction': covered/(span[1]-span[0])}


def _comparison(baseline, evidence, measurement):
    result = {'available': baseline.geometry is not None, 'experimental': measurement.coordinates,
              'experimental_dimensions': (measurement.rows, measurement.columns),
              'experimental_score': measurement.rectangle.score}
    if baseline.geometry is None:
        return result
    geom = baseline.geometry
    coords = (float(geom.x_edges[0]), float(geom.y_edges[0]), float(geom.x_edges[-1]), float(geom.y_edges[-1]))
    indices = (geom.x_indices[0], geom.y_indices[0], geom.x_indices[1], geom.y_indices[1])
    rr = measurement.rectangle
    experimental_indices = (rr.left, rr.top, rr.right, rr.bottom)
    differences = tuple(a-b for a, b in zip(measurement.coordinates, coords))
    x0, y0, x1, y1 = (indices[0]-evidence.x_start, indices[1]-evidence.y_start,
                      indices[2]-evidence.x_start, indices[3]-evidence.y_start)
    h, w = evidence.signed_evidence.shape
    score = float(evidence.signed_evidence[y0:y1, x0:x1].sum()) if 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h else None
    result.update(baseline=coords, baseline_dimensions=(geom.rows, geom.columns), baseline_score=score,
                  difference_pixels=differences,
                  difference_cells=tuple(a-b for a, b in zip(experimental_indices, indices)),
                  extents_agree=indices == experimental_indices)
    return result


def recognize_rectangular(observations, baseline, settings, fusion_diagnostics):
    cfg = settings.rectangular
    lattice = baseline.lattice
    diag = {'fusion': fusion_diagnostics, 'baseline_reasons': baseline.reasons,
            'baseline_accepted': baseline.accepted}
    def failed(reason, *, incomplete=True):
        return GlobalResult(lattice, None, False, (reason,), baseline.occupancy, baseline.weights,
                            strategy='d7.6', experimental=diag, incomplete=incomplete)
    if lattice is None:
        return failed('no valid local lattice')
    numbers = (lattice.pitch_x, lattice.pitch_y, lattice.origin_x, lattice.origin_y)
    if not all(math.isfinite(v) for v in numbers) or min(lattice.pitch_x, lattice.pitch_y) <= 0:
        return failed('invalid lattice', incomplete=False)
    h, w = baseline.occupancy.shape
    # Check index range, allocation and search effort before constructing arrays.
    try:
        x0, x1 = math.floor(-lattice.origin_x/lattice.pitch_x), math.ceil((w-lattice.origin_x)/lattice.pitch_x)
        y0, y1 = math.floor(-lattice.origin_y/lattice.pitch_y), math.ceil((h-lattice.origin_y)/lattice.pitch_y)
        columns, rows = x1-x0, y1-y0
        if (min(rows, columns) < 1 or max(abs(x0), abs(x1), abs(y0), abs(y1)) > 2**52
                or rows*columns > cfg.max_cells or min(rows, columns)**2*max(rows, columns) > cfg.max_search_work):
            return failed('excessive lattice evidence domain', incomplete=False)
        cells = integrate_cells(baseline.occupancy, baseline.weights > 0,
                                pitch_x=lattice.pitch_x, pitch_y=lattice.pitch_y,
                                origin_x=lattice.origin_x, origin_y=lattice.origin_y,
                                domain=(0., 0., float(w), float(h)))
        if not all(np.isfinite(a).all() for a in (cells.observed_area, cells.integrated_occupancy, cells.signed_evidence)):
            return failed('nonfinite rectangle evidence', incomplete=False)
        candidate = maximum_sum_rectangle(cells.signed_evidence)
        measured = measure_rectangle(cells, candidate)
    except (ValueError, OverflowError, FloatingPointError) as exc:
        return failed(f'invalid rectangle evidence: {exc}', incomplete=False)
    if measured is None:
        return failed('no rectangle evidence')
    r = measured.rectangle
    xs = lattice.origin_x + np.arange(r.left, r.right+1)*lattice.pitch_x
    ys = lattice.origin_y + np.arange(r.top, r.bottom+1)*lattice.pitch_y
    geom = GlobalGeometry(xs, ys, (r.left, r.right), (r.top, r.bottom))
    diag.update(cell_evidence=cells, rectangle=measured,
                comparison=_comparison(baseline, cells, measured))
    reasons, incomplete = [], False
    if lattice.ambiguous: reasons.append('competing lattice hypotheses')
    if any(observations[i].local_valid for i in lattice.rejected): reasons.append('incompatible valid tile lattice')
    if lattice.residual_x > settings.max_lattice_residual*lattice.pitch_x or lattice.residual_y > settings.max_lattice_residual*lattice.pitch_y:
        reasons.append('global lattice residual exceeds tolerance')
    if not r.has_positive_evidence: reasons.append('no positive rectangle evidence')
    l, t, right, bottom = candidate.left, candidate.top, candidate.right, candidate.bottom
    cov = cells.coverage[t:bottom, l:right]
    means = cells.mean_occupancy[t:bottom, l:right]
    known = cov >= cfg.min_coverage
    supported = known & (means >= settings.min_cell_occupancy)
    if not known.all():
        reasons.append('incomplete interior coverage'); incomplete = True
    if supported.mean() < cfg.min_cell_fraction or np.any(supported.mean(0) < cfg.min_strip_fraction) or np.any(supported.mean(1) < cfg.min_strip_fraction):
        reasons.append('insufficient rectangular cell support')
    if measured.mean_interior_occupancy is None or measured.mean_interior_occupancy < cfg.min_interior_occupancy:
        reasons.append('insufficient interior occupancy')
    if measured.explained_occupancy_fraction is None or measured.explained_occupancy_fraction < cfg.min_explained_fraction:
        reasons.append('unexplained exterior occupancy')
    integral = _integrals(baseline.occupancy, baseline.weights, dtype=np.float64)
    phase_coverage = np.zeros(baseline.occupancy.shape, dtype=bool)
    for i in lattice.inliers:
        x0, y0, x1, y1 = observations[i].box
        phase_coverage[y0:y1, x0:x1] = True
    phase_integral = _integrals(phase_coverage.astype(float), phase_coverage)
    phase_known = np.zeros((geom.rows, geom.columns), dtype=bool)
    for row in range(geom.rows):
        for col in range(geom.columns):
            box = (xs[col]+.2*lattice.pitch_x, ys[row]+.2*lattice.pitch_y,
                   xs[col]+.8*lattice.pitch_x, ys[row]+.8*lattice.pitch_y)
            _, coverage = _region(phase_integral, box)
            phase_known[row, col] = coverage >= cfg.min_coverage
    if not phase_known.all():
        reasons.append('incomplete valid phase tile coverage'); incomplete = True
    boundaries = {}
    for side, coordinate, pitch, span in (
        ('left', xs[0], lattice.pitch_x, (ys[0], ys[-1])), ('right', xs[-1], lattice.pitch_x, (ys[0], ys[-1])),
        ('top', ys[0], lattice.pitch_y, (xs[0], xs[-1])), ('bottom', ys[-1], lattice.pitch_y, (xs[0], xs[-1]))):
        segments = _segments(integral, geom, side, cfg, settings)
        term = _termination_agreement(observations, lattice, side, coordinate, pitch, span, cfg.max_termination_residual)
        boundaries[side] = {'continuity': segments, 'termination': term, 'alternatives': measured.alternatives[side]}
        if segments['observable_count'] != len(segments['segments']) or term['count'] == 0:
            reasons.append(f'{side} termination unobservable'); incomplete = True
        if term['outlier_count']:
            reasons.append(f'{side} termination observations conflict')
        n = len(segments['segments'])
        if segments['mean_transition'] is not None and segments['mean_transition'] < cfg.min_transition:
            reasons.append(f'{side} weak occupancy transition')
        if (segments['exterior_supported_fraction'] is not None
                and segments['exterior_supported_count'] > cfg.max_conflict_run
                and segments['exterior_supported_fraction'] > cfg.max_exterior_fraction):
            reasons.append(f'{side} excessive exterior support')
        if segments['longest_possible_conflict_run'] > cfg.max_conflict_run and segments['longest_possible_conflict_run']/n > cfg.max_conflict_run_fraction:
            reasons.append(f'{side} contiguous boundary conflict')
        for direction, alt in measured.alternatives[side].items():
            if alt.status in ('outside_domain', 'unobserved'):
                reasons.append(f'{side} {direction} boundary evidence unavailable'); incomplete = True
            # An inward alternative removes known board cells and still needs
            # nearly full coverage. Outward coverage may be partial: observed
            # strips and termination votes above establish observability, and
            # the margin below uses only genuinely observed changed area.
            elif direction == 'inward' and alt.status == 'available' and alt.changed_cell_coverage < cfg.min_coverage*n:
                reasons.append(f'{side} {direction} boundary evidence incomplete'); incomplete = True
            elif alt.status == 'available' and (alt.normalized_margin is None or alt.normalized_margin < cfg.min_boundary_margin):
                reasons.append(f'{side} ambiguous {direction} boundary')
    return GlobalResult(lattice, geom, not reasons, tuple(reasons), baseline.occupancy, baseline.weights,
                        means, cov, boundaries, {'observed_cell_fraction': float(known.mean()),
                                                'supported_cell_fraction': float(supported.mean()),
                                                'phase_covered_cell_fraction': float(phase_known.mean())},
                        strategy='d7.6', experimental=diag, incomplete=incomplete)
