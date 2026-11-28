"""Controlled, checkpoint-free global geometry fixtures."""
from dataclasses import replace
import math

import numpy as np
import pytest

from grid_geometry.phase.decoder import AxisExtent, FrequencyCandidate, GridHypothesis
from grid_geometry.phase.global_geometry import fuse_occupancy, recognize_global
from grid_geometry.dataset.targets import PhaseGeometry, phase_targets
from grid_geometry.phase.tiles import TileObservation, consensus, observe_tiles


def scene(columns=12, rows=8, pitch=16, origin=(24, 24), size=None, tile=96, step=64):
    left, top = origin
    right, bottom = left + columns*pitch, top + rows*pitch
    size = size or (int(right+24), int(bottom+24))
    width, height = size
    fields = phase_targets(size, PhaseGeometry(left, top, pitch, pitch, (left, top, right, bottom)))[0]
    def starts(n):
        if n <= tile: return [0]
        result = list(range(0, n-tile+1, step))
        if result[-1] != n-tile: result.append(n-tile)
        return result
    def axis(start, end, board_start, board_end):
        leading, trailing = board_start <= start, board_end >= end
        first = math.floor((start-board_start)/pitch) if leading else 0
        last = math.ceil((end-board_start)/pitch) if trailing else round((board_end-board_start)/pitch)
        return AxisExtent(board_start + first*pitch-start, pitch, last-first,
                          None if leading else board_start-start,
                          None if trailing else board_end-start, leading, trailing)
    observations = []
    for y in starts(height):
        for x in starts(width):
            x1, y1 = min(width, x+tile), min(height, y+tile)
            # Phase-valid tiles overlap the board; background tiles still supply
            # occupancy coverage without proposing a lattice.
            valid = min(x1, right)-max(x, left) >= 2*pitch and min(y1, bottom)-max(y, top) >= 2*pitch
            h = None
            if valid:
                ax, ay = axis(x, x1, left, right), axis(y, y1, top, bottom)
                h = GridHypothesis(ax, ay, FrequencyCandidate(pitch, ax.origin % pitch, 1, 1),
                                   FrequencyCandidate(pitch, ay.origin % pitch, 1, 1), 1, 0, 1)
            observations.append(TileObservation((x, y, x1, y1), h, valid, (), fields[:, y:y1, x:x1].copy()))
    return observations, (height, width)


def test_complete_rectangular_and_large_boards():
    for columns, rows in ((12, 8), (73, 41)):
        observations, shape = scene(columns, rows)
        result = recognize_global(observations, shape)
        assert result.accepted, result.reasons
        assert (result.geometry.columns, result.geometry.rows) == (columns, rows)
        assert all(result.boundaries[k]['count'] for k in ('left', 'right', 'top', 'bottom'))
        assert result.measurements['supported_cell_fraction'] == 1


def test_occupancy_fusion_coverage_and_padding():
    a = np.zeros((5, 4, 4), np.float32); a[4] = .2
    b = np.zeros((5, 4, 4), np.float32); b[4] = .8
    observations = [TileObservation((0, 0, 4, 4), None, False, fields=a),
                    TileObservation((2, 0, 6, 4), None, False, fields=b)]
    occ, weights = fuse_occupancy(observations, (5, 7))
    assert occ[1, 0] == pytest.approx(.2)
    assert .2 < occ[1, 2] < .8
    assert weights[1, 2] > 0 and weights[4, 0] == 0 and np.isnan(occ[4, 0])
    with pytest.raises(ValueError): fuse_occupancy([replace(observations[0], box=(0, 0, 3, 4))], (5, 7))


def test_tile_padding_is_not_observed():
    def predict(_):
        output = np.zeros((5, 32, 32), dtype=np.float32)
        output[4] = 1
        return output
    observations, layout = observe_tiles(np.zeros((12, 18, 3), np.uint8), 32, 8, predict)
    assert layout == (1, 1)
    assert observations[0].fields.shape == (5, 12, 18)
    _, weights = fuse_occupancy(observations, (16, 22))
    assert np.all(weights[:12, :18] > 0)
    assert not np.any(weights[12:, :])
    assert not np.any(weights[:, 18:])


def test_unobservable_conflicting_and_false_exterior():
    observations, shape = scene()
    no_left = [replace(o, hypothesis=replace(o.hypothesis, x=replace(o.hypothesis.x, start_censored=True, visible_start=None)))
               if o.local_valid else o for o in observations]
    result = recognize_global(no_left, shape)
    assert not result.accepted and 'left termination unobservable' in result.reasons
    first = next(i for i, o in enumerate(observations) if o.local_valid and o.hypothesis.x.visible_start is not None)
    o = observations[first]
    bad_h = replace(o.hypothesis, x=replace(o.hypothesis.x, visible_start=o.hypothesis.x.visible_start+8))
    conflict = observations.copy(); conflict[first] = replace(o, hypothesis=bad_h)
    assert not recognize_global(conflict, shape).accepted
    # An additional occupied exterior strip must never be mistaken for background.
    changed = []
    for o in observations:
        f = o.fields.copy(); x0, y0, x1, y1 = o.box
        xa, xb, ya, yb = max(x0, 12), min(x1, 24), max(y0, 32), min(y1, 136)
        if xa < xb and ya < yb:
            f[4, ya-y0:yb-y0, xa-x0:xb-x0] = 1
        changed.append(replace(o, fields=f))
    assert not recognize_global(changed, shape).accepted


def test_missing_coverage_negative_and_conflicting_lattice():
    assert not recognize_global([], (100, 100)).accepted
    observations, shape = scene()
    without_fields = [replace(o, fields=None) for o in observations]
    assert not recognize_global(without_fields, shape).accepted
    i = next(i for i, o in enumerate(observations) if o.local_valid)
    o = observations[i]
    bad = replace(o.hypothesis, x=replace(o.hypothesis.x, origin=o.hypothesis.x.origin + 7))
    observations[i] = replace(o, hypothesis=bad)
    assert not recognize_global(observations, shape).accepted


@pytest.mark.parametrize('side', ['left', 'top'])
def test_off_by_one_that_excludes_a_supported_strip_is_rejected(side):
    observations, shape = scene()
    changed = []
    for o in observations:
        h = o.hypothesis
        if h is not None:
            axis = h.x if side == 'left' else h.y
            if axis.visible_start is not None:
                axis = replace(axis, visible_start=axis.visible_start+16)
                h = replace(h, **({'x': axis} if side == 'left' else {'y': axis}))
        changed.append(replace(o, hypothesis=h))
    result = recognize_global(changed, shape)
    assert not result.accepted
    assert any('exterior' in reason for reason in result.reasons)


def test_internal_missing_cells_reject_complete_geometry():
    observations, shape = scene()
    changed = []
    for o in observations:
        f = o.fields.copy(); x0, y0, x1, y1 = o.box
        xa, xb, ya, yb = max(x0, 56), min(x1, 88), max(y0, 56), min(y1, 88)
        if xa < xb and ya < yb:
            f[4, ya-y0:yb-y0, xa-x0:xb-x0] = 0
        changed.append(replace(o, fields=f))
    result = recognize_global(changed, shape)
    assert not result.accepted and 'cluster of unsupported cells' in result.reasons


def test_long_range_pitch_fit_and_coordinate_wrap():
    observations, _ = scene(70, 8)
    varied = []
    for i, o in enumerate(observations):
        if o.local_valid:
            center = (o.box[0] + o.box[2]) / 2
            anchor = 24 + round((center-24)/16)*16
            pitch = 16 + (.015 if i % 2 else -.015)
            axis = replace(o.hypothesis.x, pitch=pitch, origin=anchor-o.box[0])
            evidence = replace(o.hypothesis.x_evidence, pitch=pitch)
            o = replace(o, hypothesis=replace(o.hypothesis, x=axis, x_evidence=evidence))
        varied.append(o)
    fit = consensus(varied)
    assert fit is not None and not fit.ambiguous
    assert fit.pitch_x == pytest.approx(16, abs=.01)
    assert fit.residual_x < .01
