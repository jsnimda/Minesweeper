from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from grid_geometry.phase.global_geometry import GlobalGeometry, GlobalResult
from grid_geometry.phase.tiles import LatticeConsensus
from grid_geometry.preprocessing import ImageTransform
from recognition_lab.global_diagnostics import GlobalEvidence, resolve_global
from recognition_lab.global_validation_diagnostics import (GlobalCellCache, cell_artifact, inspect_cell,
                                                           termination_lines, visible_cells)
from recognition_lab.recognition_state import RecognitionSelection
from recognition_lab.results import Crop, Detection, GeometryResult, RecognitionResult


def fixture(mode='whole', accepted=False, geometry=True):
    lattice = LatticeConsensus(10., 8., 9.75, 1.25, (0,), (), False, .125, .25)
    g = GlobalGeometry(np.array([-.25, 9.75, 19.75]), np.array([1.25, 9.25, 17.25]), (-1, 1), (0, 2)) if geometry else None
    cells = np.array([[.2, np.nan], [.49, .51]]) if geometry else None
    coverage = np.array([[1., .25], [.7, .95]]) if geometry else None
    sides = {'left': {'count': 2, 'estimate': -.75, 'spread_cells': .1, 'samples': (-1., -.5),
                      'confidence': (.8, .7), 'index': -1, 'snapped': -.25, 'residual_cells': .05,
                      'inside_occupancy': .8, 'outside_occupancy': .3, 'inside_coverage': 1.,
                      'outside_coverage': .8, 'exterior_supported_cells': 1,
                      'candidates': [{'index': -1, 'coordinate': -.25, 'admissible': False}]},
             'right': {'count': 0, 'estimate': None, 'spread_cells': None, 'samples': ()},
             'top': {'count': 1, 'estimate': 1.125, 'samples': (1.125,)},
             'bottom': {'count': 0, 'estimate': None, 'samples': ()}}
    result = GlobalResult(lattice, g, accepted, () if accepted else ('left exterior contains supported cells',),
                          np.ones((20, 25), np.float32), np.ones((20, 25)), cells, coverage, sides,
                          {'supported_cell_fraction': .75, 'min_row_support': .5})
    tr = ImageTransform((50, 10), (25, 20), (25, 20), (0, 0)) if mode == 'whole' else ImageTransform((25, 20), (25, 20), (25, 20), (0, 0))
    crop = Crop((100, 50, 150, 70), False)
    screenshot_hypotheses = [] if g is None else [(g.x_edges/tr.scale_x+100, g.y_edges/tr.scale_y+50, 0.)]
    grid = GeometryResult(0, crop, None, tr, None, result, screenshot_hypotheses,
                          diagnostics={'mode': mode}, phase_data={'global': result, 'observations': []})
    state = RecognitionSelection(); state.replace(RecognitionResult([Detection('grid', 1., crop.box)], [grid]))
    state.select(state.snapshot.roots[0].children[0].id)
    return state, result


@pytest.mark.parametrize('accepted', [True, False])
def test_candidate_and_accepted_geometry_are_separate(accepted):
    state, result = fixture(accepted=accepted)
    evidence = resolve_global(state)
    assert evidence.result.geometry is result.geometry
    assert (state.grid().accepted_geometry is not None) is accepted
    assert cell_artifact(evidence, 'Cell occupancy')[2] is result.cell_occupancy
    assert result.boundaries['left']['candidates'][0]['admissible'] is False
    assert result.boundaries['left']['exterior_supported_cells'] == 1
    assert 'exterior_locations' not in result.boundaries['left']


def test_lattice_without_extent_and_missing_termination_evidence():
    state, result = fixture(geometry=False)
    evidence = resolve_global(state)
    assert result.lattice is not None and result.geometry is None and state.grid().accepted_geometry is None
    assert cell_artifact(evidence, 'Cell occupancy') is None
    assert inspect_cell(evidence, 100, 50) is None
    lines = termination_lines(evidence)
    assert [name for name, _ in lines] == ['left', 'top']
    assert result.boundaries['right']['estimate'] is None
    assert termination_lines(replace(evidence, result=None)) == ()


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
def test_global_cell_mapping_and_retained_values(mode):
    state, result = fixture(mode); evidence = resolve_global(state)
    tr = evidence.grid.heatmap_transform
    # Interior of global row 1, column 0; coordinates remain fractional.
    x, y = 100 + 4.75/tr.scale_x, 50 + 13.25/tr.scale_y
    row, col, values = inspect_cell(evidence, x, y)
    assert (row, col) == (1, 0)
    assert values == {'Cell occupancy': .49, 'Cell coverage': .7}
    assert inspect_cell(evidence, 100 + 19.75/tr.scale_x, y) is None
    bad = replace(result, cell_coverage=np.ones((1, 1)))
    assert cell_artifact(replace(evidence, result=bad), 'Cell coverage') is None
    assert inspect_cell(replace(evidence, result=bad), x, y)[2] == {'Cell occupancy': .49}


@pytest.mark.parametrize('mode,left', [('whole', 98.5), ('tiled', 99.25)])
def test_termination_projection_keeps_outward_estimate(mode, left):
    state, result = fixture(mode); evidence = resolve_global(state)
    lines = dict(termination_lines(evidence))
    assert lines['left'][0] == left and lines['left'][2] == left
    assert result.boundaries['left']['estimate'] == -.75
    assert 'right' not in lines and 'bottom' not in lines
    assert all(name in ('left', 'top') for name in lines)


def test_global_cell_colors_values_bounds_and_cache_invalidation():
    state, result = fixture(); evidence = resolve_global(state); cache = GlobalCellCache()
    snapshot = state.snapshot
    rgb = cache.get(snapshot, evidence, 'Cell occupancy')
    assert cache.get(snapshot, evidence, 'Cell occupancy') is rgb
    assert tuple(rgb[0, 0]) == (51, 51, 51)
    assert tuple(rgb[0, 1]) == (255, 0, 255)
    assert tuple(rgb[1, 0]) != tuple(rgb[1, 1])  # no support classification threshold
    cells = visible_cells(evidence, rgb, (0, 0, 300, 300), artifact=cache.artifact)
    assert len(cells) == 4
    assert cells[0][2] == pytest.approx((99.5, 50.625, 119.5, 54.625))
    assert len(visible_cells(evidence, rgb, (0, 0, 300, 300), limit=1, artifact=cache.artifact)) == 1
    assert result.cell_occupancy[1, 0] == .49
    other_grid = replace(evidence.grid)
    other = replace(evidence, grid=other_grid)
    assert cache.get(snapshot, other, 'Cell occupancy') is not rgb
    old = state.selected; state.replace(state.snapshot.result, geometry_update=True)
    assert state.snapshot.resolve(old) is None
    assert cache.get(state.snapshot, resolve_global(state), 'Cell occupancy') is not rgb
    cache.get(None, None, None)
    assert cache.key is cache.rgb is cache.artifact is None


def test_rendering_of_termination_estimates_and_cell_values():
    from recognition_lab.rendering import RecognitionStyle, draw_global_cells, draw_global_terminations
    class Canvas:
        def __init__(self): self.lines = []; self.labels = []; self.rectangles = []
        def drawLine(self, *args): self.lines.append(args[:4])
        def drawString(self, text, *args): self.labels.append(text)
        def drawRect(self, rect, paint): self.rectangles.append((rect, paint.getColor()))
    canvas = Canvas()
    draw_global_terminations(canvas, (('left', (1., 2., 1., 8.)),), RecognitionStyle(), 1.)
    assert canvas.lines == [(1., 2., 1., 8.)]
    assert canvas.labels == ['left termination estimate (projected)']
    canvas.rectangles.clear()  # Label background is separate from cell shading.
    draw_global_cells(canvas, ((0, 0, (1., 2., 3., 4.), (12, 34, 56)),), .5)
    assert len(canvas.rectangles) == 1 and canvas.rectangles[0][1] >> 24 == 128
