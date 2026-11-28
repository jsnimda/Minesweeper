from dataclasses import replace

import numpy as np
import pytest

from grid_geometry.phase.decoder import AxisExtent, DecodeResult, FrequencyCandidate, GridHypothesis
from grid_geometry.phase.tiles import TileObservation
from grid_geometry.phase.validation import ValidationResult
from grid_geometry.preprocessing import ImageTransform
from recognition_lab.local_geometry import mapping
from recognition_lab.local_validation_diagnostics import (SupportCache, cell_array, cell_at, overlay_colors,
                                                           resolve_validation, side_rows)
from recognition_lab.recognition_state import RecognitionSelection
from recognition_lab.results import Crop, Detection, GeometryResult, RecognitionResult


def fixture(mode='whole', accepted=False):
    candidate = FrequencyCandidate(10., 9.75, .9, .8)
    x = AxisExtent(-.25, 10., 3, None, 28.2, True, False)
    y = AxisExtent(1.5, 8., 2, 1.75, None, False, True)
    h = GridHypothesis(x, y, candidate, candidate, 1., .08, .98)
    other = replace(h, x=replace(x, origin=.75))
    decoded = DecodeResult('ambiguous', (h, other))
    occupancy = np.array([[.2, np.nan, .7], [.8, .9, 1.]], np.float32)
    magnitude_x = np.ones((2, 3), np.float32)
    magnitude_y = np.full((2, 3), .5, np.float32)
    alignment_x = np.array([[-1., 0., 1.], [.4, .6, .8]], np.float32)
    alignment_y = np.ones((2, 3), np.float32)
    supported = np.array([[False, False, False], [True, True, True]])
    observable = np.array([[True, True, False], [True, True, True]])
    checked = ValidationResult(False, ('right termination unobservable',), {'cell_coverage': .6},
                               occupancy, magnitude_x, magnitude_y, alignment_x, alignment_y, supported,
                               np.array([0., 1.]), np.array([.5, .5, 1.]),
                               {'left': {'censored': True, 'inner_support': .8, 'exterior_occupancy': None,
                                         'visible_offset_cells': None, 'exterior_phase_support': None}},
                               accepted, () if accepted else ('cluster of unsupported cells',), observable)
    box = (0, 0, 25, 20) if mode == 'whole' else (12, 7, 37, 27)
    obs = TileObservation(box, h, accepted, fields=np.zeros((5, 20, 25), np.float32),
                          input_transform=ImageTransform((25, 20), (25, 20), (40, 40), (5, 9)),
                          decoding=decoded, local_validation=checked)
    tr = ImageTransform((50, 10), (25, 20), (25, 20), (0, 0)) if mode == 'whole' else ImageTransform((100, 100), (100, 100), (100, 100), (0, 0))
    grid = GeometryResult(0, Crop((100, 50, 200, 150), False), None, tr, None, None, [],
                          diagnostics={'mode': mode}, phase_data={'observations': [obs]})
    state = RecognitionSelection(); state.replace(RecognitionResult([Detection('grid', 1., (0, 0, 10, 10))], [grid]))
    node = state.snapshot.roots[0].children[0].children[0].children[0]
    return state, node, checked


@pytest.mark.parametrize('accepted', [True, False])
def test_evaluated_and_unevaluated_validation_association(accepted):
    state, node, validation = fixture(accepted=accepted)
    state.select(node.id)
    evidence = resolve_validation(state, 0)
    assert evidence.validation is validation
    assert evidence.status == ('Evaluated accepted' if accepted else 'Evaluated rejected')
    assert evidence.validation.accepted is False  # local complete-extent result is separate
    assert cell_array(evidence, 'cell_occupancy') is validation.cell_occupancy
    state.select(node.children[1].id)
    alternate = resolve_validation(state, 0)
    assert alternate.status == 'Not evaluated' and alternate.validation is None
    assert cell_array(alternate, 'supported') is None and side_rows(alternate) == ()
    assert overlay_colors(alternate, 'Occupancy') is None


def test_missing_validation_and_exact_hypothesis_association():
    state, node, _ = fixture(); grid = state.snapshot.grid(node.id)
    obs = grid.phase_data['observations'][0]
    grid.phase_data['observations'][0] = replace(obs, local_validation=None)
    state.replace(state.snapshot.result); node = state.snapshot.roots[0].children[0].children[0].children[0]
    state.select(node.id)
    evidence = resolve_validation(state, 0)
    assert evidence.status == 'Unavailable' and evidence.validation is None
    assert cell_at(evidence, (1, 1, 0, 0), 1, 2) is None
    grid.phase_data['observations'][0] = replace(obs, hypothesis=obs.decoding.hypotheses[1])
    state.replace(state.snapshot.result); node = state.snapshot.roots[0].children[0].children[0].children[0]
    state.select(node.id)
    assert resolve_validation(state, 0).validation is None


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
def test_cell_index_and_fractional_viewer_transforms(mode):
    state, node, checked = fixture(mode); state.select(node.id)
    evidence = resolve_validation(state, 0)
    for content, region in [('Source image', 'Full canvas'), ('Model input', 'Full canvas'),
                            ('Model output', 'Known region')]:
        sx, sy, tx, ty = tr = mapping(evidence.local, content, region)
        # Center of row 1, column 2 is measured from the equivalent boundaries.
        x, y = tx + sx*(-.25 + 2.5*10), ty + sy*(1.5 + 1.5*8)
        row, col, values = cell_at(evidence, tr, x, y)
        assert (row, col) == (1, 2)
        assert values['cell_occupancy'] == checked.cell_occupancy[1, 2]
        assert values['supported'] is True and values['observable'] is True
    assert cell_at(evidence, (1, 1, 0, 0), -.26, 2) is None
    assert cell_at(evidence, (1, 1, 0, 0), 29.75, 2) is None


def test_support_colors_masks_nonfinite_and_unavailable_observability():
    state, node, checked = fixture(); state.select(node.id)
    evidence = resolve_validation(state, 0)
    original = checked.cell_occupancy.copy()
    rgba = overlay_colors(evidence, 'Occupancy')
    assert rgba.shape == (2, 3, 4)
    assert tuple(rgba[0, 1, :3]) == (255, 0, 255)  # nonfinite and observable
    assert tuple(rgba[0, 2, :3]) == (105, 105, 105)  # unobservable
    np.testing.assert_array_equal(checked.cell_occupancy, original)
    mask = overlay_colors(evidence, 'Supported mask')
    assert tuple(mask[1, 0, :3]) == (46, 210, 126)
    assert tuple(mask[0, 0, :3]) == (245, 78, 86)
    assert tuple(overlay_colors(evidence, 'Observable mask')[0, 2, :3]) == (105, 105, 105)
    missing = replace(evidence, validation=replace(checked, observable=None))
    assert np.all(overlay_colors(missing, 'Occupancy')[..., :3] == 105)
    assert overlay_colors(missing, 'Observable mask') is None
    bad_shape = replace(evidence, validation=replace(checked, supported=np.zeros((1, 1), bool)))
    assert overlay_colors(bad_shape, 'Supported mask') is None


def test_retained_side_evidence_and_row_column_arrays():
    state, node, checked = fixture(); state.select(node.id)
    evidence = resolve_validation(state, 0)
    sides = side_rows(evidence)
    assert sides[0][:4] == ('left', -.25, None, True)
    assert sides[0][4] is checked.side_evidence['left']
    assert sides[0][4]['exterior_occupancy'] is None
    assert sides[1][1] == 29.75 and sides[1][2] == 28.2
    assert checked.row_support.tolist() == [0., 1.]
    assert checked.column_support.tolist() == [.5, .5, 1.]
    from recognition_lab.local_validation_panel import CellTable, SupportTable
    table = SupportTable(); table.set_arrays(checked.row_support, checked.column_support)
    assert table.rowCount() == 3
    assert table.data(table.index(1, 1)) == '1.0'
    assert table.data(table.index(2, 1)) == 'Unavailable'
    assert table.data(table.index(2, 2)) == '1.0'
    cells = CellTable(); cells.set_array(checked.cell_occupancy)
    assert cells.rowCount() == 6
    assert cells.data(cells.index(4, 0)) == '1' and cells.data(cells.index(4, 1)) == '1'
    assert cells.data(cells.index(4, 2)) == repr(checked.cell_occupancy[1, 1].item())
    cells.set_array(np.array([[42.]], np.float32))
    assert cells.data(cells.index(0, 2)) == '42.0'  # Inspectable even if it cannot map to the hypothesis.


def test_cache_and_overlay_invalidate_on_alternative_and_snapshot_replacement():
    state, node, _ = fixture(); state.select(node.children[0].id)
    evidence = resolve_validation(state, 0); cache = SupportCache()
    rgba = cache.get(state.snapshot, evidence, 'Occupancy')
    assert cache.get(state.snapshot, evidence, 'Occupancy') is rgba
    state.select(node.children[1].id)
    assert cache.get(state.snapshot, resolve_validation(state, 0), 'Occupancy') is None
    assert cache.key is None and cache.rgba is None
    old = node.children[0].id
    state.replace(state.snapshot.result, geometry_update=True)
    assert state.snapshot.resolve(old) is None
    assert not state.select(old) and resolve_validation(state, 0) is None
    assert cache.get(state.snapshot, None, 'Occupancy') is None
