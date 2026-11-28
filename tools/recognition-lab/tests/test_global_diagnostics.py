from types import SimpleNamespace

import numpy as np
import pytest

from grid_geometry.phase.decoder import AxisExtent, DecodeResult, FrequencyCandidate, GridHypothesis
from grid_geometry.phase.global_geometry import GlobalResult
from grid_geometry.phase.tiles import LatticeConsensus, TileObservation
from grid_geometry.preprocessing import ImageTransform
from recognition_lab.global_diagnostics import (GlobalHeatCache, inspect_global, observation_row, participation,
                                                participation_rectangles, resolve_global)
from recognition_lab.recognition_state import RecognitionSelection
from recognition_lab.results import Crop, Detection, GeometryResult, RecognitionResult


def fixture(mode='whole', lattice=True):
    candidate = FrequencyCandidate(10., 9.75, .9, .8)
    extent = AxisExtent(-.25, 10., 3, None, None, True, True)
    hypothesis = GridHypothesis(extent, extent, candidate, candidate, 1., .08, .98)
    decoded = DecodeResult('ambiguous', (hypothesis,))
    obs = [TileObservation((0, 0, 3, 2), hypothesis, True, decoding=decoded,
                           local_validation=SimpleNamespace(local_accepted=True)),
           TileObservation((0, 0, 3, 2), hypothesis, True, decoding=decoded,
                           local_validation=SimpleNamespace(local_accepted=False)),
           TileObservation((0, 0, 3, 2), None, False)]
    consensus = LatticeConsensus(10.125, 9.875, .25, .75, (0,), (1, 2), False, .125, .25) if lattice else None
    occupancy = np.array([[.2, .75, np.nan], [.1, .5, .8]], np.float32)
    weights = np.array([[1., 2., 0.], [.25, 1., 2.]], np.float64)
    global_result = GlobalResult(consensus, None, False, ('left termination unobservable',), occupancy, weights)
    tr = (ImageTransform((6, 1), (3, 2), (3, 2), (0, 0)) if mode == 'whole'
          else ImageTransform((3, 2), (3, 2), (3, 2), (0, 0)))
    grid = GeometryResult(0, Crop((100, 50, 106, 52), False), None, tr, None, global_result, [],
                          diagnostics={'mode': mode}, phase_data={'global': global_result, 'observations': obs})
    state = RecognitionSelection(); state.replace(RecognitionResult([Detection('grid', 1., (100, 50, 106, 52))], [grid]))
    state.select(state.snapshot.roots[0].children[0].id)
    return state, global_result, obs


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
def test_global_resolution_participation_and_complete_status(mode):
    state, result, obs = fixture(mode)
    evidence = resolve_global(state)
    assert evidence.result is result and evidence.observations is state.grid().phase_data['observations']
    assert evidence.mode == mode and result.lattice is not None and not result.accepted
    assert observation_row(evidence, 0)[2:5] == ('Accepted', 'Eligible', 'Inlier')
    assert observation_row(evidence, 1)[2:5] == ('Rejected', 'Eligible', 'Excluded')
    assert observation_row(evidence, 2)[2:5] == ('Unavailable', 'Not eligible', 'Excluded')
    assert observation_row(evidence, 0)[5:9] == (10., 10., 9.75, 9.75)
    assert observation_row(evidence, 0)[9:] == (None, None)
    node = state.snapshot.roots[0].children[0].children[0].children[0]
    state.select(node.id)
    assert resolve_global(state).grid is evidence.grid  # shared selection, no table selection system
    state.select(node.children[0].id)
    assert resolve_global(state).result is result


def test_missing_lattice_or_individual_membership_stays_unavailable():
    state, _, _ = fixture(lattice=False)
    evidence = resolve_global(state)
    assert participation(None, 0) == 'Unavailable'
    assert observation_row(evidence, 0)[4] == 'Unavailable'
    assert participation(SimpleNamespace(inlier_count=1, rejected_count=2), 0) == 'Unavailable'
    assert participation(SimpleNamespace(inliers=(0,), rejected=(0,)), 0) == 'Unavailable'


@pytest.mark.parametrize('mode,point', [('whole', (101., 50.25)), ('tiled', (100.5, 50.5))])
def test_fused_pixel_values_and_coordinate_spaces(mode, point):
    state, result, _ = fixture(mode)
    evidence = resolve_global(state)
    position, pixel, values = inspect_global(evidence, *point)
    assert position == pytest.approx((.5, .5)) and pixel == (0, 0)
    assert values['occupancy'] == result.occupancy[0, 0].item()
    assert values['weights'] == result.weights[0, 0].item() and values['coverage'] is True
    assert 'observation_count' not in values
    assert inspect_global(evidence, 99., 50.) is None
    rectangles = participation_rectangles(evidence)
    assert rectangles[0][1] == ((100., 50., 106., 51.) if mode == 'whole' else (100., 50., 103., 52.))
    assert rectangles[0][2:] == ('accepted', True, 'Inlier')
    assert rectangles[2][2:] == ('unavailable', False, 'Excluded')


def test_heatmap_values_fixed_opacity_dynamic_weight_range_and_cache():
    state, result, _ = fixture()
    evidence = resolve_global(state); cache = GlobalHeatCache(); snapshot = state.snapshot
    occupancy, legend = cache.get(snapshot, evidence, 'Phase occupancy')
    assert occupancy[0, 0, 3] == 25 and occupancy[0, 2, 3] == 0
    assert '0–1' in legend and cache.get(snapshot, evidence, 'Phase occupancy')[0] is occupancy
    coverage, legend = cache.get(snapshot, evidence, 'Phase coverage')
    assert coverage[0, 0, 3] == 127 and coverage[0, 2, 3] == 0
    assert 'weight > 0' in legend
    weights, legend = cache.get(snapshot, evidence, 'Phase fusion weights')
    assert weights[0, 0, 3] == 63 and weights[0, 1, 3] == 127
    assert '0–2.0' in legend and 'not an observation count or probability' in legend
    assert result.weights[0, 1] == 2. and result.occupancy[0, 0] == np.float32(.2)
    old = state.selected
    state.replace(state.snapshot.result, geometry_update=True)
    assert state.snapshot.resolve(old) is None
    new_evidence = resolve_global(state)
    assert cache.get(state.snapshot, new_evidence, 'Phase occupancy')[0] is not occupancy
    cache.get(None, None, 'None')
    assert cache.key is cache.rgba is cache.legend is None


def test_participation_overlay_style_does_not_change_desktop_or_selection():
    from recognition_lab.rendering import RecognitionStyle, draw_participation
    state, _, _ = fixture()
    evidence = resolve_global(state); selected = state.selected
    class Canvas:
        def __init__(self): self.rects = []; self.labels = []
        def drawRect(self, rect, paint):
            if paint.getStyle() == skia.Paint.kStroke_Style:
                self.rects.append(paint.getPathEffect() is not None)
        def drawString(self, text, *args): self.labels.append(text)
    import skia
    canvas = Canvas()
    draw_participation(canvas, participation_rectangles(evidence), RecognitionStyle(), 1.)
    assert canvas.rects == [False, True, True]
    assert 'local accepted / eligible / inlier' in canvas.labels[0]
    assert 'local rejected / eligible / excluded' in canvas.labels[1]
    assert 'local unavailable / not eligible / excluded' in canvas.labels[2]
    assert state.selected == selected


def test_observation_table_preserves_actual_membership():
    from recognition_lab.global_panel import ObservationTable
    state, _, _ = fixture(); table = ObservationTable(); table.set_evidence(resolve_global(state))
    assert table.rowCount() == 3
    assert table.data(table.index(1, 3)) == 'Eligible'
    assert table.data(table.index(1, 4)) == 'Excluded'
    assert table.data(table.index(1, 9)) == 'Unavailable'
    table.set_evidence(None)
    assert table.rowCount() == 0
