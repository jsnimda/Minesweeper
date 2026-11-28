from types import SimpleNamespace

import numpy as np
import pytest

from grid_geometry.phase.decoder import AxisExtent, DecodeResult, FrequencyCandidate, GridHypothesis
from grid_geometry.phase.tiles import TileObservation
from grid_geometry.preprocessing import ImageTransform
from recognition_lab.explorer import explorer_items
from recognition_lab.inspector import inspector_sections
from recognition_lab.recognition_state import RecognitionSelection
from recognition_lab.results import Crop, Detection, GeometryResult, RecognitionResult
from recognition_lab.viewer_selection import selection_region


def fixture(mode='tiled'):
    evidence = FrequencyCandidate(48., 47.25, .8125, .75)
    x = AxisExtent(-.75, 48., 2, None, 90.125, True, False)
    y = AxisExtent(1.25, 48., 1, 1.5, None, False, True)
    hypothesis = GridHypothesis(x, y, evidence, evidence, .9, .07, .91)
    decoded = DecodeResult('ambiguous', (hypothesis, hypothesis), reasons=('competing frequency hypotheses',))
    local = SimpleNamespace(local_accepted=False, local_reasons=('decoder status: ambiguous',),
                            measurements={'cell_coverage': .875})
    fields = np.zeros((5, 20, 40), np.float32)
    full = np.zeros((5, 384, 384), np.float32)
    transform = ImageTransform((40, 20), (40, 20), (384, 384), (0, 0))
    observation = TileObservation((20, 10, 60, 30), hypothesis, False, local.local_reasons, fields,
                                  np.zeros((3, 384, 384), np.float32), transform, full, decoded, local,
                                  'RGB CHW float32 divided by 255', 'none')
    missing = TileObservation((60, 10, 100, 30), None, False, fields=fields)
    if mode == 'whole':
        observation = TileObservation((0, 0, 40, 20), hypothesis, False, local.local_reasons, fields,
                                      observation.input_array, ImageTransform((20, 10), (40, 20), (40, 40), (0, 10)),
                                      full, decoded, local, observation.input_representation, 'bilinear')
        observations = [observation]
        map_to_crop = ImageTransform((20, 10), (40, 20), (40, 20), (0, 0))
    else:
        observations = [observation, missing]
        map_to_crop = ImageTransform((100, 50), (100, 50), (100, 50), (0, 0))
    detection = Detection('grid', .9, (100., 50., 200., 100.))
    grid = GeometryResult(0, Crop((100, 50, 200, 100), False), None, map_to_crop, None,
                          SimpleNamespace(lattice=object(), accepted=False, geometry=None, reasons=()), [],
                          diagnostics={'mode': mode}, phase_data={'observations': observations})
    return RecognitionResult([detection], [grid]), observations, hypothesis


def values(sections):
    return {section.title: {p.name: p.value for p in section.properties} for section in sections}


@pytest.mark.parametrize('mode,expected', [('whole', 1), ('tiled', 2)])
def test_hierarchy_identity_and_validation_semantics(mode, expected):
    result, observations, hypothesis = fixture(mode)
    state = RecognitionSelection(); state.replace(result)
    geometry = state.snapshot.roots[0].children[0]
    assert [n.id.kind for n in geometry.children] == ['observations', 'lattice', 'validation']
    group = geometry.children[0]
    assert len(group.children) == expected
    obs = group.children[0]; first, second = obs.children
    assert first.id != second.id and state.snapshot.resolve(first.id) is hypothesis
    assert state.snapshot.resolve(second.id) is hypothesis
    assert state.snapshot.resolve(obs.id) is observations[0]
    assert state.snapshot.observation(second.id) is observations[0]
    items = explorer_items(state.snapshot)[0].children[0].children[0]
    assert items.children[0].summary == 'Dec: ambiguous · Local: Rejected'
    assert [h.summary for h in items.children[0].children] == ['Evaluated rejected', 'Not evaluated']
    if mode == 'tiled':
        assert items.children[1].summary == 'Dec: Unavailable · Local: Unavailable'
        assert not items.children[1].children
    state.select(second.id)
    old_snapshot = state.snapshot
    state.replace(result, geometry_update=True)
    assert state.resolve() is result.detections[0]
    assert state.snapshot.resolve(obs.id) is None and state.snapshot.resolve(second.id) is None
    assert old_snapshot.resolve(second.id) is hypothesis
    assert not state.select(second.id)


def test_observation_and_hypothesis_inspection_and_navigation():
    result, observations, _ = fixture()
    state = RecognitionSelection(); state.replace(result)
    group = state.snapshot.roots[0].children[0].children[0]
    obs = group.children[0]; first, second = obs.children
    props = values(inspector_sections(state.snapshot, obs.id))
    assert props['Observation']['Coordinate space'] == 'Tile-local pixels (native resolution)'
    assert props['Observation']['Observation bounds (local px)'] == '0, 0, 40, 20'
    assert props['Observation']['Source bounds (px)'] == '120.0, 60.0, 160.0, 80.0'
    assert props['Model input and preprocessing']['Known rectangle (model canvas px)'] == '0, 0, 40, 20'
    assert props['Model input and preprocessing']['Padding left, top, right, bottom (px)'] == '0, 0, 344, 364'
    assert props['Diagnostic artifacts']['Full fields shape (CHW)'] == '5, 384, 384'
    assert props['Decoder and local validation']['Decoder status'] == 'ambiguous'
    assert props['Local support measurements']['cell_coverage'] == '0.875'
    h = values(inspector_sections(state.snapshot, first.id))
    assert h['Local lattice and equivalent boundaries']['Left equivalent boundary (px)'] == '-0.75'
    assert h['Local lattice and equivalent boundaries']['Periodic origin X (px modulo pitch)'] == '47.25'
    assert h['Visible occupancy crossings']['Left crossing (px)'] == 'Unavailable'
    assert h['Visible occupancy crossings']['Left censored'] == 'Yes'
    assert h['Local validation']['Applicability'] == 'Evaluated rejected'
    other = values(inspector_sections(state.snapshot, second.id))
    assert other['Local validation']['Applicability'] == 'Not evaluated'
    assert other['Local validation']['Reasons'] == 'Unavailable'
    assert selection_region(state.snapshot, first.id) == (120., 60., 160., 80.)
    assert selection_region(state.snapshot, first.id, {'tiles': False}) is None
    assert selection_region(state.snapshot, first.id, {'tiles': True}) == (120., 60., 160., 80.)
    assert observations[0].hypothesis.x.origin == -.75


def test_whole_coordinate_mapping_and_missing_validation():
    result, _, _ = fixture('whole')
    original = result.grids[0].phase_data['observations'][0]
    from dataclasses import replace
    result.grids[0].phase_data['observations'][0] = replace(original, local_validation=None)
    state = RecognitionSelection(); state.replace(result)
    obs = state.snapshot.roots[0].children[0].children[0].children[0]
    sections = values(inspector_sections(state.snapshot, obs.id))
    assert sections['Observation']['Coordinate space'] == 'Resized crop pixels (valid region, padding removed)'
    assert sections['Decoder and local validation']['Local validation status'] == 'Unavailable'
    assert selection_region(state.snapshot, obs.id) == (100., 50., 120., 60.)
    assert values(inspector_sections(state.snapshot, obs.children[0].id))['Local validation']['Applicability'] == 'Unavailable'
    assert not state.select_tile(0)


def test_accepted_local_result_and_unassociated_geometry():
    from dataclasses import replace
    result, _, _ = fixture()
    grid = result.grids[0]
    obs = grid.phase_data['observations'][0]
    local = SimpleNamespace(local_accepted=True, local_reasons=(), measurements={'cell_coverage': 1.0})
    grid.phase_data['observations'][0] = replace(obs, local_valid=True, local_validation=local)
    grid.detection_index = 8
    state = RecognitionSelection(); state.replace(result)
    orphan = state.snapshot.roots[1]
    group = orphan.children[0]
    hypothesis = group.children[0].children[0]
    assert state.snapshot.grid(hypothesis.id) is grid
    assert state.snapshot.detection_index(hypothesis.id) is None
    assert values(inspector_sections(state.snapshot, hypothesis.id))['Local validation']['Applicability'] == 'Evaluated accepted'
    assert explorer_items(state.snapshot)[1].children[0].children[0].children[0].summary == 'Evaluated accepted'


def test_tile_selection_uses_current_snapshot_and_preserves_invalid_selection():
    result, _, _ = fixture()
    state = RecognitionSelection(); state.replace(result)
    state.select(state.snapshot.roots[0].children[0].id)
    assert state.select_tile(1)
    selected = state.selected
    assert selected.kind == 'observation' and selected.observation_index == 1
    assert not state.select_tile(9) and state.selected == selected
    state.replace(result, geometry_update=True)
    assert state.snapshot.resolve(selected) is None
    assert state.select_tile(0)
    assert state.selected.observation_index == 0


def test_tile_control_updates_the_authoritative_selection():
    from recognition_lab.ui import MainWindow
    result, _, _ = fixture()
    state = RecognitionSelection(); state.replace(result)
    state.select(state.snapshot.roots[0].children[0].id)
    calls = []
    window = SimpleNamespace(recognition=state, sync_result_tree_selection=lambda: calls.append('tree'),
                             show_diagnostics=lambda: calls.append('diagnostics'),
                             update_empty_states=lambda: calls.append('empty'),
                             redraw=lambda: calls.append('redraw'))
    MainWindow.tile_index_changed(window, 1)
    assert state.selected == state.snapshot.observation_id(0, 1)
    assert calls == ['tree', 'diagnostics', 'empty', 'redraw']
    MainWindow.tile_index_changed(window, 9)
    assert state.selected == state.snapshot.observation_id(0, 1)
    assert calls[-1] == 'redraw'
