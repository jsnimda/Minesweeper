from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from grid_geometry.phase.decoder import AxisExtent, DecodeResult, FrequencyCandidate, GridHypothesis
from grid_geometry.phase.tiles import TileObservation
from grid_geometry.preprocessing import ImageTransform
from recognition_lab.local_geometry import mapping, primitives, resolve_local
from recognition_lab.recognition_state import RecognitionSelection
from recognition_lab.results import Crop, Detection, GeometryResult, RecognitionResult


def fixture(mode='whole', accepted=False):
    evidence = FrequencyCandidate(10., 9.75, .9, .8)
    x = AxisExtent(-.25, 10., 3, None, 24.25, True, False)
    y = AxisExtent(1.5, 8., 2, 1.75, None, False, True)
    first = GridHypothesis(x, y, evidence, evidence, 1., .08, .98)
    alternative = replace(first, x=replace(x, origin=1.25))
    decoded = DecodeResult('ambiguous', (first, alternative))
    local = SimpleNamespace(local_accepted=accepted, local_reasons=())
    box = (0, 0, 25, 20) if mode == 'whole' else (12, 7, 37, 27)
    observation = TileObservation(box, first, accepted, decoding=decoded, local_validation=local,
                                  input_transform=ImageTransform((25, 20), (25, 20), (40, 40), (5, 9)))
    transform = ImageTransform((50, 10), (25, 20), (25, 20), (0, 0)) if mode == 'whole' else ImageTransform((100, 100), (100, 100), (100, 100), (0, 0))
    grid = GeometryResult(0, Crop((100, 50, 200, 150), False), None, transform, None, None, [],
                          diagnostics={'mode': mode}, phase_data={'observations': [observation]})
    state = RecognitionSelection(); state.replace(RecognitionResult([Detection('grid', 1., (0, 0, 10, 10))], [grid]))
    group = state.snapshot.roots[0].children[0].children[0]
    return state, group.children[0], first, alternative


def test_resolve_selected_hypothesis_status_and_stale_selection():
    state, observation, first, alternative = fixture()
    state.select(observation.id)
    local = resolve_local(state, 0)
    assert local.hypothesis is first and local.validation == 'Evaluated rejected'
    assert local.decoder_status == 'ambiguous' and not local.fallback
    state.select(observation.children[1].id)
    local = resolve_local(state, 0)
    assert local.hypothesis is alternative and local.validation == 'Not evaluated'
    state.select(state.snapshot.roots[0].children[0].id)
    assert resolve_local(state, 0).fallback
    old = observation.children[0].id
    state.replace(state.snapshot.result, geometry_update=True)
    assert not state.select(old) and resolve_local(state, 0) is None


def test_hypothesis_switch_reuses_diagnostic_image():
    from recognition_lab.model_visualization import DiagnosticCache
    state, node, first, alternative = fixture()
    grid = state.snapshot.grid(node.id)
    observation = grid.phase_data['observations'][0]
    fields = np.zeros((5, 40, 40), np.float32)
    grid.phase_data['observations'][0] = replace(observation, full_fields=fields)
    state.replace(state.snapshot.result)
    node = state.snapshot.roots[0].children[0].children[0].children[0]
    cache = DiagnosticCache()
    state.select(node.children[0].id)
    a = resolve_local(state, 0)
    image = cache.get(state.snapshot, a.observation, 'Model output', 'Full canvas', 'X cosine')
    state.select(node.children[1].id)
    b = resolve_local(state, 0)
    assert a.hypothesis is first and b.hypothesis is alternative
    assert cache.get(state.snapshot, b.observation, 'Model output', 'Full canvas', 'X cosine') is image


def test_accepted_unavailable_and_missing_hypotheses():
    state, node, _, _ = fixture(accepted=True)
    state.select(node.id)
    assert resolve_local(state, 0).validation == 'Evaluated accepted'
    grid = state.grid()
    grid.phase_data['observations'][0] = replace(state.resolve(), local_validation=None)
    state.replace(state.snapshot.result); node = state.snapshot.roots[0].children[0].children[0].children[0]
    state.select(node.id)
    assert resolve_local(state, 0).validation == 'Unavailable'
    grid.phase_data['observations'][0] = replace(state.resolve(), hypothesis=None, decoding=None)
    state.replace(state.snapshot.result); state.select(state.snapshot.roots[0].children[0].children[0].children[0].id)
    assert resolve_local(state, 0) is None


@pytest.mark.parametrize('mode,expected', [('whole', (2., .5, 100., 50.)), ('tiled', (1., 1., 112., 57.))])
def test_coordinate_mappings(mode, expected):
    state, node, _, _ = fixture(mode); state.select(node.id)
    local = resolve_local(state, 0)
    assert mapping(local, 'Source image', 'Full canvas') == expected
    assert mapping(local, 'Model output', 'Full canvas') == (1., 1., 5, 9)
    assert mapping(local, 'Model input', 'Known region') == (1., 1., 0., 0.)
    boundary = primitives(local, mapping(local, 'Source image', 'Full canvas'),
                          (0, 0, 300, 300), {'local_boundaries': True})
    expected_left = 99.5 if mode == 'whole' else 111.75
    assert any(kind == 'boundary' and coords[0] == expected_left and coords[2] == expected_left
               for kind, coords in boundary)
    canvas = primitives(local, mapping(local, 'Model input', 'Full canvas'),
                        (-10, -10, 100, 100), {'local_boundaries': True})
    assert ('boundary', (4.75, 10.5, 4.75, 26.5)) in canvas


def test_fractional_boundaries_centers_crossings_and_censoring():
    state, node, first, _ = fixture(); state.select(node.id)
    local = resolve_local(state, 0)
    layers = {'local_lattice': True, 'local_centers': True, 'local_boundaries': True,
              'terminations': True, 'local_censoring': True}
    output = primitives(local, mapping(local, 'Model input', 'Known region'), (-10, -10, 50, 40), layers)
    lines = [c for kind, c in output if kind == 'boundary']
    assert (-.25, 1.5, -.25, 17.5) in lines
    assert (29.75, 1.5, 29.75, 17.5) in lines  # outside the observation
    assert ('center', (4.75, 5.5)) in output
    assert ('crossing', (24.25, 10.)) in output
    assert ('crossing', (12.5, 1.75)) in output
    assert ('censored', (0., 10.)) in output and ('censored', (12.5, 20.)) in output
    assert first.x.origin == -.25 and first.x.end == 29.75
    assert first.x.visible_start is None and first.x.start_censored
    assert not primitives(local, (1, 1, 0, 0), (-10, -10, 50, 40), {})


def test_viewport_clipping_invalid_geometry_and_bounded_iteration():
    state, node, _, _ = fixture(); state.select(node.id)
    local = resolve_local(state, 0)
    clipped = primitives(local, (1, 1, 0, 0), (0, 0, 25, 20), {'local_boundaries': True})
    assert all(coords[0] != -.25 for _, coords in clipped)
    assert len(primitives(replace(local, hypothesis=replace(local.hypothesis, x=replace(local.hypothesis.x, pitch=float('nan')))),
                          (1, 1, 0, 0), (0, 0, 25, 20), {'local_lattice': True})) == 0
    dense = replace(local, hypothesis=replace(local.hypothesis,
                    x=replace(local.hypothesis.x, pitch=.0001, count=1_000_000),
                    y=replace(local.hypothesis.y, pitch=.0001, count=1_000_000)))
    assert len(primitives(dense, (1, 1, 0, 0), (0, 0, 25, 20),
                          {'local_lattice': True, 'local_centers': True}, limit=40)) <= 40


def test_local_styles_and_visibility():
    from recognition_lab.rendering import RecognitionStyle, draw_local_geometry
    class Canvas:
        def __init__(self): self.effects = []; self.lines = 0
        def drawLine(self, *args):
            self.lines += 1
            self.effects.append(args[-1].getPathEffect() is not None)
        def drawCircle(self, *args): pass
        def drawRect(self, *args): pass
    elements = (('lattice', (0., 0., 2., 3.)),)
    for status, dashed in [('Evaluated accepted', False), ('Evaluated rejected', True),
                           ('Not evaluated', True), ('Unavailable', True)]:
        canvas = Canvas(); draw_local_geometry(canvas, elements, status, RecognitionStyle(), 1)
        assert canvas.lines == 2 and canvas.effects == [dashed, dashed]
    canvas = Canvas(); draw_local_geometry(canvas, (), 'Evaluated accepted', RecognitionStyle(), 1)
    assert canvas.lines == 0
