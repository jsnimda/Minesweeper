import math
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from grid_geometry.phase.tiles import TileObservation
from grid_geometry.preprocessing import ImageTransform
from recognition_lab.model_visualization import (CHANNELS, DiagnosticCache, colors, resolve_observation,
                                                  scalar_field, scale_for)
from recognition_lab.recognition_state import RecognitionSelection
from recognition_lab.results import Crop, Detection, GeometryResult, RecognitionResult
from recognition_lab.viewport import ViewportTransform


def observation():
    inputs = np.zeros((3, 4, 6), dtype=np.float32)
    inputs[:, 1, 2] = [.25, .5, 1.]
    fields = np.zeros((5, 4, 6), dtype=np.float32)
    fields[:, 1, 2] = [.5, -.5, -.25, .75, .625]
    return TileObservation((10, 20, 14, 22), None, False, fields=fields[:, 1:3, 1:5],
                           input_array=inputs, full_fields=fields,
                           input_transform=ImageTransform((4, 2), (4, 2), (6, 4), (1, 1)),
                           input_representation='RGB CHW float32 divided by 255', resize_method='none')


def selection(mode):
    obs1, obs2 = observation(), observation()
    grid = GeometryResult(0, Crop((0, 0, 30, 30), False), None, None, None, None, [],
                          diagnostics={'mode': mode}, phase_data={'observations': [obs1] if mode == 'whole' else [obs1, obs2]})
    state = RecognitionSelection(); state.replace(RecognitionResult([Detection('grid', 1., (0, 0, 20, 20))], [grid]))
    state.select(state.snapshot.roots[0].children[0].id)
    return state, obs1, obs2


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
def test_resolve_observation_uses_selection_and_fallback(mode):
    state, first, second = selection(mode)
    assert resolve_observation(state, 0) is first
    assert resolve_observation(state, 1) is (second if mode == 'tiled' else first)
    node = state.snapshot.roots[0].children[0].children[0].children[0]
    state.select(node.id)
    assert resolve_observation(state, 1) is first
    state.replace(None)
    assert resolve_observation(state, 0) is None


def test_hypotheses_resolve_to_the_same_observation():
    from dataclasses import replace
    from grid_geometry.phase.decoder import DecodeResult
    state, first, _ = selection('whole')
    hypotheses = (object(), object())
    first = replace(first, decoding=DecodeResult('ambiguous', hypotheses))
    result = state.snapshot.result
    result.grids[0].phase_data['observations'][0] = first
    state.replace(result)
    node = state.snapshot.roots[0].children[0].children[0].children[0]
    for child in node.children:
        assert state.select(child.id)
        assert resolve_observation(state, 0) is first


def test_model_input_full_and_known_pixels_and_ownership():
    obs = observation(); snapshot = object(); cache = DiagnosticCache()
    full = cache.get(snapshot, obs, 'Model input', 'Full canvas', None)
    known = cache.get(snapshot, obs, 'Model input', 'Known region', None)
    assert full.image.size == (6, 4) and known.image.size == (4, 2)
    assert full.image.getpixel((2, 1)) == (64, 128, 255)
    assert np.shares_memory(full.values, obs.input_array) and np.shares_memory(known.values, obs.input_array)
    pixel, canvas, actual = known.inspect(1.5, .5)
    assert pixel == (1, 0) and canvas == (2.5, 1.5)
    np.testing.assert_array_equal(actual, obs.input_array[:, 1, 2])
    assert known.inspect(4, 0) is None
    assert cache.get(snapshot, obs, 'Model input', 'Full canvas', None) is full
    assert obs.input_array[0, 1, 2] == .25


@pytest.mark.parametrize('channel,index', [('X cosine', 0), ('X sine', 1), ('Y cosine', 2), ('Y sine', 3), ('Occupancy', 4)])
def test_activated_channels_use_original_fields(channel, index):
    obs = observation(); view = DiagnosticCache().get(object(), obs, 'Model output', 'Known region', channel)
    assert np.shares_memory(view.values, obs.full_fields)
    assert view.inspect(1.1, .2)[2] == obs.full_fields[index, 1, 2]
    assert view.scale == scale_for(channel)


def test_derived_channels_and_fixed_ranges():
    obs = observation(); fields = obs.full_fields
    assert tuple(CHANNELS) == ('X sine', 'X cosine', 'Y sine', 'Y cosine', 'Occupancy',
                               'X wrapped phase', 'Y wrapped phase', 'X phase magnitude', 'Y phase magnitude')
    assert scalar_field(fields, 'X wrapped phase')[1, 2] == pytest.approx(math.atan2(-.5, .5))
    assert scalar_field(fields, 'Y wrapped phase')[1, 2] == pytest.approx(math.atan2(.75, -.25))
    assert scalar_field(fields, 'X phase magnitude')[1, 2] == pytest.approx(math.hypot(.5, -.5))
    assert scalar_field(fields, 'Y phase magnitude')[1, 2] == pytest.approx(math.hypot(-.25, .75))
    assert scale_for('X sine') == (-1., 1., 'diverging')
    assert scale_for('Occupancy') == (0., 1., 'sequential')
    assert scale_for('X wrapped phase') == (-math.pi, math.pi, 'cyclic')
    assert scale_for('Y phase magnitude') == (0., math.sqrt(2), 'sequential')
    wrap = colors(np.array([[-math.pi, math.pi]]), scale_for('X wrapped phase'))
    np.testing.assert_array_equal(wrap[0, 0], wrap[0, 1])
    np.testing.assert_array_equal(colors(np.array([[.5]]), scale_for('Occupancy'))[0, 0], [128, 128, 128])


def test_missing_nonfinite_and_cache_invalidation():
    from dataclasses import replace
    obs = observation(); cache = DiagnosticCache(); snapshot = object()
    obs.full_fields[0, 1, 2] = np.inf
    field = scalar_field(obs.full_fields, 'X wrapped phase')
    assert np.isnan(field[1, 2]) and np.isinf(obs.full_fields[0, 1, 2])
    view = cache.get(snapshot, obs, 'Model output', 'Full canvas', 'X wrapped phase')
    assert view.image.getpixel((2, 1)) == (255, 0, 255)
    assert np.isnan(view.inspect(2.5, 1.5)[2])
    replacement = object()
    assert cache.get(replacement, replace(obs, full_fields=None), 'Model output', 'Full canvas', 'X sine') is None
    assert cache.snapshot is replacement and all(v is None for v in cache.views.values())
    assert cache.get(replacement, replace(obs, input_representation=None), 'Model input', 'Full canvas', None) is None
    assert cache.get(replacement, replace(obs, input_transform=None), 'Model input', 'Known region', None) is None
    cache.clear()
    assert cache.snapshot is cache.observation is None and cache.views == {}


def test_source_and_diagnostic_viewport_states_and_no_replacement_for_same_observation():
    from dataclasses import replace
    from grid_geometry.phase.decoder import DecodeResult
    from recognition_lab.ui import MainWindow
    state, obs, _ = selection('whole')
    result = state.snapshot.result
    result.grids[0].phase_data['observations'][0] = replace(obs, decoding=DecodeResult('ambiguous', (object(), object())))
    state.replace(result)
    hypotheses = state.snapshot.roots[0].children[0].children[0].children[0].children
    state.select(hypotheses[0].id)
    class Choice:
        def __init__(self, text): self.text = text
        def currentText(self): return self.text
        def setEnabled(self, value): pass
    class Label:
        def setText(self, value): self.text = value
        def setToolTip(self, value): self.tooltip = value
        def clear(self): self.text = ''
        def hide(self): pass
        def show(self): pass
    class Canvas:
        def __init__(self):
            self.image_size = (100, 100); self.transform = ViewportTransform(2, 7, 9)
            self.mode = 'manual'; self._dpr = 1.; self.calls = 0
        def width(self): return 400
        def height(self): return 300
        def set_image(self, image):
            self.calls += 1; self.image_size = image.size if image else None
            self.transform = ViewportTransform(); self.mode = 'fit'
        def set_transform(self, tr): self.transform = tr; self.mode = 'manual'
        def fit(self): self.mode = 'fit'
    canvas = Canvas()
    window = SimpleNamespace(content_mode=Choice('Model input'), content_region=Choice('Full canvas'),
                             output_channel=Choice('X sine'), known_outline=Choice(''), recognition=state,
                             _diagnostic_cache=DiagnosticCache(), _diagnostic_view=None,
                             _view_key=('source', 1), _view_kind='source', _view_states={},
                             frame=SimpleNamespace(image=Image.new('RGB', (100, 100))), canvas=canvas,
                             content_label=Label(), legend_label=Label(), image_status=Label())
    MainWindow.update_viewer_content(window, 0)
    assert canvas.calls == 1 and window._view_kind == 'diagnostic'
    _, height, width = obs.input_array.shape
    assert window.image_status.text == f'Model-canvas pixels: {width} × {height}'
    canvas.set_transform(ViewportTransform(4, 11, 12))
    state.select(hypotheses[1].id)
    MainWindow.update_viewer_content(window, 0)
    assert canvas.calls == 1 and canvas.transform == ViewportTransform(4, 11, 12)
    window.content_mode.text = 'Source image'
    MainWindow.update_viewer_content(window, 0)
    assert canvas.transform == ViewportTransform(2, 7, 9)
    window.content_mode.text = 'Model input'
    MainWindow.update_viewer_content(window, 0)
    assert canvas.transform == ViewportTransform(4, 11, 12)
    window.content_mode.text = 'Source image'
    state.replace(None)
    MainWindow.update_viewer_content(window, 0)
    assert window._diagnostic_cache.snapshot is None and window._diagnostic_cache.observation is None
    assert window.image_status.text == 'Screenshot: 100 × 100'
