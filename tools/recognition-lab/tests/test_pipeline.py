from pathlib import Path
import pytest
from recognition_lab.pipeline import Pipeline, Settings
from recognition_lab.results import RecognitionResult, GeometryResult, Crop


def test_missing_models():
    pipeline = Pipeline()
    with pytest.raises(FileNotFoundError): pipeline.load_yolo(Path('/missing/yolo.pt'))
    with pytest.raises(FileNotFoundError): pipeline.load_phase(Path('/missing/phase.pt'))


def test_geometry_requires_phase_checkpoint():
    pipeline = Pipeline()
    with pytest.raises(RuntimeError, match='PhaseGridUNet'):
        pipeline.geometry(None, RecognitionResult(), Settings())


def test_empty_detection_result():
    result = RecognitionResult()
    assert not result.detections and not result.grids


def test_only_validated_geometry_is_exposed():
    from types import SimpleNamespace
    import numpy as np
    edges = (np.array([1., 2., 3.]), np.array([4., 5.]), 1.)
    grid = GeometryResult(0, Crop((0, 0, 8, 8), False), None, None, None,
                          SimpleNamespace(accepted=False), [edges])
    assert grid.accepted_geometry is None
    grid.validation = SimpleNamespace(accepted=True)
    assert grid.accepted_geometry[0] is edges[0]
    assert grid.accepted_geometry[1] is edges[1]


@pytest.mark.parametrize('decoder_status', ['fitted', None])
def test_phase_overlay_accepts_validated_edges(decoder_status):
    from types import SimpleNamespace
    import numpy as np
    import skia
    from recognition_lab.rendering import draw
    from grid_geometry.preprocessing import ImageTransform
    edges = (np.array([12., 20., 28.]), np.array([12., 20., 28.]), 1.)
    phase = {'global': SimpleNamespace(cell_occupancy=np.ones((2, 2))), 'observations': []}
    grid = GeometryResult(0, Crop((0, 0, 40, 40), False), np.zeros((1, 1, 1)),
                          ImageTransform((40, 40), (40, 40), (40, 40), (0, 0)),
                          SimpleNamespace(status=decoder_status) if decoder_status else None,
                          SimpleNamespace(accepted=True, status='accepted'),
                          [edges], phase_data=phase)
    surface = skia.Surface(40, 40)
    draw(surface.getCanvas(), RecognitionResult(grids=[grid]),
         {'boundaries': True, 'centers': True, 'cell_support': True})


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
@pytest.mark.parametrize('status,has_hypothesis', [('ambiguous', True), ('fitted', True),
                                                    ('insufficient_evidence', False)])
def test_geometry_retains_original_observation_artifacts(monkeypatch, mode, status, has_hypothesis):
    from types import SimpleNamespace
    import numpy as np
    import torch
    from PIL import Image
    from grid_geometry.phase import decoder, global_geometry, model, tiles, validation
    from grid_geometry.phase.decoder import AxisExtent, DecodeResult, FrequencyCandidate, GridHypothesis
    from recognition_lab.results import Detection

    evidence = FrequencyCandidate(48., 47., .9, .8)
    extent = AxisExtent(-1., 48., 2, None, None, True, True)
    first = GridHypothesis(extent, extent, evidence, evidence, 1., .08, .98)
    other = GridHypothesis(extent, extent, evidence, evidence, .9, .07, .97)
    decoded = DecodeResult(status, (first, other) if has_hypothesis else (),
                           frequency_candidates={'x': (evidence,), 'y': (evidence,)}, reasons=('evidence',))
    local = SimpleNamespace(local_accepted=False, local_reasons=('rejected',), measurements={'support': .25})
    seen = {}

    def decode(fields, *args, **kwargs):
        seen['decoded_fields'] = fields
        return decoded

    def validate(fields, hypothesis, *, decoder_status):
        assert fields is seen['decoded_fields'] and hypothesis is first and decoder_status == status
        return local

    def recognize(observations, shape):
        seen['global_observations'] = observations
        seen['shape'] = shape
        geometry = (SimpleNamespace(x_edges=np.array([-1., 20.]), y_edges=np.array([-2., 10.]))
                    if has_hypothesis else None)
        return SimpleNamespace(lattice=None, geometry=geometry, accepted=False, status='rejected',
                               boundaries={}, measurements={}, reasons=('global rejection',))

    monkeypatch.setattr(decoder, 'decode_phase', decode)
    monkeypatch.setattr(validation, 'validate_phase', validate)
    monkeypatch.setattr(tiles, 'decode_phase', decode)
    monkeypatch.setattr(tiles, 'validate_phase', validate)
    monkeypatch.setattr(global_geometry, 'recognize_global', recognize)
    monkeypatch.setattr(model, 'phase_predictions', lambda output: output)

    class FakeModel:
        def to(self, device): return self
        def eval(self): return self
        def __call__(self, tensor):
            seen['input'] = tensor.detach().cpu().clone().numpy()[0]
            return torch.zeros((1, 5, 384, 384), dtype=torch.float32)

    image = Image.new('RGB', (60, 40), (32, 128, 255))
    pipeline = Pipeline(); pipeline.phase = FakeModel()
    result = pipeline.geometry(image, RecognitionResult([Detection('grid', .9, (10, 5, 50, 25))]),
                               Settings(device='cpu', padding=0, mode=mode))
    grid = result.grids[0]
    observation = grid.phase_data['observations'][0]
    assert seen['global_observations'][0] is observation
    assert observation.decoding is decoded
    assert observation.hypothesis is (first if has_hypothesis else None)
    assert observation.local_validation is (local if has_hypothesis else None)
    assert not observation.local_valid
    assert observation.reasons == (('rejected',) if has_hypothesis else decoded.reasons)
    assert observation.full_fields.shape == (5, 384, 384)
    assert observation.full_fields.dtype == np.float32
    assert np.shares_memory(observation.fields, observation.full_fields)
    assert observation.fields is seen['decoded_fields']
    assert observation.input_array.shape == (3, 384, 384)
    assert observation.input_array.dtype == np.float32
    assert observation.input_representation == 'RGB CHW float32 divided by 255'
    assert observation.resize_method == ('bilinear' if mode == 'whole' else 'none')
    np.testing.assert_array_equal(observation.input_array, seen['input'])
    assert observation.input_transform.canvas_size == (384, 384)
    assert observation.input_transform.source_size == (40, 20)
    assert observation.input_transform.offset == ((0, 96) if mode == 'whole' else (0, 0))
    assert observation.input_transform.resized_size == ((384, 192) if mode == 'whole' else (40, 20))
    assert observation.known_box == ((0, 96, 384, 288) if mode == 'whole' else (0, 0, 40, 20))
    assert observation.padding == ((0, 96, 0, 96) if mode == 'whole' else (0, 0, 344, 364))
    assert observation.box == ((0, 0, 384, 192) if mode == 'whole' else (0, 0, 40, 20))
    assert seen['shape'] == ((192, 384) if mode == 'whole' else (20, 40))
    assert observation.hypothesis is None or observation.hypothesis.x.origin == -1.
    assert observation.hypothesis is None or observation.hypothesis.x.start_censored
    assert grid.validation.accepted is False
    if has_hypothesis:
        assert grid.hypotheses[0][0][0] == pytest.approx(10 - (1 / grid.heatmap_transform.scale_x))
        assert grid.hypotheses[0][1][0] == pytest.approx(5 - (2 / grid.heatmap_transform.scale_y))
    else:
        assert not grid.hypotheses
    assert grid.decoding is (decoded if mode == 'whole' else None)
    assert grid.heatmaps is (observation.fields if mode == 'whole' else None)
