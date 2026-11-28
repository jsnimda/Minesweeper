from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from recognition_lab.pipeline import Pipeline, Settings
from recognition_lab.provenance import (DetectionProvenance, GeometryProvenance, validation_errors,
                                        stage_mismatches, stage_summary, tile_index_range, tile_selection)
from recognition_lab.results import Crop, Detection, GeometryResult, RecognitionResult
from recognition_lab.recognition_state import ResultSnapshot
from recognition_lab.inspector import inspector_sections


def detection_provenance(path='/models/yolo.pt', model_id=1):
    return DetectionProvenance('auto', 'cpu', path, model_id, 1280, .25)


def geometry_provenance(mode='whole'):
    return GeometryProvenance('auto', 'cpu', '/models/phase.pt', 2, 20, mode,
                              384 if mode == 'tiled' else None, 96 if mode == 'tiled' else None,
                              (384, 384) if mode == 'whole' else None)


def test_provenance_is_immutable():
    for record in (detection_provenance(), geometry_provenance()):
        with pytest.raises(FrozenInstanceError):
            record.requested_device = 'cuda'


@pytest.mark.parametrize(('settings', 'stage', 'valid'), [
    (Settings(imgsz=0), 'detection', False), (Settings(confidence=float('nan')), 'detection', False),
    (Settings(device='bad'), 'geometry', False), (Settings(padding=-1), 'geometry', False),
    (Settings(mode='bad'), 'geometry', False), (Settings(mode='tiled', tile=383), 'geometry', False),
    (Settings(mode='tiled', overlap=-1), 'geometry', False), (Settings(mode='tiled', overlap=384), 'geometry', False),
    (Settings(mode='tiled', overlap=383), 'geometry', True), (Settings(mode='tiled', overlap=0), 'geometry', True),
    (Settings(tile=0, overlap=-5), 'geometry', True), (Settings(mode='bad', padding=-1), 'detection', True),
    (Settings(imgsz=0), 'geometry', True), (Settings(imgsz=0), 'all', False),
])
def test_stage_validation(settings, stage, valid):
    assert (not validation_errors(settings, stage)) is valid


def test_device_availability_and_mismatch_policy():
    assert validation_errors(Settings(device='cuda'), 'detection', {'cpu'})
    assert not validation_errors(Settings(device='auto'), 'detection', {'cpu'})
    det = detection_provenance()
    args = (det, det.checkpoint_path, det.model_id, det.checkpoint_path)
    assert stage_mismatches(Settings(), *args) == ()
    assert 'requested device' in stage_mismatches(Settings(device='cpu'), *args)
    assert 'loaded model identity' in stage_mismatches(Settings(), det, det.checkpoint_path, 3, det.checkpoint_path)
    assert 'editable checkpoint path' in stage_mismatches(Settings(), det, det.checkpoint_path, 1, '/other.pt')
    assert 'loaded checkpoint' in stage_mismatches(Settings(), det, None, None, det.checkpoint_path)
    assert stage_mismatches(Settings(), None, None, None, '') is None
    geo = geometry_provenance()
    assert stage_mismatches(Settings(tile=128, overlap=999), geo, geo.checkpoint_path, 2, geo.checkpoint_path) == ()
    tiled = geometry_provenance('tiled')
    assert 'tile overlap' in stage_mismatches(Settings(mode='tiled', overlap=95), tiled, tiled.checkpoint_path, 2, tiled.checkpoint_path)
    text = stage_summary('Detection', Settings(), det, det.checkpoint_path, 1, det.checkpoint_path)
    assert 'requested device auto; resolved device cpu' in text
    assert 'no recorded completed configuration' in stage_summary('Geometry', Settings(), None, None, None, '')


class Array:
    def __init__(self, value): self.value = np.asarray(value)
    def cpu(self): return self
    def numpy(self): return self.value


class FakeDetector:
    def predict(self, **kwargs):
        self.kwargs = kwargs
        boxes = SimpleNamespace(xyxy=Array([[1.125, 2.25, 10.5, 12.75]]), conf=Array([.9]), cls=Array([0]))
        return [SimpleNamespace(names={0: 'grid'}, boxes=boxes, speed={})]


class FakePhase:
    def to(self, device): self.device = device; return self
    def eval(self): return self


def test_pipeline_capture_and_geometry_lifecycle(monkeypatch):
    monkeypatch.setattr('recognition_lab.pipeline.device_for', lambda name: 'cpu')
    pipeline = Pipeline(); pipeline.yolo = FakeDetector(); pipeline.yolo_path = Path('/models/yolo.pt'); pipeline.yolo_id = 7
    pipeline.phase = FakePhase(); pipeline.phase_path = Path('/models/phase.pt'); pipeline.phase_id = 8
    detection = pipeline.detect(None, Settings())
    assert detection.detection_provenance == detection_provenance(model_id=7)
    assert detection.geometry_provenance is None and pipeline.yolo.kwargs['device'] == 'cpu'
    # A non-grid detection avoids inference while exercising the actual stage boundary.
    source = Detection('timer', .8, (1, 2, 8, 9))
    detection.detections[:] = [source]
    geometry = pipeline.geometry(None, detection, Settings())
    assert geometry.detections[0] is source
    assert geometry.detection_provenance is detection.detection_provenance
    assert geometry.geometry_provenance == replace(geometry_provenance(), model_id=8, whole_input_size=None)
    again = pipeline.geometry(None, geometry, Settings(mode='tiled'))
    assert again.detection_provenance is detection.detection_provenance
    assert again.geometry_provenance == replace(geometry_provenance('tiled'), model_id=8)
    new_detection = pipeline.detect(None, Settings())
    assert new_detection.geometry_provenance is None
    assert detection.detection_provenance is geometry.detection_provenance


def test_model_reload_identity_and_paths(monkeypatch, tmp_path):
    checkpoint = tmp_path / 'model.pt'; checkpoint.touch()
    class FakeYolo:
        def __init__(self, path): self.names = dict(enumerate(['game_region', 'grid', 'mine_counter', 'reset_button', 'timer']))
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=FakeYolo))
    pipeline = Pipeline(); pipeline.load_yolo(checkpoint)
    first = pipeline.yolo_id
    assert pipeline.yolo_path == checkpoint.resolve()
    pipeline.load_yolo(checkpoint)
    assert pipeline.yolo_id != first
    pipeline.unload_yolo()
    assert pipeline.yolo_id is None and pipeline.yolo_path is None


def test_tile_ranges_and_inspector_provenance():
    assert tile_index_range(None) is None
    grid = GeometryResult(0, Crop((0, 0, 10, 10), False), None, None, None, None, [], phase_data={'observations': []})
    assert tile_index_range(grid) is None
    grid.phase_data['observations'] = [object(), object()]
    assert tile_index_range(grid) == (0, 1)
    assert tile_selection(grid, grid, 1) == ((0, 1), 1)
    assert tile_selection(grid, grid, 2) == ((0, 1), 0)
    assert tile_selection(grid, None, 1) == ((0, 1), 0)
    assert tile_selection(None, grid, 1) == (None, 0)
    det, geo = detection_provenance(), geometry_provenance()
    result = RecognitionResult([Detection('grid', .8, (0, 0, 10, 10))], [grid], {}, det, geo)
    snapshot = ResultSnapshot(result)
    detection_sections = inspector_sections(snapshot, snapshot.roots[0].id)
    geometry_sections = inspector_sections(snapshot, snapshot.roots[0].children[0].id)
    for sections in (detection_sections, geometry_sections):
        properties = {p.name: p.value for p in sections[-1].properties}
        assert properties['Requested device'] == 'auto' and properties['Resolved execution device'] == 'cpu'
    assert result.grids[0] is grid
    assert result.detection_provenance is det and result.geometry_provenance is geo


def test_failed_or_invalid_inference_preserves_previous_result(monkeypatch):
    monkeypatch.setattr('recognition_lab.pipeline.device_for', lambda name: 'cpu')
    class FailedDetector:
        def predict(self, **kwargs): raise RuntimeError('inference failed')
    pipeline = Pipeline(); pipeline.yolo = FailedDetector()
    original = Detection('grid', .8, (1.125, 2, 10, 12))
    previous = RecognitionResult([original], detection_provenance=detection_provenance())
    with pytest.raises(RuntimeError, match='inference failed'):
        pipeline.detect(None, Settings())
    with pytest.raises(ValueError, match='image size'):
        pipeline.detect(None, Settings(imgsz=0))
    assert previous.detections[0] is original
    assert previous.detection_provenance == detection_provenance()
