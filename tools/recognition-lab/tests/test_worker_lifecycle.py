"""Exercise worker delivery guards without starting a thread or Qt application."""
from types import SimpleNamespace

from recognition_lab.pipeline import Settings
from recognition_lab.provenance import DetectionProvenance, GeometryProvenance, stage_mismatches, tile_selection
from recognition_lab.recognition_state import RecognitionSelection
from recognition_lab.results import Crop, Detection, GeometryResult, RecognitionResult
from recognition_lab.ui import MainWindow, Worker


class SignalSpy:
    def __init__(self, callback):
        self.callback = callback
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)
        self.callback(*args)


def result_fixture():
    detection = Detection('grid', .9, (1.25, 2.5, 20.75, 22.5))
    det = DetectionProvenance('auto', 'cpu', '/model/yolo.pt', 1, 1280, .25)
    geo = GeometryProvenance('auto', 'cpu', '/model/phase.pt', 2, 20, 'tiled', 384, 96)
    grid = GeometryResult(0, Crop((0, 0, 24, 24), False), None, None, None, None, [],
                          phase_data={'observations': [object(), object()]})
    return RecognitionResult([detection], [grid], detection_provenance=det, geometry_provenance=geo)


def test_failed_worker_does_not_replace_snapshot(monkeypatch):
    state = RecognitionSelection(); previous = result_fixture()
    state.replace(previous); selected = state.snapshot.roots[0].id; state.select(selected)
    before = state.snapshot
    warnings = []
    monkeypatch.setattr('recognition_lab.ui.QMessageBox',
                        SimpleNamespace(warning=lambda parent, title, error: warnings.append(error)))
    window = SimpleNamespace(guard_generation=True, generation=4,
                             worker_done=lambda result: state.replace(result))
    def fail():
        raise RuntimeError('inference failed')
    worker = SimpleNamespace(fn=fail, generation=4,
                             succeeded=SignalSpy(lambda result, generation: MainWindow.worker_succeeded(window, result, generation)),
                             failed=SignalSpy(lambda error: MainWindow.worker_failed(window, error)))
    Worker.run(worker)
    assert not worker.succeeded.calls
    assert warnings == ['RuntimeError: inference failed']
    assert state.snapshot is before and state.resolve() is previous.detections[0]
    assert state.selected == selected and state.snapshot.result.geometry_provenance is previous.geometry_provenance


def test_obsolete_worker_success_does_not_replace_current_snapshot():
    state = RecognitionSelection(); current = result_fixture()
    state.replace(current); selected = state.snapshot.roots[0].id; state.select(selected)
    before = state.snapshot
    window = SimpleNamespace(guard_generation=True, generation=5,
                             worker_done=lambda result: state.replace(result))
    stale = result_fixture()
    worker = SimpleNamespace(fn=lambda: stale, generation=4,
                             succeeded=SignalSpy(lambda result, generation: MainWindow.worker_succeeded(window, result, generation)),
                             failed=SignalSpy(lambda error: None))
    Worker.run(worker)
    assert worker.succeeded.calls == [(stale, 4)] and not worker.failed.calls
    assert state.snapshot is before and state.selected == selected
    assert state.snapshot.result is current and state.snapshot.result.detection_provenance is current.detection_provenance


def test_geometry_update_preserves_detection_provenance_selection_and_resets_tile():
    state = RecognitionSelection(); previous = result_fixture()
    state.replace(previous); old_grid = previous.grids[0]
    state.select(state.snapshot.roots[0].children[0].id)
    next_geometry = GeometryProvenance('cpu', 'cpu', '/model/new-phase.pt', 3, 12, 'whole', whole_input_size=(384, 384))
    new_grid = GeometryResult(0, Crop((0, 0, 24, 24), False), None, None, None, None, [],
                              phase_data={'observations': [object(), object(), object()]})
    updated = RecognitionResult(list(previous.detections), [new_grid],
                                detection_provenance=previous.detection_provenance,
                                geometry_provenance=next_geometry)
    state.replace(updated, geometry_update=True)
    assert state.resolve() is previous.detections[0]
    assert state.snapshot.result.detection_provenance is previous.detection_provenance
    assert state.snapshot.result.geometry_provenance is next_geometry
    assert tile_selection(state.grid(), old_grid, 1) == ((0, 2), 0)


def test_pending_settings_unload_and_replacement_preserve_snapshot():
    state = RecognitionSelection(); previous = result_fixture()
    state.replace(previous); identity = state.snapshot.roots[0].id; state.select(identity)
    snapshot = state.snapshot
    provenance = previous.detection_provenance
    assert 'confidence' in stage_mismatches(Settings(confidence=.5), provenance,
                                            provenance.checkpoint_path, 1, provenance.checkpoint_path)
    assert 'loaded model identity' in stage_mismatches(Settings(), provenance, None, None, provenance.checkpoint_path)
    assert 'loaded model identity' in stage_mismatches(Settings(), provenance,
                                                        provenance.checkpoint_path, 9, provenance.checkpoint_path)
    assert state.snapshot is snapshot and state.selected == identity
    assert state.resolve() is previous.detections[0] and previous.detection_provenance is provenance
