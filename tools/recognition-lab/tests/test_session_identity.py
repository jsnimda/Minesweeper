import hashlib
import sys
from types import SimpleNamespace

import pytest

from recognition_lab.model_identity import fingerprint_before_load, identity_after_load
from recognition_lab.pipeline import Pipeline, Settings
from recognition_lab.session_log import SessionLog


def test_log_capacity_order_and_clear():
    log = SessionLog(capacity=2)
    log.add('info', 'load', 'first', 1)
    log.add('warning', 'recognition', 'second', 2)
    log.add('error', 'capture', 'third')
    text = log.text()
    assert 'first' not in text and text.index('second') < text.index('third')
    assert '| WARNING | recognition #2 | second' in text
    assert '| ERROR | capture | third' in text
    log.clear()
    assert log.text() == ''


def test_fingerprint_and_changed_or_missing_checkpoint(tmp_path):
    path = tmp_path / 'checkpoint'; path.write_bytes(b'weights')
    digest, signature, status = fingerprint_before_load(path)
    assert digest == hashlib.sha256(b'weights').hexdigest() and status == 'available'
    identity = identity_after_load(path, digest, signature, status, object())
    assert identity.sha256 == digest and identity.parameters is None
    path.write_bytes(b'other weights')
    changed = identity_after_load(path, digest, signature, status, object())
    assert changed.sha256 is None and 'changed' in changed.fingerprint_status
    path.unlink()
    missing = identity_after_load(path, digest, signature, status, object())
    assert missing.sha256 is None and 'unavailable' in missing.fingerprint_status
    assert fingerprint_before_load(path)[0] is None


def test_replacement_failure_and_completed_provenance_keep_identity(monkeypatch, tmp_path):
    path = tmp_path / 'checkpoint'; path.write_bytes(b'first')
    names = dict(enumerate(['game_region', 'grid', 'mine_counter', 'reset_button', 'timer']))
    class FakeYolo:
        def __init__(self, path): self.names = names
        def predict(self, **kwargs):
            class Array:
                def cpu(self): return self
                def numpy(self): return []
            return [SimpleNamespace(names={}, boxes=SimpleNamespace(xyxy=Array(), conf=Array(), cls=Array()), speed={})]
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=FakeYolo))
    monkeypatch.setattr('recognition_lab.pipeline.device_for', lambda device: 'cpu')
    pipeline = Pipeline(); pipeline.load_yolo(path)
    old_model, old_identity, old_id = pipeline.yolo, pipeline.yolo_identity, pipeline.yolo_id
    result = pipeline.detect(None, Settings())
    path.write_bytes(b'second')
    names[0] = 'wrong'
    with pytest.raises(ValueError): pipeline.load_yolo(path)
    assert (pipeline.yolo, pipeline.yolo_identity, pipeline.yolo_id) == (old_model, old_identity, old_id)
    names[0] = 'game_region'
    pipeline.load_yolo(path)
    assert result.detection_provenance.checkpoint_sha256 == hashlib.sha256(b'first').hexdigest()
    assert pipeline.yolo_identity.sha256 == hashlib.sha256(b'second').hexdigest()
