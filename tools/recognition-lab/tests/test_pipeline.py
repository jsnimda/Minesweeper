from pathlib import Path
import pytest
from recognition_lab.pipeline import Pipeline
from recognition_lab.results import RecognitionResult


def test_missing_models():
    pipeline = Pipeline()
    with pytest.raises(FileNotFoundError): pipeline.load_yolo(Path('/missing/yolo.pt'))
    with pytest.raises(FileNotFoundError): pipeline.load_unet(Path('/missing/unet.pt'), Path('/missing/config.toml'))


def test_empty_detection_result():
    result = RecognitionResult()
    assert not result.detections and not result.grids
