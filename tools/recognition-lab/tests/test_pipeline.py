from pathlib import Path
import pytest
from recognition_lab.pipeline import Pipeline, Settings
from recognition_lab.results import RecognitionResult, GeometryResult, Crop


def test_missing_models():
    pipeline = Pipeline()
    with pytest.raises(FileNotFoundError): pipeline.load_yolo(Path('/missing/yolo.pt'))
    with pytest.raises(FileNotFoundError): pipeline.load_unet(Path('/missing/unet.pt'), Path('/missing/config.toml'))
    with pytest.raises(FileNotFoundError): pipeline.load_phase(Path('/missing/phase.pt'))


def test_phase_backend_requires_phase_checkpoint():
    pipeline = Pipeline()
    with pytest.raises(RuntimeError, match='PhaseGridUNet'):
        pipeline.geometry(None, RecognitionResult(), Settings(geometry_backend='phase'))


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


def test_phase_overlay_accepts_validated_edges():
    from types import SimpleNamespace
    import numpy as np
    import skia
    from recognition_lab.rendering import draw
    from grid_geometry.preprocessing import ImageTransform
    edges = (np.array([12., 20., 28.]), np.array([12., 20., 28.]), 1.)
    phase = {'global': SimpleNamespace(cell_occupancy=np.ones((2, 2))), 'observations': []}
    grid = GeometryResult(0, Crop((0, 0, 40, 40), False), np.zeros((1, 1, 1)),
                          ImageTransform((40, 40), (40, 40), (40, 40), (0, 0)),
                          SimpleNamespace(status='fitted'), SimpleNamespace(accepted=True, status='accepted'),
                          [edges], phase_data=phase)
    surface = skia.Surface(40, 40)
    draw(surface.getCanvas(), RecognitionResult(grids=[grid]),
         {'boundaries': True, 'centers': True, 'cell_support': True})
