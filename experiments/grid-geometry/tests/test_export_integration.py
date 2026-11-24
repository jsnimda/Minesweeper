from dataclasses import asdict
from importlib import import_module
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from grid_geometry.dataset import GeometryDataset
from grid_geometry.heatmaps import geometry_heatmaps
from grid_geometry.model import GridUNet


@pytest.fixture
def diagnostic_root():
    project = Path(__file__).resolve().parents[1]
    path = os.environ.get("GRID_GEOMETRY_DIAGNOSTICS")
    if path is not None:
        return Path(path)
    default = project / "outputs" / "geometry-diagnostics" / "geometry"
    if not default.is_dir():
        pytest.skip("Export diagnostic samples with scripts/export_geometry_diagnostics.py")
    return default


def test_actual_export_matches_reference_heatmaps(diagnostic_root, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "synthetic-scenes" / "src"
    monkeypatch.syspath_prepend(str(source))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    reference = import_module("synthetic_scenes.exporters")

    dataset = GeometryDataset(diagnostic_root, "train", canvas_size=(96, 80))
    samples = [dataset.load_sample(index) for index in range(len(dataset))]
    assert {sample.style for sample in samples} == {"classic", "windows7", "microsoft", "google"}
    expert = next(sample for sample in samples if sample.style == "microsoft")
    assert (expert.geometry.rows, expert.geometry.columns) == (16, 30)
    widths = np.diff(expert.geometry.x_edges)
    assert widths[0] < widths[1]
    assert widths[-1] < widths[-2]
    assert np.any(expert.source_geometry.centers % 1 != 0)
    validation = GeometryDataset(diagnostic_root, "val").load_sample(0)
    assert validation.board_id not in {sample.board_id for sample in samples}

    for sample in samples:
        for geometry in (sample.source_geometry, sample.geometry):
            annotation = asdict(geometry)
            annotation["x_edges"] = geometry.x_edges.tolist()
            annotation["y_edges"] = geometry.y_edges.tolist()
            for include_outer in (True, False):
                target = geometry_heatmaps(geometry, sigma=2.0, include_outer=include_outer)
                expected = reference.geometry_heatmaps(
                    annotation, sigma=2.0, include_outer=include_outer,
                )
                np.testing.assert_array_equal(target, expected)


def test_actual_export_cpu_forward_backward(diagnostic_root):
    dataset = GeometryDataset(diagnostic_root, "train", canvas_size=(80, 64))
    images, targets = next(iter(DataLoader(dataset, batch_size=2)))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = GridUNet().train()

    predictions = model(images).sigmoid()
    weights = 1.0 + 10.0 * targets
    loss = (weights * (predictions - targets).square()).mean()
    loss.backward()

    assert predictions.shape == targets.shape == (2, 3, 64, 80)
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(model.enc1.layers[0].weight.grad) > 0
    assert torch.count_nonzero(model.segmentation_head.weight.grad) > 0
