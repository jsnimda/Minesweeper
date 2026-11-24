import csv
from dataclasses import replace
import hashlib
import json
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from grid_geometry import evaluate
from grid_geometry.dataset import GeometryDataset
from grid_geometry.evaluation_artifacts import SampleRecord, hypothesis_record
from grid_geometry.evaluation_metrics import BoardDecoding, BoardEstimate, truth_from_geometry
from grid_geometry.geometry import Geometry
from grid_geometry.lattice import EdgeObservation
from grid_geometry.model import GridUNet
from grid_geometry.preprocessing import ImageTransform
from grid_geometry.training import weighted_mse
from grid_geometry.training_artifacts import CHECKPOINT_VERSION
from grid_geometry.training_config import load_config


@pytest.fixture
def evaluation_run(tmp_path, annotation):
    root = tmp_path / "annotation export"
    split = root / "real-holdout"
    split.mkdir(parents=True)
    for index in range(3):
        position_shift = index / 2
        record = {
            **annotation,
            "board_id": f"board-{index}", "style": "reference-classic" if index else "microsoft",
            "grid_box": [value + position_shift for value in annotation["grid_box"]],
            "x_edges": [value + position_shift for value in annotation["x_edges"]],
            "y_edges": [value + position_shift for value in annotation["y_edges"]],
            "cell_boxes": [[[coordinate + position_shift for coordinate in box] for box in row]
                           for row in annotation["cell_boxes"]],
            "centers": [[[coordinate + position_shift for coordinate in center] for center in row]
                        for row in annotation["centers"]],
        }
        (split / f"scene-{index}.json").write_text(json.dumps(record), encoding="utf-8")
        Image.new("RGB", (31, 19), (index * 75, 155, 20)).save(split / f"scene-{index}.png")
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    settings = checkpoints / "config.toml"
    settings.write_text(
        '[dataset]\nroot = "unused-training-data"\nwidth = 40\nheight = 32\nsigma = 1.25\ninclude_outer = false\n'
        '[training]\nloss_weight = 3\nepochs = 2\n[run]\ndirectory = "training-run"\n',
        encoding="utf-8",
    )
    best, last = checkpoints / "best.pt", checkpoints / "last.pt"
    weights = GridUNet().state_dict()
    for tensor in weights.values():
        tensor.zero_()
    torch.save(weights, best)
    torch.save({
        "format_version": CHECKPOINT_VERSION, "model": weights, "epoch": 2,
        "contract": {
            "dataset": {"width": 40, "height": 32, "sigma": 1.25, "include_outer": False},
            "loss_weight": 3,
        },
    }, last)
    return evaluate.EvaluationConfig(best, root, split.name, tmp_path / "evaluation", device="cpu", batch_size=2), last


def test_best_and_explicit_last_checkpoint_load_strictly(evaluation_run):
    config, last = evaluation_run
    settings = load_config(config.settings_path)
    best_model, best_identity = evaluate.load_evaluation_model(config.checkpoint, "best", settings)
    last_model, last_identity = evaluate.load_evaluation_model(last, "last", settings)
    assert not best_model.training and not last_model.training
    assert last_identity["epoch"] == 2
    assert best_identity["sha256"] == hashlib.sha256(config.checkpoint.read_bytes()).hexdigest()
    assert all(torch.equal(value, last_model.state_dict()[key]) for key, value in best_model.state_dict().items())
    with pytest.raises(ValueError, match="checkpoint-kind last"):
        evaluate.load_evaluation_model(last, "best", settings)
    with pytest.raises(ValueError, match="versioned"):
        evaluate.load_evaluation_model(config.checkpoint, "last", settings)
    with pytest.raises(ValueError, match="sigma"):
        mismatched = replace(settings, dataset=replace(settings.dataset, sigma=2))
        evaluate.load_evaluation_model(last, "last", mismatched)


@pytest.mark.parametrize("changes", [
    {"batch_size": 0}, {"workers": -1}, {"split": "../escape"}, {"decoder": "custom"},
    {"visualizations": -1}, {"checkpoint_kind": "archive"}, {"device": "gpu"},
    {"decoder": "none", "save_hypotheses": True},
])
def test_invalid_evaluation_configuration(evaluation_run, changes):
    config, _ = evaluation_run
    with pytest.raises(ValueError):
        replace(config, **changes)


def test_inference_returns_probabilities_once_with_eval_and_no_grad():
    class Instrumented(nn.Module):
        def forward(self, images):
            assert not self.training
            assert not torch.is_grad_enabled()
            return images * 0

    model = Instrumented().train()
    images = torch.randn(2, 3, 4, 4, requires_grad=True)
    predictions, elapsed = evaluate.predict_batch(model, images, torch.device("cpu"))
    torch.testing.assert_close(predictions, torch.full_like(predictions, 0.5))
    assert not predictions.requires_grad and elapsed >= 0
    rng = torch.Generator().manual_seed(83)
    targets = torch.rand(images.shape, generator=rng)
    channels = evaluate.heatmap_losses(predictions, targets, weight=3)
    direct = weighted_mse(torch.zeros_like(images), targets, weight=3)
    np.testing.assert_allclose(channels.mean(axis=0), direct.channels.detach().numpy(), rtol=1e-6)


def test_accelerator_timing_synchronizes_around_forward(monkeypatch):
    events = []

    class MockImages:
        def to(self, device, *, non_blocking):
            events.append(("transfer", device.type, non_blocking))
            return torch.zeros(1, 3, 2, 2)

    class Model(nn.Module):
        def forward(self, images):
            events.append("forward")
            return images

    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append(("sync", device.type)))
    evaluate.predict_batch(Model(), MockImages(), torch.device("cuda"))
    assert events == [("transfer", "cuda", True), ("sync", "cuda"), "forward", ("sync", "cuda")]


def test_evaluation_outputs_for_named_split_with_uneven_batches_and_exports(evaluation_run):
    config, _ = evaluation_run
    config = replace(config, save_heatmaps=True, save_hypotheses=True)
    result = evaluate.evaluate(config)
    output = config.output_dir
    saved_settings = json.loads((output / "evaluation.json").read_text())
    assert saved_settings["dataset"]["split"] == "real-holdout"
    assert saved_settings["dataset"]["boards"] == 3
    assert saved_settings["input"]["sigma"] == 1.25
    assert saved_settings["input"]["include_outer"] is False
    assert saved_settings["evaluation"]["device"] == "cpu"
    assert (output / "training_config.toml").read_bytes() == config.settings_path.read_bytes()
    assert result == json.loads((output / "metrics.json").read_text())
    assert result["boards"] == 3 and set(result["geometry"]) == {"independent", "joint"}
    assert result["latency"]["inference_batch_ms"]["count"] == 2
    assert result["latency"]["preprocessing_ms"]["count"] == 3
    assert result["latency"]["independent_validation_ms"]["count"] == 3
    for strict in result["validation"].values():
        assert strict["accepted"] == {"count": 0, "total": 3, "rate": 0}
        assert strict["geometry_accuracy_among_accepted"]["rate"] is None
        assert strict["rejection_reasons"]["no_hypothesis"]["count"] == 3
        assert "labelled_completeness" not in strict
    for stats in result["geometry"].values():
        assert stats["statuses"]["insufficient_evidence"]["count"] == 3
        assert stats["center_max_px"]["count"] == 0
        assert stats["visible_endpoints"]["coverage"]["total"] == 12
    dataset = GeometryDataset.from_split_directory(
        config.dataset_root / config.split, canvas_size=(40, 32), sigma=1.25, include_outer=False,
    )
    targets = torch.stack([dataset[index][1] for index in range(3)])
    expected = weighted_mse(torch.zeros_like(targets), targets, 3)
    assert result["heatmaps"]["weighted_mse"] == pytest.approx(float(expected.total))
    with (output / "boards.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6 and {row["inference_batch_size"] for row in rows} == {"1", "2"}
    assert {row["style"] for row in rows} == {"microsoft", "reference-classic"}
    assert rows[0]["truth_pitch_x"] and rows[0]["center_max_px"] == ""
    manifest_path = output / "predictions" / "manifest.jsonl"
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert len(manifest) == 3
    assert manifest[0]["coordinates"]["source_scene_crop_xyxy"]
    first = manifest[0]
    predictions = json.loads((manifest_path.parent / first["hypotheses"]).read_text())
    assert set(predictions) == {"independent", "joint"}
    assert first["validation"]["joint"] == predictions["joint"]["validation"]
    assert predictions["joint"]["status"] == "insufficient_evidence"
    assert predictions["joint"]["validation"]["status"] == "rejected"
    with np.load(manifest_path.parent / first["probabilities"], allow_pickle=False) as archive:
        values = archive["probabilities"]
        assert values.dtype == np.float32 and values.shape == (3, 32, 40)
        np.testing.assert_array_equal(values, 0.5)
    assert result["visualizations"]
    with Image.open(output / "visualizations" / result["visualizations"][0]) as preview:
        assert preview.width > 400 and preview.height > 400
    saved_metrics = (output / "metrics.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        evaluate.evaluate(config)
    assert (output / "metrics.json").read_bytes() == saved_metrics


def test_fitted_but_strictly_rejected_prediction_exports_local_support(evaluation_run, tmp_path, monkeypatch):
    config, _ = evaluation_run
    dataset = GeometryDataset.from_split_directory(config.dataset_root / config.split, canvas_size=(40, 32))
    truth = truth_from_geometry(dataset.load_sample(0).geometry)
    estimate = BoardEstimate(
        truth.origin_x + truth.pitch_x * np.arange(truth.columns + 1),
        truth.origin_y + truth.pitch_y * np.arange(truth.rows + 1),
        tuple(EdgeObservation(position, 1, False) for position in truth.visible_extent),
    )
    historical = BoardDecoding("fitted", (estimate,))
    monkeypatch.setattr(evaluate, "decode_predictions", lambda *_: ({"independent": historical}, {}))
    labels = tmp_path / "labels.json"
    labels.write_text('{"scene-0": false, "scene-1": true}')
    thresholds = tmp_path / "validation.toml"
    thresholds.write_text("[validation]\nmax_boundary_gap = 2\n")
    config = replace(config, decoder="independent", save_hypotheses=True, validation_config=thresholds,
                     completeness_labels=labels)
    result = evaluate.evaluate(config)
    assert result["geometry"]["independent"]["accepted"]["count"] == 3
    stats = result["validation"]["independent"]
    assert stats["accepted"]["count"] == 0 and stats["rejected_correct"]["count"] >= 1
    assert stats["labelled_completeness"]["boards"] == 2
    assert stats["labelled_completeness"]["complete_board_rejection"]["count"] == 1
    settings = json.loads((config.output_dir / "evaluation.json").read_text())
    assert settings["validation"]["thresholds"]["max_boundary_gap"] == 2
    assert settings["validation"]["completeness_labels"]["boards"] == 2
    with (config.output_dir / "boards.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["observation_complete"] for row in rows] == ["False", "True", ""]
    assert rows[0]["status"] == "fitted" and rows[0]["validation_status"] == "rejected"
    assert rows[0]["validation_center_coverage"] == "0.0"
    assert "center_coverage" in json.loads(rows[0]["validation_reasons"])
    predictions = json.loads((config.output_dir / "predictions" / "000000.json").read_text())
    strict = predictions["independent"]["validation"]
    assert strict["support"]["centers"]["supported"] == [[False] * 3] * 2
    assert len(strict["support"]["vertical"]["continuity"]) == 2
    assert any("rejected_correct_geometry" in name for name in result["visualizations"])


def test_historical_mode_and_invalid_labels(evaluation_run, tmp_path):
    config, _ = evaluation_run
    labels = tmp_path / "labels.json"
    labels.write_text('{"wrong-id": true}')
    with pytest.raises(ValueError, match="unknown sample IDs"):
        evaluate.evaluate(replace(config, completeness_labels=labels))
    labels.write_text('{"scene-0": 0}')
    with pytest.raises(ValueError, match="boolean"):
        evaluate.evaluate(replace(config, completeness_labels=labels))
    assert not config.output_dir.exists()
    result = evaluate.evaluate(replace(config, strict_validation=False, visualizations=0))
    assert result["validation"] == {}
    assert not any(name.endswith("_validation_ms") for name in result["latency"])


def test_source_crop_and_scene_export_keep_fractional_and_censored_coordinates(annotation):
    fields = ("size", "rows", "columns", "grid_box", "x_edges", "y_edges", "cell_boxes", "centers")
    geometry = Geometry(**{field: annotation[field] for field in fields})
    transform = ImageTransform((20, 10), (40, 30), (48, 40), (4, 5))
    sample = SampleRecord(0, "sample", "board", None, "microsoft", geometry, transform, (100.25, 40.5, 130.25, 80.5), 0)
    estimate = BoardEstimate(
        np.array([8.5, 13.5, 18.5]), np.array([11.75, 18.75, 25.75]),
        (EdgeObservation(0.5, 1.0, True), None, EdgeObservation(18.25, 0.7, False), None),
    )
    saved = hypothesis_record(estimate, sample)
    np.testing.assert_allclose(saved["source_crop"]["ideal_x_edges"], [2.25, 4.75, 7.25])
    np.testing.assert_allclose(saved["source_scene"]["ideal_x_edges"], [103.625, 107.375, 111.125])
    np.testing.assert_allclose(saved["source_scene"]["ideal_y_edges"], [49.5, 58.8333333333, 68.1666666667])
    assert saved["outer_edges"]["left"]["state"] == "censored"
    assert saved["outer_edges"]["top"] == {"state": "unknown"}
    assert saved["outer_edges"]["right"]["crop_observation"] == pytest.approx(7.125)


def test_cli_heatmap_only_with_spawn_and_no_prediction_exports(evaluation_run):
    config, last = evaluation_run
    args = [
        sys.executable, "-m", "grid_geometry.evaluate", "--checkpoint", str(last), "--checkpoint-kind", "last",
        "--dataset-root", str(config.dataset_root), "--split", config.split, "--output-dir", str(config.output_dir),
        "--device", "cpu", "--batch-size", "2", "--workers", "1", "--decoder", "none", "--visualizations", "0",
    ]
    process = subprocess.run(args, capture_output=True, text=True, timeout=75)
    assert process.returncode == 0, process.stdout + process.stderr
    assert "Evaluated 3 boards" in process.stdout
    result = json.loads((config.output_dir / "metrics.json").read_text())
    assert result["geometry"] == {}
    assert result["validation"] == {}
    assert not any(name.endswith("joint_total_ms") for name in result["latency"])
    assert not (config.output_dir / "predictions").exists()


def test_cli_requires_explicit_paths_and_refuses_unavailable_cuda(evaluation_run, monkeypatch, capsys):
    config, _ = evaluation_run
    with pytest.raises(SystemExit) as error:
        evaluate.main([])
    assert error.value.code == 2
    capsys.readouterr()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert evaluate.main([
        "--checkpoint", str(config.checkpoint), "--dataset-root", str(config.dataset_root), "--split", config.split,
        "--device", "cuda", "--output-dir", str(config.output_dir),
    ]) == 1
    assert "CUDA was requested" in capsys.readouterr().err
    assert not config.output_dir.exists()
