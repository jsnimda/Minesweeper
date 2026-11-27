import json

import pytest

from grid_geometry.phase import runs
from grid_geometry.phase.model import MODEL_ID, REPRESENTATION_ID


def test_training_config_and_fresh_run_preflight(tmp_path, monkeypatch):
    config = runs.training_config(tmp_path, "smoke", "run", {"epochs": 2, "batch_size": 8, "amp": True})
    assert config.dataset_root == str((tmp_path / "datasets" / "smoke").resolve())
    assert config.output_dir == str((tmp_path / "runs" / "run").resolve())
    assert (config.epochs, config.batch_size, config.amp) == (2, 8, True)
    seen = []
    monkeypatch.setattr(runs, "validate_training_dataset", lambda root: seen.append(root) or {"split_counts": {"train": 1}})
    assert runs.prepare_training(config)["split_counts"] == {"train": 1}
    assert seen == [tmp_path / "datasets" / "smoke"]
    (tmp_path / "runs" / "run").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="fresh run"):
        runs.prepare_training(config)
    assert len(seen) == 1


@pytest.mark.parametrize("dataset,run", [("../data", "run"), ("data", "../run"), (".", "run")])
def test_training_config_rejects_paths(tmp_path, dataset, run):
    with pytest.raises(ValueError, match="single"):
        runs.training_config(tmp_path, dataset, run, {})


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    directory = tmp_path / "runs" / "run"
    directory.mkdir(parents=True)
    for name in ("last.pt", "best.pt"):
        (directory / name).touch()
    last = {"config": {"dataset_root": str(root), "device": "auto", "amp": True},
            "dataset_identity": "identity", "model_id": MODEL_ID, "representation_id": REPRESENTATION_ID,
            "history": [{"epoch": 1}], "epoch": 1, "best_epoch": 1, "best": 0.5,
            "environment": {"device": "cpu"}, "amp_effective": False}
    best = {"dataset_identity": "identity", "model_id": MODEL_ID, "representation_id": REPRESENTATION_ID}
    monkeypatch.setattr(runs.torch, "load", lambda path, **kwargs: last if path.name == "last.pt" else best)
    monkeypatch.setattr(runs, "validate_saved_dataset", lambda path: {"input_size": [384, 384]})
    monkeypatch.setattr(runs, "dataset_identity", lambda path: "identity")
    return tmp_path / "runs", root, directory, last, best


def test_load_without_training_or_optional_summary(saved_run):
    root_dir, root, directory, last, best = saved_run
    run = runs.load_saved_run(root_dir, "run")
    assert run.dataset_root == root.resolve()
    assert run.state is last and run.best_state is best
    assert run.last_path == directory / "last.pt" and run.best_path == directory / "best.pt"
    assert run.history == [{"epoch": 1}]
    assert run.outcomes["completed_epochs"] == 1 and run.outcomes["best_value"] == 0.5
    assert run.outcomes["device_requested"] == "auto" and run.outcomes["device_effective"] == "cpu"
    assert run.outcomes["amp_requested"] is True and run.outcomes["amp_effective"] is False
    assert run.outcomes["aggregate_samples_per_second"] is None
    assert run.outcomes["successful_updates"] is None


def test_saved_summary_takes_precedence(saved_run):
    root_dir, _, directory, _, _ = saved_run
    (directory / "summary.json").write_text(json.dumps({"stopping_reason": "patience", "best_epoch": 2,
                                                       "aggregate_samples_per_second": {"train": 5.0}}))
    outcomes = runs.load_saved_run(root_dir, "run").outcomes
    assert outcomes["stopping_reason"] == "patience" and outcomes["best_epoch"] == 2
    assert outcomes["aggregate_samples_per_second"] == {"train": 5.0}


@pytest.mark.parametrize("checkpoint,key,value", [
    ("last", "model_id", "other"), ("best", "model_id", "other"),
    ("last", "representation_id", "other"), ("best", "representation_id", "other"),
    ("last", "dataset_identity", ""), ("best", "dataset_identity", "other"),
])
def test_rejects_checkpoint_identity_mismatch(saved_run, checkpoint, key, value):
    root_dir, _, _, last, best = saved_run
    (last if checkpoint == "last" else best)[key] = value
    with pytest.raises(ValueError):
        runs.load_saved_run(root_dir, "run")


def test_rejects_dataset_changes_and_missing_history(saved_run, monkeypatch):
    root_dir, _, _, last, _ = saved_run
    monkeypatch.setattr(runs, "dataset_identity", lambda path: "changed")
    with pytest.raises(ValueError, match="dataset identity"):
        runs.load_saved_run(root_dir, "run")
    monkeypatch.setattr(runs, "dataset_identity", lambda path: "identity")
    monkeypatch.setattr(runs, "validate_saved_dataset", lambda path: {"input_size": [256, 256]})
    with pytest.raises(ValueError, match="384"):
        runs.load_saved_run(root_dir, "run")
    monkeypatch.setattr(runs, "validate_saved_dataset", lambda path: {"input_size": [384, 384]})
    last["history"] = []
    with pytest.raises(ValueError, match="no training history"):
        runs.load_saved_run(root_dir, "run")


def test_missing_checkpoint_and_dataset_config(saved_run):
    root_dir, _, directory, last, _ = saved_run
    (directory / "best.pt").unlink()
    with pytest.raises(FileNotFoundError):
        runs.load_saved_run(root_dir, "run")
    (directory / "best.pt").touch()
    last["config"] = {}
    with pytest.raises(ValueError, match="dataset configuration"):
        runs.load_saved_run(root_dir, "run")
