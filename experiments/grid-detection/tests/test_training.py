import json
from types import SimpleNamespace

import pytest
import yaml

from grid_detection import training


@pytest.fixture
def training_call(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "validate_training_dataset", lambda root: ({"revision": 7}, []))
    monkeypatch.setattr(training.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(training, "run_provenance", lambda *args: {"revision": "test"})
    return dict(dataset_name="data", run_name="run", preset="smoke", requested_device="auto",
                device="cpu", model_path=tmp_path / "weights" / "yolo.pt",
                training={"epochs": 2, "device": "cpu", "amp": False},
                data_dir=tmp_path / "datasets", runs_dir=tmp_path / "runs", project_root=tmp_path)


def test_existing_run_is_protected(training_call, monkeypatch):
    run_dir = training_call["runs_dir"] / "run"
    run_dir.mkdir(parents=True)
    marker = run_dir / "experiment.json"
    marker.write_text("original")
    monkeypatch.setattr(training, "YOLO", lambda path: pytest.fail("model initialized"))
    with pytest.raises(FileExistsError):
        training.train_model(**training_call)
    assert marker.read_text() == "original"


def test_metadata_records_requested_and_effective_settings(training_call, monkeypatch):
    class Model:
        def __init__(self, path):
            assert path == str(training_call["model_path"])

        def add_callback(self, event, callback):
            assert event == "on_train_start"
            self.callback = callback

        def train(self, **options):
            metadata = json.loads((training_call["runs_dir"] / "run" / "experiment.json").read_text())
            assert metadata["resolved_device"] is None
            assert options == metadata["training_options"]
            self.callback(SimpleNamespace(device="cpu", amp=False, args=SimpleNamespace(workers=0)))

    monkeypatch.setattr(training, "YOLO", Model)
    run_dir = training.train_model(**training_call)
    metadata = json.loads((run_dir / "experiment.json").read_text())
    assert metadata["device_requested"] == "auto"
    assert metadata["device_selected"] == "cpu"
    assert metadata["amp_requested"] is False
    assert (metadata["resolved_device"], metadata["resolved_amp"], metadata["resolved_workers"]) == ("cpu", False, 0)
    assert metadata["dataset_config"] == {"revision": 7}
    assert metadata["training_options"]["epochs"] == 2


def test_failed_initialization_keeps_reserved_run(training_call, monkeypatch):
    def fail(path):
        raise RuntimeError("initialization failed")
    monkeypatch.setattr(training, "YOLO", fail)
    with pytest.raises(RuntimeError, match="initialization failed"):
        training.train_model(**training_call)
    assert (training_call["runs_dir"] / "run" / "experiment.json").is_file()


def test_optional_augmentation_rejected_before_reservation(training_call, monkeypatch):
    monkeypatch.setattr(training.importlib.util, "find_spec", lambda name: object())
    with pytest.raises(RuntimeError, match="Albumentations"):
        training.train_model(**training_call)
    assert not (training_call["runs_dir"] / "run").exists()


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "manifest.jsonl").write_text('{"split": "val"}\n')
    directory = tmp_path / "runs" / "run"
    directory.mkdir(parents=True)
    (directory / "experiment.json").write_text(json.dumps({"dataset_path": str(root), "dataset_config": {"revision": 7}}))
    (directory / "args.yaml").write_text(yaml.safe_dump({"imgsz": 1280, "batch": 8}))
    (directory / "results.csv").write_text(' epoch, train/loss\n 1, 0.5\n')
    monkeypatch.setattr(training, "validate_saved_dataset", lambda path: {"revision": 7})
    monkeypatch.setattr(training, "validate_descriptor_path", lambda path: None)
    return tmp_path / "runs", root, directory


def test_load_saved_run_without_checkpoints(saved_run):
    runs, root, directory = saved_run
    run = training.load_saved_run(runs, "run")
    assert run.dataset_root == root.resolve()
    assert run.records == [{"split": "val"}]
    assert run.history == [{"epoch": "1", "train/loss": "0.5"}]
    assert run.args["imgsz"] == 1280
    assert run.best_path == directory / "weights" / "best.pt"
    assert run.last_path == directory / "weights" / "last.pt"
    assert not run.best_path.exists()


def test_load_rejects_changed_config_and_descriptor(saved_run, monkeypatch):
    runs, _, _ = saved_run
    monkeypatch.setattr(training, "validate_saved_dataset", lambda path: {"revision": 8})
    with pytest.raises(ValueError, match="configuration differs"):
        training.load_saved_run(runs, "run")
    monkeypatch.setattr(training, "validate_saved_dataset", lambda path: {"revision": 7})
    def reject(path):
        raise ValueError("descriptor mismatch")
    monkeypatch.setattr(training, "validate_descriptor_path", reject)
    with pytest.raises(ValueError, match="descriptor mismatch"):
        training.load_saved_run(runs, "run")


def test_load_requires_history(saved_run):
    runs, _, directory = saved_run
    (directory / "results.csv").write_text('epoch,loss\n')
    with pytest.raises(ValueError, match="No training history"):
        training.load_saved_run(runs, "run")
