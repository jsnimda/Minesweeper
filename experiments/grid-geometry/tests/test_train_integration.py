import csv
from dataclasses import replace
import json
import subprocess
import sys

from PIL import Image
import pytest
import torch

from grid_geometry import train
from grid_geometry.model import GridUNet
from grid_geometry.training import seed_everything
from grid_geometry.training_artifacts import HISTORY_FIELDS, load_last, own_run
from grid_geometry.training_config import Config, DatasetConfig, RunConfig, TrainingConfig, load_config


@pytest.fixture
def tiny_exports(tmp_path, annotation):
    root = tmp_path / "geometry"
    for split, count in (("train", 3), ("val", 2)):
        folder = root / split
        folder.mkdir(parents=True)
        for index in range(count):
            metadata = {**annotation, "board_id": f"{split}-{index}"}
            (folder / f"{index}.json").write_text(json.dumps(metadata), encoding="utf-8")
            Image.new("RGB", (31, 19), (index * 95, 180 - index * 25, 50)).save(folder / f"{index}.png")
    return root


def _config(root, directory, *, epochs=2):
    return Config(
        DatasetConfig(root, width=32, height=24),
        TrainingConfig(batch_size=2, epochs=epochs, device="cpu", log_every_batches=1),
        RunConfig(directory, archive_every=1),
    )


def _metrics(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == HISTORY_FIELDS
        return list(reader)


def _quiet(_message):
    pass


def test_interrupted_run_resumes_with_portable_state_and_repaired_history(tiny_exports, tmp_path):
    config = _config(tiny_exports, tmp_path / "interrupted")
    directory = config.run.directory
    seed_everything(config.training.seed)
    initial = GridUNet().state_dict()
    inside_second_epoch = False

    def interrupt_after_update(message):
        nonlocal inside_second_epoch
        if message == "Epoch 2/2":
            inside_second_epoch = True
        if inside_second_epoch and "train batch 1/2" in message:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        train.train_model(config, report=interrupt_after_update)
    assert not (directory / ".training.lock").exists()
    first = load_last(directory)
    assert first["epoch"] == 1
    assert len(first["optimizer"]["state"]) > 0
    assert all(value["step"].item() == 2 for value in first["optimizer"]["state"].values())
    assert any(not torch.equal(initial[key], value) for key, value in first["model"].items())
    assert (directory / "epoch-0001.pt").exists()

    # A crash between publishing the last checkpoint and derived files can
    # leave their contents out of sync. Resume uses the checkpoint's history.
    (directory / "history.csv").write_text("incomplete\n", encoding="utf-8")
    (directory / "best.pt").write_bytes(b"incomplete")
    resumed = replace(config, run=replace(config.run, resume=True))
    assert train.train_model(resumed, report=_quiet) == 2
    last = load_last(directory)
    metrics = _metrics(directory / "history.csv")
    assert [int(row["epoch"]) for row in metrics] == [1, 2]
    assert all(value["step"].item() == 4 for value in last["optimizer"]["state"].values())
    assert last["scaler"] is None
    assert last["best_epoch"] == min(last["history"], key=lambda row: row["validation_loss"])["epoch"]
    assert last["best_validation_loss"] == min(float(row["validation_loss"]) for row in metrics)
    assert int(metrics[0]["train_samples"]) == 3 and int(metrics[0]["validation_samples"]) == 2
    assert (directory / "epoch-0002.pt").is_file()

    state_dict = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    GridUNet().load_state_dict(state_dict, strict=True)
    assert set(state_dict) == set(last["model"])
    assert all(tensor.device.type == "cpu" for tensor in state_dict.values())
    assert all(torch.equal(tensor, last["best_model"][name]) for name, tensor in state_dict.items())
    assert all(value["exp_avg"].device.type == "cpu" for value in last["optimizer"]["state"].values())

    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["initialization"] == "scratch" and metadata["seed"] == 42
    assert [session["resume_epoch"] for session in metadata["sessions"]] == [None, 1]
    assert metadata["sessions"][0]["dataset"]["train_samples"] == 3
    assert metadata["sessions"][0]["device"] == "cpu"
    assert metadata["sessions"][0]["pytorch"] == str(torch.__version__)
    assert load_config(directory / "config.toml") == resumed

    uninterrupted = _config(tiny_exports, tmp_path / "uninterrupted")
    train.train_model(uninterrupted, report=_quiet)
    reference = load_last(uninterrupted.run.directory)
    assert last["history"] == reference["history"]
    assert all(torch.equal(tensor, reference["model"][name]) for name, tensor in last["model"].items())
    assert all(
        torch.equal(state["exp_avg"], reference["optimizer"]["state"][key]["exp_avg"])
        for key, state in last["optimizer"]["state"].items()
    )

    checkpoint_bytes = (directory / "last.pt").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        train.train_model(config, report=_quiet)
    assert (directory / "last.pt").read_bytes() == checkpoint_bytes

    different_weight = replace(resumed, training=replace(config.training, loss_weight=2))
    with pytest.raises(ValueError, match="loss_weight"):
        train.train_model(different_weight, report=_quiet)
    annotation = tiny_exports / "train" / "0.json"
    annotation.write_bytes(annotation.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="inventory_sha256"):
        train.train_model(resumed, report=_quiet)


def test_output_lock_refuses_second_owner(tmp_path):
    directory = tmp_path / "locked"
    with own_run(RunConfig(directory)):
        with pytest.raises(ValueError, match="run is locked"):
            with own_run(RunConfig(directory, resume=True)):
                pytest.fail("run was already owned")
        assert (directory / ".training.lock").exists()
    assert directory.is_dir() and not (directory / ".training.lock").exists()


def _write_config(path, *, workers=0):
    path.write_text(
        '[dataset]\nroot = "geometry"\nwidth = 32\nheight = 24\n'
        f'[training]\nworkers = {workers}\nbatch_size = 2\nepochs = 1\ndevice = "cpu"\n'
        '[run]\ndirectory = "default-run"\n', encoding="utf-8",
    )


def test_module_cli_and_spawned_workers(tiny_exports, tmp_path):
    path = tmp_path / "smoke.toml"
    _write_config(path, workers=1)
    directory = tmp_path / "run with spaces"
    args = [sys.executable, "-m", "grid_geometry.train", "--config", str(path), "--output-dir", str(directory)]
    process = subprocess.run(args, capture_output=True, text=True, timeout=75)
    assert process.returncode == 0, process.stdout + process.stderr
    assert "Epoch 1/1" in process.stdout and "validation loss" in process.stdout
    assert load_last(directory)["epoch"] == 1
    assert not (tmp_path / "default-run").exists()


def test_cli_argument_validation_and_interrupt_exit(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.toml"
    _write_config(path)
    with pytest.raises(SystemExit) as error:
        train.main(["--device", "gpu"])
    assert error.value.code == 2
    assert train.main(["--config", str(path), "--epochs", "0"]) == 1
    assert "epochs" in capsys.readouterr().err

    def interrupt(config, *, report):
        assert config.run.resume is True
        assert config.training.epochs == 3
        assert config.dataset.root == (tmp_path / "alternate").resolve()
        raise KeyboardInterrupt

    monkeypatch.setattr(train, "train_model", interrupt)
    assert train.main([
        "--config", str(path), "--resume", "--epochs", "3", "--dataset-root", str(tmp_path / "alternate"),
    ]) == 130
    assert "last committed epoch" in capsys.readouterr().err


def test_explicit_cpu_rejects_cuda_mixed_precision(tiny_exports, tmp_path):
    config = _config(tiny_exports, tmp_path / "unsupported")
    config = replace(config, training=replace(config.training, mixed_precision=True))
    with pytest.raises(ValueError, match="requires a CUDA"):
        train.train_model(config, report=_quiet)
    assert not config.run.directory.exists()
