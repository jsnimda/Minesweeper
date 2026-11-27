"""Prepare notebook training runs and inspect saved phase experiments."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from ..dataset.validation import validate_saved_dataset, validate_training_dataset
from .model import MODEL_ID, REPRESENTATION_ID
from .train import Config, dataset_identity


def project_directory(cwd):
    """Find the project when invoked from its root or notebook directory."""
    project = Path(cwd).resolve()
    if project.name == "notebooks":
        project = project.parent
    if not (project / "src" / "grid_geometry").is_dir():
        raise RuntimeError("Open the notebook from the grid-geometry project root or notebooks directory")
    return project


def _directory_name(name, label):
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(f"Use a single {label} directory name: {name!r}")


def training_config(project, dataset_name, run_name, training):
    """Construct the trainer's native validated configuration."""
    _directory_name(dataset_name, "dataset")
    _directory_name(run_name, "run")
    project = Path(project)
    return Config(dataset_root=str((project / "datasets" / dataset_name).resolve()),
                  output_dir=str((project / "runs" / run_name).resolve()), **training)


def prepare_training(config):
    """Protect a fresh run and validate its dataset before invoking the trainer."""
    run_dir = Path(config.output_dir)
    if run_dir.exists():
        raise FileExistsError(f"Choose a fresh run name: {run_dir}")
    verification = validate_training_dataset(Path(config.dataset_root))
    print("Split counts: " + ", ".join(f"{k}={v}" for k, v in verification["split_counts"].items()))
    print(f"Training destination: {run_dir}")
    return verification


@dataclass
class SavedRun:
    directory: Path
    last_path: Path
    best_path: Path
    state: dict
    best_state: dict
    config: dict
    dataset_root: Path
    dataset_config: dict
    history: list
    outcomes: dict


def load_saved_run(runs_dir, run_name):
    """Load checkpoints and check dataset and representation compatibility."""
    _directory_name(run_name, "run")
    directory = (Path(runs_dir) / run_name).resolve()
    last_path, best_path = directory / "last.pt", directory / "best.pt"
    for path in (last_path, best_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    state = torch.load(last_path, map_location="cpu", weights_only=False)
    config = state.get("config")
    if not isinstance(config, dict) or not config.get("dataset_root"):
        raise ValueError("Saved checkpoint lacks dataset configuration")
    if not isinstance(state.get("dataset_identity"), str) or not state["dataset_identity"]:
        raise ValueError("Saved checkpoint lacks dataset identity")
    root = Path(config["dataset_root"]).resolve()
    dataset_config = validate_saved_dataset(root)
    if tuple(dataset_config["input_size"]) != (384, 384):
        raise ValueError("This workflow requires the model-compatible 384 x 384 input size")
    if dataset_identity(root) != state["dataset_identity"]:
        raise ValueError("Saved dataset identity differs from the current dataset inventory")
    best_state = torch.load(best_path, map_location="cpu", weights_only=False)
    if best_state.get("dataset_identity") != state["dataset_identity"]:
        raise ValueError("Best and last checkpoints have different dataset identities")
    for checkpoint in (state, best_state):
        if checkpoint.get("model_id") != MODEL_ID or checkpoint.get("representation_id") != REPRESENTATION_ID:
            raise ValueError("checkpoint is not a PhaseGridUNet phase-occupancy checkpoint")
    history = state["history"]
    if not history:
        raise ValueError("Saved checkpoint has no training history")
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    outcomes = {
        "completed_epochs": summary.get("completed_epochs", state.get("epoch")),
        "best_epoch": summary.get("best_epoch", state.get("best_epoch")),
        "best_value": summary.get("best_value", state.get("best")),
        "stopping_reason": summary.get("stopping_reason", state.get("stopping_reason")),
        "device_requested": summary.get("device_requested", config.get("device")),
        "device_effective": summary.get("device_effective", state.get("environment", {}).get("device")),
        "amp_requested": summary.get("amp_requested", config.get("amp")),
        "amp_effective": summary.get("amp_effective", state.get("amp_effective")),
        "total_training_seconds": summary.get("total_training_seconds"),
        "aggregate_samples_per_second": summary.get("aggregate_samples_per_second"),
        "successful_updates": summary.get("successful_updates"),
        "skipped_updates": summary.get("skipped_updates"),
    }
    return SavedRun(directory, last_path, best_path, state, best_state, config, root,
                    dataset_config, history, outcomes)
