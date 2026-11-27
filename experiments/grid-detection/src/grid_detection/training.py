"""Training and saved-run support for the detection experiment."""

import csv
import importlib.util
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import TORCH_2_0

from grid_detection.dataset.validation import (
    validate_descriptor_path,
    validate_saved_dataset,
    validate_training_dataset,
)
from synthetic_scenes.provenance import run_provenance


def validate_directory_name(name, label):
    """Reject paths where a single dataset or run directory name is expected."""
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(f"Use a single {label} directory name: {name!r}")


def resolve_device(requested):
    value = str(requested).lower().strip()
    mps_available = bool(TORCH_2_0 and hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if value == "auto":
        value = "0" if torch.cuda.is_available() else "mps" if mps_available else "cpu"
    if value == "cpu":
        return "cpu"
    if value in {"mps", "mps:0"}:
        if not mps_available:
            raise RuntimeError("MPS requested but unavailable or unsupported by this Ultralytics/PyTorch version")
        return "mps"
    if value == "cuda":
        value = "0"
    if value.startswith("cuda:"):
        value = value[5:]
    if value.isdigit():
        if not torch.cuda.is_available() or int(value) >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {value} requested but unavailable")
        return str(int(value))
    raise ValueError(f"Unsupported device: {requested!r}; choose auto, cpu, mps, or a single CUDA device")


def prepare_training_options(training, device):
    """Add device-dependent settings and validate native Ultralytics options."""
    options = dict(training, device=device, amp=device not in {"cpu", "mps"})
    get_cfg(overrides=options)
    return options


def train_model(*, dataset_name, run_name, preset, requested_device, device,
                model_path, training, data_dir, runs_dir, project_root):
    """Validate and reserve a run, record provenance, then invoke native training."""
    validate_directory_name(dataset_name, "dataset")
    validate_directory_name(run_name, "run")
    dataset_root = (Path(data_dir) / dataset_name).resolve()
    dataset_config, _ = validate_training_dataset(dataset_root)
    if importlib.util.find_spec("albumentations") is not None:
        raise RuntimeError("Remove optional Albumentations to preserve the documented augmentation policy")
    run_dir = Path(runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    model_path = Path(model_path)
    options = dict(training, data=str(dataset_root / "data.yaml"), project=str(runs_dir), name=run_name, exist_ok=True)
    metadata = {
        "dataset_name": dataset_name, "dataset_path": str(dataset_root), "dataset_config": dataset_config,
        "model_initialization": str(model_path), "preset": preset, "training_options": options,
        "device_requested": requested_device, "device_selected": device, "amp_requested": training["amp"],
        "resolved_device": None, "resolved_amp": None, "resolved_workers": None,
        "provenance": run_provenance(Path(project_root) / "src" / "grid_detection", "grid-detection"),
        "versions": {"python": platform.python_version(), "pytorch": str(torch.__version__),
                     "ultralytics": ultralytics.__version__, "cuda_runtime": torch.version.cuda},
    }
    metadata_path = run_dir / "experiment.json"

    def save_metadata():
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    save_metadata()
    print(f"Training destination: {run_dir}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    def capture_resolved(trainer):
        metadata["resolved_device"] = str(trainer.device)
        metadata["resolved_amp"] = bool(trainer.amp)
        metadata["resolved_workers"] = trainer.args.workers
        save_metadata()

    model.add_callback("on_train_start", capture_resolved)
    model.train(**options)
    print(f"Training artifacts: {run_dir}")
    return run_dir


@dataclass
class SavedRun:
    directory: Path
    metadata: dict
    args: dict
    dataset_root: Path
    dataset_config: dict
    records: list
    data_yaml: Path
    best_path: Path
    last_path: Path
    history: list


def load_saved_run(runs_dir, run_name):
    """Load a compatible run and its dataset without initializing a model."""
    validate_directory_name(run_name, "run")
    directory = Path(runs_dir) / run_name
    metadata_path, args_path = directory / "experiment.json", directory / "args.yaml"
    for path in (metadata_path, args_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    saved = json.loads(metadata_path.read_text())
    saved_args = yaml.safe_load(args_path.read_text())
    root = Path(saved["dataset_path"]).resolve()
    config = validate_saved_dataset(root)
    validate_descriptor_path(root)
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines() if line.strip()]
    if config != saved["dataset_config"]:
        raise ValueError("Current dataset configuration differs from the configuration saved with this run")
    results_path = directory / "results.csv"
    with results_path.open(newline="") as stream:
        history = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(stream, skipinitialspace=True)]
    if not history:
        raise ValueError(f"No training history in {results_path}")
    return SavedRun(directory, saved, saved_args, root, config, records, root / "data.yaml",
                    directory / "weights" / "best.pt", directory / "weights" / "last.pt", history)
