"""Run ownership, environment metadata, and portable epoch artifacts."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import csv
from datetime import UTC, datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import BinaryIO

import torch

from .training import CHANNELS
from .training_config import DatasetConfig, RunConfig


CHECKPOINT_VERSION = 1
HISTORY_FIELDS = (
    "epoch",
    *(f"{phase}_{metric}" for phase in ("train", "validation") for metric in ("loss", *CHANNELS, "samples")),
)


@contextmanager
def own_run(config: RunConfig) -> Iterator[None]:
    directory = config.directory
    if not config.resume:
        try:
            directory.mkdir(parents=True)
        except FileExistsError as exc:
            raise ValueError(f"run directory already exists: {directory}. Choose a new directory or resume") from exc
    elif not directory.is_dir():
        raise ValueError(f"resume requires an existing run directory: {directory}")

    lock = directory / ".training.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError(f"run is locked: {lock}. Verify the prior trainer has exited before clearing a stale lock") from exc
    try:
        with handle:
            handle.write(f"pid={os.getpid()}\n")
        yield
    finally:
        lock.unlink(missing_ok=True)


def _atomic_write(path: Path, writer: Callable[[BinaryIO], object]) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_text(path: Path, content: str) -> None:
    _atomic_write(path, lambda handle: handle.write(content.encode("utf-8")))


def _save_torch(path: Path, state: dict) -> None:
    _atomic_write(path, lambda handle: torch.save(state, handle))


def cpu_snapshot(value):
    """Copy training state to CPU without moving or aliasing the live tensors."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        snapshot = value.copy()
        for key, item in value.items():
            snapshot[key] = cpu_snapshot(item)
        if hasattr(value, "_metadata"):
            snapshot._metadata = cpu_snapshot(value._metadata)
        return snapshot
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return value


def dataset_identity(config: DatasetConfig, *, train_count: int, validation_count: int) -> dict:
    digest = hashlib.sha256()
    for split in (config.train_split, config.validation_split):
        paths = sorted(
            (path for path in (config.root / split).iterdir() if path.suffix in (".png", ".json")),
            key=lambda path: path.name,
        )
        for path in paths:
            stat = path.stat()
            entry = [split, path.name, stat.st_size, stat.st_mtime_ns]
            digest.update((json.dumps(entry) + "\n").encode("utf-8"))
    return {
        "method": "sha256 of sorted split, filename, size, and mtime_ns records",
        "inventory_sha256": digest.hexdigest(),
        "train_samples": train_count,
        "validation_samples": validation_count,
    }


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=Path(__file__).resolve().parent, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def write_metadata(
    directory: Path, *, device: torch.device, dataset: dict, root: Path, seed: int,
    resume_epoch: int | None, requested_epochs: int,
) -> None:
    path = directory / "run_metadata.json"
    if resume_epoch is None:
        metadata = {"model": "GridUNet", "initialization": "scratch", "seed": seed, "sessions": []}
    else:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    status = _git("status", "--porcelain", "--", "../..")
    session = {
        "started_utc": datetime.now(UTC).isoformat(),
        "resume_epoch": resume_epoch,
        "requested_epochs": requested_epochs,
        "dataset": {"root": str(root), **dataset},
        "git_revision": _git("rev-parse", "HEAD"),
        "git_project_dirty": bool(status) if status is not None else None,
        "python": sys.version,
        "pytorch": str(torch.__version__),
        "os": platform.platform(),
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    metadata["sessions"].append(session)
    write_text(path, json.dumps(metadata, indent=2) + "\n")


def load_last(directory: Path) -> dict:
    path = directory / "last.pt"
    if not path.is_file():
        raise ValueError(f"no resumable checkpoint at {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported training checkpoint format at {path}")
    history = checkpoint["history"]
    if checkpoint["epoch"] != len(history) or any(row["epoch"] != i for i, row in enumerate(history, 1)):
        raise ValueError(f"inconsistent epoch history at {path}")
    return checkpoint


def publish_history(directory: Path, checkpoint: dict, *, write_best: bool) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
    writer.writeheader()
    writer.writerows(checkpoint["history"])
    write_text(directory / "history.csv", stream.getvalue())
    if write_best and checkpoint["best_model"] is not None:
        _save_torch(directory / "best.pt", checkpoint["best_model"])


def commit_epoch(directory: Path, checkpoint: dict, *, archive_every: int, best_changed: bool) -> None:
    # The last checkpoint is authoritative. Resume repairs sidecar files from it.
    _save_torch(directory / "last.pt", checkpoint)
    publish_history(directory, checkpoint, write_best=best_changed)
    epoch = checkpoint["epoch"]
    if archive_every and epoch and epoch % archive_every == 0:
        _save_torch(directory / f"epoch-{epoch:04d}.pt", checkpoint)
