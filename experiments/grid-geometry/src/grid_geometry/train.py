"""Train a fresh GridUNet or resume a Phase 4A run."""

import argparse
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
import sys
from time import perf_counter

import torch

from .dataset import GeometryDataset
from .model import GridUNet
from .training import CHANNELS, EpochMetrics, make_loader, make_optimizer, run_epoch, seed_everything, select_device
from .training_artifacts import (
    CHECKPOINT_VERSION,
    commit_epoch,
    cpu_snapshot,
    dataset_identity,
    load_last,
    own_run,
    publish_history,
    write_metadata,
    write_text,
)
from .training_config import Config, DEVICES, load_config, resolved_toml


def _history_row(epoch: int, train: EpochMetrics, validation: EpochMetrics) -> dict:
    row = {"epoch": epoch}
    for name, metrics in (("train", train), ("validation", validation)):
        row[f"{name}_loss"] = metrics.loss
        row[f"{name}_samples"] = metrics.samples
        for channel, loss in zip(CHANNELS, metrics.channels, strict=True):
            row[f"{name}_{channel}"] = loss
    return row


def _training_contract(config: Config, identity: dict) -> dict:
    data = asdict(config.dataset)
    del data["root"]
    training = config.training
    return {
        "dataset": data,
        "inventory_sha256": identity["inventory_sha256"],
        "batch_size": training.batch_size,
        "optimizer": training.optimizer,
        "learning_rate": training.learning_rate,
        "loss_weight": training.loss_weight,
        "seed": training.seed,
        "mixed_precision": training.mixed_precision,
    }


def train_model(config: Config, *, report: Callable[[str], None] = print) -> int:
    """Run to the configured total epoch count and return the last completed epoch."""
    data = config.dataset
    training = config.training
    directory = config.run.directory
    device = select_device(training.device)
    if training.mixed_precision and device.type != "cuda":
        raise ValueError("mixed precision requires a CUDA training device")
    train_data = GeometryDataset(
        data.root, data.train_split, canvas_size=(data.width, data.height),
        sigma=data.sigma, include_outer=data.include_outer,
    )
    validation_data = GeometryDataset(
        data.root, data.validation_split, canvas_size=(data.width, data.height),
        sigma=data.sigma, include_outer=data.include_outer,
    )
    identity = dataset_identity(data, train_count=len(train_data), validation_count=len(validation_data))
    contract = _training_contract(config, identity)

    with own_run(config.run):
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        seed_everything(training.seed)
        model = GridUNet().to(device)
        optimizer = make_optimizer(model, training)
        scaler = torch.amp.GradScaler("cuda") if training.mixed_precision else None

        if config.run.resume:
            checkpoint = load_last(directory)
            differences = sorted(key for key in contract if contract[key] != checkpoint["contract"].get(key))
            if differences:
                raise ValueError(f"resume settings differ from the checkpoint: {', '.join(differences)}")
            if training.epochs < checkpoint["epoch"]:
                raise ValueError("configured epochs is below the checkpoint's completed epoch count")
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None:
                scaler.load_state_dict(checkpoint["scaler"])
            resume_epoch = checkpoint["epoch"]
        else:
            checkpoint = {
                "format_version": CHECKPOINT_VERSION,
                "contract": contract,
                "epoch": 0,
                "model": cpu_snapshot(model.state_dict()),
                "optimizer": cpu_snapshot(optimizer.state_dict()),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_model": None,
                "best_epoch": None,
                "best_validation_loss": None,
                "history": [],
            }
            resume_epoch = None

        write_metadata(
            directory, device=device, dataset=identity, root=data.root, seed=training.seed,
            resume_epoch=resume_epoch, requested_epochs=training.epochs,
        )
        write_text(directory / "config.toml", resolved_toml(config, device=device.type))
        if resume_epoch is None:
            commit_epoch(directory, checkpoint, archive_every=0, best_changed=False)
        else:
            publish_history(directory, checkpoint, write_best=True)
        report(
            f"GridUNet on {device} | {len(train_data)} training samples, {len(validation_data)} validation samples"
            f" | completed {checkpoint['epoch']}/{training.epochs} epochs"
        )
        report(f"Run directory {directory}")
        if checkpoint["epoch"] == training.epochs:
            return checkpoint["epoch"]

        train_loader = make_loader(train_data, training, shuffle=True, device=device)
        validation_loader = make_loader(validation_data, training, shuffle=False, device=device)
        for epoch in range(checkpoint["epoch"] + 1, training.epochs + 1):
            start = perf_counter()
            epoch_seed = (training.seed + 2 * epoch) % 2**64
            seed_everything(epoch_seed)
            train_loader.generator.manual_seed(epoch_seed)
            validation_loader.generator.manual_seed((epoch_seed + 1) % 2**64)
            report(f"Epoch {epoch}/{training.epochs}")
            train = run_epoch(
                model, train_loader, device=device, weight=training.loss_weight,
                optimizer=optimizer, scaler=scaler, progress_every=training.log_every_batches, report=report,
            )
            validation = run_epoch(
                model, validation_loader, device=device, weight=training.loss_weight,
                scaler=scaler, progress_every=training.log_every_batches, report=report,
            )
            latest_model = cpu_snapshot(model.state_dict())
            checkpoint.update(
                epoch=epoch,
                model=latest_model,
                optimizer=cpu_snapshot(optimizer.state_dict()),
                scaler=scaler.state_dict() if scaler is not None else None,
            )
            checkpoint["history"].append(_history_row(epoch, train, validation))
            previous_best = checkpoint["best_validation_loss"]
            improved = previous_best is None or validation.loss < previous_best
            if improved:
                checkpoint.update(best_model=latest_model, best_epoch=epoch, best_validation_loss=validation.loss)
            commit_epoch(
                directory, checkpoint, archive_every=config.run.archive_every, best_changed=improved,
            )
            for phase, metrics in (("train", train), ("validation", validation)):
                channels = "  ".join(f"{name} {value:.6f}" for name, value in zip(CHANNELS, metrics.channels, strict=True))
                report(f"  {phase} loss {metrics.loss:.6f}  {channels}")
            report(f"  best validation epoch {checkpoint['best_epoch']}  |  {perf_counter() - start:.1f}s")
        return checkpoint["epoch"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.toml"), help="training TOML file")
    parser.add_argument("--dataset-root", type=Path, help="override the geometry dataset directory")
    parser.add_argument("--output-dir", type=Path, help="override the run directory")
    parser.add_argument("--epochs", type=int, help="override the total target epoch count")
    parser.add_argument("--device", choices=DEVICES, help="override the training device")
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.dataset_root is not None:
            config = replace(config, dataset=replace(config.dataset, root=args.dataset_root.expanduser().resolve()))
        overrides = {
            key: getattr(args, key) for key in ("epochs", "device", "mixed_precision") if getattr(args, key) is not None
        }
        run_changes = {}
        if args.output_dir is not None:
            run_changes["directory"] = args.output_dir.expanduser().resolve()
        if args.resume is not None:
            run_changes["resume"] = args.resume
        config = replace(config, training=replace(config.training, **overrides), run=replace(config.run, **run_changes))
        train_model(config, report=lambda text: print(text, flush=True))
    except KeyboardInterrupt:
        print("Training interrupted. Resume this directory from its last committed epoch.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Training error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
