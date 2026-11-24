"""Evaluate GridUNet heatmaps and optional decoded geometry on an exported split."""

import argparse
from contextlib import nullcontext
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import GeometryDataset
from .decoder import decode_geometry, fit_independent
from .evaluation_artifacts import PreviewSelection, SampleRecord, export_prediction
from .evaluation_metrics import (
    BoardDecoding, BoardEstimate, GEOMETRY_FIELDS, NORMALIZED_THRESHOLDS, PIXEL_TOLERANCE,
    evaluate_geometry, summarize_geometry, truth_from_geometry,
)
from .lattice import rank_axis_pairs
from .model import GridUNet
from .training import CHANNELS, select_device, weighted_squared_errors
from .training_artifacts import CHECKPOINT_VERSION, inventory_sha256, write_text
from .training_config import Config, DEVICES, load_config


MODES = ("both", "independent", "joint", "none")
HEATMAP_FIELDS = ("loss", *(f"{channel}_loss" for channel in CHANNELS))
DECODING_TIMES = ("axis_fit_ms", "independent_ms", "joint_additional_ms", "joint_total_ms")
CSV_FIELDS = (
    "index", "sample_id", "board_id", "game_state_id", "style", "decoder", "coordinates",
    "truth_rows", "truth_columns", "truth_pitch_x", "truth_pitch_y", "truth_origin_x", "truth_origin_y",
    *HEATMAP_FIELDS, "preprocessing_ms", "inference_batch_ms", "inference_batch_size",
    "inference_amortized_board_ms", *DECODING_TIMES, *GEOMETRY_FIELDS,
)


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint: Path
    dataset_root: Path
    split: str
    output_dir: Path
    training_config: Path | None = None
    checkpoint_kind: str = "best"
    device: str = "auto"
    batch_size: int = 8
    workers: int = 0
    decoder: str = "both"
    save_heatmaps: bool = False
    save_hypotheses: bool = False
    visualizations: int = 6

    def __post_init__(self) -> None:
        if not self.split.strip() or self.split in (".", "..") or any(c in self.split for c in "/\\:"):
            raise ValueError("split must be a single directory name under the dataset root")
        if self.checkpoint_kind not in ("best", "last"):
            raise ValueError("checkpoint_kind must be 'best' or 'last'")
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {', '.join(DEVICES)}")
        if self.decoder not in MODES:
            raise ValueError(f"decoder must be one of {', '.join(MODES)}")
        for label, value, minimum in (
            ("batch_size", self.batch_size, 1), ("workers", self.workers, 0),
            ("visualizations", self.visualizations, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{label} must be an integer of at least {minimum}")
        if self.decoder == "none" and self.save_hypotheses:
            raise ValueError("saving hypotheses requires independent or joint decoding")

    @property
    def settings_path(self) -> Path:
        return self.training_config if self.training_config is not None else self.checkpoint.with_name("config.toml")


def load_evaluation_model(path: Path, kind: str, settings: Config) -> tuple[GridUNet, dict]:
    if kind not in ("best", "last"):
        raise ValueError("checkpoint kind must be 'best' or 'last'")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        handle.seek(0)
        payload = torch.load(handle, map_location="cpu", weights_only=True)

    identity = {"path": str(path.resolve()), "sha256": digest, "kind": kind}
    if kind == "last":
        if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_VERSION:
            raise ValueError("last checkpoint must be a versioned Phase 4A training checkpoint")
        contract = payload.get("contract")
        if not isinstance(contract, dict) or not isinstance(contract.get("dataset"), dict):
            raise ValueError("last checkpoint is missing its training settings")
        mismatches = [
            name for name in ("width", "height", "sigma", "include_outer")
            if contract["dataset"].get(name) != getattr(settings.dataset, name)
        ]
        if contract.get("loss_weight") != settings.training.loss_weight:
            mismatches.append("loss_weight")
        if mismatches:
            raise ValueError(f"training configuration disagrees with last checkpoint: {', '.join(mismatches)}")
        if "model" not in payload:
            raise ValueError("last checkpoint is missing its model entry")
        identity.update(epoch=payload.get("epoch"), format_version=CHECKPOINT_VERSION)
        weights = payload["model"]
    else:
        if isinstance(payload, dict) and "format_version" in payload:
            raise ValueError("use --checkpoint-kind last to read a resumable training checkpoint")
        weights = payload
    if not isinstance(weights, dict) or not weights or not all(isinstance(value, torch.Tensor) for value in weights.values()):
        raise ValueError("checkpoint model must be a tensor state dictionary")
    model = GridUNet()
    model.load_state_dict(weights, strict=True)
    return model.eval(), identity


class EvaluationDataset(Dataset):
    """Time preparation in the process that reads each annotated sample."""

    def __init__(self, dataset: GeometryDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        start = perf_counter()
        sample = self.dataset.load_sample(index)
        elapsed = (perf_counter() - start) * 1000
        source_crop = tuple(map(float, sample.source_crop)) if sample.source_crop is not None else None
        record = SampleRecord(
            index, sample.sample_id, sample.board_id, sample.game_state_id, sample.style,
            sample.geometry, sample.transform, source_crop, elapsed,
        )
        return sample.image, sample.heatmaps, record


def _collate(batch):
    images, targets, records = zip(*batch, strict=True)
    return torch.stack(images), torch.stack(targets), records


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def predict_batch(
    model: torch.nn.Module, images: torch.Tensor, device: torch.device,
) -> tuple[torch.Tensor, float]:
    model.eval()
    images = images.to(device, non_blocking=device.type == "cuda")
    _synchronize(device)
    start = perf_counter()
    probabilities = model(images).float().sigmoid()
    _synchronize(device)
    elapsed = (perf_counter() - start) * 1000
    return probabilities.cpu(), elapsed


def heatmap_losses(probabilities: torch.Tensor, targets: torch.Tensor, weight: float) -> np.ndarray:
    """Return a float32 [board, channel] matrix of the training objective."""
    losses = weighted_squared_errors(probabilities, targets, weight).mean(dim=(2, 3))
    values = losses.detach().cpu().numpy()
    if not np.isfinite(values).all():
        raise RuntimeError("evaluation heatmap loss is nonfinite")
    return values


def decode_predictions(probabilities: np.ndarray, mode: str) -> tuple[dict[str, BoardDecoding], dict[str, float]]:
    if mode not in MODES:
        raise ValueError(f"unknown decoder mode {mode!r}")
    if mode == "none":
        return {}, {}
    start = perf_counter()
    fits = fit_independent(probabilities)
    fit_ms = (perf_counter() - start) * 1000
    timing = {"axis_fit_ms": fit_ms}
    decodings = {}
    if mode in ("independent", "both"):
        start = perf_counter()
        pairs = rank_axis_pairs(fits.x, fits.y)[:8]
        status = "insufficient_evidence"
        if pairs:
            status = "fitted" if fits.x.status == fits.y.status == "fitted" else "ambiguous"
        hypotheses = tuple(
            BoardEstimate(
                pair.x.boundaries, pair.y.boundaries,
                (pair.x.left_edge, pair.y.left_edge, pair.x.right_edge, pair.y.right_edge),
                pair.score,
            )
            for pair in pairs
        )
        decodings["independent"] = BoardDecoding(status, hypotheses, (fits.x.status, fits.y.status))
        timing["independent_ms"] = fit_ms + (perf_counter() - start) * 1000
    if mode in ("joint", "both"):
        start = perf_counter()
        joint = decode_geometry(probabilities, independent=fits)
        hypotheses = tuple(
            BoardEstimate(item.x_edges, item.y_edges, item.outer_edges, item.score) for item in joint.hypotheses
        )
        decodings["joint"] = BoardDecoding(joint.status, hypotheses, (fits.x.status, fits.y.status))
        timing["joint_additional_ms"] = (perf_counter() - start) * 1000
        timing["joint_total_ms"] = fit_ms + timing["joint_additional_ms"]
    return decodings, timing


def _json(path: Path, content: dict) -> None:
    write_text(path, json.dumps(content, indent=2, allow_nan=False) + "\n")


def _effective_config(
    config: EvaluationConfig, settings: Config, source_config: bytes, checkpoint: dict,
    device: torch.device, sample_count: int,
) -> dict:
    data = settings.dataset
    return {
        "format_version": 1,
        "dataset": {
            "root": str(config.dataset_root.resolve()), "split": config.split, "boards": sample_count,
            "inventory_sha256": inventory_sha256(config.dataset_root, (config.split,)),
            "inventory_method": "sha256 of sorted split, filename, size, and mtime_ns records",
        },
        "checkpoint": checkpoint,
        "training_config": {
            "path": str(config.settings_path.resolve()),
            "sha256": hashlib.sha256(source_config).hexdigest(),
        },
        "input": {
            "width": data.width, "height": data.height, "sigma": data.sigma, "include_outer": data.include_outer,
            "loss_weight": settings.training.loss_weight, "channels": CHANNELS,
        },
        "thresholds": {
            "max_center_error_px": PIXEL_TOLERANCE,
            "max_center_error_pitch_normalized": [value for _, value in NORMALIZED_THRESHOLDS],
        },
        "evaluation": {
            "output_dir": str(config.output_dir.resolve()),
            "device_requested": config.device, "device": str(device), "batch_size": config.batch_size,
            "workers": config.workers, "decoder": config.decoder, "model_precision": "float32",
            "save_heatmaps": config.save_heatmaps, "save_hypotheses": config.save_hypotheses,
            "visualizations": config.visualizations,
        },
        "latency_conventions": {
            "unit": "ms",
            "preprocessing_ms": "per-board PNG/JSON load, resize/pad, and target generation, excluding collation and queueing",
            "inference_batch_ms": "forward and sigmoid per batch, synchronized on accelerators, excluding transfers and loss",
            "inference_amortized_board_ms": "batch inference time divided by its actual board count",
            "axis_fit_ms": "mean projections and Phase 3B axis fits per board",
            "independent_ms": "axis fit plus independent pairing and hypothesis conversion per board",
            "joint_additional_ms": "joint decoding and hypothesis conversion with precomputed axis fits per board",
            "joint_total_ms": "axis fit plus joint additional time per board",
            "warmup": "first batch and first board are included",
        },
        "environment": {
            "python": sys.version, "pytorch": str(torch.__version__), "os": platform.platform(),
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }


def _latency_summary(durations: dict[str, list[float]]) -> dict:
    return {
        name: {"count": len(values), "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
        for name, values in durations.items() if values
    }


def _run(
    config: EvaluationConfig, model: GridUNet, loader: DataLoader, device: torch.device, weight: float,
    writer: csv.DictWriter, manifest,
) -> dict:
    methods = {name: [] for name in ("independent", "joint") if config.decoder in ("both", name)}
    latencies: dict[str, list[float]] = {
        name: [] for name in ("preprocessing_ms", "inference_batch_ms", "inference_amortized_board_ms", *DECODING_TIMES)
    }
    previews = PreviewSelection(config.visualizations)
    sums = np.zeros(3, dtype=np.float64)
    count = 0
    for images, targets, samples in loader:
        probabilities, batch_ms = predict_batch(model, images, device)
        losses = heatmap_losses(probabilities, targets, weight)
        sums += losses.sum(axis=0, dtype=np.float64)
        batch_size = len(samples)
        count += batch_size
        latencies["inference_batch_ms"].append(batch_ms)
        for sample, image, target, probability, loss in zip(samples, images, targets, probabilities, losses, strict=True):
            probability_array = probability.numpy()
            decodings, timing = decode_predictions(probability_array, config.decoder)
            truth = truth_from_geometry(sample.geometry)
            latencies["preprocessing_ms"].append(sample.preprocessing_ms)
            latencies["inference_amortized_board_ms"].append(batch_ms / batch_size)
            for name, elapsed in timing.items():
                latencies[name].append(elapsed)
            common = {
                "index": sample.index, "sample_id": sample.sample_id, "board_id": sample.board_id,
                "game_state_id": sample.game_state_id, "style": sample.style, "truth_rows": truth.rows,
                "truth_columns": truth.columns, "truth_pitch_x": truth.pitch_x, "truth_pitch_y": truth.pitch_y,
                "truth_origin_x": truth.origin_x, "truth_origin_y": truth.origin_y,
                "coordinates": json.dumps(sample.coordinates(), allow_nan=False),
                "loss": float(loss.mean(dtype=np.float64)), "preprocessing_ms": sample.preprocessing_ms,
                "inference_batch_ms": batch_ms, "inference_batch_size": batch_size,
                "inference_amortized_board_ms": batch_ms / batch_size, **timing,
                **{f"{name}_loss": float(value) for name, value in zip(CHANNELS, loss, strict=True)},
            }
            failures = []
            for method, decoding in decodings.items():
                result = evaluate_geometry(truth, decoding)
                methods[method].append(result)
                writer.writerow(dict(common, decoder=method, **result))
                if result["failure"] is not None:
                    failures.append(result["failure"])
            if not decodings:
                writer.writerow(dict(common, decoder="none"))
            outcome = failures[-1] if failures else "correct" if decodings else "heatmaps_only"
            previews.consider(sample, image.numpy(), target.numpy(), probability_array, decodings, outcome.replace(".", "_"))
            if manifest is not None:
                export = export_prediction(
                    config.output_dir / "predictions", sample, truth, probability_array, decodings,
                    save_heatmaps=config.save_heatmaps, save_hypotheses=config.save_hypotheses,
                )
                manifest.write(json.dumps(export, allow_nan=False) + "\n")
    if not count:
        raise ValueError("evaluation split produced no samples")
    channels = sums / count
    return {
        "boards": count,
        "heatmaps": {"weighted_mse": float(channels.mean()), "channels": dict(zip(CHANNELS, map(float, channels), strict=True))},
        "geometry": {method: summarize_geometry(records) for method, records in methods.items()},
        "latency": _latency_summary(latencies),
        "visualizations": previews.write(config.output_dir / "visualizations"),
    }


def evaluate(config: EvaluationConfig) -> dict:
    device = select_device(config.device)
    source_config = config.settings_path.read_bytes()
    settings = load_config(config.settings_path)
    model, checkpoint = load_evaluation_model(config.checkpoint, config.checkpoint_kind, settings)
    data = settings.dataset
    dataset = GeometryDataset.from_split_directory(
        config.dataset_root / config.split, canvas_size=(data.width, data.height),
        sigma=data.sigma, include_outer=data.include_outer,
    )
    run_config = _effective_config(config, settings, source_config, checkpoint, device, len(dataset))
    try:
        config.output_dir.mkdir(parents=True)
    except FileExistsError as exc:
        raise ValueError(f"evaluation directory already exists: {config.output_dir}") from exc
    _json(config.output_dir / "evaluation.json", run_config)
    (config.output_dir / "training_config.toml").write_bytes(source_config)
    save_predictions = config.save_heatmaps or config.save_hypotheses
    if save_predictions:
        (config.output_dir / "predictions").mkdir()
    loader = DataLoader(
        EvaluationDataset(dataset), batch_size=config.batch_size, num_workers=config.workers,
        multiprocessing_context="spawn" if config.workers else None,
        shuffle=False, pin_memory=device.type == "cuda", collate_fn=_collate,
    )
    model = model.to(device)
    manifest_context = (
        (config.output_dir / "predictions" / "manifest.jsonl").open("w", encoding="utf-8")
        if save_predictions else nullcontext(None)
    )
    with (config.output_dir / "boards.csv").open("w", newline="", encoding="utf-8") as stream, manifest_context as manifest:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        result = _run(config, model, loader, device, settings.training.loss_weight, writer, manifest)
    _json(config.output_dir / "metrics.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-kind", choices=("best", "last"), default="best")
    parser.add_argument("--training-config", type=Path, help="resolved training TOML, defaults to config.toml beside the checkpoint")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", required=True, help="a named directory inside the dataset root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=DEVICES, default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--decoder", choices=MODES, default="both")
    parser.add_argument("--save-heatmaps", action="store_true")
    parser.add_argument("--save-hypotheses", action="store_true")
    parser.add_argument("--visualizations", type=int, default=6)
    args = parser.parse_args(argv)
    try:
        result = evaluate(EvaluationConfig(**vars(args)))
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 1
    print(f"Evaluated {result['boards']} boards. Weighted MSE {result['heatmaps']['weighted_mse']:.6f}")
    for method, metrics in result["geometry"].items():
        complete = metrics["complete_px"]
        print(f"{method}: complete geometry {complete['count']}/{complete['total']} at {PIXEL_TOLERANCE:.2f} px")
    print(f"Evaluation directory {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
