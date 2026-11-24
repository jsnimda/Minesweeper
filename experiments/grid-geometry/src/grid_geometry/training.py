"""Numerical training and validation steps for RGB-to-heatmap models."""

from contextlib import nullcontext
from dataclasses import dataclass
import math
import random
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import GeometryDataset
from .training_config import DEVICES, TrainingConfig


CHANNELS = ("center", "vertical", "horizontal")


@dataclass(frozen=True)
class HeatmapLoss:
    total: torch.Tensor
    channels: torch.Tensor


def weighted_squared_errors(
    probabilities: torch.Tensor, targets: torch.Tensor, weight: float = 10.0,
) -> torch.Tensor:
    """The shared training and evaluation objective before reduction."""
    if (
        probabilities.ndim != 4 or probabilities.shape[1] != 3
        or probabilities.shape != targets.shape or not probabilities.numel()
    ):
        raise ValueError("predictions and targets must have equal, nonempty [B, 3, H, W] shapes")
    actual = targets.float()
    return (1 + weight * actual) * (probabilities.float() - actual).square()


def weighted_mse(logits: torch.Tensor, targets: torch.Tensor, weight: float = 10.0) -> HeatmapLoss:
    """Legacy sigmoid weighted MSE, with each channel reduced in float32."""
    weighted_error = weighted_squared_errors(logits.float().sigmoid(), targets, weight)
    channels = weighted_error.mean(dim=(0, 2, 3))
    return HeatmapLoss(channels.mean(), channels)


def select_device(requested: str) -> torch.device:
    if requested not in DEVICES:
        raise ValueError(f"unknown training device {requested!r}")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but PyTorch reports no available CUDA device")
        return torch.device("cuda", torch.cuda.current_device())
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but PyTorch reports it unavailable")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed % 2**64)


def _seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(
    dataset: GeometryDataset, config: TrainingConfig, *, shuffle: bool, device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.workers,
        generator=torch.Generator(),
        worker_init_fn=_seed_worker,
        multiprocessing_context="spawn" if config.workers else None,
        pin_memory=device.type == "cuda",
    )


def make_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    optimizer_type = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[config.optimizer]
    return optimizer_type(model.parameters(), lr=config.learning_rate)


@dataclass(frozen=True)
class EpochMetrics:
    channels: tuple[float, float, float]
    samples: int

    @property
    def loss(self) -> float:
        return sum(self.channels) / len(self.channels)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    weight: float = 10.0,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    progress_every: int = 0,
    report: Callable[[str], None] = print,
) -> EpochMetrics:
    """Update the model when an optimizer is supplied, or measure validation loss."""
    training = optimizer is not None
    mixed_precision = scaler is not None
    if mixed_precision and device.type != "cuda":
        raise ValueError("mixed precision is supported only on CUDA")
    model.train(training)
    totals = torch.zeros(3, device=device)
    samples = 0
    phase = "train" if training else "validation"
    transfer_async = device.type == "cuda"
    with torch.set_grad_enabled(training):
        for batch, (images, targets) in enumerate(loader, 1):
            images = images.to(device, non_blocking=transfer_async)
            targets = targets.to(device, non_blocking=transfer_async)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            amp_context = torch.autocast("cuda", dtype=torch.float16) if mixed_precision else nullcontext()
            with amp_context:
                logits = model(images)
            loss = weighted_mse(logits, targets, weight)

            if optimizer is not None:
                if scaler is None:
                    loss.total.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss.total).backward()
                    scaler.step(optimizer)
                    scaler.update()

            count = images.shape[0]
            totals += count * loss.channels.detach()
            samples += count
            if progress_every and (batch % progress_every == 0 or batch == len(loader)):
                report(f"  {phase} batch {batch}/{len(loader)}  loss {float(totals.mean() / samples):.6f}")

    if not samples:
        raise ValueError(f"{phase} loader produced no samples")
    channels = tuple(float(value) for value in (totals / samples).tolist())
    if not all(math.isfinite(value) for value in channels):
        raise RuntimeError(f"{phase} loss is nonfinite")
    return EpochMetrics(channels, samples)
