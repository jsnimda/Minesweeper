"""Masked phase regression and soft occupancy loss."""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PhaseLoss:
    total: torch.Tensor
    phase: torch.Tensor
    occupancy: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor


def phase_loss(logits, targets, phase_mask, occupancy_mask, occupancy_weight=1.0):
    if not math.isfinite(occupancy_weight) or occupancy_weight < 0:
        raise ValueError("occupancy_weight must be finite and nonnegative")
    if logits.ndim != 4 or logits.shape[1] != 5 or targets.shape != logits.shape:
        raise ValueError("expected matching (batch, 5, height, width) tensors")
    expected_mask_shape = (logits.shape[0], *logits.shape[2:])
    if phase_mask.shape != expected_mask_shape or occupancy_mask.shape != expected_mask_shape:
        raise ValueError("mask shapes must match batch and spatial dimensions")
    logits, targets = logits.float(), targets.float()
    phase_weights = phase_mask.to(dtype=logits.dtype).unsqueeze(1)
    squared = (logits[:, :4].tanh() - targets[:, :4]).square() * phase_weights
    denominator = phase_weights.sum().clamp_min(1)
    x = squared[:, :2].sum() / (2 * denominator)
    y = squared[:, 2:].sum() / (2 * denominator)
    phase = (x + y) / 2
    weights = occupancy_mask.to(dtype=logits.dtype)
    occupancy = (F.binary_cross_entropy_with_logits(logits[:, 4], targets[:, 4], reduction="none") * weights).sum() / weights.sum().clamp_min(1)
    return PhaseLoss(phase + occupancy_weight * occupancy, phase, occupancy, x, y)
