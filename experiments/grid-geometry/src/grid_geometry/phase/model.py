"""Five-channel variant of the original, checkpoint-compatible U-Net backbone."""
import torch
from torch import nn

from ..model import GridUNet

MODEL_ID = "PhaseGridUNet"
REPRESENTATION_ID = "directional-phase-occupancy-v1"


class PhaseGridUNet(GridUNet):
    """Return raw phase and occupancy logits at the input resolution."""

    def __init__(self) -> None:
        super().__init__()
        self.segmentation_head = nn.Conv2d(16, 5, kernel_size=1)


def phase_predictions(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits once, retaining the unconstrained complex magnitudes."""
    if logits.ndim != 4 or logits.shape[1] != 5:
        raise ValueError("expected logits of shape (batch, 5, height, width)")
    return torch.cat((logits[:, :4].tanh(), logits[:, 4:5].sigmoid()), dim=1)


def load_phase_weights(model: PhaseGridUNet, checkpoint: dict) -> None:
    """Require explicit identity even when the weights are shape compatible."""
    if checkpoint.get("model_id") != MODEL_ID or checkpoint.get("representation_id") != REPRESENTATION_ID:
        raise ValueError("checkpoint is not a PhaseGridUNet phase-occupancy checkpoint")
    model.load_state_dict(checkpoint["model"])
