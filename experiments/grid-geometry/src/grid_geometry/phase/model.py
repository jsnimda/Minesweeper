"""Phase U-Net and checkpoint-compatible prediction helpers."""

import torch
import torch.nn as nn

MODEL_ID = "PhaseGridUNet"
REPRESENTATION_ID = "directional-phase-occupancy-v1"


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PhaseGridUNet(nn.Module):
    """Map RGB batches to phase and occupancy logits at the input resolution.

    Both spatial dimensions must be positive multiples of eight.
    """

    def __init__(self) -> None:
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(3, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(64, 128)

        # Decoder
        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        self.dec1 = DoubleConv(128 + 64, 64)
        self.dec2 = DoubleConv(64 + 32, 32)
        self.dec3 = DoubleConv(32 + 16, 16)

        # Preserve the original initialization sequence for seeded training.
        self.segmentation_head = nn.Conv2d(16, 3, kernel_size=1)
        self.segmentation_head = nn.Conv2d(16, 5, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(x)
        x = self.pool(skip1)

        skip2 = self.enc2(x)
        x = self.pool(skip2)

        skip3 = self.enc3(x)
        x = self.pool(skip3)

        x = self.bottleneck(x)

        x = self.up(x)
        x = self.dec1(torch.cat([x, skip3], dim=1))

        x = self.up(x)
        x = self.dec2(torch.cat([x, skip2], dim=1))

        x = self.up(x)
        x = self.dec3(torch.cat([x, skip1], dim=1))

        return self.segmentation_head(x)


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
