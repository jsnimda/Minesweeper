"""Grid U-Net architecture extracted from the legacy grid notebook."""

import torch
import torch.nn as nn


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


class GridUNet(nn.Module):
    """Map RGB batches to three heatmap logits at the input resolution.

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

        self.segmentation_head = nn.Conv2d(16, 3, kernel_size=1)

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
