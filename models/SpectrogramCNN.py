"""Spectrogram CNN baseline inspired by EEG spectrogram literature."""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectrogramCNN(nn.Module):
    """A small 2D CNN for channel-wise spectrogram inputs.

    Input shape:
        x: [batch, channels, freq_bins, time_bins]
    Output shape:
        logits: [batch, num_classes]
        features: [batch, feature_dim]
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        base_channels: int = 16,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one forward pass."""

        encoded = self.encoder(x)
        features = encoded.flatten(1)
        logits = self.classifier(encoded)
        return logits, features
