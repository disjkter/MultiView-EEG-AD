"""Minimal EEGNet baseline for EEG classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """Compact EEGNet-style classifier.

    Input shape:
        x: [batch, channels, time]
    Output shape:
        logits: [batch, num_classes]
        features: [batch, feature_dim]
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
        separable_filters: int = 16,
        temporal_kernel: int = 64,
        separable_kernel: int = 16,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        spatial_filters = temporal_filters * depth_multiplier

        self.block1 = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(temporal_filters),
            nn.Conv2d(
                temporal_filters,
                spatial_filters,
                kernel_size=(num_channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                kernel_size=(1, separable_kernel),
                padding=(0, separable_kernel // 2),
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(spatial_filters, separable_filters, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(separable_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(separable_filters, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass."""

        x = x.unsqueeze(1)  # [B, 1, C, T]
        x = self.block1(x)
        x = self.block2(x)
        features = x.mean(dim=(-1, -2))
        logits = self.classifier(x)
        return logits, features
