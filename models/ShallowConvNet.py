"""Minimal ShallowConvNet baseline for EEG classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class ShallowConvNet(nn.Module):
    """A compact ShallowConvNet-style model.

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
        temporal_filters: int = 40,
        temporal_kernel: int = 25,
        pool_kernel: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, temporal_kernel), bias=False),
            nn.Conv2d(temporal_filters, temporal_filters, kernel_size=(num_channels, 1), bias=False),
            nn.BatchNorm2d(temporal_filters),
            nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_stride)),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(temporal_filters, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one forward pass."""

        x = x.unsqueeze(1)  # [B, 1, C, T]
        x = self.features[0](x)
        x = self.features[1](x)
        x = self.features[2](x)
        x = torch.square(x)
        x = self.features[3](x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.features[4](x)
        features = x.mean(dim=(-1, -2))
        logits = self.classifier(x)
        return logits, features
