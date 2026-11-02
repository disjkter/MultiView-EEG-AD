"""Minimal Temporal Convolutional Network baseline for EEG classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """Residual temporal block with dilated convolutions.

    Input / output shape:
        x: [batch, channels, time]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def _causal_crop(self, x: torch.Tensor, crop_size: int) -> torch.Tensor:
        """Trim right padding to preserve sequence length."""

        if crop_size == 0:
            return x
        return x[..., :-crop_size]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one residual block."""

        residual = self.residual(x)
        crop1 = self.conv1.padding[0]
        crop2 = self.conv2.padding[0]

        out = self.conv1(x)
        out = self._causal_crop(out, crop1)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self._causal_crop(out, crop2)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout(out)

        return self.activation(out + residual)


class TCN(nn.Module):
    """Compact TCN baseline.

    Input shape:
        x: [batch, channels, time]
    Output shape:
        logits: [batch, num_classes]
        features: [batch, hidden_channels]
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        hidden_channels: int = 32,
        depth: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = num_channels
        for level in range(depth):
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
            in_channels = hidden_channels

        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass."""

        x = self.encoder(x)  # [B, hidden_channels, T]
        features = self.pool(x).squeeze(-1)  # [B, hidden_channels]
        logits = self.classifier(features)
        return logits, features
