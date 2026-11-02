"""EEGInception baseline adapted for the local subject-split runner."""

from __future__ import annotations

import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    """1D inception-style temporal block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, ...] = (8, 16, 32),
        bottleneck_channels: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_bottleneck = bottleneck_channels > 0
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if self.use_bottleneck
            else nn.Identity()
        )
        branch_in = bottleneck_channels if self.use_bottleneck else in_channels
        branch_out = out_channels // len(kernel_sizes)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    branch_in,
                    branch_out,
                    kernel_size=k,
                    stride=1,
                    padding=k // 2,
                    bias=False,
                )
                for k in kernel_sizes
            ]
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run temporal inception branches."""

        x = self.bottleneck(x)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.drop(x)
        return x


class SpatialBlock(nn.Module):
    """Depthwise-separable spatial refinement block."""

    def __init__(self, in_channels: int, depth_multiplier: int = 2) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * depth_multiplier,
            kernel_size=(1, 1),
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels * depth_multiplier,
            in_channels,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine channel interactions while preserving temporal length."""

        x = x.unsqueeze(2)  # [B, C, 1, T]
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x.squeeze(2)


class EEGInception(nn.Module):
    """EEGInception classifier for [B, C, T] EEG input."""

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        channels: tuple[int, ...] = (96, 192, 384),
        kernel_sizes: tuple[int, ...] = (8, 16, 32),
        depth_multiplier: int = 2,
        bottleneck_channels: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = num_channels
        for out_ch in channels:
            blocks.append(
                InceptionBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    dropout=dropout,
                )
            )
            blocks.append(
                SpatialBlock(
                    in_channels=out_ch,
                    depth_multiplier=depth_multiplier,
                )
            )
            in_ch = out_ch
        self.feature_extractor = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass.

        Args:
            x: [B, C, T]
        Returns:
            logits: [B, num_classes]
            features: [B, final_channels]
        """

        x = self.feature_extractor(x)
        x = self.global_pool(x)
        features = x.flatten(1)
        logits = self.classifier(x)
        return logits, features
