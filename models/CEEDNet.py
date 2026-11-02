"""CEEDNet 1D-ResNet baseline adapted from the official CAUEEG implementation."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn


def _program_conv_filters(
    sequence_length: int,
    conv_filter_list: list[dict[str, int]],
    output_lower_bound: int = 4,
    output_upper_bound: int = 8,
) -> int:
    """Choose CEEDNet pool/stride values for a target temporal output length."""

    mid = (output_upper_bound + output_lower_bound) / 2.0
    in_out_ratio = float(sequence_length) / mid
    base_stride = np.power(
        in_out_ratio / np.prod([cf["kernel_size"] for cf in conv_filter_list], dtype=np.float64),
        1.0 / len(conv_filter_list),
    )

    for idx, cf in enumerate(conv_filter_list):
        if idx == 0 and len(conv_filter_list) > 1:
            total_stride = max(1.0, base_stride * cf["kernel_size"] * 0.7)
            cf["pool"] = max(1, round(np.sqrt(total_stride) * 0.3))
            cf["stride"] = max(1, round(total_stride / cf["pool"]))
        else:
            total_stride = max(1.0, base_stride * cf["kernel_size"])
            cf["stride"] = min(max(1, round(np.sqrt(total_stride))), round(total_stride))
            cf["pool"] = max(1, round(total_stride / cf["stride"]))

    current_length = sequence_length
    success = False
    for _ in range(5):
        if success:
            break
        for pivot in reversed(range(len(conv_filter_list))):
            current_length = sequence_length
            for cf in conv_filter_list:
                current_length = current_length // cf.get("pool", 1)
                effective_kernel = cf["kernel_size"] - 1
                padding = 2 * (cf["kernel_size"] // 2)
                current_length = (current_length + padding - effective_kernel - 1) // cf["stride"] + 1

            pool = conv_filter_list[pivot]["pool"]
            stride = conv_filter_list[pivot]["stride"]
            if current_length < output_lower_bound:
                if stride > 1:
                    conv_filter_list[pivot]["stride"] = max(1, stride - 1)
                else:
                    conv_filter_list[pivot]["pool"] = max(1, pool - 1)
            elif current_length > output_upper_bound:
                conv_filter_list[pivot]["stride"] = stride + 1
            else:
                success = True
                break

    if not success:
        raise RuntimeError("Failed to determine CEEDNet convolution filter parameters.")
    return int(current_length)


class _BasicBlock1D(nn.Module):
    """Basic 1D residual block used by CEEDNet ResNet-18."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm1d,
        activation: Callable[..., nn.Module] = nn.Mish,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm1 = norm_layer(out_channels)
        self.act1 = activation()
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm2 = norm_layer(out_channels)
        self.act2 = activation()
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block to ``x`` with shape [B, C, T]."""

        identity = x
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.act2(out + identity)


class CEEDNet1DResNet18(nn.Module):
    """Official-style CEEDNet 1D-ResNet-18 for EEG windows.

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
        seq_length: int = 640,
        base_channels: int = 64,
        fc_stages: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.current_channels = base_channels
        conv_filter_list = [
            {"kernel_size": 41},
            {"kernel_size": 9},
            {"kernel_size": 9},
            {"kernel_size": 9},
            {"kernel_size": 9},
        ]
        self.output_length = _program_conv_filters(seq_length, conv_filter_list)

        input_filter = conv_filter_list[0]
        input_stage: list[nn.Module] = []
        if input_filter["pool"] > 1:
            input_stage.append(nn.MaxPool1d(input_filter["pool"]))
        input_stage.extend(
            [
                nn.Conv1d(
                    num_channels,
                    self.current_channels,
                    kernel_size=input_filter["kernel_size"],
                    stride=input_filter["stride"],
                    padding=input_filter["kernel_size"] // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(self.current_channels),
                nn.Mish(),
            ]
        )
        self.input_stage = nn.Sequential(*input_stage)

        layers = [2, 2, 2, 2]
        channels = [base_channels, 2 * base_channels, 4 * base_channels, 8 * base_channels]
        self.conv_stage1 = self._make_stage(channels[0], layers[0], conv_filter_list[1])
        self.conv_stage2 = self._make_stage(channels[1], layers[1], conv_filter_list[2])
        self.conv_stage3 = self._make_stage(channels[2], layers[2], conv_filter_list[3])
        self.conv_stage4 = self._make_stage(channels[3], layers[3], conv_filter_list[4])

        self.final_pool = nn.AdaptiveAvgPool1d(1)
        fc_layers: list[nn.Module] = []
        feature_dim = self.current_channels
        for _ in range(fc_stages - 1):
            fc_layers.append(
                nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 2, bias=False),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(feature_dim // 2),
                    nn.Mish(),
                )
            )
            feature_dim //= 2
        self.feature_dim = feature_dim
        fc_layers.append(nn.Linear(feature_dim, num_classes))
        self.fc_stage = nn.Sequential(*fc_layers)
        self.reset_weights()

    def _make_stage(self, channels: int, blocks: int, conv_filter: dict[str, int]) -> nn.Sequential:
        stage: list[nn.Module] = []
        if conv_filter["pool"] > 1:
            stage.append(nn.MaxPool1d(conv_filter["pool"]))

        stride = conv_filter["stride"]
        downsample = None
        if stride != 1 or self.current_channels != channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.current_channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channels),
            )
        stage.append(_BasicBlock1D(self.current_channels, channels, conv_filter["kernel_size"], stride, downsample))
        self.current_channels = channels

        for _ in range(1, blocks):
            stage.append(_BasicBlock1D(self.current_channels, channels, conv_filter["kernel_size"]))
        return nn.Sequential(*stage)

    def reset_weights(self) -> None:
        """Initialize weights following the official CEEDNet implementation."""

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def compute_feature_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate feature embedding for ``x`` with shape [B, C, T]."""

        x = self.input_stage(x)
        x = self.conv_stage1(x)
        x = self.conv_stage2(x)
        x = self.conv_stage3(x)
        x = self.conv_stage4(x)
        x = self.final_pool(x).reshape(x.size(0), -1)
        for layer in self.fc_stage[:-1]:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run CEEDNet and return logits plus penultimate features."""

        features = self.compute_feature_embedding(x)
        logits = self.fc_stage[-1](features)
        return logits, features
