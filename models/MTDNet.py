"""Minimal MTDNet baseline adapted from the official implementation."""

from __future__ import annotations

import torch
import torch.nn as nn


class MTDNet(nn.Module):
    """Multiscale Temporal Deep Network for EEG classification.

    Input shape:
        x: [batch, channels, time]
    Output shape:
        logits: [batch, num_classes]
        features: [batch, hidden_dim]
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        first_layer_channels: int = 32,
        lstm_hidden_size: int = 16,
        post_hidden_dims: tuple[int, ...] = (16,),
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.first_layer_channels = first_layer_channels
        self.lstm_hidden_size = lstm_hidden_size

        # The three branches follow the official MTDNet repository:
        # scale-1: Conv(k=10,s=10)
        # scale-2: Conv(k=5,s=5) + AvgPool(2)
        # scale-3: Conv(k=2,s=2) + AvgPool(5)
        self.time_layers_s1 = nn.Sequential(
            nn.Conv1d(num_channels, first_layer_channels, kernel_size=10, stride=10, bias=True),
            nn.BatchNorm1d(first_layer_channels),
            nn.ReLU(inplace=False),
        )
        self.time_layers_s2 = nn.Sequential(
            nn.Conv1d(num_channels, first_layer_channels, kernel_size=5, stride=5, bias=True),
            nn.BatchNorm1d(first_layer_channels),
            nn.ReLU(inplace=False),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )
        self.time_layers_s3 = nn.Sequential(
            nn.Conv1d(num_channels, first_layer_channels, kernel_size=2, stride=2, bias=True),
            nn.BatchNorm1d(first_layer_channels),
            nn.ReLU(inplace=False),
            nn.AvgPool1d(kernel_size=5, stride=5),
        )

        self.lstm = nn.LSTM(
            input_size=first_layer_channels * 3,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )
        self.time_norm = nn.BatchNorm1d(lstm_hidden_size)

        post_layers: list[nn.Module] = []
        in_features = lstm_hidden_size
        for hidden_dim in post_hidden_dims:
            post_layers.extend(
                [
                    nn.Linear(in_features, hidden_dim, bias=True),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=False),
                ]
            )
            in_features = hidden_dim
        post_layers.append(nn.Linear(in_features, num_classes, bias=True))
        self.post_lstm = nn.Sequential(*post_layers)

    def multiscale_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract aligned multiscale temporal features."""

        x_s1 = self.time_layers_s1(x)
        x_s2 = self.time_layers_s2(x)
        x_s3 = self.time_layers_s3(x)
        return torch.cat([x_s1, x_s2, x_s3], dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass."""

        batch_size, _, num_timepoints = x.shape
        multiscale = self.multiscale_features(x)
        sequence = multiscale.view(batch_size, num_timepoints // 10, -1).contiguous()
        lstm_out, _ = self.lstm(sequence)
        features = self.time_norm(lstm_out[:, -1, :])
        logits = self.post_lstm(features)
        return logits, features
