"""EEGConformer baseline adapted for the local subject-split runner."""

from __future__ import annotations

import torch
import torch.nn as nn


class ShallowNetEmbedding(nn.Module):
    """Shallow convolutional token embedding from EEG-Conformer.

    Input:
        x: [B, T, C]
    Output:
        tokens: [B, N, D]
    """

    def __init__(self, num_channels: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.shallow_net = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=(1, 25), stride=(1, 1)),
            nn.Conv2d(d_model, d_model, kernel_size=(num_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 4)),
            nn.Dropout(dropout),
        )
        self.projection = nn.Conv2d(d_model, d_model, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project EEG into transformer tokens."""

        x = x.permute(0, 2, 1).unsqueeze(1)  # [B, 1, C, T]
        x = self.shallow_net(x)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class EEGConformer(nn.Module):
    """Compact EEGConformer classifier for [B, C, T] EEG input."""

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        seq_len: int = 128,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = ShallowNetEmbedding(num_channels=num_channels, d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        with torch.no_grad():
            dummy = torch.zeros(1, seq_len, num_channels, dtype=torch.float32)
            feat = self.encoder(self.embedding(dummy))
            fc_in = int(feat.reshape(1, -1).shape[1])

        self.classifier = nn.Linear(fc_in, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass.

        Args:
            x: [B, C, T]
        Returns:
            logits: [B, num_classes]
            features: [B, N * D]
        """

        x = x.transpose(1, 2)  # [B, T, C]
        tokens = self.embedding(x)
        encoded = self.encoder(tokens)
        features = self.dropout(self.activation(encoded)).reshape(encoded.shape[0], -1)
        logits = self.classifier(features)
        return logits, features
