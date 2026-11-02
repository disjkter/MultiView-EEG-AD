"""Minimal iTransformer-style baseline for EEG classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class ITransformer(nn.Module):
    """Invert the time/channel roles and run self-attention across channels.

    Input shape:
        x: [batch, channels, time]
    Output shape:
        logits: [batch, num_classes]
        features: [batch, embed_dim]
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        depth: int = 2,
        ff_multiplier: int = 4,
        dropout: float = 0.3,
        pooled_time_steps: int = 128,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.pooled_time_steps = pooled_time_steps

        self.time_pool = nn.AdaptiveAvgPool1d(pooled_time_steps)
        self.input_projection = nn.Linear(pooled_time_steps, embed_dim)
        self.channel_embedding = nn.Parameter(torch.zeros(1, num_channels, embed_dim))
        self.norm_in = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ff_multiplier,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.channel_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass."""

        # x: [B, C, T]
        x = self.time_pool(x)  # [B, C, pooled_time_steps]
        x = self.input_projection(x)  # [B, C, embed_dim]
        x = self.norm_in(x + self.channel_embedding)
        x = self.encoder(x)  # [B, C, embed_dim]
        features = self.norm_out(x.mean(dim=1))  # [B, embed_dim]
        logits = self.classifier(self.dropout(features))
        return logits, features
