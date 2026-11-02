"""TF-C baseline adapted to multichannel 1-second EEG windows.

The design follows the official TF-C implementation by Zhang et al. (NeurIPS
2022): separate time- and frequency-space Transformer encoders, projection
heads for cross-space contrast, and concatenated embeddings for downstream
classification. The original implementation was released under the MIT
License at https://github.com/mims-harvard/TFC-pretraining.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TFCEncoder(nn.Module):
    """Encode time and frequency views of EEG into a shared feature space."""

    def __init__(
        self,
        num_channels: int = 19,
        seq_len: int = 128,
        model_dim: int = 128,
        projection_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_channels = int(num_channels)
        self.seq_len = int(seq_len)
        self.freq_bins = self.seq_len // 2 + 1

        self.time_input = nn.Linear(self.seq_len, model_dim)
        self.freq_input = nn.Linear(self.freq_bins, model_dim)
        self.channel_embedding = nn.Parameter(torch.zeros(1, self.num_channels, model_dim))
        layer_kwargs = dict(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=2 * model_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.time_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**layer_kwargs), num_layers=num_layers
        )
        self.freq_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**layer_kwargs), num_layers=num_layers
        )
        self.time_norm = nn.LayerNorm(model_dim)
        self.freq_norm = nn.LayerNorm(model_dim)
        self.time_projector = nn.Sequential(
            nn.Linear(model_dim, 2 * projection_dim),
            nn.BatchNorm1d(2 * projection_dim),
            nn.ReLU(),
            nn.Linear(2 * projection_dim, projection_dim),
        )
        self.freq_projector = nn.Sequential(
            nn.Linear(model_dim, 2 * projection_dim),
            nn.BatchNorm1d(2 * projection_dim),
            nn.ReLU(),
            nn.Linear(2 * projection_dim, projection_dim),
        )
        nn.init.normal_(self.channel_embedding, std=0.02)

    def frequency_view(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-magnitude one-sided spectra with shape [B, C, F]."""

        return torch.log1p(torch.abs(torch.fft.rfft(x, dim=-1)))

    def forward(
        self, x_time: torch.Tensor, x_frequency: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return time features, time projections, frequency features, and projections."""

        if x_time.ndim != 3 or x_time.shape[1:] != (self.num_channels, self.seq_len):
            raise ValueError(
                f"Expected time input [B, {self.num_channels}, {self.seq_len}], got {tuple(x_time.shape)}"
            )
        if x_frequency is None:
            x_frequency = self.frequency_view(x_time)
        expected_freq = (self.num_channels, self.freq_bins)
        if x_frequency.ndim != 3 or x_frequency.shape[1:] != expected_freq:
            raise ValueError(f"Expected frequency input [B, {expected_freq[0]}, {expected_freq[1]}]")

        time_tokens = self.time_input(x_time) + self.channel_embedding
        freq_tokens = self.freq_input(x_frequency) + self.channel_embedding
        h_time = self.time_norm(self.time_encoder(time_tokens).mean(dim=1))
        h_freq = self.freq_norm(self.freq_encoder(freq_tokens).mean(dim=1))
        z_time = self.time_projector(h_time)
        z_freq = self.freq_projector(h_freq)
        return h_time, z_time, h_freq, z_freq


class TFCClassifier(nn.Module):
    """TF-C encoder with a two-class downstream classification head."""

    def __init__(
        self,
        num_classes: int = 2,
        num_channels: int = 19,
        seq_len: int = 128,
        model_dim: int = 128,
        projection_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.6,
    ) -> None:
        super().__init__()
        self.encoder = TFCEncoder(
            num_channels=num_channels,
            seq_len=seq_len,
            model_dim=model_dim,
            projection_dim=projection_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=min(dropout, 0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return class logits and the concatenated TF-C projection."""

        _, z_time, _, z_freq = self.encoder(x)
        features = torch.cat([F.normalize(z_time, dim=-1), F.normalize(z_freq, dim=-1)], dim=-1)
        return self.classifier(features), features

    def load_pretrained_encoder(self, checkpoint: str, map_location: object = "cpu") -> int:
        """Load compatible TF-C encoder tensors and return their count."""

        payload = torch.load(checkpoint, map_location=map_location)
        state = payload.get("encoder_state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise ValueError("TF-C checkpoint does not contain an encoder state dictionary")
        current = self.encoder.state_dict()
        compatible = {
            key.replace("encoder.", "", 1): value
            for key, value in state.items()
            if isinstance(key, str)
            and isinstance(value, torch.Tensor)
            and key.replace("encoder.", "", 1) in current
            and current[key.replace("encoder.", "", 1)].shape == value.shape
        }
        if not compatible:
            raise ValueError(f"No compatible TF-C encoder tensors found in {checkpoint}")
        self.encoder.load_state_dict(compatible, strict=False)
        return len(compatible)
