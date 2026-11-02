"""MSCNN-only EEG classifier used as a temporal-branch control."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from models.MSCNN import MSCNN
except ImportError:
    from .MSCNN import MSCNN


class MSCNNClassifier(nn.Module):
    """Classify EEG using only the multi-scale temporal encoder."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 19,
        temporal_out_dim: int = 256,
        classifier_dropout: float = 0.6,
    ) -> None:
        super().__init__()
        self.temporal_encoder = MSCNN(in_channels=num_channels, embed_dim=temporal_out_dim)
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(temporal_out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and temporal features for EEG shaped [B, C, T]."""

        features = self.temporal_encoder(x)
        return self.fusion_classifier(features), features
