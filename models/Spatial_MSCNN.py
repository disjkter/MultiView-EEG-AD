"""Dual-stream EEG classifier with configurable spatial and temporal branches."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.MSCNN import MSCNN
    from models.neurophysiological_prior import DEFAULT_19_CHANNELS, NeurophysiologicalPriorLayer
except ImportError:
    from .MSCNN import MSCNN
    from .neurophysiological_prior import DEFAULT_19_CHANNELS, NeurophysiologicalPriorLayer


class SpatialExpert(nn.Module):
    """Lightweight spatial branch for cross-channel EEG topology cues."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        temporal_channels: int = 128,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.spatial_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.temporal_conv = nn.Conv1d(hidden_channels, temporal_channels, kernel_size=15, padding=7)
        self.bn2 = nn.BatchNorm1d(temporal_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(temporal_channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a spatial embedding with shape [B, out_dim]."""

        x = F.gelu(self.bn1(self.spatial_conv(x)))
        x = F.gelu(self.bn2(self.temporal_conv(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class SpatialMSCNN_DualStream(nn.Module):
    """Dual-stream EEG classifier with optional neurophysiological priors."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int,
        spatial_out_dim: int = 256,
        temporal_out_dim: int = 256,
        spatial_hidden_channels: int = 64,
        spatial_temporal_channels: int = 128,
        classifier_hidden_dim: int | None = None,
        classifier_num_layers: int = 2,
        classifier_dropout: float = 0.6,
        use_neuro_prior: bool = False,
        sampling_rate: float = 128.0,
        prior_out_dim: int = 64,
        use_sampling_rate_embedding: bool = False,
        sampling_rate_options: tuple[float, ...] = (128.0, 256.0),
    ) -> None:
        super().__init__()
        if use_neuro_prior and num_channels != len(DEFAULT_19_CHANNELS):
            raise ValueError("NeurophysiologicalPriorLayer currently expects the standard 19-channel layout.")

        self.spatial_expert = SpatialExpert(
            in_channels=num_channels,
            hidden_channels=spatial_hidden_channels,
            temporal_channels=spatial_temporal_channels,
            out_dim=spatial_out_dim,
        )
        self.femba = MSCNN(in_channels=num_channels, embed_dim=temporal_out_dim)
        self.use_neuro_prior = use_neuro_prior

        prior_feature_dim = 0
        if self.use_neuro_prior:
            self.neuro_prior = NeurophysiologicalPriorLayer(sampling_rate=sampling_rate)
            prior_input_dim = len(self.neuro_prior.region_names) + len(self.neuro_prior.band_names) + (
                num_channels * (num_channels - 1) // 2
            )
            self.prior_projector = nn.Sequential(
                nn.Linear(prior_input_dim, prior_out_dim),
                nn.BatchNorm1d(prior_out_dim),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
            )
            prior_feature_dim = prior_out_dim

        fused_dim = spatial_out_dim + temporal_out_dim + prior_feature_dim
        self.use_sampling_rate_embedding = use_sampling_rate_embedding
        self.sampling_rate_options = tuple(float(value) for value in sampling_rate_options)
        self.default_sampling_rate = float(sampling_rate)
        if self.use_sampling_rate_embedding:
            self.sampling_rate_embedding = nn.Embedding(len(self.sampling_rate_options), fused_dim)

        if classifier_num_layers == 1:
            self.fusion_classifier = nn.Sequential(
                nn.Dropout(classifier_dropout),
                nn.Linear(fused_dim, num_classes),
            )
        elif classifier_num_layers == 2:
            hidden_dim = classifier_hidden_dim or max(fused_dim // 2, num_classes)
            self.fusion_classifier = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError("classifier_num_layers must be 1 or 2.")

    def _sampling_rate_ids(self, sampling_rate: float | torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        """Map sampling-rate values to embedding IDs with shape [B]."""

        value = self.default_sampling_rate if sampling_rate is None else sampling_rate
        options = torch.tensor(self.sampling_rate_options, dtype=torch.float32, device=device)
        if torch.is_tensor(value):
            rates = value.to(device=device, dtype=torch.float32).view(-1)
            if rates.numel() == 1:
                rates = rates.expand(batch_size)
        else:
            rates = torch.full((batch_size,), float(value), dtype=torch.float32, device=device)
        return torch.argmin(torch.abs(rates.unsqueeze(-1) - options.unsqueeze(0)), dim=-1)

    def forward(self, x: torch.Tensor, sampling_rate: float | torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and fused features with shape [B, fused_dim]."""

        feat_spatial = self.spatial_expert(x)
        feat_temporal = self.femba(x)
        feature_parts = [feat_spatial, feat_temporal]
        if self.use_neuro_prior:
            prior = self.neuro_prior(x)
            upper_i, upper_j = torch.triu_indices(x.shape[1], x.shape[1], offset=1, device=x.device)
            connectivity_features = prior.connectivity[:, upper_i, upper_j]
            prior_features = torch.cat(
                [
                    prior.region_power,
                    prior.band_power.mean(dim=1),
                    connectivity_features,
                ],
                dim=-1,
            )
            feature_parts.append(self.prior_projector(prior_features))
        fused_features = torch.cat(feature_parts, dim=-1)
        if self.use_sampling_rate_embedding:
            sr_ids = self._sampling_rate_ids(sampling_rate, x.shape[0], x.device)
            fused_features = fused_features + self.sampling_rate_embedding(sr_ids)
        logits = self.fusion_classifier(fused_features)
        return logits, fused_features
