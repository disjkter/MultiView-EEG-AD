"""EEGNet backbone with domain-generalization prediction heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.EEGNet import EEGNet
from models.GraphTemporalTACD import gradient_reverse


class EEGNetDomainDecoupling(nn.Module):
    """Expose EEGNet features through the shared DG training contract.

    Input shape:
        x: [batch, channels, time]
    Output features:
        disease_feature: [batch, disease_dim]
        domain_feature: [batch, domain_dim]
    """

    def __init__(
        self,
        num_classes: int = 2,
        num_domains: int = 2,
        num_channels: int = 19,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
        separable_filters: int = 16,
        temporal_kernel: int = 64,
        separable_kernel: int = 16,
        disease_dim: int = 256,
        domain_dim: int = 128,
        classifier_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone = EEGNet(
            num_channels=num_channels,
            num_classes=num_classes,
            temporal_filters=temporal_filters,
            depth_multiplier=depth_multiplier,
            separable_filters=separable_filters,
            temporal_kernel=temporal_kernel,
            separable_kernel=separable_kernel,
            dropout=classifier_dropout,
        )
        self.disease_projector = nn.Sequential(
            nn.Linear(separable_filters, disease_dim),
            nn.LayerNorm(disease_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
        )
        self.domain_projector = nn.Sequential(
            nn.Linear(separable_filters, domain_dim),
            nn.LayerNorm(domain_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
        )
        self.disease_classifier = nn.Linear(disease_dim, num_classes)
        self.domain_adversary = nn.Sequential(
            nn.Linear(disease_dim, domain_dim),
            nn.GELU(),
            nn.Linear(domain_dim, num_domains),
        )
        self.domain_classifier = nn.Linear(domain_dim, num_domains)

    def forward(
        self,
        x: torch.Tensor,
        grl_scale: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Return the same outputs as the graph-temporal DG model."""

        _, backbone_feature = self.backbone(x)
        disease_feature = self.disease_projector(backbone_feature)
        domain_feature = self.domain_projector(backbone_feature)
        return {
            "disease_logits": self.disease_classifier(disease_feature),
            "domain_adversarial_logits": self.domain_adversary(
                gradient_reverse(disease_feature, grl_scale)
            ),
            "domain_specific_logits": self.domain_classifier(domain_feature),
            "disease_feature": disease_feature,
            "domain_feature": domain_feature,
            "backbone_feature": backbone_feature,
        }
