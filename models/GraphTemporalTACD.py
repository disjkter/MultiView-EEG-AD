"""Topology-aware graph-temporal network with causal domain decoupling."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from models.GraphTemporalNet import GraphTemporalNet


class GradientReversalFunction(Function):
    """Identity forward operation with a sign-reversed backward gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def gradient_reverse(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply gradient reversal with the requested scale."""

    return GradientReversalFunction.apply(x, scale)


def dann_schedule(step: int, total_steps: int) -> float:
    """Return the standard smooth DANN coefficient in [0, 1)."""

    progress = float(step) / max(total_steps - 1, 1)
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


class GraphTemporalTACD(nn.Module):
    """Decouple disease-discriminative and domain-specific EEG features."""

    def __init__(
        self,
        num_classes: int = 2,
        num_domains: int = 2,
        num_channels: int = 19,
        graph_out_dim: int = 256,
        temporal_out_dim: int = 256,
        disease_dim: int = 256,
        domain_dim: int = 128,
        classifier_dropout: float = 0.5,
        num_neighbors: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = GraphTemporalNet(
            num_classes=num_classes,
            num_channels=num_channels,
            graph_out_dim=graph_out_dim,
            temporal_out_dim=temporal_out_dim,
            classifier_dropout=classifier_dropout,
            num_neighbors=num_neighbors,
        )
        fused_dim = graph_out_dim + temporal_out_dim
        self.disease_projector = nn.Sequential(
            nn.Linear(fused_dim, disease_dim),
            nn.LayerNorm(disease_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
        )
        self.domain_projector = nn.Sequential(
            nn.Linear(fused_dim, domain_dim),
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
        """Return disease, adversarial-domain, and specific-domain outputs."""

        graph_feature, temporal_feature, fused_feature = self.backbone.encode(x)
        disease_feature = self.disease_projector(fused_feature)
        domain_feature = self.domain_projector(fused_feature)
        return {
            "disease_logits": self.disease_classifier(disease_feature),
            "domain_adversarial_logits": self.domain_adversary(
                gradient_reverse(disease_feature, grl_scale)
            ),
            "domain_specific_logits": self.domain_classifier(domain_feature),
            "disease_feature": disease_feature,
            "domain_feature": domain_feature,
            "graph_feature": graph_feature,
            "temporal_feature": temporal_feature,
        }


def cross_covariance_penalty(
    disease_feature: torch.Tensor,
    domain_feature: torch.Tensor,
) -> torch.Tensor:
    """Penalize linear dependence between centered disease/domain features."""

    disease_centered = disease_feature - disease_feature.mean(dim=0, keepdim=True)
    domain_centered = domain_feature - domain_feature.mean(dim=0, keepdim=True)
    covariance = disease_centered.T @ domain_centered
    covariance = covariance / max(disease_feature.shape[0] - 1, 1)
    return covariance.square().mean()


def class_conditional_margin_alignment(
    disease_logits: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    variance_weight: float = 0.1,
) -> torch.Tensor:
    """Align binary disease-logit margins across source domains per class."""

    if disease_logits.ndim != 2 or disease_logits.shape[1] != 2:
        raise ValueError(
            "class_conditional_margin_alignment requires binary [N, 2] logits"
        )
    if labels.ndim != 1 or domains.ndim != 1:
        raise ValueError("labels and domains must be one-dimensional")
    if not (
        disease_logits.shape[0] == labels.shape[0] == domains.shape[0]
    ):
        raise ValueError("logits, labels, and domains must share the batch axis")
    if variance_weight < 0:
        raise ValueError("variance_weight must be non-negative")

    margins = disease_logits[:, 1] - disease_logits[:, 0]
    class_penalties: list[torch.Tensor] = []
    for label in torch.unique(labels):
        class_mask = labels == label
        group_margins = [
            margins[class_mask & (domains == domain)]
            for domain in torch.unique(domains[class_mask])
        ]
        group_margins = [group for group in group_margins if group.numel() > 0]
        if len(group_margins) < 2:
            continue

        means = torch.stack([group.mean() for group in group_margins])
        penalty = (means - means.mean()).square().mean()

        variance_groups = [group for group in group_margins if group.numel() > 1]
        variances = (
            torch.stack([group.var(unbiased=False) for group in variance_groups])
            if len(variance_groups) >= 2
            else None
        )
        if variances is not None:
            penalty = penalty + variance_weight * (
                variances - variances.mean()
            ).square().mean()
        class_penalties.append(penalty)

    if not class_penalties:
        return disease_logits.sum() * 0.0
    return torch.stack(class_penalties).mean()


def multi_domain_coral_loss(
    features: torch.Tensor,
    domains: torch.Tensor,
) -> torch.Tensor:
    """Align feature covariances between every pair of observed source domains."""

    if features.ndim != 2 or domains.ndim != 1:
        raise ValueError("features must be [N, D] and domains must be [N]")
    if features.shape[0] != domains.shape[0]:
        raise ValueError("features and domains must share the batch axis")
    covariances: list[torch.Tensor] = []
    for domain in torch.unique(domains):
        group = features[domains == domain]
        if group.shape[0] < 2:
            continue
        centered = group - group.mean(dim=0, keepdim=True)
        covariances.append(centered.T @ centered / (group.shape[0] - 1))
    if len(covariances) < 2:
        return features.sum() * 0.0
    feature_dim = features.shape[1]
    penalties = [
        (left - right).square().sum() / (4.0 * feature_dim * feature_dim)
        for index, left in enumerate(covariances)
        for right in covariances[index + 1 :]
    ]
    return torch.stack(penalties).mean()


def _multi_kernel_rbf(
    left: torch.Tensor,
    right: torch.Tensor,
    bandwidths: tuple[float, ...],
) -> torch.Tensor:
    """Evaluate a mean multi-bandwidth RBF kernel matrix."""

    squared_distance = torch.cdist(left, right, p=2).square()
    kernels = [
        torch.exp(-squared_distance / (2.0 * bandwidth * bandwidth))
        for bandwidth in bandwidths
    ]
    return torch.stack(kernels).mean(dim=0)


def multi_domain_mmd_loss(
    features: torch.Tensor,
    domains: torch.Tensor,
    bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    """Align every pair of source domains with multi-kernel biased MMD."""

    if features.ndim != 2 or domains.ndim != 1:
        raise ValueError("features must be [N, D] and domains must be [N]")
    if features.shape[0] != domains.shape[0]:
        raise ValueError("features and domains must share the batch axis")
    if not bandwidths or any(value <= 0 for value in bandwidths):
        raise ValueError("bandwidths must contain positive values")
    normalized = F.normalize(features, dim=1)
    groups = [normalized[domains == domain] for domain in torch.unique(domains)]
    groups = [group for group in groups if group.shape[0] > 0]
    if len(groups) < 2:
        return features.sum() * 0.0
    penalties = []
    for index, left in enumerate(groups):
        for right in groups[index + 1 :]:
            penalties.append(
                _multi_kernel_rbf(left, left, bandwidths).mean()
                + _multi_kernel_rbf(right, right, bandwidths).mean()
                - 2.0 * _multi_kernel_rbf(left, right, bandwidths).mean()
            )
    return torch.stack(penalties).mean()
