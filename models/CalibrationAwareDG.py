"""Calibration-aware source-only domain-generalization objectives."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


def aggregate_participant_probabilities(
    logits: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    subjects: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average differentiable window probabilities for each participant."""

    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("participant aggregation requires binary [N, 2] logits")
    if not (logits.shape[0] == labels.shape[0] == domains.shape[0] == len(subjects)):
        raise ValueError("logits, labels, domains, and subjects must share the batch axis")
    probabilities = torch.softmax(logits, dim=1)
    grouped: dict[str, list[int]] = {}
    for index, subject in enumerate(subjects):
        grouped.setdefault(str(subject), []).append(index)

    participant_probabilities = []
    participant_labels = []
    participant_domains = []
    for indices in grouped.values():
        index_tensor = torch.as_tensor(indices, device=logits.device)
        group_labels = labels[index_tensor]
        group_domains = domains[index_tensor]
        if not torch.all(group_labels == group_labels[0]):
            raise ValueError("one participant has multiple disease labels in a batch")
        if not torch.all(group_domains == group_domains[0]):
            raise ValueError("one participant appears in multiple source domains")
        participant_probabilities.append(probabilities[index_tensor].mean(dim=0))
        participant_labels.append(group_labels[0])
        participant_domains.append(group_domains[0])
    return (
        torch.stack(participant_probabilities),
        torch.stack(participant_labels),
        torch.stack(participant_domains),
    )


def participant_classification_loss(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute balanced participant-level negative log-likelihood."""

    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("probabilities must have shape [participants, 2]")
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities and labels must share the batch axis")
    return F.nll_loss(probabilities.clamp_min(1e-7).log(), labels)


def class_conditional_quantile_alignment(
    logits: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> torch.Tensor:
    """Align class-conditional score quantiles across observed source cohorts.

    Aligning conditional rather than marginal score distributions avoids forcing
    cohorts with different class prevalences to have identical predictions.
    """

    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("quantile alignment requires binary logits with shape [N, 2]")
    if labels.ndim != 1 or domains.ndim != 1:
        raise ValueError("labels and domains must be one-dimensional")
    if logits.shape[0] != labels.shape[0] or labels.shape[0] != domains.shape[0]:
        raise ValueError("logits, labels, and domains must share the batch axis")
    if not quantiles or any(value < 0.0 or value > 1.0 for value in quantiles):
        raise ValueError("quantiles must be non-empty and lie in [0, 1]")

    margin = logits[:, 1] - logits[:, 0]
    q = margin.new_tensor(quantiles)
    penalties: list[torch.Tensor] = []
    for label in torch.unique(labels):
        summaries = []
        for domain in torch.unique(domains):
            values = margin[(labels == label) & (domains == domain)]
            if values.numel() >= 2:
                summaries.append(torch.quantile(values, q))
        if len(summaries) < 2:
            continue
        stacked = torch.stack(summaries)
        barycenter = stacked.mean(dim=0, keepdim=True)
        penalties.append((stacked - barycenter).square().mean())
    if not penalties:
        return logits.sum() * 0.0
    return torch.stack(penalties).mean()


def _kernel_calibration_error(
    probability: torch.Tensor,
    labels: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    """Return a differentiable squared kernel calibration residual."""

    residual = probability - labels.to(probability.dtype)
    distance = probability[:, None] - probability[None, :]
    kernel = torch.exp(-distance.square() / (2.0 * bandwidth * bandwidth))
    return (residual[:, None] * kernel * residual[None, :]).mean().clamp_min(0.0)


def domain_calibration_risk(
    logits: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    bandwidth: float = 0.1,
    equalization_weight: float = 1.0,
) -> torch.Tensor:
    """Minimize and equalize differentiable calibration error across cohorts."""

    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    if equalization_weight < 0.0:
        raise ValueError("equalization_weight must be non-negative")
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("calibration risk requires binary logits with shape [N, 2]")
    if logits.shape[0] != labels.shape[0] or labels.shape[0] != domains.shape[0]:
        raise ValueError("logits, labels, and domains must share the batch axis")

    probability = torch.softmax(logits, dim=1)[:, 1]
    risks = [
        _kernel_calibration_error(
            probability[domains == domain], labels[domains == domain], bandwidth
        )
        for domain in torch.unique(domains)
        if int((domains == domain).sum()) >= 2
    ]
    if not risks:
        return logits.sum() * 0.0
    stacked = torch.stack(risks)
    return stacked.mean() + equalization_weight * stacked.var(unbiased=False)


def error_aware_entropy_regularization(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Suppress low-entropy predictions in proportion to their error evidence."""

    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("entropy regularization requires logits with shape [N, C]")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must share the batch axis")
    probability = torch.softmax(logits, dim=1).clamp_min(1e-7)
    true_probability = probability.gather(1, labels[:, None]).squeeze(1)
    normalized_entropy = -(probability * probability.log()).sum(dim=1) / math.log(
        probability.shape[1]
    )
    return ((1.0 - true_probability) * (1.0 - normalized_entropy)).mean()


def calibration_aware_dg_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    *,
    calibration_bandwidth: float = 0.1,
    calibration_equalization_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute the three CARE-DG regularizers under one stable interface."""

    return {
        "score_alignment": class_conditional_quantile_alignment(
            logits, labels, domains
        ),
        "calibration": domain_calibration_risk(
            logits,
            labels,
            domains,
            bandwidth=calibration_bandwidth,
            equalization_weight=calibration_equalization_weight,
        ),
        "error_entropy": error_aware_entropy_regularization(logits, labels),
    }
