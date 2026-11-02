"""Adapt established EEG classifiers to the shared source-only LODO contract."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.EEGConformer import EEGConformer
from models.ShallowConvNet import ShallowConvNet


class StandardEEGBackboneAdapter(nn.Module):
    """Expose an ERM-only EEG backbone through the common output dictionary.

    The adapter deliberately provides no domain-generalization heads. It is used
    only for controlled ERM architecture-sensitivity experiments.
    """

    def __init__(
        self,
        backbone_name: str,
        num_domains: int,
        num_channels: int = 19,
        num_classes: int = 2,
        seq_len: int = 128,
        dropout: float = 0.5,
        conformer_d_model: int = 128,
        conformer_num_heads: int = 8,
        conformer_num_layers: int = 4,
        conformer_ff_dim: int = 256,
    ) -> None:
        super().__init__()
        if backbone_name == "shallow_convnet":
            self.backbone = ShallowConvNet(
                num_channels=num_channels,
                num_classes=num_classes,
                dropout=dropout,
            )
        elif backbone_name == "eeg_conformer":
            self.backbone = EEGConformer(
                num_channels=num_channels,
                num_classes=num_classes,
                seq_len=seq_len,
                d_model=conformer_d_model,
                n_heads=conformer_num_heads,
                e_layers=conformer_num_layers,
                d_ff=conformer_ff_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported standard EEG backbone: {backbone_name}")
        self.num_domains = int(num_domains)

    def forward(
        self,
        x: torch.Tensor,
        grl_scale: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Return disease logits and contract-compatible unused tensors."""

        del grl_scale
        logits, features = self.backbone(x)
        zero_domains = logits.new_zeros((logits.shape[0], self.num_domains))
        return {
            "disease_logits": logits,
            "disease_feature": features,
            "domain_feature": features,
            "domain_adversarial_logits": zero_domains,
            "domain_specific_logits": zero_domains,
        }
