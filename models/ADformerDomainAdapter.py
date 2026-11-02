"""Adapt the official ADformer implementation to the shared LODO contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


from external.ADformer.models.ADformer import Model as OfficialADformer  # noqa: E402


class ADformerDomainAdapter(nn.Module):
    """Expose official ADformer logits through the common DG output dictionary."""

    def __init__(
        self,
        num_domains: int,
        num_channels: int = 19,
        seq_len: int = 128,
        patch_len_list: str = "2",
        up_dim_list: str = "19",
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 6,
        d_ff: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("ADformer d_model must be divisible by n_heads")
        config = SimpleNamespace(
            task_name="supervised",
            seq_len=seq_len,
            output_attention=False,
            enc_in=num_channels,
            no_temporal_block=False,
            no_spatial_block=False,
            patch_len_list=patch_len_list,
            up_dim_list=up_dim_list,
            augmentations="none",
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            no_inter_attn=False,
            d_ff=d_ff,
            activation="gelu",
            e_layers=e_layers,
            num_class=2,
        )
        self.backbone = OfficialADformer(config)
        self.num_domains = num_domains

    def forward(self, x: torch.Tensor, grl_scale: float = 0.0) -> dict[str, torch.Tensor]:
        """Return ADformer disease logits and unused contract-compatible tensors.

        Args:
            x: EEG windows shaped ``[batch, 19, 128]``.
            grl_scale: Accepted for interface compatibility; P0-D uses ERM only.
        """

        del grl_scale
        sequence = x.transpose(1, 2)  # [B, T, C]
        logits = self.backbone(sequence, None, None, None)
        zero_domains = logits.new_zeros((logits.shape[0], self.num_domains))
        return {
            "disease_logits": logits,
            "disease_feature": logits,
            "domain_feature": logits,
            "domain_adversarial_logits": zero_domains,
            "domain_specific_logits": zero_domains,
        }
