"""Masked node/time reconstruction pretraining for GraphTemporalNet."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.GraphTemporalNet import GraphTemporalNet
except ImportError:
    from .GraphTemporalNet import GraphTemporalNet


@dataclass
class MaskedReconstructionOutput:
    """Outputs required to optimize and inspect masked reconstruction."""

    reconstruction: torch.Tensor
    masked_input: torch.Tensor
    mask: torch.Tensor
    node_mask: torch.Tensor
    time_mask: torch.Tensor


def make_node_time_mask(
    x: torch.Tensor,
    node_mask_ratio: float,
    time_mask_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create per-sample node masks and contiguous temporal masks."""

    if not 0.0 <= node_mask_ratio < 1.0 or not 0.0 <= time_mask_ratio < 1.0:
        raise ValueError("Mask ratios must be in [0, 1).")
    batch_size, num_channels, num_times = x.shape
    node_mask = torch.zeros((batch_size, num_channels), dtype=torch.bool, device=x.device)
    time_mask = torch.zeros((batch_size, num_times), dtype=torch.bool, device=x.device)

    num_masked_nodes = int(round(num_channels * node_mask_ratio))
    num_masked_times = int(round(num_times * time_mask_ratio))
    if node_mask_ratio > 0.0:
        num_masked_nodes = max(1, num_masked_nodes)
    if time_mask_ratio > 0.0:
        num_masked_times = max(1, num_masked_times)

    for batch_idx in range(batch_size):
        if num_masked_nodes:
            indices = torch.randperm(num_channels, device=x.device)[:num_masked_nodes]
            node_mask[batch_idx, indices] = True
        if num_masked_times:
            start_max = num_times - num_masked_times
            start = int(torch.randint(0, start_max + 1, (1,), device=x.device).item())
            time_mask[batch_idx, start : start + num_masked_times] = True

    full_mask = node_mask.unsqueeze(-1) | time_mask.unsqueeze(1)
    if not full_mask.any():
        raise ValueError("At least one mask ratio must be greater than zero.")
    return full_mask, node_mask, time_mask


class GraphTemporalMaskedAutoencoder(nn.Module):
    """Pretrain graph and temporal encoders by reconstructing masked EEG."""

    def __init__(
        self,
        num_channels: int = 19,
        seq_len: int = 128,
        graph_out_dim: int = 256,
        temporal_out_dim: int = 256,
        classifier_dropout: float = 0.3,
        num_neighbors: int = 4,
        decoder_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = GraphTemporalNet(
            num_classes=2,
            num_channels=num_channels,
            graph_out_dim=graph_out_dim,
            temporal_out_dim=temporal_out_dim,
            classifier_dropout=classifier_dropout,
            num_neighbors=num_neighbors,
        )
        graph_hidden_dim = self.encoder.graph_encoder.graph_hidden_dim
        self.temporal_to_node = nn.Linear(temporal_out_dim, graph_hidden_dim)
        self.signal_decoder = nn.Sequential(
            nn.Linear(graph_hidden_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, seq_len),
        )
        self.num_channels = num_channels
        self.seq_len = seq_len

    def forward(
        self,
        x: torch.Tensor,
        node_mask_ratio: float,
        time_mask_ratio: float,
    ) -> MaskedReconstructionOutput:
        """Mask EEG, encode both views, and reconstruct all channel samples."""

        if x.shape[1:] != (self.num_channels, self.seq_len):
            raise ValueError(
                f"Expected [B, {self.num_channels}, {self.seq_len}], got {tuple(x.shape)}."
            )
        mask, node_mask, time_mask = make_node_time_mask(
            x,
            node_mask_ratio=node_mask_ratio,
            time_mask_ratio=time_mask_ratio,
        )
        masked_input = x.masked_fill(mask, 0.0)
        nodes = self.encoder.graph_encoder.encode_nodes(masked_input)
        temporal = self.encoder.temporal_encoder(masked_input)
        nodes = nodes + self.temporal_to_node(temporal).unsqueeze(1)
        reconstruction = self.signal_decoder(nodes)
        return MaskedReconstructionOutput(
            reconstruction=reconstruction,
            masked_input=masked_input,
            mask=mask,
            node_mask=node_mask,
            time_mask=time_mask,
        )

    @staticmethod
    def masked_loss(output: MaskedReconstructionOutput, target: torch.Tensor) -> torch.Tensor:
        """Compute MSE only on values hidden from the encoder."""

        return F.mse_loss(output.reconstruction[output.mask], target[output.mask])
