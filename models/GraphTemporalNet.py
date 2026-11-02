"""Lightweight topology-aware graph-temporal EEG classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.MSCNN import MSCNN
except ImportError:
    from .MSCNN import MSCNN


STANDARD_19_CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
)

# Approximate 2-D 10-20 positions, used only to define local electrode topology.
STANDARD_19_POSITIONS = torch.tensor(
    [
        [-0.50, 1.00], [0.50, 1.00],
        [-1.00, 0.50], [-0.50, 0.50], [0.00, 0.55], [0.50, 0.50], [1.00, 0.50],
        [-1.00, 0.00], [-0.50, 0.00], [0.00, 0.00], [0.50, 0.00], [1.00, 0.00],
        [-1.00, -0.50], [-0.50, -0.50], [0.00, -0.55], [0.50, -0.50], [1.00, -0.50],
        [-0.50, -1.00], [0.50, -1.00],
    ],
    dtype=torch.float32,
)


def build_knn_adjacency(positions: torch.Tensor, num_neighbors: int = 4) -> torch.Tensor:
    """Build a symmetric, self-looped, normalized k-nearest-neighbor graph."""

    num_nodes = int(positions.shape[0])
    if not 1 <= num_neighbors < num_nodes:
        raise ValueError("num_neighbors must be in [1, num_nodes - 1].")
    distances = torch.cdist(positions, positions)
    nearest = torch.topk(distances, k=num_neighbors + 1, largest=False).indices[:, 1:]
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=positions.dtype)
    adjacency.scatter_(1, nearest, 1.0)
    adjacency = torch.maximum(adjacency, adjacency.T)
    adjacency.fill_diagonal_(1.0)
    degree_inv_sqrt = adjacency.sum(dim=1).clamp_min(1.0).pow(-0.5)
    return degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]


class GraphConv(nn.Module):
    """Graph convolution for node tensors shaped [B, N, F]."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Propagate node features through a shared normalized adjacency matrix."""

        propagated = torch.einsum("ij,bjf->bif", adjacency, x)
        return F.gelu(self.norm(self.linear(propagated)))


class ElectrodeGraphEncoder(nn.Module):
    """Encode per-channel temporal features on the standard 19-electrode graph."""

    def __init__(
        self,
        num_channels: int = 19,
        node_dim: int = 32,
        graph_hidden_dim: int = 64,
        out_dim: int = 256,
        num_neighbors: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_channels != len(STANDARD_19_CHANNELS):
            raise ValueError("ElectrodeGraphEncoder requires the standard 19-channel layout.")

        adjacency = build_knn_adjacency(STANDARD_19_POSITIONS, num_neighbors)
        self.register_buffer("adjacency", adjacency)
        self.node_temporal_encoder = nn.Sequential(
            nn.Conv1d(1, node_dim, kernel_size=15, padding=7),
            nn.BatchNorm1d(node_dim),
            nn.GELU(),
            nn.Conv1d(node_dim, node_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(node_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.graph_conv1 = GraphConv(node_dim, graph_hidden_dim)
        self.graph_conv2 = GraphConv(graph_hidden_dim, graph_hidden_dim)
        self.graph_hidden_dim = graph_hidden_dim
        self.node_attention = nn.Linear(graph_hidden_dim, 1)
        self.projector = nn.Sequential(
            nn.Linear(graph_hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """Return topology-aware node embeddings shaped [B, 19, graph_hidden_dim]."""

        batch_size, num_channels, num_times = x.shape
        nodes = x.reshape(batch_size * num_channels, 1, num_times)
        nodes = self.node_temporal_encoder(nodes).squeeze(-1)
        nodes = nodes.reshape(batch_size, num_channels, -1)
        nodes = self.graph_conv1(nodes, self.adjacency)
        nodes = self.graph_conv2(nodes, self.adjacency) + nodes
        return nodes

    def pool_nodes(self, nodes: torch.Tensor) -> torch.Tensor:
        """Attention-pool node embeddings and project them to graph features."""

        weights = torch.softmax(self.node_attention(nodes), dim=1)
        graph_feature = torch.sum(weights * nodes, dim=1)
        return self.projector(graph_feature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return graph features from EEG shaped [B, 19, T]."""

        return self.pool_nodes(self.encode_nodes(x))


class GraphTemporalNet(nn.Module):
    """Fuse electrode-graph and multi-scale temporal EEG representations."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 19,
        graph_out_dim: int = 256,
        temporal_out_dim: int = 256,
        classifier_dropout: float = 0.6,
        num_neighbors: int = 4,
    ) -> None:
        super().__init__()
        self.graph_encoder = ElectrodeGraphEncoder(
            num_channels=num_channels,
            out_dim=graph_out_dim,
            num_neighbors=num_neighbors,
        )
        self.temporal_encoder = MSCNN(in_channels=num_channels, embed_dim=temporal_out_dim)
        fused_dim = graph_out_dim + temporal_out_dim
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(fused_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return class logits and fused graph-temporal features."""

        graph_feature, temporal_feature, features = self.encode(x)
        return self.fusion_classifier(features), features

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return graph, temporal, and fused representations."""

        graph_feature = self.graph_encoder(x)
        temporal_feature = self.temporal_encoder(x)
        features = torch.cat([graph_feature, temporal_feature], dim=-1)
        return graph_feature, temporal_feature, features
