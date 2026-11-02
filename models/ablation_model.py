"""Ablation-ready dual-stream EEG model with configurable architecture components.

Usage::

    from models.ablation_model import build_ablation_model, ABLATION_VARIANTS

    model = build_ablation_model(
        variant="full",             # or "no_spatial", "no_mscnn", ...
        num_classes=2, num_channels=19,
        checkpoint="checkpoints/tage_timefreq_consistency_adssl_1s_lead.pth",
    )
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.MSCNN import MSCNN
    from models.Spatial_MSCNN import SpatialExpert
    from models.neurophysiological_prior import (
        DEFAULT_19_CHANNELS,
        NeurophysiologicalPriorLayer,
    )
except ImportError:
    from .MSCNN import MSCNN
    from .Spatial_MSCNN import SpatialExpert
    from .neurophysiological_prior import (
        DEFAULT_19_CHANNELS,
        NeurophysiologicalPriorLayer,
    )


# ---------------------------------------------------------------------------
# Ablated MSCNN variants
# ---------------------------------------------------------------------------


class MSCNN_NoMultiScale(nn.Module):
    """MSCNN with a single-scale convolution (kernel=51) instead of multi-scale."""

    def __init__(self, in_channels: int, embed_dim: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=51, padding=25)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=51, padding=25)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=51, padding=25)
        self.bn3 = nn.BatchNorm1d(256)

        # SE Block
        self.se_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.se_fc = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 256),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.gelu(self.bn3(self.conv3(x)))
        se_weight = self.se_avg_pool(x).squeeze(-1)
        se_weight = self.se_fc(se_weight).unsqueeze(-1)
        x = x * se_weight
        x = x.mean(dim=-1)
        return self.proj(x)


class MSCNN_NoSE(nn.Module):
    """MSCNN without SE channel attention block."""

    def __init__(self, in_channels: int, embed_dim: int = 256) -> None:
        super().__init__()
        branch_channels_1 = 64 // 3
        rem_1 = 64 - 2 * branch_channels_1
        self.branch1_1 = nn.Conv1d(
            in_channels, branch_channels_1, kernel_size=51, padding=25
        )
        self.branch1_2 = nn.Conv1d(
            in_channels, branch_channels_1, kernel_size=25, padding=12
        )
        self.branch1_3 = nn.Conv1d(in_channels, rem_1, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4)

        branch_channels_2 = 128 // 3
        rem_2 = 128 - 2 * branch_channels_2
        self.branch2_1 = nn.Conv1d(64, branch_channels_2, kernel_size=51, padding=25)
        self.branch2_2 = nn.Conv1d(64, branch_channels_2, kernel_size=25, padding=12)
        self.branch2_3 = nn.Conv1d(64, rem_2, kernel_size=11, padding=5)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)

        branch_channels_3 = 256 // 3
        rem_3 = 256 - 2 * branch_channels_3
        self.branch3_1 = nn.Conv1d(128, branch_channels_3, kernel_size=51, padding=25)
        self.branch3_2 = nn.Conv1d(128, branch_channels_3, kernel_size=25, padding=12)
        self.branch3_3 = nn.Conv1d(128, rem_3, kernel_size=11, padding=5)
        self.bn3 = nn.BatchNorm1d(256)

        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.branch1_1(x)
        x2 = self.branch1_2(x)
        x3 = self.branch1_3(x)
        x = F.gelu(self.bn1(torch.cat([x1, x2, x3], dim=1)))
        x = self.pool1(x)

        x1 = self.branch2_1(x)
        x2 = self.branch2_2(x)
        x3 = self.branch2_3(x)
        x = F.gelu(self.bn2(torch.cat([x1, x2, x3], dim=1)))
        x = self.pool2(x)

        x1 = self.branch3_1(x)
        x2 = self.branch3_2(x)
        x3 = self.branch3_3(x)
        x = F.gelu(self.bn3(torch.cat([x1, x2, x3], dim=1)))
        x = x.mean(dim=-1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------


class AttentionFusion(nn.Module):
    """Learnable attention-weighted fusion of two feature vectors."""

    def __init__(self, dim_a: int, dim_b: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.proj_a = nn.Linear(dim_a, hidden_dim)
        self.proj_b = nn.Linear(dim_b, hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        proj_a = self.proj_a(feat_a)
        proj_b = self.proj_b(feat_b)
        attn_input = torch.cat([proj_a, proj_b], dim=-1)
        weights = self.attention(attn_input)  # [B, 2]
        return weights[:, 0:1] * feat_a + weights[:, 1:2] * feat_b


class GatedFusion(nn.Module):
    """Gated fusion: gate = sigmoid(W·[a;b]), output = gate*a + (1-gate)*b."""

    def __init__(self, dim_a: int, dim_b: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim_a + dim_b, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim_a),
            nn.Sigmoid(),
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([feat_a, feat_b], dim=-1))
        return gate * feat_a + (1.0 - gate) * feat_b


# ---------------------------------------------------------------------------
# Build classifier head with configurable depth
# ---------------------------------------------------------------------------


def _build_classifier(
    in_dim: int,
    num_classes: int,
    hidden_dim: int | None,
    dropout: float,
    num_layers: int = 2,
) -> nn.Sequential:
    """Build a classifier head with 1-3 linear layers + BN + GELU + Dropout."""

    if num_layers == 1:
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, num_classes))

    layers: list[nn.Module] = []
    current_dim = in_dim
    for i in range(num_layers - 1):
        h_dim = hidden_dim or max(in_dim // 2, num_classes)
        if i == 0:
            layers.extend(
                [
                    nn.Linear(current_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        else:
            next_h = max(h_dim // 2, num_classes * 2)
            layers.extend(
                [
                    nn.Linear(current_dim, next_h),
                    nn.BatchNorm1d(next_h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = next_h
        current_dim = h_dim if i == 0 else next_h

    layers.append(nn.Linear(current_dim, num_classes))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Main ablation model
# ---------------------------------------------------------------------------


class SpatialMSCNN_Ablation(nn.Module):
    """Configurable dual-stream EEG classifier for ablation studies.

    Parameters
    ----------
    ablation : dict
        Configuration dict with keys:
        - use_spatial_expert (bool)
        - use_mscnn (bool)
        - mscnn_multiscale (bool)
        - mscnn_se_attention (bool)
        - fusion_mode: "concat" | "attention" | "gate"
        - num_classifier_layers: 1 | 2 | 3
        - dropout (float)
        - use_neuro_prior (bool)
    """

    _DEFAULT_ABLATION: dict[str, object] = {
        "use_spatial_expert": True,
        "use_mscnn": True,
        "mscnn_multiscale": True,
        "mscnn_se_attention": True,
        "fusion_mode": "concat",
        "num_classifier_layers": 2,
        "dropout": 0.6,
        "use_neuro_prior": False,
    }

    def __init__(
        self,
        num_classes: int,
        num_channels: int,
        spatial_out_dim: int = 256,
        temporal_out_dim: int = 256,
        spatial_hidden_channels: int = 64,
        spatial_temporal_channels: int = 128,
        classifier_hidden_dim: int | None = None,
        classifier_dropout: float = 0.6,
        use_neuro_prior: bool = False,
        sampling_rate: float = 128.0,
        prior_out_dim: int = 64,
        ablation: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        cfg = dict(self._DEFAULT_ABLATION)
        if ablation:
            cfg.update(ablation)
        self.ablation_cfg = cfg

        # Validate: need at least one branch
        if not cfg["use_spatial_expert"] and not cfg["use_mscnn"]:
            raise ValueError(
                "At least one branch (spatial_expert or mscnn) must be enabled."
            )

        # --- Spatial Expert branch ---
        if cfg["use_spatial_expert"]:
            self.spatial_expert = SpatialExpert(
                in_channels=num_channels,
                hidden_channels=spatial_hidden_channels,
                temporal_channels=spatial_temporal_channels,
                out_dim=spatial_out_dim,
            )
        else:
            self.spatial_expert = None

        # --- MSCNN branch ---
        if cfg["use_mscnn"]:
            use_ms = bool(cfg.get("mscnn_multiscale", True))
            use_se = bool(cfg.get("mscnn_se_attention", True))
            if use_ms and use_se:
                self.femba = MSCNN(in_channels=num_channels, embed_dim=temporal_out_dim)
            elif use_ms and not use_se:
                self.femba = MSCNN_NoSE(
                    in_channels=num_channels, embed_dim=temporal_out_dim
                )
            elif not use_ms and use_se:
                self.femba = MSCNN_NoMultiScale(
                    in_channels=num_channels, embed_dim=temporal_out_dim
                )
            else:
                # No multi-scale, no SE — single-scale convs without attention
                self.femba = MSCNN_NoMultiScale(
                    in_channels=num_channels, embed_dim=temporal_out_dim
                )
        else:
            self.femba = None

        # --- Neuro Prior ---
        prior_feature_dim = 0
        self.use_neuro_prior = bool(cfg.get("use_neuro_prior", False))
        if self.use_neuro_prior:
            if num_channels != len(DEFAULT_19_CHANNELS):
                raise ValueError(
                    "NeurophysiologicalPriorLayer requires 19-channel layout."
                )
            self.neuro_prior = NeurophysiologicalPriorLayer(sampling_rate=sampling_rate)
            prior_input_dim = (
                len(self.neuro_prior.region_names)
                + len(self.neuro_prior.band_names)
                + (num_channels * (num_channels - 1) // 2)
            )
            self.prior_projector = nn.Sequential(
                nn.Linear(prior_input_dim, prior_out_dim),
                nn.BatchNorm1d(prior_out_dim),
                nn.GELU(),
                nn.Dropout(cfg.get("dropout", classifier_dropout)),
            )
            prior_feature_dim = prior_out_dim

        # --- Fusion ---
        self.fusion_mode = str(cfg.get("fusion_mode", "concat"))
        branch_dims = []
        if cfg["use_spatial_expert"]:
            branch_dims.append(spatial_out_dim)
        if cfg["use_mscnn"]:
            branch_dims.append(temporal_out_dim)
        total_branch_dim = sum(branch_dims) + prior_feature_dim

        if len(branch_dims) == 2 and self.fusion_mode == "attention":
            self.fusion = AttentionFusion(branch_dims[0], branch_dims[1])
            fused_dim = max(branch_dims[0], branch_dims[1]) + prior_feature_dim
        elif len(branch_dims) == 2 and self.fusion_mode == "gate":
            self.fusion = GatedFusion(branch_dims[0], branch_dims[1])
            fused_dim = branch_dims[0] + prior_feature_dim
        else:
            self.fusion = None
            fused_dim = total_branch_dim

        # --- Classifier ---
        num_cls_layers = int(cfg.get("num_classifier_layers", 2))
        if num_cls_layers not in (1, 2, 3):
            num_cls_layers = 2
        self.classifier = _build_classifier(
            in_dim=fused_dim,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
            dropout=float(cfg.get("dropout", classifier_dropout)),
            num_layers=num_cls_layers,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, fused_features)."""

        features: list[torch.Tensor] = []

        if self.spatial_expert is not None:
            features.append(self.spatial_expert(x))
        if self.femba is not None:
            features.append(self.femba(x))

        if self.use_neuro_prior:
            prior = self.neuro_prior(x)
            upper_i, upper_j = torch.triu_indices(
                x.shape[1], x.shape[1], offset=1, device=x.device
            )
            connectivity_features = prior.connectivity[:, upper_i, upper_j]
            prior_features = torch.cat(
                [
                    prior.region_power,
                    prior.band_power.mean(dim=1),
                    connectivity_features,
                ],
                dim=-1,
            )
            features.append(self.prior_projector(prior_features))

        if self.fusion is not None and len(features) == 2:
            fused_branches = self.fusion(features[0], features[1])
            if len(features) > 2:
                fused_features = torch.cat([fused_branches, *features[2:]], dim=-1)
            else:
                fused_features = fused_branches
        else:
            fused_features = torch.cat(features, dim=-1)

        logits = self.classifier(fused_features)
        return logits, fused_features


# ---------------------------------------------------------------------------
# Variant registry + builder
# ---------------------------------------------------------------------------

ABLATION_VARIANTS: dict[str, dict[str, object]] = {
    "full": {
        "label": "Full Dual-Stream",
        "use_spatial_expert": True,
        "use_mscnn": True,
        "mscnn_multiscale": True,
        "mscnn_se_attention": True,
        "fusion_mode": "concat",
        "num_classifier_layers": 2,
        "dropout": 0.6,
        "use_neuro_prior": False,
    },
    "no_spatial": {
        "label": "w/o Spatial Expert",
        "use_spatial_expert": False,
        "use_mscnn": True,
        "mscnn_multiscale": True,
        "mscnn_se_attention": True,
        "fusion_mode": "concat",
        "num_classifier_layers": 2,
        "dropout": 0.6,
        "use_neuro_prior": False,
    },
    "no_mscnn": {
        "label": "w/o MSCNN (Temporal)",
        "use_spatial_expert": True,
        "use_mscnn": False,
        "mscnn_multiscale": True,
        "mscnn_se_attention": True,
        "fusion_mode": "concat",
        "num_classifier_layers": 2,
        "dropout": 0.6,
        "use_neuro_prior": False,
    },
    "no_multiscale": {
        "label": "w/o Multi-Scale Conv",
        "use_spatial_expert": True,
        "use_mscnn": True,
        "mscnn_multiscale": False,
        "mscnn_se_attention": True,
        "fusion_mode": "concat",
        "num_classifier_layers": 2,
        "dropout": 0.6,
        "use_neuro_prior": False,
    },
    "no_se": {
        "label": "w/o SE Attention",
        "use_spatial_expert": True,
        "use_mscnn": True,
        "mscnn_multiscale": True,
        "mscnn_se_attention": False,
        "fusion_mode": "concat",
        "num_classifier_layers": 2,
        "dropout": 0.6,
        "use_neuro_prior": False,
    },
    "attn_fusion": {
        "label": "Attention Fusion",
        "use_spatial_expert": True,
        "use_mscnn": True,
        "mscnn_multiscale": True,
        "mscnn_se_attention": True,
        "fusion_mode": "attention",
        "num_classifier_layers": 2,
        "dropout": 0.6,
        "use_neuro_prior": False,
    },
    "gate_fusion": {
        "label": "Gated Fusion",
        "use_spatial_expert": True,
        "use_mscnn": True,
        "mscnn_multiscale": True,
        "mscnn_se_attention": True,
        "fusion_mode": "gate",
        "num_classifier_layers": 2,
        "dropout": 0.6,
        "use_neuro_prior": False,
    },
    "cls_1layer": {
        "label": "1-Layer Classifier",
        "use_spatial_expert": True,
        "use_mscnn": True,
        "mscnn_multiscale": True,
        "mscnn_se_attention": True,
        "fusion_mode": "concat",
        "num_classifier_layers": 1,
        "dropout": 0.6,
        "use_neuro_prior": False,
    },
    "cls_2layer": {
        "label": "2-Layer Classifier",
        "use_spatial_expert": True,
        "use_mscnn": True,
        "mscnn_multiscale": True,
        "mscnn_se_attention": True,
        "fusion_mode": "concat",
        "num_classifier_layers": 2,
        "dropout": 0.6,
        "use_neuro_prior": False,
    },
    "cls_3layer": {
        "label": "3-Layer Classifier",
        "use_spatial_expert": True,
        "use_mscnn": True,
        "mscnn_multiscale": True,
        "mscnn_se_attention": True,
        "fusion_mode": "concat",
        "num_classifier_layers": 3,
        "dropout": 0.6,
        "use_neuro_prior": False,
    },
    "neuro_prior": {
        "label": "+ Neuro Prior",
        "use_spatial_expert": True,
        "use_mscnn": True,
        "mscnn_multiscale": True,
        "mscnn_se_attention": True,
        "fusion_mode": "concat",
        "num_classifier_layers": 2,
        "dropout": 0.6,
        "use_neuro_prior": True,
    },
}


def build_ablation_model(
    variant: str,
    num_classes: int,
    num_channels: int,
    checkpoint: str | None = None,
    device: torch.device | None = None,
    **kwargs: object,
) -> SpatialMSCNN_Ablation:
    """Build an ablation variant model, optionally loading pretrained weights.

    Parameters
    ----------
    variant : str
        One of ABLATION_VARIANTS keys.
    num_classes : int
    num_channels : int
    checkpoint : str | None
        Path to pretrained checkpoint.
    device : torch.device | None
    **kwargs
        Override for any SpatialMSCNN_Ablation parameter.

    Returns
    -------
    SpatialMSCNN_Ablation
    """

    if variant not in ABLATION_VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. Available: {list(ABLATION_VARIANTS)}"
        )

    ablation = dict(ABLATION_VARIANTS[variant])
    model_kwargs: dict[str, object] = {
        "num_classes": num_classes,
        "num_channels": num_channels,
        "ablation": ablation,
    }
    model_kwargs.update(
        {
            k: v
            for k, v in kwargs.items()
            if k not in ("ablation", "num_classes", "num_channels")
        }
    )

    model = SpatialMSCNN_Ablation(**model_kwargs)  # type: ignore[arg-type]
    if device is not None:
        model = model.to(device)

    if checkpoint:
        state_dict = torch.load(checkpoint, map_location=device or "cpu")
        model_state = model.state_dict()
        compatible = {
            k: v
            for k, v in state_dict.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        skipped = len(state_dict) - len(compatible)
        model.load_state_dict(compatible, strict=False)
        if skipped:
            print(
                f"  [ablation] Loaded {len(compatible)}/{len(state_dict)} weights, skipped {skipped}"
            )

    return model
