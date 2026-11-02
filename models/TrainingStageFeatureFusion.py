"""Training-stage fusion of two compatible EEG feature encoders."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


FusionMode = Literal["pre_only", "source_only", "fixed_average", "concat", "learned_scalar"]


def _checkpoint_state(payload: object) -> dict[str, torch.Tensor]:
    """Extract a model state dictionary from a supported checkpoint payload."""

    if isinstance(payload, dict) and "encoder_state_dict" in payload:
        payload = payload["encoder_state_dict"]
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a state dictionary.")
    return {str(key): value for key, value in payload.items() if isinstance(value, torch.Tensor)}


def load_encoder_checkpoint(
    encoder: nn.Module,
    checkpoint: str | Path,
    allowed_prefixes: tuple[str, ...],
    map_location: torch.device | str = "cpu",
) -> int:
    """Load shape-compatible feature-extractor tensors and return their count."""

    payload = torch.load(checkpoint, map_location=map_location)
    checkpoint_state = _checkpoint_state(payload)
    model_state = encoder.state_dict()
    compatible = {
        key: value
        for key, value in checkpoint_state.items()
        if key.startswith(allowed_prefixes)
        and key in model_state
        and model_state[key].shape == value.shape
    }
    if not compatible:
        raise ValueError(f"No compatible encoder tensors found in {checkpoint}.")
    encoder.load_state_dict(compatible, strict=False)
    return len(compatible)


class TrainingStageFeatureFusion(nn.Module):
    """Fuse normalized representations from pretraining and source-task encoders.

    Both encoders must return ``(logits, features)`` and the two feature tensors
    must have shape ``[B, feature_dim]``. The learned mode predicts one scalar
    gate per EEG segment, which keeps the added parameter count small.
    """

    MODES = ("pre_only", "source_only", "fixed_average", "concat", "learned_scalar")

    def __init__(
        self,
        pretrained_encoder: nn.Module,
        source_encoder: nn.Module,
        feature_dim: int,
        num_classes: int,
        mode: FusionMode = "learned_scalar",
        gate_hidden_dim: int = 64,
        classifier_dropout: float = 0.2,
        freeze_encoders: bool = True,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown fusion mode: {mode}")
        if feature_dim <= 0 or gate_hidden_dim <= 0:
            raise ValueError("feature_dim and gate_hidden_dim must be positive.")

        self.pretrained_encoder = pretrained_encoder
        self.source_encoder = source_encoder
        self.feature_dim = int(feature_dim)
        self.mode = mode
        self.freeze_encoders = bool(freeze_encoders)
        self.pre_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        self.source_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)

        self.gate: nn.Module | None = None
        if mode == "learned_scalar":
            self.gate = nn.Sequential(
                nn.Linear(2 * feature_dim, gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(gate_hidden_dim, 1),
                nn.Sigmoid(),
            )
        classifier_dim = 2 * feature_dim if mode == "concat" else feature_dim
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_dim, num_classes),
        )
        self.set_encoders_frozen(self.freeze_encoders)

    def set_encoders_frozen(self, frozen: bool) -> None:
        """Enable or disable gradient updates for both feature encoders."""

        self.freeze_encoders = bool(frozen)
        for encoder in (self.pretrained_encoder, self.source_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad = not self.freeze_encoders
            if self.freeze_encoders:
                encoder.eval()

    def train(self, mode: bool = True) -> "TrainingStageFeatureFusion":
        """Keep frozen encoders in evaluation mode so BatchNorm state is fixed."""

        super().train(mode)
        if self.freeze_encoders:
            self.pretrained_encoder.eval()
            self.source_encoder.eval()
        return self

    @staticmethod
    def _features(encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Extract the feature tensor returned by an EEG encoder."""

        output = encoder(x)
        if not isinstance(output, tuple) or len(output) < 2:
            raise TypeError("Each encoder must return (logits, features).")
        return output[1]

    def forward_with_gate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits, fused features, and pretraining weights ``alpha``."""

        if self.mode == "pre_only":
            pre = self.pre_norm(self._features(self.pretrained_encoder, x))
            if pre.shape[-1] != self.feature_dim:
                raise ValueError(f"Expected [B, {self.feature_dim}] features, got {tuple(pre.shape)}.")
            alpha = pre.new_ones((pre.shape[0], 1))
            return self.classifier(pre), pre, alpha
        if self.mode == "source_only":
            source = self.source_norm(self._features(self.source_encoder, x))
            if source.shape[-1] != self.feature_dim:
                raise ValueError(f"Expected [B, {self.feature_dim}] features, got {tuple(source.shape)}.")
            alpha = source.new_zeros((source.shape[0], 1))
            return self.classifier(source), source, alpha

        pre = self.pre_norm(self._features(self.pretrained_encoder, x))
        source = self.source_norm(self._features(self.source_encoder, x))
        if pre.shape != source.shape or pre.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected matching [B, {self.feature_dim}] features, got "
                f"{tuple(pre.shape)} and {tuple(source.shape)}."
            )

        if self.mode == "fixed_average":
            alpha = pre.new_full((pre.shape[0], 1), 0.5)
            fused = 0.5 * (pre + source)
        elif self.mode == "concat":
            alpha = pre.new_full((pre.shape[0], 1), float("nan"))
            fused = torch.cat([pre, source], dim=-1)
        else:
            assert self.gate is not None
            alpha = self.gate(torch.cat([pre, source], dim=-1))
            fused = alpha * pre + (1.0 - alpha) * source
        return self.classifier(fused), fused, alpha

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return class logits and fused features."""

        logits, fused, _ = self.forward_with_gate(x)
        return logits, fused
