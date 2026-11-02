"""Tests for training-stage EEG feature fusion."""

import unittest
from pathlib import Path

import torch
import torch.nn as nn

from models.TrainingStageFeatureFusion import TrainingStageFeatureFusion, load_encoder_checkpoint


class DummyEncoder(nn.Module):
    """Small encoder with BatchNorm for checking freeze behavior."""

    def __init__(self, input_dim: int = 5, feature_dim: int = 4) -> None:
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(input_dim, feature_dim), nn.BatchNorm1d(feature_dim))
        self.head = nn.Linear(feature_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and features."""

        features = self.feature(x)
        return self.head(features), features


class TrainingStageFeatureFusionTests(unittest.TestCase):
    """Verify fusion shapes, gates, freezing, and checkpoint loading."""

    def make_model(self, mode: str, frozen: bool = True) -> TrainingStageFeatureFusion:
        """Build a small fusion model."""

        return TrainingStageFeatureFusion(
            DummyEncoder(), DummyEncoder(), feature_dim=4, num_classes=2,
            mode=mode, gate_hidden_dim=3, classifier_dropout=0.0,
            freeze_encoders=frozen,
        )

    def test_all_modes_return_expected_shapes(self) -> None:
        """Every ablation mode produces valid logits and scalar gates."""

        x = torch.randn(6, 5)
        for mode in TrainingStageFeatureFusion.MODES:
            model = self.make_model(mode)
            logits, features, alpha = model.forward_with_gate(x)
            self.assertEqual(tuple(logits.shape), (6, 2))
            expected_dim = 8 if mode == "concat" else 4
            self.assertEqual(tuple(features.shape), (6, expected_dim))
            self.assertEqual(tuple(alpha.shape), (6, 1))

    def test_gate_values_and_fixed_baselines(self) -> None:
        """Scalar gates use the documented range and fixed weights."""

        x = torch.randn(5, 5)
        learned = self.make_model("learned_scalar")
        alpha = learned.forward_with_gate(x)[2]
        self.assertTrue(torch.all((alpha >= 0.0) & (alpha <= 1.0)))
        self.assertTrue(torch.allclose(self.make_model("pre_only").forward_with_gate(x)[2], torch.ones(5, 1)))
        self.assertTrue(torch.allclose(self.make_model("source_only").forward_with_gate(x)[2], torch.zeros(5, 1)))
        self.assertTrue(torch.allclose(self.make_model("fixed_average").forward_with_gate(x)[2], torch.full((5, 1), 0.5)))

    def test_frozen_encoders_remain_in_eval_mode(self) -> None:
        """Training the fusion head does not update encoder BatchNorm state."""

        model = self.make_model("learned_scalar", frozen=True)
        model.train()
        self.assertFalse(model.pretrained_encoder.training)
        self.assertFalse(model.source_encoder.training)
        self.assertTrue(model.classifier.training)
        self.assertFalse(any(parameter.requires_grad for parameter in model.pretrained_encoder.parameters()))

    def test_checkpoint_loader_filters_to_encoder_prefix(self) -> None:
        """Checkpoint loading ignores heads and loads compatible encoder tensors."""

        source = DummyEncoder()
        target = DummyEncoder()
        payload = {"encoder_state_dict": source.state_dict()}
        Path("outputs").mkdir(exist_ok=True)
        path = Path("outputs/_test_stage_feature_fusion_checkpoint.pth")
        try:
            torch.save(payload, path)
            count = load_encoder_checkpoint(target, path, ("feature.",))
        finally:
            path.unlink(missing_ok=True)
        self.assertGreater(count, 0)
        self.assertTrue(torch.equal(source.feature[0].weight, target.feature[0].weight))


if __name__ == "__main__":
    unittest.main()
