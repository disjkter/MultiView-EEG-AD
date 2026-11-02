"""Tests for TAGE-Net optional neurophysiological prior branch."""

from __future__ import annotations

import unittest

import torch

from models.Spatial_MSCNN import SpatialMSCNN_DualStream


class TAGEWithPriorTest(unittest.TestCase):
    """Validate the dual-stream backbone with and without prior features."""

    def test_default_model_keeps_original_feature_dimension(self) -> None:
        """Default construction should preserve the existing 512-d feature path."""

        model = SpatialMSCNN_DualStream(num_classes=2, num_channels=19)
        model.eval()
        x = torch.randn(2, 19, 256)

        with torch.no_grad():
            logits, features = model(x)

        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(features.shape, (2, 512))

    def test_neuro_prior_adds_projected_prior_embedding(self) -> None:
        """The optional prior branch should append a compact 64-d embedding."""

        model = SpatialMSCNN_DualStream(
            num_classes=2,
            num_channels=19,
            use_neuro_prior=True,
            sampling_rate=128.0,
            prior_out_dim=64,
        )
        model.eval()
        x = torch.randn(2, 19, 256)

        with torch.no_grad():
            logits, features = model(x)

        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(features.shape, (2, 576))

    def test_one_layer_classifier_preserves_encoder_features(self) -> None:
        """A one-layer head should change only the classifier."""

        model = SpatialMSCNN_DualStream(
            num_classes=2,
            num_channels=19,
            classifier_num_layers=1,
        )
        model.eval()
        x = torch.randn(2, 19, 128)

        with torch.no_grad():
            logits, features = model(x)

        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(features.shape, (2, 512))


if __name__ == "__main__":
    unittest.main()
