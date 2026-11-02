"""Shape and optimization checks for the TF-C external baseline."""

from __future__ import annotations

import unittest

import torch

from models.TFCBaseline import TFCClassifier, TFCEncoder
from scripts.pretrain_tfc_baseline import nt_xent


class TFCBaselineTest(unittest.TestCase):
    """Verify multichannel TF-C outputs and the contrastive loss."""

    def test_encoder_and_classifier_shapes(self) -> None:
        """The model should accept [B, 19, 128] and emit fixed embeddings."""

        encoder = TFCEncoder(model_dim=32, projection_dim=16, num_layers=1, num_heads=4)
        encoder.eval()
        x = torch.randn(4, 19, 128)
        with torch.no_grad():
            h_t, z_t, h_f, z_f = encoder(x)
        self.assertEqual(h_t.shape, (4, 32))
        self.assertEqual(h_f.shape, (4, 32))
        self.assertEqual(z_t.shape, (4, 16))
        self.assertEqual(z_f.shape, (4, 16))

        classifier = TFCClassifier(
            model_dim=32, projection_dim=16, num_layers=1, num_heads=4
        )
        classifier.eval()
        with torch.no_grad():
            logits, features = classifier(x)
        self.assertEqual(logits.shape, (4, 2))
        self.assertEqual(features.shape, (4, 32))

    def test_contrastive_loss_is_finite_and_differentiable(self) -> None:
        """The NT-Xent implementation should produce usable gradients."""

        first = torch.randn(8, 16, requires_grad=True)
        second = first.detach() + 0.05 * torch.randn(8, 16)
        loss = nt_xent(first, second, temperature=0.2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(first.grad)
        self.assertTrue(torch.isfinite(first.grad).all())


if __name__ == "__main__":
    unittest.main()
