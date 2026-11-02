"""Smoke tests for the iTransformer baseline."""

from __future__ import annotations

import unittest

import torch

from models.iTransformer import ITransformer


class TestITransformer(unittest.TestCase):
    """Basic forward-pass checks."""

    def test_forward_shapes_for_8s_windows(self) -> None:
        model = ITransformer(num_channels=19, num_classes=2)
        x = torch.randn(4, 19, 1024)
        logits, features = model(x)
        self.assertEqual(logits.shape, (4, 2))
        self.assertEqual(features.shape, (4, 64))

    def test_forward_shapes_for_5s_windows(self) -> None:
        model = ITransformer(num_channels=19, num_classes=2)
        x = torch.randn(3, 19, 640)
        logits, features = model(x)
        self.assertEqual(logits.shape, (3, 2))
        self.assertEqual(features.shape, (3, 64))


if __name__ == "__main__":
    unittest.main()
