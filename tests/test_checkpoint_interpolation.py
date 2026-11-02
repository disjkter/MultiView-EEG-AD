"""Tests for encoder checkpoint interpolation."""

import unittest

import torch

from scripts.interpolate_encoder_checkpoints import interpolate_encoder_states


class CheckpointInterpolationTests(unittest.TestCase):
    def test_interpolates_encoder_and_excludes_classifier(self) -> None:
        ssl = {
            "spatial_expert.weight": torch.zeros(2),
            "fusion_classifier.weight": torch.zeros(2),
        }
        source = {
            "spatial_expert.weight": torch.full((2,), 4.0),
            "fusion_classifier.weight": torch.full((2,), 9.0),
        }
        result = interpolate_encoder_states(ssl, source, 0.25)
        self.assertEqual(set(result), {"spatial_expert.weight"})
        self.assertTrue(torch.equal(result["spatial_expert.weight"], torch.ones(2)))

    def test_rejects_invalid_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            interpolate_encoder_states({}, {}, 1.5)


if __name__ == "__main__":
    unittest.main()

