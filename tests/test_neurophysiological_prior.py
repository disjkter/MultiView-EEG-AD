"""Tests for neurophysiological prior feature extraction."""

from __future__ import annotations

import unittest

import torch

from models.neurophysiological_prior import NeurophysiologicalPriorLayer


class NeurophysiologicalPriorLayerTest(unittest.TestCase):
    """Validate shapes and numerical properties of EEG prior features."""

    def test_prior_outputs_have_expected_shapes(self) -> None:
        """The layer should expose region, band, and connectivity features."""

        layer = NeurophysiologicalPriorLayer(sampling_rate=128.0)
        x = torch.randn(4, 19, 256)

        output = layer(x)

        self.assertEqual(output.region_power.shape, (4, 5))
        self.assertEqual(output.band_power.shape, (4, 19, 5))
        self.assertEqual(output.connectivity.shape, (4, 19, 19))
        self.assertEqual(output.region_names, ("frontal", "temporal", "central", "parietal", "occipital"))
        self.assertEqual(output.band_names, ("delta", "theta", "alpha", "beta", "gamma"))

    def test_connectivity_is_symmetric_with_unit_diagonal(self) -> None:
        """Pearson connectivity should be symmetric and have diagonal ones."""

        layer = NeurophysiologicalPriorLayer(sampling_rate=128.0)
        x = torch.randn(2, 19, 128)

        connectivity = layer(x).connectivity
        diagonal = torch.diagonal(connectivity, dim1=1, dim2=2)

        self.assertTrue(torch.allclose(connectivity, connectivity.transpose(1, 2), atol=1e-5))
        self.assertTrue(torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-5))

    def test_constant_input_is_numerically_stable(self) -> None:
        """Constant EEG segments should not produce NaN or Inf values."""

        layer = NeurophysiologicalPriorLayer(sampling_rate=128.0)
        x = torch.ones(3, 19, 128)

        output = layer(x)

        self.assertFalse(torch.isnan(output.region_power).any())
        self.assertFalse(torch.isnan(output.band_power).any())
        self.assertFalse(torch.isnan(output.connectivity).any())
        self.assertFalse(torch.isinf(output.region_power).any())
        self.assertFalse(torch.isinf(output.band_power).any())
        self.assertFalse(torch.isinf(output.connectivity).any())


if __name__ == "__main__":
    unittest.main()
