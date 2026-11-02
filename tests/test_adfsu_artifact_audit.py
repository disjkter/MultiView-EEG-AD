"""Unit tests for participant-level ADFSU artifact-audit utilities."""

import unittest

import numpy as np

from scripts.audit_adfsu_artifacts import participant_features, stable_indices


class ADFSUArtifactAuditTests(unittest.TestCase):
    """Check deterministic sampling and participant-level feature extraction."""

    def test_stable_indices_preserves_endpoints(self) -> None:
        indices = np.arange(20)
        selected = stable_indices(indices, 5)
        self.assertEqual(selected.tolist(), [0, 4, 9, 14, 19])

    def test_participant_features_are_finite(self) -> None:
        rng = np.random.default_rng(9)
        data = rng.normal(size=(3, 19, 128)).astype(np.float32)
        qa, spectral, channel = participant_features(data, 128.0)
        self.assertEqual(len(channel), 38)
        self.assertIn("alpha_relative_power", spectral)
        self.assertTrue(all(np.isfinite(value) for value in qa.values()))
        self.assertTrue(all(np.isfinite(value) for value in spectral.values()))


if __name__ == "__main__":
    unittest.main()
