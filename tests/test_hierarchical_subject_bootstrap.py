"""Tests for hierarchical participant-level bootstrap intervals."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.analyze_hierarchical_subject_bootstrap import (
    hierarchical_bootstrap,
    stratified_indices,
)


class HierarchicalSubjectBootstrapTests(unittest.TestCase):
    """Validate class preservation and deterministic interval generation."""

    def test_stratified_resampling_preserves_class_counts(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1])
        indices = stratified_indices(labels, np.random.default_rng(4))
        sampled = labels[indices]
        self.assertEqual(int((sampled == 0).sum()), 3)
        self.assertEqual(int((sampled == 1).sum()), 2)

    def test_hierarchical_bootstrap_is_finite(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        runs = {
            1: [{"target": "A", "labels": labels, "probability": np.asarray([0.1, 0.4, 0.6, 0.9]), "threshold": 0.5}],
            2: [{"target": "A", "labels": labels, "probability": np.asarray([0.2, 0.3, 0.7, 0.8]), "threshold": 0.5}],
        }
        point, samples = hierarchical_bootstrap(runs, 100, np.random.default_rng(8))
        self.assertTrue(all(np.isfinite(value) for value in point.values()))
        self.assertTrue(all(np.all(np.isfinite(value)) for value in samples.values()))


if __name__ == "__main__":
    unittest.main()
