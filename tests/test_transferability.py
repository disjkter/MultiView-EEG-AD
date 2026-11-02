"""Tests for transferability metrics."""

from __future__ import annotations

import unittest

import numpy as np

from utils.transferability import (
    conditional_prototype_distance,
    leep_score,
    logme_score,
    rank_aggregate,
    transrate_score,
)


class TransferabilityMetricsTest(unittest.TestCase):
    """Validate metric directionality and finite outputs."""

    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        labels = np.repeat([0, 1], 20)
        informative = rng.normal(0.0, 0.3, size=(40, 8))
        informative[labels == 1, :4] += 2.0
        noise = rng.normal(size=(40, 8))
        self.labels = labels
        self.informative = informative
        self.noise = noise

    def test_logme_prefers_informative_features(self) -> None:
        self.assertGreater(
            logme_score(self.informative, self.labels),
            logme_score(self.noise, self.labels),
        )

    def test_transrate_is_finite(self) -> None:
        self.assertTrue(np.isfinite(transrate_score(self.informative, self.labels)))

    def test_leep_prefers_correct_probabilities(self) -> None:
        correct = np.column_stack(
            [np.where(self.labels == 0, 0.9, 0.1), np.where(self.labels == 1, 0.9, 0.1)]
        )
        uniform = np.full((len(self.labels), 2), 0.5)
        self.assertGreater(leep_score(correct, self.labels), leep_score(uniform, self.labels))

    def test_matching_prototypes_have_lower_distance(self) -> None:
        shifted = self.informative.copy()
        mismatched = self.informative[self.labels[::-1].argsort()]
        matching_distance = conditional_prototype_distance(
            self.informative, self.labels, shifted, self.labels
        )
        mismatched_distance = conditional_prototype_distance(
            self.informative, self.labels, mismatched, self.labels
        )
        self.assertLess(matching_distance, mismatched_distance)

    def test_rank_aggregate_prefers_consistent_row(self) -> None:
        rows = [
            {
                "logme": 2.0,
                "transrate": 2.0,
                "leep": -0.1,
                "gradient_cosine": 0.8,
                "graph_prototype_distance": 0.1,
                "temporal_prototype_distance": 0.1,
                "fused_prototype_distance": 0.1,
            },
            {
                "logme": 1.0,
                "transrate": 1.0,
                "leep": -0.8,
                "gradient_cosine": -0.2,
                "graph_prototype_distance": 0.8,
                "temporal_prototype_distance": 0.8,
                "fused_prototype_distance": 0.8,
            },
        ]
        scores = rank_aggregate(rows)
        self.assertGreater(scores[0], scores[1])

    def test_single_source_rank_is_one(self) -> None:
        row = {
            "logme": 1.0,
            "transrate": 1.0,
            "leep": -0.5,
            "gradient_cosine": 0.2,
            "graph_prototype_distance": 0.3,
            "temporal_prototype_distance": 0.3,
            "fused_prototype_distance": 0.3,
        }
        self.assertEqual(rank_aggregate([row]), [1.0])

    def test_near_constant_metric_does_not_control_rank(self) -> None:
        rows = [
            {
                "logme": -1.072360,
                "transrate": 1.0,
                "leep": -0.2,
                "gradient_cosine": 0.8,
                "graph_prototype_distance": 0.1,
                "temporal_prototype_distance": 0.1,
                "fused_prototype_distance": 0.1,
            },
            {
                "logme": -1.072370,
                "transrate": 0.0,
                "leep": -0.8,
                "gradient_cosine": -0.5,
                "graph_prototype_distance": 0.8,
                "temporal_prototype_distance": 0.8,
                "fused_prototype_distance": 0.8,
            },
        ]
        scores = rank_aggregate(rows)
        self.assertEqual(scores, [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
