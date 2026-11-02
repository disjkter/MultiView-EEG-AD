"""Tests for margin-alignment result aggregation."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.aggregate_margin_alignment import (
    add_holm_adjustment,
    cohort_standardized_predictions,
    exact_sign_flip,
)


class CohortScoreStandardizationTests(unittest.TestCase):
    """Validate the unlabeled cohort score transformation."""

    def test_predictions_are_invariant_to_logit_shift_and_positive_scale(self) -> None:
        logits = np.asarray([-2.0, -1.0, 0.5, 2.5])
        shifted_scaled_logits = 3.0 * logits + 5.0
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        shifted_scaled_probabilities = 1.0 / (
            1.0 + np.exp(-shifted_scaled_logits)
        )

        original = cohort_standardized_predictions(probabilities.tolist())
        transformed = cohort_standardized_predictions(
            shifted_scaled_probabilities.tolist()
        )

        np.testing.assert_array_equal(original, transformed)

    def test_exact_sign_flip_has_expected_resolution(self) -> None:
        self.assertEqual(exact_sign_flip([1.0, 1.0, 1.0]), 0.25)
        self.assertEqual(exact_sign_flip([1.0] * 10), 2.0 / 1024.0)

    def test_holm_adjustment_is_monotonic_within_metric(self) -> None:
        rows = [
            {"metric": "auc", "exact_sign_flip_pvalue": 0.01},
            {"metric": "auc", "exact_sign_flip_pvalue": 0.03},
            {"metric": "auc", "exact_sign_flip_pvalue": 0.20},
            {"metric": "f1", "exact_sign_flip_pvalue": 0.04},
        ]

        adjusted = add_holm_adjustment(rows)

        self.assertAlmostEqual(adjusted[0]["holm_adjusted_pvalue"], 0.03)
        self.assertAlmostEqual(adjusted[1]["holm_adjusted_pvalue"], 0.06)
        self.assertAlmostEqual(adjusted[2]["holm_adjusted_pvalue"], 0.20)
        self.assertAlmostEqual(adjusted[3]["holm_adjusted_pvalue"], 0.04)


if __name__ == "__main__":
    unittest.main()
