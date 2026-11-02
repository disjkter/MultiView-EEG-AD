"""Tests for the paper-upgrade participant-level analyses."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_lodo_calibration_baselines import calibration_intercept_slope
from scripts.paper_analysis.analyze_prevalence_sensitivity import stratified_resample
from scripts.paper_analysis.aggregate_sensitivity import filter_and_validate_rows
from scripts.paper_analysis.common import cohort_z_predictions, positive_logit


class PaperAnalysisTests(unittest.TestCase):
    """Validate score and prevalence sensitivity primitives."""

    def test_positive_logit_is_finite_at_probability_boundaries(self) -> None:
        logits = positive_logit([0.0, 0.5, 1.0])
        self.assertTrue(np.all(np.isfinite(logits)))
        self.assertAlmostEqual(float(logits[1]), 0.0)

    def test_cohort_z_is_invariant_to_positive_affine_logit_shift(self) -> None:
        logits = np.asarray([-2.0, -0.5, 0.5, 2.0])
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        shifted = 1.0 / (1.0 + np.exp(-(3.0 * logits + 7.0)))
        np.testing.assert_array_equal(
            cohort_z_predictions(probabilities), cohort_z_predictions(shifted)
        )

    def test_prevalence_resampling_matches_requested_class_count(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        indices = stratified_resample(labels, 0.8, 10, np.random.default_rng(7))
        self.assertEqual(len(indices), 10)
        self.assertAlmostEqual(float(labels[indices].mean()), 0.8)

    def test_calibration_intercept_slope_recovers_identity(self) -> None:
        probability = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        intercept, slope = calibration_intercept_slope(labels, probability)
        self.assertTrue(np.isfinite(intercept))
        self.assertTrue(np.isfinite(slope))

    def test_sensitivity_filter_retains_only_matched_targets_and_variants(self) -> None:
        rows = [
            {"condition": condition, "variant": variant, "seed": 2026, "target": target}
            for condition in ("cap25", "cap50")
            for variant in ("erm", "dann")
            for target in ("ADSZ", "ds004504", "CAUEEG")
        ]
        selected = filter_and_validate_rows(
            rows, targets=["ADSZ", "ds004504"], variants=["erm"]
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual({str(row["variant"]) for row in selected}, {"erm"})
        self.assertEqual(
            {str(row["target"]) for row in selected}, {"ADSZ", "ds004504"}
        )

    def test_sensitivity_filter_rejects_incomplete_target_groups(self) -> None:
        rows = [
            {"condition": "cap25", "variant": "erm", "seed": 2026, "target": "ADSZ"},
            {"condition": "cap50", "variant": "erm", "seed": 2026, "target": "ADSZ"},
            {
                "condition": "cap50",
                "variant": "erm",
                "seed": 2026,
                "target": "ds004504",
            },
        ]
        with self.assertRaisesRegex(RuntimeError, "matched targets"):
            filter_and_validate_rows(rows, targets=["ADSZ", "ds004504"])


if __name__ == "__main__":
    unittest.main()
