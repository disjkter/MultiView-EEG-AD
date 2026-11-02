"""Tests for cross-cohort calibration analysis."""

from __future__ import annotations

import unittest

from scripts.analyze_cross_cohort_calibration import (
    calibration_metrics,
    parse_result_spec,
    reliability_bins,
)


class CalibrationMetricTests(unittest.TestCase):
    """Validate probabilistic metrics and bin assignment."""

    def test_perfect_predictions_have_near_zero_error(self) -> None:
        metrics = calibration_metrics([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(metrics["brier"], 0.0)
        self.assertLess(metrics["nll"], 1e-5)
        self.assertAlmostEqual(metrics["ece"], 0.0)

    def test_constant_prevalence_prediction_is_calibrated(self) -> None:
        metrics = calibration_metrics([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(metrics["brier"], 0.25)
        self.assertAlmostEqual(metrics["nll"], 0.6931471805599453)
        self.assertAlmostEqual(metrics["ece"], 0.0)

    def test_probability_one_is_assigned_to_last_bin(self) -> None:
        bins = reliability_bins([0, 1], [0.0, 1.0], num_bins=10)
        self.assertEqual([row["bin"] for row in bins], [0, 9])

    def test_result_spec_preserves_windows_drive_separator(self) -> None:
        name, path = parse_result_spec("development=F:/results/run")
        self.assertEqual(name, "development")
        self.assertEqual(str(path).replace("\\", "/"), "F:/results/run")


if __name__ == "__main__":
    unittest.main()
