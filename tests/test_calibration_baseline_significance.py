"""Tests for paired source-calibration significance analysis."""

from __future__ import annotations

import unittest

from scripts.analyze_calibration_baseline_significance import paired_tests


class CalibrationBaselineSignificanceTests(unittest.TestCase):
    """Validate comparison direction and metric-family correction."""

    def test_lower_error_is_counted_as_improvement(self) -> None:
        rows = []
        for seed in range(3):
            for calibrator, offset in (("identity", 0.0), ("temperature", -0.1), ("platt", 0.1)):
                rows.append(
                    {
                        "variant": "erm",
                        "calibrator": calibrator,
                        "seed": str(seed),
                        "ece": str(0.3 + offset),
                        "brier": str(0.3 + offset),
                        "nll": str(0.8 + offset),
                        "balanced_accuracy": str(0.6 - offset),
                    }
                )
        comparisons = paired_tests(rows)
        temperature_ece = next(
            row
            for row in comparisons
            if row["calibrator"] == "temperature" and row["metric"] == "ece"
        )
        self.assertEqual(temperature_ece["improved_seeds"], 3)
        self.assertLess(float(temperature_ece["mean_difference"]), 0.0)


if __name__ == "__main__":
    unittest.main()
