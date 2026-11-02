"""Tests for source-domain-robust calibration and threshold selection."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from scripts.evaluate_domain_robust_calibration import (
    choose_minimax_threshold,
    fit_robust_platt,
    fit_robust_temperature,
)
from scripts.evaluate_lodo_calibration_baselines import apply_calibrator


class DomainRobustCalibrationTests(unittest.TestCase):
    """Check constraints and source-domain minimax behavior."""

    def setUp(self) -> None:
        self.labels = [
            np.asarray([0, 0, 1, 1]),
            np.asarray([0, 0, 1, 1]),
        ]
        self.probabilities = [
            np.asarray([0.05, 0.20, 0.70, 0.90]),
            np.asarray([0.25, 0.45, 0.55, 0.75]),
        ]

    def test_robust_calibrators_are_monotonic_and_finite(self) -> None:
        temperature = fit_robust_temperature(self.labels, self.probabilities)
        slope, intercept = fit_robust_platt(self.labels, self.probabilities)
        self.assertGreater(temperature, 0.0)
        self.assertGreater(slope, 0.0)
        self.assertTrue(np.isfinite(intercept))

    def test_minimax_threshold_reports_its_domain_scores(self) -> None:
        threshold, worst_score, mean_score = choose_minimax_threshold(
            self.labels, self.probabilities
        )
        scores = [
            balanced_accuracy_score(y, p >= threshold)
            for y, p in zip(self.labels, self.probabilities)
        ]
        self.assertAlmostEqual(worst_score, min(scores))
        self.assertAlmostEqual(mean_score, float(np.mean(scores)))
        self.assertGreaterEqual(worst_score, 0.5)

    def test_minimax_decisions_are_invariant_to_monotonic_mapping(self) -> None:
        identity_threshold, _, _ = choose_minimax_threshold(
            self.labels, self.probabilities
        )
        parameters = (1.7, -0.4)
        transformed = [
            apply_calibrator(probability, "platt", parameters)
            for probability in self.probabilities
        ]
        transformed_threshold, _, _ = choose_minimax_threshold(self.labels, transformed)
        for original, mapped in zip(self.probabilities, transformed):
            np.testing.assert_array_equal(
                original >= identity_threshold,
                mapped >= transformed_threshold,
            )


if __name__ == "__main__":
    unittest.main()
