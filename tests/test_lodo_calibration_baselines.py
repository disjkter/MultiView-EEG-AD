"""Tests for source-only post-hoc calibration baselines."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import log_loss

from scripts.evaluate_lodo_calibration_baselines import (
    apply_calibrator,
    expected_calibration_error,
    fit_platt,
    fit_temperature,
)


class LODOCalibrationBaselineTests(unittest.TestCase):
    """Validate calibration fitting without target-label access."""

    def setUp(self) -> None:
        self.y_true = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        self.overconfident = np.asarray(
            [0.01, 0.03, 0.10, 0.80, 0.20, 0.90, 0.97, 0.99], dtype=np.float64
        )

    def test_temperature_is_positive_and_does_not_worsen_fit(self) -> None:
        temperature = fit_temperature(self.y_true, self.overconfident)
        calibrated = apply_calibrator(
            self.overconfident, "temperature", (temperature,)
        )
        self.assertGreater(temperature, 0.0)
        self.assertLessEqual(
            log_loss(self.y_true, calibrated),
            log_loss(self.y_true, self.overconfident) + 1e-8,
        )

    def test_platt_returns_finite_probabilities(self) -> None:
        parameters = fit_platt(self.y_true, self.overconfident)
        calibrated = apply_calibrator(self.overconfident, "platt", parameters)
        self.assertGreater(parameters[0], 0.0)
        self.assertTrue(np.all(np.isfinite(calibrated)))
        self.assertTrue(np.all((calibrated > 0.0) & (calibrated < 1.0)))
        order = np.argsort(self.overconfident)
        self.assertTrue(np.all(np.diff(calibrated[order]) >= 0.0))

    def test_ece_is_zero_for_balanced_half_probabilities(self) -> None:
        y_true = np.asarray([0, 1, 0, 1])
        probability = np.full(4, 0.5)
        self.assertAlmostEqual(
            expected_calibration_error(y_true, probability), 0.0, places=12
        )


if __name__ == "__main__":
    unittest.main()
