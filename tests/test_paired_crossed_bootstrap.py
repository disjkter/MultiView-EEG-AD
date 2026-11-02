"""Tests for paired crossed-bootstrap metric helpers."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.analyze_paired_crossed_bootstrap import (
    adaptive_ece,
    compute_metrics,
    fixed_ece,
    participant_keys,
)


class PairedCrossedBootstrapTests(unittest.TestCase):
    """Validate calibration and threshold endpoint calculations."""

    def test_perfect_probabilities_have_near_zero_calibration_error(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        probability = np.asarray([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(fixed_ece(labels, probability), 0.0)
        self.assertAlmostEqual(adaptive_ece(labels, probability), 0.0)

    def test_balanced_half_probabilities_have_chance_auc(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        probability = np.full(4, 0.5)
        metrics = compute_metrics(labels, probability, 0.5)
        self.assertAlmostEqual(metrics["auc"], 0.5)
        self.assertAlmostEqual(metrics["brier"], 0.25)
        self.assertAlmostEqual(metrics["nll"], np.log(2.0))

    def test_old_summary_positions_are_stable_anonymous_participant_keys(self) -> None:
        subjects, source = participant_keys({}, "target", 2)
        self.assertEqual(subjects, ("target::position_0000", "target::position_0001"))
        self.assertEqual(source, "deterministic_sorted_position")


if __name__ == "__main__":
    unittest.main()
