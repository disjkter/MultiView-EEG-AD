"""Tests for calibration-aware source-only domain generalization losses."""

from __future__ import annotations

import unittest

import torch

from models.CalibrationAwareDG import (
    aggregate_participant_probabilities,
    calibration_aware_dg_loss,
    class_conditional_quantile_alignment,
    domain_calibration_risk,
    error_aware_entropy_regularization,
    participant_classification_loss,
)


class CalibrationAwareDGTests(unittest.TestCase):
    """Validate invariances, directionality, and differentiability."""

    def test_quantile_alignment_is_zero_for_matching_conditional_scores(self) -> None:
        margin = torch.tensor([-2.0, -1.0, 1.0, 2.0] * 2)
        logits = torch.stack([-0.5 * margin, 0.5 * margin], dim=1)
        labels = torch.tensor([0, 0, 1, 1] * 2)
        domains = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        loss = class_conditional_quantile_alignment(logits, labels, domains)

        self.assertAlmostEqual(float(loss), 0.0, places=7)

    def test_quantile_alignment_detects_class_conditional_score_shift(self) -> None:
        margin = torch.tensor(
            [-2.0, -1.0, 1.0, 2.0, 0.0, 1.0, 3.0, 4.0],
            requires_grad=True,
        )
        logits = torch.stack([-0.5 * margin, 0.5 * margin], dim=1)
        labels = torch.tensor([0, 0, 1, 1] * 2)
        domains = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        loss = class_conditional_quantile_alignment(logits, labels, domains)
        loss.backward()

        self.assertGreater(float(loss), 0.0)
        self.assertIsNotNone(margin.grad)

    def test_calibration_risk_prefers_correct_confident_predictions(self) -> None:
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        domains = torch.tensor([0, 0, 0, 1, 1, 1])
        good = torch.tensor([[4.0, -4.0], [-4.0, 4.0]] * 3)
        bad = -good

        good_loss = domain_calibration_risk(good, labels, domains)
        bad_loss = domain_calibration_risk(bad, labels, domains)

        self.assertLess(float(good_loss), float(bad_loss))

    def test_error_aware_entropy_targets_confident_errors(self) -> None:
        labels = torch.tensor([0, 1])
        correct = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])
        wrong = -correct

        self.assertLess(
            float(error_aware_entropy_regularization(correct, labels)),
            float(error_aware_entropy_regularization(wrong, labels)),
        )

    def test_combined_loss_backpropagates(self) -> None:
        logits = torch.randn(12, 2, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 0, 1] * 2)
        domains = torch.tensor([0] * 6 + [1] * 6)

        losses = calibration_aware_dg_loss(logits, labels, domains)
        total = sum(losses.values())
        total.backward()

        self.assertEqual(set(losses), {"score_alignment", "calibration", "error_entropy"})
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_participant_aggregation_is_differentiable(self) -> None:
        logits = torch.tensor(
            [[2.0, -2.0], [1.0, -1.0], [-2.0, 2.0], [-1.0, 1.0]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 0, 1, 1])
        domains = torch.tensor([0, 0, 1, 1])
        subjects = ["A", "A", "B", "B"]

        probability, participant_y, participant_domain = (
            aggregate_participant_probabilities(
                logits, labels, domains, subjects
            )
        )
        loss = participant_classification_loss(probability, participant_y)
        loss.backward()

        self.assertEqual(tuple(probability.shape), (2, 2))
        self.assertEqual(participant_y.tolist(), [0, 1])
        self.assertEqual(participant_domain.tolist(), [0, 1])
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.all(logits.grad.abs().sum(dim=1) > 0))


if __name__ == "__main__":
    unittest.main()
