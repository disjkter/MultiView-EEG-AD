"""Tests for validation-guided source selection."""

from __future__ import annotations

import unittest

from scripts.evaluate_validation_guided_source_selection import (
    choose_source_by_validation_loss,
)


class ValidationGuidedSourceSelectionTest(unittest.TestCase):
    """Validate that test metrics cannot affect source selection."""

    def test_lowest_validation_loss_is_selected(self) -> None:
        name, _ = choose_source_by_validation_loss(
            {
                "source_a": {"best_val_loss": 0.8, "metrics": {"auc": 1.0}},
                "source_b": {"best_val_loss": 0.5, "metrics": {"auc": 0.0}},
            }
        )
        self.assertEqual(name, "source_b")


if __name__ == "__main__":
    unittest.main()
