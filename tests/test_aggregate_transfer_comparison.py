"""Tests for paired transfer-comparison statistics."""

import unittest

import numpy as np

from scripts.aggregate_transfer_comparison import (
    bootstrap_ci,
    exact_sign_flip_pvalue,
)


class AggregateTransferComparisonTests(unittest.TestCase):
    def test_identical_pairs_have_pvalue_one(self) -> None:
        self.assertEqual(exact_sign_flip_pvalue(np.zeros(5)), 1.0)

    def test_consistent_positive_differences_are_significant(self) -> None:
        differences = np.ones(10)
        self.assertLess(exact_sign_flip_pvalue(differences), 0.01)

    def test_bootstrap_interval_is_deterministic(self) -> None:
        differences = np.asarray([0.1, 0.2, 0.3])
        first = bootstrap_ci(differences, samples=1000, seed=7)
        second = bootstrap_ci(differences, samples=1000, seed=7)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
