"""Tests for overlap-controlled ADFSU paired inference."""

from __future__ import annotations

import unittest

from scripts.analyze_overlap_controlled_adfsu import exact_sign_flip, holm_adjust


class OverlapControlledAdfsuTests(unittest.TestCase):
    """Check exact inference helpers."""

    def test_exact_sign_flip_for_consistent_differences(self) -> None:
        self.assertAlmostEqual(exact_sign_flip([1.0] * 10), 2.0 / 1024.0)

    def test_holm_adjustment_is_monotone(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.50})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.04)
        self.assertAlmostEqual(adjusted["c"], 0.50)


if __name__ == "__main__":
    unittest.main()
