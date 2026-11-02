"""Tests for support-curve aggregation."""

from __future__ import annotations

import unittest

import pandas as pd

from scripts.aggregate_transfer_support_curve import compare_paired_runs
from scripts.aggregate_transfer_comparison import METRICS


class AggregateTransferSupportCurveTest(unittest.TestCase):
    """Validate paired gain direction and grouping."""

    def test_transfer_gain_is_source_minus_target(self) -> None:
        rows = []
        for seed in (1, 2):
            for initialization, value in (
                ("target_only", 0.5),
                ("source_transfer", 0.7),
            ):
                rows.append(
                    {
                        "support_per_class": 1,
                        "seed": seed,
                        "initialization": initialization,
                        **{metric: value for metric in METRICS},
                    }
                )

        comparisons = compare_paired_runs(
            pd.DataFrame(rows), bootstrap_samples=100, bootstrap_seed=7
        )

        self.assertTrue((comparisons["transfer_gain"] > 0).all())
        self.assertTrue((comparisons["positive_seeds"] == 2).all())


if __name__ == "__main__":
    unittest.main()
