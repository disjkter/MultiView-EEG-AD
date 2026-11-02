"""Tests for train-subject filtering in source-target similarity analysis."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.analyze_source_target_similarity import choose_indices, exponential_similarity


class SourceTargetSimilaritySplitTest(unittest.TestCase):
    """Ensure target statistics cannot sample held-out subjects."""

    def test_choose_indices_respects_eligible_indices(self) -> None:
        subject_ids = np.asarray(["train-a"] * 5 + ["test-b"] * 5, dtype=object)
        eligible = np.arange(5, dtype=np.int64)

        selected = choose_indices(
            n_segments=10,
            sub_ids=subject_ids,
            max_segments=10,
            max_segments_per_subject=3,
            seed=2026,
            eligible_indices=eligible,
        )

        self.assertEqual(len(selected), 3)
        self.assertTrue(np.all(selected < 5))

    def test_exponential_similarity_is_positive_and_ordered(self) -> None:
        """Distance weighting must not hard-zero the least similar source."""

        similarities = exponential_similarity([0.1, 0.2, 0.4], temperature=1.0)
        self.assertEqual(similarities[0], 1.0)
        self.assertGreater(similarities[0], similarities[1])
        self.assertGreater(similarities[1], similarities[2])
        self.assertGreater(similarities[2], 0.0)


if __name__ == "__main__":
    unittest.main()
