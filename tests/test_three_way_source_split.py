"""Tests for disjoint source-only checkpoint and calibration splits."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.run_graph_temporal_lodo_tacd import DomainStore, split_source_indices_three_way


class ThreeWaySourceSplitTests(unittest.TestCase):
    """Verify participant separation and class retention."""

    def test_three_splits_are_subject_disjoint_and_stratified(self) -> None:
        subjects = np.asarray(
            [f"class{label}_subject{subject}" for label in (0, 1) for subject in range(10) for _ in range(2)],
            dtype=object,
        )
        labels = np.asarray([label for label in (0, 1) for _ in range(10) for _ in range(2)])
        store = DomainStore(
            name="synthetic",
            x=np.zeros((len(labels), 19, 128), dtype=np.float32),
            y=labels,
            subject_ids=subjects,
        )
        splits = split_source_indices_three_way(store, 0.2, 0.2, 0, 2, 2026)
        subject_sets = [set(store.subject_ids[indices]) for indices in splits]
        self.assertFalse(subject_sets[0] & subject_sets[1])
        self.assertFalse(subject_sets[0] & subject_sets[2])
        self.assertFalse(subject_sets[1] & subject_sets[2])
        for indices in splits:
            self.assertEqual(set(store.y[indices]), {0, 1})


if __name__ == "__main__":
    unittest.main()
