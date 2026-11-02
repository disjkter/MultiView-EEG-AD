"""Tests for exact participant-level duplicate detection."""

import unittest

import numpy as np

from scripts.deduplicate_processed_participants import find_duplicate_groups


class DeduplicateParticipantTests(unittest.TestCase):
    """Verify duplicate grouping and conflicting-label protection."""

    def test_finds_exact_participant_duplicates(self) -> None:
        first = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
        data = np.concatenate([first, first, first + 1], axis=0)
        labels = np.asarray([1, 1, 0])
        ids = np.asarray(["a", "b", "c"])
        groups = find_duplicate_groups(data, labels, ids)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["kept_participant"], "a")
        self.assertEqual(groups[0]["removed_participants"], ["b"])

    def test_rejects_conflicting_duplicate_labels(self) -> None:
        first = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
        data = np.concatenate([first, first], axis=0)
        labels = np.asarray([0, 1])
        ids = np.asarray(["a", "b"])
        with self.assertRaises(ValueError):
            find_duplicate_groups(data, labels, ids)


if __name__ == "__main__":
    unittest.main()
