"""Tests for strict and few-shot subject split loading."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from scripts.run_dual_stream_subject_split import subsample_training_subjects
from utils.subject_split import load_subject_split_masks


class SubjectSplitUtilsTest(unittest.TestCase):
    """Verify explicit exclusions without weakening normal split checks."""

    def write_split(self, payload: dict[str, object]) -> Path:
        Path("outputs").mkdir(exist_ok=True)
        path = Path("outputs/_test_subject_split_utils.json")
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_explicit_excluded_subject_is_allowed(self) -> None:
        path = self.write_split(
            {
                "train_subjects": ["s1"],
                "val_subjects": ["s2"],
                "test_subjects": ["s3"],
                "excluded_subjects": ["s4"],
            }
        )
        subjects = np.asarray(["s1", "s2", "s3", "s4"], dtype=object)
        train, val, test = load_subject_split_masks(path, subjects)
        self.assertEqual(train.tolist(), [True, False, False, False])
        self.assertEqual(val.tolist(), [False, True, False, False])
        self.assertEqual(test.tolist(), [False, False, True, False])

    def test_unlisted_subject_remains_an_error(self) -> None:
        path = self.write_split(
            {
                "train_subjects": ["s1"],
                "val_subjects": ["s2"],
                "test_subjects": ["s3"],
            }
        )
        subjects = np.asarray(["s1", "s2", "s3", "s4"], dtype=object)
        with self.assertRaisesRegex(ValueError, "does not assign all subjects"):
            load_subject_split_masks(path, subjects)

    def test_exclusion_cannot_overlap_training(self) -> None:
        path = self.write_split(
            {
                "train_subjects": ["s1"],
                "val_subjects": ["s2"],
                "test_subjects": ["s3"],
                "excluded_subjects": ["s1"],
            }
        )
        subjects = np.asarray(["s1", "s2", "s3"], dtype=object)
        with self.assertRaisesRegex(ValueError, "overlapping subjects"):
            load_subject_split_masks(path, subjects)

    def test_split_subject_absent_from_dataset_is_an_error(self) -> None:
        path = self.write_split(
            {
                "train_subjects": ["s1"],
                "val_subjects": ["s2"],
                "test_subjects": ["s3", "s_missing"],
            }
        )
        subjects = np.asarray(["s1", "s2", "s3"], dtype=object)
        with self.assertRaisesRegex(ValueError, "absent from the dataset"):
            load_subject_split_masks(path, subjects)

    def test_training_subject_fraction_is_stratified_and_deterministic(self) -> None:
        subjects = np.repeat(
            np.asarray(["a", "b", "c", "d", "e", "f"], dtype=object), 2
        )
        labels = np.repeat(np.asarray([0, 0, 0, 1, 1, 1]), 2)
        indices = np.arange(len(subjects))
        first = subsample_training_subjects(indices, labels, subjects, 0.5, 2026)
        second = subsample_training_subjects(indices, labels, subjects, 0.5, 2026)
        np.testing.assert_array_equal(first, second)
        retained = np.unique(subjects[first])
        retained_labels = {
            int(labels[np.where(subjects == subject)[0][0]]) for subject in retained
        }
        self.assertEqual(len(retained), 4)
        self.assertEqual(retained_labels, {0, 1})


if __name__ == "__main__":
    unittest.main()
