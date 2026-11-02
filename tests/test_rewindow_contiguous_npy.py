"""Tests for contiguous NPY rewindowing."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.rewindow_contiguous_npy import rewindow


class RewindowContiguousNpyTest(unittest.TestCase):
    """Verify ordering and participant-boundary handling."""

    def test_never_combines_different_subjects(self) -> None:
        source = Path("outputs/_rewindow_test_source")
        output = Path("outputs/_rewindow_test_output")
        source.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        x = np.arange(7 * 2 * 3, dtype=np.float32).reshape(7, 2, 3)
        np.save(source / "x_19ch.npy", x)
        np.save(source / "y_19ch.npy", np.asarray([0, 0, 0, 1, 1, 1, 1]))
        np.save(
            source / "sub_ids_19ch.npy",
            np.asarray(["a", "a", "a", "b", "b", "b", "b"], dtype=object),
        )

        summary = rewindow(source, output, factor=2)
        combined = np.load(output / "x_19ch.npy")
        subjects = np.load(output / "sub_ids_19ch.npy")

        self.assertEqual(summary["num_segments"], 3)
        self.assertEqual(summary["dropped_source_windows_at_run_ends"], 1)
        np.testing.assert_array_equal(subjects, np.asarray(["a", "b", "b"]))
        np.testing.assert_array_equal(
            combined[0], x[0:2].transpose(1, 0, 2).reshape(2, 6)
        )


if __name__ == "__main__":
    unittest.main()
