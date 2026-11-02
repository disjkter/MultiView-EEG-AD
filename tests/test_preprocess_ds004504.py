"""Tests for canonical ds004504 preprocessing helpers."""

import unittest

import numpy as np

from scripts.preprocess_ds004504 import segment_signal


class DS004504PreprocessingTest(unittest.TestCase):
    """Verify one-second non-overlapping segmentation."""

    def test_segment_signal_drops_only_short_tail(self) -> None:
        """Two full windows and an incomplete tail should produce two windows."""

        signal = np.arange(19 * 300, dtype=np.float32).reshape(19, 300)
        windows = segment_signal(signal, window_size=128)
        self.assertEqual(windows.shape, (2, 19, 128))
        np.testing.assert_array_equal(windows[0], signal[:, :128])
        np.testing.assert_array_equal(windows[1], signal[:, 128:256])


if __name__ == "__main__":
    unittest.main()
