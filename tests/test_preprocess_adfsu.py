"""Tests for canonical ADFSU preprocessing helpers."""

import unittest

import numpy as np

from scripts.preprocess_adfsu import CHANNELS, segment_signal


class ADFSUPreprocessingTest(unittest.TestCase):
    """Check deterministic window construction without filesystem fixtures."""

    def test_eight_seconds_produces_eight_windows(self) -> None:
        """A 1024-sample 128 Hz recording must yield eight 1-second windows."""

        signal = np.zeros((len(CHANNELS), 1024), dtype=np.float32)
        windows = segment_signal(signal, window_size=128, stride=128)
        self.assertEqual(windows.shape, (8, 19, 128))


if __name__ == "__main__":
    unittest.main()
