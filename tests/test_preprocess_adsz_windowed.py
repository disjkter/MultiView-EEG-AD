"""Tests for the corrected ADSZ preprocessing pipeline."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from scripts.preprocess_adsz_windowed import ADSZ_CHANNELS, canonical_adsz_subject_stem, extract_epochs


class ADSZPreprocessingTest(unittest.TestCase):
    """Verify windowing and subject identity preservation."""

    def test_extracts_all_non_overlapping_one_second_windows(self) -> None:
        """A two-second 128 Hz recording must produce two 1-second windows."""

        rng = np.random.default_rng(7)
        signal = rng.normal(size=(256, len(ADSZ_CHANNELS))).astype(np.float32)
        with patch("scripts.preprocess_adsz_windowed.np.loadtxt", return_value=signal):
            epochs = extract_epochs(
                Path("subject01.out"),
                fs=128.0,
                lowcut=0.5,
                highcut=45.0,
                filter_order=4,
                window_size=128,
                stride=128,
                normalize=False,
            )

        self.assertIsNotNone(epochs)
        self.assertEqual(epochs.shape, (2, 19, 128))

    def test_ec_and_eo_recordings_share_participant_id(self) -> None:
        """Paired recording conditions must not become separate subjects."""

        self.assertEqual(canonical_adsz_subject_stem("ec0103"), "0103")
        self.assertEqual(canonical_adsz_subject_stem("eo0103"), "0103")
        self.assertEqual(canonical_adsz_subject_stem("eeg20c1"), "eeg20_1")
        self.assertEqual(canonical_adsz_subject_stem("eeg20o1"), "eeg20_1")


if __name__ == "__main__":
    unittest.main()
