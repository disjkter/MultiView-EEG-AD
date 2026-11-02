"""Tests for covariance alignment strategies."""

import unittest

import numpy as np

from scripts.run_dual_stream_loso_subject import (
    apply_euclidean_alignment,
    fit_euclidean_alignment,
    fit_log_euclidean_alignment,
)


class AlignmentStrategyTests(unittest.TestCase):
    """Check numerical and shape properties of EEG alignment matrices."""

    def setUp(self) -> None:
        rng = np.random.default_rng(2026)
        self.samples = rng.normal(size=(24, 5, 64)).astype(np.float32)

    def test_alignment_matrices_are_finite_and_symmetric(self) -> None:
        for fit in (fit_euclidean_alignment, fit_log_euclidean_alignment):
            matrix = fit(self.samples)
            self.assertEqual(matrix.shape, (5, 5))
            self.assertTrue(np.isfinite(matrix).all())
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-5)

    def test_log_euclidean_alignment_preserves_input_shape(self) -> None:
        matrix = fit_log_euclidean_alignment(self.samples)
        aligned = apply_euclidean_alignment(self.samples, matrix)
        self.assertEqual(aligned.shape, self.samples.shape)
        self.assertEqual(aligned.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
