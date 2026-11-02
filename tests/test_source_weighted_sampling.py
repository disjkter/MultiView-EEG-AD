"""Tests for source-aware EEG pretraining sampling."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.source_weighted_sampling import (
    MultiSourceEEGDataset,
    SourceWeightedSampler,
    load_similarity_weights,
)


class SourceWeightedSamplingTest(unittest.TestCase):
    """Verify domain mapping, normalization, and source probabilities."""

    @staticmethod
    def temporary_directory() -> tempfile.TemporaryDirectory[str]:
        """Create test data under the writable repository test directory."""

        root = Path(__file__).resolve().parent / "_tmp"
        root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=root)

    def test_multisource_dataset_maps_global_indices(self) -> None:
        with self.temporary_directory() as tmp:
            root = Path(tmp)
            first = root / "first.npy"
            second = root / "second.npy"
            np.save(first, np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4))
            np.save(second, np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4) + 10)

            dataset = MultiSourceEEGDataset([first, second], ["first", "second"])

            self.assertEqual(len(dataset), 5)
            self.assertEqual(dataset.source_lengths, [2, 3])
            self.assertEqual([dataset.source_index(i) for i in range(5)], [0, 0, 1, 1, 1])
            self.assertEqual(tuple(dataset[4].shape), (2, 4))
            np.testing.assert_allclose(dataset[4].mean(dim=-1).numpy(), np.zeros(2), atol=1e-5)
            dataset.close()

    def test_sampler_tracks_requested_source_distribution(self) -> None:
        sampler = SourceWeightedSampler([2, 100], [0.8, 0.2], num_samples=10000, seed=7)
        indices = list(sampler)
        first_count = sum(index < 2 for index in indices)

        self.assertEqual(len(indices), 10000)
        self.assertAlmostEqual(first_count / len(indices), 0.8, delta=0.02)
        self.assertEqual(sum(sampler.last_domain_counts), 10000)

    def test_load_similarity_weights_preserves_source_order(self) -> None:
        with self.temporary_directory() as tmp:
            path = Path(tmp) / "similarity.json"
            path.write_text(
                json.dumps(
                    [
                        {"target": "target-a", "source": "source-b", "sampling_weight": 0.25},
                        {"target": "target-a", "source": "source-a", "sampling_weight": 0.75},
                    ]
                ),
                encoding="utf-8",
            )

            weights = load_similarity_weights(path, "target-a", ["source-a", "source-b"])

            np.testing.assert_allclose(weights, [0.75, 0.25])


if __name__ == "__main__":
    unittest.main()
