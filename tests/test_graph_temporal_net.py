"""Tests for the topology-aware graph-temporal classifier."""

import unittest

import torch

from models.GraphTemporalNet import GraphTemporalNet, STANDARD_19_POSITIONS, build_knn_adjacency
from models.MSCNNClassifier import MSCNNClassifier


class GraphTemporalNetTest(unittest.TestCase):
    def test_adjacency_is_symmetric_and_normalized(self) -> None:
        adjacency = build_knn_adjacency(STANDARD_19_POSITIONS, num_neighbors=4)
        self.assertEqual(tuple(adjacency.shape), (19, 19))
        self.assertTrue(torch.allclose(adjacency, adjacency.T))
        self.assertTrue(torch.all(torch.diag(adjacency) > 0))
        self.assertTrue(torch.isfinite(adjacency).all())

    def test_forward_and_backward(self) -> None:
        model = GraphTemporalNet(num_classes=2, num_channels=19)
        x = torch.randn(2, 19, 128)
        logits, features = model(x)
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(tuple(features.shape), (2, 512))
        logits.sum().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_mscnn_only_control(self) -> None:
        model = MSCNNClassifier(num_classes=2, num_channels=19)
        logits, features = model(torch.randn(2, 19, 128))
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(tuple(features.shape), (2, 256))


if __name__ == "__main__":
    unittest.main()
