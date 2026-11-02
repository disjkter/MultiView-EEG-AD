"""Tests for Graph-Temporal masked reconstruction pretraining."""

import unittest

import torch

from models.GraphTemporalMAE import GraphTemporalMaskedAutoencoder, make_node_time_mask
from models.GraphTemporalNet import GraphTemporalNet


class GraphTemporalMAETest(unittest.TestCase):
    def test_node_and_time_masks_have_expected_structure(self) -> None:
        x = torch.randn(3, 19, 128)
        mask, node_mask, time_mask = make_node_time_mask(x, 0.3, 0.25)
        self.assertEqual(tuple(mask.shape), (3, 19, 128))
        self.assertTrue(torch.all(node_mask.sum(dim=1) == 6))
        self.assertTrue(torch.all(time_mask.sum(dim=1) == 32))
        for row in time_mask:
            indices = torch.where(row)[0]
            self.assertEqual(int(indices[-1] - indices[0] + 1), len(indices))

    def test_masked_reconstruction_backward(self) -> None:
        model = GraphTemporalMaskedAutoencoder(
            num_channels=19,
            seq_len=128,
            graph_out_dim=32,
            temporal_out_dim=32,
            decoder_hidden_dim=32,
        )
        x = torch.randn(2, 19, 128)
        output = model(x, node_mask_ratio=0.3, time_mask_ratio=0.3)
        loss = model.masked_loss(output, x)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.encoder.parameters()))

    def test_encoder_checkpoint_is_downstream_compatible(self) -> None:
        pretrain_model = GraphTemporalMaskedAutoencoder()
        downstream_model = GraphTemporalNet(num_classes=2)
        downstream_state = downstream_model.state_dict()
        compatible = {
            key: value
            for key, value in pretrain_model.encoder.state_dict().items()
            if not key.startswith("fusion_classifier.")
            and key in downstream_state
            and value.shape == downstream_state[key].shape
        }
        downstream_model.load_state_dict(compatible, strict=False)
        self.assertGreater(len(compatible), 0)
        self.assertTrue(any(key.startswith("graph_encoder.") for key in compatible))
        self.assertTrue(any(key.startswith("temporal_encoder.") for key in compatible))


if __name__ == "__main__":
    unittest.main()
