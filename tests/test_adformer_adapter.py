"""Tests for the official ADformer LODO adapter."""

from __future__ import annotations

import unittest

import torch

from models.ADformerDomainAdapter import ADformerDomainAdapter


class ADformerAdapterTests(unittest.TestCase):
    """Validate shapes and gradients through the official implementation."""

    def test_forward_and_backward(self) -> None:
        model = ADformerDomainAdapter(
            num_domains=2,
            d_model=32,
            n_heads=4,
            e_layers=1,
            d_ff=64,
            dropout=0.0,
        )
        output = model(torch.randn(2, 19, 128))
        self.assertEqual(output["disease_logits"].shape, (2, 2))
        self.assertEqual(output["domain_adversarial_logits"].shape, (2, 2))
        output["disease_logits"].sum().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
