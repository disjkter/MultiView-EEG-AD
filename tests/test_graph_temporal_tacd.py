"""Tests for graph-temporal causal domain decoupling."""

from __future__ import annotations

import unittest
from argparse import Namespace

import numpy as np
import torch

from models.GraphTemporalTACD import (
    GraphTemporalTACD,
    class_conditional_margin_alignment,
    cross_covariance_penalty,
    dann_schedule,
    multi_domain_coral_loss,
    multi_domain_mmd_loss,
)
from models.EEGNetDomainDecoupling import EEGNetDomainDecoupling
from models.StandardEEGBackboneAdapter import StandardEEGBackboneAdapter
from scripts.run_graph_temporal_lodo_tacd import (
    DomainStore,
    DomainBalancedBatchSampler,
    ParticipantBalancedBatchSampler,
    SampleRef,
    binary_metrics,
    build_model,
)


class GraphTemporalTACDTests(unittest.TestCase):
    """Validate model contracts and differentiability."""

    def test_forward_shapes(self) -> None:
        model = GraphTemporalTACD(
            graph_out_dim=16,
            temporal_out_dim=16,
            disease_dim=12,
            domain_dim=8,
        )
        output = model(torch.randn(3, 19, 128), grl_scale=0.5)
        self.assertEqual(tuple(output["disease_logits"].shape), (3, 2))
        self.assertEqual(tuple(output["domain_adversarial_logits"].shape), (3, 2))
        self.assertEqual(tuple(output["domain_specific_logits"].shape), (3, 2))
        self.assertEqual(tuple(output["disease_feature"].shape), (3, 12))
        self.assertEqual(tuple(output["domain_feature"].shape), (3, 8))

    def test_combined_loss_backpropagates(self) -> None:
        model = GraphTemporalTACD(
            graph_out_dim=16,
            temporal_out_dim=16,
            disease_dim=12,
            domain_dim=8,
        )
        output = model(torch.randn(4, 19, 128), grl_scale=0.5)
        loss = (
            output["disease_logits"].square().mean()
            + output["domain_adversarial_logits"].square().mean()
            + output["domain_specific_logits"].square().mean()
            + cross_covariance_penalty(
                output["disease_feature"], output["domain_feature"]
            )
        )
        loss.backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )

    def test_eegnet_domain_decoupling_forward_shapes(self) -> None:
        model = EEGNetDomainDecoupling(
            disease_dim=12,
            domain_dim=8,
            separable_filters=8,
        )
        output = model(torch.randn(3, 19, 128), grl_scale=0.5)
        self.assertEqual(tuple(output["disease_logits"].shape), (3, 2))
        self.assertEqual(tuple(output["domain_adversarial_logits"].shape), (3, 2))
        self.assertEqual(tuple(output["domain_specific_logits"].shape), (3, 2))
        self.assertEqual(tuple(output["disease_feature"].shape), (3, 12))
        self.assertEqual(tuple(output["domain_feature"].shape), (3, 8))

    def test_model_factory_builds_eegnet(self) -> None:
        args = Namespace(
            backbone="eegnet",
            eegnet_temporal_filters=4,
            eegnet_depth_multiplier=2,
            eegnet_separable_filters=8,
            eegnet_temporal_kernel=32,
            eegnet_separable_kernel=8,
            disease_dim=12,
            domain_dim=8,
            dropout=0.25,
            graph_out_dim=16,
            temporal_out_dim=16,
            graph_num_neighbors=4,
        )
        model = build_model(args, num_domains=2)
        self.assertIsInstance(model, EEGNetDomainDecoupling)

    def test_model_factory_builds_standard_erm_backbones(self) -> None:
        common = dict(
            method="erm",
            lambda_margin=0.0,
            dropout=0.25,
            conformer_d_model=32,
            conformer_num_heads=4,
            conformer_num_layers=1,
            conformer_ff_dim=64,
        )
        for backbone in ("shallow_convnet", "eeg_conformer"):
            with self.subTest(backbone=backbone):
                model = build_model(Namespace(backbone=backbone, **common), num_domains=2)
                self.assertIsInstance(model, StandardEEGBackboneAdapter)
                output = model(torch.randn(2, 19, 128))
                self.assertEqual(tuple(output["disease_logits"].shape), (2, 2))
                self.assertEqual(tuple(output["domain_adversarial_logits"].shape), (2, 2))

    def test_dann_schedule_is_monotonic(self) -> None:
        values = [dann_schedule(step, 10) for step in range(10)]
        self.assertEqual(values[0], 0.0)
        self.assertTrue(all(left <= right for left, right in zip(values, values[1:])))
        self.assertLess(values[-1], 1.0)

    def test_binary_metrics_respects_decision_threshold(self) -> None:
        y_true = np.asarray([0, 0, 1, 1], dtype=np.int64)
        positive_probability = np.asarray([0.2, 0.6, 0.55, 0.8])
        y_prob = np.column_stack([1.0 - positive_probability, positive_probability])

        fixed = binary_metrics(y_true, y_prob, threshold=0.5)
        calibrated = binary_metrics(y_true, y_prob, threshold=0.7)

        self.assertEqual(fixed["y_pred"], [0, 1, 1, 1])
        self.assertEqual(calibrated["y_pred"], [0, 0, 0, 1])

    def test_margin_alignment_is_zero_for_matching_domains(self) -> None:
        margins = torch.tensor([-2.0, -1.0, 1.0, 2.0] * 2)
        logits = torch.stack([-0.5 * margins, 0.5 * margins], dim=1)
        labels = torch.tensor([0, 0, 1, 1] * 2)
        domains = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        loss = class_conditional_margin_alignment(logits, labels, domains)

        self.assertAlmostEqual(float(loss), 0.0, places=7)

    def test_margin_alignment_penalizes_domain_shift_and_backpropagates(self) -> None:
        margins = torch.tensor(
            [-2.0, -1.0, 1.0, 2.0, 0.0, 1.0, 3.0, 4.0],
            requires_grad=True,
        )
        logits = torch.stack([-0.5 * margins, 0.5 * margins], dim=1)
        labels = torch.tensor([0, 0, 1, 1] * 2)
        domains = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        loss = class_conditional_margin_alignment(logits, labels, domains)
        loss.backward()

        self.assertGreater(float(loss), 0.0)
        self.assertIsNotNone(margins.grad)

    def test_coral_is_zero_for_identical_domain_features(self) -> None:
        group = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
        features = torch.cat([group, group], dim=0)
        domains = torch.tensor([0, 0, 0, 1, 1, 1])

        loss = multi_domain_coral_loss(features, domains)

        self.assertAlmostEqual(float(loss), 0.0, places=7)

    def test_coral_penalizes_covariance_shift_and_backpropagates(self) -> None:
        features = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
            requires_grad=True,
        )
        domains = torch.tensor([0, 0, 0, 1, 1, 1])

        loss = multi_domain_coral_loss(features, domains)
        loss.backward()

        self.assertGreater(float(loss), 0.0)
        self.assertIsNotNone(features.grad)

    def test_mmd_is_zero_for_identical_domain_features(self) -> None:
        group = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        features = torch.cat([group, group], dim=0)
        domains = torch.tensor([0, 0, 0, 1, 1, 1])

        loss = multi_domain_mmd_loss(features, domains)

        self.assertAlmostEqual(float(loss), 0.0, places=7)

    def test_mmd_penalizes_distribution_shift_and_backpropagates(self) -> None:
        features = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
            requires_grad=True,
        )
        domains = torch.tensor([0, 0, 1, 1])

        loss = multi_domain_mmd_loss(features, domains)
        loss.backward()

        self.assertGreater(float(loss), 0.0)
        self.assertIsNotNone(features.grad)

    def test_domain_balanced_batches_have_equal_domain_counts(self) -> None:
        refs = [
            SampleRef(0, index, 0) for index in range(3)
        ] + [
            SampleRef(1, index, 1) for index in range(7)
        ]
        sampler = DomainBalancedBatchSampler(refs, 2, batch_size=4, seed=2026)

        for batch in sampler:
            domains = [refs[index].domain_index for index in batch]
            self.assertEqual(domains.count(0), 2)
            self.assertEqual(domains.count(1), 2)

    def test_participant_batches_balance_domain_class_and_subject(self) -> None:
        stores = []
        refs = []
        for domain in range(2):
            labels = []
            subjects = []
            for label in range(2):
                for subject_index in range(3):
                    for _ in range(2):
                        labels.append(label)
                        subjects.append(f"d{domain}_c{label}_s{subject_index}")
            store = DomainStore(
                name=f"D{domain}",
                x=np.zeros((len(labels), 19, 128), dtype=np.float32),
                y=np.asarray(labels),
                subject_ids=np.asarray(subjects, dtype=object),
            )
            stores.append(store)
            refs.extend(
                SampleRef(domain, index, domain) for index in range(len(labels))
            )

        sampler = ParticipantBalancedBatchSampler(
            stores,
            refs,
            subjects_per_domain_class=2,
            windows_per_subject=2,
            seed=2026,
        )
        for batch in sampler:
            cells: dict[tuple[int, int], set[str]] = {}
            windows: dict[str, int] = {}
            for dataset_index in batch:
                ref = refs[dataset_index]
                store = stores[ref.store_index]
                label = int(store.y[ref.segment_index])
                subject = str(store.subject_ids[ref.segment_index])
                cells.setdefault((ref.domain_index, label), set()).add(subject)
                windows[subject] = windows.get(subject, 0) + 1
            self.assertEqual(set(cells), {(0, 0), (0, 1), (1, 0), (1, 1)})
            self.assertTrue(all(len(subjects) == 2 for subjects in cells.values()))
            self.assertTrue(all(count == 2 for count in windows.values()))
            self.assertEqual(len(batch), 16)


if __name__ == "__main__":
    unittest.main()
