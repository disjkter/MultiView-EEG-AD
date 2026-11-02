"""Tests for subject-split checkpoint CLI validation."""

import unittest

import torch

from scripts.run_dual_stream_subject_split import (
    capture_l2sp_anchor,
    l2sp_penalty,
    resolve_checkpoint_argument,
    select_compatible_checkpoint_tensors,
)


class CheckpointArgumentTests(unittest.TestCase):
    def test_checkpoint_path_is_preserved(self) -> None:
        self.assertEqual(resolve_checkpoint_argument("model.pth", False), "model.pth")

    def test_no_checkpoint_resolves_to_none(self) -> None:
        self.assertIsNone(resolve_checkpoint_argument(None, True))

    def test_conflicting_flags_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            resolve_checkpoint_argument("model.pth", True)

    def test_compatible_checkpoint_selection_respects_scope_and_shape(self) -> None:
        model_state = {
            "temporal_encoder.weight": torch.zeros(2, 3),
            "fusion_classifier.weight": torch.zeros(2, 2),
        }
        checkpoint = {
            "temporal_encoder.weight": torch.ones(2, 3),
            "fusion_classifier.weight": torch.ones(2, 2),
            "temporal_encoder.bad_shape": torch.ones(1),
        }

        selected = select_compatible_checkpoint_tensors(
            checkpoint,
            model_state,
            ("temporal_encoder.",),
        )

        self.assertEqual(list(selected), ["temporal_encoder.weight"])

    def test_dual_stream_encoder_scope_excludes_classifier(self) -> None:
        model_state = {
            "spatial_expert.fc.weight": torch.zeros(2, 3),
            "femba.projection.weight": torch.zeros(2, 3),
            "fusion_classifier.1.weight": torch.zeros(2, 2),
        }
        checkpoint = {key: torch.ones_like(value) for key, value in model_state.items()}

        selected = select_compatible_checkpoint_tensors(
            checkpoint,
            model_state,
            ("spatial_expert.", "femba.", "neuro_prior.", "prior_projector."),
        )

        self.assertEqual(
            set(selected),
            {"spatial_expert.fc.weight", "femba.projection.weight"},
        )
        self.assertNotIn("fusion_classifier.1.weight", selected)

    def test_l2sp_anchor_constrains_only_selected_encoder(self) -> None:
        model = torch.nn.Module()
        model.spatial_expert = torch.nn.Linear(2, 1, bias=False)
        model.fusion_classifier = torch.nn.Linear(1, 1, bias=False)
        anchor = capture_l2sp_anchor(model, "encoders")
        self.assertEqual(set(anchor), {"spatial_expert.weight"})

        with torch.no_grad():
            model.spatial_expert.weight.add_(1.0)
            model.fusion_classifier.weight.add_(5.0)

        self.assertAlmostEqual(float(l2sp_penalty(model, anchor).detach()), 2.0)


if __name__ == "__main__":
    unittest.main()
