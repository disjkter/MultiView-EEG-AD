"""Tests for transferability source-model command construction."""

from __future__ import annotations

import unittest

from scripts.run_transferability_source_models import build_command


class TransferabilitySourceRunnerTest(unittest.TestCase):
    """Ensure source training stays compatible with the subject-split runner."""

    def test_build_command_uses_graph_temporal_without_checkpoint(self) -> None:
        source = {
            "name": "demo",
            "data_dir": "data/demo",
            "split_file": "outputs/splits/demo.json",
            "checkpoint": "outputs/source_models/demo/best_model.pth",
        }
        command = build_command(
            "python",
            source,
            {"epochs": 3, "batch_size": 8, "max_segments_per_subject": 12},
            seed=2026,
        )
        self.assertIn("--no-checkpoint", command)
        self.assertEqual(command[command.index("--model-type") + 1], "graph_temporal")
        self.assertEqual(command[command.index("--epochs") + 1], "3")
        self.assertEqual(command[command.index("--selection-metric") + 1], "auc")
        self.assertEqual(
            command[command.index("--max-train-segments-per-subject") + 1],
            "12",
        )
        self.assertEqual(
            command[command.index("--output-dir") + 1],
            "outputs\\source_models\\demo",
        )

    def test_source_can_override_selection_metric(self) -> None:
        source = {
            "name": "demo",
            "data_dir": "data/demo",
            "split_file": "outputs/splits/demo.json",
            "checkpoint": "outputs/source_models/demo/best_model.pth",
            "selection_metric": "loss",
        }
        command = build_command("python", source, {}, seed=2026)
        self.assertEqual(command[command.index("--selection-metric") + 1], "loss")


if __name__ == "__main__":
    unittest.main()
