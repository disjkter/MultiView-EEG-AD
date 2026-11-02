"""Tests for controlled transfer-validation command construction."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_transferability_validation import (
    build_finetune_command,
    run_with_retries,
)


class TransferabilityValidationCommandTest(unittest.TestCase):
    """Validate transfer initialization and threshold options."""

    def test_encoder_checkpoint_and_threshold_calibration_are_forwarded(self) -> None:
        command = build_finetune_command(
            "python",
            "target_data",
            Path("split.json"),
            Path("output"),
            "source.pth",
            {
                "checkpoint_scope": "encoders",
                "tune_threshold": True,
                "threshold_objective": "balanced_acc",
            },
            2026,
        )

        self.assertIn("--checkpoint-scope", command)
        self.assertEqual(command[command.index("--checkpoint-scope") + 1], "encoders")
        self.assertIn("--tune-threshold", command)
        self.assertEqual(
            command[command.index("--threshold-objective") + 1],
            "balanced_acc",
        )

    def test_target_only_does_not_receive_checkpoint_scope(self) -> None:
        command = build_finetune_command(
            "python",
            "target_data",
            Path("split.json"),
            Path("output"),
            None,
            {"tune_threshold": True},
            2026,
        )

        self.assertIn("--no-checkpoint", command)
        self.assertNotIn("--checkpoint-scope", command)
        self.assertIn("--tune-threshold", command)

    @patch("scripts.run_transferability_validation.time.sleep")
    @patch("scripts.run_transferability_validation.subprocess.run")
    def test_training_retries_transient_native_failure(
        self, run_mock, sleep_mock
    ) -> None:
        failed = unittest.mock.Mock(returncode=3221226505)
        succeeded = unittest.mock.Mock(returncode=0)
        run_mock.side_effect = [failed, succeeded]

        run_with_retries(["python", "train.py"], retries=3)

        self.assertEqual(run_mock.call_count, 2)
        sleep_mock.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
