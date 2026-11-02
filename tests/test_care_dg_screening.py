"""Tests for the prespecified CQ-CARE-DG screening decision."""

from __future__ import annotations

import unittest
import subprocess
import sys
from pathlib import Path

from scripts.evaluate_care_dg_screening import apply_gate


def make_runs(delta: dict[str, float]) -> tuple[dict, dict]:
    """Create three matched seeds and targets with known differences."""

    baseline = {}
    proposed = {}
    values = {
        "balanced_accuracy": 0.60,
        "auc": 0.75,
        "ece": 0.35,
        "brier": 0.30,
        "nll": 0.80,
    }
    for seed in (2026, 2027, 2028):
        baseline[seed] = {}
        proposed[seed] = {}
        for target in ("A", "B", "C"):
            baseline[seed][target] = dict(values)
            proposed[seed][target] = {
                metric: value + delta.get(metric, 0.0)
                for metric, value in values.items()
            }
    return proposed, baseline


class CareDGScreeningTests(unittest.TestCase):
    """Check that the gate follows the frozen directions and tolerance."""

    def test_passes_calibration_gain_without_material_discrimination_loss(self) -> None:
        proposed, baseline = make_runs(
            {"ece": -0.02, "brier": -0.01, "nll": -0.03, "auc": -0.01}
        )
        self.assertEqual(apply_gate(proposed, baseline)["decision"], "PASS")

    def test_stops_when_auc_loss_exceeds_tolerance(self) -> None:
        proposed, baseline = make_runs(
            {"ece": -0.02, "brier": -0.01, "nll": -0.03, "auc": -0.03}
        )
        self.assertEqual(apply_gate(proposed, baseline)["decision"], "STOP")

    def test_stops_without_brier_improvement(self) -> None:
        proposed, baseline = make_runs({"ece": -0.02, "brier": 0.01})
        self.assertEqual(apply_gate(proposed, baseline)["decision"], "STOP")

    def test_script_help_runs_outside_project_directory(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "evaluate_care_dg_screening.py"),
                "--help",
            ],
            cwd=project_root.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--baseline-dir", result.stdout)


if __name__ == "__main__":
    unittest.main()
