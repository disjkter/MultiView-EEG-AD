"""Apply the prespecified CQ-CARE-DG development screening gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_cross_cohort_calibration import calibration_metrics

METRICS = ("balanced_accuracy", "auc", "ece", "brier", "nll")


def collect_variant(root: Path, variant: str) -> dict[int, dict[str, dict[str, float]]]:
    """Collect participant-level fold metrics for one variant by seed and target."""

    rows: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for path in sorted(root.rglob(f"{variant}/lodo_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        for fold in payload["folds"]:
            target = str(fold["target"])
            metrics = fold["metrics"]
            calibration = calibration_metrics(metrics["y_true"], metrics["y_prob"])
            rows[seed][target] = {
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "auc": float(metrics["auc"]),
                "ece": float(calibration["ece"]),
                "brier": float(calibration["brier"]),
                "nll": float(calibration["nll"]),
            }
    if not rows:
        raise FileNotFoundError(f"No results for variant={variant} under {root}")
    return dict(rows)


def apply_gate(
    proposed: dict[int, dict[str, dict[str, float]]],
    baseline: dict[int, dict[str, dict[str, float]]],
    max_discrimination_drop: float = 0.02,
) -> dict[str, object]:
    """Compare matched seeds and targets using the frozen screening criteria."""

    seeds = sorted(set(proposed).intersection(baseline))
    if len(seeds) < 3:
        raise ValueError(f"At least three matched seeds are required, found {seeds}")
    targets = sorted(
        set.intersection(
            *(set(proposed[seed]).intersection(baseline[seed]) for seed in seeds)
        )
    )
    if len(targets) < 3:
        raise ValueError(f"At least three matched targets are required, found {targets}")

    differences: dict[str, list[float]] = {metric: [] for metric in METRICS}
    target_brier: dict[str, float] = {}
    for target in targets:
        per_target = {metric: [] for metric in METRICS}
        for seed in seeds:
            for metric in METRICS:
                value = (
                    proposed[seed][target][metric]
                    - baseline[seed][target][metric]
                )
                differences[metric].append(value)
                per_target[metric].append(value)
        target_brier[target] = mean(per_target["brier"])

    mean_difference = {
        metric: mean(values) for metric, values in differences.items()
    }
    criteria = {
        "mean_ece_improves": mean_difference["ece"] < 0.0,
        "mean_brier_improves": mean_difference["brier"] < 0.0,
        "brier_improves_on_at_least_two_targets": sum(
            value < 0.0 for value in target_brier.values()
        )
        >= 2,
        "auc_drop_within_tolerance": mean_difference["auc"]
        >= -max_discrimination_drop,
        "balanced_accuracy_drop_within_tolerance": mean_difference[
            "balanced_accuracy"
        ]
        >= -max_discrimination_drop,
    }
    return {
        "decision": "PASS" if all(criteria.values()) else "STOP",
        "matched_seeds": seeds,
        "matched_targets": targets,
        "max_discrimination_drop": max_discrimination_drop,
        "mean_proposed_minus_baseline": mean_difference,
        "target_brier_difference": target_brier,
        "criteria": criteria,
    }


def main() -> None:
    """Read existing ERM+Margin and CQ-CARE-DG results and write the gate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--care-dir", type=Path, required=True)
    parser.add_argument("--baseline-variant", default="erm_margin")
    parser.add_argument("--care-variant", default="care_dg")
    parser.add_argument("--max-discrimination-drop", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = apply_gate(
        collect_variant(args.care_dir, args.care_variant),
        collect_variant(args.baseline_dir, args.baseline_variant),
        args.max_discrimination_drop,
    )
    result["baseline_variant"] = args.baseline_variant
    result["care_variant"] = args.care_variant
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
