"""Aggregate training-size or window-length LODO sensitivity experiments."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper_analysis.common import (
    cohort_z_predictions,
    iter_lodo_folds,
    parse_named_path,
    participant_metrics,
    write_csv,
    write_json,
)
from sklearn.metrics import balanced_accuracy_score


def collect(condition_specs: list[str]) -> list[dict[str, object]]:
    """Collect fold metrics and effective participant counts by condition."""

    rows: list[dict[str, object]] = []
    for value in condition_specs:
        condition, root = parse_named_path(value)
        for record in iter_lodo_folds(root, condition):
            fold = record["fold"]
            metrics = fold["metrics"]
            labels = np.asarray(metrics["y_true"], dtype=np.int64)
            probabilities = np.asarray(metrics["y_prob"], dtype=np.float64)
            reliability = participant_metrics(
                labels, probabilities, float(fold["decision_threshold"])
            )
            row: dict[str, object] = {
                "condition": condition,
                "variant": record["variant"],
                "method": record["method"],
                "backbone": record["backbone"],
                "seed": record["seed"],
                "target": record["target"],
                "source_subjects": int(sum(fold["source_train_subjects"].values())),
                "source_validation_subjects": int(sum(fold["source_val_subjects"].values())),
                "target_subjects": int(fold["target_test_subjects"]),
                "source_threshold": float(fold["decision_threshold"]),
                "source_threshold_ba": reliability["balanced_accuracy"],
                "cohort_z_ba": float(
                    balanced_accuracy_score(labels, cohort_z_predictions(probabilities))
                ),
                "auc": reliability["auc"],
                "ece": reliability["ece"],
                "brier": reliability["brier"],
                "nll": reliability["nll"],
            }
            rows.append(row)
    return rows


def filter_and_validate_rows(
    rows: list[dict[str, object]],
    targets: list[str] | None = None,
    variants: list[str] | None = None,
) -> list[dict[str, object]]:
    """Filter rows and require matched target sets for every condition and seed."""

    expected_targets = set(targets or [])
    expected_variants = set(variants or [])
    selected = [
        row
        for row in rows
        if (not expected_targets or str(row["target"]) in expected_targets)
        and (not expected_variants or str(row["variant"]) in expected_variants)
    ]
    if not selected:
        raise RuntimeError("No sensitivity rows remain after target/variant filtering")

    if expected_targets:
        grouped_targets: dict[tuple[str, str, int], set[str]] = defaultdict(set)
        for row in selected:
            key = (str(row["condition"]), str(row["variant"]), int(row["seed"]))
            grouped_targets[key].add(str(row["target"]))
        mismatches = [
            {"group": key, "targets": sorted(actual)}
            for key, actual in sorted(grouped_targets.items())
            if actual != expected_targets
        ]
        if mismatches:
            raise RuntimeError(
                "Sensitivity groups do not contain the requested matched targets: "
                f"{mismatches[:5]}"
            )
    return selected


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Macro-average targets within seed, then summarize matched seeds."""

    seed_groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        seed_groups[(str(row["condition"]), str(row["variant"]), int(row["seed"]))].append(row)
    metrics = ("source_threshold_ba", "cohort_z_ba", "auc", "ece", "brier", "nll")
    seeds: list[dict[str, object]] = []
    for key, group in sorted(seed_groups.items()):
        item: dict[str, object] = {
            "condition": key[0], "variant": key[1], "seed": key[2],
            "num_targets": len(group),
            "worst_cohort_ba": min(float(row["source_threshold_ba"]) for row in group),
        }
        for metric in metrics:
            item[metric] = float(np.mean([float(row[metric]) for row in group]))
        seeds.append(item)
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in seeds:
        groups[(str(row["condition"]), str(row["variant"]))].append(row)
    output: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        item: dict[str, object] = {"condition": key[0], "variant": key[1], "num_seeds": len(group)}
        for metric in (*metrics, "worst_cohort_ba"):
            values = np.asarray([float(row[metric]) for row in group])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(item)
    return output


def main() -> None:
    """Aggregate completed sensitivity conditions into CSV and JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help="Retain this target and require it in every condition/variant/seed group",
    )
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="Retain this training variant (repeat for multiple variants)",
    )
    parser.add_argument("--expected-seeds", type=int, default=None)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    rows = filter_and_validate_rows(collect(args.condition), args.targets, args.variants)
    summary = summarize(rows)
    if args.expected_seeds is not None:
        incomplete = [row for row in summary if int(row["num_seeds"]) != args.expected_seeds]
        if incomplete:
            raise RuntimeError(f"Incomplete sensitivity groups: {incomplete[:5]}")
    prefix = args.output_prefix
    write_csv(prefix.with_name(prefix.name + "_folds.csv"), rows)
    write_csv(prefix.with_name(prefix.name + "_summary.csv"), summary)
    write_json(prefix.with_suffix(".json"), {"folds": rows, "summary": summary})
    print(f"Aggregated {len(rows)} fold runs into {len(summary)} condition summaries")


if __name__ == "__main__":
    main()
