"""Aggregate margin-alignment LODO variants and threshold controls."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


METRICS = (
    "accuracy",
    "balanced_accuracy",
    "f1",
    "auc",
    "sensitivity",
    "specificity",
)


def cohort_standardized_predictions(y_prob: list[float]) -> np.ndarray:
    """Predict after removing an unlabeled cohort's logit mean and scale."""

    probabilities = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(probabilities / (1.0 - probabilities))
    standardized = (logits - logits.mean()) / (logits.std() + 1e-8)
    return (standardized >= 0.0).astype(np.int64)


def cohort_standardized_metrics(
    y_true: list[int],
    y_prob: list[float],
    auc: float,
) -> dict[str, float]:
    """Evaluate scores after unlabeled cohort-wise logit standardization."""

    labels = np.asarray(y_true, dtype=np.int64)
    predictions = cohort_standardized_predictions(y_prob)
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": float(auc),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
    }


def collect(results_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Collect aggregate and fold-level rows from variant summaries."""

    aggregate_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for path in sorted(results_dir.rglob("lodo_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        variant = path.parent.name
        folds = payload["folds"]
        row: dict[str, object] = {
            "variant": variant,
            "method": payload["method"],
            "seed": payload["seed"],
            "num_folds": len(folds),
        }
        for metric in METRICS:
            calibrated = [float(fold["metrics"][metric]) for fold in folds]
            fixed = [float(fold["fixed_threshold_metrics"][metric]) for fold in folds]
            row[f"calibrated_{metric}"] = mean(calibrated)
            row[f"fixed_{metric}"] = mean(fixed)
            row[f"delta_{metric}"] = mean(calibrated) - mean(fixed)
        cohort_metrics = [
            cohort_standardized_metrics(
                fold["metrics"]["y_true"],
                fold["metrics"]["y_prob"],
                float(fold["metrics"]["auc"]),
            )
            for fold in folds
        ]
        for metric in METRICS:
            row[f"cohort_z_{metric}"] = mean(
                values[metric] for values in cohort_metrics
            )
        row["worst_calibrated_balanced_accuracy"] = min(
            float(fold["metrics"]["balanced_accuracy"]) for fold in folds
        )
        aggregate_rows.append(row)

        for fold, cohort_metric in zip(folds, cohort_metrics):
            fold_row: dict[str, object] = {
                "variant": variant,
                "method": payload["method"],
                "seed": payload["seed"],
                "target": fold["target"],
                "decision_threshold": fold["decision_threshold"],
                "threshold_objective_score": fold["threshold_objective_score"],
            }
            for metric in METRICS:
                calibrated_value = float(fold["metrics"][metric])
                fixed_value = float(fold["fixed_threshold_metrics"][metric])
                fold_row[f"calibrated_{metric}"] = calibrated_value
                fold_row[f"fixed_{metric}"] = fixed_value
                fold_row[f"delta_{metric}"] = calibrated_value - fixed_value
                fold_row[f"cohort_z_{metric}"] = cohort_metric[metric]
            fold_rows.append(fold_row)
    return aggregate_rows, fold_rows


def summarize_variants(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize repeated seeds for each variant."""

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)

    summaries: list[dict[str, object]] = []
    numeric_fields = [
        key
        for key in rows[0]
        if key not in {"variant", "method", "seed", "num_folds"}
    ]
    for variant, variant_rows in sorted(grouped.items()):
        summary: dict[str, object] = {
            "variant": variant,
            "method": variant_rows[0]["method"],
            "num_seeds": len(variant_rows),
        }
        for field in numeric_fields:
            values = [float(row[field]) for row in variant_rows]
            summary[f"{field}_mean"] = mean(values)
            summary[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def exact_sign_flip(differences: list[float]) -> float:
    """Return a two-sided exact paired sign-flip p-value."""

    observed = abs(mean(differences))
    null_values = (
        abs(mean(sign * value for sign, value in zip(signs, differences)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    )
    extreme = sum(value >= observed - 1e-15 for value in null_values)
    return extreme / (2 ** len(differences))


def paired_tests(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Compare TA-CD + margin against matched-seed baselines."""

    by_variant_seed = {
        (str(row["variant"]), int(row["seed"])): row for row in rows
    }
    comparisons: list[dict[str, object]] = []
    fields = (
        "calibrated_balanced_accuracy",
        "cohort_z_balanced_accuracy",
        "calibrated_auc",
        "calibrated_f1",
    )
    comparison_specs = (
        ("care_dg", "erm_margin"),
        ("care_dg", "erm"),
        ("care_dg", "dann"),
        ("care_dg", "coral"),
        ("care_dg", "mmd"),
        ("care_dg", "tacd_margin"),
        ("care_dg", "tacd"),
        ("care_dg", "care_no_score"),
        ("care_dg", "care_no_calibration"),
        ("care_dg", "care_no_entropy"),
        ("care_dg", "pooled_mmce"),
        ("care_dg", "multidomain_calibration"),
        ("tacd_margin", "erm_margin"),
        ("tacd_margin", "erm"),
        ("tacd_margin", "dann"),
        ("tacd_margin", "tacd"),
        ("tacd_margin", "coral"),
        ("tacd_margin", "mmd"),
        ("erm_margin", "erm"),
        ("erm", "dann"),
        ("tacd", "dann"),
        ("coral", "erm"),
        ("mmd", "erm"),
    )
    for reference, baseline in comparison_specs:
        seeds = sorted(
            seed
            for variant, seed in by_variant_seed
            if variant == reference
            and (baseline, seed) in by_variant_seed
        )
        if not seeds:
            continue
        for field in fields:
            differences = [
                float(by_variant_seed[(reference, seed)][field])
                - float(by_variant_seed[(baseline, seed)][field])
                for seed in seeds
            ]
            difference_std = stdev(differences) if len(differences) > 1 else 0.0
            comparisons.append(
                {
                    "reference": reference,
                    "baseline": baseline,
                    "metric": field,
                    "num_pairs": len(differences),
                    "mean_difference": mean(differences),
                    "difference_std": difference_std,
                    "cohen_dz": (
                        mean(differences) / difference_std
                        if difference_std > 0
                        else None
                    ),
                    "wins": sum(value > 0 for value in differences),
                    "ties": sum(value == 0 for value in differences),
                    "exact_sign_flip_pvalue": exact_sign_flip(differences),
                }
            )
    return add_holm_adjustment(comparisons)


def add_holm_adjustment(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Add Holm-adjusted p-values within each reported metric family."""

    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row["metric"]), []).append(index)
    for indices in grouped.values():
        ordered = sorted(
            indices,
            key=lambda index: float(rows[index]["exact_sign_flip_pvalue"]),
        )
        running_max = 0.0
        family_size = len(ordered)
        for rank, index in enumerate(ordered):
            raw = float(rows[index]["exact_sign_flip_pvalue"])
            adjusted = min(1.0, (family_size - rank) * raw)
            running_max = max(running_max, adjusted)
            rows[index]["holm_adjusted_pvalue"] = running_max
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV with a stable header."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Parse arguments and write JSON plus aggregate/fold CSV files."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=None)
    args = parser.parse_args()

    aggregate_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for results_dir in args.results_dir:
        directory_aggregate, directory_folds = collect(results_dir)
        aggregate_rows.extend(directory_aggregate)
        fold_rows.extend(directory_folds)
    if not aggregate_rows:
        raise FileNotFoundError(
            f"No */lodo_summary.json files found under {args.results_dir}"
        )
    if args.expected_seeds is not None:
        counts: dict[str, int] = {}
        for row in aggregate_rows:
            variant = str(row["variant"])
            counts[variant] = counts.get(variant, 0) + 1
        incomplete = {
            variant: count
            for variant, count in counts.items()
            if count != args.expected_seeds
        }
        if incomplete:
            raise RuntimeError(
                f"Incomplete variant seed counts; expected {args.expected_seeds}: "
                f"{incomplete}"
            )

    prefix = args.output_prefix
    variant_rows = summarize_variants(aggregate_rows)
    comparison_rows = paired_tests(aggregate_rows)
    write_csv(prefix.with_name(prefix.name + "_aggregate.csv"), aggregate_rows)
    write_csv(prefix.with_name(prefix.name + "_folds.csv"), fold_rows)
    write_csv(prefix.with_name(prefix.name + "_variants.csv"), variant_rows)
    write_csv(prefix.with_name(prefix.name + "_paired_tests.csv"), comparison_rows)
    json_path = prefix.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "results_dirs": [str(path) for path in args.results_dir],
                    "cohort_z_uses_target_labels": False,
                    "cohort_z_uses_unlabeled_target_scores": True,
                    "cohort_z_protocol": "z-score target-cohort subject logits; threshold at zero",
                },
                "aggregate": aggregate_rows,
                "variant_summary": variant_rows,
                "paired_tests": comparison_rows,
                "folds": fold_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    ranked = sorted(
        variant_rows,
        key=lambda row: float(row["cohort_z_balanced_accuracy_mean"]),
        reverse=True,
    )
    for row in ranked:
        print(
            f"{row['variant']}: "
            f"source_BA={float(row['calibrated_balanced_accuracy_mean']):.4f} "
            f"cohort_z_BA={float(row['cohort_z_balanced_accuracy_mean']):.4f} "
            f"AUC={float(row['calibrated_auc_mean']):.4f} "
            f"worst_source_BA={float(row['worst_calibrated_balanced_accuracy_mean']):.4f}"
        )


if __name__ == "__main__":
    main()
