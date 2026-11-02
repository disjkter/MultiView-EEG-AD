"""Evaluate target-prevalence sensitivity from saved participant predictions."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_lodo_calibration_baselines import calibration_intercept_slope
from scripts.paper_analysis.common import (
    iter_lodo_folds,
    parse_named_path,
    write_csv,
    write_json,
)
from scripts.analyze_cross_cohort_calibration import calibration_metrics


def stratified_resample(
    labels: np.ndarray,
    prevalence: float,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample participant indices with replacement at a specified prevalence."""

    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    if not len(negative) or not len(positive):
        raise ValueError("Both classes are required for prevalence sensitivity")
    n_positive = min(sample_size - 1, max(1, int(round(sample_size * prevalence))))
    n_negative = sample_size - n_positive
    return np.concatenate(
        [
            rng.choice(negative, n_negative, replace=True),
            rng.choice(positive, n_positive, replace=True),
        ]
    )


def evaluate_sample(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute threshold and calibration metrics for one resample."""

    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    intercept, slope = calibration_intercept_slope(labels, probabilities)
    row = {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }
    row.update(calibration_metrics(labels, probabilities))
    return row


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize resampling distributions with percentile intervals."""

    groups: dict[tuple[str, str, str, int, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["experiment"]), str(row["variant"]), str(row["target"]),
            int(row["seed"]), float(row["requested_prevalence"]),
        )
        groups[key].append(row)
    metrics = (
        "balanced_accuracy", "auc", "sensitivity", "specificity", "ece", "brier",
        "nll", "calibration_intercept", "calibration_slope",
    )
    output: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        item: dict[str, object] = {
            "experiment": key[0], "variant": key[1], "target": key[2],
            "seed": key[3], "requested_prevalence": key[4], "resamples": len(group),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            values = values[np.isfinite(values)]
            item[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            item[f"{metric}_ci_low"] = float(np.quantile(values, 0.025)) if len(values) else float("nan")
            item[f"{metric}_ci_high"] = float(np.quantile(values, 0.975)) if len(values) else float("nan")
        output.append(item)
    return output


def main() -> None:
    """Run participant-level prevalence sensitivity without model retraining."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--target", default="ADFSU")
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--prevalences", nargs="+", type=float, default=[0.2, 0.4, 0.5, 0.6, 0.8])
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--expected-seeds", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=260812)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if any(value <= 0 or value >= 1 for value in args.prevalences):
        parser.error("Prevalences must lie strictly between zero and one")

    rng = np.random.default_rng(args.bootstrap_seed)
    rows: list[dict[str, object]] = []
    for value in args.results:
        experiment, root = parse_named_path(value)
        for record in iter_lodo_folds(root, experiment):
            if record["target"] != args.target:
                continue
            if args.variant and record["variant"] not in args.variant:
                continue
            fold = record["fold"]
            labels = np.asarray(fold["metrics"]["y_true"], dtype=np.int64)
            probabilities = np.asarray(fold["metrics"]["y_prob"], dtype=np.float64)
            sample_size = args.sample_size or len(labels)
            threshold = float(fold["decision_threshold"])
            for prevalence in args.prevalences:
                for resample in range(args.resamples):
                    indices = stratified_resample(labels, prevalence, sample_size, rng)
                    item: dict[str, object] = {
                        "experiment": experiment,
                        "variant": record["variant"],
                        "method": record["method"],
                        "seed": record["seed"],
                        "target": record["target"],
                        "requested_prevalence": prevalence,
                        "realized_prevalence": float(labels[indices].mean()),
                        "sample_size": sample_size,
                        "resample": resample,
                        "source_threshold": threshold,
                    }
                    item.update(evaluate_sample(labels[indices], probabilities[indices], threshold))
                    rows.append(item)
    if not rows:
        raise RuntimeError(f"No folds matched target={args.target} variants={args.variant}")
    if args.expected_seeds is not None:
        coverage: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            coverage[str(row["variant"])].add(int(row["seed"]))
        incomplete = {
            variant: sorted(seeds)
            for variant, seeds in coverage.items()
            if len(seeds) != args.expected_seeds
        }
        if incomplete:
            raise RuntimeError(
                f"Incomplete seed coverage; expected {args.expected_seeds}: {incomplete}"
            )
    summary = summarize(rows)
    prefix = args.output_prefix
    write_csv(prefix.with_name(prefix.name + "_resamples.csv"), rows)
    write_csv(prefix.with_name(prefix.name + "_summary.csv"), summary)
    write_json(
        prefix.with_suffix(".json"),
        {
            "metadata": {
                "participant_level": True,
                "model_retraining": False,
                "target_labels_used_for_evaluation_only": True,
                "not_clinical_prevalence_estimation": True,
                "bootstrap_seed": args.bootstrap_seed,
                "expected_seeds": args.expected_seeds,
            },
            "summary": summary,
        },
    )
    print(f"Saved {len(rows)} prevalence resamples to {prefix.parent}")


if __name__ == "__main__":
    main()
