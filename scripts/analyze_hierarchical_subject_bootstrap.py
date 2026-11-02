"""Compute hierarchical seed-and-subject bootstrap intervals from LODO summaries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METRICS = ("balanced_accuracy", "auc", "sensitivity", "specificity", "ece", "brier", "nll")


def stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample participants with replacement within each observed class."""

    sampled = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        sampled.append(rng.choice(members, size=len(members), replace=True))
    return np.concatenate(sampled)


def compute_metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute threshold, discrimination, and calibration endpoints."""

    prediction = probability >= threshold
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    positive = labels == 1
    negative = ~positive
    sensitivity = float(prediction[positive].mean())
    specificity = float((~prediction[negative]).mean())
    ranks = rankdata(probability, method="average")
    num_positive = int(positive.sum())
    num_negative = int(negative.sum())
    auc = float(
        (ranks[positive].sum() - num_positive * (num_positive + 1) / 2)
        / (num_positive * num_negative)
    )
    assignments = np.minimum(np.floor(clipped * 10).astype(np.int64), 9)
    counts = np.bincount(assignments, minlength=10)
    probability_sums = np.bincount(assignments, weights=clipped, minlength=10)
    label_sums = np.bincount(assignments, weights=labels, minlength=10)
    occupied = counts > 0
    ece = float(
        np.sum(
            counts[occupied]
            / len(labels)
            * np.abs(
                probability_sums[occupied] / counts[occupied]
                - label_sums[occupied] / counts[occupied]
            )
        )
    )
    return {
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ece": ece,
        "brier": float(np.mean((probability - labels) ** 2)),
        "nll": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))),
    }


def load_runs(results_dir: Path) -> dict[str, dict[int, list[dict[str, object]]]]:
    """Load target predictions grouped by variant and seed."""

    grouped: dict[str, dict[int, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    paths = sorted(results_dir.glob("seed*/**/lodo_summary.json"))
    if not paths:
        raise FileNotFoundError(f"No lodo_summary.json files under {results_dir}")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        variant = path.parent.name
        seed = int(payload["seed"])
        for fold in payload["folds"]:
            metrics = fold["metrics"]
            grouped[variant][seed].append(
                {
                    "target": str(fold["target"]),
                    "labels": np.asarray(metrics["y_true"], dtype=np.int64),
                    "probability": np.asarray(metrics["y_prob"], dtype=np.float64),
                    "threshold": float(fold["decision_threshold"]),
                }
            )
    return grouped


def hierarchical_bootstrap(
    runs: dict[int, list[dict[str, object]]],
    num_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Bootstrap matched seeds and stratified target participants."""

    seeds = sorted(runs)
    targets = {str(fold["target"]) for fold in runs[seeds[0]]}
    if any({str(fold["target"]) for fold in runs[seed]} != targets for seed in seeds):
        raise ValueError("Target sets differ across seeds")

    point_values: dict[str, list[float]] = defaultdict(list)
    for seed in seeds:
        for fold in runs[seed]:
            values = compute_metrics(fold["labels"], fold["probability"], fold["threshold"])
            for metric, value in values.items():
                point_values[metric].append(value)
    point = {metric: float(np.mean(values)) for metric, values in point_values.items()}

    bootstrap = {metric: np.empty(num_bootstrap, dtype=np.float64) for metric in METRICS}
    for replicate in range(num_bootstrap):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        replicate_values: dict[str, list[float]] = defaultdict(list)
        for seed in sampled_seeds:
            for fold in runs[int(seed)]:
                labels = fold["labels"]
                probability = fold["probability"]
                indices = stratified_indices(labels, rng)
                values = compute_metrics(labels[indices], probability[indices], fold["threshold"])
                for metric, value in values.items():
                    replicate_values[metric].append(value)
        for metric in METRICS:
            bootstrap[metric][replicate] = float(np.mean(replicate_values[metric]))
    return point, bootstrap


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write interval rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run hierarchical bootstrap for every completed objective variant."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--num-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2608)
    args = parser.parse_args()
    if args.num_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates")

    grouped = load_runs(Path(args.results_dir))
    rows: list[dict[str, object]] = []
    for variant, runs in sorted(grouped.items()):
        rng = np.random.default_rng(args.bootstrap_seed)
        point, samples = hierarchical_bootstrap(runs, args.num_bootstrap, rng)
        targets = sorted({str(fold["target"]) for folds in runs.values() for fold in folds})
        participants = {
            str(fold["target"]): len(fold["labels"])
            for fold in next(iter(runs.values()))
        }
        for metric in METRICS:
            lower, upper = np.percentile(samples[metric], [2.5, 97.5])
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "point_estimate": point[metric],
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "num_seeds": len(runs),
                    "num_targets": len(targets),
                    "target_participants": json.dumps(participants, sort_keys=True),
                    "num_bootstrap": args.num_bootstrap,
                    "bootstrap_seed": args.bootstrap_seed,
                }
            )
        print(f"Bootstrapped {variant}: seeds={len(runs)} targets={len(targets)}")

    prefix = Path(args.output_prefix)
    write_csv(prefix.with_suffix(".csv"), rows)
    payload = {
        "metadata": {
            "results_dir": str(args.results_dir),
            "sampling": "seeds with replacement, then participants within class and target",
            "interval": "percentile 95%",
            "num_bootstrap": args.num_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "rows": rows,
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved bootstrap intervals to {prefix.parent}")


if __name__ == "__main__":
    main()
