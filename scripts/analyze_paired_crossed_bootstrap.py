"""Participant-preserving paired crossed bootstrap for the development LODO matrix."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


METRICS = (
    "source_threshold_balanced_accuracy",
    "fixed_balanced_accuracy",
    "cohort_z_balanced_accuracy",
    "auc",
    "ece",
    "adaptive_ece",
    "brier",
    "nll",
)
COMPARISONS = (
    ("adformer_erm", "erm"),
    ("dann", "erm"),
    ("coral", "erm"),
    ("mmd", "erm"),
    ("tacd", "erm"),
    ("erm_margin", "erm"),
    ("tacd_margin", "tacd"),
    ("tacd_margin", "erm_margin"),
    ("erm_nested_matched", "erm"),
    ("tacd_margin_nested_matched", "tacd_margin"),
    ("tacd_margin_nested_matched", "erm_nested_matched"),
)


def participant_keys(metrics: dict[str, object], target: str, count: int) -> tuple[tuple[str, ...], str]:
    """Return serialized IDs or stable anonymous keys for legacy summaries."""

    serialized = metrics.get("subject_ids")
    if serialized is not None:
        subjects = tuple(str(value) for value in serialized)
        if len(subjects) != count:
            raise ValueError(f"subject_ids length mismatch for {target}")
        return subjects, "serialized"
    return (
        tuple(f"{target}::position_{index:04d}" for index in range(count)),
        "deterministic_sorted_position",
    )


def balanced_accuracy(labels: np.ndarray, prediction: np.ndarray) -> float:
    """Return binary balanced accuracy."""

    positive = labels == 1
    negative = ~positive
    return float((prediction[positive].mean() + (~prediction[negative]).mean()) / 2.0)


def fixed_ece(labels: np.ndarray, probability: np.ndarray, num_bins: int = 10) -> float:
    """Return equal-width expected calibration error."""

    assignments = np.minimum(np.floor(probability * num_bins).astype(np.int64), num_bins - 1)
    total = labels.size
    value = 0.0
    for bin_index in range(num_bins):
        mask = assignments == bin_index
        if mask.any():
            value += float(mask.mean()) * abs(float(probability[mask].mean() - labels[mask].mean()))
    return value


def adaptive_ece(labels: np.ndarray, probability: np.ndarray, num_bins: int = 10) -> float:
    """Return equal-count ECE using deterministic probability ordering."""

    ordered = np.argsort(probability, kind="stable")
    value = 0.0
    for indices in np.array_split(ordered, min(num_bins, labels.size)):
        if indices.size:
            value += indices.size / labels.size * abs(
                float(probability[indices].mean() - labels[indices].mean())
            )
    return value


def compute_metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute all participant-level endpoints from one target prediction vector."""

    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped / (1.0 - clipped))
    logit_std = float(logits.std())
    cohort_prediction = logits >= float(logits.mean()) if logit_std > 0 else np.zeros_like(labels, dtype=bool)
    positive = labels == 1
    negative = ~positive
    ranks = rankdata(probability, method="average")
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    auc = float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )
    return {
        "source_threshold_balanced_accuracy": balanced_accuracy(labels, probability >= threshold),
        "fixed_balanced_accuracy": balanced_accuracy(labels, probability >= 0.5),
        "cohort_z_balanced_accuracy": balanced_accuracy(labels, cohort_prediction),
        "auc": auc,
        "ece": fixed_ece(labels, probability),
        "adaptive_ece": adaptive_ece(labels, probability),
        "brier": float(np.mean((probability - labels) ** 2)),
        "nll": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))),
    }


def load_runs(results_dirs: list[Path]) -> dict[str, dict[int, dict[str, dict[str, object]]]]:
    """Load and validate all method, seed, target participant predictions."""

    runs: dict[str, dict[int, dict[str, dict[str, object]]]] = defaultdict(lambda: defaultdict(dict))
    paths = sorted(
        path
        for results_dir in results_dirs
        for path in results_dir.glob("seed*/**/lodo_summary.json")
    )
    for path in paths:
        variant = path.parent.name
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        for fold in payload["folds"]:
            if any(
                bool(fold.get(field, False))
                for field in (
                    "target_data_used_during_training",
                    "target_data_used_for_model_selection",
                    "target_data_used_for_threshold_selection",
                )
            ):
                raise ValueError(f"Target-use protocol violation in {path}")
            metrics = fold["metrics"]
            target = str(fold["target"])
            labels = np.asarray(metrics["y_true"], dtype=np.int64)
            subjects, subject_id_source = participant_keys(metrics, target, labels.size)
            if target in runs[variant][seed]:
                raise ValueError(f"Duplicate run for {variant}, seed {seed}, target {target}")
            runs[variant][seed][target] = {
                "subjects": subjects,
                "subject_id_source": subject_id_source,
                "labels": labels,
                "probability": np.asarray(metrics["y_prob"], dtype=np.float64),
                "threshold": float(fold["decision_threshold"]),
            }
    variants = sorted(runs)
    if not variants:
        raise FileNotFoundError(f"No lodo_summary.json files under {results_dirs}")
    reference = runs[variants[0]]
    if len(reference) != 10:
        raise ValueError(f"Expected 10 seeds, found {len(reference)}")
    for variant in variants:
        if set(runs[variant]) != set(reference):
            raise ValueError(f"Seed mismatch for {variant}")
        for seed in reference:
            if set(runs[variant][seed]) != set(reference[seed]):
                raise ValueError(f"Target mismatch for {variant}, seed {seed}")
            for target in reference[seed]:
                first = reference[seed][target]
                current = runs[variant][seed][target]
                if len(first["subjects"]) != len(current["subjects"]) or not np.array_equal(first["labels"], current["labels"]):
                    raise ValueError(f"Participant mismatch for {variant}, seed {seed}, target {target}")
                # Legacy summaries omit ID strings. Exact label-vector agreement
                # plus deterministic evaluator sorting establishes a common
                # target-position key across old and new summaries.
                canonical = tuple(
                    f"{target}::position_{index:04d}"
                    for index in range(len(current["subjects"]))
                )
                current["subjects"] = canonical
                first["subjects"] = canonical
    return runs


def point_estimates(
    runs: dict[str, dict[int, dict[str, dict[str, object]]]]
) -> dict[str, dict[str, float]]:
    """Macro-average endpoints over matched seeds and targets."""

    output: dict[str, dict[str, float]] = {}
    for variant, seed_runs in runs.items():
        values: dict[str, list[float]] = defaultdict(list)
        for target_runs in seed_runs.values():
            for run in target_runs.values():
                metrics = compute_metrics(run["labels"], run["probability"], run["threshold"])
                for metric, value in metrics.items():
                    values[metric].append(value)
        output[variant] = {metric: float(np.mean(metric_values)) for metric, metric_values in values.items()}
    return output


def bootstrap_differences(
    runs: dict[str, dict[int, dict[str, dict[str, object]]]],
    comparisons: tuple[tuple[str, str], ...],
    num_bootstrap: int,
    random_seed: int,
) -> dict[tuple[str, str, str], np.ndarray]:
    """Resample crossed seeds and target participants with shared paired indices."""

    rng = np.random.default_rng(random_seed)
    variants = sorted(runs)
    seeds = sorted(next(iter(runs.values())))
    targets = sorted(next(iter(next(iter(runs.values())).values())))
    output = {
        (reference, baseline, metric): np.empty(num_bootstrap, dtype=np.float64)
        for reference, baseline in comparisons
        for metric in METRICS
    }
    template = next(iter(runs.values()))[seeds[0]]
    for replicate in range(num_bootstrap):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_participants = {}
        for target in targets:
            labels = template[target]["labels"]
            sampled_participants[target] = np.concatenate(
                [
                    rng.choice(np.flatnonzero(labels == label), size=int(np.sum(labels == label)), replace=True)
                    for label in np.unique(labels)
                ]
            )
        method_values: dict[str, dict[str, list[float]]] = {
            variant: defaultdict(list) for variant in variants
        }
        for sampled_seed in sampled_seeds:
            for target in targets:
                indices = sampled_participants[target]
                for variant in variants:
                    run = runs[variant][int(sampled_seed)][target]
                    metrics = compute_metrics(
                        run["labels"][indices], run["probability"][indices], run["threshold"]
                    )
                    for metric, value in metrics.items():
                        method_values[variant][metric].append(value)
        means = {
            variant: {metric: float(np.mean(values)) for metric, values in metrics.items()}
            for variant, metrics in method_values.items()
        }
        for reference, baseline in comparisons:
            for metric in METRICS:
                output[(reference, baseline, metric)][replicate] = (
                    means[reference][metric] - means[baseline][metric]
                )
    return output


def occupancy_rows(
    runs: dict[str, dict[int, dict[str, dict[str, object]]]], num_bins: int = 10
) -> list[dict[str, object]]:
    """Return fixed-bin participant occupancy for every method, seed, and target."""

    rows = []
    for variant, seed_runs in sorted(runs.items()):
        for seed, target_runs in sorted(seed_runs.items()):
            for target, run in sorted(target_runs.items()):
                probability = run["probability"]
                assignments = np.minimum(np.floor(probability * num_bins).astype(np.int64), num_bins - 1)
                for bin_index in range(num_bins):
                    mask = assignments == bin_index
                    rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "target": target,
                            "bin": bin_index,
                            "lower": bin_index / num_bins,
                            "upper": (bin_index + 1) / num_bins,
                            "count": int(mask.sum()),
                            "fraction": float(mask.mean()),
                            "mean_probability": float(probability[mask].mean()) if mask.any() else None,
                            "positive_fraction": float(run["labels"][mask].mean()) if mask.any() else None,
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the paired crossed bootstrap and calibration robustness audit."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2608)
    args = parser.parse_args()
    runs = load_runs(args.results_dir)
    estimates = point_estimates(runs)
    comparisons = tuple(pair for pair in COMPARISONS if pair[0] in runs and pair[1] in runs)
    samples = bootstrap_differences(runs, comparisons, args.num_bootstrap, args.bootstrap_seed)
    rows = []
    for reference, baseline in comparisons:
        for metric in METRICS:
            difference = estimates[reference][metric] - estimates[baseline][metric]
            lower, upper = np.percentile(samples[(reference, baseline, metric)], [2.5, 97.5])
            rows.append(
                {
                    "reference": reference,
                    "baseline": baseline,
                    "metric": metric,
                    "reference_estimate": estimates[reference][metric],
                    "baseline_estimate": estimates[baseline][metric],
                    "mean_difference": difference,
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "num_seeds": len(runs[reference]),
                    "num_targets": len(next(iter(runs[reference].values()))),
                    "num_bootstrap": args.num_bootstrap,
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paired_crossed_bootstrap.csv", rows)
    write_csv(args.output_dir / "ece_fixed_bin_occupancy.csv", occupancy_rows(runs))
    payload = {
        "metadata": {
            "results_dirs": [str(path) for path in args.results_dir],
            "sampling": "shared seed indices; shared participant indices within target and class; cohorts fixed",
            "difference": "reference minus baseline",
            "calibration_metric_direction": "negative favors reference",
            "cohort_z": "transductive diagnostic control using unlabeled target score distribution",
            "num_bootstrap": args.num_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "participant_key_note": (
                "Older summaries omit subject_ids. For those files, target-specific positions are "
                "used as anonymous participant keys after exact label-vector alignment across all "
                "methods and seeds; the training evaluator deterministically sorts subject IDs."
            ),
        },
        "point_estimates": estimates,
        "comparisons": rows,
    }
    (args.output_dir / "paired_crossed_bootstrap.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Validated {len(runs)} methods and {sum(len(value) for value in runs.values())} runs")
    print(f"Saved {len(rows)} paired intervals to {args.output_dir}")


if __name__ == "__main__":
    main()
