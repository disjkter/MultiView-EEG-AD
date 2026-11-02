"""Paired inference for the overlap-controlled ADFSU sensitivity analysis."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy.stats import rankdata


METRICS = ("balanced_accuracy", "auc", "ece", "brier", "nll")


def compute_metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute participant-level discrimination, threshold, and calibration metrics."""

    prediction = probability >= threshold
    positive = labels == 1
    negative = ~positive
    sensitivity = float(prediction[positive].mean())
    specificity = float((~prediction[negative]).mean())
    ranks = rankdata(probability, method="average")
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    auc = float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    assignments = np.minimum(np.floor(clipped * 10).astype(np.int64), 9)
    counts = np.bincount(assignments, minlength=10)
    probability_sums = np.bincount(assignments, weights=clipped, minlength=10)
    label_sums = np.bincount(assignments, weights=labels, minlength=10)
    occupied = counts > 0
    ece = float(
        np.sum(
            counts[occupied]
            / labels.size
            * np.abs(
                probability_sums[occupied] / counts[occupied]
                - label_sums[occupied] / counts[occupied]
            )
        )
    )
    return {
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "auc": auc,
        "ece": ece,
        "brier": float(np.mean((probability - labels) ** 2)),
        "nll": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))),
    }


def exact_sign_flip(differences: list[float]) -> float:
    """Return a two-sided exact paired sign-flip p-value."""

    observed = abs(mean(differences))
    extreme = sum(
        abs(mean(sign * value for sign, value in zip(signs, differences)))
        >= observed - 1e-15
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    )
    return extreme / (2 ** len(differences))


def holm_adjust(raw_pvalues: dict[str, float]) -> dict[str, float]:
    """Apply Holm correction across the prespecified metric family."""

    ordered = sorted(raw_pvalues, key=raw_pvalues.get)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, metric in enumerate(ordered):
        value = min(1.0, (len(ordered) - rank) * raw_pvalues[metric])
        running_max = max(running_max, value)
        adjusted[metric] = running_max
    return adjusted


def load_runs(results_dir: Path, variants: tuple[str, str]) -> dict[str, dict[int, dict[str, object]]]:
    """Load matched participant predictions for two variants."""

    runs: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for path in sorted(results_dir.glob("seed*/**/lodo_summary.json")):
        variant = path.parent.name
        if variant not in variants:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if len(payload["folds"]) != 1 or payload["folds"][0]["target"] != "ADFSU":
            raise ValueError(f"Expected exactly one ADFSU fold in {path}")
        fold = payload["folds"][0]
        metrics = fold["metrics"]
        subjects = [str(value) for value in metrics["subject_ids"]]
        runs[variant][int(payload["seed"])] = {
            "subjects": subjects,
            "labels": np.asarray(metrics["y_true"], dtype=np.int64),
            "probability": np.asarray(metrics["y_prob"], dtype=np.float64),
            "threshold": float(fold["decision_threshold"]),
            "sources": tuple(fold["sources"]),
        }
    seeds = sorted(set(runs[variants[0]]) & set(runs[variants[1]]))
    if len(seeds) != 10:
        raise ValueError(f"Expected 10 matched seeds, found {len(seeds)}")
    for seed in seeds:
        left, right = runs[variants[0]][seed], runs[variants[1]][seed]
        if left["subjects"] != right["subjects"] or not np.array_equal(left["labels"], right["labels"]):
            raise ValueError(f"Participant alignment differs at seed {seed}")
        if "ADSZ" in left["sources"] or "ADSZ" in right["sources"]:
            raise ValueError(f"ADSZ source leakage detected at seed {seed}")
    return runs


def paired_bootstrap(
    runs: dict[str, dict[int, dict[str, object]]],
    variants: tuple[str, str],
    num_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Resample matched seeds and class-stratified participants."""

    rng = np.random.default_rng(seed)
    seeds = sorted(runs[variants[0]])
    samples = {metric: np.empty(num_bootstrap) for metric in METRICS}
    for replicate in range(num_bootstrap):
        values: dict[str, list[float]] = defaultdict(list)
        for sampled_seed in rng.choice(seeds, size=len(seeds), replace=True):
            first = runs[variants[0]][int(sampled_seed)]
            labels = first["labels"]
            indices = np.concatenate(
                [
                    rng.choice(np.flatnonzero(labels == label), size=int(np.sum(labels == label)), replace=True)
                    for label in np.unique(labels)
                ]
            )
            method_values = {}
            for variant in variants:
                run = runs[variant][int(sampled_seed)]
                method_values[variant] = compute_metrics(
                    run["labels"][indices], run["probability"][indices], run["threshold"]
                )
            for metric in METRICS:
                values[metric].append(method_values[variants[0]][metric] - method_values[variants[1]][metric])
        for metric in METRICS:
            samples[metric][replicate] = mean(values[metric])
    return samples


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a stable CSV table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run paired inference and save paper-audit artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--num-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2608)
    args = parser.parse_args()
    variants = ("tacd", "erm")
    runs = load_runs(args.results_dir, variants)
    seed_differences: dict[str, list[float]] = defaultdict(list)
    for seed in sorted(runs[variants[0]]):
        per_variant = {
            variant: compute_metrics(
                runs[variant][seed]["labels"],
                runs[variant][seed]["probability"],
                runs[variant][seed]["threshold"],
            )
            for variant in variants
        }
        for metric in METRICS:
            seed_differences[metric].append(per_variant[variants[0]][metric] - per_variant[variants[1]][metric])
    raw_pvalues = {metric: exact_sign_flip(seed_differences[metric]) for metric in METRICS}
    adjusted = holm_adjust(raw_pvalues)
    bootstrap = paired_bootstrap(runs, variants, args.num_bootstrap, args.bootstrap_seed)
    rows = []
    for metric in METRICS:
        lower, upper = np.percentile(bootstrap[metric], [2.5, 97.5])
        differences = seed_differences[metric]
        rows.append(
            {
                "reference": variants[0],
                "baseline": variants[1],
                "metric": metric,
                "num_pairs": len(differences),
                "mean_difference": mean(differences),
                "difference_std": stdev(differences),
                "wins": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "exact_sign_flip_pvalue": raw_pvalues[metric],
                "holm_adjusted_pvalue": adjusted[metric],
                "paired_bootstrap_ci_lower": float(lower),
                "paired_bootstrap_ci_upper": float(upper),
            }
        )
    prefix = args.output_prefix
    write_csv(prefix.with_suffix(".csv"), rows)
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "metadata": {
                    "analysis": "overlap-controlled retrospective sensitivity analysis",
                    "difference": "tacd minus erm",
                    "calibration_metric_direction": "negative favors tacd",
                    "target_data_used_during_training": False,
                    "adsz_used_as_source": False,
                    "bootstrap": "matched seeds, then matched class-stratified participants",
                    "num_bootstrap": args.num_bootstrap,
                    "bootstrap_seed": args.bootstrap_seed,
                },
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    for row in rows:
        print(
            f"{row['metric']}: delta={row['mean_difference']:+.4f} "
            f"95% CI [{row['paired_bootstrap_ci_lower']:+.4f}, "
            f"{row['paired_bootstrap_ci_upper']:+.4f}] "
            f"Holm p={row['holm_adjusted_pvalue']:.4f}"
        )


if __name__ == "__main__":
    main()
