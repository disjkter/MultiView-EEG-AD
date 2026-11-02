"""Aggregate paired multi-seed few-shot transfer comparisons."""

from __future__ import annotations

import argparse
import json
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "accuracy",
    "balanced_accuracy",
    "f1",
    "auc",
    "sensitivity",
    "specificity",
)

DEFAULT_VARIANTS = (
    "target_only",
    "CAUEEG",
    "CAUEEG_encoders",
    "GraphMAE",
    "target_only_calibrated",
    "CAUEEG_encoders_calibrated",
    "GraphMAE_calibrated",
)


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Return an exact two-sided paired sign-flip p-value."""

    differences = np.asarray(differences, dtype=np.float64)
    observed = abs(float(differences.mean()))
    null_means = np.asarray(
        [
            np.mean(differences * np.asarray(signs, dtype=np.float64))
            for signs in product((-1.0, 1.0), repeat=len(differences))
        ]
    )
    return float(np.mean(np.abs(null_means) >= observed - 1e-12))


def bootstrap_ci(
    differences: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Return a paired percentile-bootstrap confidence interval."""

    differences = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(differences),
        size=(samples, len(differences)),
    )
    means = differences[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def load_runs(
    root: Path,
    target: str,
    seeds: list[int],
    variants: list[str],
) -> pd.DataFrame:
    """Load all expected summary files into a paired run table."""

    rows: list[dict[str, object]] = []
    for seed in seeds:
        for variant in variants:
            path = (
                root
                / f"support4_seed{seed}"
                / target
                / variant
                / "summary.json"
            )
            if not path.exists():
                raise FileNotFoundError(f"Missing summary: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "best_epoch": payload.get("best_epoch"),
                    "decision_threshold": payload.get("decision_threshold"),
                    **{metric: float(metrics[metric]) for metric in METRICS},
                }
            )
    return pd.DataFrame(rows)


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and sample standard deviation for each method."""

    aggregate = runs.groupby("variant", sort=False)[list(METRICS)].agg(
        ["mean", "std", "min", "max"]
    )
    aggregate.columns = [
        f"{metric}_{statistic}" for metric, statistic in aggregate.columns
    ]
    return aggregate.reset_index()


def compare_variants(
    runs: pd.DataFrame,
    variants: list[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Run paired tests for every method pair and metric."""

    indexed = {
        variant: runs[runs["variant"] == variant].sort_values("seed").set_index("seed")
        for variant in variants
    }
    rows: list[dict[str, object]] = []
    for left, right in combinations(variants, 2):
        if not indexed[left].index.equals(indexed[right].index):
            raise ValueError(f"Seed mismatch between {left} and {right}")
        for metric_index, metric in enumerate(METRICS):
            differences = (
                indexed[left][metric].to_numpy()
                - indexed[right][metric].to_numpy()
            )
            ci_low, ci_high = bootstrap_ci(
                differences,
                samples=bootstrap_samples,
                seed=bootstrap_seed + metric_index,
            )
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "left_mean": float(indexed[left][metric].mean()),
                    "right_mean": float(indexed[right][metric].mean()),
                    "mean_difference": float(differences.mean()),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
                    "positive_seeds": int(np.sum(differences > 0)),
                    "negative_seeds": int(np.sum(differences < 0)),
                    "ties": int(np.sum(differences == 0)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Aggregate the complete transfer comparison matrix."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--target", default="ds004504")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(2026, 2036)),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    runs = load_runs(
        Path(args.root),
        args.target,
        args.seeds,
        args.variants,
    )
    aggregate = aggregate_runs(runs)
    comparisons = compare_variants(
        runs,
        args.variants,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_prefix.with_name(output_prefix.name + "_runs.csv"), index=False)
    aggregate.to_csv(
        output_prefix.with_name(output_prefix.name + "_aggregate.csv"),
        index=False,
    )
    comparisons.to_csv(
        output_prefix.with_name(output_prefix.name + "_comparisons.csv"),
        index=False,
    )
    payload = {
        "target": args.target,
        "seeds": args.seeds,
        "variants": args.variants,
        "aggregate": aggregate.to_dict(orient="records"),
        "comparisons": comparisons.to_dict(orient="records"),
    }
    output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))
    print(f"Saved comparison outputs with prefix: {output_prefix}")


if __name__ == "__main__":
    main()
