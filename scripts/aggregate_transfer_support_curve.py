"""Aggregate a paired multi-seed few-shot transfer support curve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_transfer_comparison import (
    METRICS,
    bootstrap_ci,
    exact_sign_flip_pvalue,
)


def load_summary(path: Path) -> dict[str, object]:
    """Load one subject-level experiment summary."""

    if not path.exists():
        raise FileNotFoundError(f"Missing summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
        raise ValueError(f"Invalid summary: {path}")
    return payload


def load_runs(
    curve_root: Path,
    support4_root: Path,
    target: str,
    supports: list[int],
    seeds: list[int],
    source_name: str,
    support4_target_variant: str,
    support4_source_variant: str,
) -> pd.DataFrame:
    """Load paired target-only and source-transfer runs at each support size."""

    rows: list[dict[str, object]] = []
    for support in supports:
        for seed in seeds:
            if support == 4:
                paths = {
                    "target_only": (
                        support4_root
                        / f"support4_seed{seed}"
                        / target
                        / support4_target_variant
                        / "summary.json"
                    ),
                    "source_transfer": (
                        support4_root
                        / f"support4_seed{seed}"
                        / target
                        / support4_source_variant
                        / "summary.json"
                    ),
                }
            else:
                run_root = curve_root / f"support{support}_seed{seed}" / target
                paths = {
                    "target_only": run_root / "target_only" / "summary.json",
                    "source_transfer": run_root / source_name / "summary.json",
                }
            for initialization, path in paths.items():
                payload = load_summary(path)
                metrics = payload["metrics"]
                rows.append(
                    {
                        "support_per_class": support,
                        "seed": seed,
                        "initialization": initialization,
                        "decision_threshold": payload.get("decision_threshold"),
                        **{metric: float(metrics[metric]) for metric in METRICS},
                    }
                )
    return pd.DataFrame(rows)


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for each support and initialization."""

    aggregate = runs.groupby(
        ["support_per_class", "initialization"], sort=True
    )[list(METRICS)].agg(["mean", "std", "min", "max"])
    aggregate.columns = [
        f"{metric}_{statistic}" for metric, statistic in aggregate.columns
    ]
    return aggregate.reset_index()


def compare_paired_runs(
    runs: pd.DataFrame,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Compare source transfer against target-only within every support size."""

    rows: list[dict[str, object]] = []
    for support in sorted(runs["support_per_class"].unique()):
        support_runs = runs[runs["support_per_class"] == support]
        target = (
            support_runs[support_runs["initialization"] == "target_only"]
            .sort_values("seed")
            .set_index("seed")
        )
        transfer = (
            support_runs[support_runs["initialization"] == "source_transfer"]
            .sort_values("seed")
            .set_index("seed")
        )
        if not target.index.equals(transfer.index):
            raise ValueError(f"Seed mismatch at support={support}")
        for metric_index, metric in enumerate(METRICS):
            differences = transfer[metric].to_numpy() - target[metric].to_numpy()
            ci_low, ci_high = bootstrap_ci(
                differences,
                samples=bootstrap_samples,
                seed=bootstrap_seed + int(support) * len(METRICS) + metric_index,
            )
            rows.append(
                {
                    "support_per_class": int(support),
                    "metric": metric,
                    "target_only_mean": float(target[metric].mean()),
                    "source_transfer_mean": float(transfer[metric].mean()),
                    "transfer_gain": float(differences.mean()),
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
    """Aggregate and save the complete support-size curve."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-root", required=True)
    parser.add_argument("--support4-root", required=True)
    parser.add_argument("--target", default="CAUEEG")
    parser.add_argument("--source-name", default="ADFSU")
    parser.add_argument("--supports", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(range(2026, 2036))
    )
    parser.add_argument(
        "--support4-target-variant", default="target_only_calibrated"
    )
    parser.add_argument(
        "--support4-source-variant", default="ADFSU_full_finetune_calibrated"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    runs = load_runs(
        Path(args.curve_root),
        Path(args.support4_root),
        args.target,
        args.supports,
        args.seeds,
        args.source_name,
        args.support4_target_variant,
        args.support4_source_variant,
    )
    aggregate = aggregate_runs(runs)
    comparisons = compare_paired_runs(
        runs, args.bootstrap_samples, args.bootstrap_seed
    )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_prefix.with_name(output_prefix.name + "_runs.csv"), index=False)
    aggregate.to_csv(
        output_prefix.with_name(output_prefix.name + "_aggregate.csv"), index=False
    )
    comparisons.to_csv(
        output_prefix.with_name(output_prefix.name + "_comparisons.csv"), index=False
    )
    output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "target": args.target,
                "source": args.source_name,
                "supports": args.supports,
                "seeds": args.seeds,
                "aggregate": aggregate.to_dict(orient="records"),
                "comparisons": comparisons.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))
    print(f"Saved support-curve outputs with prefix: {output_prefix}")


if __name__ == "__main__":
    main()
