"""Aggregate matched random, SSL-only, and staged-transfer support curves."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_transfer_comparison import bootstrap_ci, exact_sign_flip_pvalue

METRICS = ("accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity")
COMPARISONS = (
    ("ssl_only", "random_init"),
    ("staged_transfer", "ssl_only"),
    ("staged_transfer", "random_init"),
)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values while preserving input order."""

    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def main() -> None:
    """Load completed summaries and save tables plus paired statistics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/staged_ds004504_to_caueeg.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    experiment = config["experiment"]
    root = Path(experiment["output_root"])
    seeds = [int(seed) for seed in experiment["seeds"]]
    supports = [int(support) for support in experiment["supports_per_class"]]
    conditions = [str(value) for value in config["initialization"]["conditions"]]

    rows: list[dict[str, object]] = []
    for support in supports:
        for condition in conditions:
            for seed in seeds:
                path = root / f"support_{support}" / condition / f"seed_{seed}" / "summary.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                metrics = payload["metrics"]
                rows.append(
                    {
                        "support_per_class": support,
                        "condition": condition,
                        "seed": seed,
                        **{metric: float(metrics[metric]) for metric in METRICS},
                    }
                )
    runs = pd.DataFrame(rows)
    aggregate = runs.groupby(["support_per_class", "condition"])[list(METRICS)].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate = aggregate.reset_index()

    comparison_rows: list[dict[str, object]] = []
    for support in supports:
        support_runs = runs[runs["support_per_class"] == support]
        for better, reference in COMPARISONS:
            better_runs = support_runs[support_runs["condition"] == better].set_index("seed").sort_index()
            reference_runs = support_runs[support_runs["condition"] == reference].set_index("seed").sort_index()
            if not better_runs.index.equals(reference_runs.index):
                raise ValueError(f"Seed mismatch: support={support}, {better} vs {reference}")
            for metric_index, metric in enumerate(METRICS):
                differences = better_runs[metric].to_numpy() - reference_runs[metric].to_numpy()
                low, high = bootstrap_ci(
                    differences,
                    samples=100_000,
                    seed=20261002 + support * 100 + metric_index,
                )
                comparison_rows.append(
                    {
                        "support_per_class": support,
                        "comparison": f"{better}_minus_{reference}",
                        "metric": metric,
                        "reference_mean": float(reference_runs[metric].mean()),
                        "candidate_mean": float(better_runs[metric].mean()),
                        "mean_difference": float(differences.mean()),
                        "bootstrap_ci95_low": low,
                        "bootstrap_ci95_high": high,
                        "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
                        "positive_seeds": int(np.sum(differences > 0)),
                    }
                )
    comparisons = pd.DataFrame(comparison_rows)
    for comparison in comparisons["comparison"].unique():
        mask = (comparisons["comparison"] == comparison) & (comparisons["metric"] == "auc")
        comparisons.loc[mask, "auc_holm_p"] = holm_adjust(
            comparisons.loc[mask, "exact_sign_flip_p"].tolist()
        )

    prefix = root / "aggregate"
    runs.to_csv(prefix.with_name("aggregate_runs.csv"), index=False)
    aggregate.to_csv(prefix.with_name("aggregate_summary.csv"), index=False)
    comparisons.to_csv(prefix.with_name("aggregate_comparisons.csv"), index=False)
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": "ds004504",
                "target": "CAUEEG",
                "supports_per_class": supports,
                "seeds": seeds,
                "aggregate": aggregate.to_dict(orient="records"),
                "comparisons": comparisons.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))
    print(f"Saved staged-transfer aggregate: {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()

