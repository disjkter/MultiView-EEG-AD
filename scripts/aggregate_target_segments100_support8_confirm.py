"""Aggregate the paired ten-seed support-8 segment-budget confirmation."""

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


def main() -> None:
    """Combine pilot and confirmation seeds and compute paired statistics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/target_segments100_support8_confirm.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    experiment = config["experiment"]
    root = Path(experiment["output_root"])
    pilot_root = Path(experiment["pilot_root"])
    reference_root = Path(experiment["reference_root"])
    pilot_seeds = {int(value) for value in experiment["completed_pilot_seeds"]}
    seeds = [int(value) for value in experiment["all_seeds"]]
    support = int(experiment["support_per_class"])
    rows: list[dict[str, object]] = []
    for seed in seeds:
        candidate_path = (
            pilot_root / f"support_{support}" / "segments_100" / f"seed_{seed}" / "summary.json"
            if seed in pilot_seeds
            else root / f"seed_{seed}" / "summary.json"
        )
        paths = {
            "segments_20": reference_root / f"support_{support}" / "ssl_only" / f"seed_{seed}" / "summary.json",
            "segments_100": candidate_path,
        }
        for condition, path in paths.items():
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "validation_auc": payload.get("best_val_auc"),
                    **{metric: float(metrics[metric]) for metric in METRICS},
                }
            )
    runs = pd.DataFrame(rows)
    summary = runs.groupby("condition")[["validation_auc", *METRICS]].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part) if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    baseline = runs[runs["condition"] == "segments_20"].set_index("seed").sort_index()
    candidate = runs[runs["condition"] == "segments_100"].set_index("seed").sort_index()
    comparisons: list[dict[str, object]] = []
    for index, metric in enumerate(METRICS):
        differences = candidate[metric].to_numpy() - baseline[metric].to_numpy()
        low, high = bootstrap_ci(differences, samples=100_000, seed=20261004 + index)
        comparisons.append(
            {
                "metric": metric,
                "segments_20_mean": float(baseline[metric].mean()),
                "segments_100_mean": float(candidate[metric].mean()),
                "mean_difference": float(differences.mean()),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
                "positive_seeds": int(np.sum(differences > 0)),
                "negative_seeds": int(np.sum(differences < 0)),
            }
        )
    result = {
        "support_per_class": support,
        "seeds": seeds,
        "summary": summary.to_dict(orient="records"),
        "paired_comparisons": comparisons,
    }
    runs.to_csv(root / "aggregate_runs.csv", index=False)
    summary.to_csv(root / "aggregate_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(root / "aggregate_comparisons.csv", index=False)
    (root / "aggregate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()

