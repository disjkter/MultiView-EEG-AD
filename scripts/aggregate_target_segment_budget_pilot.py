"""Aggregate the target-stage segment-budget pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    """Select a segment budget using target-validation AUROC."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/target_segment_budget_pilot.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    experiment = config["experiment"]
    root = Path(experiment["output_root"])
    reference = Path(experiment["reference_root"])
    seeds = [int(value) for value in experiment["seeds"]]
    supports = [int(value) for value in experiment["supports_per_class"]]
    candidates = [f"segments_{int(value)}" for value in experiment["train_segments_per_subject"]]
    rows: list[dict[str, object]] = []
    for support in supports:
        for seed in seeds:
            paths = {
                "segments_20": reference / f"support_{support}" / "ssl_only" / f"seed_{seed}" / "summary.json",
                **{
                    condition: root / f"support_{support}" / condition / f"seed_{seed}" / "summary.json"
                    for condition in candidates
                },
            }
            for condition, path in paths.items():
                payload = json.loads(path.read_text(encoding="utf-8"))
                metrics = payload["metrics"]
                rows.append(
                    {
                        "support_per_class": support,
                        "seed": seed,
                        "condition": condition,
                        "validation_auc": payload.get("best_val_auc"),
                        "auc": metrics["auc"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "f1": metrics["f1"],
                    }
                )
    runs = pd.DataFrame(rows)
    summary = runs.groupby(["condition", "support_per_class"])[
        ["validation_auc", "auc", "balanced_accuracy", "f1"]
    ].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part) if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    ranking = runs.groupby("condition")["validation_auc"].mean().sort_values(ascending=False)
    result = {
        "selection_rule": "Highest mean target-validation AUROC across prespecified supports and seeds.",
        "selected_condition": str(ranking.index[0]),
        "validation_ranking": {key: float(value) for key, value in ranking.items()},
        "summary": summary.to_dict(orient="records"),
    }
    runs.to_csv(root / "aggregate_runs.csv", index=False)
    summary.to_csv(root / "aggregate_summary.csv", index=False)
    (root / "aggregate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Selected segment budget: {ranking.index[0]}")


if __name__ == "__main__":
    main()

