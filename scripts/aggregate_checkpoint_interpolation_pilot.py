"""Aggregate a checkpoint-interpolation pilot and select by validation AUROC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_checkpoint_interpolation_pilot import weight_name


def main() -> None:
    """Compare interpolation candidates with direct SSL and source transfer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/checkpoint_interpolation_pilot.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    experiment = config["experiment"]
    root = Path(experiment["output_root"])
    reference = Path(experiment["reference_root"])
    seeds = [int(value) for value in experiment["seeds"]]
    supports = [int(value) for value in experiment["supports_per_class"]]
    candidates = [weight_name(float(value)) for value in experiment["interpolation_weights"]]
    rows: list[dict[str, object]] = []
    for support in supports:
        for seed in seeds:
            paths = {
                "ssl_only": reference / f"support_{support}" / "ssl_only" / f"seed_{seed}" / "summary.json",
                "original_staged": reference / f"support_{support}" / "staged_transfer" / f"seed_{seed}" / "summary.json",
                **{
                    name: root / "target" / f"support_{support}" / name / f"seed_{seed}" / "summary.json"
                    for name in candidates
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
    ranking = (
        runs[runs["condition"].isin(candidates)]
        .groupby("condition")["validation_auc"]
        .mean()
        .sort_values(ascending=False)
    )
    result = {
        "selection_rule": "Highest mean target-validation AUROC across prespecified supports and seeds.",
        "selected_candidate": str(ranking.index[0]),
        "validation_ranking": {key: float(value) for key, value in ranking.items()},
        "summary": summary.to_dict(orient="records"),
    }
    runs.to_csv(root / "aggregate_runs.csv", index=False)
    summary.to_csv(root / "aggregate_summary.csv", index=False)
    (root / "aggregate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Selected candidate by validation AUROC: {ranking.index[0]}")


if __name__ == "__main__":
    main()

