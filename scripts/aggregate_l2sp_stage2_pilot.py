"""Aggregate the stage-2 regularization pilot and select by validation AUROC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def read_summary(path: Path) -> dict[str, object]:
    """Read a completed experiment summary."""

    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Combine reference and candidate runs without selecting on test AUROC."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/l2sp_stage2_pilot.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    experiment = config["experiment"]
    output_root = Path(experiment["output_root"])
    reference_root = Path(experiment["reference_root"])
    seeds = [int(value) for value in experiment["seeds"]]
    supports = [int(value) for value in experiment["supports_per_class"]]
    candidate_names = [str(item["name"]) for item in config["initialization"]["stage2_variants"]]
    rows: list[dict[str, object]] = []

    for support in supports:
        for seed in seeds:
            reference_paths = {
                "ssl_only": reference_root / f"support_{support}" / "ssl_only" / f"seed_{seed}" / "summary.json",
                "original_staged": reference_root / f"support_{support}" / "staged_transfer" / f"seed_{seed}" / "summary.json",
            }
            candidate_paths = {
                name: output_root / "target" / f"support_{support}" / name / f"seed_{seed}" / "summary.json"
                for name in candidate_names
            }
            for condition, path in {**reference_paths, **candidate_paths}.items():
                payload = read_summary(path)
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
    summary = (
        runs.groupby(["condition", "support_per_class"])[
            ["validation_auc", "auc", "balanced_accuracy", "f1"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part) if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    validation_ranking = (
        runs[runs["condition"].isin(candidate_names)]
        .groupby("condition")["validation_auc"]
        .mean()
        .sort_values(ascending=False)
    )
    selected = str(validation_ranking.index[0])
    result = {
        "selection_rule": "Highest mean target-validation AUROC across the prespecified pilot supports and seeds.",
        "selected_candidate": selected,
        "validation_ranking": {key: float(value) for key, value in validation_ranking.items()},
        "summary": summary.to_dict(orient="records"),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_root / "aggregate_runs.csv", index=False)
    summary.to_csv(output_root / "aggregate_summary.csv", index=False)
    (output_root / "aggregate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Selected candidate by validation AUROC: {selected}")


if __name__ == "__main__":
    main()

