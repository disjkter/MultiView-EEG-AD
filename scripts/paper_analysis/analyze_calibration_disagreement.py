"""Analyze disagreement among source-fitted calibration metrics under shift."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper_analysis.common import parse_named_path, write_csv, write_json


def main() -> None:
    """Compare each calibrator with matched identity results."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", action="append", required=True, metavar="EXPERIMENT=CSV")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    raw: list[dict[str, str]] = []
    for value in args.folds:
        experiment, path = parse_named_path(value)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                row["experiment"] = experiment
                raw.append(row)
    identity = {
        (row["experiment"], row["backbone"], row["variant"], row["seed"], row["target"]): row
        for row in raw if row["calibrator"] == "identity"
    }
    metrics = ("ece", "brier", "nll")
    deltas: list[dict[str, object]] = []
    for row in raw:
        if row["calibrator"] == "identity":
            continue
        key = (row["experiment"], row["backbone"], row["variant"], row["seed"], row["target"])
        reference = identity.get(key)
        if reference is None:
            continue
        item: dict[str, object] = {
            "experiment": row["experiment"], "backbone": row["backbone"],
            "variant": row["variant"], "seed": int(row["seed"]),
            "target": row["target"], "calibrator": row["calibrator"],
        }
        for metric in metrics:
            item[f"delta_{metric}"] = float(row[metric]) - float(reference[metric])
        signs = [float(item[f"delta_{metric}"]) < 0 for metric in metrics]
        item["metrics_disagree"] = len(set(signs)) > 1
        deltas.append(item)
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in deltas:
        groups[(str(row["experiment"]), str(row["target"]), str(row["calibrator"]))].append(row)
    summary: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        item: dict[str, object] = {
            "experiment": key[0], "target": key[1], "calibrator": key[2],
            "fold_runs": len(group),
            "metric_disagreement_fraction": float(np.mean([bool(row["metrics_disagree"]) for row in group])),
        }
        for metric in metrics:
            values = np.asarray([float(row[f"delta_{metric}"]) for row in group])
            item[f"delta_{metric}_mean"] = float(values.mean())
            item[f"delta_{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            item[f"{metric}_improvement_fraction"] = float(np.mean(values < 0))
        summary.append(item)
    prefix = args.output_prefix
    write_csv(prefix.with_name(prefix.name + "_folds.csv"), deltas)
    write_csv(prefix.with_name(prefix.name + "_summary.csv"), summary)
    write_json(prefix.with_suffix(".json"), {"folds": deltas, "summary": summary})
    print(f"Analyzed {len(deltas)} matched calibrator-fold comparisons")


if __name__ == "__main__":
    main()
