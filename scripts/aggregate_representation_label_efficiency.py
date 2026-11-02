"""Aggregate representation-transfer and label-efficiency experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity")


def read_run(path: Path) -> dict[str, float]:
    """Read participant-level metrics from one summary file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    return {metric: float(metrics[metric]) for metric in METRICS}


def main() -> None:
    """Collect all completed runs and export mean and sample deviation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()

    root = Path(args.input_root)
    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.glob("**/seed_*/summary.json")):
        relative = summary_path.relative_to(root)
        parts = relative.parts
        if len(parts) < 4:
            continue
        experiment, dataset, condition, seed_dir = parts[:4]
        seed = int(seed_dir.removeprefix("seed_"))
        if seed not in args.seeds:
            continue
        row: dict[str, object] = {
            "experiment": experiment,
            "dataset": dataset,
            "condition": condition,
            "seed": seed,
            "source": str(summary_path),
        }
        row.update(read_run(summary_path))
        rows.append(row)
    if not rows:
        raise ValueError(f"No completed runs found under {root}")

    keys = sorted({(str(r["experiment"]), str(r["dataset"]), str(r["condition"])) for r in rows})
    summary: list[dict[str, object]] = []
    for experiment, dataset, condition in keys:
        selected = [
            row for row in rows
            if row["experiment"] == experiment
            and row["dataset"] == dataset
            and row["condition"] == condition
        ]
        record: dict[str, object] = {
            "experiment": experiment,
            "dataset": dataset,
            "condition": condition,
            "num_seeds": len(selected),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in selected])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(record)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    prefix.with_suffix(".json").write_text(
        json.dumps({"runs": rows, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
