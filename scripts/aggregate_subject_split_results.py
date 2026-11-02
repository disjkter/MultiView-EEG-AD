"""Aggregate subject-split experiment summaries across seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "f1",
    "auc",
    "sensitivity",
    "specificity",
)


def load_summary(path: Path) -> dict:
    """Load one experiment summary JSON."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_mean_std(values: list[float]) -> str:
    """Format a metric list as mean +/- population std."""
    if not values:
        return ""
    return f"{mean(values):.6f} +/- {pstdev(values):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="Experiment output directories containing summary.json files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path. A JSON summary is also written next to it.",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for input_dir in args.input_dirs:
        summary_path = Path(input_dir) / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary file: {summary_path}")
        summary = load_summary(summary_path)
        metrics = summary.get("metrics", {})
        row: dict[str, object] = {
            "seed": summary.get("seed", Path(input_dir).name),
            "best_epoch": summary.get("best_epoch", ""),
            "threshold": summary.get("decision_threshold", ""),
        }
        for key in METRIC_KEYS:
            row[key] = metrics.get(key, "")
        rows.append(row)

    aggregate = {}
    for key in METRIC_KEYS:
        values = [float(row[key]) for row in rows if row[key] != ""]
        aggregate[key] = {
            "mean": mean(values) if values else None,
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "formatted": format_mean_std(values),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seed", "best_epoch", "threshold", *METRIC_KEYS],
        )
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({})
        writer.writerow({"seed": "aggregate"})
        for key in METRIC_KEYS:
            writer.writerow({"seed": key, "best_epoch": aggregate[key]["formatted"]})

    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"runs": rows, "aggregate": aggregate}, f, indent=2)

    print("Aggregate subject-split results")
    for key in METRIC_KEYS:
        print(f"{key}: {aggregate[key]['formatted']}")
    print(f"Saved CSV to: {output_path}")
    print(f"Saved JSON to: {json_path}")


if __name__ == "__main__":
    main()
