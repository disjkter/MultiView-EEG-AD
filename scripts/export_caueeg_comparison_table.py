"""Export CAUEEG model comparison table from saved experiment summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = ["accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity"]
METRIC_LABELS = {
    "accuracy": "Acc",
    "balanced_accuracy": "Bal Acc",
    "f1": "F1",
    "auc": "AUC",
    "sensitivity": "Sen",
    "specificity": "Spec",
}


def summarize(values: list[float]) -> tuple[float, float]:
    """Compute mean/std for one metric."""

    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan"), float("nan")
    return statistics.fmean(finite), statistics.pstdev(finite) if len(finite) > 1 else 0.0


def fmt(mean: float, std: float, percent: bool = True) -> str:
    """Format mean plus std."""

    if math.isnan(mean):
        return "-"
    scale = 100.0 if percent else 1.0
    return f"{mean * scale:.2f}±{std * scale:.2f}"


def load_from_seed_summaries(model: str, run_dirs: list[Path]) -> dict[str, object] | None:
    """Load metrics from one summary per seed."""

    summaries = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            return None
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    aggregate = {}
    for metric in METRICS:
        values = [float(summary["metrics"][metric]) for summary in summaries]
        mean, std = summarize(values)
        aggregate[metric] = {"mean": mean, "std": std}

    return {
        "model": model,
        "aggregate": aggregate,
        "seeds": [summary["seed"] for summary in summaries],
    }


def load_from_aggregate(model: str, aggregate_path: Path) -> dict[str, object] | None:
    """Load metrics from aggregate_summary.json."""

    if not aggregate_path.exists():
        return None
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate_source = payload.get("aggregate_metrics", payload.get("aggregate", {}))
    aggregate = {}
    for metric in METRICS:
        stats = aggregate_source.get(metric)
        if not stats:
            return None
        aggregate[metric] = {"mean": float(stats["mean"]), "std": float(stats["std"])}
    return {
        "model": model,
        "aggregate": aggregate,
        "seeds": payload.get("seeds"),
    }


def rows_to_markdown(rows: list[dict[str, object]]) -> str:
    """Build a Markdown table."""

    headers = ["Model", *[METRIC_LABELS[metric] for metric in METRICS]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---" + "|---:" * len(METRICS) + "|",
    ]
    for row in rows:
        aggregate = row["aggregate"]
        values = [
            str(row["model"]),
            *[fmt(aggregate[metric]["mean"], aggregate[metric]["std"]) for metric in METRICS],
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Write a CSV table with formatted metric cells."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", *[METRIC_LABELS[metric] for metric in METRICS]])
        for row in rows:
            aggregate = row["aggregate"]
            writer.writerow(
                [
                    row["model"],
                    *[fmt(aggregate[metric]["mean"], aggregate[metric]["std"]) for metric in METRICS],
                ]
            )


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description="Export CAUEEG model comparison table.")
    parser.add_argument("--output-md", default="outputs/caueeg_model_comparison.md")
    parser.add_argument("--output-csv", default="outputs/caueeg_model_comparison.csv")
    parser.add_argument(
        "--tage-run-dirs",
        nargs="+",
        default=[
            "outputs/caueeg_tage_ssl_e50_fixedsplit_seed2026",
            "outputs/caueeg_tage_ssl_e50_fixedsplit_seed2027",
            "outputs/caueeg_tage_ssl_e50_fixedsplit_seed2028",
        ],
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="Model=path/to/aggregate_summary.json. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    """Export available comparison rows."""

    args = parse_args()
    rows = []
    tage = load_from_seed_summaries("TAGE-SSL + EA", [Path(path) for path in args.tage_run_dirs])
    if tage:
        rows.append(tage)

    for item in args.baseline:
        if "=" not in item:
            raise ValueError(f"Expected --baseline Model=path, got {item!r}")
        model, path = item.split("=", 1)
        loaded = load_from_aggregate(model, Path(path))
        if loaded:
            rows.append(loaded)

    if not rows:
        raise RuntimeError("No result rows were found.")

    md = rows_to_markdown(rows)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md, encoding="utf-8")
    write_csv(rows, Path(args.output_csv))
    print(md)
    print(f"Saved: {args.output_md}")
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
