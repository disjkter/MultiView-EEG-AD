"""Collect experiment result summaries under outputs into one CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = ["accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity"]


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""

    return json.loads(path.read_text(encoding="utf-8"))


def infer_dataset(name: str) -> str:
    """Infer dataset name from an output directory or file name."""

    lowered = name.lower()
    if "caueeg" in lowered:
        return "CAUEEG"
    if "adfsu" in lowered:
        return "ADFSU"
    if "ds004504" in lowered:
        return "ds004504"
    if "adsz" in lowered:
        return "ADSZ"
    if "40hz" in lowered:
        return "40Hz"
    return ""


def infer_protocol(name: str) -> str:
    """Infer evaluation protocol from an output name."""

    lowered = name.lower()
    if "subject_split" in lowered:
        return "subject_holdout"
    if "loso" in lowered:
        return "LOSO"
    if "lodo" in lowered:
        return "LODO"
    if "cd_" in lowered or "cross" in lowered:
        return "cross_domain"
    return ""


def infer_model(name: str, payload: dict[str, Any]) -> str:
    """Infer model name from metadata or output path."""

    if payload.get("model"):
        return str(payload["model"])

    lowered = name.lower()
    aliases = [
        "adformer",
        "shallowconvnet",
        "eegnet",
        "ceednet",
        "itransformer",
        "mtdnet",
        "tcn",
        "svm",
        "biofoundation",
        "timefreq",
        "tage",
    ]
    for alias in aliases:
        if alias in lowered:
            return alias
    return ""


def format_metric(mean: Any, std: Any) -> str:
    """Format mean/std pair for markdown-friendly display."""

    if mean is None:
        return ""
    if std is None:
        return f"{float(mean):.4f}"
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def extract_aggregate(path: Path) -> dict[str, Any] | None:
    """Extract a row from known aggregate summary formats."""

    payload = read_json(path)
    aggregate = payload.get("aggregate") or payload.get("aggregate_metrics")
    if not isinstance(aggregate, dict):
        return None

    row: dict[str, Any] = {
        "source_file": str(path),
        "experiment": path.parent.name if path.name == "aggregate_summary.json" else path.stem,
        "dataset": infer_dataset(str(path)),
        "protocol": infer_protocol(str(path)),
        "model": infer_model(str(path), payload),
        "num_seeds": "",
    }
    seeds = payload.get("seeds")
    runs = payload.get("runs") or payload.get("seed_runs")
    if isinstance(seeds, list):
        row["num_seeds"] = len(seeds)
    elif isinstance(runs, list):
        row["num_seeds"] = len(runs)

    for metric in METRICS:
        item = aggregate.get(metric)
        if isinstance(item, dict):
            mean = item.get("mean")
            std = item.get("std")
        else:
            mean = None
            std = None
        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std
        row[metric] = format_metric(mean, std)
    return row


def collect(outputs_dir: Path) -> list[dict[str, Any]]:
    """Collect aggregate result rows under an outputs directory."""

    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    patterns = ["aggregate_summary.json", "*_3seed_summary.json"]
    for pattern in patterns:
        for path in outputs_dir.rglob(pattern):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                row = extract_aggregate(path)
            except (json.JSONDecodeError, OSError, ValueError):
                row = None
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda item: (item.get("dataset", ""), item.get("protocol", ""), item.get("model", ""), item.get("experiment", "")))
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write result rows to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "protocol", "model", "experiment", "num_seeds"]
    for metric in METRICS:
        fields.extend([metric, f"{metric}_mean", f"{metric}_std"])
    fields.append("source_file")
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--output", default="outputs/experiment_results_registry.csv")
    args = parser.parse_args()

    rows = collect(Path(args.outputs_dir))
    write_csv(rows, Path(args.output))
    print(f"Collected {len(rows)} aggregate result files")
    print(f"Saved CSV to: {args.output}")


if __name__ == "__main__":
    main()
