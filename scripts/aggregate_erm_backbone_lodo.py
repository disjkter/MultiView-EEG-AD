"""Aggregate participant-level ERM LODO results across EEG backbones and seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "balanced_accuracy",
    "auc",
    "sensitivity",
    "specificity",
    "ece",
    "brier",
    "nll",
)


def calibration_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Compute ten-bin ECE and binary proper scoring rules."""

    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    assignments = np.minimum(np.floor(clipped * 10).astype(np.int64), 9)
    ece = 0.0
    for bin_index in range(10):
        mask = assignments == bin_index
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(clipped[mask].mean() - labels[mask].mean())
            )
    return {
        "ece": float(ece),
        "brier": float(np.mean((clipped - labels) ** 2)),
        "nll": float(
            -np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
        ),
    }


def load_rows(results_dir: Path) -> list[dict[str, object]]:
    """Load one macro-averaged row per backbone and seed."""

    rows: list[dict[str, object]] = []
    for path in sorted(results_dir.glob("*/seed*/erm/lodo_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        folds = payload["folds"]
        fold_values: dict[str, list[float]] = defaultdict(list)
        worst_balanced_accuracy = 1.0
        for fold in folds:
            metrics = fold["metrics"]
            for metric in ("balanced_accuracy", "auc", "sensitivity", "specificity"):
                fold_values[metric].append(float(metrics[metric]))
            labels = np.asarray(metrics["y_true"], dtype=np.int64)
            probability = np.asarray(metrics["y_prob"], dtype=np.float64)
            for metric, value in calibration_metrics(labels, probability).items():
                fold_values[metric].append(value)
            worst_balanced_accuracy = min(
                worst_balanced_accuracy, float(metrics["balanced_accuracy"])
            )
        rows.append(
            {
                "backbone": str(payload["backbone"]),
                "seed": int(payload["seed"]),
                **{metric: float(np.mean(fold_values[metric])) for metric in METRICS},
                "worst_cohort_balanced_accuracy": worst_balanced_accuracy,
                "summary_path": str(path),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No ERM LODO summaries found under {results_dir}")
    return rows


def summarize(rows: list[dict[str, object]], expected_seeds: int) -> dict[str, object]:
    """Return mean and sample standard deviation for every backbone."""

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["backbone"])].append(row)
    summaries: dict[str, object] = {}
    for backbone, backbone_rows in sorted(grouped.items()):
        if len(backbone_rows) != expected_seeds:
            raise ValueError(
                f"{backbone} has {len(backbone_rows)} seeds; expected {expected_seeds}"
            )
        metric_names = (*METRICS, "worst_cohort_balanced_accuracy")
        summaries[backbone] = {
            "num_seeds": len(backbone_rows),
            "seeds": sorted(int(row["seed"]) for row in backbone_rows),
            "metrics": {
                metric: {
                    "mean": float(np.mean([float(row[metric]) for row in backbone_rows])),
                    "std": (
                        float(np.std([float(row[metric]) for row in backbone_rows], ddof=1))
                        if len(backbone_rows) > 1
                        else 0.0
                    ),
                }
                for metric in metric_names
            },
        }
    return {"expected_seeds": expected_seeds, "backbones": summaries}


def main() -> None:
    """Aggregate completed architecture-breadth experiments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--expected-seeds", type=int, default=10)
    args = parser.parse_args()

    rows = load_rows(Path(args.results_dir))
    summary = summarize(rows, args.expected_seeds)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
