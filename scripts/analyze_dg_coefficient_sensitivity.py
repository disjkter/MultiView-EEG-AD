"""Audit and aggregate the P1-C domain-generalization coefficient sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


DEFAULT_VARIANT = {
    "dann": "dann_ld020",
    "coral": "coral_la1",
    "mmd": "mmd_la1",
    "tacd": "tacd_scale100",
}


def binary_calibration(y_true: list[int], y_prob: list[float]) -> dict[str, float]:
    """Return participant-level fixed-bin ECE, adaptive ECE, Brier, and NLL."""
    n = len(y_true)
    if n == 0 or n != len(y_prob):
        raise ValueError("Invalid participant predictions")
    probs = [min(max(float(p), 1e-7), 1.0 - 1e-7) for p in y_prob]
    labels = [int(y) for y in y_true]
    brier = mean((p - y) ** 2 for y, p in zip(labels, probs))
    nll = -mean(y * math.log(p) + (1 - y) * math.log(1 - p) for y, p in zip(labels, probs))

    ece = 0.0
    for bin_index in range(10):
        lo, hi = bin_index / 10.0, (bin_index + 1) / 10.0
        members = [i for i, p in enumerate(probs) if lo <= p < hi or (bin_index == 9 and p == 1.0)]
        if members:
            confidence = mean(probs[i] for i in members)
            accuracy = mean(labels[i] for i in members)
            ece += len(members) / n * abs(confidence - accuracy)

    ordered = sorted(range(n), key=probs.__getitem__)
    adaptive = 0.0
    for bin_index in range(10):
        start = bin_index * n // 10
        stop = (bin_index + 1) * n // 10
        members = ordered[start:stop]
        if members:
            confidence = mean(probs[i] for i in members)
            accuracy = mean(labels[i] for i in members)
            adaptive += len(members) / n * abs(confidence - accuracy)
    return {"ece": ece, "adaptive_ece": adaptive, "brier": brier, "nll": nll}


def load_records(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load fold records and return protocol-audit issues."""
    records: list[dict[str, Any]] = []
    issues: list[str] = []
    files = sorted(root.rglob("lodo_summary.json"))
    if len(files) != 36:
        issues.append(f"Expected 36 summaries, found {len(files)}")
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("backbone") != "graph_temporal":
            issues.append(f"Unexpected backbone: {path}")
        for fold in payload.get("folds", []):
            for field in (
                "target_data_used_during_training",
                "target_data_used_for_model_selection",
                "target_data_used_for_threshold_selection",
            ):
                if fold.get(field) is not False:
                    issues.append(f"{path}: {fold.get('target')} has {field}={fold.get(field)!r}")
            metric = fold["metrics"]
            row = {
                "path": str(path),
                "method": payload["method"],
                "variant": payload["variant"],
                "seed": int(payload["seed"]),
                "target": fold["target"],
                "source_validation_auc": float(fold["source_validation_metrics"]["auc"]),
                "source_threshold_ba": float(metric["balanced_accuracy"]),
                "accuracy": float(metric["accuracy"]),
                "f1": float(metric["f1"]),
                "auc": float(metric["auc"]),
                "sensitivity": float(metric["sensitivity"]),
                "specificity": float(metric["specificity"]),
                "fixed_threshold_ba": float(fold["fixed_threshold_metrics"]["balanced_accuracy"]),
                "target_subjects": int(fold["target_test_subjects"]),
            }
            row.update(binary_calibration(metric["y_true"], metric["y_prob"]))
            records.append(row)
    return records, issues


def select_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select one coefficient per method, seed, and fold using source-validation AUROC only."""
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault((row["method"], row["seed"], row["target"]), []).append(row)
    selected = []
    for (method, _, _), rows in sorted(groups.items()):
        default = DEFAULT_VARIANT[method]
        winner = sorted(
            rows,
            key=lambda row: (
                -row["source_validation_auc"],
                row["variant"] != default,
                row["variant"],
            ),
        )[0].copy()
        winner["selection_tied"] = sum(
            abs(row["source_validation_auc"] - winner["source_validation_auc"]) < 1e-12
            for row in rows
        ) > 1
        selected.append(winner)
    return selected


def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Macro-average result rows over all matched seed-target cells."""
    metrics = (
        "source_validation_auc", "source_threshold_ba", "fixed_threshold_ba", "auc",
        "ece", "adaptive_ece", "brier", "nll", "sensitivity", "specificity",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for group, members in sorted(groups.items()):
        item = {key: value for key, value in zip(keys, group)}
        item["cells"] = len(members)
        for metric in metrics:
            values = [float(member[metric]) for member in members]
            item[metric] = mean(values)
            item[f"{metric}_std"] = pstdev(values)
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to a UTF-8 CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the complete P1-C audit and source-only coefficient selection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    records, issues = load_records(args.input_root)
    selected = select_records(records)
    all_summary = summarize(records, ("method", "variant"))
    selected_summary = summarize(selected, ("method",))
    counts = {
        method: dict(sorted(Counter(row["variant"] for row in selected if row["method"] == method).items()))
        for method in sorted({row["method"] for row in selected})
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "dg_coefficient_all_folds.csv", records)
    write_csv(args.output_root / "dg_coefficient_all_variants_summary.csv", all_summary)
    write_csv(args.output_root / "dg_coefficient_selected_folds.csv", selected)
    write_csv(args.output_root / "dg_coefficient_selected_summary.csv", selected_summary)
    result = {
        "summary_files": len(list(args.input_root.rglob("lodo_summary.json"))),
        "fold_records": len(records),
        "selected_cells": len(selected),
        "protocol_issues": issues,
        "selection_rule": "max source-validation AUROC per method/seed/target; ties prefer locked default",
        "selection_counts": counts,
        "selected_summary": selected_summary,
    }
    (args.output_root / "dg_coefficient_sensitivity.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
