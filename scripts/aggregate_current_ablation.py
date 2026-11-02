"""Aggregate current hold-out ablation results.

Reads ``outputs/current_ablation/**/summary.json`` produced by
``run_current_ablation_batch.py`` and writes compact tables for paper reporting.

Example
-------
    python scripts/aggregate_current_ablation.py \
        --results-dir outputs/current_ablation \
        --output-prefix outputs/current_ablation/current_ablation_summary
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ["accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity"]
VARIANT_ORDER = [
    "full",
    "no_spatial",
    "no_mscnn",
    "no_multiscale",
    "no_se",
    "cls_1layer",
    "cls_2layer",
    "cls_3layer",
    "attn_fusion",
    "gate_fusion",
    "neuro_prior",
]


def parse_run_tag(tag: str) -> tuple[str, int] | None:
    """Parse tags like ``full_s2026`` or ``full_s2026_n30``."""

    if "_s" not in tag:
        return None
    variant, rest = tag.rsplit("_s", 1)
    seed_text = rest.split("_", 1)[0]
    try:
        return variant, int(seed_text)
    except ValueError:
        return None


def collect(results_dir: Path) -> list[dict[str, object]]:
    """Collect all summary rows from a current ablation directory."""

    rows: list[dict[str, object]] = []
    for summary_path in sorted(results_dir.rglob("summary.json")):
        rel = summary_path.relative_to(results_dir)
        if len(rel.parts) < 3:
            continue
        dataset = rel.parts[0]
        parsed = parse_run_tag(rel.parts[1])
        if parsed is None:
            continue
        variant, seed = parsed
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        metrics = summary.get("metrics", {})
        row = {
            "dataset": dataset,
            "variant": variant,
            "label": summary.get("label", variant),
            "seed": seed,
            "protocol": summary.get("protocol", ""),
            "checkpoint": summary.get("checkpoint", ""),
        }
        for metric in METRICS:
            row[metric] = metrics.get(metric)
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Compute mean/std per dataset and variant."""

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["variant"]))].append(row)

    dataset_order = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    variant_rank = {variant: i for i, variant in enumerate(VARIANT_ORDER)}

    summary_rows: list[dict[str, object]] = []
    for dataset in dataset_order:
        variants = sorted(
            [variant for ds, variant in grouped if ds == dataset],
            key=lambda item: variant_rank.get(item, 999),
        )
        baseline = None
        if (dataset, "full") in grouped:
            vals = [
                float(row["balanced_accuracy"])
                for row in grouped[(dataset, "full")]
                if row.get("balanced_accuracy") is not None
            ]
            baseline = float(np.mean(vals)) if vals else None

        for variant in variants:
            entries = grouped[(dataset, variant)]
            out = {
                "dataset": dataset,
                "variant": variant,
                "label": entries[0].get("label", variant),
                "num_seeds": len(entries),
                "seeds": ",".join(str(row["seed"]) for row in sorted(entries, key=lambda r: int(r["seed"]))),
            }
            for metric in METRICS:
                vals = [
                    float(row[metric])
                    for row in entries
                    if row.get(metric) is not None and not np.isnan(float(row[metric]))
                ]
                out[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
                out[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            bal = out["balanced_accuracy_mean"]
            out["delta_balanced_accuracy"] = (
                float(bal - baseline) if baseline is not None and not np.isnan(float(bal)) else float("nan")
            )
            summary_rows.append(out)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a compact Markdown table focused on balanced accuracy."""

    lines = [
        "# Current Ablation Summary",
        "",
        "| Dataset | Variant | Seeds | BalAcc | Delta | F1 | AUC | Sens | Spec |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {label} | {num_seeds} | {bal:.4f} | {delta:+.4f} | "
            "{f1:.4f} | {auc:.4f} | {sens:.4f} | {spec:.4f} |".format(
                dataset=row["dataset"],
                label=row["label"],
                num_seeds=row["num_seeds"],
                bal=float(row["balanced_accuracy_mean"]),
                delta=float(row["delta_balanced_accuracy"]),
                f1=float(row["f1_mean"]),
                auc=float(row["auc_mean"]),
                sens=float(row["sensitivity_mean"]),
                spec=float(row["specificity_mean"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate current hold-out ablation results")
    parser.add_argument("--results-dir", default="outputs/current_ablation")
    parser.add_argument("--output-prefix", default="outputs/current_ablation/current_ablation_summary")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_prefix = Path(args.output_prefix)
    rows = collect(results_dir)
    summary_rows = summarize(rows)

    write_csv(output_prefix.with_suffix(".raw.csv"), rows)
    write_csv(output_prefix.with_suffix(".csv"), summary_rows)
    write_markdown(output_prefix.with_suffix(".md"), summary_rows)
    output_prefix.with_suffix(".json").write_text(
        json.dumps({"raw": rows, "summary": summary_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Collected {len(rows)} runs from {results_dir}")
    print(f"Wrote {len(summary_rows)} summary rows")
    print(f"CSV: {output_prefix.with_suffix('.csv')}")
    print(f"Markdown: {output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
