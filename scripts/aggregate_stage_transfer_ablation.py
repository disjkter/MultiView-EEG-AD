"""Verify and aggregate the four-condition staged-transfer ablation."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aggregate_jbhi_revision_experiments import (  # noqa: E402
    METRICS,
    expected_test_ids,
    extract_metrics,
)


CONDITIONS = {
    "random_init": "Random initialization",
    "ssl_only": "Self-supervised pretraining",
    "source_only": "Source-task training",
    "ssl_source": "Self-supervised pretraining + source-task training",
}


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Return an exact two-sided sign-flip p-value for paired differences."""

    observed = abs(float(differences.mean()))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted = abs(float(np.mean(differences * np.asarray(signs))))
        exceed += int(permuted >= observed - 1e-15)
        total += 1
    return exceed / total


def main() -> None:
    """Audit fixed test identities and export paired multi-seed summaries."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="outputs/jbhi_revision/stage_transfer_ablation")
    parser.add_argument("--split-root", default="outputs/jbhi_revision/fixed_splits")
    parser.add_argument("--output-prefix", default="outputs/jbhi_revision/stage_transfer_ablation/aggregate")
    parser.add_argument("--dataset", default="ds004504_processed_19ch_1s_lead")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2026, 2036)))
    args = parser.parse_args()

    root = Path(args.input_root)
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for condition, label in CONDITIONS.items():
        for seed in args.seeds:
            summary_path = root / "target" / condition / f"seed_{seed}" / "summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics, actual_ids = extract_metrics(payload, summary_path)
            expected_ids = expected_test_ids(Path(args.split_root), args.dataset, seed)
            exact = actual_ids == expected_ids
            if not exact:
                raise ValueError(
                    f"Test participants differ for {condition}, seed {seed}: "
                    f"missing={sorted(expected_ids-actual_ids)[:5]}, "
                    f"unexpected={sorted(actual_ids-expected_ids)[:5]}"
                )
            row: dict[str, object] = {
                "condition": condition,
                "label": label,
                "seed": seed,
            }
            row.update({metric: float(metrics[metric]) for metric in METRICS})
            rows.append(row)
            audits.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "expected_test_n": len(expected_ids),
                    "actual_test_n": len(actual_ids),
                    "exact_test_id_match": exact,
                    "source": str(summary_path),
                }
            )

    aggregates: list[dict[str, object]] = []
    for condition, label in CONDITIONS.items():
        selected = [row for row in rows if row["condition"] == condition]
        record: dict[str, object] = {
            "condition": condition,
            "label": label,
            "num_seeds": len(selected),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in selected])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregates.append(record)

    full_by_seed = {
        int(row["seed"]): float(row["auc"])
        for row in rows
        if row["condition"] == "ssl_source"
    }
    paired: list[dict[str, object]] = []
    for comparator in ("random_init", "ssl_only", "source_only"):
        comparator_by_seed = {
            int(row["seed"]): float(row["auc"])
            for row in rows
            if row["condition"] == comparator
        }
        differences = np.asarray(
            [full_by_seed[seed] - comparator_by_seed[seed] for seed in args.seeds],
            dtype=np.float64,
        )
        paired.append(
            {
                "comparison": f"ssl_source - {comparator}",
                "mean_auc_difference": float(differences.mean()),
                "wins": int((differences > 0).sum()),
                "ties": int((differences == 0).sum()),
                "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
            }
        )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "seeds": args.seeds,
                "runs": rows,
                "summary": aggregates,
                "paired_auc": paired,
                "identity_audit": audits,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Staged-transfer ablation",
        "",
        "Values are mean +/- sample standard deviation across fixed participant splits.",
        "",
        "| Condition | Accuracy | Balanced accuracy | F1 | AUROC |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        def fmt(metric: str) -> str:
            return f"{float(row[f'{metric}_mean']):.3f} +/- {float(row[f'{metric}_std']):.3f}"
        lines.append(
            f"| {row['label']} | {fmt('accuracy')} | {fmt('balanced_accuracy')} | "
            f"{fmt('f1')} | {fmt('auc')} |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": aggregates, "paired_auc": paired}, indent=2))


if __name__ == "__main__":
    main()
