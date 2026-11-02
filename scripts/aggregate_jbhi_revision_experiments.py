"""Aggregate the fixed-split experiments prepared for the JBHI revision.

The script verifies that every run evaluated exactly the participants recorded
in its split manifest before computing means and standard deviations.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity")
OBJECTIVE_LABELS = {
    "random_init": "Random initialization",
    "masked_signal_only": "Masked signal reconstruction",
    "full": "Full pretraining",
    "minus_signal": "Without signal reconstruction",
    "minus_spectrum": "Without spectral reconstruction",
    "minus_branch": "Without cross-branch agreement",
    "minus_crop": "Without temporal-view consistency",
}
BRANCH_LABELS = {
    "full": "Dual-branch model",
    "no_spatial": "Temporal branch only",
    "no_mscnn": "Cross-channel branch only",
}


def read_json(path: Path) -> dict[str, object]:
    """Read one JSON object."""

    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def expected_test_ids(split_root: Path, dataset: str, seed: int) -> set[str]:
    """Load the expected test participant identifiers for one run."""

    split = read_json(split_root / f"{dataset}_seed{seed}.json")
    return {str(value) for value in split["test_subjects"]}


def extract_metrics(summary: dict[str, object], source: Path) -> tuple[dict[str, object], set[str]]:
    """Extract required metrics and participant identifiers from a summary."""

    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing metrics object: {source}")
    missing = [name for name in METRICS if name not in metrics]
    if missing:
        raise ValueError(f"Missing metrics {missing}: {source}")
    ids = metrics.get("subject_ids")
    if ids is None:
        predictions = summary.get("participant_predictions")
        ids = predictions.get("subject_ids") if isinstance(predictions, dict) else None
    if ids is None:
        raise ValueError(f"Missing participant identifiers: {source}")
    return metrics, {str(value) for value in ids}


def metric_row(
    *, experiment: str, dataset: str, condition: str, label: str, seed: int,
    summary_path: Path, split_root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build one metric row and its split-identity audit row."""

    summary = read_json(summary_path)
    metrics, actual_ids = extract_metrics(summary, summary_path)
    expected_ids = expected_test_ids(split_root, dataset, seed)
    exact = actual_ids == expected_ids
    audit = {
        "experiment": experiment,
        "dataset": dataset,
        "condition": condition,
        "seed": seed,
        "expected_test_n": len(expected_ids),
        "actual_test_n": len(actual_ids),
        "exact_test_id_match": exact,
        "source": str(summary_path),
    }
    if not exact:
        missing = sorted(expected_ids - actual_ids)[:5]
        unexpected = sorted(actual_ids - expected_ids)[:5]
        raise ValueError(
            f"Test participant mismatch in {summary_path}; "
            f"missing={missing}, unexpected={unexpected}"
        )
    row: dict[str, object] = {
        "experiment": experiment,
        "dataset": dataset,
        "condition": condition,
        "label": label,
        "seed": seed,
        "source": str(summary_path),
    }
    row.update({name: float(metrics[name]) for name in METRICS})
    return row, audit


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Compute mean and sample standard deviation for each condition."""

    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["experiment"]), str(row["dataset"]),
            str(row["condition"]), str(row["label"]),
        )
        grouped[key].append(row)
    output: list[dict[str, object]] = []
    for key, entries in grouped.items():
        experiment, dataset, condition, label = key
        record: dict[str, object] = {
            "experiment": experiment,
            "dataset": dataset,
            "condition": condition,
            "label": label,
            "num_seeds": len(entries),
            "seeds": ",".join(str(row["seed"]) for row in sorted(entries, key=lambda x: int(x["seed"]))),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in entries], dtype=np.float64)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(record)
    return sorted(output, key=lambda row: (str(row["experiment"]), str(row["dataset"]), str(row["condition"])))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a list of dictionaries as UTF-8 CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a compact paper-oriented result table."""

    lines = [
        "# Fixed-split revision experiments",
        "",
        "Values are mean +/- sample standard deviation across random seeds.",
        "",
        "| Experiment | Dataset | Condition | Seeds | Accuracy | Balanced accuracy | F1 | AUROC |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(metric: str) -> str:
            return f"{float(row[f'{metric}_mean']):.3f} +/- {float(row[f'{metric}_std']):.3f}"
        lines.append(
            f"| {row['experiment']} | {row['dataset']} | {row['label']} | {row['num_seeds']} | "
            f"{fmt('accuracy')} | {fmt('balanced_accuracy')} | {fmt('f1')} | {fmt('auc')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Verify and aggregate all requested revision experiments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective-root", default="outputs/jbhi_revision/objective_ablation")
    parser.add_argument("--branch-root", default="outputs/jbhi_revision/branch_ablation")
    parser.add_argument("--split-root", default="outputs/jbhi_revision/fixed_splits")
    parser.add_argument("--output-prefix", default="outputs/jbhi_revision/aggregate")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2026, 2036)))
    parser.add_argument(
        "--datasets", nargs="+",
        default=["CAUEEG_processed_19ch_ad_vs_hc_1s_lead", "ds004504_processed_19ch_1s_lead"],
    )
    args = parser.parse_args()

    objective_root = Path(args.objective_root)
    branch_root = Path(args.branch_root)
    split_root = Path(args.split_root)
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []

    objective_dataset = "CAUEEG_processed_19ch_ad_vs_hc_1s_lead"
    for condition, label in OBJECTIVE_LABELS.items():
        for seed in args.seeds:
            row, audit = metric_row(
                experiment="pretraining_objective",
                dataset=objective_dataset,
                condition=condition,
                label=label,
                seed=seed,
                summary_path=objective_root / condition / f"seed_{seed}" / "summary.json",
                split_root=split_root,
            )
            rows.append(row)
            audits.append(audit)

    for dataset in args.datasets:
        for condition, label in BRANCH_LABELS.items():
            for seed in args.seeds:
                row, audit = metric_row(
                    experiment="encoder_branch",
                    dataset=dataset,
                    condition=condition,
                    label=label,
                    seed=seed,
                    summary_path=branch_root / dataset / f"{condition}_s{seed}" / "summary.json",
                    split_root=split_root,
                )
                rows.append(row)
                audits.append(audit)

    aggregates = aggregate_rows(rows)
    prefix = Path(args.output_prefix)
    write_csv(prefix.with_suffix(".runs.csv"), rows)
    write_csv(prefix.with_suffix(".csv"), aggregates)
    write_csv(prefix.with_suffix(".identity.csv"), audits)
    write_markdown(prefix.with_suffix(".md"), aggregates)
    prefix.with_suffix(".json").write_text(
        json.dumps({"runs": rows, "summary": aggregates, "identity_audit": audits}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Verified {len(rows)} runs and wrote {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
