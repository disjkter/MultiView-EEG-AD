"""Aggregate the matched CAUEEG self-supervised objective ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CONDITIONS = (
    ("signal", "Signal reconstruction"),
    ("signal_freq", "+ spectrum reconstruction"),
    ("signal_freq_branch", "+ branch consistency"),
    ("full", "+ crop consistency (full)"),
)
METRICS = ("accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity")


def load_summary(path: Path) -> dict[str, object]:
    """Load one downstream summary and require participant-level outputs."""

    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or "subject_ids" not in metrics:
        raise ValueError(f"Missing participant predictions in {path}")
    return payload


def main() -> None:
    """Check participant identity, aggregate metrics, and write paper-ready files."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="outputs/ssl_objective_ablation_caueeg")
    parser.add_argument("--split-root", default="outputs/paper_fixed_splits")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--output-prefix", default="outputs/ssl_objective_ablation_caueeg/aggregate")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    records: list[dict[str, object]] = []
    identity_audit: list[dict[str, object]] = []
    for seed in args.seeds:
        split_path = Path(args.split_root) / f"CAUEEG_processed_19ch_ad_vs_hc_1s_lead_seed{seed}.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        expected_ids = {str(value) for value in split["test_subjects"]}
        for key, label in CONDITIONS:
            summary_path = input_root / key / f"seed_{seed}" / "summary.json"
            summary = load_summary(summary_path)
            metrics = summary["metrics"]
            actual_ids = {str(value) for value in metrics["subject_ids"]}
            matched = actual_ids == expected_ids
            identity_audit.append(
                {
                    "seed": seed,
                    "condition": key,
                    "expected_test_n": len(expected_ids),
                    "actual_test_n": len(actual_ids),
                    "exact_test_id_match": matched,
                    "source": str(summary_path),
                }
            )
            if not matched:
                raise ValueError(f"Test participant mismatch: condition={key}, seed={seed}")
            row: dict[str, object] = {"seed": seed, "condition": key, "label": label}
            row.update({metric: float(metrics[metric]) for metric in METRICS})
            records.append(row)

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for key, _ in CONDITIONS:
        rows = [row for row in records if row["condition"] == key]
        aggregate[key] = {}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            aggregate[key][metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "condition", "label", *METRICS])
        writer.writeheader()
        writer.writerows(records)
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {"seeds": args.seeds, "aggregate": aggregate, "identity_audit": identity_audit},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    table_lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{Ablation of the self-supervised objectives on matched CAUEEG test participants. Values are means $\pm$ population standard deviations across three seeds. BA denotes balanced accuracy.}",
        r"\label{tab:ssl_objective_ablation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Pretraining objective & Accuracy & BA & F1 & AUROC \\",
        r"\midrule",
    ]
    for key, label in CONDITIONS:
        values = aggregate[key]
        formatted = [f"${values[m]['mean']:.3f}\\pm{values[m]['std']:.3f}$" for m in ("accuracy", "balanced_accuracy", "f1", "auc")]
        table_lines.append(f"{label} & " + " & ".join(formatted) + r" \\")
    table_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    prefix.with_suffix(".tex").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"Saved {prefix.with_suffix('.csv')}, {prefix.with_suffix('.json')}, and {prefix.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
