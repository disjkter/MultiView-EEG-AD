"""Run paired significance checks for initialization ablation experiments.

This script reads 3-seed summary JSON files and compares selected initialization
strategies with exact paired sign-flip permutation tests. It is intentionally
small and dependency-light so it can run in the current project environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Condition:
    """One experiment condition used in the significance comparison."""

    label: str
    summary_path: str


SUMMARY_GROUPS: dict[str, dict[str, Condition]] = {
    "CAUEEG": {
        "auditory40hz_single": Condition(
            label="auditory40hz single-source",
            summary_path="outputs/subject_split_caueeg_tage_timefreq_consistency_auditory40hz_1s_lead_3seed_summary.json",
        ),
        "brainlat19_single": Condition(
            label="BrainLat19 single-source",
            summary_path="outputs/subject_split_caueeg_tage_timefreq_consistency_brainlat19_1s_lead_3seed_summary.json",
        ),
        "multi_source": Condition(
            label="Multi-source pretraining",
            summary_path="outputs/caueeg_biofoundation_v1_3seed_summary.json",
        ),
    },
    "ds004504": {
        "random_init": Condition(
            label="Random init",
            summary_path="outputs/subject_split_ds004504_tage_randominit_1s_lead_3seed_summary.json",
        ),
        "auditory40hz_single": Condition(
            label="auditory40hz single-source",
            summary_path="outputs/subject_split_ds004504_tage_timefreq_consistency_auditory40hz_1s_lead_3seed_summary.json",
        ),
        "multi_source": Condition(
            label="Multi-source pretraining",
            summary_path="outputs/subject_split_ds004504_tage_timefreq_consistency_adssl_1s_lead_3seed_summary.json",
        ),
        # This file exists locally but should be treated carefully if you later
        # replace it with a cleaner BrainLat19-only summary.
        "bl19_40hz_single": Condition(
            label="bl19_40hz single-source",
            summary_path="outputs/subject_split_ds004504_tage_timefreq_consistency_bl19_40hz_3seed_summary.json",
        ),
    },
}

PAIRINGS: dict[str, list[tuple[str, str]]] = {
    "CAUEEG": [
        ("multi_source", "brainlat19_single"),
        ("multi_source", "auditory40hz_single"),
        ("brainlat19_single", "auditory40hz_single"),
    ],
    "ds004504": [
        ("multi_source", "random_init"),
        ("multi_source", "auditory40hz_single"),
        ("random_init", "auditory40hz_single"),
        ("multi_source", "bl19_40hz_single"),
    ],
}

METRICS = (
    "accuracy",
    "balanced_accuracy",
    "f1",
    "auc",
    "sensitivity",
    "specificity",
)


def load_runs(path: Path) -> list[dict[str, float]]:
    """Load the per-seed runs from one aggregate summary file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if not runs:
        raise ValueError(f"No runs found in {path}")
    return runs


def exact_sign_flip_pvalue(diffs: Iterable[float]) -> tuple[float, float]:
    """Return observed mean difference and exact two-sided sign-flip p-value."""

    diffs = [float(x) for x in diffs]
    n = len(diffs)
    observed = sum(diffs) / n
    null_means = []
    for signs in product((-1.0, 1.0), repeat=n):
        null_means.append(sum(d * s for d, s in zip(diffs, signs)) / n)
    extreme = sum(abs(x) >= abs(observed) - 1e-12 for x in null_means)
    pvalue = extreme / len(null_means)
    return observed, pvalue


def compare_pair(
    lhs_runs: list[dict[str, float]],
    rhs_runs: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Compare two conditions metric by metric."""

    if len(lhs_runs) != len(rhs_runs):
        raise ValueError("Seed counts do not match between compared conditions.")

    results: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        lhs_values = [float(run[metric]) for run in lhs_runs]
        rhs_values = [float(run[metric]) for run in rhs_runs]
        diffs = [a - b for a, b in zip(lhs_values, rhs_values)]
        mean_diff, pvalue = exact_sign_flip_pvalue(diffs)
        results[metric] = {
            "lhs_mean": sum(lhs_values) / len(lhs_values),
            "rhs_mean": sum(rhs_values) / len(rhs_values),
            "mean_diff": mean_diff,
            "pvalue": pvalue,
        }
    return results


def format_report(report: dict[str, list[dict[str, object]]]) -> str:
    """Render a compact markdown report."""

    lines = [
        "# Initialization Ablation Significance Report",
        "",
        "Exact paired sign-flip permutation test over per-seed metrics.",
        "",
    ]
    for target, rows in report.items():
        lines.append(f"## Target = {target}")
        lines.append("")
        lines.append("| Comparison | Metric | Mean Diff | p-value |")
        lines.append("|---|---:|---:|---:|")
        for row in rows:
            comparison = row["comparison"]
            metric = row["metric"]
            mean_diff = row["mean_diff"]
            pvalue = row["pvalue"]
            lines.append(
                f"| {comparison} | {metric} | {mean_diff:+.4f} | {pvalue:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Run comparisons and write JSON plus markdown summaries."""

    base_dir = Path(".")
    report: dict[str, list[dict[str, object]]] = {}

    for target, pairings in PAIRINGS.items():
        conditions = SUMMARY_GROUPS[target]
        target_rows: list[dict[str, object]] = []
        for lhs_key, rhs_key in pairings:
            lhs = conditions[lhs_key]
            rhs = conditions[rhs_key]
            lhs_runs = load_runs(base_dir / lhs.summary_path)
            rhs_runs = load_runs(base_dir / rhs.summary_path)
            pair_result = compare_pair(lhs_runs, rhs_runs)
            for metric in ("balanced_accuracy", "auc", "sensitivity", "specificity"):
                values = pair_result[metric]
                target_rows.append(
                    {
                        "comparison": f"{lhs.label} vs {rhs.label}",
                        "metric": metric,
                        "mean_diff": values["mean_diff"],
                        "pvalue": values["pvalue"],
                        "lhs_mean": values["lhs_mean"],
                        "rhs_mean": values["rhs_mean"],
                    }
                )
        report[target] = target_rows

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "initialization_ablation_significance.json"
    md_path = out_dir / "initialization_ablation_significance.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(format_report(report), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
