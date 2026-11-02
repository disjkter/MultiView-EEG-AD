"""Generate a paper-ready significance report for key EEG transfer results.

This script focuses on the current CAUEEG / ds004504 paper storyline:
1. whether multi-source pretraining is better than simpler initialization;
2. whether single-source priors are source-dependent.

It supports both aggregate summary formats used in this repository:
- {"runs": [...]} from subject-split summaries
- {"seed_runs": [...]} from multi-seed aggregate summaries
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path


@dataclass(frozen=True)
class Condition:
    """One experimental condition used in a paired significance comparison."""

    label: str
    summary_path: str
    metric_scope: str = "flat"


COMPARISONS: dict[str, list[tuple[Condition, Condition]]] = {
    "CAUEEG 单数据集微调": [
        (
            Condition(
                label="多源预训练",
                summary_path="outputs/caueeg_biofoundation_v1_3seed_summary.json",
            ),
            Condition(
                label="BrainLat19 单源预训练",
                summary_path="outputs/subject_split_caueeg_tage_timefreq_consistency_brainlat19_1s_lead_3seed_summary.json",
            ),
        ),
        (
            Condition(
                label="多源预训练",
                summary_path="outputs/caueeg_biofoundation_v1_3seed_summary.json",
            ),
            Condition(
                label="auditory40hz 单源预训练",
                summary_path="outputs/subject_split_caueeg_tage_timefreq_consistency_auditory40hz_1s_lead_3seed_summary.json",
            ),
        ),
    ],
    "ds004504 单数据集微调": [
        (
            Condition(
                label="多源预训练",
                summary_path="outputs/subject_split_ds004504_tage_timefreq_consistency_adssl_1s_lead_3seed_summary.json",
            ),
            Condition(
                label="随机初始化",
                summary_path="outputs/subject_split_ds004504_tage_randominit_1s_lead_3seed_summary.json",
            ),
        ),
        (
            Condition(
                label="多源预训练",
                summary_path="outputs/subject_split_ds004504_tage_timefreq_consistency_adssl_1s_lead_3seed_summary.json",
            ),
            Condition(
                label="auditory40hz 单源预训练",
                summary_path="outputs/subject_split_ds004504_tage_timefreq_consistency_auditory40hz_1s_lead_3seed_summary.json",
            ),
        ),
    ],
}

METRICS = ("accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity")


def load_runs(path: Path) -> list[dict[str, float]]:
    """Load per-seed results from one summary file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if "runs" in payload:
        return payload["runs"]
    if "seed_runs" in payload:
        rows: list[dict[str, float]] = []
        for seed_run in payload["seed_runs"]:
            target_metrics = seed_run.get("target_metrics", {})
            row = {"seed": seed_run.get("seed")}
            for metric in METRICS:
                if metric in target_metrics:
                    row[metric] = float(target_metrics[metric])
                elif metric in seed_run:
                    row[metric] = float(seed_run[metric])
            rows.append(row)
        return rows
    raise ValueError(f"Unsupported summary format: {path}")


def exact_sign_flip(diffs: list[float]) -> tuple[float, float]:
    """Return observed mean diff and exact two-sided sign-flip p-value."""

    observed = sum(diffs) / len(diffs)
    null_means = []
    for signs in product((-1.0, 1.0), repeat=len(diffs)):
        null_means.append(sum(d * s for d, s in zip(diffs, signs)) / len(diffs))
    extreme = sum(abs(x) >= abs(observed) - 1e-12 for x in null_means)
    return observed, extreme / len(null_means)


def compare(lhs_runs: list[dict[str, float]], rhs_runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Compare two conditions seed-by-seed."""

    if len(lhs_runs) != len(rhs_runs):
        raise ValueError("Mismatched seed counts.")

    results: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        lhs_values = [float(run[metric]) for run in lhs_runs]
        rhs_values = [float(run[metric]) for run in rhs_runs]
        diffs = [a - b for a, b in zip(lhs_values, rhs_values)]
        mean_diff, pvalue = exact_sign_flip(diffs)
        results[metric] = {
            "lhs_mean": sum(lhs_values) / len(lhs_values),
            "rhs_mean": sum(rhs_values) / len(rhs_values),
            "mean_diff": mean_diff,
            "pvalue": pvalue,
        }
    return results


def verdict(mean_diff: float) -> str:
    """Return a concise direction label."""

    if mean_diff > 0:
        return "前者更高"
    if mean_diff < 0:
        return "后者更高"
    return "两者相同"


def build_report() -> dict[str, object]:
    """Build the full report payload."""

    report: dict[str, object] = {
        "note": (
            "当前多数比较只有 3 个随机种子。对 3-seed 的双侧精确 sign-flip 检验，"
            "最小可达到的 p 值为 0.25，因此它更适合作为趋势验证，而不能作为强显著性证据。"
        ),
        "sections": {},
    }
    for section, pairs in COMPARISONS.items():
        section_rows: list[dict[str, object]] = []
        for lhs, rhs in pairs:
            lhs_runs = load_runs(Path(lhs.summary_path))
            rhs_runs = load_runs(Path(rhs.summary_path))
            pair_result = compare(lhs_runs, rhs_runs)
            for metric in METRICS:
                values = pair_result[metric]
                section_rows.append(
                    {
                        "comparison": f"{lhs.label} vs {rhs.label}",
                        "metric": metric,
                        "lhs_mean": values["lhs_mean"],
                        "rhs_mean": values["rhs_mean"],
                        "mean_diff": values["mean_diff"],
                        "pvalue": values["pvalue"],
                        "verdict": verdict(values["mean_diff"]),
                    }
                )
        report["sections"][section] = section_rows
    return report


def render_markdown(report: dict[str, object]) -> str:
    """Render a Chinese markdown report."""

    lines = [
        "# 关键实验显著性检验报告",
        "",
        report["note"],
        "",
    ]
    for section, rows in report["sections"].items():
        lines.append(f"## {section}")
        lines.append("")
        lines.append("| 比较 | 指标 | 前者均值 | 后者均值 | 均值差 | p 值 | 结论 |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for row in rows:
            lines.append(
                f"| {row['comparison']} | {row['metric']} | "
                f"{row['lhs_mean']:.4f} | {row['rhs_mean']:.4f} | "
                f"{row['mean_diff']:+.4f} | {row['pvalue']:.4f} | {row['verdict']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 论文写作建议",
            "",
            "1. 目前可以写“在 3 个随机种子下，多源预训练在多数关键指标上呈现更优趋势”。",
            "2. 不建议现在写“达到统计显著差异”，因为 3-seed 双侧精确检验无法支持 p < 0.05。",
            "3. 若要形成 IEEE Trans 更稳的统计证据，建议把关键实验补到 5 个或以上随机种子。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Write JSON and markdown outputs."""

    report = build_report()
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "paper_significance_report.json"
    md_path = out_dir / "paper_significance_report_cn.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8-sig")
    md_path.write_text(render_markdown(report), encoding="utf-8-sig")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
