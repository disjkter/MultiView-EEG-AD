"""Paired statistical analysis for the fixed-split SSL objective ablation.

The complete objective is compared with every prespecified comparator on the
same ten participant splits. AUROC is the primary endpoint. Confidence
intervals use a paired bootstrap over split-level differences, and p-values
use an exact two-sided sign-flip test followed by Holm correction.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np


COMPARATORS = (
    "random_init",
    "masked_signal_only",
    "minus_signal",
    "minus_spectrum",
    "minus_branch",
    "minus_crop",
)
LABELS = {
    "random_init": "Random initialization",
    "masked_signal_only": "Masked signal only",
    "minus_signal": "Without signal reconstruction",
    "minus_spectrum": "Without spectral reconstruction",
    "minus_branch": "Without cross-branch agreement",
    "minus_crop": "Without temporal-view consistency",
}


def exact_sign_flip(differences: np.ndarray) -> float:
    """Return the exact two-sided sign-flip p-value for the mean difference."""

    observed = abs(float(differences.mean()))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(differences))))
    permuted = np.abs((signs * differences[None, :]).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def bootstrap_ci(
    differences: np.ndarray, *, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Return a percentile 95% CI from paired split-level bootstrap samples."""

    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def holm_adjust(raw_p_values: list[float]) -> list[float]:
    """Apply Holm's step-down family-wise error correction."""

    order = np.argsort(raw_p_values)
    adjusted = np.empty(len(raw_p_values), dtype=np.float64)
    running = 0.0
    count = len(raw_p_values)
    for rank, original_index in enumerate(order):
        candidate = (count - rank) * raw_p_values[int(original_index)]
        running = max(running, candidate)
        adjusted[int(original_index)] = min(1.0, running)
    return adjusted.tolist()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write dictionaries to a UTF-8 CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latex_p(value: float) -> str:
    """Format a p-value for a compact manuscript table."""

    return "$<0.001$" if value < 0.001 else f"{value:.3f}"


def write_latex(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the primary AUROC comparisons as a paper-ready LaTeX table."""

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Paired comparison of the complete pretraining objective against prespecified alternatives on the same ten CAUEEG splits. $\Delta$AUROC is complete pretraining minus the comparator. Confidence intervals use 100,000 paired bootstrap resamples. Raw $p$-values are from exact two-sided sign-flip tests; $p_{\mathrm{Holm}}$ controls the family-wise error rate across six comparisons.}",
        r"\label{tab:objective_statistics}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Comparator & Mean $\Delta$AUROC & 95\% CI & Wins & Raw $p$ & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['comparator_label']} & {float(row['mean_difference']):+.3f} & "
            f"[{float(row['ci_lower']):+.3f}, {float(row['ci_upper']):+.3f}] & "
            f"{row['wins']}/{row['num_pairs']} & {latex_p(float(row['raw_p']))} & "
            f"{latex_p(float(row['holm_p']))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Load verified runs and export paired AUROC statistics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="paper/byqiu_revision/verified_results_20260928/aggregate.json",
    )
    parser.add_argument(
        "--output-prefix",
        default="paper/byqiu_revision/verified_results_20260928/objective_paired_statistics",
    )
    parser.add_argument("--metric", default="auc")
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260928)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    objective_runs = [
        row for row in payload["runs"] if row["experiment"] == "pretraining_objective"
    ]
    lookup = {
        (str(row["condition"]), int(row["seed"])): float(row[args.metric])
        for row in objective_runs
    }
    seeds = sorted(seed for condition, seed in lookup if condition == "full")
    if len(seeds) != 10:
        raise ValueError(f"Expected 10 complete-objective seeds, found {len(seeds)}")

    rng = np.random.default_rng(args.bootstrap_seed)
    summary_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for comparator in COMPARATORS:
        complete = np.asarray([lookup[("full", seed)] for seed in seeds])
        reference = np.asarray([lookup[(comparator, seed)] for seed in seeds])
        differences = complete - reference
        lower, upper = bootstrap_ci(differences, samples=args.bootstrap_samples, rng=rng)
        summary_rows.append(
            {
                "metric": args.metric,
                "complete_condition": "full",
                "comparator": comparator,
                "comparator_label": LABELS[comparator],
                "num_pairs": len(seeds),
                "mean_complete": float(complete.mean()),
                "mean_comparator": float(reference.mean()),
                "mean_difference": float(differences.mean()),
                "ci_lower": lower,
                "ci_upper": upper,
                "wins": int(np.sum(differences > 0)),
                "ties": int(np.sum(np.isclose(differences, 0.0, atol=1e-15))),
                "losses": int(np.sum(differences < 0)),
                "raw_p": exact_sign_flip(differences),
            }
        )
        for seed, complete_value, comparator_value, difference in zip(
            seeds, complete, reference, differences
        ):
            paired_rows.append(
                {
                    "metric": args.metric,
                    "seed": seed,
                    "split_id": f"CAUEEG_processed_19ch_ad_vs_hc_1s_lead_seed{seed}",
                    "complete_condition": "full",
                    "comparator": comparator,
                    "complete_value": float(complete_value),
                    "comparator_value": float(comparator_value),
                    "paired_difference": float(difference),
                }
            )

    adjusted = holm_adjust([float(row["raw_p"]) for row in summary_rows])
    for row, adjusted_p in zip(summary_rows, adjusted):
        row["holm_p"] = adjusted_p
        row["holm_significant_0_05"] = adjusted_p < 0.05

    prefix = Path(args.output_prefix)
    write_csv(prefix.with_suffix(".csv"), summary_rows)
    write_csv(prefix.with_name(prefix.name + "_per_seed.csv"), paired_rows)
    output = {
        "analysis": {
            "metric": args.metric,
            "paired_unit": "fixed participant split / seed",
            "num_pairs": len(seeds),
            "seeds": seeds,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "confidence_interval": "paired percentile bootstrap",
            "test": "exact two-sided sign-flip test over all 2^10 assignments",
            "multiplicity": "Holm correction across six prespecified comparisons",
        },
        "comparisons": summary_rows,
        "per_seed": paired_rows,
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_latex(prefix.parent.parent / "objective_statistics_table.tex", summary_rows)
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
