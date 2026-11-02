"""Run paired significance tests for source-only calibration baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aggregate_margin_alignment import add_holm_adjustment, exact_sign_flip


METRICS = ("ece", "brier", "nll", "balanced_accuracy")


def load_rows(path: Path) -> list[dict[str, str]]:
    """Load seed-macro calibration results."""

    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_tests(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Compare each fitted calibrator with identity at matched seeds."""

    indexed = {
        (row["variant"], row["calibrator"], int(row["seed"])): row
        for row in rows
    }
    variants = sorted({row["variant"] for row in rows})
    comparisons: list[dict[str, object]] = []
    for metric in METRICS:
        metric_rows: list[dict[str, object]] = []
        for variant in variants:
            for calibrator in ("temperature", "platt"):
                seeds = sorted(
                    seed
                    for candidate_variant, candidate_calibrator, seed in indexed
                    if candidate_variant == variant
                    and candidate_calibrator == calibrator
                    and (variant, "identity", seed) in indexed
                )
                differences = [
                    float(indexed[(variant, calibrator, seed)][metric])
                    - float(indexed[(variant, "identity", seed)][metric])
                    for seed in seeds
                ]
                difference_std = stdev(differences) if len(differences) > 1 else 0.0
                metric_rows.append(
                    {
                        "variant": variant,
                        "calibrator": calibrator,
                        "baseline": "identity",
                        "metric": metric,
                        "lower_is_better": metric in {"ece", "brier", "nll"},
                        "num_pairs": len(differences),
                        "mean_difference": mean(differences),
                        "difference_std": difference_std,
                        "cohen_dz": (
                            mean(differences) / difference_std
                            if difference_std > 0
                            else None
                        ),
                        "improved_seeds": sum(
                            value < 0
                            if metric in {"ece", "brier", "nll"}
                            else value > 0
                            for value in differences
                        ),
                        "ties": sum(value == 0 for value in differences),
                        "exact_sign_flip_pvalue": exact_sign_flip(differences),
                    }
                )
        comparisons.extend(add_holm_adjustment(metric_rows))
    return comparisons


def main() -> None:
    """Load seed results and save corrected paired comparisons."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-results", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    comparisons = paired_tests(load_rows(Path(args.seed_results)))
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(comparisons)} paired calibration comparisons")


if __name__ == "__main__":
    main()
