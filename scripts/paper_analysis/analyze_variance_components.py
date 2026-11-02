"""Estimate descriptive fixed-effect contributions across cohorts and methods."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper_analysis.common import parse_named_path, write_csv, write_json


def dummy_matrix(values: list[str]) -> np.ndarray:
    """Return treatment-coded categorical columns excluding one reference level."""

    levels = sorted(set(values))
    return np.column_stack(
        [np.asarray([value == level for value in values], dtype=np.float64) for level in levels[1:]]
    ) if len(levels) > 1 else np.empty((len(values), 0))


def design(rows: list[dict[str, object]], factors: list[str]) -> np.ndarray:
    """Construct an intercept plus categorical fixed-effect design matrix."""

    columns = [np.ones((len(rows), 1), dtype=np.float64)]
    for factor in factors:
        columns.append(dummy_matrix([str(row[factor]) for row in rows]))
    return np.column_stack(columns)


def residual_sum_squares(y: np.ndarray, matrix: np.ndarray) -> float:
    """Return ordinary least-squares residual sum of squares."""

    fitted = matrix @ np.linalg.lstsq(matrix, y, rcond=None)[0]
    return float(np.sum((y - fitted) ** 2))


def effect_rows(rows: list[dict[str, object]], metric: str) -> list[dict[str, object]]:
    """Compute partial R-squared for prespecified categorical effects."""

    factors = ["target", "variant", "backbone", "seed"]
    y = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    full_sse = residual_sum_squares(y, design(rows, factors))
    output: list[dict[str, object]] = []
    for factor in factors[:-1]:
        reduced = [name for name in factors if name != factor]
        reduced_sse = residual_sum_squares(y, design(rows, reduced))
        extra = max(0.0, reduced_sse - full_sse)
        output.append(
            {
                "metric": metric,
                "effect": factor,
                "n_observations": len(rows),
                "levels": len({str(row[factor]) for row in rows}),
                "partial_r_squared": float(extra / (extra + full_sse)) if extra + full_sse else 0.0,
                "interpretation": "descriptive fixed-effect contribution; not a population site variance estimate",
            }
        )
    return output


def main() -> None:
    """Combine matched backbone fold tables and estimate effect contributions."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", action="append", required=True, metavar="BACKBONE=CSV")
    parser.add_argument(
        "--metrics", nargs="+",
        default=["calibrated_balanced_accuracy", "cohort_z_balanced_accuracy", "calibrated_auc"],
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for value in args.folds:
        backbone, path = parse_named_path(value)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                row["backbone"] = backbone
                rows.append(row)
    results = [item for metric in args.metrics for item in effect_rows(rows, metric)]
    prefix = args.output_prefix
    write_csv(prefix.with_suffix(".csv"), results)
    write_json(
        prefix.with_suffix(".json"),
        {
            "metadata": {
                "model": "metric ~ target + variant + backbone + seed",
                "balanced_matched_design_required": True,
                "inferential_claim": False,
            },
            "effects": results,
        },
    )
    print(f"Saved descriptive variance analysis with {len(rows)} fold observations")


if __name__ == "__main__":
    main()
