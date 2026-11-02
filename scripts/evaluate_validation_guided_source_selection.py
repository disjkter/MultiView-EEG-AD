"""Evaluate source initialization selected only by target validation loss."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_transfer_comparison import (  # noqa: E402
    METRICS,
    bootstrap_ci,
    exact_sign_flip_pvalue,
)


def choose_source_by_validation_loss(
    candidates: dict[str, dict[str, object]],
) -> tuple[str, dict[str, object]]:
    """Return the candidate with the lowest best validation loss."""

    if not candidates:
        raise ValueError("At least one source candidate is required.")
    for name, payload in candidates.items():
        if payload.get("best_val_loss") is None:
            raise ValueError(f"Candidate {name} has no best_val_loss.")
    name = min(candidates, key=lambda item: float(candidates[item]["best_val_loss"]))
    return name, candidates[name]


def load_runs(
    root: Path,
    target: str,
    seeds: list[int],
    baseline_variant: str,
    source_variants: list[str],
) -> pd.DataFrame:
    """Load candidates, select by validation loss, and retain test metrics."""

    rows: list[dict[str, object]] = []
    for seed in seeds:
        run_root = root / f"support4_seed{seed}" / target
        baseline_path = run_root / baseline_variant / "summary.json"
        if not baseline_path.exists():
            raise FileNotFoundError(f"Missing summary: {baseline_path}")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        candidates: dict[str, dict[str, object]] = {}
        for variant in source_variants:
            path = run_root / variant / "summary.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing summary: {path}")
            candidates[variant] = json.loads(path.read_text(encoding="utf-8"))
        selected_name, selected = choose_source_by_validation_loss(candidates)
        row: dict[str, object] = {
            "seed": seed,
            "selected_source": selected_name,
            "selected_val_loss": float(selected["best_val_loss"]),
        }
        for metric in METRICS:
            baseline_value = float(baseline["metrics"][metric])
            selected_value = float(selected["metrics"][metric])
            row[f"baseline_{metric}"] = baseline_value
            row[f"selected_{metric}"] = selected_value
            row[f"gain_{metric}"] = selected_value - baseline_value
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_runs(
    runs: pd.DataFrame,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Compute paired uncertainty and exact tests for selected-source gains."""

    rows: list[dict[str, object]] = []
    for metric_index, metric in enumerate(METRICS):
        differences = runs[f"gain_{metric}"].to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_ci(
            differences,
            samples=bootstrap_samples,
            seed=bootstrap_seed + metric_index,
        )
        rows.append(
            {
                "metric": metric,
                "baseline_mean": float(runs[f"baseline_{metric}"].mean()),
                "selected_mean": float(runs[f"selected_{metric}"].mean()),
                "mean_gain": float(differences.mean()),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
                "positive_seeds": int(np.sum(differences > 0)),
                "negative_seeds": int(np.sum(differences < 0)),
                "ties": int(np.sum(differences == 0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run validation-guided source selection analysis."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--target", default="CAUEEG")
    parser.add_argument("--baseline-variant", default="target_only_calibrated")
    parser.add_argument("--source-variants", nargs="+", required=True)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(range(2026, 2036))
    )
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    runs = load_runs(
        Path(args.root),
        args.target,
        args.seeds,
        args.baseline_variant,
        args.source_variants,
    )
    summary = summarize_runs(runs, args.bootstrap_samples, args.bootstrap_seed)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_prefix.with_name(output_prefix.name + "_runs.csv"), index=False)
    summary.to_csv(
        output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False
    )
    output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "selection_rule": "minimum best_val_loss; test metrics unused",
                "target": args.target,
                "source_variants": args.source_variants,
                "seeds": args.seeds,
                "selection_counts": runs["selected_source"].value_counts().to_dict(),
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(runs[["seed", "selected_source", "selected_val_loss"]].to_string(index=False))
    print(summary.to_string(index=False))
    print(f"Saved validation-guided selection outputs with prefix: {output_prefix}")


if __name__ == "__main__":
    main()
