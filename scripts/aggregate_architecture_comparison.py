"""Aggregate paired multi-seed architecture comparison results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


METRICS = (
    "accuracy",
    "balanced_accuracy",
    "f1",
    "auc",
    "sensitivity",
    "specificity",
)
MODEL_ORDER = ("dual_stream", "mscnn_only", "graph_temporal")
PAIR_ORDER = (
    ("graph_temporal", "dual_stream"),
    ("graph_temporal", "mscnn_only"),
    ("mscnn_only", "dual_stream"),
)


def load_runs(results_dir: Path) -> list[dict[str, Any]]:
    """Load valid experiment summaries below a comparison directory."""

    runs: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*_seed*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = str(payload.get("model_type", path.parent.name.rsplit("_seed", 1)[0]))
        if model not in MODEL_ORDER:
            continue
        metrics = payload.get("metrics", {})
        missing = [metric for metric in METRICS if metric not in metrics]
        if missing:
            raise ValueError(f"{path} is missing metrics: {missing}")
        runs.append(
            {
                "model": model,
                "seed": int(payload["seed"]),
                "best_epoch": int(payload.get("best_epoch", -1)),
                "path": str(path),
                **{metric: float(metrics[metric]) for metric in METRICS},
            }
        )
    if not runs:
        raise FileNotFoundError(f"No *_seed*/summary.json files found under {results_dir}")
    return runs


def bootstrap_mean_ci(values: np.ndarray, seed: int = 2026) -> tuple[float, float]:
    """Return a percentile 95% CI for the mean paired difference."""

    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def paired_summary(
    left_model: str,
    right_model: str,
    by_model: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Compute paired left-model minus right-model statistics."""

    left_by_seed = {row["seed"]: row for row in by_model[left_model]}
    right_by_seed = {row["seed"]: row for row in by_model[right_model]}
    paired_seeds = sorted(set(left_by_seed).intersection(right_by_seed))
    if not paired_seeds:
        return None

    paired: dict[str, Any] = {
        "left_model": left_model,
        "right_model": right_model,
        "difference": f"{left_model}_minus_{right_model}",
        "seeds": paired_seeds,
        "metrics": {},
    }
    for metric in METRICS:
        left = np.asarray([left_by_seed[seed][metric] for seed in paired_seeds], dtype=np.float64)
        right = np.asarray([right_by_seed[seed][metric] for seed in paired_seeds], dtype=np.float64)
        differences = left - right
        ci_low, ci_high = bootstrap_mean_ci(differences)
        t_result = ttest_rel(left, right)
        try:
            w_result = wilcoxon(left, right, alternative="two-sided")
            wilcoxon_stat = float(w_result.statistic)
            wilcoxon_p = float(w_result.pvalue)
        except ValueError:
            wilcoxon_stat = 0.0
            wilcoxon_p = 1.0
        paired["metrics"][metric] = {
            left_model: left.tolist(),
            right_model: right.tolist(),
            "differences": differences.tolist(),
            "mean_difference": float(differences.mean()),
            "ci95_bootstrap": [ci_low, ci_high],
            "wins": int(np.sum(differences > 0)),
            "ties": int(np.sum(differences == 0)),
            "losses": int(np.sum(differences < 0)),
            "paired_t_statistic": float(t_result.statistic),
            "paired_t_pvalue": float(t_result.pvalue),
            "wilcoxon_statistic": wilcoxon_stat,
            "wilcoxon_pvalue": wilcoxon_p,
        }
    return paired


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute model summaries and all available paired comparisons."""

    by_model = {
        model: sorted((run for run in runs if run["model"] == model), key=lambda row: row["seed"])
        for model in MODEL_ORDER
    }
    model_summary: dict[str, Any] = {}
    for model, rows in by_model.items():
        if not rows:
            continue
        model_summary[model] = {
            "n": len(rows),
            "seeds": [row["seed"] for row in rows],
            "metrics": {
                metric: {
                    "mean": float(np.mean([row[metric] for row in rows])),
                    "std": float(np.std([row[metric] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
                }
                for metric in METRICS
            },
        }

    comparisons: dict[str, Any] = {}
    for left_model, right_model in PAIR_ORDER:
        comparison = paired_summary(left_model, right_model, by_model)
        if comparison is not None:
            comparisons[f"{left_model}_vs_{right_model}"] = comparison
    if not comparisons:
        raise ValueError("No model pair has matching seeds.")

    result = {"models": model_summary, "pairwise_comparisons": comparisons}
    legacy_key = "graph_temporal_vs_dual_stream"
    if legacy_key in comparisons:
        result["paired_comparison"] = comparisons[legacy_key]
    return result


def write_runs_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    """Write one row per model and seed."""

    fields = ("model", "seed", "best_epoch", *METRICS, "path")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in runs)


def print_summary(summary: dict[str, Any]) -> None:
    """Print compact model and paired-comparison tables."""

    print("\nMean +/- sample standard deviation")
    for model in MODEL_ORDER:
        if model not in summary["models"]:
            continue
        print(f"\n{model} (n={summary['models'][model]['n']})")
        for metric in METRICS:
            values = summary["models"][model]["metrics"][metric]
            print(f"  {metric:18s}: {values['mean']:.4f} +/- {values['std']:.4f}")

    for name, comparison in summary["pairwise_comparisons"].items():
        print(f"\nPaired comparison: {comparison['difference']}")
        for metric in METRICS:
            values = comparison["metrics"][metric]
            low, high = values["ci95_bootstrap"]
            print(
                f"  {metric:18s}: delta={values['mean_difference']:+.4f} "
                f"CI=[{low:+.4f}, {high:+.4f}] "
                f"W/T/L={values['wins']}/{values['ties']}/{values['losses']} "
                f"t-p={values['paired_t_pvalue']:.4g} "
                f"wilcoxon-p={values['wilcoxon_pvalue']:.4g}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    runs = load_runs(results_dir)
    summary = summarize(runs)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    write_runs_csv(csv_path, runs)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(summary)
    print(f"\nSaved {csv_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
