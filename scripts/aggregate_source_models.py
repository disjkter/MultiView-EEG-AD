"""Aggregate repeated subject-level source-model training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = (
    "accuracy",
    "balanced_accuracy",
    "f1",
    "auc",
    "sensitivity",
    "specificity",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--run-pattern", default="ds004504_seed{seed}")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2026, 2036)))
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def load_runs(results_dir: Path, run_pattern: str, seeds: list[int]) -> pd.DataFrame:
    """Load expected summaries and fail when a run or checkpoint is missing."""

    rows: list[dict[str, object]] = []
    for seed in seeds:
        run_dir = results_dir / run_pattern.format(seed=seed)
        summary_path = run_dir / "summary.json"
        checkpoint_path = run_dir / "best_model.pth"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary: {summary_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        rows.append(
            {
                "seed": seed,
                "best_epoch": payload.get("best_epoch"),
                "best_val_loss": payload.get("best_val_loss"),
                "best_val_auc": payload.get("best_val_auc"),
                **{metric: float(metrics[metric]) for metric in METRICS},
            }
        )
    return pd.DataFrame(rows)


def aggregate_runs(runs: pd.DataFrame) -> dict[str, object]:
    """Return JSON-serializable descriptive statistics."""

    aggregate: dict[str, object] = {"num_runs": int(len(runs))}
    for metric in ("best_epoch", "best_val_loss", "best_val_auc", *METRICS):
        values = runs[metric]
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return aggregate


def main() -> None:
    """Aggregate source-model summaries and save CSV/JSON outputs."""

    args = parse_args()
    runs = load_runs(args.results_dir, args.run_pattern, args.seeds)
    aggregate = aggregate_runs(runs)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_name(args.output_prefix.name + "_runs.csv")
    json_path = args.output_prefix.with_suffix(".json")
    runs.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print(runs.to_string(index=False))
    print("\nAggregate")
    for metric in METRICS:
        stats = aggregate[metric]
        print(f"{metric:17s}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
