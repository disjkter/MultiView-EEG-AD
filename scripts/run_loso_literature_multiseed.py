"""Run literature-inspired LOSO baselines over multiple seeds."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


def summarize_metric(values: list[float]) -> dict[str, float]:
    """Compute mean and standard deviation for one metric."""

    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return {"mean": float("nan"), "std": float("nan"), "num_valid": 0}

    mean = statistics.fmean(finite_values)
    std = statistics.pstdev(finite_values) if len(finite_values) > 1 else 0.0
    return {"mean": float(mean), "std": float(std), "num_valid": len(finite_values)}


def build_command(args: argparse.Namespace, seed: int, seed_output_dir: Path) -> list[str]:
    """Build the command for one seeded literature baseline run."""

    if args.model == "fc_svm":
        command = [
            args.python_exe,
            "scripts/run_loso_fc_svm.py",
            "--data-dir",
            args.data_dir,
            "--output-dir",
            str(seed_output_dir),
            "--val-ratio",
            str(args.val_ratio),
            "--seed",
            str(seed),
        ]
    else:
        command = [
            args.python_exe,
            "scripts/run_loso_spectrogram_cnn.py",
            "--data-dir",
            args.data_dir,
            "--output-dir",
            str(seed_output_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--dropout",
            str(args.dropout),
            "--base-channels",
            str(args.base_channels),
            "--val-ratio",
            str(args.val_ratio),
            "--patience",
            str(args.patience),
            "--n-fft",
            str(args.n_fft),
            "--hop-length",
            str(args.hop_length),
            "--device",
            args.device,
            "--seed",
            str(seed),
        ]

    if args.use_ea:
        command.append("--use-ea")
    if args.tune_threshold:
        command.append("--tune-threshold")
        command.extend(["--threshold-objective", args.threshold_objective])
    if args.max_subjects is not None:
        command.extend(["--max-subjects", str(args.max_subjects)])

    return command


def main() -> None:
    """Parse args, run per-seed jobs, and aggregate results."""

    parser = argparse.ArgumentParser(description="Multi-seed wrapper for literature-inspired LOSO baselines")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", choices=["fc_svm", "spectrogram_cnn"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--use-ea", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--threshold-objective", choices=["balanced_acc", "f1"], default="balanced_acc")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--n-fft", type=int, default=64)
    parser.add_argument("--hop-length", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_summaries: list[dict[str, object]] = []
    metrics_to_aggregate = ["accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity"]

    for seed in args.seeds:
        seed_output_dir = output_dir / f"seed_{seed}"
        command = build_command(args, seed, seed_output_dir)
        print(f"\nRunning seed {seed}")
        subprocess.run(command, check=True)
        summary_path = seed_output_dir / "summary.json"
        seed_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    aggregate_metrics: dict[str, dict[str, float]] = {}
    for metric in metrics_to_aggregate:
        values = [float(summary["metrics"][metric]) for summary in seed_summaries]
        aggregate_metrics[metric] = summarize_metric(values)

    aggregate = {
        "data_dir": args.data_dir,
        "model": args.model,
        "seeds": args.seeds,
        "use_ea": args.use_ea,
        "tune_threshold": args.tune_threshold,
        "threshold_objective": args.threshold_objective if args.tune_threshold else None,
        "metrics": aggregate_metrics,
        "seed_runs": [
            {
                "seed": seed,
                "metrics": summary["metrics"],
            }
            for seed, summary in zip(args.seeds, seed_summaries)
        ],
    }

    with (output_dir / "aggregate_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(aggregate, fp, ensure_ascii=False, indent=2)

    print("\nAggregate multi-seed literature LOSO results")
    for metric, values in aggregate_metrics.items():
        print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
