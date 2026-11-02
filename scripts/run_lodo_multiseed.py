"""Run dual-stream LODO experiments over multiple seeds."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dual_stream_config import add_config_argument, apply_yaml_defaults


def summarize_metric(values: list[float]) -> dict[str, float]:
    """Compute mean and standard deviation for one metric."""

    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {"mean": float(mean), "std": float(std)}


def main() -> None:
    """Parse args, run per-seed LODO jobs, and aggregate results."""

    parser = argparse.ArgumentParser(description="Multi-seed wrapper for dual-stream LODO experiments")
    add_config_argument(parser)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--domains", nargs="+", required=True, help="Format: name=path")
    parser.add_argument("--method", choices=["plain", "dann"], default="plain")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use-ea", action="store_true")
    parser.add_argument("--linear-probe", action="store_true")
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--threshold-objective", choices=["balanced_acc", "f1", "youden"], default="balanced_acc")
    parser.add_argument("--lambda-domain-max", type=float, default=0.5)
    args = parser.parse_args()
    args = apply_yaml_defaults(
        parser,
        args,
        [
            "checkpoint",
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "val_ratio",
            "patience",
            "use_ea",
            "linear_probe",
            "disable_class_weights",
            "tune_threshold",
            "threshold_objective",
            "lambda_domain_max",
        ],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_summaries: list[dict[str, object]] = []
    metrics_to_aggregate = ["accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity"]

    for seed in args.seeds:
        seed_output_dir = output_dir / f"seed_{seed}"
        command = [
            args.python_exe,
            "scripts/run_dual_stream_lodo.py",
            "--config",
            str(args.config),
            "--domains",
            *args.domains,
            "--method",
            args.method,
            "--output-dir",
            str(seed_output_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--val-ratio",
            str(args.val_ratio),
            "--patience",
            str(args.patience),
            "--seed",
            str(seed),
            "--lambda-domain-max",
            str(args.lambda_domain_max),
        ]
        if args.checkpoint:
            command.extend(["--checkpoint", args.checkpoint])
        if args.use_ea:
            command.append("--use-ea")
        if args.linear_probe:
            command.append("--linear-probe")
        if args.disable_class_weights:
            command.append("--disable-class-weights")
        if args.tune_threshold:
            command.append("--tune-threshold")
            command.extend(["--threshold-objective", args.threshold_objective])

        print(f"\nRunning seed {seed}")
        subprocess.run(command, check=True)
        summary_path = seed_output_dir / "lodo_summary.json"
        seed_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    aggregate_metrics: dict[str, dict[str, float]] = {}
    for metric in metrics_to_aggregate:
        values = [float(summary["aggregate_mean"][metric]) for summary in seed_summaries]
        aggregate_metrics[metric] = summarize_metric(values)

    aggregate = {
        "domains": args.domains,
        "method": args.method,
        "checkpoint": args.checkpoint,
        "seeds": args.seeds,
        "use_ea": args.use_ea,
        "linear_probe": args.linear_probe,
        "disable_class_weights": args.disable_class_weights,
        "tune_threshold": args.tune_threshold,
        "threshold_objective": args.threshold_objective if args.tune_threshold else None,
        "metrics": aggregate_metrics,
        "seed_runs": [
            {
                "seed": seed,
                "aggregate_mean": summary["aggregate_mean"],
                "folds": summary["folds"],
            }
            for seed, summary in zip(args.seeds, seed_summaries)
        ],
    }

    with (output_dir / "aggregate_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(aggregate, fp, ensure_ascii=False, indent=2)

    print("\nAggregate multi-seed LODO results")
    for metric, values in aggregate_metrics.items():
        print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
