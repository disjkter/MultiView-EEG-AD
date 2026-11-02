"""Run the dual-stream cross-domain experiment over multiple seeds."""

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
    """Parse arguments, run per-seed jobs, and aggregate summaries."""

    parser = argparse.ArgumentParser(description="Multi-seed wrapper for dual-stream cross-domain experiments")
    add_config_argument(parser)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
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
    parser.add_argument("--decision-threshold", type=float, default=None)
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--threshold-objective", choices=["balanced_acc", "f1", "youden"], default="balanced_acc")
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
        ],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_summaries: list[dict[str, object]] = []
    metrics_to_aggregate = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "auc",
        "sensitivity",
        "specificity",
    ]

    for seed in args.seeds:
        seed_output_dir = output_dir / f"seed_{seed}"
        command = [
            args.python_exe,
            "scripts/run_dual_stream_cross_domain.py",
            "--config",
            str(args.config),
            "--source-dir",
            args.source_dir,
            "--target-dir",
            args.target_dir,
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
        ]
        if args.checkpoint:
            command.extend(["--checkpoint", args.checkpoint])
        if args.use_ea:
            command.append("--use-ea")
        if args.linear_probe:
            command.append("--linear-probe")
        if args.disable_class_weights:
            command.append("--disable-class-weights")
        if args.decision_threshold is not None:
            command.extend(["--decision-threshold", str(args.decision_threshold)])
        if args.tune_threshold:
            command.append("--tune-threshold")
            command.extend(["--threshold-objective", args.threshold_objective])

        print(f"\nRunning seed {seed}")
        subprocess.run(command, check=True)

        summary_path = seed_output_dir / "summary.json"
        seed_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    aggregate_metrics: dict[str, dict[str, float]] = {}
    for metric in metrics_to_aggregate:
        values = [float(summary["target_metrics"][metric]) for summary in seed_summaries]
        aggregate_metrics[metric] = summarize_metric(values)

    aggregate = {
        "source_dir": args.source_dir,
        "target_dir": args.target_dir,
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
                "seed": summary["seed"],
                "best_epoch": summary["best_epoch"],
                "decision_threshold": summary.get("decision_threshold"),
                "target_metrics": summary["target_metrics"],
            }
            for summary in seed_summaries
        ],
    }

    with open(output_dir / "aggregate_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(aggregate, file_obj, ensure_ascii=False, indent=2)

    print("\nAggregate multi-seed results")
    for metric, values in aggregate_metrics.items():
        print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
