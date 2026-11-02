"""Run dual-stream LOSO experiments over multiple seeds."""

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
    """Parse args, run per-seed LOSO jobs, and aggregate results."""

    parser = argparse.ArgumentParser(description="Multi-seed wrapper for dual-stream LOSO experiments")
    add_config_argument(parser)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--use-ea", action="store_true")
    parser.add_argument("--linear-probe", action="store_true")
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument(
        "--threshold-objective",
        choices=["accuracy", "balanced_acc", "f1", "precision", "specificity", "youden"],
        default="balanced_acc",
    )
    parser.add_argument("--selection-metric", choices=["loss", "auc"], default="loss")
    parser.add_argument("--spatial-out-dim", type=int, default=256)
    parser.add_argument("--temporal-out-dim", type=int, default=256)
    parser.add_argument("--spatial-hidden-channels", type=int, default=64)
    parser.add_argument("--spatial-temporal-channels", type=int, default=128)
    parser.add_argument("--classifier-hidden-dim", type=int, default=None)
    parser.add_argument("--classifier-dropout", type=float, default=0.6)
    parser.add_argument("--use-neuro-prior", action="store_true")
    parser.add_argument("--sampling-rate", type=float, default=128.0)
    parser.add_argument("--use-sampling-rate-embedding", action="store_true")
    parser.add_argument("--sampling-rate-options", nargs="+", type=float, default=[128.0, 256.0])
    parser.add_argument("--prior-out-dim", type=int, default=64)
    parser.add_argument("--augment-train", action="store_true")
    parser.add_argument("--augment-noise-std", type=float, default=0.0)
    parser.add_argument("--augment-max-shift", type=int, default=0)
    parser.add_argument("--augment-scale-min", type=float, default=1.0)
    parser.add_argument("--augment-scale-max", type=float, default=1.0)
    parser.add_argument("--augment-channel-dropout", type=float, default=0.0)
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
            "selection_metric",
            "spatial_out_dim",
            "temporal_out_dim",
            "spatial_hidden_channels",
            "spatial_temporal_channels",
            "classifier_hidden_dim",
            "classifier_dropout",
            "use_neuro_prior",
            "sampling_rate",
            "use_sampling_rate_embedding",
            "sampling_rate_options",
            "prior_out_dim",
            "augment_train",
            "augment_noise_std",
            "augment_max_shift",
            "augment_scale_min",
            "augment_scale_max",
            "augment_channel_dropout",
        ],
    )
    if args.no_checkpoint:
        args.checkpoint = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_summaries: list[dict[str, object]] = []
    metrics_to_aggregate = ["accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity"]

    for seed in args.seeds:
        seed_output_dir = output_dir / f"seed_{seed}"
        command = [
            args.python_exe,
            "scripts/run_dual_stream_loso_subject.py",
            "--config",
            str(args.config),
            "--data-dir",
            args.data_dir,
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
            "--selection-metric",
            str(args.selection_metric),
            "--spatial-out-dim",
            str(args.spatial_out_dim),
            "--temporal-out-dim",
            str(args.temporal_out_dim),
            "--spatial-hidden-channels",
            str(args.spatial_hidden_channels),
            "--spatial-temporal-channels",
            str(args.spatial_temporal_channels),
            "--classifier-dropout",
            str(args.classifier_dropout),
            "--sampling-rate",
            str(args.sampling_rate),
            "--prior-out-dim",
            str(args.prior_out_dim),
        ]
        if args.classifier_hidden_dim is not None:
            command.extend(["--classifier-hidden-dim", str(args.classifier_hidden_dim)])
        if args.checkpoint:
            command.extend(["--checkpoint", args.checkpoint])
        if args.no_checkpoint:
            command.append("--no-checkpoint")
        if args.use_ea:
            command.append("--use-ea")
        if args.linear_probe:
            command.append("--linear-probe")
        if args.disable_class_weights:
            command.append("--disable-class-weights")
        if args.tune_threshold:
            command.append("--tune-threshold")
            command.extend(["--threshold-objective", args.threshold_objective])
        if args.use_neuro_prior:
            command.append("--use-neuro-prior")
        if args.use_sampling_rate_embedding:
            command.append("--use-sampling-rate-embedding")
            command.append("--sampling-rate-options")
            command.extend(str(value) for value in args.sampling_rate_options)
        if args.augment_train:
            command.append("--augment-train")
            command.extend(
                [
                    "--augment-noise-std",
                    str(args.augment_noise_std),
                    "--augment-max-shift",
                    str(args.augment_max_shift),
                    "--augment-scale-min",
                    str(args.augment_scale_min),
                    "--augment-scale-max",
                    str(args.augment_scale_max),
                    "--augment-channel-dropout",
                    str(args.augment_channel_dropout),
                ]
            )

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
        "checkpoint": args.checkpoint,
        "seeds": args.seeds,
        "use_ea": args.use_ea,
        "linear_probe": args.linear_probe,
        "disable_class_weights": args.disable_class_weights,
        "tune_threshold": args.tune_threshold,
        "threshold_objective": args.threshold_objective if args.tune_threshold else None,
        "selection_metric": args.selection_metric,
        "spatial_out_dim": args.spatial_out_dim,
        "temporal_out_dim": args.temporal_out_dim,
        "spatial_hidden_channels": args.spatial_hidden_channels,
        "spatial_temporal_channels": args.spatial_temporal_channels,
        "classifier_hidden_dim": args.classifier_hidden_dim,
        "classifier_dropout": args.classifier_dropout,
        "use_neuro_prior": args.use_neuro_prior,
        "sampling_rate": args.sampling_rate,
        "use_sampling_rate_embedding": args.use_sampling_rate_embedding,
        "sampling_rate_options": args.sampling_rate_options,
        "prior_out_dim": args.prior_out_dim,
        "augment_train": args.augment_train,
        "augment_noise_std": args.augment_noise_std if args.augment_train else None,
        "augment_max_shift": args.augment_max_shift if args.augment_train else None,
        "augment_scale_min": args.augment_scale_min if args.augment_train else None,
        "augment_scale_max": args.augment_scale_max if args.augment_train else None,
        "augment_channel_dropout": args.augment_channel_dropout if args.augment_train else None,
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

    print("\nAggregate multi-seed LOSO results")
    for metric, values in aggregate_metrics.items():
        print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
