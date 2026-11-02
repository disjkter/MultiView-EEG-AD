"""Evaluate interpolated SSL and source-task encoder checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def require_file(path: Path) -> None:
    """Require one input file."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing required input: {path}")


def run_checked(command: list[str], output: Path, label: str) -> None:
    """Run a resumable command and require its output."""

    if output.is_file():
        print(f"SKIP {label}", flush=True)
        return
    print(f"START {label}", flush=True)
    subprocess.run(command, check=True)
    require_file(output)
    print(f"DONE {label}", flush=True)


def weight_name(value: float) -> str:
    """Return a filesystem-safe interpolation label."""

    return f"source_weight_{value:.2f}".replace(".", "p")


def main() -> None:
    """Create interpolated checkpoints and fine-tune them on matched target splits."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/checkpoint_interpolation_pilot.yaml")
    parser.add_argument("--python-exe", default=sys.executable)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    data = config["data"]
    checkpoints = config["checkpoints"]
    training = config["target_training"]
    evaluation = config["evaluation"]
    output_root = Path(experiment["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    ssl_checkpoint = Path(checkpoints["ssl"])
    require_file(ssl_checkpoint)

    common = [
        "--model-type", "dual_stream",
        "--epochs", str(training["epochs"]),
        "--batch-size", str(training["batch_size"]),
        "--lr", str(training["learning_rate"]),
        "--weight-decay", str(training["weight_decay"]),
        "--patience", str(training["patience"]),
        "--max-train-segments-per-subject", str(training["max_train_segments_per_subject"]),
        "--max-val-segments-per-subject", str(training["max_val_segments_per_subject"]),
        "--max-test-segments-per-subject", str(training["max_test_segments_per_subject"]),
        "--alignment", str(evaluation["alignment"]),
        "--max-ea-samples", str(evaluation["max_alignment_samples"]),
        "--selection-metric", str(evaluation["selection_metric"]),
    ]
    if bool(evaluation["tune_threshold"]):
        common.extend(
            ["--tune-threshold", "--threshold-objective", str(evaluation["threshold_objective"])]
        )

    for seed in [int(value) for value in experiment["seeds"]]:
        source_checkpoint = Path(str(checkpoints["source_pattern"]).format(seed=seed))
        require_file(source_checkpoint)
        for source_weight in [float(value) for value in experiment["interpolation_weights"]]:
            condition = weight_name(source_weight)
            merged_checkpoint = output_root / "checkpoints" / f"seed_{seed}" / f"{condition}.pth"
            run_checked(
                [
                    args.python_exe,
                    "scripts/interpolate_encoder_checkpoints.py",
                    "--ssl-checkpoint", str(ssl_checkpoint),
                    "--source-checkpoint", str(source_checkpoint),
                    "--source-weight", str(source_weight),
                    "--output", str(merged_checkpoint),
                ],
                merged_checkpoint,
                f"interpolate {condition} seed={seed}",
            )
            for support in [int(value) for value in experiment["supports_per_class"]]:
                split = Path(str(data["target_split_patterns"][str(support)]).format(seed=seed))
                require_file(split)
                payload = json.loads(split.read_text(encoding="utf-8"))
                if len(payload.get("train_subjects", [])) != 2 * support:
                    raise ValueError(f"Unexpected support split: {split}")
                output = output_root / "target" / f"support_{support}" / condition / f"seed_{seed}"
                run_checked(
                    [
                        args.python_exe,
                        "scripts/run_dual_stream_subject_split.py",
                        "--data-dir", str(data["target"]),
                        "--split-file", str(split),
                        "--checkpoint", str(merged_checkpoint),
                        "--checkpoint-scope", "encoders",
                        "--output-dir", str(output),
                        "--seed", str(seed),
                        *common,
                    ],
                    output / "summary.json",
                    f"target support={support} {condition} seed={seed}",
                )

    subprocess.run(
        [args.python_exe, "scripts/aggregate_checkpoint_interpolation_pilot.py", "--config", str(config_path)],
        check=True,
    )
    print("CHECKPOINT INTERPOLATION PILOT COMPLETED", flush=True)


if __name__ == "__main__":
    main()

