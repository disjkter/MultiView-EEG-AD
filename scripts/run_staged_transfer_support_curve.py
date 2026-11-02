"""Run matched dual-stream ds004504-to-CAUEEG staged transfer experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def require_file(path: Path) -> None:
    """Fail early when an experiment input is unavailable."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing required input: {path}")


def run_checked(command: list[str], done_file: Path, label: str) -> None:
    """Run one resumable training job and verify its declared output."""

    if done_file.is_file():
        print(f"SKIP {label}", flush=True)
        return
    print(f"START {label}", flush=True)
    subprocess.run(command, check=True)
    require_file(done_file)
    print(f"DONE {label}", flush=True)


def training_arguments(settings: dict[str, object], evaluation: dict[str, object]) -> list[str]:
    """Translate YAML training settings into the shared runner arguments."""

    args = [
        "--model-type", "dual_stream",
        "--epochs", str(settings["epochs"]),
        "--batch-size", str(settings["batch_size"]),
        "--lr", str(settings["learning_rate"]),
        "--weight-decay", str(settings["weight_decay"]),
        "--patience", str(settings["patience"]),
        "--max-train-segments-per-subject", str(settings["max_train_segments_per_subject"]),
        "--max-val-segments-per-subject", str(settings["max_val_segments_per_subject"]),
        "--max-test-segments-per-subject", str(settings["max_test_segments_per_subject"]),
        "--alignment", str(evaluation["alignment"]),
        "--max-ea-samples", str(evaluation["max_alignment_samples"]),
        "--selection-metric", str(evaluation["selection_metric"]),
    ]
    if bool(evaluation["tune_threshold"]):
        args.extend(
            ["--tune-threshold", "--threshold-objective", str(evaluation["threshold_objective"])]
        )
    return args


def validate_target_split(path: Path, support: int) -> None:
    """Verify that a target split is a balanced few-shot participant split."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    train_subjects = payload.get("train_subjects", [])
    excluded_subjects = payload.get("excluded_subjects", [])
    if len(train_subjects) != 2 * support:
        raise ValueError(
            f"Invalid support-{support} split {path}: expected {2 * support} train subjects, "
            f"found {len(train_subjects)}"
        )
    if not excluded_subjects:
        raise ValueError(f"Invalid few-shot split {path}: excluded_subjects is empty")


def main() -> None:
    """Train the source stage and compare three matched target initializations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/staged_ds004504_to_caueeg.yaml")
    parser.add_argument("--python-exe", default=sys.executable)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    data = config["data"]
    initialization = config["initialization"]
    evaluation = config["evaluation"]
    seeds = [int(seed) for seed in experiment["seeds"]]
    supports = [int(support) for support in experiment["supports_per_class"]]
    output_root = Path(experiment["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    ssl_checkpoint = Path(initialization["ssl_checkpoint"])
    require_file(ssl_checkpoint)
    source_common = training_arguments(config["source_training"], evaluation)
    target_common = training_arguments(config["target_training"], evaluation)
    runner = "scripts/run_dual_stream_subject_split.py"

    for seed in seeds:
        source_split = Path(str(data["source_split_pattern"]).format(seed=seed))
        require_file(source_split)
        source_dir = output_root / "source_ds004504" / f"seed_{seed}"
        source_checkpoint = source_dir / "best_model.pth"
        source_command = [
            args.python_exe, runner,
            "--data-dir", str(data["source"]),
            "--split-file", str(source_split),
            "--checkpoint", str(ssl_checkpoint),
            "--checkpoint-scope", "encoders",
            "--output-dir", str(source_dir),
            "--seed", str(seed),
            *source_common,
        ]
        run_checked(source_command, source_checkpoint, f"source ds004504 seed={seed}")

        for support in supports:
            pattern = data["target_split_patterns"][str(support)]
            target_split = Path(str(pattern).format(seed=seed))
            require_file(target_split)
            validate_target_split(target_split, support)
            for condition in initialization["conditions"]:
                target_dir = output_root / f"support_{support}" / str(condition) / f"seed_{seed}"
                command = [
                    args.python_exe, runner,
                    "--data-dir", str(data["target"]),
                    "--split-file", str(target_split),
                    "--output-dir", str(target_dir),
                    "--seed", str(seed),
                    *target_common,
                ]
                if condition == "random_init":
                    command.append("--no-checkpoint")
                elif condition == "ssl_only":
                    command.extend(["--checkpoint", str(ssl_checkpoint), "--checkpoint-scope", "encoders"])
                elif condition == "staged_transfer":
                    command.extend(["--checkpoint", str(source_checkpoint), "--checkpoint-scope", "encoders"])
                else:
                    raise ValueError(f"Unknown initialization condition: {condition}")
                run_checked(
                    command,
                    target_dir / "summary.json",
                    f"target CAUEEG support={support} condition={condition} seed={seed}",
                )

    aggregate_command = [
        args.python_exe,
        "scripts/aggregate_staged_transfer_support_curve.py",
        "--config", str(config_path),
    ]
    subprocess.run(aggregate_command, check=True)
    print("STAGED TRANSFER SUPPORT CURVE COMPLETED", flush=True)


if __name__ == "__main__":
    main()

