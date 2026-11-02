"""Run a small stage-2 regularization pilot before a full multi-seed study."""

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
    """Run one resumable command and verify its output."""

    if output.is_file():
        print(f"SKIP {label}", flush=True)
        return
    print(f"START {label}", flush=True)
    subprocess.run(command, check=True)
    require_file(output)
    print(f"DONE {label}", flush=True)


def common_arguments(settings: dict[str, object], evaluation: dict[str, object]) -> list[str]:
    """Build shared arguments for one subject-level training job."""

    learning_rate = settings.get("learning_rate", settings.get("classifier_learning_rate"))
    result = [
        "--model-type", "dual_stream",
        "--epochs", str(settings["epochs"]),
        "--batch-size", str(settings["batch_size"]),
        "--lr", str(learning_rate),
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
        result.extend(
            ["--tune-threshold", "--threshold-objective", str(evaluation["threshold_objective"])]
        )
    return result


def validate_support_split(path: Path, support: int) -> None:
    """Check the number of target training participants."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload.get("train_subjects", [])) != 2 * support:
        raise ValueError(f"Unexpected training-subject count in {path}")


def main() -> None:
    """Train regularized source encoders and evaluate them on matched target splits."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/l2sp_stage2_pilot.yaml")
    parser.add_argument("--python-exe", default=sys.executable)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    data = config["data"]
    initialization = config["initialization"]
    evaluation = config["evaluation"]
    output_root = Path(experiment["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    ssl_checkpoint = Path(initialization["ssl_checkpoint"])
    require_file(ssl_checkpoint)
    source_common = common_arguments(config["source_training"], evaluation)
    target_common = common_arguments(config["target_training"], evaluation)
    runner = "scripts/run_dual_stream_subject_split.py"

    for variant in initialization["stage2_variants"]:
        variant_name = str(variant["name"])
        for seed in [int(value) for value in experiment["seeds"]]:
            source_split = Path(str(data["source_split_pattern"]).format(seed=seed))
            require_file(source_split)
            source_dir = output_root / "source" / variant_name / f"seed_{seed}"
            source_checkpoint = source_dir / "best_model.pth"
            source_command = [
                args.python_exe, runner,
                "--data-dir", str(data["source"]),
                "--split-file", str(source_split),
                "--checkpoint", str(ssl_checkpoint),
                "--checkpoint-scope", "encoders",
                "--encoder-lr", str(variant["encoder_learning_rate"]),
                "--l2sp-weight", str(variant["l2sp_weight"]),
                "--l2sp-scope", "encoders",
                "--output-dir", str(source_dir),
                "--seed", str(seed),
                *source_common,
            ]
            run_checked(source_command, source_checkpoint, f"source {variant_name} seed={seed}")

            for support in [int(value) for value in experiment["supports_per_class"]]:
                split_pattern = data["target_split_patterns"][str(support)]
                target_split = Path(str(split_pattern).format(seed=seed))
                require_file(target_split)
                validate_support_split(target_split, support)
                target_dir = output_root / "target" / f"support_{support}" / variant_name / f"seed_{seed}"
                target_command = [
                    args.python_exe, runner,
                    "--data-dir", str(data["target"]),
                    "--split-file", str(target_split),
                    "--checkpoint", str(source_checkpoint),
                    "--checkpoint-scope", "encoders",
                    "--output-dir", str(target_dir),
                    "--seed", str(seed),
                    *target_common,
                ]
                run_checked(
                    target_command,
                    target_dir / "summary.json",
                    f"target support={support} {variant_name} seed={seed}",
                )

    subprocess.run(
        [
            args.python_exe,
            "scripts/aggregate_l2sp_stage2_pilot.py",
            "--config", str(config_path),
        ],
        check=True,
    )
    print("L2-SP STAGE-2 PILOT COMPLETED", flush=True)


if __name__ == "__main__":
    main()

