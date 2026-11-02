"""Tune the target-stage update strength for few-shot SSL transfer."""

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
    """Run one resumable job and verify its output."""

    if output.is_file():
        print(f"SKIP {label}", flush=True)
        return
    print(f"START {label}", flush=True)
    subprocess.run(command, check=True)
    require_file(output)
    print(f"DONE {label}", flush=True)


def main() -> None:
    """Compare discriminative fine-tuning and linear probing."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/target_finetune_pilot.yaml")
    parser.add_argument("--python-exe", default=sys.executable)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    data = config["data"]
    initialization = config["initialization"]
    training = config["training"]
    evaluation = config["evaluation"]
    root = Path(experiment["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(initialization["ssl_checkpoint"])
    require_file(checkpoint)
    common = [
        "--model-type", "dual_stream",
        "--epochs", str(training["epochs"]),
        "--batch-size", str(training["batch_size"]),
        "--lr", str(training["classifier_learning_rate"]),
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

    for support in [int(value) for value in experiment["supports_per_class"]]:
        for seed in [int(value) for value in experiment["seeds"]]:
            split = Path(str(data["target_split_patterns"][str(support)]).format(seed=seed))
            require_file(split)
            payload = json.loads(split.read_text(encoding="utf-8"))
            if len(payload.get("train_subjects", [])) != 2 * support:
                raise ValueError(f"Unexpected support split: {split}")
            for condition in initialization["conditions"]:
                name = str(condition["name"])
                output = root / f"support_{support}" / name / f"seed_{seed}"
                command = [
                    args.python_exe,
                    "scripts/run_dual_stream_subject_split.py",
                    "--data-dir", str(data["target"]),
                    "--split-file", str(split),
                    "--checkpoint", str(checkpoint),
                    "--checkpoint-scope", "encoders",
                    "--output-dir", str(output),
                    "--seed", str(seed),
                    *common,
                ]
                if bool(condition["linear_probe"]):
                    command.append("--linear-probe")
                else:
                    command.extend(["--encoder-lr", str(condition["encoder_learning_rate"])])
                run_checked(
                    command,
                    output / "summary.json",
                    f"support={support} {name} seed={seed}",
                )

    subprocess.run(
        [args.python_exe, "scripts/aggregate_target_finetune_pilot.py", "--config", str(config_path)],
        check=True,
    )
    print("TARGET FINE-TUNING PILOT COMPLETED", flush=True)


if __name__ == "__main__":
    main()

