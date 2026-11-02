"""Complete the ten-seed support-8 confirmation for 100 segments per subject."""

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


def main() -> None:
    """Run only the seven seeds not already completed in the pilot."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/target_segments100_support8_confirm.yaml")
    parser.add_argument("--python-exe", default=sys.executable)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    data = config["data"]
    training = config["training"]
    evaluation = config["evaluation"]
    checkpoint = Path(config["initialization"]["ssl_checkpoint"])
    require_file(checkpoint)
    root = Path(experiment["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    support = int(experiment["support_per_class"])
    segment_budget = int(experiment["train_segments_per_subject"])

    for seed in [int(value) for value in experiment["confirmation_seeds"]]:
        split = Path(str(data["split_pattern"]).format(seed=seed))
        require_file(split)
        split_payload = json.loads(split.read_text(encoding="utf-8"))
        if len(split_payload.get("train_subjects", [])) != 2 * support:
            raise ValueError(f"Unexpected support split: {split}")
        output = root / f"seed_{seed}"
        summary = output / "summary.json"
        if summary.is_file():
            print(f"SKIP support={support} segments={segment_budget} seed={seed}", flush=True)
            continue
        command = [
            args.python_exe,
            "scripts/run_dual_stream_subject_split.py",
            "--data-dir", str(data["target"]),
            "--split-file", str(split),
            "--checkpoint", str(checkpoint),
            "--checkpoint-scope", "encoders",
            "--output-dir", str(output),
            "--seed", str(seed),
            "--model-type", "dual_stream",
            "--epochs", str(training["epochs"]),
            "--batch-size", str(training["batch_size"]),
            "--lr", str(training["learning_rate"]),
            "--weight-decay", str(training["weight_decay"]),
            "--patience", str(training["patience"]),
            "--max-train-segments-per-subject", str(segment_budget),
            "--max-val-segments-per-subject", str(training["max_val_segments_per_subject"]),
            "--max-test-segments-per-subject", str(training["max_test_segments_per_subject"]),
            "--alignment", str(evaluation["alignment"]),
            "--max-ea-samples", str(evaluation["max_alignment_samples"]),
            "--selection-metric", str(evaluation["selection_metric"]),
        ]
        if bool(evaluation["tune_threshold"]):
            command.extend(
                ["--tune-threshold", "--threshold-objective", str(evaluation["threshold_objective"])]
            )
        print(f"START support={support} segments={segment_budget} seed={seed}", flush=True)
        subprocess.run(command, check=True)
        require_file(summary)
        print(f"DONE support={support} segments={segment_budget} seed={seed}", flush=True)

    subprocess.run(
        [
            args.python_exe,
            "scripts/aggregate_target_segments100_support8_confirm.py",
            "--config", str(config_path),
        ],
        check=True,
    )
    print("SUPPORT-8 SEGMENTS-100 TEN-SEED CONFIRMATION COMPLETED", flush=True)


if __name__ == "__main__":
    main()

