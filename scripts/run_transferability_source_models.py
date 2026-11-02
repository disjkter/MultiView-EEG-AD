"""Train source GraphTemporalNet checkpoints required by transferability audit."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def build_command(
    python_exe: str,
    source: dict[str, object],
    training: dict[str, object],
    seed: int,
) -> list[str]:
    """Build one source-only GraphTemporalNet training command."""

    checkpoint = Path(str(source["checkpoint"]))
    output_dir = checkpoint.parent
    command = [
        python_exe,
        "scripts/run_dual_stream_subject_split.py",
        "--data-dir",
        str(source["data_dir"]),
        "--split-file",
        str(source["split_file"]),
        "--output-dir",
        str(output_dir),
        "--no-checkpoint",
        "--model-type",
        "graph_temporal",
        "--epochs",
        str(int(training.get("epochs", 50))),
        "--batch-size",
        str(int(training.get("batch_size", 64))),
        "--lr",
        str(float(training.get("learning_rate", 1e-3))),
        "--weight-decay",
        str(float(training.get("weight_decay", 1e-4))),
        "--patience",
        str(int(training.get("patience", 8))),
        "--selection-metric",
        str(source.get("selection_metric", training.get("selection_metric", "auc"))),
        "--seed",
        str(seed),
        "--graph-num-neighbors",
        str(int(training.get("graph_num_neighbors", 4))),
        "--spatial-out-dim",
        str(int(training.get("graph_out_dim", 256))),
        "--temporal-out-dim",
        str(int(training.get("temporal_out_dim", 256))),
        "--classifier-dropout",
        str(float(training.get("classifier_dropout", 0.6))),
    ]
    max_segments = int(training.get("max_segments_per_subject", 100))
    if max_segments > 0:
        command.extend(
            [
                "--max-train-segments-per-subject",
                str(max_segments),
                "--max-val-segments-per-subject",
                str(max_segments),
                "--max-test-segments-per-subject",
                str(max_segments),
            ]
        )
    return command


def quote_command(command: list[str]) -> str:
    """Render a readable PowerShell-safe command preview."""

    return " ".join(f'"{part}"' if " " in part else part for part in command)


def main() -> None:
    """Train all source models listed in the transferability audit config."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transferability_audit.yaml")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--source", action="append", default=None, help="Train only this source")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Retrain completed checkpoints")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("config must define a non-empty sources list")
    training = config.get("source_training", {})
    if not isinstance(training, dict):
        raise ValueError("source_training must be a mapping")
    seed = int(args.seed if args.seed is not None else config.get("seed", 2026))
    selected_names = set(args.source or [])
    matched = 0

    for source in sources:
        name = str(source["name"])
        if selected_names and name not in selected_names:
            continue
        matched += 1
        split_file = Path(str(source["split_file"]))
        data_dir = Path(str(source["data_dir"]))
        checkpoint = Path(str(source["checkpoint"]))
        if not data_dir.exists():
            raise FileNotFoundError(f"Missing source data directory: {data_dir}")
        if not split_file.exists():
            raise FileNotFoundError(f"Missing source split file: {split_file}")
        if checkpoint.exists() and not args.force:
            print(f"SKIP {name}: checkpoint exists at {checkpoint}")
            continue

        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        command = build_command(args.python_exe, source, training, seed)
        print(f"\nSOURCE {name}")
        print(quote_command(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Training finished but expected checkpoint was not created: {checkpoint}"
                )

    if matched == 0:
        raise ValueError(f"No configured source matched: {sorted(selected_names)}")


if __name__ == "__main__":
    main()
