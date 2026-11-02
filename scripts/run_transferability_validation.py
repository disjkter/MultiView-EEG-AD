"""Validate transferability rankings with controlled few-shot fine-tuning."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


METRICS = ("accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity")


def load_json(path: Path) -> dict[str, object]:
    """Load a JSON object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def run_with_retries(command: list[str], retries: int) -> None:
    """Run one training process, retrying transient native-process failures."""

    if retries < 1:
        raise ValueError("retries must be at least 1")
    for attempt in range(1, retries + 1):
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            return
        if attempt < retries:
            print(
                f"Training process failed with exit code {result.returncode}; "
                f"retrying ({attempt + 1}/{retries})..."
            )
            time.sleep(2)
    raise subprocess.CalledProcessError(result.returncode, command)


def create_fewshot_split(
    original_split: Path,
    support_subjects: list[str],
    output_path: Path,
    target_name: str,
    seed: int,
) -> None:
    """Create a split with audit support subjects as train and untouched val/test."""

    payload = load_json(original_split)
    original_train = {str(subject) for subject in payload.get("train_subjects", [])}
    unknown = set(support_subjects).difference(original_train)
    if unknown:
        raise ValueError(f"Support subjects are outside original train split: {sorted(unknown)}")
    excluded_subjects = sorted(original_train.difference(support_subjects))
    output = {
        "metadata": {
            "target": target_name,
            "seed": seed,
            "purpose": "transferability_fewshot_validation",
            "original_split": str(original_split),
        },
        "train_subjects": support_subjects,
        "val_subjects": payload["val_subjects"],
        "test_subjects": payload["test_subjects"],
        "excluded_subjects": excluded_subjects,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")


def build_finetune_command(
    python_exe: str,
    data_dir: str,
    split_file: Path,
    output_dir: Path,
    checkpoint: str | None,
    training: dict[str, object],
    seed: int,
) -> list[str]:
    """Build one controlled GraphTemporalNet fine-tuning command."""

    command = [
        python_exe,
        "scripts/run_dual_stream_subject_split.py",
        "--data-dir",
        data_dir,
        "--split-file",
        str(split_file),
        "--output-dir",
        str(output_dir),
        "--model-type",
        "graph_temporal",
        "--epochs",
        str(int(training.get("epochs", 30))),
        "--batch-size",
        str(int(training.get("batch_size", 32))),
        "--lr",
        str(float(training.get("learning_rate", 1e-3))),
        "--weight-decay",
        str(float(training.get("weight_decay", 1e-4))),
        "--patience",
        str(int(training.get("patience", 8))),
        "--selection-metric",
        str(training.get("selection_metric", "auc")),
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
        "--max-train-segments-per-subject",
        str(int(training.get("train_segments_per_subject", 20))),
        "--max-val-segments-per-subject",
        str(int(training.get("eval_segments_per_subject", 100))),
        "--max-test-segments-per-subject",
        str(int(training.get("eval_segments_per_subject", 100))),
    ]
    if checkpoint is None:
        command.append("--no-checkpoint")
    else:
        command.extend(
            [
                "--checkpoint",
                checkpoint,
                "--checkpoint-scope",
                str(training.get("checkpoint_scope", "all")),
            ]
        )
    if bool(training.get("tune_threshold", False)):
        command.extend(
            [
                "--tune-threshold",
                "--threshold-objective",
                str(training.get("threshold_objective", "balanced_acc")),
            ]
        )
    return command


def read_metrics(summary_path: Path) -> dict[str, float]:
    """Read scalar test metrics from a subject-split summary."""

    metrics = load_json(summary_path)["metrics"]
    return {name: float(metrics[name]) for name in METRICS}


def write_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Write validation summary in JSON, CSV, and Markdown formats."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Few-shot Transferability Validation",
        "",
        "| Target | Init | Rank | BA | AUROC | Delta BA | Delta AUROC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rank = "" if row["rank_score"] is None else f"{row['rank_score']:.4f}"
        lines.append(
            f"| {row['target']} | {row['initialization']} | {rank} | "
            f"{row['balanced_accuracy']:.4f} | {row['auc']:.4f} | "
            f"{row['delta_balanced_accuracy']:.4f} | {row['delta_auc']:.4f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run target-only and source-initialized few-shot comparisons."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transferability_audit_local.yaml")
    parser.add_argument("--audit-json", default=None)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--target", action="append", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Maximum attempts for each training subprocess.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    audit_path = Path(
        args.audit_json
        or str(config.get("output_prefix", "outputs/transferability_audit_local/seed2026"))
        + ".json"
    )
    audit = load_json(audit_path)
    audit_rows = audit.get("rows")
    metadata = audit.get("metadata")
    if not isinstance(audit_rows, list) or not isinstance(metadata, dict):
        raise ValueError(f"Invalid audit payload: {audit_path}")
    target_meta = {item["name"]: item for item in metadata["targets"]}
    target_specs = {item["name"]: item for item in config["targets"]}
    training = config.get("validation_training", {})
    seed = int(config.get("seed", 2026))
    output_dir = Path(
        str(config.get("validation_output_dir", "outputs/transferability_validation_local"))
    )
    selected_targets = set(args.target or [])
    grouped_rows: dict[str, list[dict[str, object]]] = {}
    for row in audit_rows:
        grouped_rows.setdefault(str(row["target"]), []).append(row)

    completed: list[dict[str, object]] = []
    for target_name, source_rows in grouped_rows.items():
        if selected_targets and target_name not in selected_targets:
            continue
        target_spec = target_specs[target_name]
        support_subjects = [str(subject) for subject in target_meta[target_name]["support_subjects"]]
        fewshot_split = output_dir / "splits" / f"{target_name}_seed{seed}.json"
        create_fewshot_split(
            Path(str(target_spec["split_file"])),
            support_subjects,
            fewshot_split,
            target_name,
            seed,
        )
        variants: list[tuple[str, str | None, float | None]] = [("target_only", None, None)]
        variants.extend(
            (
                str(row["source"]),
                str(row["checkpoint"]),
                float(row["rank_score"]),
            )
            for row in source_rows
        )
        variant_metrics: dict[str, dict[str, float]] = {}
        for initialization, checkpoint, rank_score in variants:
            run_dir = output_dir / target_name / initialization
            summary_path = run_dir / "summary.json"
            command = build_finetune_command(
                args.python_exe,
                str(target_spec["data_dir"]),
                fewshot_split,
                run_dir,
                checkpoint,
                training,
                seed,
            )
            print("\n" + " ".join(command))
            if not args.dry_run and (args.force or not summary_path.exists()):
                run_with_retries(command, args.retries)
            if not args.dry_run:
                variant_metrics[initialization] = read_metrics(summary_path)

        if args.dry_run:
            continue
        baseline = variant_metrics["target_only"]
        rank_by_source = {str(row["source"]): float(row["rank_score"]) for row in source_rows}
        for initialization, metrics in variant_metrics.items():
            completed.append(
                {
                    "target": target_name,
                    "initialization": initialization,
                    "rank_score": rank_by_source.get(initialization),
                    **metrics,
                    "delta_balanced_accuracy": metrics["balanced_accuracy"]
                    - baseline["balanced_accuracy"],
                    "delta_auc": metrics["auc"] - baseline["auc"],
                }
            )

    if not args.dry_run:
        if not completed:
            raise ValueError("No validation runs were completed")
        existing_path = output_dir / "summary.json"
        existing_rows: list[dict[str, object]] = []
        if existing_path.exists():
            existing_payload = json.loads(existing_path.read_text(encoding="utf-8"))
            if isinstance(existing_payload, list):
                existing_rows = existing_payload
        merged = {
            (str(row["target"]), str(row["initialization"])): row
            for row in existing_rows
        }
        for row in completed:
            merged[(str(row["target"]), str(row["initialization"]))] = row
        all_rows = sorted(
            merged.values(),
            key=lambda row: (str(row["target"]), str(row["initialization"])),
        )
        write_summary(all_rows, output_dir)
        print(f"Saved validation summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
