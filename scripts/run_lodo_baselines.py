"""Run leave-one-domain-out baseline experiments across processed EEG datasets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def resolve_domain_files(domain_dir: Path) -> tuple[Path, Path, Path]:
    """Resolve x/y/sub_id files from a processed domain directory."""

    candidates = [
        ("x_19ch.npy", "y_19ch.npy", "sub_ids_19ch.npy"),
        ("x.npy", "y.npy", "sub_ids.npy"),
    ]
    for x_name, y_name, id_name in candidates:
        x_path = domain_dir / x_name
        y_path = domain_dir / y_name
        id_path = domain_dir / id_name
        if x_path.exists() and y_path.exists() and id_path.exists():
            return x_path, y_path, id_path
    raise FileNotFoundError(f"Could not find x/y/sub_id npy files under {domain_dir}")


def load_domain(domain_name: str, domain_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one domain and prefix subject ids with domain name."""

    x_path, y_path, id_path = resolve_domain_files(domain_dir)
    x = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    sub_ids = np.load(id_path, allow_pickle=True)
    prefixed_ids = np.asarray([f"{domain_name}::{sid}" for sid in sub_ids], dtype=object)
    return x, y, prefixed_ids


def match_time_length(x: np.ndarray, target_length: int) -> np.ndarray:
    """Crop or zero-pad EEG samples to a shared temporal length."""

    current_length = x.shape[-1]
    if current_length == target_length:
        return x
    if current_length > target_length:
        return x[..., :target_length]

    padded = np.zeros((x.shape[0], x.shape[1], target_length), dtype=x.dtype)
    padded[..., :current_length] = x
    return padded


def save_merged_domain(domain_specs: list[tuple[str, Path]], output_dir: Path) -> int:
    """Merge multiple source domains into one temporary training domain."""

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    id_parts: list[np.ndarray] = []
    loaded_domains: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    source_lengths: list[int] = []

    for domain_name, domain_dir in domain_specs:
        x, y, sub_ids = load_domain(domain_name, domain_dir)
        loaded_domains.append((x, y, sub_ids))
        source_lengths.append(int(x.shape[-1]))

    common_length = min(source_lengths)
    for x, y, sub_ids in loaded_domains:
        x_parts.append(match_time_length(x, common_length))
        y_parts.append(y)
        id_parts.append(sub_ids)

    x_all = np.concatenate(x_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    sub_ids_all = np.concatenate(id_parts, axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "x_19ch.npy", x_all)
    np.save(output_dir / "y_19ch.npy", y_all)
    np.save(output_dir / "sub_ids_19ch.npy", sub_ids_all)
    with (output_dir / "merge_info.json").open("w", encoding="utf-8") as fp:
        json.dump({"common_time_length": common_length}, fp, indent=2, ensure_ascii=False)
    return common_length


def read_target_metrics(summary_path: Path) -> dict[str, float]:
    """Read target metrics from a completed baseline run."""

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = data["target_metrics"]
    return {
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "f1": float(metrics["f1"]),
        "auc": float(metrics["auc"]),
        "sensitivity": float(metrics["sensitivity"]),
        "specificity": float(metrics["specificity"]),
    }


def main() -> None:
    """Run LODO baselines across all provided domains."""

    parser = argparse.ArgumentParser(description="Leave-one-domain-out wrapper for baseline models")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--domains", nargs="+", required=True, help="Format: name=path")
    parser.add_argument("--model", choices=["svm", "eegnet"], required=True)
    parser.add_argument("--output-dir", default="outputs/lodo_baselines")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--use-ea", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--threshold-objective", choices=["balanced_acc", "f1"], default="balanced_acc")
    args = parser.parse_args()

    domain_specs: list[tuple[str, Path]] = []
    for item in args.domains:
        if "=" not in item:
            raise ValueError(f"Invalid domain spec: {item}. Use name=path.")
        name, path_str = item.split("=", 1)
        domain_specs.append((name, Path(path_str)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = output_dir / "_tmp_sources"
    tmp_root.mkdir(parents=True, exist_ok=True)

    fold_summaries: list[dict[str, object]] = []
    for target_name, target_dir in domain_specs:
        source_specs = [(name, path) for name, path in domain_specs if name != target_name]
        merged_source_dir = tmp_root / f"source_for_{target_name}"
        common_length = save_merged_domain(source_specs, merged_source_dir)

        fold_output_dir = output_dir / f"fold_{target_name}"
        command = [
            args.python_exe,
            "scripts/run_cross_domain_baselines.py",
            "--source-dir",
            str(merged_source_dir),
            "--target-dir",
            str(target_dir),
            "--model",
            args.model,
            "--output-dir",
            str(fold_output_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--dropout",
            str(args.dropout),
            "--val-ratio",
            str(args.val_ratio),
            "--patience",
            str(args.patience),
            "--seed",
            str(args.seed),
        ]
        if args.use_ea:
            command.append("--use-ea")
        if args.tune_threshold:
            command.append("--tune-threshold")
            command.extend(["--threshold-objective", args.threshold_objective])

        print(f"\nRunning LODO baseline fold with target domain: {target_name}")
        subprocess.run(command, check=True)

        metrics = read_target_metrics(fold_output_dir / "summary.json")
        fold_summaries.append(
            {
                "target_domain": target_name,
                "source_domains": [name for name, _ in source_specs],
                "common_time_length": common_length,
                "metrics": metrics,
            }
        )

    metric_names = ["accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity"]
    aggregate_mean: dict[str, float] = {}
    for metric_name in metric_names:
        values = [float(fold["metrics"][metric_name]) for fold in fold_summaries]
        aggregate_mean[metric_name] = float(sum(values) / len(values))

    result = {
        "model": args.model,
        "domains": [name for name, _ in domain_specs],
        "use_ea": args.use_ea,
        "tune_threshold": args.tune_threshold,
        "threshold_objective": args.threshold_objective if args.tune_threshold else None,
        "folds": fold_summaries,
        "aggregate_mean": aggregate_mean,
    }
    with (output_dir / "lodo_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)

    print("\nLODO baseline aggregate mean results")
    for metric_name, value in aggregate_mean.items():
        print(f"{metric_name}: {value:.4f}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
