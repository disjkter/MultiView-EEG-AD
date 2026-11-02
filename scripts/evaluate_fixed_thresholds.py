"""Re-evaluate saved subject-split models with shared decision thresholds."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_dual_stream_loso_subject import build_model  # noqa: E402
from scripts.run_dual_stream_subject_split import (  # noqa: E402
    IndexedEEGDataset,
    collect_subject_probabilities,
    compute_metrics,
    fit_euclidean_alignment,
    load_domain_mmap,
    sample_indices_per_subject,
)
from utils.subject_split import load_subject_split_masks  # noqa: E402


METRICS = ["accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity"]


def summarize(values: list[float]) -> dict[str, float]:
    """Compute population mean/std over finite values."""

    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(statistics.fmean(finite)),
        "std": float(statistics.pstdev(finite)) if len(finite) > 1 else 0.0,
    }


def load_summary(run_dir: Path) -> dict[str, object]:
    """Load a run summary."""

    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def evaluate_run(
    run_dir: Path,
    threshold: float,
    split_file: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate one saved run with a fixed threshold."""

    summary = load_summary(run_dir)
    x_all, y_all, sub_ids_all = load_domain_mmap(Path(str(summary["data_dir"])))
    train_mask, _, test_mask = load_subject_split_masks(split_file, sub_ids_all)
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    seed = int(summary["seed"])
    train_indices = sample_indices_per_subject(
        train_indices,
        sub_ids_all,
        summary.get("max_train_segments_per_subject"),
        seed + 11,
    )
    test_indices = sample_indices_per_subject(
        test_indices,
        sub_ids_all,
        summary.get("max_test_segments_per_subject"),
        seed + 23,
    )

    ea_matrix = None
    if bool(summary.get("use_ea", False)):
        ea_indices = train_indices
        max_ea_samples = summary.get("max_ea_samples")
        if max_ea_samples is not None and len(ea_indices) > int(max_ea_samples):
            rng = np.random.default_rng(seed + 31)
            ea_indices = rng.choice(ea_indices, size=int(max_ea_samples), replace=False)
        ea_matrix = fit_euclidean_alignment(np.asarray(x_all[ea_indices], dtype=np.float32))

    test_dataset = IndexedEEGDataset(x_all, y_all, sub_ids_all, test_indices, ea_matrix=ea_matrix)
    model = build_model(
        num_classes=int(y_all.max() + 1),
        num_channels=int(x_all.shape[1]),
        checkpoint=None,
        linear_probe=bool(summary.get("linear_probe", False)),
        device=device,
        spatial_out_dim=int(summary.get("spatial_out_dim", 256)),
        temporal_out_dim=int(summary.get("temporal_out_dim", 256)),
        spatial_hidden_channels=int(summary.get("spatial_hidden_channels", 64)),
        spatial_temporal_channels=int(summary.get("spatial_temporal_channels", 128)),
        classifier_hidden_dim=summary.get("classifier_hidden_dim"),
        classifier_num_layers=int(summary.get("classifier_num_layers", 2)),
        classifier_dropout=float(summary.get("classifier_dropout", 0.6)),
        use_neuro_prior=bool(summary.get("use_neuro_prior", False)),
        sampling_rate=float(summary.get("sampling_rate", 128.0)),
        prior_out_dim=int(summary.get("prior_out_dim", 64)),
    )
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device))
    subjects, y_true, y_prob = collect_subject_probabilities(model, test_dataset, batch_size, device)
    metrics = compute_metrics(subjects, y_true, y_prob, threshold)
    return {
        "run_dir": str(run_dir),
        "seed": seed,
        "threshold": threshold,
        "metrics": {metric: metrics[metric] for metric in METRICS},
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description="Re-evaluate saved models with fixed thresholds.")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    """Evaluate all thresholds and print aggregate results."""

    args = parse_args()
    device = torch.device(args.device)
    run_dirs = [Path(path) for path in args.run_dirs]

    threshold_results = []
    for threshold in args.thresholds:
        seed_runs = [
            evaluate_run(run_dir, threshold, args.split_file, args.batch_size, device)
            for run_dir in run_dirs
        ]
        aggregate = {
            metric: summarize([float(run["metrics"][metric]) for run in seed_runs])
            for metric in METRICS
        }
        threshold_results.append(
            {
                "threshold": threshold,
                "aggregate": aggregate,
                "seed_runs": seed_runs,
            }
        )

    print("\nFixed-threshold aggregate results")
    print("threshold  acc_mean  bal_mean  f1_mean  auc_mean  sen_mean  spec_mean")
    for result in threshold_results:
        agg = result["aggregate"]
        print(
            f"{result['threshold']:<9.3f} "
            f"{agg['accuracy']['mean']:.4f}   "
            f"{agg['balanced_accuracy']['mean']:.4f}    "
            f"{agg['f1']['mean']:.4f}   "
            f"{agg['auc']['mean']:.4f}   "
            f"{agg['sensitivity']['mean']:.4f}   "
            f"{agg['specificity']['mean']:.4f}"
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(threshold_results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
