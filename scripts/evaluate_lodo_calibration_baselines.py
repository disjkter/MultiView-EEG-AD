"""Evaluate source-only calibration baselines from completed LODO checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import brier_score_loss, log_loss
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_dual_stream_subject_split import choose_threshold  # noqa: E402
from scripts.run_graph_temporal_lodo_tacd import (  # noqa: E402
    MultiDomainDataset,
    binary_metrics,
    build_model,
    collect_subject_probabilities,
    load_store,
    make_refs,
    parse_domains,
    set_seed,
    split_source_indices,
)


def positive_logit(probability: np.ndarray) -> np.ndarray:
    """Convert positive-class probabilities to finite logits."""

    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(clipped) - np.log1p(-clipped)


def sigmoid(logit: np.ndarray) -> np.ndarray:
    """Evaluate a numerically stable sigmoid."""

    logit = np.asarray(logit, dtype=np.float64)
    output = np.empty_like(logit)
    positive = logit >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exp_values = np.exp(logit[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def fit_temperature(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Fit one positive temperature on source-validation subjects."""

    logits = positive_logit(probability)

    def objective(log_temperature: float) -> float:
        calibrated = sigmoid(logits / math.exp(log_temperature))
        return float(log_loss(y_true, calibrated, labels=[0, 1]))

    result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    return float(math.exp(result.x))


def fit_platt(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """Fit a positive Platt slope and intercept on source validation."""

    logits = positive_logit(probability)

    def objective(parameters: np.ndarray) -> float:
        slope = math.exp(float(parameters[0]))
        calibrated = sigmoid(slope * logits + float(parameters[1]))
        return float(log_loss(y_true, calibrated, labels=[0, 1]))

    result = minimize(
        objective,
        x0=np.asarray([0.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(-5.0, 5.0), (-20.0, 20.0)],
    )
    if not result.success:
        raise RuntimeError(f"Platt optimization failed: {result.message}")
    return float(math.exp(result.x[0])), float(result.x[1])


def apply_calibrator(
    probability: np.ndarray,
    calibrator: str,
    parameters: tuple[float, ...],
) -> np.ndarray:
    """Apply one fitted monotonic probability calibrator."""

    logits = positive_logit(probability)
    if calibrator == "identity":
        return np.asarray(probability, dtype=np.float64)
    if calibrator == "temperature":
        return sigmoid(logits / parameters[0])
    if calibrator == "platt":
        return sigmoid(parameters[0] * logits + parameters[1])
    raise ValueError(f"Unknown calibrator: {calibrator}")


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    num_bins: int = 10,
) -> float:
    """Compute equal-width binary expected calibration error."""

    edges = np.linspace(0.0, 1.0, num_bins + 1)
    assignments = np.minimum(np.digitize(probability, edges[1:-1]), num_bins - 1)
    total = len(y_true)
    error = 0.0
    for index in range(num_bins):
        mask = assignments == index
        if not np.any(mask):
            continue
        error += float(mask.sum() / total) * abs(
            float(probability[mask].mean()) - float(y_true[mask].mean())
        )
    return float(error)


def probability_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute discrimination, threshold, and calibration metrics."""

    two_class = np.column_stack([1.0 - probability, probability])
    metrics = binary_metrics(y_true, two_class, threshold)
    return {
        key: float(metrics[key])
        for key in (
            "accuracy",
            "balanced_accuracy",
            "f1",
            "auc",
            "sensitivity",
            "specificity",
        )
    } | {
        "ece": expected_calibration_error(y_true, probability),
        "brier": float(brier_score_loss(y_true, probability)),
        "nll": float(log_loss(y_true, probability, labels=[0, 1])),
    }


def calibration_intercept_slope(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, float]:
    """Estimate evaluation-only logistic calibration intercept and slope."""

    labels = np.asarray(y_true, dtype=np.float64)
    logits = positive_logit(probability)
    if len(np.unique(labels)) < 2 or float(np.std(logits)) < 1e-12:
        return float("nan"), float("nan")

    design = np.column_stack([np.ones_like(logits), logits])
    parameters = np.asarray([0.0, 1.0], dtype=np.float64)
    ridge = np.diag([1e-8, 1e-8])
    for _ in range(50):
        fitted = sigmoid(design @ parameters)
        weights = np.clip(fitted * (1.0 - fitted), 1e-8, None)
        gradient = design.T @ (labels - fitted) - ridge @ parameters
        information = (design.T * weights) @ design + ridge
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        parameters += step
        parameters = np.clip(parameters, -20.0, 20.0)
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return float(parameters[0]), float(parameters[1])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write homogeneous dictionaries to CSV."""

    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Macro-average targets within seed, then aggregate across seeds."""

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "auc",
        "sensitivity",
        "specificity",
        "ece",
        "brier",
        "nll",
    ]
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), str(row["calibrator"]), int(row["seed"]))].append(row)

    seed_rows: list[dict[str, object]] = []
    for (variant, calibrator, seed), group in sorted(grouped.items()):
        seed_row: dict[str, object] = {
            "variant": variant,
            "calibrator": calibrator,
            "seed": seed,
            "num_targets": len(group),
        }
        for metric in metric_names:
            seed_row[metric] = float(np.mean([float(row[metric]) for row in group]))
        seed_rows.append(seed_row)

    aggregate_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        aggregate_groups[(str(row["variant"]), str(row["calibrator"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (variant, calibrator), group in sorted(aggregate_groups.items()):
        summary_row: dict[str, object] = {
            "variant": variant,
            "calibrator": calibrator,
            "num_seeds": len(group),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in group])
            summary_row[f"{metric}_mean"] = float(values.mean())
            summary_row[f"{metric}_std"] = float(values.std(ddof=0))
        summary_rows.append(summary_row)
    return seed_rows, summary_rows


def main() -> None:
    """Reconstruct source validation predictions and evaluate calibration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--backbone", choices=["graph_temporal", "eegnet"], required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-source-subjects-per-class", type=int, default=50)
    parser.add_argument("--max-segments-per-subject", type=int, default=20)
    parser.add_argument("--graph-out-dim", type=int, default=256)
    parser.add_argument("--temporal-out-dim", type=int, default=256)
    parser.add_argument("--disease-dim", type=int, default=256)
    parser.add_argument("--domain-dim", type=int, default=128)
    parser.add_argument("--graph-num-neighbors", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--eegnet-temporal-filters", type=int, default=8)
    parser.add_argument("--eegnet-depth-multiplier", type=int, default=2)
    parser.add_argument("--eegnet-separable-filters", type=int, default=16)
    parser.add_argument("--eegnet-temporal-kernel", type=int, default=64)
    parser.add_argument("--eegnet-separable-kernel", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--limit-summaries",
        type=int,
        default=0,
        help="Process only the first N summaries for smoke testing; 0 processes all.",
    )
    args = parser.parse_args()

    device_name = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(device_name)
    domain_paths = parse_domains(args.domains)
    stores = {name: load_store(name, path) for name, path in domain_paths.items()}
    results_dir = Path(args.results_dir)
    summary_paths = sorted(results_dir.glob("seed*/**/lodo_summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No lodo_summary.json files under {results_dir}")
    if args.limit_summaries > 0:
        summary_paths = summary_paths[: args.limit_summaries]

    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("backbone", "graph_temporal") != args.backbone:
            raise ValueError(f"Backbone mismatch in {summary_path}")
        seed = int(payload["seed"])
        variant = str(payload.get("variant", summary_path.parent.name))
        set_seed(seed)
        for fold in payload["folds"]:
            target_name = str(fold["target"])
            source_stores = [store for name, store in stores.items() if name != target_name]
            val_sets: list[np.ndarray] = []
            for index, store in enumerate(source_stores):
                _, val_indices = split_source_indices(
                    store,
                    args.val_ratio,
                    args.max_source_subjects_per_class,
                    args.max_segments_per_subject,
                    seed + 100 * index,
                )
                val_sets.append(val_indices)
            val_refs = make_refs(source_stores, val_sets)
            val_loader = DataLoader(
                MultiDomainDataset(source_stores, val_refs),
                batch_size=args.batch_size,
                shuffle=False,
            )
            model = build_model(args, len(source_stores)).to(device)
            checkpoint_path = summary_path.parent / f"target_{target_name}" / "best_model.pth"
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            val_subjects, val_true, val_prob_two = collect_subject_probabilities(
                model, val_loader, device
            )
            val_probability = val_prob_two[:, 1]

            target_metrics = fold["metrics"]
            target_true = np.asarray(target_metrics["y_true"], dtype=np.int64)
            target_probability = np.asarray(target_metrics["y_prob"], dtype=np.float64)
            target_subjects = target_metrics.get("subject_ids")
            if target_subjects is None:
                target_subjects = [f"target_index_{index:04d}" for index in range(len(target_true))]
            calibrators: dict[str, tuple[float, ...]] = {
                "identity": (),
                "temperature": (fit_temperature(val_true, val_probability),),
                "platt": fit_platt(val_true, val_probability),
            }
            for calibrator, parameters in calibrators.items():
                calibrated_val = apply_calibrator(val_probability, calibrator, parameters)
                calibrated_target = apply_calibrator(
                    target_probability, calibrator, parameters
                )
                threshold, threshold_score = choose_threshold(
                    val_true,
                    np.column_stack([1.0 - calibrated_val, calibrated_val]),
                    "balanced_acc",
                )
                metrics = probability_metrics(target_true, calibrated_target, threshold)
                intercept, slope = calibration_intercept_slope(
                    target_true, calibrated_target
                )
                rows.append(
                    {
                        "backbone": args.backbone,
                        "variant": variant,
                        "method": payload["method"],
                        "seed": seed,
                        "target": target_name,
                        "calibrator": calibrator,
                        "source_validation_subjects": len(val_true),
                        "target_subjects": len(target_true),
                        "threshold": float(threshold),
                        "threshold_score": float(threshold_score),
                        "parameter_1": parameters[0] if parameters else "",
                        "parameter_2": parameters[1] if len(parameters) > 1 else "",
                        "calibration_intercept": intercept,
                        "calibration_slope": slope,
                    }
                    | metrics
                )
                for split, subjects, labels, raw_values, calibrated_values in (
                    ("source_validation", val_subjects, val_true, val_probability, calibrated_val),
                    ("target", target_subjects, target_true, target_probability, calibrated_target),
                ):
                    for subject, label, raw_value, calibrated_value in zip(
                        subjects, labels, raw_values, calibrated_values
                    ):
                        prediction_rows.append(
                            {
                                "backbone": args.backbone,
                                "variant": variant,
                                "method": payload["method"],
                                "seed": seed,
                                "target": target_name,
                                "calibrator": calibrator,
                                "split": split,
                                "subject_id": str(subject),
                                "y_true": int(label),
                                "raw_probability": float(raw_value),
                                "probability": float(calibrated_value),
                            }
                        )
            print(f"Processed {variant} seed={seed} target={target_name}")

    seed_rows, summary_rows = summarize(rows)
    prefix = Path(args.output_prefix)
    write_csv(prefix.with_name(prefix.name + "_folds.csv"), rows)
    write_csv(prefix.with_name(prefix.name + "_seeds.csv"), seed_rows)
    write_csv(prefix.with_name(prefix.name + "_summary.csv"), summary_rows)
    write_csv(prefix.with_name(prefix.name + "_predictions.csv"), prediction_rows)
    output = {
        "metadata": {
            "backbone": args.backbone,
            "source_validation_labels_used_for_calibration": True,
            "target_samples_used_for_calibration": False,
            "target_labels_used_for_calibration": False,
            "calibrators": ["identity", "temperature", "platt"],
        },
        "summary": summary_rows,
        "seed_macro": seed_rows,
        "folds": rows,
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved calibration baseline results to {prefix.parent}")


if __name__ == "__main__":
    main()
