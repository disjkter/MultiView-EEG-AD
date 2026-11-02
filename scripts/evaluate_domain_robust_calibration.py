"""Evaluate source-domain-robust calibration and threshold selection for LODO models."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import balanced_accuracy_score, log_loss
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_lodo_calibration_baselines import (  # noqa: E402
    apply_calibrator,
    fit_platt,
    fit_temperature,
    positive_logit,
    probability_metrics,
    sigmoid,
    summarize,
    write_csv,
)
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


def smooth_max(values: np.ndarray, sharpness: float) -> float:
    """Return a stable log-sum-exp approximation to the maximum."""

    values = np.asarray(values, dtype=np.float64)
    maximum = float(values.max())
    return maximum + math.log(float(np.exp(sharpness * (values - maximum)).mean())) / sharpness


def domain_nll(
    labels: list[np.ndarray],
    probabilities: list[np.ndarray],
) -> np.ndarray:
    """Compute one equally weighted NLL value per source domain."""

    return np.asarray(
        [log_loss(y, p, labels=[0, 1]) for y, p in zip(labels, probabilities)],
        dtype=np.float64,
    )


def fit_robust_temperature(
    labels: list[np.ndarray],
    probabilities: list[np.ndarray],
    sharpness: float = 10.0,
    identity_penalty: float = 0.01,
) -> float:
    """Fit temperature by minimizing smooth worst-source NLL."""

    logits = [positive_logit(probability) for probability in probabilities]

    def objective(log_temperature: float) -> float:
        calibrated = [sigmoid(logit / math.exp(log_temperature)) for logit in logits]
        return smooth_max(domain_nll(labels, calibrated), sharpness) + identity_penalty * log_temperature**2

    result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Robust temperature optimization failed: {result.message}")
    return float(math.exp(result.x))


def fit_robust_platt(
    labels: list[np.ndarray],
    probabilities: list[np.ndarray],
    sharpness: float = 10.0,
    identity_penalty: float = 0.01,
) -> tuple[float, float]:
    """Fit monotonic Platt parameters using smooth worst-source NLL."""

    logits = [positive_logit(probability) for probability in probabilities]

    def objective(parameters: np.ndarray) -> float:
        log_slope, intercept = float(parameters[0]), float(parameters[1])
        slope = math.exp(log_slope)
        calibrated = [sigmoid(slope * logit + intercept) for logit in logits]
        regularizer = identity_penalty * (log_slope**2 + intercept**2)
        return smooth_max(domain_nll(labels, calibrated), sharpness) + regularizer

    result = minimize(
        objective,
        x0=np.asarray([0.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(-5.0, 5.0), (-20.0, 20.0)],
    )
    if not result.success:
        raise RuntimeError(f"Robust Platt optimization failed: {result.message}")
    return float(math.exp(result.x[0])), float(result.x[1])


def choose_minimax_threshold(
    labels: list[np.ndarray],
    probabilities: list[np.ndarray],
) -> tuple[float, float, float]:
    """Select the threshold maximizing worst-source then mean-source BA."""

    pooled = np.concatenate(probabilities)
    unique = np.unique(np.clip(pooled, 0.0, 1.0))
    candidates = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, 0.5, 1.0]),
                unique,
                (unique[:-1] + unique[1:]) / 2.0,
            ]
        )
    )
    best_key = (-float("inf"), -float("inf"))
    best_threshold = 0.5
    for threshold in candidates:
        scores = np.asarray(
            [
                balanced_accuracy_score(y, p >= threshold)
                for y, p in zip(labels, probabilities)
            ],
            dtype=np.float64,
        )
        key = (float(scores.min()), float(scores.mean()))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold, best_key[0], best_key[1]


def parse_args() -> argparse.Namespace:
    """Parse evaluation and model-reconstruction arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--backbone", choices=["graph_temporal", "eegnet"], required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-source-subjects-per-class", type=int, default=50)
    parser.add_argument("--max-segments-per-subject", type=int, default=20)
    parser.add_argument("--robustness", type=float, default=10.0)
    parser.add_argument("--identity-penalty", type=float, default=0.01)
    parser.add_argument("--graph-out-dim", type=int, default=256)
    parser.add_argument("--temporal-out-dim", type=int, default=256)
    parser.add_argument("--disease-dim", type=int, default=None)
    parser.add_argument("--domain-dim", type=int, default=None)
    parser.add_argument("--graph-num-neighbors", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--eegnet-temporal-filters", type=int, default=8)
    parser.add_argument("--eegnet-depth-multiplier", type=int, default=2)
    parser.add_argument("--eegnet-separable-filters", type=int, default=16)
    parser.add_argument("--eegnet-temporal-kernel", type=int, default=64)
    parser.add_argument("--eegnet-separable-kernel", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--limit-summaries", type=int, default=0)
    args = parser.parse_args()
    if args.disease_dim is None:
        args.disease_dim = 64 if args.backbone == "eegnet" else 256
    if args.domain_dim is None:
        args.domain_dim = 32 if args.backbone == "eegnet" else 128
    return args


def main() -> None:
    """Run matched source-only calibration and threshold comparisons."""

    args = parse_args()
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    domain_paths = parse_domains(args.domains)
    stores = {name: load_store(name, path) for name, path in domain_paths.items()}
    results_dir = Path(args.results_dir)
    summary_paths = sorted(results_dir.glob("seed*/**/lodo_summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No lodo_summary.json files under {results_dir}")
    if args.seeds is not None:
        selected_seeds = set(args.seeds)
        summary_paths = [
            path
            for path in summary_paths
            if int(json.loads(path.read_text(encoding="utf-8"))["seed"])
            in selected_seeds
        ]
        if not summary_paths:
            raise ValueError(f"No summaries matched seeds {sorted(selected_seeds)}")
    if args.limit_summaries > 0:
        summary_paths = summary_paths[: args.limit_summaries]

    rows: list[dict[str, object]] = []
    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("backbone", "graph_temporal") != args.backbone:
            raise ValueError(f"Backbone mismatch in {summary_path}")
        seed = int(payload["seed"])
        variant = summary_path.parent.name
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

            model = build_model(args, len(source_stores)).to(device)
            checkpoint_path = summary_path.parent / f"target_{target_name}" / "best_model.pth"
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))

            source_labels: list[np.ndarray] = []
            source_probabilities: list[np.ndarray] = []
            for store, indices in zip(source_stores, val_sets):
                loader = DataLoader(
                    MultiDomainDataset([store], make_refs([store], [indices])),
                    batch_size=args.batch_size,
                    shuffle=False,
                )
                _, labels, probabilities = collect_subject_probabilities(model, loader, device)
                source_labels.append(labels)
                source_probabilities.append(probabilities[:, 1])

            pooled_labels = np.concatenate(source_labels)
            pooled_probability = np.concatenate(source_probabilities)
            target_metrics = fold["metrics"]
            target_labels = np.asarray(target_metrics["y_true"], dtype=np.int64)
            target_probability = np.asarray(target_metrics["y_prob"], dtype=np.float64)

            pooled_parameters = {
                "identity": (),
                "temperature": (fit_temperature(pooled_labels, pooled_probability),),
                "platt": fit_platt(pooled_labels, pooled_probability),
            }
            robust_parameters = {
                "temperature": (
                    fit_robust_temperature(
                        source_labels,
                        source_probabilities,
                        args.robustness,
                        args.identity_penalty,
                    ),
                ),
                "platt": fit_robust_platt(
                    source_labels,
                    source_probabilities,
                    args.robustness,
                    args.identity_penalty,
                ),
            }
            raw_minimax_threshold, raw_worst_ba, raw_mean_ba = choose_minimax_threshold(
                source_labels, source_probabilities
            )
            strategies = [
                ("identity_pooled", "identity", pooled_parameters["identity"], "pooled"),
                ("identity_minimax", "identity", pooled_parameters["identity"], "minimax"),
                ("temperature_pooled", "temperature", pooled_parameters["temperature"], "pooled"),
                ("platt_pooled", "platt", pooled_parameters["platt"], "pooled"),
                ("temperature_minimax", "temperature", robust_parameters["temperature"], "minimax"),
                ("platt_minimax", "platt", robust_parameters["platt"], "minimax"),
            ]
            for strategy, calibrator, parameters, threshold_rule in strategies:
                calibrated_sources = [
                    apply_calibrator(probability, calibrator, parameters)
                    for probability in source_probabilities
                ]
                calibrated_pooled = np.concatenate(calibrated_sources)
                if threshold_rule == "pooled":
                    threshold, threshold_score = choose_threshold(
                        pooled_labels,
                        np.column_stack([1.0 - calibrated_pooled, calibrated_pooled]),
                        "balanced_acc",
                    )
                    source_worst_ba = min(
                        balanced_accuracy_score(y, p >= threshold)
                        for y, p in zip(source_labels, calibrated_sources)
                    )
                    source_mean_ba = float(
                        np.mean(
                            [
                                balanced_accuracy_score(y, p >= threshold)
                                for y, p in zip(source_labels, calibrated_sources)
                            ]
                        )
                    )
                else:
                    threshold = raw_minimax_threshold
                    source_worst_ba = raw_worst_ba
                    source_mean_ba = raw_mean_ba
                    threshold_score = source_worst_ba
                calibrated_target = apply_calibrator(target_probability, calibrator, parameters)
                metrics = probability_metrics(target_labels, calibrated_target, threshold)
                if threshold_rule == "minimax":
                    decision_metrics = binary_metrics(
                        target_labels,
                        np.column_stack([1.0 - target_probability, target_probability]),
                        raw_minimax_threshold,
                    )
                    for metric_name in (
                        "accuracy",
                        "balanced_accuracy",
                        "f1",
                        "auc",
                        "sensitivity",
                        "specificity",
                    ):
                        metrics[metric_name] = float(decision_metrics[metric_name])
                rows.append(
                    {
                        "backbone": args.backbone,
                        "variant": variant,
                        "method": payload["method"],
                        "seed": seed,
                        "target": target_name,
                        "calibrator": strategy,
                        "source_validation_subjects": len(pooled_labels),
                        "target_subjects": len(target_labels),
                        "threshold": float(threshold),
                        "threshold_score": float(threshold_score),
                        "source_worst_balanced_accuracy": float(source_worst_ba),
                        "source_mean_balanced_accuracy": float(source_mean_ba),
                        "parameter_1": parameters[0] if parameters else "",
                        "parameter_2": parameters[1] if len(parameters) > 1 else "",
                    }
                    | metrics
                )
            print(f"Processed {variant} seed={seed} target={target_name}")

    seed_rows, summary_rows = summarize(rows)
    prefix = Path(args.output_prefix)
    write_csv(prefix.with_name(prefix.name + "_folds.csv"), rows)
    write_csv(prefix.with_name(prefix.name + "_seeds.csv"), seed_rows)
    write_csv(prefix.with_name(prefix.name + "_summary.csv"), summary_rows)
    output = {
        "metadata": {
            "backbone": args.backbone,
            "source_validation_labels_used": True,
            "target_samples_used": False,
            "target_labels_used": False,
            "robustness": args.robustness,
            "identity_penalty": args.identity_penalty,
            "selection_objective": "lexicographic worst-source then mean-source balanced accuracy",
        },
        "summary": summary_rows,
        "seed_macro": seed_rows,
        "folds": rows,
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved domain-robust calibration results to {prefix.parent}")


if __name__ == "__main__":
    main()
