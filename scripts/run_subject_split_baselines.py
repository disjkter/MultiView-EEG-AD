"""Run subject-level train/val/test baseline experiments on one EEG dataset.

This script mirrors ``run_dual_stream_subject_split.py`` so baseline models can
be compared against TAGE-SSL under the same CAUEEG split protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.EEGNet import EEGNet  # noqa: E402
from models.iTransformer import ITransformer  # noqa: E402
from models.MTDNet import MTDNet  # noqa: E402
from models.ShallowConvNet import ShallowConvNet  # noqa: E402
from models.TCN import TCN  # noqa: E402
from scripts.run_dual_stream_loso_subject import choose_threshold, safe_auc, set_seed  # noqa: E402
from scripts.run_dual_stream_subject_split import (  # noqa: E402
    IndexedEEGDataset,
    collect_subject_probabilities,
    compute_metrics,
    count_subjects,
    fit_euclidean_alignment,
    load_domain_mmap,
    sample_indices_per_subject,
    stratified_subject_split,
)
from utils.subject_split import load_subject_split_masks, save_subject_split  # noqa: E402
from models.CEEDNet import CEEDNet1DResNet18  # noqa: E402
from models.EEGConformer import EEGConformer  # noqa: E402
from models.EEGInception import EEGInception  # noqa: E402
from models.LCADNet import LCADNet  # noqa: E402


METRICS = ["accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity"]


def build_model(
    model_name: str,
    num_channels: int,
    num_classes: int,
    dropout: float,
) -> nn.Module:
    """Instantiate one baseline model."""

    if model_name == "eegnet":
        return EEGNet(num_channels=num_channels, num_classes=num_classes, dropout=dropout)
    if model_name == "shallowconvnet":
        return ShallowConvNet(num_channels=num_channels, num_classes=num_classes, dropout=dropout)
    if model_name == "tcn":
        return TCN(num_channels=num_channels, num_classes=num_classes, dropout=dropout)
    if model_name == "mtdnet":
        return MTDNet(num_channels=num_channels, num_classes=num_classes)
    if model_name == "itransformer":
        return ITransformer(num_channels=num_channels, num_classes=num_classes, dropout=dropout)
    if model_name == "ceednet":
        return CEEDNet1DResNet18(num_channels=num_channels, num_classes=num_classes, dropout=dropout)
    if model_name == "eegconformer":
        return EEGConformer(num_channels=num_channels, num_classes=num_classes, dropout=dropout)
    if model_name == "eeginception":
        return EEGInception(num_channels=num_channels, num_classes=num_classes, dropout=dropout)
    if model_name == "lcadnet":
        return LCADNet(num_channels=num_channels, num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unsupported model: {model_name}")


def build_svm(seed: int) -> Pipeline:
    """Build an RBF-SVM baseline for flattened EEG windows."""

    return Pipeline(
        steps=[
            ("svd", TruncatedSVD(n_components=128, algorithm="randomized", random_state=seed)),
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        ]
    )


def flatten_indexed_dataset(dataset: IndexedEEGDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize an indexed EEG dataset into flattened samples."""

    x_flat: list[np.ndarray] = []
    y: list[int] = []
    subjects: list[str] = []
    old_return_subject = dataset.return_subject
    dataset.return_subject = True
    for idx in range(len(dataset)):
        sample, label, subject = dataset[idx]
        x_flat.append(sample.numpy().reshape(-1))
        y.append(int(label))
        subjects.append(str(subject))
    dataset.return_subject = old_return_subject
    return np.asarray(x_flat, dtype=np.float32), np.asarray(y, dtype=np.int64), np.asarray(subjects, dtype=object)


def collect_subject_probabilities_svm(
    model: Pipeline,
    dataset: IndexedEEGDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate SVM segment probabilities into subject-level probabilities."""

    x_flat, y, subjects = flatten_indexed_dataset(dataset)
    prob_all = model.predict_proba(x_flat).astype(np.float32, copy=False)
    subject_probs: dict[str, list[np.ndarray]] = {}
    subject_labels: dict[str, int] = {}
    for subject, label, prob in zip(subjects, y, prob_all):
        subject_probs.setdefault(str(subject), []).append(prob)
        subject_labels[str(subject)] = int(label)

    unique_subjects = np.asarray(sorted(subject_probs.keys()), dtype=object)
    y_true = np.asarray([subject_labels[str(subject)] for subject in unique_subjects], dtype=np.int64)
    y_prob = np.asarray([np.mean(subject_probs[str(subject)], axis=0) for subject in unique_subjects], dtype=np.float32)
    return unique_subjects, y_true, y_prob


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one train/eval epoch."""

    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    preds: list[int] = []
    targets: list[int] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        with torch.set_grad_enabled(is_train):
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(float(loss.item()))
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        targets.extend(batch_y.detach().cpu().numpy().tolist())

    acc = float(np.mean(np.asarray(preds) == np.asarray(targets))) if targets else 0.0
    return float(np.mean(losses)) if losses else 0.0, acc


def summarize_metric(values: list[float]) -> dict[str, float]:
    """Return mean/std for finite metric values."""

    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return {"mean": float("nan"), "std": float("nan"), "num_valid": 0}
    return {
        "mean": float(statistics.fmean(finite_values)),
        "std": float(statistics.pstdev(finite_values)) if len(finite_values) > 1 else 0.0,
        "num_valid": len(finite_values),
    }


def run_one_seed(args: argparse.Namespace, seed: int, output_dir: Path) -> dict[str, object]:
    """Run one seeded subject-level baseline experiment."""

    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_all, y_all, sub_ids_all = load_domain_mmap(Path(args.data_dir))
    if args.split_file and Path(args.split_file).exists():
        train_mask, val_mask, test_mask = load_subject_split_masks(args.split_file, sub_ids_all)
    else:
        train_mask, val_mask, test_mask = stratified_subject_split(y_all, sub_ids_all, args.train_ratio, args.val_ratio, seed)
    if args.save_split_file:
        save_subject_split(
            args.save_split_file,
            sub_ids_all,
            train_mask,
            val_mask,
            test_mask,
            metadata={
                "data_dir": args.data_dir,
                "seed": seed,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
            },
        )
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]
    train_indices = sample_indices_per_subject(train_indices, sub_ids_all, args.max_train_segments_per_subject, seed + 11)
    val_indices = sample_indices_per_subject(val_indices, sub_ids_all, args.max_val_segments_per_subject, seed + 17)
    test_indices = sample_indices_per_subject(test_indices, sub_ids_all, args.max_test_segments_per_subject, seed + 23)

    ea_matrix = None
    if args.use_ea:
        ea_indices = train_indices
        if len(ea_indices) > args.max_ea_samples:
            rng = np.random.default_rng(seed + 31)
            ea_indices = np.sort(rng.choice(ea_indices, size=args.max_ea_samples, replace=False))
        ea_samples = []
        for idx in ea_indices:
            sample = np.asarray(x_all[int(idx)], dtype=np.float32).copy()
            mean = sample.mean(axis=1, keepdims=True)
            std = sample.std(axis=1, keepdims=True) + 1e-6
            ea_samples.append((sample - mean) / std)
        ea_matrix = fit_euclidean_alignment(np.asarray(ea_samples, dtype=np.float32))

    train_dataset = IndexedEEGDataset(x_all, y_all, sub_ids_all, train_indices, ea_matrix=ea_matrix)
    val_dataset = IndexedEEGDataset(x_all, y_all, sub_ids_all, val_indices, ea_matrix=ea_matrix)
    test_dataset = IndexedEEGDataset(x_all, y_all, sub_ids_all, test_indices, ea_matrix=ea_matrix)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)

    if args.model == "svm":
        model = build_svm(seed)
        x_train_flat, y_train_flat, _ = flatten_indexed_dataset(train_dataset)
        model.fit(x_train_flat, y_train_flat)

        if args.tune_threshold:
            _, val_true, val_prob = collect_subject_probabilities_svm(model, val_dataset)
            threshold, threshold_score = choose_threshold(val_true, val_prob, args.threshold_objective)
        else:
            threshold = None
            threshold_score = None

        test_subjects, test_true, test_prob = collect_subject_probabilities_svm(model, test_dataset)
        metrics = compute_metrics(test_subjects, test_true, test_prob, threshold)
        summary = {
            "model": args.model,
            "data_dir": args.data_dir,
            "seed": seed,
            "split_file": args.split_file,
            "save_split_file": args.save_split_file,
            "split": {
                "train_segments": int(len(train_indices)),
                "val_segments": int(len(val_indices)),
                "test_segments": int(len(test_indices)),
                "train_subjects_by_label": count_subjects(y_all, sub_ids_all, train_indices),
                "val_subjects_by_label": count_subjects(y_all, sub_ids_all, val_indices),
                "test_subjects_by_label": count_subjects(y_all, sub_ids_all, test_indices),
            },
            "use_ea": args.use_ea,
            "max_ea_samples": args.max_ea_samples if args.use_ea else None,
            "max_train_segments_per_subject": args.max_train_segments_per_subject,
            "max_val_segments_per_subject": args.max_val_segments_per_subject,
            "max_test_segments_per_subject": args.max_test_segments_per_subject,
            "disable_class_weights": None,
            "tune_threshold": args.tune_threshold,
            "threshold_objective": args.threshold_objective,
            "decision_threshold": threshold,
            "threshold_score": threshold_score,
            "selection_metric": None,
            "best_epoch": None,
            "best_val_loss": None,
            "best_val_auc": safe_auc(val_true, val_prob) if args.tune_threshold else None,
            "history": [],
            "metrics": metrics,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("\nSubject-level test summary")
        print(f"Accuracy     : {metrics['accuracy']:.4f}")
        print(f"Balanced Acc : {metrics['balanced_accuracy']:.4f}")
        print(f"F1           : {metrics['f1']:.4f}")
        print(f"AUC          : {metrics['auc']:.4f}")
        print(f"Sensitivity  : {metrics['sensitivity']:.4f}")
        print(f"Specificity  : {metrics['specificity']:.4f}")
        print(metrics["report"])
        return summary

    model = build_model(
        model_name=args.model,
        num_channels=int(x_all.shape[1]),
        num_classes=int(len(np.unique(y_all))),
        dropout=args.dropout,
    ).to(device)

    class_weight = None
    if not args.disable_class_weights:
        class_counts = np.bincount(y_all[train_indices], minlength=int(y_all.max() + 1))
        class_weights = len(train_indices) / (len(class_counts) * np.maximum(class_counts, 1))
        class_weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_model_path = output_dir / "best_model.pth"
    best_val_loss = float("inf")
    best_val_auc = -float("inf")
    best_epoch = 0
    patience_left = args.patience
    history: list[dict[str, float | int | None]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        _, val_true_epoch, val_prob_epoch = collect_subject_probabilities(model, val_dataset, args.batch_size, device)
        val_auc = safe_auc(val_true_epoch, val_prob_epoch)
        print(
            f"Seed {seed} | {args.model} | Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_auc": None if np.isnan(val_auc) else float(val_auc),
            }
        )

        if args.selection_metric == "auc" and not np.isnan(val_auc):
            improved = val_auc > best_val_auc
        else:
            improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_epoch = epoch
            patience_left = args.patience
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    if not best_model_path.exists():
        torch.save(model.state_dict(), best_model_path)
        best_epoch = int(history[-1]["epoch"]) if history else 0

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    if args.tune_threshold:
        _, val_true, val_prob = collect_subject_probabilities(model, val_dataset, args.batch_size, device)
        threshold, threshold_score = choose_threshold(val_true, val_prob, args.threshold_objective)
    else:
        threshold = None
        threshold_score = None

    test_subjects, test_true, test_prob = collect_subject_probabilities(model, test_dataset, args.batch_size, device)
    metrics = compute_metrics(test_subjects, test_true, test_prob, threshold)

    summary = {
        "model": args.model,
        "data_dir": args.data_dir,
        "seed": seed,
        "split_file": args.split_file,
        "save_split_file": args.save_split_file,
        "split": {
            "train_segments": int(len(train_indices)),
            "val_segments": int(len(val_indices)),
            "test_segments": int(len(test_indices)),
            "train_subjects_by_label": count_subjects(y_all, sub_ids_all, train_indices),
            "val_subjects_by_label": count_subjects(y_all, sub_ids_all, val_indices),
            "test_subjects_by_label": count_subjects(y_all, sub_ids_all, test_indices),
        },
        "use_ea": args.use_ea,
        "max_ea_samples": args.max_ea_samples if args.use_ea else None,
        "max_train_segments_per_subject": args.max_train_segments_per_subject,
        "max_val_segments_per_subject": args.max_val_segments_per_subject,
        "max_test_segments_per_subject": args.max_test_segments_per_subject,
        "disable_class_weights": args.disable_class_weights,
        "tune_threshold": args.tune_threshold,
        "threshold_objective": args.threshold_objective,
        "decision_threshold": threshold,
        "threshold_score": threshold_score,
        "selection_metric": args.selection_metric,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_auc": None if np.isnan(best_val_auc) else float(best_val_auc),
        "history": history,
        "metrics": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSubject-level test summary")
    print(f"Accuracy     : {metrics['accuracy']:.4f}")
    print(f"Balanced Acc : {metrics['balanced_accuracy']:.4f}")
    print(f"F1           : {metrics['f1']:.4f}")
    print(f"AUC          : {metrics['auc']:.4f}")
    print(f"Sensitivity  : {metrics['sensitivity']:.4f}")
    print(f"Specificity  : {metrics['specificity']:.4f}")
    print(metrics["report"])
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Subject-level baseline runner for CAUEEG-style datasets")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--model",
        choices=["svm", "eegnet", "shallowconvnet", "tcn", "mtdnet", "itransformer", "ceednet", "lcadnet", "eegconformer", "eeginception"],
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--save-split-file", default=None)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--use-ea", action="store_true")
    parser.add_argument("--max-ea-samples", type=int, default=5000)
    parser.add_argument("--max-train-segments-per-subject", type=int, default=None)
    parser.add_argument("--max-val-segments-per-subject", type=int, default=None)
    parser.add_argument("--max-test-segments-per-subject", type=int, default=None)
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--threshold-objective", choices=["accuracy", "balanced_acc", "f1", "precision", "specificity", "youden"], default="balanced_acc")
    parser.add_argument("--selection-metric", choices=["loss", "auc"], default="auc")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    """Run all requested seeds and write an aggregate summary."""

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_summaries = []
    for seed in args.seeds:
        seed_output_dir = output_dir / f"seed_{seed}"
        seed_summaries.append(run_one_seed(args, seed, seed_output_dir))

    aggregate_metrics = {
        metric: summarize_metric([float(summary["metrics"][metric]) for summary in seed_summaries])
        for metric in METRICS
    }
    aggregate = {
        "model": args.model,
        "seeds": args.seeds,
        "aggregate_metrics": aggregate_metrics,
        "seed_runs": [
            {
                "seed": summary["seed"],
                "best_epoch": summary["best_epoch"],
                "decision_threshold": summary["decision_threshold"],
                "metrics": {metric: summary["metrics"][metric] for metric in METRICS},
            }
            for summary in seed_summaries
        ],
    }
    (output_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print("\nAggregate multi-seed results")
    for metric, stats in aggregate_metrics.items():
        print(f"{metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")


if __name__ == "__main__":
    main()
