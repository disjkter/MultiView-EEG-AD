"""Run cross-domain experiments with the dual-stream EEG classifier.

This script trains ``SpatialMSCNN_DualStream`` on a source domain stored as
NumPy arrays and evaluates on a target domain with strict subject-level
aggregation. It supports:

- optional pretrained initialization
- Euclidean Alignment (fit on source only, then applied to target)
- linear probing or full fine-tuning
- source-domain subject-level train/val split for early stopping
- target-domain subject-level voting and metrics export
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import fractional_matrix_power
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.Spatial_MSCNN import SpatialMSCNN_DualStream
from utils.dual_stream_config import add_config_argument, apply_yaml_defaults


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def load_domain(domain_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one domain as ``(x, y, subject_ids)``."""

    x_path, y_path, id_path = resolve_domain_files(domain_dir)
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    sub_ids = np.load(id_path, allow_pickle=True)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    if y.dtype != np.int64:
        y = y.astype(np.int64, copy=False)
    return x, y, sub_ids


def fit_euclidean_alignment(x_train: np.ndarray) -> np.ndarray:
    """Fit Euclidean Alignment on source-domain samples only."""

    n_trials, n_channels, _ = x_train.shape
    covs = np.zeros((n_trials, n_channels, n_channels), dtype=np.float64)
    for i in range(n_trials):
        covs[i] = np.cov(x_train[i])
    mean_cov = np.mean(covs, axis=0)
    mean_cov += np.eye(n_channels, dtype=np.float64) * 1e-6
    r_inv_half = fractional_matrix_power(mean_cov, -0.5)
    return np.real(r_inv_half).astype(np.float32)


def apply_euclidean_alignment(x: np.ndarray, r_inv_half: np.ndarray) -> np.ndarray:
    """Apply a fitted Euclidean Alignment transform to each sample."""

    if x.nbytes >= 256 * 1024 * 1024:
        fd, tmp_path = tempfile.mkstemp(prefix="ea_dual_stream_", suffix=".dat")
        os.close(fd)
        aligned = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=x.shape)
    else:
        aligned = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        aligned[i] = r_inv_half @ x[i]
    return aligned


def split_source_subjects(
    y: np.ndarray,
    sub_ids: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a subject-level source train/val split."""

    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(sub_ids)
    subject_labels = {
        subject: int(y[np.where(sub_ids == subject)[0][0]]) for subject in unique_subjects
    }

    train_subjects: list[object] = []
    val_subjects: list[object] = []
    for label in sorted(set(subject_labels.values())):
        label_subjects = [s for s in unique_subjects if subject_labels[s] == label]
        label_subjects = list(label_subjects)
        rng.shuffle(label_subjects)
        n_val = max(1, int(round(len(label_subjects) * val_ratio)))
        if n_val >= len(label_subjects) and len(label_subjects) > 1:
            n_val = len(label_subjects) - 1
        val_subjects.extend(label_subjects[:n_val])
        train_subjects.extend(label_subjects[n_val:])

    if not train_subjects or not val_subjects:
        raise ValueError("Source subject split failed; adjust val_ratio or dataset size.")

    train_mask = np.isin(sub_ids, np.array(train_subjects, dtype=object))
    val_mask = np.isin(sub_ids, np.array(val_subjects, dtype=object))
    return train_mask, val_mask


class EEGDataset(Dataset):
    """Simple EEG dataset with per-sample channel-wise normalization."""

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        self.x = np.asarray(x_data, dtype=np.float32)
        self.y = np.asarray(y_data, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = np.array(self.x[idx], dtype=np.float32, copy=True)
        mean = sample.mean(axis=1, keepdims=True)
        std = sample.std(axis=1, keepdims=True) + 1e-6
        sample = (sample - mean) / std
        return torch.from_numpy(sample), torch.tensor(self.y[idx], dtype=torch.long)


@dataclass
class EpochResult:
    """Container for epoch statistics."""

    loss: float
    acc: float


def collect_subject_probabilities(
    model: nn.Module,
    x_data: np.ndarray,
    y_data: np.ndarray,
    sub_ids: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate sample probabilities into subject-level probabilities."""

    loader = DataLoader(EEGDataset(x_data, y_data), batch_size=batch_size, shuffle=False)
    model.eval()
    probs_all: list[np.ndarray] = []
    labels_all: list[int] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits, _ = model(batch_x.to(device))
            probs_all.extend(F.softmax(logits, dim=1).cpu().numpy())
            labels_all.extend(batch_y.numpy().tolist())

    probs_all = np.asarray(probs_all, dtype=np.float32)
    labels_all = np.asarray(labels_all, dtype=np.int64)
    subject_ids_unique = np.unique(sub_ids)
    subject_true: list[int] = []
    subject_probs: list[np.ndarray] = []

    for subject in subject_ids_unique:
        idx = np.where(sub_ids == subject)[0]
        subject_probs.append(probs_all[idx].mean(axis=0))
        subject_true.append(int(labels_all[idx[0]]))

    return (
        subject_ids_unique,
        np.asarray(subject_true, dtype=np.int64),
        np.asarray(subject_probs, dtype=np.float32),
    )


def subject_predictions_from_probs(y_prob: np.ndarray, threshold: float | None) -> np.ndarray:
    """Convert subject-level probabilities to hard predictions."""

    if y_prob.shape[1] == 2 and threshold is not None:
        return (y_prob[:, 1] >= threshold).astype(np.int64)
    return np.argmax(y_prob, axis=1).astype(np.int64)


def choose_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str,
) -> tuple[float, float]:
    """Tune a binary decision threshold on validation subjects."""

    if y_prob.shape[1] != 2:
        return 0.5, float("nan")

    candidate_thresholds = np.linspace(0.1, 0.9, 33)
    best_threshold = 0.5
    best_score = -float("inf")

    for threshold in candidate_thresholds:
        y_pred = subject_predictions_from_probs(y_prob, threshold)
        if objective == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif objective == "balanced_acc":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            tn = float(np.sum((y_true == 0) & (y_pred == 0)))
            fp = float(np.sum((y_true == 0) & (y_pred == 1)))
            fn = float(np.sum((y_true == 1) & (y_pred == 0)))
            tp = float(np.sum((y_true == 1) & (y_pred == 1)))
            sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            score = sensitivity + specificity - 1.0
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_score


def compute_subject_metrics(
    subject_ids_unique: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None,
) -> dict[str, object]:
    """Compute subject-level metrics from probabilities."""

    y_pred = subject_predictions_from_probs(y_prob, threshold)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    auc = roc_auc_score(y_true, y_prob[:, 1]) if len(np.unique(y_true)) == 2 else float("nan")

    return {
        "subject_ids": subject_ids_unique.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
        "decision_threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "auc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "report": classification_report(y_true, y_pred, digits=4, zero_division=0),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochResult:
    """Run one train or eval epoch."""

    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    all_preds: list[int] = []
    all_targets: list[int] = []

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
        all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        all_targets.extend(batch_y.detach().cpu().numpy().tolist())

    return EpochResult(
        loss=float(np.mean(losses)) if losses else 0.0,
        acc=float(accuracy_score(all_targets, all_preds)) if all_targets else 0.0,
    )


    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    auc = roc_auc_score(y_true, y_prob[:, 1]) if len(np.unique(y_true)) == 2 else float("nan")

    return {
        "subject_ids": subject_ids_unique.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
        "decision_threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "auc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "report": classification_report(y_true, y_pred, digits=4, zero_division=0),
    }


def build_model(
    num_classes: int,
    num_channels: int,
    checkpoint: str | None,
    linear_probe: bool,
    device: torch.device,
) -> nn.Module:
    """Create the dual-stream model and optionally load pretrained weights."""

    model = SpatialMSCNN_DualStream(num_classes=num_classes, num_channels=num_channels).to(device)
    if checkpoint:
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    if linear_probe:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fusion_classifier.parameters():
            param.requires_grad = True

    return model


def main() -> None:
    """Parse arguments and run the cross-domain experiment."""

    parser = argparse.ArgumentParser(description="Dual-stream cross-domain EEG experiment")
    add_config_argument(parser)
    parser.add_argument("--source-dir", required=True, help="Processed source domain directory")
    parser.add_argument("--target-dir", required=True, help="Processed target domain directory")
    parser.add_argument("--checkpoint", default=None, help="Optional pretrained model checkpoint")
    parser.add_argument("--output-dir", default="outputs/cross_domain_dual_stream", help="Result directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--use-ea", action="store_true", help="Apply Euclidean Alignment fit on source only")
    parser.add_argument("--linear-probe", action="store_true", help="Freeze backbone and train classifier only")
    parser.add_argument("--disable-class-weights", action="store_true", help="Use unweighted cross-entropy")
    parser.add_argument("--decision-threshold", type=float, default=None, help="Fixed positive-class threshold for binary tasks")
    parser.add_argument("--tune-threshold", action="store_true", help="Tune binary threshold on source validation subjects")
    parser.add_argument(
        "--threshold-objective",
        choices=["balanced_acc", "f1", "youden"],
        default="balanced_acc",
        help="Validation objective for threshold tuning",
    )
    args = parser.parse_args()
    args = apply_yaml_defaults(
        parser,
        args,
        [
            "checkpoint",
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "val_ratio",
            "patience",
            "seed",
            "use_ea",
            "linear_probe",
            "disable_class_weights",
            "tune_threshold",
            "threshold_objective",
        ],
    )

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    x_source, y_source, sub_ids_source = load_domain(source_dir)
    x_target, y_target, sub_ids_target = load_domain(target_dir)

    train_mask, val_mask = split_source_subjects(y_source, sub_ids_source, args.val_ratio, args.seed)
    x_train, y_train = x_source[train_mask], y_source[train_mask]
    x_val, y_val = x_source[val_mask], y_source[val_mask]

    if args.use_ea:
        r_inv_half = fit_euclidean_alignment(x_train)
        x_train = apply_euclidean_alignment(x_train, r_inv_half)
        x_val = apply_euclidean_alignment(x_val, r_inv_half)
        x_target = apply_euclidean_alignment(x_target, r_inv_half)
        np.save(output_dir / "ea_matrix.npy", r_inv_half)

    train_loader = DataLoader(EEGDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(EEGDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        num_classes=int(max(y_source.max(), y_target.max()) + 1),
        num_channels=x_source.shape[1],
        checkpoint=args.checkpoint,
        linear_probe=args.linear_probe,
        device=device,
    )

    criterion_weight = None
    if not args.disable_class_weights:
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * np.maximum(class_counts, 1))
        criterion_weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=criterion_weight)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    patience_left = args.patience
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_result = run_epoch(model, val_loader, criterion, device, optimizer=None)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_result.loss,
                "train_acc": train_result.acc,
                "val_loss": val_result.loss,
                "val_acc": val_result.acc,
            }
        )
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_result.loss:.4f} train_acc={train_result.acc:.4f} | "
            f"val_loss={val_result.loss:.4f} val_acc={val_result.acc:.4f}"
        )

        if val_result.loss < best_val_loss:
            best_val_loss = val_result.loss
            best_epoch = epoch
            patience_left = args.patience
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(output_dir / "best_model.pth", map_location=device))
    validation_threshold = args.decision_threshold
    threshold_score = None
    if args.tune_threshold:
        val_subject_ids, val_subject_true, val_subject_prob = collect_subject_probabilities(
            model=model,
            x_data=x_val,
            y_data=y_val,
            sub_ids=sub_ids_source[val_mask],
            batch_size=args.batch_size,
            device=device,
        )
        validation_threshold, threshold_score = choose_threshold(
            y_true=val_subject_true,
            y_prob=val_subject_prob,
            objective=args.threshold_objective,
        )
        val_threshold_metrics = compute_subject_metrics(
            subject_ids_unique=val_subject_ids,
            y_true=val_subject_true,
            y_prob=val_subject_prob,
            threshold=validation_threshold,
        )
    else:
        val_threshold_metrics = None

    target_subject_ids, target_subject_true, target_subject_prob = collect_subject_probabilities(
        model=model,
        x_data=x_target,
        y_data=y_target,
        sub_ids=sub_ids_target,
        batch_size=args.batch_size,
        device=device,
    )
    target_metrics = compute_subject_metrics(
        subject_ids_unique=target_subject_ids,
        y_true=target_subject_true,
        y_prob=target_subject_prob,
        threshold=validation_threshold,
    )

    summary = {
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "checkpoint": args.checkpoint,
        "linear_probe": args.linear_probe,
        "use_ea": args.use_ea,
        "disable_class_weights": args.disable_class_weights,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "decision_threshold": validation_threshold,
        "threshold_objective": args.threshold_objective if args.tune_threshold else None,
        "threshold_objective_score": threshold_score,
        "validation_threshold_metrics": val_threshold_metrics,
        "history": history,
        "target_metrics": target_metrics,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print("\nCross-domain target-domain subject-level results")
    print(f"Accuracy   : {target_metrics['accuracy']:.4f}")
    print(f"Balanced Acc: {target_metrics['balanced_accuracy']:.4f}")
    print(f"F1         : {target_metrics['f1']:.4f}")
    print(f"AUC        : {target_metrics['auc']:.4f}")
    print(f"Sensitivity: {target_metrics['sensitivity']:.4f}")
    print(f"Specificity: {target_metrics['specificity']:.4f}")
    if validation_threshold is not None and not math.isnan(validation_threshold):
        print(f"Threshold  : {validation_threshold:.4f}")
    print(target_metrics["report"])
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
