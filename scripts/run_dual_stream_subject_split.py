"""Train/evaluate TAGE-Net with a subject-level train/val/test split.

This runner is designed for large datasets such as CAUEEG where LOSO would
require hundreds of re-training folds. Splits are performed by subject to avoid
segment leakage.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_dual_stream_loso_subject import (  # noqa: E402
    FocalLoss,
    SupConClassificationLoss,
    apply_euclidean_alignment,
    build_model,
    choose_threshold,
    fit_euclidean_alignment,
    fit_log_euclidean_alignment,
    resolve_domain_files,
    safe_auc,
    set_seed,
)
from utils.dual_stream_config import add_config_argument, apply_yaml_defaults  # noqa: E402
from utils.subject_split import load_subject_split_masks, save_subject_split  # noqa: E402
from models.GraphTemporalNet import GraphTemporalNet  # noqa: E402
from models.MSCNNClassifier import MSCNNClassifier  # noqa: E402
from models.TrainingStageFeatureFusion import (  # noqa: E402
    TrainingStageFeatureFusion,
    load_encoder_checkpoint,
)
from models.TFCBaseline import TFCClassifier  # noqa: E402


def load_domain_mmap(domain_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load processed EEG files with memory mapping for X."""

    x_path, y_path, id_path = resolve_domain_files(domain_dir)
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path).astype(np.int64, copy=False)
    sub_ids = np.load(id_path, allow_pickle=True)
    sub_ids = disambiguate_mixed_label_subject_ids(y, sub_ids)
    return x, y, sub_ids


def disambiguate_mixed_label_subject_ids(y: np.ndarray, sub_ids: np.ndarray) -> np.ndarray:
    """Make subject ids unique when a dataset reuses ids across labels."""

    label_sets: dict[str, set[int]] = {}
    for subject, label in zip(sub_ids, y):
        label_sets.setdefault(str(subject), set()).add(int(label))

    mixed_subjects = {subject for subject, labels in label_sets.items() if len(labels) > 1}
    if not mixed_subjects:
        return sub_ids

    fixed_ids = np.asarray(
        [
            f"label{int(label)}__{subject}" if str(subject) in mixed_subjects else str(subject)
            for subject, label in zip(sub_ids, y)
        ],
        dtype=object,
    )
    print(
        f"Warning: disambiguated {len(mixed_subjects)} subject ids reused across labels "
        "by prefixing the label."
    )
    return fixed_ids


def subject_labels(y: np.ndarray, sub_ids: np.ndarray) -> dict[object, int]:
    """Return one label per subject."""

    labels: dict[object, int] = {}
    for subject in np.unique(sub_ids):
        idx = np.where(sub_ids == subject)[0]
        labels[subject] = int(y[idx[0]])
    return labels


def stratified_subject_split(
    y: np.ndarray,
    sub_ids: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create stratified subject-level train/val/test masks."""

    if train_ratio <= 0.0 or val_ratio <= 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Expected train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1")

    rng = np.random.default_rng(seed)
    labels = subject_labels(y, sub_ids)
    train_subjects: list[object] = []
    val_subjects: list[object] = []
    test_subjects: list[object] = []

    for label in sorted(set(labels.values())):
        label_subjects = [subject for subject, value in labels.items() if value == label]
        rng.shuffle(label_subjects)
        n_total = len(label_subjects)
        n_train = max(1, int(round(n_total * train_ratio)))
        n_val = max(1, int(round(n_total * val_ratio)))
        if n_train + n_val >= n_total:
            n_val = max(1, n_total - n_train - 1)
        train_subjects.extend(label_subjects[:n_train])
        val_subjects.extend(label_subjects[n_train : n_train + n_val])
        test_subjects.extend(label_subjects[n_train + n_val :])

    train_mask = np.isin(sub_ids, np.asarray(train_subjects, dtype=object))
    val_mask = np.isin(sub_ids, np.asarray(val_subjects, dtype=object))
    test_mask = np.isin(sub_ids, np.asarray(test_subjects, dtype=object))
    return train_mask, val_mask, test_mask


def sample_indices_per_subject(
    indices: np.ndarray,
    sub_ids: np.ndarray,
    max_segments_per_subject: int | None,
    seed: int,
) -> np.ndarray:
    """Optionally cap the number of segments used for each subject."""

    if max_segments_per_subject is None or max_segments_per_subject <= 0:
        return indices

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for subject in np.unique(sub_ids[indices]):
        subject_indices = indices[sub_ids[indices] == subject]
        if len(subject_indices) > max_segments_per_subject:
            subject_indices = rng.choice(subject_indices, size=max_segments_per_subject, replace=False)
        selected.append(np.asarray(subject_indices, dtype=np.int64))
    return np.sort(np.concatenate(selected))


def subsample_training_subjects(
    indices: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Keep a deterministic stratified fraction of training subjects."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("train_subject_fraction must be in (0, 1].")
    if fraction >= 1.0:
        return np.asarray(indices, dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected_subjects: list[object] = []
    index_subjects = subject_ids[indices]
    index_labels = labels[indices]
    for label in np.unique(index_labels):
        class_subjects = np.unique(index_subjects[index_labels == label])
        keep = max(1, int(round(len(class_subjects) * fraction)))
        keep = min(keep, len(class_subjects))
        selected_subjects.extend(
            rng.choice(class_subjects, size=keep, replace=False).tolist()
        )
    mask = np.isin(subject_ids[indices], np.asarray(selected_subjects, dtype=object))
    return np.asarray(indices[mask], dtype=np.int64)


class IndexedEEGDataset(Dataset):
    """Memory-mapped EEG dataset indexed by selected segment ids.

    Each returned sample has shape [C, T].
    """

    def __init__(
        self,
        x_all: np.ndarray,
        y_all: np.ndarray,
        sub_ids_all: np.ndarray,
        indices: np.ndarray,
        ea_matrix: np.ndarray | None = None,
        augment: bool = False,
        noise_std: float = 0.0,
        max_shift: int = 0,
        scale_min: float = 1.0,
        scale_max: float = 1.0,
        channel_dropout_prob: float = 0.0,
        return_subject: bool = False,
    ) -> None:
        self.x_all = x_all
        self.y_all = y_all
        self.sub_ids_all = sub_ids_all
        self.indices = np.asarray(indices, dtype=np.int64)
        self.ea_matrix = ea_matrix
        self.augment = augment
        self.noise_std = noise_std
        self.max_shift = max_shift
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.channel_dropout_prob = channel_dropout_prob
        self.return_subject = return_subject

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        sample = np.asarray(self.x_all[idx], dtype=np.float32).copy()
        mean = sample.mean(axis=1, keepdims=True)
        std = sample.std(axis=1, keepdims=True) + 1e-6
        sample = (sample - mean) / std
        if self.ea_matrix is not None:
            sample = self.ea_matrix @ sample
        if self.augment:
            sample = self._augment(sample)

        label = int(self.y_all[idx])
        if self.return_subject:
            return torch.from_numpy(sample.astype(np.float32, copy=False)), torch.tensor(label), str(self.sub_ids_all[idx])
        return torch.from_numpy(sample.astype(np.float32, copy=False)), torch.tensor(label)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Apply lightweight augmentations on normalized EEG."""

        if self.noise_std > 0.0:
            x = x + np.random.normal(0.0, self.noise_std, x.shape).astype(np.float32)
        if self.max_shift > 0:
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            if shift != 0:
                x = np.roll(x, shift, axis=-1)
        if self.scale_min < self.scale_max:
            x = x * np.float32(np.random.uniform(self.scale_min, self.scale_max))
        if self.channel_dropout_prob > 0.0:
            channel_mask = np.random.rand(x.shape[0]) < self.channel_dropout_prob
            if channel_mask.any() and channel_mask.sum() < x.shape[0]:
                x[channel_mask] = 0.0
        return x.astype(np.float32, copy=False)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    l2sp_anchor: dict[str, torch.Tensor] | None = None,
    l2sp_weight: float = 0.0,
) -> tuple[float, float]:
    """Run one train/eval epoch and return loss/accuracy."""

    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    preds: list[int] = []
    targets: list[int] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        with torch.set_grad_enabled(is_train):
            logits, features = model(batch_x)
            if isinstance(criterion, SupConClassificationLoss):
                loss = criterion(logits, batch_y, features)
            else:
                loss = criterion(logits, batch_y)
            if is_train and l2sp_anchor and l2sp_weight > 0.0:
                loss = loss + l2sp_weight * l2sp_penalty(model, l2sp_anchor)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(float(loss.item()))
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        targets.extend(batch_y.detach().cpu().numpy().tolist())

    acc = float(np.mean(np.asarray(preds) == np.asarray(targets))) if targets else 0.0
    return float(np.mean(losses)) if losses else 0.0, acc


def l2sp_penalty(
    model: nn.Module,
    anchor: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return the summed squared distance from selected initial parameters."""

    named_parameters = dict(model.named_parameters())
    missing = sorted(set(anchor) - set(named_parameters))
    if missing:
        raise ValueError(f"L2-SP anchor contains unknown parameters: {missing[:3]}")
    return torch.stack(
        [torch.sum((named_parameters[name] - reference) ** 2) for name, reference in anchor.items()]
    ).sum()


def capture_l2sp_anchor(
    model: nn.Module,
    scope: str,
) -> dict[str, torch.Tensor]:
    """Clone the initialized parameters used as the L2-SP reference point."""

    encoder_prefixes = (
        "spatial_expert.",
        "femba.",
        "neuro_prior.",
        "prior_projector.",
        "sampling_rate_embedding.",
    )
    selected = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if scope == "all" or name.startswith(encoder_prefixes)
    }
    if not selected:
        raise ValueError(f"No parameters selected for L2-SP scope={scope}.")
    return selected


def collect_subject_probabilities(
    model: nn.Module,
    dataset: IndexedEEGDataset,
    batch_size: int,
    device: torch.device,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate segment probabilities into subject-level probabilities."""

    dataset.return_subject = True
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    subject_probs: dict[str, list[np.ndarray]] = {}
    subject_labels_map: dict[str, int] = {}

    with torch.no_grad():
        for batch_x, batch_y, batch_subjects in loader:
            logits, _ = model(batch_x.to(device))
            probs = F.softmax(logits / temperature, dim=1).cpu().numpy()
            labels = batch_y.numpy().tolist()
            for subject, label, prob in zip(batch_subjects, labels, probs):
                subject = str(subject)
                subject_probs.setdefault(subject, []).append(prob)
                subject_labels_map[subject] = int(label)

    subjects = np.asarray(sorted(subject_probs.keys()), dtype=object)
    y_true = np.asarray([subject_labels_map[str(subject)] for subject in subjects], dtype=np.int64)
    y_prob = np.asarray([np.mean(subject_probs[str(subject)], axis=0) for subject in subjects], dtype=np.float32)
    dataset.return_subject = False
    return subjects, y_true, y_prob


def collect_subject_gate_statistics(
    model: TrainingStageFeatureFusion,
    dataset: IndexedEEGDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    """Summarize learned pretraining weights overall and for each subject."""

    dataset.return_subject = True
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    by_subject: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch_x, _, batch_subjects in loader:
            _, _, alpha = model.forward_with_gate(batch_x.to(device))
            for subject, value in zip(batch_subjects, alpha.squeeze(-1).cpu().numpy().tolist()):
                by_subject.setdefault(str(subject), []).append(float(value))
    dataset.return_subject = False
    subject_means = {subject: float(np.mean(values)) for subject, values in sorted(by_subject.items())}
    finite_values = np.asarray(list(subject_means.values()), dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    return {
        "meaning": "alpha is the normalized pretraining-feature weight",
        "subject_mean": subject_means,
        "mean": float(np.mean(finite_values)) if finite_values.size else None,
        "std": float(np.std(finite_values)) if finite_values.size else None,
        "min": float(np.min(finite_values)) if finite_values.size else None,
        "max": float(np.max(finite_values)) if finite_values.size else None,
    }


def compute_metrics(subject_ids: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, threshold: float | None) -> dict[str, object]:
    """Compute subject-level binary classification metrics."""

    if threshold is not None and y_prob.shape[1] == 2:
        y_pred = (y_prob[:, 1] >= threshold).astype(np.int64)
    else:
        y_pred = np.argmax(y_prob, axis=1).astype(np.int64)

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    auc = roc_auc_score(y_true, y_prob[:, 1]) if len(np.unique(y_true)) == 2 else float("nan")

    return {
        "subject_ids": subject_ids.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "report": classification_report(y_true, y_pred, digits=4, zero_division=0),
    }


def count_subjects(y: np.ndarray, sub_ids: np.ndarray, indices: Iterable[int]) -> dict[str, int]:
    """Count subjects per class in a split."""

    indices = np.asarray(list(indices), dtype=np.int64)
    labels = subject_labels(y[indices], sub_ids[indices])
    return {str(label): int(sum(value == label for value in labels.values())) for label in sorted(set(labels.values()))}


def require_two_classes_in_split(split_name: str, y: np.ndarray, sub_ids: np.ndarray, indices: np.ndarray) -> None:
    """Fail fast if a subject-level split contains only one class."""

    counts = count_subjects(y, sub_ids, indices)
    if len(counts) < 2:
        raise ValueError(
            f"{split_name} split contains fewer than two subject classes: {counts}. "
            "Use a stratified subject-level split and do not reuse an invalid split file."
        )


def resolve_checkpoint_argument(checkpoint: str | None, no_checkpoint: bool) -> str | None:
    """Resolve checkpoint CLI flags without silently discarding a supplied path."""

    if checkpoint and no_checkpoint:
        raise ValueError("--checkpoint and --no-checkpoint are mutually exclusive.")
    return None if no_checkpoint else checkpoint


def select_compatible_checkpoint_tensors(
    checkpoint_payload: object,
    model_state: dict[str, torch.Tensor],
    allowed_prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Select checkpoint tensors matching the requested scope and tensor shape."""

    if isinstance(checkpoint_payload, dict) and "encoder_state_dict" in checkpoint_payload:
        checkpoint_payload = checkpoint_payload["encoder_state_dict"]
    if not isinstance(checkpoint_payload, dict):
        raise ValueError("Checkpoint must contain a state dictionary.")
    return {
        key: value
        for key, value in checkpoint_payload.items()
        if isinstance(key, str)
        and isinstance(value, torch.Tensor)
        and key.startswith(allowed_prefixes)
        and key in model_state
        and model_state[key].shape == value.shape
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Subject-level train/val/test runner for large EEG datasets")
    add_config_argument(parser)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--checkpoint-scope",
        choices=["all", "encoders", "graph", "temporal"],
        default="all",
        help="Limit which GraphTemporalNet tensors are initialized from a checkpoint.",
    )
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=None,
        help="Optional encoder learning rate; the classifier keeps --lr.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--save-split-file", default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--use-ea", action="store_true")
    parser.add_argument(
        "--alignment",
        choices=["none", "euclidean", "log_euclidean"],
        default=None,
        help="Training-only covariance alignment. --use-ea remains an alias for euclidean.",
    )
    parser.add_argument("--max-ea-samples", type=int, default=5000)
    parser.add_argument("--max-train-segments-per-subject", type=int, default=None)
    parser.add_argument("--max-val-segments-per-subject", type=int, default=None)
    parser.add_argument("--max-test-segments-per-subject", type=int, default=None)
    parser.add_argument("--train-subject-fraction", type=float, default=1.0)
    parser.add_argument("--linear-probe", action="store_true")
    parser.add_argument(
        "--l2sp-weight",
        type=float,
        default=0.0,
        help="Weight of the L2-SP penalty around the initialized parameters.",
    )
    parser.add_argument(
        "--l2sp-scope",
        choices=["encoders", "all"],
        default="encoders",
        help="Parameters constrained by L2-SP.",
    )
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--loss-type", choices=["ce", "focal", "supcon"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--supcon-weight", type=float, default=0.1)
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--threshold-objective", choices=["accuracy", "balanced_acc", "f1", "precision", "specificity", "youden"], default="balanced_acc")
    parser.add_argument("--selection-metric", choices=["loss", "auc"], default="auc")
    parser.add_argument(
        "--model-type",
        choices=["dual_stream", "graph_temporal", "mscnn_only", "stage_fusion", "tfc"],
        default="dual_stream",
    )
    parser.add_argument("--fusion-base-model", choices=["dual_stream", "graph_temporal"], default="dual_stream")
    parser.add_argument("--pretrained-checkpoint", default=None)
    parser.add_argument("--source-checkpoint", default=None)
    parser.add_argument(
        "--fusion-mode",
        choices=list(TrainingStageFeatureFusion.MODES),
        default="learned_scalar",
    )
    parser.add_argument("--gate-hidden-dim", type=int, default=64)
    parser.add_argument("--unfreeze-fusion-encoders", action="store_true")
    parser.add_argument("--graph-num-neighbors", type=int, default=4)
    parser.add_argument("--spatial-out-dim", type=int, default=256)
    parser.add_argument("--temporal-out-dim", type=int, default=256)
    parser.add_argument("--spatial-hidden-channels", type=int, default=64)
    parser.add_argument("--spatial-temporal-channels", type=int, default=128)
    parser.add_argument("--classifier-hidden-dim", type=int, default=None)
    parser.add_argument("--classifier-num-layers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--classifier-dropout", type=float, default=0.6)
    parser.add_argument("--use-neuro-prior", action="store_true")
    parser.add_argument("--sampling-rate", type=float, default=128.0)
    parser.add_argument("--use-sampling-rate-embedding", action="store_true")
    parser.add_argument("--sampling-rate-options", nargs="+", type=float, default=[128.0, 256.0])
    parser.add_argument("--prior-out-dim", type=int, default=64)
    parser.add_argument("--augment-train", action="store_true")
    parser.add_argument("--augment-noise-std", type=float, default=0.0)
    parser.add_argument("--augment-max-shift", type=int, default=0)
    parser.add_argument("--augment-scale-min", type=float, default=1.0)
    parser.add_argument("--augment-scale-max", type=float, default=1.0)
    parser.add_argument("--augment-channel-dropout", type=float, default=0.0)
    parser.add_argument("--tfc-model-dim", type=int, default=128)
    parser.add_argument("--tfc-projection-dim", type=int, default=128)
    parser.add_argument("--tfc-num-layers", type=int, default=2)
    parser.add_argument("--tfc-num-heads", type=int, default=4)
    args = parser.parse_args()
    args = apply_yaml_defaults(
        parser,
        args,
        [
            "checkpoint",
            "checkpoint_scope",
            "epochs",
            "batch_size",
            "lr",
            "encoder_lr",
            "weight_decay",
            "patience",
            "seed",
            "use_ea",
            "alignment",
            "linear_probe",
            "l2sp_weight",
            "l2sp_scope",
            "disable_class_weights",
            "loss_type",
            "focal_gamma",
            "supcon_weight",
            "supcon_temperature",
            "tune_threshold",
            "threshold_objective",
            "selection_metric",
            "model_type",
            "fusion_base_model",
            "pretrained_checkpoint",
            "source_checkpoint",
            "fusion_mode",
            "gate_hidden_dim",
            "unfreeze_fusion_encoders",
            "graph_num_neighbors",
            "spatial_out_dim",
            "temporal_out_dim",
            "spatial_hidden_channels",
            "spatial_temporal_channels",
            "classifier_hidden_dim",
            "classifier_num_layers",
            "classifier_dropout",
            "use_neuro_prior",
            "sampling_rate",
            "use_sampling_rate_embedding",
            "sampling_rate_options",
            "prior_out_dim",
            "augment_train",
            "augment_noise_std",
            "augment_max_shift",
            "augment_scale_min",
            "augment_scale_max",
            "augment_channel_dropout",
            "tfc_model_dim",
            "tfc_projection_dim",
            "tfc_num_layers",
            "tfc_num_heads",
        ],
    )
    args.checkpoint = resolve_checkpoint_argument(args.checkpoint, args.no_checkpoint)
    args.alignment = args.alignment or ("euclidean" if args.use_ea else "none")
    if args.l2sp_weight < 0.0:
        raise ValueError("--l2sp-weight must be non-negative.")
    if args.l2sp_weight > 0.0 and not args.checkpoint:
        raise ValueError("L2-SP requires an initialization checkpoint.")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_all, y_all, sub_ids_all = load_domain_mmap(Path(args.data_dir))
    if args.split_file and Path(args.split_file).exists():
        train_mask, val_mask, test_mask = load_subject_split_masks(args.split_file, sub_ids_all)
    else:
        train_mask, val_mask, test_mask = stratified_subject_split(y_all, sub_ids_all, args.train_ratio, args.val_ratio, args.seed)
    if args.save_split_file:
        save_subject_split(
            args.save_split_file,
            sub_ids_all,
            train_mask,
            val_mask,
            test_mask,
            metadata={
                "data_dir": args.data_dir,
                "seed": args.seed,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
            },
        )
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]
    train_indices = subsample_training_subjects(
        train_indices,
        y_all,
        sub_ids_all,
        args.train_subject_fraction,
        args.seed + 7,
    )
    train_indices = sample_indices_per_subject(train_indices, sub_ids_all, args.max_train_segments_per_subject, args.seed + 11)
    val_indices = sample_indices_per_subject(val_indices, sub_ids_all, args.max_val_segments_per_subject, args.seed + 17)
    test_indices = sample_indices_per_subject(test_indices, sub_ids_all, args.max_test_segments_per_subject, args.seed + 23)
    require_two_classes_in_split("train", y_all, sub_ids_all, train_indices)
    require_two_classes_in_split("val", y_all, sub_ids_all, val_indices)
    require_two_classes_in_split("test", y_all, sub_ids_all, test_indices)

    ea_matrix = None
    if args.alignment != "none":
        ea_indices = train_indices
        if len(ea_indices) > args.max_ea_samples:
            rng = np.random.default_rng(args.seed + 31)
            ea_indices = rng.choice(ea_indices, size=args.max_ea_samples, replace=False)
        ea_samples = np.asarray(x_all[ea_indices], dtype=np.float32)
        if args.alignment == "log_euclidean":
            ea_matrix = fit_log_euclidean_alignment(ea_samples)
        else:
            ea_matrix = fit_euclidean_alignment(ea_samples)

    train_dataset = IndexedEEGDataset(
        x_all,
        y_all,
        sub_ids_all,
        train_indices,
        ea_matrix=ea_matrix,
        augment=args.augment_train,
        noise_std=args.augment_noise_std,
        max_shift=args.augment_max_shift,
        scale_min=args.augment_scale_min,
        scale_max=args.augment_scale_max,
        channel_dropout_prob=args.augment_channel_dropout,
    )
    val_dataset = IndexedEEGDataset(x_all, y_all, sub_ids_all, val_indices, ea_matrix=ea_matrix)
    test_dataset = IndexedEEGDataset(x_all, y_all, sub_ids_all, test_indices, ea_matrix=ea_matrix)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == "stage_fusion":
        if args.checkpoint or args.no_checkpoint:
            raise ValueError("stage_fusion uses --pretrained-checkpoint and --source-checkpoint.")
        if not args.pretrained_checkpoint or not args.source_checkpoint:
            raise ValueError("stage_fusion requires both encoder checkpoints.")
        if args.linear_probe:
            raise ValueError("Use the fusion freeze flag instead of --linear-probe for stage_fusion.")

        def make_fusion_encoder() -> nn.Module:
            if args.fusion_base_model == "graph_temporal":
                return GraphTemporalNet(
                    num_classes=int(y_all.max() + 1),
                    num_channels=int(x_all.shape[1]),
                    graph_out_dim=args.spatial_out_dim,
                    temporal_out_dim=args.temporal_out_dim,
                    classifier_dropout=args.classifier_dropout,
                    num_neighbors=args.graph_num_neighbors,
                )
            return build_model(
                num_classes=int(y_all.max() + 1),
                num_channels=int(x_all.shape[1]),
                checkpoint=None,
                linear_probe=False,
                device=device,
                spatial_out_dim=args.spatial_out_dim,
                temporal_out_dim=args.temporal_out_dim,
                spatial_hidden_channels=args.spatial_hidden_channels,
                spatial_temporal_channels=args.spatial_temporal_channels,
                classifier_hidden_dim=args.classifier_hidden_dim,
                classifier_num_layers=args.classifier_num_layers,
                classifier_dropout=args.classifier_dropout,
                use_neuro_prior=args.use_neuro_prior,
                sampling_rate=args.sampling_rate,
                prior_out_dim=args.prior_out_dim,
                use_sampling_rate_embedding=args.use_sampling_rate_embedding,
                sampling_rate_options=tuple(args.sampling_rate_options),
            )

        pretrained_encoder = make_fusion_encoder()
        source_encoder = make_fusion_encoder()
        if args.fusion_base_model == "graph_temporal":
            encoder_prefixes = ("graph_encoder.", "temporal_encoder.")
        else:
            encoder_prefixes = ("spatial_expert.", "femba.", "neuro_prior.", "prior_projector.")
        pre_count = load_encoder_checkpoint(
            pretrained_encoder, args.pretrained_checkpoint, encoder_prefixes, map_location=device
        )
        source_count = load_encoder_checkpoint(
            source_encoder, args.source_checkpoint, encoder_prefixes, map_location=device
        )
        feature_dim = args.spatial_out_dim + args.temporal_out_dim
        if args.use_neuro_prior:
            feature_dim += args.prior_out_dim
        model = TrainingStageFeatureFusion(
            pretrained_encoder=pretrained_encoder,
            source_encoder=source_encoder,
            feature_dim=feature_dim,
            num_classes=int(y_all.max() + 1),
            mode=args.fusion_mode,
            gate_hidden_dim=args.gate_hidden_dim,
            classifier_dropout=args.classifier_dropout,
            freeze_encoders=not args.unfreeze_fusion_encoders,
        ).to(device)
        print(
            f"Loaded stage-fusion encoders: pretrained={pre_count} tensors, "
            f"source={source_count} tensors, base={args.fusion_base_model}, mode={args.fusion_mode}"
        )
    elif args.model_type == "tfc":
        if args.checkpoint_scope != "all":
            raise ValueError("TF-C loads its complete encoder; use --checkpoint-scope all.")
        model = TFCClassifier(
            num_classes=int(y_all.max() + 1),
            num_channels=int(x_all.shape[1]),
            seq_len=int(x_all.shape[2]),
            model_dim=args.tfc_model_dim,
            projection_dim=args.tfc_projection_dim,
            num_layers=args.tfc_num_layers,
            num_heads=args.tfc_num_heads,
            dropout=args.classifier_dropout,
        ).to(device)
        if args.checkpoint:
            loaded = model.load_pretrained_encoder(args.checkpoint, map_location=device)
            print(f"Loaded {loaded} TF-C encoder tensors from {args.checkpoint}")
        if args.linear_probe:
            if not args.checkpoint:
                raise ValueError("TF-C linear probing requires a pretraining checkpoint.")
            for parameter in model.encoder.parameters():
                parameter.requires_grad = False
    elif args.model_type == "graph_temporal":
        model = GraphTemporalNet(
            num_classes=int(y_all.max() + 1),
            num_channels=int(x_all.shape[1]),
            graph_out_dim=args.spatial_out_dim,
            temporal_out_dim=args.temporal_out_dim,
            classifier_dropout=args.classifier_dropout,
            num_neighbors=args.graph_num_neighbors,
        ).to(device)
        if args.checkpoint:
            checkpoint_payload = torch.load(args.checkpoint, map_location=device)
            if isinstance(checkpoint_payload, dict) and "encoder_state_dict" in checkpoint_payload:
                checkpoint_state = checkpoint_payload["encoder_state_dict"]
            else:
                checkpoint_state = checkpoint_payload
            model_state = model.state_dict()
            allowed_prefixes = {
                "all": ("",),
                "encoders": ("graph_encoder.", "temporal_encoder."),
                "graph": ("graph_encoder.",),
                "temporal": ("temporal_encoder.",),
            }[args.checkpoint_scope]
            compatible_state = {
                key: value
                for key, value in checkpoint_state.items()
                if key.startswith(allowed_prefixes)
                and key in model_state
                and model_state[key].shape == value.shape
            }
            if not compatible_state:
                raise ValueError(f"No compatible GraphTemporalNet weights found in {args.checkpoint}.")
            model.load_state_dict(compatible_state, strict=False)
            print(
                f"Loaded {len(compatible_state)} GraphTemporalNet tensors "
                f"(scope={args.checkpoint_scope}) from {args.checkpoint}"
            )
        if args.linear_probe:
            for parameter in model.parameters():
                parameter.requires_grad = False
            for parameter in model.fusion_classifier.parameters():
                parameter.requires_grad = True

    elif args.model_type == "mscnn_only":
        model = MSCNNClassifier(
            num_classes=int(y_all.max() + 1),
            num_channels=int(x_all.shape[1]),
            temporal_out_dim=args.temporal_out_dim,
            classifier_dropout=args.classifier_dropout,
        ).to(device)
        if args.checkpoint:
            if args.checkpoint_scope == "graph":
                raise ValueError("MSCNNClassifier has no graph encoder to load.")
            checkpoint_payload = torch.load(args.checkpoint, map_location=device)
            allowed_prefixes = {
                "all": ("",),
                "encoders": ("temporal_encoder.",),
                "temporal": ("temporal_encoder.",),
            }[args.checkpoint_scope]
            compatible_state = select_compatible_checkpoint_tensors(
                checkpoint_payload,
                model.state_dict(),
                allowed_prefixes,
            )
            if not compatible_state:
                raise ValueError(
                    f"No compatible MSCNNClassifier weights found in {args.checkpoint}."
                )
            model.load_state_dict(compatible_state, strict=False)
            print(
                f"Loaded {len(compatible_state)} MSCNNClassifier tensors "
                f"(scope={args.checkpoint_scope}) from {args.checkpoint}"
            )
        if args.linear_probe:
            if not args.checkpoint:
                raise ValueError(
                    "Linear probing requires an MSCNNClassifier pretraining checkpoint."
                )
            for parameter in model.parameters():
                parameter.requires_grad = False
            for parameter in model.fusion_classifier.parameters():
                parameter.requires_grad = True
    else:
        model = build_model(
            num_classes=int(y_all.max() + 1),
            num_channels=int(x_all.shape[1]),
            checkpoint=None,
            linear_probe=False,
            device=device,
            spatial_out_dim=args.spatial_out_dim,
            temporal_out_dim=args.temporal_out_dim,
            spatial_hidden_channels=args.spatial_hidden_channels,
            spatial_temporal_channels=args.spatial_temporal_channels,
            classifier_hidden_dim=args.classifier_hidden_dim,
            classifier_num_layers=args.classifier_num_layers,
            classifier_dropout=args.classifier_dropout,
            use_neuro_prior=args.use_neuro_prior,
            sampling_rate=args.sampling_rate,
            prior_out_dim=args.prior_out_dim,
            use_sampling_rate_embedding=args.use_sampling_rate_embedding,
            sampling_rate_options=tuple(args.sampling_rate_options),
        )
        if args.checkpoint:
            allowed_prefixes = {
                "all": ("",),
                "encoders": (
                    "spatial_expert.",
                    "femba.",
                    "neuro_prior.",
                    "prior_projector.",
                    "sampling_rate_embedding.",
                ),
                "graph": ("spatial_expert.", "neuro_prior.", "prior_projector."),
                "temporal": ("femba.",),
            }[args.checkpoint_scope]
            checkpoint_payload = torch.load(args.checkpoint, map_location=device)
            compatible_state = select_compatible_checkpoint_tensors(
                checkpoint_payload,
                model.state_dict(),
                allowed_prefixes,
            )
            if not compatible_state:
                raise ValueError(
                    f"No compatible dual-stream weights found in {args.checkpoint}."
                )
            model.load_state_dict(compatible_state, strict=False)
            print(
                f"Loaded {len(compatible_state)} dual-stream tensors "
                f"(scope={args.checkpoint_scope}) from {args.checkpoint}"
            )
        if args.linear_probe:
            if not args.checkpoint:
                raise ValueError("Dual-stream linear probing requires a checkpoint.")
            for parameter in model.parameters():
                parameter.requires_grad = False
            for parameter in model.fusion_classifier.parameters():
                parameter.requires_grad = True

    l2sp_anchor = (
        capture_l2sp_anchor(model, args.l2sp_scope)
        if args.l2sp_weight > 0.0
        else None
    )

    class_weight = None
    if not args.disable_class_weights:
        class_counts = np.bincount(y_all[train_indices], minlength=int(y_all.max() + 1))
        class_weights = len(train_indices) / (len(class_counts) * np.maximum(class_counts, 1))
        class_weight = torch.tensor(class_weights, dtype=torch.float32, device=device)

    if args.loss_type == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weight)
    elif args.loss_type == "supcon":
        criterion = SupConClassificationLoss(args.supcon_weight, args.supcon_temperature, weight=class_weight)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight)

    if args.encoder_lr is None:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        classifier = getattr(model, "fusion_classifier", None)
        if classifier is None:
            classifier = getattr(model, "classifier", None)
        if classifier is None:
            raise ValueError("--encoder-lr requires a model with a classifier module.")
        classifier_ids = {id(parameter) for parameter in classifier.parameters()}
        encoder_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in classifier_ids
        ]
        classifier_parameters = [
            parameter for parameter in classifier.parameters() if parameter.requires_grad
        ]
        parameter_groups: list[dict[str, object]] = []
        if encoder_parameters:
            parameter_groups.append({"params": encoder_parameters, "lr": args.encoder_lr})
        if classifier_parameters:
            parameter_groups.append({"params": classifier_parameters, "lr": args.lr})
        if not parameter_groups:
            raise ValueError("No trainable parameters remain for optimization.")
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    best_model_path = output_dir / "best_model.pth"
    best_val_loss = float("inf")
    best_val_auc = -float("inf")
    best_epoch = 0
    patience_left = args.patience
    history: list[dict[str, float | int | None]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            l2sp_anchor=l2sp_anchor,
            l2sp_weight=args.l2sp_weight,
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        _, val_true_epoch, val_prob_epoch = collect_subject_probabilities(model, val_dataset, args.batch_size, device)
        val_auc = safe_auc(val_true_epoch, val_prob_epoch)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
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

        improved = False
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

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    threshold = None
    if args.tune_threshold:
        val_subjects, val_true, val_prob = collect_subject_probabilities(model, val_dataset, args.batch_size, device)
        threshold, threshold_score = choose_threshold(val_true, val_prob, args.threshold_objective)
    else:
        threshold_score = None

    test_subjects, test_true, test_prob = collect_subject_probabilities(model, test_dataset, args.batch_size, device)
    metrics = compute_metrics(test_subjects, test_true, test_prob, threshold)
    gate_statistics = None
    if isinstance(model, TrainingStageFeatureFusion):
        gate_statistics = collect_subject_gate_statistics(model, test_dataset, args.batch_size, device)

    split_summary = {
        "train_segments": int(len(train_indices)),
        "val_segments": int(len(val_indices)),
        "test_segments": int(len(test_indices)),
        "train_subjects_by_label": count_subjects(y_all, sub_ids_all, train_indices),
        "val_subjects_by_label": count_subjects(y_all, sub_ids_all, val_indices),
        "test_subjects_by_label": count_subjects(y_all, sub_ids_all, test_indices),
    }
    summary = {
        "data_dir": args.data_dir,
        "checkpoint": args.checkpoint,
        "checkpoint_scope": args.checkpoint_scope,
        "seed": args.seed,
        "split_file": args.split_file,
        "save_split_file": args.save_split_file,
        "split": split_summary,
        "use_ea": args.alignment != "none",
        "alignment": args.alignment,
        "max_ea_samples": args.max_ea_samples if args.alignment != "none" else None,
        "max_train_segments_per_subject": args.max_train_segments_per_subject,
        "max_val_segments_per_subject": args.max_val_segments_per_subject,
        "max_test_segments_per_subject": args.max_test_segments_per_subject,
        "train_subject_fraction": args.train_subject_fraction,
        "encoder_lr": args.encoder_lr,
        "linear_probe": args.linear_probe,
        "l2sp_weight": args.l2sp_weight,
        "l2sp_scope": args.l2sp_scope if args.l2sp_weight > 0.0 else None,
        "disable_class_weights": args.disable_class_weights,
        "loss_type": args.loss_type,
        "tune_threshold": args.tune_threshold,
        "threshold_objective": args.threshold_objective if args.tune_threshold else None,
        "decision_threshold": threshold,
        "threshold_score": threshold_score,
        "selection_metric": args.selection_metric,
        "model_type": args.model_type,
        "pretrained_checkpoint": args.pretrained_checkpoint if args.model_type == "stage_fusion" else None,
        "source_checkpoint": args.source_checkpoint if args.model_type == "stage_fusion" else None,
        "fusion_base_model": args.fusion_base_model if args.model_type == "stage_fusion" else None,
        "fusion_mode": args.fusion_mode if args.model_type == "stage_fusion" else None,
        "gate_hidden_dim": args.gate_hidden_dim if args.model_type == "stage_fusion" else None,
        "fusion_encoders_frozen": (not args.unfreeze_fusion_encoders) if args.model_type == "stage_fusion" else None,
        "graph_num_neighbors": args.graph_num_neighbors if (
            args.model_type == "graph_temporal"
            or (args.model_type == "stage_fusion" and args.fusion_base_model == "graph_temporal")
        ) else None,
        "classifier_num_layers": args.classifier_num_layers,
        "classifier_hidden_dim": args.classifier_hidden_dim,
        "classifier_dropout": args.classifier_dropout,
        "sampling_rate": args.sampling_rate,
        "use_sampling_rate_embedding": args.use_sampling_rate_embedding,
        "sampling_rate_options": args.sampling_rate_options,
        "tfc_model_dim": args.tfc_model_dim if args.model_type == "tfc" else None,
        "tfc_projection_dim": args.tfc_projection_dim if args.model_type == "tfc" else None,
        "tfc_num_layers": args.tfc_num_layers if args.model_type == "tfc" else None,
        "tfc_num_heads": args.tfc_num_heads if args.model_type == "tfc" else None,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_auc": None if np.isnan(best_val_auc) else float(best_val_auc),
        "history": history,
        "metrics": metrics,
        "gate_statistics": gate_statistics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "subject_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "y_true", "y_pred", "prob_class_0", "prob_class_1"])
        for subject, truth, prediction, probabilities in zip(
            metrics["subject_ids"], metrics["y_true"], metrics["y_pred"], metrics["y_prob"]
        ):
            writer.writerow([subject, truth, prediction, *probabilities])
    if gate_statistics is not None:
        (output_dir / "gate_statistics.json").write_text(
            json.dumps(gate_statistics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("\nSubject-level test summary")
    print(f"Accuracy     : {metrics['accuracy']:.4f}")
    print(f"Balanced Acc : {metrics['balanced_accuracy']:.4f}")
    print(f"F1           : {metrics['f1']:.4f}")
    print(f"AUC          : {metrics['auc']:.4f}")
    print(f"Sensitivity  : {metrics['sensitivity']:.4f}")
    print(f"Specificity  : {metrics['specificity']:.4f}")
    print(metrics["report"])
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
