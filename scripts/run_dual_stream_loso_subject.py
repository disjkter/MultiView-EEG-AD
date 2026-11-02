"""Run subject-level leave-one-subject-out evaluation on a single dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import fractional_matrix_power
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.Spatial_MSCNN import SpatialMSCNN_DualStream
from utils.dual_stream_config import add_config_argument, apply_yaml_defaults


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_domain_files(domain_dir: Path) -> tuple[Path, Path, Path]:
    """Resolve processed EEG npy files."""

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
    """Load one processed domain."""

    x_path, y_path, id_path = resolve_domain_files(domain_dir)
    x = np.load(x_path, mmap_mode="r")
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
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


def fit_euclidean_alignment(x_train: np.ndarray) -> np.ndarray:
    """Fit Euclidean Alignment from channel-standardized training samples only."""

    x_train = np.asarray(x_train, dtype=np.float64)
    means = x_train.mean(axis=-1, keepdims=True)
    stds = x_train.std(axis=-1, keepdims=True) + 1e-6
    x_normalized = (x_train - means) / stds
    n_trials, n_channels, _ = x_normalized.shape
    covs = np.zeros((n_trials, n_channels, n_channels), dtype=np.float64)
    for i in range(n_trials):
        covs[i] = np.cov(x_normalized[i])
    mean_cov = np.mean(covs, axis=0)
    mean_cov += np.eye(n_channels, dtype=np.float64) * 1e-6
    r_inv_half = fractional_matrix_power(mean_cov, -0.5)
    return np.real(r_inv_half).astype(np.float32)


def fit_log_euclidean_alignment(x_train: np.ndarray) -> np.ndarray:
    """Fit log-Euclidean covariance alignment using training samples only.

    The reference covariance is the exponential of the arithmetic mean of
    trial covariance matrix logarithms. Eigenvalues are clipped before the
    logarithm so the operation remains stable for short EEG windows.
    """

    x_train = np.asarray(x_train, dtype=np.float64)
    means = x_train.mean(axis=-1, keepdims=True)
    stds = x_train.std(axis=-1, keepdims=True) + 1e-6
    x_normalized = (x_train - means) / stds
    n_trials, n_channels, _ = x_normalized.shape
    mean_log_cov = np.zeros((n_channels, n_channels), dtype=np.float64)
    identity = np.eye(n_channels, dtype=np.float64)
    for trial in x_normalized:
        covariance = np.cov(trial) + 1e-6 * identity
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        mean_log_cov += (eigenvectors * np.log(eigenvalues)) @ eigenvectors.T
    mean_log_cov /= n_trials
    log_values, log_vectors = np.linalg.eigh(mean_log_cov)
    reference_covariance = (log_vectors * np.exp(log_values)) @ log_vectors.T
    reference_covariance += 1e-6 * identity
    r_inv_half = fractional_matrix_power(reference_covariance, -0.5)
    return np.real(r_inv_half).astype(np.float32)


def apply_euclidean_alignment(x: np.ndarray, r_inv_half: np.ndarray) -> np.ndarray:
    """Apply Euclidean Alignment to samples."""

    aligned = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        aligned[i] = r_inv_half @ x[i]
    return aligned


def split_subjects_for_validation(
    y: np.ndarray,
    sub_ids: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a subject-level train/val split from the remaining subjects."""

    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(sub_ids)
    subject_labels = {
        subject: int(y[np.where(sub_ids == subject)[0][0]]) for subject in unique_subjects
    }

    train_subjects: list[object] = []
    val_subjects: list[object] = []
    for label in sorted(set(subject_labels.values())):
        label_subjects = [s for s in unique_subjects if subject_labels[s] == label]
        rng.shuffle(label_subjects)
        n_val = max(1, int(round(len(label_subjects) * val_ratio)))
        if n_val >= len(label_subjects) and len(label_subjects) > 1:
            n_val = len(label_subjects) - 1
        val_subjects.extend(label_subjects[:n_val])
        train_subjects.extend(label_subjects[n_val:])

    train_mask = np.isin(sub_ids, np.asarray(train_subjects, dtype=object))
    val_mask = np.isin(sub_ids, np.asarray(val_subjects, dtype=object))
    return train_mask, val_mask


class EEGDataset(Dataset):
    """Per-sample normalized EEG dataset with optional lightweight augmentation."""

    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        augment: bool = False,
        noise_std: float = 0.0,
        max_shift: int = 0,
        scale_min: float = 1.0,
        scale_max: float = 1.0,
        channel_dropout_prob: float = 0.0,
    ):
        self.x = x_data.astype(np.float32, copy=False)
        self.y = y_data.astype(np.int64, copy=False)
        self.augment = augment
        self.noise_std = noise_std
        self.max_shift = max_shift
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.channel_dropout_prob = channel_dropout_prob
        for i in range(len(self.x)):
            mean = self.x[i].mean(axis=1, keepdims=True)
            std = self.x[i].std(axis=1, keepdims=True) + 1e-6
            self.x[i] = (self.x[i] - mean) / std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.x[idx].copy()
        if self.augment:
            sample = self._augment(sample)
        return torch.from_numpy(sample), torch.tensor(self.y[idx], dtype=torch.long)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Apply lightweight EEG-safe augmentations on normalized sample [C, T]."""

        if self.noise_std > 0.0:
            x = x + np.random.normal(0.0, self.noise_std, x.shape).astype(np.float32)

        if self.max_shift > 0:
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            if shift != 0:
                x = np.roll(x, shift, axis=-1)

        if self.scale_min < self.scale_max:
            scale = np.random.uniform(self.scale_min, self.scale_max)
            x = x * np.float32(scale)

        if self.channel_dropout_prob > 0.0:
            channel_mask = np.random.rand(x.shape[0]) < self.channel_dropout_prob
            if channel_mask.any() and channel_mask.sum() < x.shape[0]:
                x[channel_mask] = 0.0

        return x.astype(np.float32, copy=False)


@dataclass
class EpochResult:
    """One epoch classification stats."""

    loss: float
    acc: float


class FocalLoss(nn.Module):
    """Multi-class focal loss built on top of cross entropy."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss from logits and integer targets."""

        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


class SupConClassificationLoss(nn.Module):
    """Cross-entropy plus supervised contrastive loss on fused features."""

    def __init__(
        self,
        supcon_weight: float = 0.1,
        temperature: float = 0.07,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.supcon_weight = supcon_weight
        self.temperature = temperature
        self.weight = weight

    def supervised_contrastive_loss(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute supervised contrastive loss for one-view features."""

        if features.ndim != 2 or features.shape[0] < 2:
            return features.new_tensor(0.0)

        features = F.normalize(features, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        targets = targets.view(-1, 1)
        positive_mask = (targets == targets.T).float()
        self_mask = torch.eye(features.shape[0], device=features.device, dtype=torch.float32)
        positive_mask = positive_mask - self_mask

        exp_logits = torch.exp(logits) * (1.0 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        positive_count = positive_mask.sum(dim=1)
        valid_mask = positive_count > 0
        if not torch.any(valid_mask):
            return features.new_tensor(0.0)

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / torch.clamp(positive_count, min=1.0)
        loss = -mean_log_prob_pos[valid_mask].mean()
        return loss

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CE + weighted supervised contrastive loss."""

        ce_loss = F.cross_entropy(logits, targets, weight=self.weight)
        supcon_loss = self.supervised_contrastive_loss(features, targets)
        return ce_loss + self.supcon_weight * supcon_loss


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
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(float(loss.item()))
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        targets.extend(batch_y.detach().cpu().numpy().tolist())

    acc = float(np.mean(np.asarray(preds) == np.asarray(targets))) if targets else 0.0
    return EpochResult(loss=float(np.mean(losses)) if losses else 0.0, acc=acc)


def build_model(
    num_classes: int,
    num_channels: int,
    checkpoint: str | None,
    linear_probe: bool,
    device: torch.device,
    spatial_out_dim: int,
    temporal_out_dim: int,
    spatial_hidden_channels: int,
    spatial_temporal_channels: int,
    classifier_hidden_dim: int | None,
    classifier_dropout: float,
    use_neuro_prior: bool,
    sampling_rate: float,
    prior_out_dim: int,
    use_sampling_rate_embedding: bool,
    sampling_rate_options: tuple[float, ...],
    classifier_num_layers: int = 2,
) -> nn.Module:
    """Instantiate the dual-stream model."""

    model = SpatialMSCNN_DualStream(
        num_classes=num_classes,
        num_channels=num_channels,
        spatial_out_dim=spatial_out_dim,
        temporal_out_dim=temporal_out_dim,
        spatial_hidden_channels=spatial_hidden_channels,
        spatial_temporal_channels=spatial_temporal_channels,
        classifier_hidden_dim=classifier_hidden_dim,
        classifier_num_layers=classifier_num_layers,
        classifier_dropout=classifier_dropout,
        use_neuro_prior=use_neuro_prior,
        sampling_rate=sampling_rate,
        prior_out_dim=prior_out_dim,
        use_sampling_rate_embedding=use_sampling_rate_embedding,
        sampling_rate_options=sampling_rate_options,
    ).to(device)
    if checkpoint:
        state_dict = torch.load(checkpoint, map_location=device)
        model_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        model.load_state_dict(compatible_state, strict=False)

    if linear_probe:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fusion_classifier.parameters():
            param.requires_grad = True

    return model


def choose_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str,
) -> tuple[float, float]:
    """Tune a binary threshold on the validation subjects."""

    if y_prob.shape[1] != 2:
        return 0.5, float("nan")

    candidate_thresholds = np.linspace(0.1, 0.9, 33)
    best_threshold = 0.5
    best_score = -float("inf")

    for threshold in candidate_thresholds:
        y_pred = (y_prob[:, 1] >= threshold).astype(np.int64)
        if objective == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif objective == "accuracy":
            score = accuracy_score(y_true, y_pred)
        elif objective == "balanced_acc":
            score = balanced_accuracy_score(y_true, y_pred)
        elif objective == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif objective == "specificity":
            tn = float(np.sum((y_true == 0) & (y_pred == 0)))
            fp = float(np.sum((y_true == 0) & (y_pred == 1)))
            score = tn / (tn + fp) if (tn + fp) else 0.0
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


def collect_subject_probabilities(
    model: nn.Module,
    x_data: np.ndarray,
    y_data: np.ndarray,
    sub_ids: np.ndarray,
    batch_size: int,
    device: torch.device,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate sample probabilities into subject-level probabilities."""

    loader = DataLoader(EEGDataset(x_data, y_data), batch_size=batch_size, shuffle=False)
    model.eval()
    probs_all: list[np.ndarray] = []
    labels_all: list[int] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits, _ = model(batch_x.to(device))
            probs_all.extend(F.softmax(logits / temperature, dim=1).cpu().numpy())
            labels_all.extend(batch_y.numpy().tolist())

    probs_all = np.asarray(probs_all, dtype=np.float32)
    labels_all = np.asarray(labels_all, dtype=np.int64)
    unique_subjects = np.unique(sub_ids)
    subject_true: list[int] = []
    subject_prob: list[np.ndarray] = []

    for subject in unique_subjects:
        idx = np.where(sub_ids == subject)[0]
        subject_true.append(int(labels_all[idx[0]]))
        subject_prob.append(probs_all[idx].mean(axis=0))

    return unique_subjects, np.asarray(subject_true, dtype=np.int64), np.asarray(subject_prob, dtype=np.float32)


def collect_logits_and_labels(
    model: nn.Module,
    x_data: np.ndarray,
    y_data: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect sample-level logits and labels."""

    loader = DataLoader(EEGDataset(x_data, y_data), batch_size=batch_size, shuffle=False)
    model.eval()
    logits_all: list[np.ndarray] = []
    labels_all: list[int] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits, _ = model(batch_x.to(device))
            logits_all.append(logits.cpu().numpy())
            labels_all.extend(batch_y.numpy().tolist())
    return np.concatenate(logits_all, axis=0).astype(np.float32), np.asarray(labels_all, dtype=np.int64)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a scalar temperature on validation logits."""

    if logits.shape[1] != 2 or len(np.unique(labels)) < 2:
        return 1.0

    def objective(log_temperature: np.ndarray) -> float:
        temperature = float(np.exp(log_temperature[0]))
        scaled_logits = logits / temperature
        scaled_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(scaled_logits)
        probs = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-8, None)
        nll = -np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-8, 1.0)))
        return float(nll)

    result = minimize(objective, x0=np.asarray([0.0], dtype=np.float64), method="L-BFGS-B", bounds=[(-3.0, 3.0)])
    if not result.success:
        return 1.0
    return float(np.exp(result.x[0]))


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute binary AUC safely."""

    if y_prob.shape[1] != 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob[:, 1]))
    except ValueError:
        return float("nan")


def compute_metrics(
    subject_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
    y_pred: np.ndarray | None = None,
) -> dict[str, object]:
    """Compute subject-level metrics."""

    if y_pred is not None:
        y_pred = np.asarray(y_pred, dtype=np.int64)
    elif y_prob.shape[1] == 2 and threshold is not None:
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


def main() -> None:
    """Run cross-subject LOSO inside a single dataset."""

    parser = argparse.ArgumentParser(description="Subject-level LOSO evaluation on a single dataset")
    add_config_argument(parser)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--output-dir", default="outputs/loso_subject")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--use-ea", action="store_true")
    parser.add_argument("--linear-probe", action="store_true")
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--loss-type", choices=["ce", "focal", "supcon"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--supcon-weight", type=float, default=0.1)
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument(
        "--threshold-objective",
        choices=["accuracy", "balanced_acc", "f1", "precision", "specificity", "youden"],
        default="balanced_acc",
    )
    parser.add_argument("--selection-metric", choices=["loss", "auc"], default="loss")
    parser.add_argument("--temperature-scaling", action="store_true")
    parser.add_argument("--spatial-out-dim", type=int, default=256)
    parser.add_argument("--temporal-out-dim", type=int, default=256)
    parser.add_argument("--spatial-hidden-channels", type=int, default=64)
    parser.add_argument("--spatial-temporal-channels", type=int, default=128)
    parser.add_argument("--classifier-hidden-dim", type=int, default=None)
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
    parser.add_argument("--max-subjects", type=int, default=None, help="Debug option to limit evaluated subjects")
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
            "loss_type",
            "focal_gamma",
            "supcon_weight",
            "supcon_temperature",
            "tune_threshold",
            "threshold_objective",
            "selection_metric",
            "temperature_scaling",
            "spatial_out_dim",
            "temporal_out_dim",
            "spatial_hidden_channels",
            "spatial_temporal_channels",
            "classifier_hidden_dim",
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
        ],
    )
    if args.no_checkpoint:
        args.checkpoint = None

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_all, y_all, sub_ids_all = load_domain(Path(args.data_dir))
    unique_subjects = np.unique(sub_ids_all)
    if args.max_subjects is not None:
        unique_subjects = unique_subjects[: args.max_subjects]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(y_all.max() + 1)
    fold_results: list[dict[str, object]] = []
    all_subject_ids: list[object] = []
    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    all_y_prob: list[np.ndarray] = []

    for fold_idx, test_subject in enumerate(unique_subjects, start=1):
        print(f"\nRunning LOSO fold {fold_idx}/{len(unique_subjects)} | subject={test_subject}")
        test_mask = sub_ids_all == test_subject
        remain_mask = ~test_mask

        remain_indices = np.where(remain_mask)[0]
        test_indices = np.where(test_mask)[0]

        y_remain = y_all[remain_indices]
        sub_remain = sub_ids_all[remain_indices]
        y_test = y_all[test_indices]
        sub_test = sub_ids_all[test_indices]

        train_mask, val_mask = split_subjects_for_validation(y_remain, sub_remain, args.val_ratio, args.seed + fold_idx)
        train_indices = remain_indices[train_mask]
        val_indices = remain_indices[val_mask]

        x_train, y_train = x_all[train_indices], y_all[train_indices]
        x_val, y_val = x_all[val_indices], y_all[val_indices]
        x_test = x_all[test_indices]
        sub_val = sub_ids_all[val_indices]

        if args.use_ea:
            r_inv_half = fit_euclidean_alignment(x_train)
            x_train = apply_euclidean_alignment(x_train, r_inv_half)
            x_val = apply_euclidean_alignment(x_val, r_inv_half)
            x_test = apply_euclidean_alignment(x_test, r_inv_half)

        train_loader = DataLoader(
            EEGDataset(
                x_train,
                y_train,
                augment=args.augment_train,
                noise_std=args.augment_noise_std,
                max_shift=args.augment_max_shift,
                scale_min=args.augment_scale_min,
                scale_max=args.augment_scale_max,
                channel_dropout_prob=args.augment_channel_dropout,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(EEGDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)

        model = build_model(
            num_classes=num_classes,
            num_channels=x_all.shape[1],
            checkpoint=args.checkpoint,
            linear_probe=args.linear_probe,
            device=device,
            spatial_out_dim=args.spatial_out_dim,
            temporal_out_dim=args.temporal_out_dim,
            spatial_hidden_channels=args.spatial_hidden_channels,
            spatial_temporal_channels=args.spatial_temporal_channels,
            classifier_hidden_dim=args.classifier_hidden_dim,
            classifier_dropout=args.classifier_dropout,
            use_neuro_prior=args.use_neuro_prior,
            sampling_rate=args.sampling_rate,
            prior_out_dim=args.prior_out_dim,
            use_sampling_rate_embedding=args.use_sampling_rate_embedding,
            sampling_rate_options=tuple(args.sampling_rate_options),
        )

        class_weight = None
        if not args.disable_class_weights:
            class_counts = np.bincount(y_train)
            class_weights = len(y_train) / (len(class_counts) * np.maximum(class_counts, 1))
            class_weight = torch.tensor(class_weights, dtype=torch.float32, device=device)

        if args.loss_type == "focal":
            criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weight)
        elif args.loss_type == "supcon":
            criterion = SupConClassificationLoss(
                supcon_weight=args.supcon_weight,
                temperature=args.supcon_temperature,
                weight=class_weight,
            )
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weight)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_val_loss = float("inf")
        best_val_auc = -float("inf")
        patience_left = args.patience
        best_model_path = output_dir / f"best_model_fold_{fold_idx:03d}.pth"
        epoch_history: list[dict[str, float | int | None]] = []
        best_epoch = 0

        for epoch in range(1, args.epochs + 1):
            train_result = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
            val_result = run_epoch(model, val_loader, criterion, device, optimizer=None)
            print(
                f"  Epoch {epoch:03d} | train_loss={train_result.loss:.4f} train_acc={train_result.acc:.4f} | "
                f"val_loss={val_result.loss:.4f} val_acc={val_result.acc:.4f}"
            )

            val_auc = float("nan")
            if args.selection_metric == "auc":
                _, val_subject_true_epoch, val_subject_prob_epoch = collect_subject_probabilities(
                    model=model,
                    x_data=x_val,
                    y_data=y_val,
                    sub_ids=sub_val,
                    batch_size=args.batch_size,
                    device=device,
                )
                val_auc = safe_auc(val_subject_true_epoch, val_subject_prob_epoch)

            epoch_history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_result.loss),
                    "train_acc": float(train_result.acc),
                    "val_loss": float(val_result.loss),
                    "val_acc": float(val_result.acc),
                    "val_auc": None if np.isnan(val_auc) else float(val_auc),
                }
            )

            improved = False
            if args.selection_metric == "auc" and not np.isnan(val_auc):
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    improved = True
            elif val_result.loss < best_val_loss:
                improved = True

            if improved:
                best_val_loss = val_result.loss
                best_epoch = epoch
                patience_left = args.patience
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("  Early stopping triggered.")
                    break

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        temperature = 1.0
        if args.temperature_scaling:
            val_logits, val_labels = collect_logits_and_labels(
                model=model,
                x_data=x_val,
                y_data=y_val,
                batch_size=args.batch_size,
                device=device,
            )
            temperature = fit_temperature(val_logits, val_labels)

        threshold = None
        if args.tune_threshold:
            val_subject_ids, val_subject_true, val_subject_prob = collect_subject_probabilities(
                model=model,
                x_data=x_val,
                y_data=y_val,
                sub_ids=sub_val,
                batch_size=args.batch_size,
                device=device,
                temperature=temperature,
            )
            threshold, _ = choose_threshold(
                y_true=val_subject_true,
                y_prob=val_subject_prob,
                objective=args.threshold_objective,
            )

        test_subject_ids, test_subject_true, test_subject_prob = collect_subject_probabilities(
            model=model,
            x_data=x_test,
            y_data=y_test,
            sub_ids=sub_test,
            batch_size=args.batch_size,
            device=device,
            temperature=temperature,
        )

        fold_pred = int((test_subject_prob[:, 1] >= threshold).astype(np.int64)[0]) if (threshold is not None and test_subject_prob.shape[1] == 2) else int(np.argmax(test_subject_prob[0]))
        all_subject_ids.extend(test_subject_ids.tolist())
        all_y_true.extend(test_subject_true.tolist())
        all_y_prob.extend(test_subject_prob.tolist())
        all_y_pred.append(fold_pred)
        fold_results.append(
            {
                "fold_index": fold_idx,
                "subject_id": str(test_subject),
                "y_true": int(test_subject_true[0]),
                "y_pred": fold_pred,
                "decision_threshold": threshold,
                "temperature": float(temperature),
                "best_epoch": best_epoch,
                "best_val_loss": float(best_val_loss),
                "best_val_auc": None if best_val_auc == -float("inf") else float(best_val_auc),
                "history": epoch_history,
            }
        )

    all_subject_ids_arr = np.asarray(all_subject_ids, dtype=object)
    all_y_true_arr = np.asarray(all_y_true, dtype=np.int64)
    all_y_prob_arr = np.asarray(all_y_prob, dtype=np.float32)
    metrics = compute_metrics(
        subject_ids=all_subject_ids_arr,
        y_true=all_y_true_arr,
        y_prob=all_y_prob_arr,
        y_pred=np.asarray(all_y_pred, dtype=np.int64),
    )

    summary = {
        "data_dir": args.data_dir,
        "checkpoint": args.checkpoint,
        "use_ea": args.use_ea,
        "linear_probe": args.linear_probe,
        "disable_class_weights": args.disable_class_weights,
        "loss_type": args.loss_type,
        "focal_gamma": args.focal_gamma if args.loss_type == "focal" else None,
        "supcon_weight": args.supcon_weight if args.loss_type == "supcon" else None,
        "supcon_temperature": args.supcon_temperature if args.loss_type == "supcon" else None,
        "tune_threshold": args.tune_threshold,
        "threshold_objective": args.threshold_objective if args.tune_threshold else None,
        "selection_metric": args.selection_metric,
        "temperature_scaling": args.temperature_scaling,
        "spatial_out_dim": args.spatial_out_dim,
        "temporal_out_dim": args.temporal_out_dim,
        "spatial_hidden_channels": args.spatial_hidden_channels,
        "spatial_temporal_channels": args.spatial_temporal_channels,
        "classifier_hidden_dim": args.classifier_hidden_dim,
        "classifier_dropout": args.classifier_dropout,
        "use_neuro_prior": args.use_neuro_prior,
        "sampling_rate": args.sampling_rate,
        "use_sampling_rate_embedding": args.use_sampling_rate_embedding,
        "sampling_rate_options": args.sampling_rate_options,
        "prior_out_dim": args.prior_out_dim,
        "augment_train": args.augment_train,
        "augment_noise_std": args.augment_noise_std if args.augment_train else None,
        "augment_max_shift": args.augment_max_shift if args.augment_train else None,
        "augment_scale_min": args.augment_scale_min if args.augment_train else None,
        "augment_scale_max": args.augment_scale_max if args.augment_train else None,
        "augment_channel_dropout": args.augment_channel_dropout if args.augment_train else None,
        "max_subjects": args.max_subjects,
        "num_subjects_evaluated": int(len(unique_subjects)),
        "folds": fold_results,
        "metrics": metrics,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print("\nDataset-level LOSO summary")
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
