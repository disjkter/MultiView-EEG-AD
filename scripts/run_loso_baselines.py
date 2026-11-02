"""Run subject-level LOSO baselines on one processed EEG dataset."""

from __future__ import annotations

import argparse
import json
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
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.EEGNet import EEGNet
from models.CEEDNet import CEEDNet1DResNet18
from models.iTransformer import ITransformer
from models.MTDNet import MTDNet
from models.ShallowConvNet import ShallowConvNet
from models.TCN import TCN


def set_seed(seed: int) -> None:
    """Set random seeds."""

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
    """Load one processed dataset."""

    x_path, y_path, id_path = resolve_domain_files(domain_dir)
    x = np.load(x_path, mmap_mode="r")
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    y = np.load(y_path, mmap_mode="r")
    if y.dtype != np.int64:
        y = y.astype(np.int64, copy=False)
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
    """Fit EA on training trials only."""

    n_trials, n_channels, _ = x_train.shape
    covs = np.zeros((n_trials, n_channels, n_channels), dtype=np.float64)
    for i in range(n_trials):
        covs[i] = np.cov(x_train[i])
    mean_cov = np.mean(covs, axis=0)
    mean_cov += np.eye(n_channels, dtype=np.float64) * 1e-6
    r_inv_half = fractional_matrix_power(mean_cov, -0.5)
    return np.real(r_inv_half).astype(np.float32)


def apply_euclidean_alignment(x: np.ndarray, r_inv_half: np.ndarray) -> np.ndarray:
    """Apply Euclidean Alignment."""

    if x.nbytes >= 256 * 1024 * 1024:
        fd, tmp_path = tempfile.mkstemp(prefix="ea_aligned_", suffix=".dat")
        os.close(fd)
        aligned = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=x.shape)
    else:
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
    """Per-sample normalized EEG dataset."""

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
    """One epoch stats."""

    loss: float
    acc: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochResult:
    """Run one training or validation epoch."""

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
    return EpochResult(loss=float(np.mean(losses)) if losses else 0.0, acc=acc)


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray, objective: str) -> tuple[float, float]:
    """Tune binary threshold on validation subjects."""

    if y_prob.shape[1] != 2:
        return 0.5, float("nan")

    candidate_thresholds = np.linspace(0.1, 0.9, 33)
    best_threshold = 0.5
    best_score = -float("inf")

    for threshold in candidate_thresholds:
        y_pred = (y_prob[:, 1] >= threshold).astype(np.int64)
        if objective == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_score


def compute_metrics(
    subject_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, object]:
    """Compute subject-level binary metrics."""

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob[:, 1]) if y_prob.shape[1] == 2 else float("nan")
    except ValueError:
        auc = float("nan")

    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "subject_ids": subject_ids.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
        "auc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "report": classification_report(y_true, y_pred, digits=4, zero_division=0),
    }


def aggregate_subject_probabilities(
    sub_ids: np.ndarray,
    y_data: np.ndarray,
    prob_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate sample probabilities to subject-level mean probabilities."""

    unique_subjects = np.unique(sub_ids)
    subject_true: list[int] = []
    subject_prob: list[np.ndarray] = []
    for subject in unique_subjects:
        mask = sub_ids == subject
        subject_true.append(int(y_data[mask][0]))
        subject_prob.append(prob_all[mask].mean(axis=0))
    return unique_subjects, np.asarray(subject_true, dtype=np.int64), np.asarray(subject_prob, dtype=np.float32)


def collect_subject_probabilities_torch(
    model: nn.Module,
    x_data: np.ndarray,
    y_data: np.ndarray,
    sub_ids: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect subject-level probabilities for a torch model."""

    loader = DataLoader(EEGDataset(x_data, y_data), batch_size=batch_size, shuffle=False)
    model.eval()
    probs_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _ in loader:
            logits, _ = model(batch_x.to(device))
            probs_all.extend(F.softmax(logits, dim=1).cpu().numpy())
    return aggregate_subject_probabilities(sub_ids, y_data, np.asarray(probs_all, dtype=np.float32))


def flatten_trials(x_data: np.ndarray) -> np.ndarray:
    """Flatten trials for classical ML baselines."""

    x_norm = x_data.astype(np.float32, copy=True)
    for i in range(len(x_norm)):
        mean = x_norm[i].mean(axis=1, keepdims=True)
        std = x_norm[i].std(axis=1, keepdims=True) + 1e-6
        x_norm[i] = (x_norm[i] - mean) / std
    return x_norm.reshape(x_norm.shape[0], -1)


def collect_subject_probabilities_svm(
    model: Pipeline,
    x_data: np.ndarray,
    y_data: np.ndarray,
    sub_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect subject-level probabilities for SVM."""

    chunk_size = 256
    prob_chunks: list[np.ndarray] = []
    for start in range(0, len(x_data), chunk_size):
        end = min(start + chunk_size, len(x_data))
        x_flat = flatten_trials(x_data[start:end])
        prob_chunks.append(model.predict_proba(x_flat).astype(np.float32, copy=False))
    prob_all = np.concatenate(prob_chunks, axis=0) if prob_chunks else np.empty((0, 2), dtype=np.float32)
    return aggregate_subject_probabilities(sub_ids, y_data, prob_all)


def build_svm(seed: int) -> Pipeline:
    """Build a classical SVM pipeline."""

    return Pipeline(
        steps=[
            # Reduce the flattened trial first to keep large cross-domain fits memory-safe.
            ("svd", TruncatedSVD(n_components=128, algorithm="randomized", random_state=seed)),
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        ]
    )


def build_random_forest(seed: int) -> Pipeline:
    """Build a Random Forest pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, svd_solver="full", random_state=seed)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description="Run subject-level LOSO baselines.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        choices=["svm", "rf", "eegnet", "shallowconvnet", "mtdnet", "itransformer", "tcn", "ceednet"],
        required=True,
    )
    parser.add_argument("--output-dir", type=str, required=True)
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
    parser.add_argument("--threshold-objective", type=str, choices=["balanced_acc", "f1"], default="balanced_acc")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    """Run full LOSO baseline evaluation."""

    args = parse_args()
    set_seed(args.seed)

    domain_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_all, y_all, sub_ids_all = load_domain(domain_dir)
    unique_subjects = np.unique(sub_ids_all)
    if args.max_subjects is not None:
        unique_subjects = unique_subjects[: args.max_subjects]

    num_classes = int(len(np.unique(y_all)))
    num_channels = int(x_all.shape[1])
    device = torch.device(args.device)

    fold_summaries: list[dict[str, object]] = []
    subject_true_all: list[int] = []
    subject_pred_all: list[int] = []
    subject_prob_all: list[np.ndarray] = []
    subject_ids_eval: list[object] = []

    for fold_index, test_subject in enumerate(unique_subjects, start=1):
        print(f"\nRunning LOSO fold {fold_index}/{len(unique_subjects)} | subject={test_subject}")
        test_mask = sub_ids_all == test_subject
        remain_mask = ~test_mask

        x_remain = x_all[remain_mask]
        y_remain = y_all[remain_mask]
        sub_remain = sub_ids_all[remain_mask]

        train_mask_local, val_mask_local = split_subjects_for_validation(
            y_remain,
            sub_remain,
            val_ratio=args.val_ratio,
            seed=args.seed + fold_index,
        )

        x_train = x_remain[train_mask_local]
        y_train = y_remain[train_mask_local]
        x_val = x_remain[val_mask_local]
        y_val = y_remain[val_mask_local]
        sub_val = sub_remain[val_mask_local]
        x_test = x_all[test_mask]
        y_test = y_all[test_mask]
        sub_test = sub_ids_all[test_mask]

        if args.use_ea:
            r_inv_half = fit_euclidean_alignment(x_train)
            x_train = apply_euclidean_alignment(x_train, r_inv_half)
            x_val = apply_euclidean_alignment(x_val, r_inv_half)
            x_test = apply_euclidean_alignment(x_test, r_inv_half)

        if args.model in {"svm", "rf"}:
            model = build_svm(args.seed + fold_index) if args.model == "svm" else build_random_forest(args.seed + fold_index)
            model.fit(flatten_trials(x_train), y_train)
            _, val_true, val_prob = collect_subject_probabilities_svm(model, x_val, y_val, sub_val)
            _, test_true, test_prob = collect_subject_probabilities_svm(model, x_test, y_test, sub_test)
        else:
            if args.model == "eegnet":
                model = EEGNet(
                    num_channels=num_channels,
                    num_classes=num_classes,
                    dropout=args.dropout,
                ).to(device)
            elif args.model == "mtdnet":
                model = MTDNet(
                    num_channels=num_channels,
                    num_classes=num_classes,
                ).to(device)
            elif args.model == "itransformer":
                model = ITransformer(
                    num_channels=num_channels,
                    num_classes=num_classes,
                    dropout=args.dropout,
                ).to(device)
            elif args.model == "tcn":
                model = TCN(
                    num_channels=num_channels,
                    num_classes=num_classes,
                    dropout=args.dropout,
                ).to(device)
            elif args.model == "ceednet":
                model = CEEDNet1DResNet18(
                    num_channels=num_channels,
                    num_classes=num_classes,
                    seq_length=int(x_all.shape[2]),
                    dropout=args.dropout,
                ).to(device)
            else:
                model = ShallowConvNet(
                    num_channels=num_channels,
                    num_classes=num_classes,
                    dropout=args.dropout,
                ).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            train_loader = DataLoader(
                EEGDataset(x_train, y_train),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
            )
            val_loader = DataLoader(EEGDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)

            best_state = None
            best_val_loss = float("inf")
            stale_epochs = 0
            for epoch in range(1, args.epochs + 1):
                train_result = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
                val_result = run_epoch(model, val_loader, criterion, device, optimizer=None)
                print(
                    f"  Epoch {epoch:03d} | train_loss={train_result.loss:.4f} "
                    f"train_acc={train_result.acc:.4f} | val_loss={val_result.loss:.4f} "
                    f"val_acc={val_result.acc:.4f}"
                )
                if val_result.loss < best_val_loss:
                    best_val_loss = val_result.loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= args.patience:
                        print("  Early stopping triggered.")
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            _, val_true, val_prob = collect_subject_probabilities_torch(
                model, x_val, y_val, sub_val, args.batch_size, device
            )
            _, test_true, test_prob = collect_subject_probabilities_torch(
                model, x_test, y_test, sub_test, args.batch_size, device
            )

        decision_threshold = 0.5
        if args.tune_threshold and num_classes == 2:
            decision_threshold, _ = choose_threshold(val_true, val_prob, args.threshold_objective)

        if num_classes == 2:
            test_pred = (test_prob[:, 1] >= decision_threshold).astype(np.int64)
        else:
            test_pred = np.argmax(test_prob, axis=1)

        fold_summaries.append(
            {
                "fold_index": fold_index,
                "subject_id": str(test_subject),
                "y_true": int(test_true[0]),
                "y_pred": int(test_pred[0]),
                "decision_threshold": float(decision_threshold),
            }
        )
        subject_ids_eval.extend(sub_test[:1].tolist())
        subject_true_all.extend(test_true.tolist())
        subject_pred_all.extend(test_pred.tolist())
        subject_prob_all.extend(test_prob.tolist())

    metrics = compute_metrics(
        np.asarray(subject_ids_eval, dtype=object),
        np.asarray(subject_true_all, dtype=np.int64),
        np.asarray(subject_pred_all, dtype=np.int64),
        np.asarray(subject_prob_all, dtype=np.float32),
    )

    summary = {
        "data_dir": str(domain_dir),
        "model": args.model,
        "use_ea": bool(args.use_ea),
        "tune_threshold": bool(args.tune_threshold),
        "threshold_objective": args.threshold_objective,
        "max_subjects": args.max_subjects,
        "num_subjects_evaluated": int(len(unique_subjects)),
        "folds": fold_summaries,
        "metrics": metrics,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

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
