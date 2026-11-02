"""Run a spectrogram CNN LOSO baseline inspired by EEG spectrogram literature."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import fractional_matrix_power
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.SpectrogramCNN import SpectrogramCNN  # noqa: E402
from scripts.run_loso_baselines import (  # noqa: E402
    EpochResult,
    choose_threshold,
    compute_metrics,
    load_domain,
    set_seed,
    split_subjects_for_validation,
)


def fit_euclidean_alignment(x_train: np.ndarray) -> np.ndarray:
    """Fit Euclidean alignment using training trials only."""

    n_trials, n_channels, _ = x_train.shape
    covs = np.zeros((n_trials, n_channels, n_channels), dtype=np.float64)
    for i in range(n_trials):
        covs[i] = np.cov(x_train[i])
    mean_cov = np.mean(covs, axis=0)
    mean_cov += np.eye(n_channels, dtype=np.float64) * 1e-6
    r_inv_half = fractional_matrix_power(mean_cov, -0.5)
    return np.real(r_inv_half).astype(np.float32)


def apply_euclidean_alignment(x_data: np.ndarray, r_inv_half: np.ndarray) -> np.ndarray:
    """Apply Euclidean alignment."""

    aligned = np.zeros_like(x_data, dtype=np.float32)
    for i in range(x_data.shape[0]):
        aligned[i] = r_inv_half @ x_data[i]
    return aligned


def build_spectrogram_tensor(
    x_data: np.ndarray,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """Convert EEG trials to log-magnitude spectrograms.

    Output shape:
        [num_trials, channels, freq_bins, time_bins]
    """

    x_tensor = torch.from_numpy(x_data.astype(np.float32, copy=False))
    window = torch.hann_window(n_fft)
    specs: list[torch.Tensor] = []

    for trial in x_tensor:
        trial = (trial - trial.mean(dim=1, keepdim=True)) / (trial.std(dim=1, keepdim=True) + 1e-6)
        trial_specs: list[torch.Tensor] = []
        for ch_signal in trial:
            stft = torch.stft(
                ch_signal,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                return_complex=True,
            )
            mag = torch.log1p(torch.abs(stft))
            trial_specs.append(mag)
        specs.append(torch.stack(trial_specs, dim=0))

    return torch.stack(specs, dim=0).numpy().astype(np.float32)


class SpectrogramDataset(Dataset):
    """Dataset for precomputed spectrogram tensors."""

    def __init__(self, x_spec: np.ndarray, y_data: np.ndarray):
        self.x = x_spec.astype(np.float32, copy=False)
        self.y = y_data.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.long)


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


def aggregate_subject_probabilities(
    sub_ids: np.ndarray,
    y_data: np.ndarray,
    prob_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate sample probabilities into subject means."""

    unique_subjects = np.unique(sub_ids)
    subject_true: list[int] = []
    subject_prob: list[np.ndarray] = []
    for subject in unique_subjects:
        mask = sub_ids == subject
        subject_true.append(int(y_data[mask][0]))
        subject_prob.append(prob_all[mask].mean(axis=0))
    return unique_subjects, np.asarray(subject_true, dtype=np.int64), np.asarray(subject_prob, dtype=np.float32)


def collect_subject_probabilities(
    model: nn.Module,
    x_spec: np.ndarray,
    y_data: np.ndarray,
    sub_ids: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict subject-level probabilities."""

    loader = DataLoader(SpectrogramDataset(x_spec, y_data), batch_size=batch_size, shuffle=False)
    model.eval()
    probs_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _ in loader:
            logits, _ = model(batch_x.to(device))
            probs_all.extend(F.softmax(logits, dim=1).cpu().numpy())
    return aggregate_subject_probabilities(sub_ids, y_data, np.asarray(probs_all, dtype=np.float32))


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description="Run LOSO spectrogram CNN baseline.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--use-ea", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--threshold-objective", type=str, choices=["balanced_acc", "f1"], default="balanced_acc")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--n-fft", type=int, default=64)
    parser.add_argument("--hop-length", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    """Run full LOSO evaluation."""

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
        print(f"\nRunning LOSO Spectrogram-CNN fold {fold_index}/{len(unique_subjects)} | subject={test_subject}")
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

        x_train_spec = build_spectrogram_tensor(x_train, args.n_fft, args.hop_length)
        x_val_spec = build_spectrogram_tensor(x_val, args.n_fft, args.hop_length)
        x_test_spec = build_spectrogram_tensor(x_test, args.n_fft, args.hop_length)

        model = SpectrogramCNN(
            num_channels=num_channels,
            num_classes=num_classes,
            base_channels=args.base_channels,
            dropout=args.dropout,
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        train_loader = DataLoader(
            SpectrogramDataset(x_train_spec, y_train),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(SpectrogramDataset(x_val_spec, y_val), batch_size=args.batch_size, shuffle=False)

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

        _, val_true, val_prob = collect_subject_probabilities(
            model, x_val_spec, y_val, sub_val, args.batch_size, device
        )
        _, test_true, test_prob = collect_subject_probabilities(
            model, x_test_spec, y_test, sub_test, args.batch_size, device
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
        "model": "spectrogram_cnn",
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
