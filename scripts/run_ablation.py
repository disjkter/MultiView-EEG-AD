"""Run ablation study: LOSO subject-level evaluation across model variants.

Usage (single)::

    E:\\anaconda3\\envs\\BioFoundation\\python.exe scripts/run_ablation.py \
        --variant no_spatial \
        --data-dir ADSZ_processed_19ch_1s_lead \
        --checkpoint checkpoints/tage_timefreq_consistency_adssl_1s_lead.pth \
        --epochs 20 --batch-size 16 --seed 2026

Batch (all variants × datasets)::

    E:\\anaconda3\\envs\\BioFoundation\\python.exe scripts/run_ablation.py \
        --batch-mode
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import fractional_matrix_power
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ablation_model import ABLATION_VARIANTS, build_ablation_model


# ---------------------------------------------------------------------------
# Utility functions (from run_dual_stream_loso_subject.py)
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_domain_files(domain_dir: Path) -> tuple[Path, Path, Path]:
    candidates = [
        ("x_19ch.npy", "y_19ch.npy", "sub_ids_19ch.npy"),
        ("x.npy", "y.npy", "sub_ids.npy"),
    ]
    for x_name, y_name, id_name in candidates:
        xp = domain_dir / x_name
        yp = domain_dir / y_name
        ip = domain_dir / id_name
        if xp.exists() and yp.exists() and ip.exists():
            return xp, yp, ip
    raise FileNotFoundError(f"No x/y/sub_id npy under {domain_dir}")


def load_domain_mmap(domain_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Memory-safe: keep x as mmap, only load y and sub_ids into RAM."""
    xp, yp, ip = resolve_domain_files(domain_dir)
    x = np.load(xp, mmap_mode="r")
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    y = np.load(yp).astype(np.int64, copy=False)
    sub_ids = np.load(ip, allow_pickle=True)
    sub_ids = disambiguate_mixed_label_subject_ids(y, sub_ids)
    return x, y, sub_ids


class MmapEEGDataset(Dataset):
    """Memory-efficient: only loads indices into RAM, fetches from mmap on __getitem__."""

    def __init__(self, x_mmap: np.ndarray, y_arr: np.ndarray, indices: np.ndarray) -> None:
        self.x_mmap = x_mmap
        self.y_arr = y_arr
        self.indices = indices.astype(np.int64, copy=False)
        self.subj_means = np.zeros((len(indices), x_mmap.shape[1]), dtype=np.float32)
        self.subj_stds = np.zeros((len(indices), x_mmap.shape[1]), dtype=np.float32)
        for i, idx in enumerate(self.indices):
            xi = self.x_mmap[idx]
            self.subj_means[i] = xi.mean(axis=1)
            self.subj_stds[i] = xi.std(axis=1) + 1e-6

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.indices[idx]
        xi = self.x_mmap[i].astype(np.float32, copy=True)
        xi = (xi - self.subj_means[idx][:, None]) / self.subj_stds[idx][:, None]
        return torch.from_numpy(xi), torch.tensor(self.y_arr[i], dtype=torch.long)


def disambiguate_mixed_label_subject_ids(y: np.ndarray, sub_ids: np.ndarray) -> np.ndarray:
    label_sets: dict[str, set[int]] = {}
    for s, l in zip(sub_ids, y):
        label_sets.setdefault(str(s), set()).add(int(l))
    mixed = {s for s, ls in label_sets.items() if len(ls) > 1}
    if not mixed:
        return sub_ids
    return np.asarray(
        [f"label{int(l)}__{s}" if str(s) in mixed else str(s) for s, l in zip(sub_ids, y)],
        dtype=object,
    )


def fit_euclidean_alignment(x_train: np.ndarray) -> np.ndarray:
    n, c, _ = x_train.shape
    covs = np.zeros((n, c, c), dtype=np.float64)
    for i in range(n):
        covs[i] = np.cov(x_train[i])
    mean_cov = np.mean(covs, axis=0) + np.eye(c, dtype=np.float64) * 1e-6
    return np.real(fractional_matrix_power(mean_cov, -0.5)).astype(np.float32)


def apply_euclidean_alignment(x: np.ndarray, r_inv_half: np.ndarray) -> np.ndarray:
    aligned = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        aligned[i] = r_inv_half @ x[i]
    return aligned


def split_subjects_for_validation(
    y: np.ndarray, sub_ids: np.ndarray, val_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = np.unique(sub_ids)
    sub_labs = {s: int(y[np.where(sub_ids == s)[0][0]]) for s in unique}
    train_s, val_s = [], []
    for lab in sorted(set(sub_labs.values())):
        ls = [s for s in unique if sub_labs[s] == lab]
        rng.shuffle(ls)
        nv = max(1, int(round(len(ls) * val_ratio)))
        if nv >= len(ls) and len(ls) > 1:
            nv = len(ls) - 1
        val_s.extend(ls[:nv])
        train_s.extend(ls[nv:])
    return np.isin(sub_ids, np.asarray(train_s, dtype=object)), np.isin(sub_ids, np.asarray(val_s, dtype=object))


class EEGDataset(Dataset):
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        self.x = x_data.astype(np.float32, copy=False)
        self.y = y_data.astype(np.int64, copy=False)
        for i in range(len(self.x)):
            m = self.x[i].mean(axis=1, keepdims=True)
            s = self.x[i].std(axis=1, keepdims=True) + 1e-6
            self.x[i] = (self.x[i] - m) / s

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
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses, preds, targets = [], [], []
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        with torch.set_grad_enabled(is_train):
            logits, _ = model(bx)
            loss = criterion(logits, by)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(float(loss.item()))
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        targets.extend(by.detach().cpu().numpy().tolist())
    acc = float(np.mean(np.asarray(preds) == np.asarray(targets))) if targets else 0.0
    return float(np.mean(losses)) if losses else 0.0, acc


def collect_subject_probabilities(
    model: nn.Module, x_data: np.ndarray, y_data: np.ndarray,
    sub_ids: np.ndarray, batch_size: int, device: torch.device, temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(EEGDataset(x_data, y_data), batch_size=batch_size, shuffle=False)
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for bx, by in loader:
            logits, _ = model(bx.to(device))
            probs_all.extend(F.softmax(logits / temperature, dim=1).cpu().numpy())
            labels_all.extend(by.numpy().tolist())
    probs_all = np.asarray(probs_all, dtype=np.float32)
    labels_all = np.asarray(labels_all, dtype=np.int64)
    unique = np.unique(sub_ids)
    sub_true, sub_prob = [], []
    for s in unique:
        idx = np.where(sub_ids == s)[0]
        sub_true.append(int(labels_all[idx[0]]))
        sub_prob.append(probs_all[idx].mean(axis=0))
    return unique, np.asarray(sub_true, dtype=np.int64), np.asarray(sub_prob, dtype=np.float32)


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray, objective: str) -> tuple[float, float]:
    if y_prob.shape[1] != 2:
        return 0.5, float("nan")
    best_t, best_s = 0.5, -float("inf")
    for t in np.linspace(0.1, 0.9, 33):
        yp = (y_prob[:, 1] >= t).astype(np.int64)
        if objective == "f1":
            s = f1_score(y_true, yp, zero_division=0)
        elif objective == "accuracy":
            s = accuracy_score(y_true, yp)
        elif objective == "balanced_acc":
            s = balanced_accuracy_score(y_true, yp)
        elif objective == "precision":
            s = precision_score(y_true, yp, zero_division=0)
        else:
            tn = float(np.sum((y_true == 0) & (yp == 0)))
            fp = float(np.sum((y_true == 0) & (yp == 1)))
            s = tn / (tn + fp) if (tn + fp) else 0.0
        if s > best_s:
            best_s, best_t = float(s), float(t)
    return best_t, best_s


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_prob.shape[1] != 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob[:, 1]))
    except ValueError:
        return float("nan")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray | None = None) -> dict[str, object]:
    if y_pred is not None:
        y_pred = np.asarray(y_pred, dtype=np.int64)
    else:
        y_pred = np.argmax(y_prob, axis=1).astype(np.int64)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    auc_v = safe_auc(y_true, y_prob)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(auc_v),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# Main LOSO evaluation for one variant
# ---------------------------------------------------------------------------

def run_loso_ablation(
    variant: str,
    data_dir: str,
    checkpoint: str | None,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_ratio: float,
    patience: int,
    use_ea: bool,
    use_class_weights: bool,
    tune_threshold: bool,
    threshold_objective: str,
    device: torch.device,
    max_subjects: int | None = None,
) -> dict[str, object]:
    """Run LOSO evaluation for one ablation variant on one dataset."""

    data_path = PROJECT_ROOT / data_dir
    x_all, y_all, sub_ids_all = load_domain_mmap(data_path)
    unique_subjects = np.unique(sub_ids_all)
    if max_subjects is not None and max_subjects > 0:
        unique_subjects = unique_subjects[:max_subjects]

    # Pre-filter to only selected subjects — map global indices to local
    selected_mask = np.isin(sub_ids_all, unique_subjects)
    global_indices = np.where(selected_mask)[0]  # maps local idx -> global mmap idx
    y_all = y_all[global_indices]
    sub_ids_all = sub_ids_all[global_indices]

    def x_slice(local_indices: np.ndarray) -> np.ndarray:
        gi = global_indices[local_indices]
        return np.array(x_all[gi], dtype=np.float32)

    num_classes = int(y_all.max() + 1)
    num_channels = x_all.shape[1]

    all_y_true, all_y_pred, all_y_prob = [], [], []
    fold_results = []

    for fold_idx, test_subject in enumerate(unique_subjects, start=1):
        print(f"  Fold {fold_idx}/{len(unique_subjects)} subject={test_subject}", end="", flush=True)

        test_mask = sub_ids_all == test_subject
        remain_mask = ~test_mask
        remain_idx = np.where(remain_mask)[0]
        test_idx = np.where(test_mask)[0]

        y_remain = y_all[remain_idx]
        sub_remain = sub_ids_all[remain_idx]
        y_test = y_all[test_idx]
        sub_test = sub_ids_all[test_idx]

        train_mask, val_mask = split_subjects_for_validation(
            y_remain, sub_remain, val_ratio, seed + fold_idx
        )
        train_idx = remain_idx[train_mask]
        val_idx = remain_idx[val_mask]

        x_train = x_slice(train_idx)
        y_train = y_all[train_idx]
        x_val = x_slice(val_idx)
        y_val = y_all[val_idx]
        x_test = x_slice(test_idx)
        sub_val = sub_ids_all[val_idx]

        if use_ea:
            r_inv = fit_euclidean_alignment(x_train)
            x_train = apply_euclidean_alignment(x_train, r_inv)
            x_val = apply_euclidean_alignment(x_val, r_inv)
            x_test = apply_euclidean_alignment(x_test, r_inv)

        train_loader = DataLoader(
            EEGDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(EEGDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

        model = build_ablation_model(
            variant=variant,
            num_classes=num_classes,
            num_channels=num_channels,
            checkpoint=checkpoint,
            device=device,
        )

        class_weight = None
        if use_class_weights:
            counts = np.bincount(y_train)
            cw = len(y_train) / (len(counts) * np.maximum(counts, 1))
            class_weight = torch.tensor(cw, dtype=torch.float32, device=device)

        criterion = nn.CrossEntropyLoss(weight=class_weight)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay,
        )

        best_loss = float("inf")
        patience_left = patience
        best_epoch = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            vl_loss, vl_acc = run_epoch(model, val_loader, criterion, device)
            if vl_loss < best_loss:
                best_loss = vl_loss
                best_epoch = epoch
                patience_left = patience
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Test evaluation
        _, test_true, test_prob = collect_subject_probabilities(
            model, x_test, y_test, sub_test, batch_size, device,
        )

        threshold = 0.5
        if tune_threshold:
            _, val_true, val_prob = collect_subject_probabilities(
                model, x_val, y_val, sub_val, batch_size, device,
            )
            threshold, _ = choose_threshold(val_true, val_prob, threshold_objective)

        fold_pred = int((test_prob[:, 1] >= threshold).astype(np.int64)[0]) if test_prob.shape[1] == 2 else int(np.argmax(test_prob[0]))
        all_y_true.extend(test_true.tolist())
        all_y_prob.extend(test_prob.tolist())
        all_y_pred.append(fold_pred)
        fold_results.append({
            "fold_index": fold_idx,
            "subject_id": str(test_subject),
            "y_true": int(test_true[0]),
            "y_pred": fold_pred,
            "threshold": float(threshold),
            "best_epoch": best_epoch,
            "best_val_loss": float(best_loss),
        })
        print(f"  pred={fold_pred} true={test_true[0]}")

    all_y_true_a = np.asarray(all_y_true, dtype=np.int64)
    all_y_prob_a = np.asarray(all_y_prob, dtype=np.float32)
    all_y_pred_a = np.asarray(all_y_pred, dtype=np.int64)
    metrics = compute_metrics(all_y_true_a, all_y_prob_a, all_y_pred_a)

    return {
        "variant": variant,
        "label": ABLATION_VARIANTS[variant]["label"],
        "data_dir": data_dir,
        "checkpoint": checkpoint,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "use_ea": use_ea,
        "use_class_weights": use_class_weights,
        "num_subjects": int(len(unique_subjects)),
        "metrics": metrics,
        "folds": fold_results,
    }


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

BATCH_DATASETS = [
    "ADSZ_processed_19ch_1s_lead",
    "ADFSU_processed_19ch_1s_lead",
    "ds004504_processed_19ch_1s_lead",
]

BATCH_VARIANTS = [
    "full",
    "no_spatial",
    "no_mscnn",
    "no_multiscale",
    "no_se",
    "attn_fusion",
    "gate_fusion",
    "cls_1layer",
    "cls_3layer",
    "neuro_prior",
]

BATCH_SEEDS = [2026, 2027, 2028]

BATCH_CHECKPOINT = "checkpoints/tage_timefreq_consistency_adssl_1s_lead.pth"


def run_batch(output_dir: str, max_subjects: int | None = None) -> None:
    """Run all variants × datasets × seeds."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    total = len(BATCH_DATASETS) * len(BATCH_VARIANTS) * len(BATCH_SEEDS)
    idx = 0

    for dataset in BATCH_DATASETS:
        data_path = PROJECT_ROOT / dataset
        if not data_path.exists():
            print(f"[SKIP] Dataset not found: {dataset}")
            continue

        for variant in BATCH_VARIANTS:
            for seed in BATCH_SEEDS:
                idx += 1
                tag = f"{variant}_s{seed}"
                exp_dir = out_root / dataset / tag
                result_path = exp_dir / "summary.json"

                if result_path.exists():
                    print(f"[{idx}/{total}] SKIP (exists): {dataset} / {tag}")
                    continue

                exp_dir.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                print(f"\n{'='*70}")
                print(f"[{idx}/{total}] {dataset} / {tag}")
                print(f"{'='*70}")

                set_seed(seed)
                result = run_loso_ablation(
                    variant=variant,
                    data_dir=dataset,
                    checkpoint=BATCH_CHECKPOINT,
                    seed=seed,
                    epochs=20,
                    batch_size=16,
                    lr=5e-5,
                    weight_decay=1e-4,
                    val_ratio=0.2,
                    patience=5,
                    use_ea=False,
                    use_class_weights=True,
                    tune_threshold=True,
                    threshold_objective="balanced_acc",
                    device=device,
                    max_subjects=max_subjects,
                )

                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                m = result["metrics"]
                elapsed = time.time() - t0
                print(f"  Done in {elapsed:.0f}s | Acc={m['accuracy']:.4f} BalAcc={m['balanced_accuracy']:.4f} "
                      f"F1={m['f1']:.4f} AUC={m['auc']:.4f} Sens={m['sensitivity']:.4f} Spec={m['specificity']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study LOSO evaluation")
    parser.add_argument("--variant", choices=list(ABLATION_VARIANTS), default="full")
    parser.add_argument("--data-dir", default="ADSZ_processed_19ch_1s_lead")
    parser.add_argument("--checkpoint", default="checkpoints/tage_timefreq_consistency_adssl_1s_lead.pth")
    parser.add_argument("--output-dir", default="outputs/ablation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--use-ea", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-threshold-tune", action="store_true")
    parser.add_argument("--threshold-objective", default="balanced_acc")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--batch-mode", action="store_true")
    args = parser.parse_args()

    if args.batch_mode:
        run_batch(args.output_dir, max_subjects=args.max_subjects)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    t0 = time.time()
    result = run_loso_ablation(
        variant=args.variant,
        data_dir=args.data_dir,
        checkpoint=args.checkpoint if not args.checkpoint == "" else None,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        patience=args.patience,
        use_ea=args.use_ea,
        use_class_weights=not args.no_class_weights,
        tune_threshold=not args.no_threshold_tune,
        threshold_objective=args.threshold_objective,
        device=device,
        max_subjects=args.max_subjects,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    m = result["metrics"]
    elapsed = time.time() - t0
    print(f"\n=== {ABLATION_VARIANTS[args.variant]['label']} on {args.data_dir} ===")
    print(f"Time: {elapsed:.0f}s")
    print(f"Accuracy:      {m['accuracy']:.4f}")
    print(f"Balanced Acc:  {m['balanced_accuracy']:.4f}")
    print(f"F1 Score:      {m['f1']:.4f}")
    print(f"AUC:           {m['auc']:.4f}")
    print(f"Sensitivity:   {m['sensitivity']:.4f}")
    print(f"Specificity:   {m['specificity']:.4f}")


if __name__ == "__main__":
    main()
