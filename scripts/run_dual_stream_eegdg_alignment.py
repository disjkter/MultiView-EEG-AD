"""Run a minimal EEG-DG style alignment experiment with the dual-stream EEG model.

The training objective combines:
- supervised source-domain classification loss
- marginal feature alignment between source and target
- conditional alignment using source labels and target pseudo-labels
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.Spatial_MSCNN import SpatialMSCNN_DualStream
from scripts.run_dual_stream_cross_domain import (
    apply_euclidean_alignment,
    choose_threshold,
    collect_subject_probabilities,
    compute_subject_metrics,
    fit_euclidean_alignment,
    split_source_subjects,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

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
    """Load one processed domain."""

    x_path, y_path, id_path = resolve_domain_files(domain_dir)
    x = np.load(x_path)
    y = np.load(y_path)
    sub_ids = np.load(id_path, allow_pickle=True)
    return x.astype(np.float32), y.astype(np.int64), sub_ids


class EEGDataset(Dataset):
    """EEG dataset with per-sample channel-wise normalization."""

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray | None = None):
        self.x = np.asarray(x_data, dtype=np.float32)
        self.y = None if y_data is None else np.asarray(y_data, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = np.array(self.x[idx], dtype=np.float32, copy=True)
        mean = sample.mean(axis=1, keepdims=True)
        std = sample.std(axis=1, keepdims=True) + 1e-6
        x = torch.from_numpy((sample - mean) / std)
        if self.y is None:
            return x, torch.tensor(-1, dtype=torch.long)
        return x, torch.tensor(self.y[idx], dtype=torch.long)


@dataclass
class TrainStats:
    """One epoch of alignment training statistics."""

    total_loss: float
    cls_loss: float
    marginal_loss: float
    conditional_loss: float
    source_acc: float
    val_loss: float
    val_acc: float


def next_batch(iterator, loader):
    """Cycle through a loader without stopping."""

    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def gaussian_kernel(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
) -> torch.Tensor:
    """Compute a multi-kernel RBF similarity matrix."""

    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0)
    total1 = total.unsqueeze(1)
    l2_distance = ((total0 - total1) ** 2).sum(2)
    bandwidth = torch.sum(l2_distance.detach()) / (total.shape[0] ** 2 - total.shape[0] + 1e-6)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    kernels = [
        torch.exp(-l2_distance / (bandwidth * (kernel_mul**i) + 1e-6))
        for i in range(kernel_num)
    ]
    return sum(kernels)


def marginal_mmd_loss(source_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
    """Estimate marginal alignment loss with multi-kernel MMD."""

    kernels = gaussian_kernel(source_feat, target_feat)
    batch_size = source_feat.shape[0]
    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]
    return xx.mean() + yy.mean() - xy.mean() - yx.mean()


def conditional_alignment_loss(
    source_feat: torch.Tensor,
    source_y: torch.Tensor,
    target_feat: torch.Tensor,
    target_logits: torch.Tensor,
    num_classes: int,
    confidence_threshold: float,
) -> torch.Tensor:
    """Align class centers between source labels and target pseudo-labels."""

    target_prob = F.softmax(target_logits, dim=1)
    target_conf, target_pseudo = torch.max(target_prob, dim=1)
    total_loss = source_feat.new_tensor(0.0)
    matched_classes = 0

    for class_idx in range(num_classes):
        source_mask = source_y == class_idx
        target_mask = (target_pseudo == class_idx) & (target_conf >= confidence_threshold)
        if int(source_mask.sum()) == 0 or int(target_mask.sum()) == 0:
            continue

        source_center = source_feat[source_mask].mean(dim=0)
        target_center = target_feat[target_mask].mean(dim=0)
        total_loss = total_loss + F.mse_loss(source_center, target_center)
        matched_classes += 1

    if matched_classes == 0:
        return source_feat.new_tensor(0.0)
    return total_loss / matched_classes


def evaluate_source_validation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate classification loss/accuracy on source validation."""

    model.eval()
    losses: list[float] = []
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            losses.append(float(loss.item()))
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            all_targets.extend(batch_y.cpu().numpy().tolist())

    accuracy = float(np.mean(np.asarray(all_preds) == np.asarray(all_targets))) if all_targets else 0.0
    return float(np.mean(losses)) if losses else 0.0, accuracy


def build_model(
    num_classes: int,
    num_channels: int,
    checkpoint: str | None,
    linear_probe: bool,
    device: torch.device,
) -> nn.Module:
    """Instantiate the dual-stream model and optionally freeze the backbone."""

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


def train_one_epoch(
    model: nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    val_loader: DataLoader,
    class_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_marginal: float,
    lambda_conditional: float,
    confidence_threshold: float,
    num_classes: int,
) -> TrainStats:
    """Train one epoch with EEG-DG style alignment losses."""

    model.train()
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    num_steps = max(len(source_loader), len(target_loader))

    total_losses: list[float] = []
    cls_losses: list[float] = []
    marginal_losses: list[float] = []
    conditional_losses: list[float] = []
    source_preds: list[int] = []
    source_true: list[int] = []

    for _ in range(num_steps):
        (source_x, source_y), source_iter = next_batch(source_iter, source_loader)
        (target_x, _), target_iter = next_batch(target_iter, target_loader)

        source_x = source_x.to(device)
        source_y = source_y.to(device)
        target_x = target_x.to(device)

        source_logits, source_feat = model(source_x)
        target_logits, target_feat = model(target_x)

        cls_loss = class_criterion(source_logits, source_y)
        marginal_loss = marginal_mmd_loss(source_feat, target_feat)
        conditional_loss = conditional_alignment_loss(
            source_feat=source_feat,
            source_y=source_y,
            target_feat=target_feat,
            target_logits=target_logits,
            num_classes=num_classes,
            confidence_threshold=confidence_threshold,
        )
        loss = cls_loss + lambda_marginal * marginal_loss + lambda_conditional * conditional_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_losses.append(float(loss.item()))
        cls_losses.append(float(cls_loss.item()))
        marginal_losses.append(float(marginal_loss.item()))
        conditional_losses.append(float(conditional_loss.item()))
        source_preds.extend(torch.argmax(source_logits, dim=1).detach().cpu().numpy().tolist())
        source_true.extend(source_y.detach().cpu().numpy().tolist())

    source_acc = float(np.mean(np.asarray(source_preds) == np.asarray(source_true))) if source_true else 0.0
    val_loss, val_acc = evaluate_source_validation(model, val_loader, class_criterion, device)
    return TrainStats(
        total_loss=float(np.mean(total_losses)) if total_losses else 0.0,
        cls_loss=float(np.mean(cls_losses)) if cls_losses else 0.0,
        marginal_loss=float(np.mean(marginal_losses)) if marginal_losses else 0.0,
        conditional_loss=float(np.mean(conditional_losses)) if conditional_losses else 0.0,
        source_acc=source_acc,
        val_loss=val_loss,
        val_acc=val_acc,
    )


def main() -> None:
    """Parse arguments, train, and evaluate the EEG-DG style model."""

    parser = argparse.ArgumentParser(description="Minimal EEG-DG style dual-stream alignment experiment")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/eegdg_dual_stream")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--use-ea", action="store_true")
    parser.add_argument("--linear-probe", action="store_true")
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--lambda-marginal", type=float, default=0.3)
    parser.add_argument("--lambda-conditional", type=float, default=0.3)
    parser.add_argument("--pseudo-confidence", type=float, default=0.6)
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--threshold-objective", choices=["balanced_acc", "f1", "youden"], default="balanced_acc")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_source, y_source, sub_ids_source = load_domain(Path(args.source_dir))
    x_target, y_target, sub_ids_target = load_domain(Path(args.target_dir))
    train_mask, val_mask = split_source_subjects(y_source, sub_ids_source, args.val_ratio, args.seed)
    x_train, y_train = x_source[train_mask], y_source[train_mask]
    x_val, y_val = x_source[val_mask], y_source[val_mask]

    if args.use_ea:
        r_inv_half = fit_euclidean_alignment(x_train)
        x_train = apply_euclidean_alignment(x_train, r_inv_half)
        x_val = apply_euclidean_alignment(x_val, r_inv_half)
        x_target = apply_euclidean_alignment(x_target, r_inv_half)
        np.save(output_dir / "ea_matrix.npy", r_inv_half)

    source_train_loader = DataLoader(EEGDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True, drop_last=False)
    source_val_loader = DataLoader(EEGDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
    target_loader = DataLoader(EEGDataset(x_target, None), batch_size=args.batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(max(y_source.max(), y_target.max()) + 1)
    model = build_model(
        num_classes=num_classes,
        num_channels=x_source.shape[1],
        checkpoint=args.checkpoint,
        linear_probe=args.linear_probe,
        device=device,
    )

    class_weight = None
    if not args.disable_class_weights:
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * np.maximum(class_counts, 1))
        class_weight = torch.tensor(class_weights, dtype=torch.float32, device=device)

    class_criterion = nn.CrossEntropyLoss(weight=class_weight)
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
        stats = train_one_epoch(
            model=model,
            source_loader=source_train_loader,
            target_loader=target_loader,
            val_loader=source_val_loader,
            class_criterion=class_criterion,
            optimizer=optimizer,
            device=device,
            lambda_marginal=args.lambda_marginal,
            lambda_conditional=args.lambda_conditional,
            confidence_threshold=args.pseudo_confidence,
            num_classes=num_classes,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_total_loss": stats.total_loss,
                "train_cls_loss": stats.cls_loss,
                "train_marginal_loss": stats.marginal_loss,
                "train_conditional_loss": stats.conditional_loss,
                "train_source_acc": stats.source_acc,
                "val_loss": stats.val_loss,
                "val_acc": stats.val_acc,
            }
        )
        print(
            f"Epoch {epoch:03d} | total={stats.total_loss:.4f} cls={stats.cls_loss:.4f} "
            f"marg={stats.marginal_loss:.4f} cond={stats.conditional_loss:.4f} "
            f"src_acc={stats.source_acc:.4f} | val_loss={stats.val_loss:.4f} val_acc={stats.val_acc:.4f}"
        )

        if stats.val_loss < best_val_loss:
            best_val_loss = stats.val_loss
            best_epoch = epoch
            patience_left = args.patience
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(output_dir / "best_model.pth", map_location=device))
    validation_threshold = None
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
        validation_threshold_metrics = compute_subject_metrics(
            subject_ids_unique=val_subject_ids,
            y_true=val_subject_true,
            y_prob=val_subject_prob,
            threshold=validation_threshold,
        )
    else:
        validation_threshold_metrics = None

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
        "source_dir": args.source_dir,
        "target_dir": args.target_dir,
        "checkpoint": args.checkpoint,
        "use_ea": args.use_ea,
        "linear_probe": args.linear_probe,
        "disable_class_weights": args.disable_class_weights,
        "lambda_marginal": args.lambda_marginal,
        "lambda_conditional": args.lambda_conditional,
        "pseudo_confidence": args.pseudo_confidence,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "decision_threshold": validation_threshold,
        "threshold_objective": args.threshold_objective if args.tune_threshold else None,
        "threshold_objective_score": threshold_score,
        "validation_threshold_metrics": validation_threshold_metrics,
        "history": history,
        "target_metrics": target_metrics,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    print("\nEEG-DG style target-domain subject-level results")
    print(f"Accuracy     : {target_metrics['accuracy']:.4f}")
    print(f"Balanced Acc : {target_metrics['balanced_accuracy']:.4f}")
    print(f"F1           : {target_metrics['f1']:.4f}")
    print(f"AUC          : {target_metrics['auc']:.4f}")
    print(f"Sensitivity  : {target_metrics['sensitivity']:.4f}")
    print(f"Specificity  : {target_metrics['specificity']:.4f}")
    if validation_threshold is not None:
        print(f"Threshold    : {validation_threshold:.4f}")
    print(target_metrics["report"])
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
