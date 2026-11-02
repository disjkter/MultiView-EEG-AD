"""Run multi-source LODO with a minimal EEG-DG style alignment strategy."""

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
from scripts.run_dual_stream_lodo import load_domain, match_time_length


def set_seed(seed: int) -> None:
    """Set reproducible random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EEGDataset(Dataset):
    """EEG dataset that returns data, labels, and domain ids."""

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, domain_ids: np.ndarray):
        self.x = x_data.astype(np.float32, copy=True)
        self.y = y_data.astype(np.int64, copy=False)
        self.domain_ids = domain_ids.astype(np.int64, copy=False)
        for i in range(len(self.x)):
            mean = self.x[i].mean(axis=1, keepdims=True)
            std = self.x[i].std(axis=1, keepdims=True) + 1e-6
            self.x[i] = (self.x[i] - mean) / std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.x[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
            torch.tensor(self.domain_ids[idx], dtype=torch.long),
        )


@dataclass
class TrainStats:
    """One epoch of multi-source alignment stats."""

    total_loss: float
    cls_loss: float
    marginal_loss: float
    conditional_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


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
    """Multi-kernel MMD loss."""

    kernels = gaussian_kernel(source_feat, target_feat)
    batch_size = source_feat.shape[0]
    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]
    return xx.mean() + yy.mean() - xy.mean() - yx.mean()


def pairwise_multi_source_marginal_loss(features: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
    """Average marginal alignment across source domains."""

    unique_domains = torch.unique(domain_ids)
    losses: list[torch.Tensor] = []
    for i in range(len(unique_domains)):
        for j in range(i + 1, len(unique_domains)):
            mask_i = domain_ids == unique_domains[i]
            mask_j = domain_ids == unique_domains[j]
            if int(mask_i.sum()) < 2 or int(mask_j.sum()) < 2:
                continue
            losses.append(marginal_mmd_loss(features[mask_i], features[mask_j]))
    if not losses:
        return features.new_tensor(0.0)
    return torch.stack(losses).mean()


def pairwise_multi_source_conditional_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    domain_ids: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Align per-class feature centers across source domains."""

    unique_domains = torch.unique(domain_ids)
    losses: list[torch.Tensor] = []
    for class_idx in range(num_classes):
        for i in range(len(unique_domains)):
            for j in range(i + 1, len(unique_domains)):
                mask_i = (domain_ids == unique_domains[i]) & (labels == class_idx)
                mask_j = (domain_ids == unique_domains[j]) & (labels == class_idx)
                if int(mask_i.sum()) == 0 or int(mask_j.sum()) == 0:
                    continue
                center_i = features[mask_i].mean(dim=0)
                center_j = features[mask_j].mean(dim=0)
                losses.append(F.mse_loss(center_i, center_j))
    if not losses:
        return features.new_tensor(0.0)
    return torch.stack(losses).mean()


def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate validation loss and accuracy."""

    model.eval()
    losses: list[float] = []
    preds: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for batch_x, batch_y, _ in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            losses.append(float(loss.item()))
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            targets.extend(batch_y.cpu().numpy().tolist())
    acc = float(np.mean(np.asarray(preds) == np.asarray(targets))) if targets else 0.0
    return float(np.mean(losses)) if losses else 0.0, acc


def main() -> None:
    """Run multi-source LODO with EEG-DG style source alignment."""

    parser = argparse.ArgumentParser(description="Multi-source LODO EEG-DG style experiment")
    parser.add_argument("--domains", nargs="+", required=True, help="Format: name=path")
    parser.add_argument("--target-domain", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/lodo_eegdg")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--use-ea", action="store_true")
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--lambda-marginal", type=float, default=0.3)
    parser.add_argument("--lambda-conditional", type=float, default=0.3)
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--threshold-objective", choices=["balanced_acc", "f1", "youden"], default="balanced_acc")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_specs: list[tuple[str, Path]] = []
    for item in args.domains:
        if "=" not in item:
            raise ValueError(f"Invalid domain spec: {item}. Use name=path.")
        name, path_str = item.split("=", 1)
        domain_specs.append((name, Path(path_str)))

    target_spec = None
    source_specs: list[tuple[str, Path]] = []
    for name, path in domain_specs:
        if name == args.target_domain:
            target_spec = (name, path)
        else:
            source_specs.append((name, path))

    if target_spec is None:
        raise ValueError(f"Target domain {args.target_domain} not found in domain specs.")

    source_loaded = []
    target_x, target_y, target_sub_ids = load_domain(target_spec[0], target_spec[1])
    source_lengths = [int(target_x.shape[-1])]
    for domain_idx, (domain_name, domain_path) in enumerate(source_specs):
        x, y, sub_ids = load_domain(domain_name, domain_path)
        source_loaded.append((domain_idx, domain_name, x, y, sub_ids))
        source_lengths.append(int(x.shape[-1]))

    common_length = min(source_lengths)
    target_x = match_time_length(target_x, common_length)
    source_x_parts: list[np.ndarray] = []
    source_y_parts: list[np.ndarray] = []
    source_sub_parts: list[np.ndarray] = []
    source_domain_parts: list[np.ndarray] = []

    for domain_idx, domain_name, x, y, sub_ids in source_loaded:
        x = match_time_length(x, common_length)
        source_x_parts.append(x)
        source_y_parts.append(y)
        source_sub_parts.append(np.asarray([f"{domain_name}::{sid}" for sid in sub_ids], dtype=object))
        source_domain_parts.append(np.full(shape=(x.shape[0],), fill_value=domain_idx, dtype=np.int64))

    x_source = np.concatenate(source_x_parts, axis=0)
    y_source = np.concatenate(source_y_parts, axis=0)
    sub_ids_source = np.concatenate(source_sub_parts, axis=0)
    domain_ids_source = np.concatenate(source_domain_parts, axis=0)

    train_mask, val_mask = split_source_subjects(y_source, sub_ids_source, args.val_ratio, args.seed)
    x_train, y_train, d_train = x_source[train_mask], y_source[train_mask], domain_ids_source[train_mask]
    x_val, y_val, d_val = x_source[val_mask], y_source[val_mask], domain_ids_source[val_mask]

    if args.use_ea:
        r_inv_half = fit_euclidean_alignment(x_train)
        x_train = apply_euclidean_alignment(x_train, r_inv_half)
        x_val = apply_euclidean_alignment(x_val, r_inv_half)
        target_x = apply_euclidean_alignment(target_x, r_inv_half)
        np.save(output_dir / "ea_matrix.npy", r_inv_half)

    train_loader = DataLoader(EEGDataset(x_train, y_train, d_train), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(EEGDataset(x_val, y_val, d_val), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(max(y_source.max(), target_y.max()) + 1)
    model = SpatialMSCNN_DualStream(num_classes=num_classes, num_channels=x_source.shape[1]).to(device)
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    class_weight = None
    if not args.disable_class_weights:
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * np.maximum(class_counts, 1))
        class_weight = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_losses: list[float] = []
        cls_losses: list[float] = []
        marginal_losses: list[float] = []
        conditional_losses: list[float] = []
        train_preds: list[int] = []
        train_true: list[int] = []

        for batch_x, batch_y, batch_domain in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_domain = batch_domain.to(device)

            logits, features = model(batch_x)
            cls_loss = criterion(logits, batch_y)
            marginal_loss = pairwise_multi_source_marginal_loss(features, batch_domain)
            conditional_loss = pairwise_multi_source_conditional_loss(
                features=features,
                labels=batch_y,
                domain_ids=batch_domain,
                num_classes=num_classes,
            )
            loss = cls_loss + args.lambda_marginal * marginal_loss + args.lambda_conditional * conditional_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_losses.append(float(loss.item()))
            cls_losses.append(float(cls_loss.item()))
            marginal_losses.append(float(marginal_loss.item()))
            conditional_losses.append(float(conditional_loss.item()))
            train_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
            train_true.extend(batch_y.detach().cpu().numpy().tolist())

        train_acc = float(np.mean(np.asarray(train_preds) == np.asarray(train_true))) if train_true else 0.0
        val_loss, val_acc = evaluate_validation(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
                "train_cls_loss": float(np.mean(cls_losses)) if cls_losses else 0.0,
                "train_marginal_loss": float(np.mean(marginal_losses)) if marginal_losses else 0.0,
                "train_conditional_loss": float(np.mean(conditional_losses)) if conditional_losses else 0.0,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"Epoch {epoch:03d} | total={history[-1]['train_total_loss']:.4f} "
            f"cls={history[-1]['train_cls_loss']:.4f} marg={history[-1]['train_marginal_loss']:.4f} "
            f"cond={history[-1]['train_conditional_loss']:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
        x_data=target_x,
        y_data=target_y,
        sub_ids=np.asarray([f"{target_spec[0]}::{sid}" for sid in target_sub_ids], dtype=object),
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
        "target_domain": args.target_domain,
        "source_domains": [name for name, _ in source_specs],
        "checkpoint": args.checkpoint,
        "use_ea": args.use_ea,
        "disable_class_weights": args.disable_class_weights,
        "lambda_marginal": args.lambda_marginal,
        "lambda_conditional": args.lambda_conditional,
        "seed": args.seed,
        "common_time_length": common_length,
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

    print("\nMulti-source EEG-DG style LODO target-domain results")
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
