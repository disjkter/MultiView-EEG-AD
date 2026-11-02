"""Run domain-adversarial cross-domain training with the dual-stream EEG model."""

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
from torch.autograd import Function
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
from utils.dual_stream_config import add_config_argument, apply_yaml_defaults


def set_seed(seed: int) -> None:
    """Set global random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_domain_files(domain_dir: Path) -> tuple[Path, Path, Path]:
    """Resolve processed EEG domain files."""

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
    """EEG dataset with per-sample channel normalization."""

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray | None = None):
        self.x = x_data.astype(np.float32, copy=True)
        self.y = None if y_data is None else y_data.astype(np.int64, copy=False)
        for i in range(len(self.x)):
            mean = self.x[i].mean(axis=1, keepdims=True)
            std = self.x[i].std(axis=1, keepdims=True) + 1e-6
            self.x[i] = (self.x[i] - mean) / std

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = torch.from_numpy(self.x[idx])
        if self.y is None:
            return x, torch.tensor(-1, dtype=torch.long)
        return x, torch.tensor(self.y[idx], dtype=torch.long)


class GradientReversalFunction(Function):
    """Identity in forward, negative scaling in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_grl: float) -> torch.Tensor:
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_grl * grad_output, None


class GradientReversal(nn.Module):
    """Gradient reversal layer for DANN-style training."""

    def forward(self, x: torch.Tensor, lambda_grl: float) -> torch.Tensor:
        return GradientReversalFunction.apply(x, lambda_grl)


class DomainAdversarialDualStream(nn.Module):
    """Dual-stream classifier plus domain head."""

    def __init__(self, num_classes: int, num_channels: int, checkpoint: str | None, linear_probe: bool):
        super().__init__()
        self.backbone = SpatialMSCNN_DualStream(num_classes=num_classes, num_channels=num_channels)
        if checkpoint:
            state_dict = torch.load(checkpoint, map_location="cpu")
            self.backbone.load_state_dict(state_dict, strict=False)
        self.feature_dim = self.backbone.fusion_classifier[0].in_features
        self.grl = GradientReversal()
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.BatchNorm1d(self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim // 2, 2),
        )

        if linear_probe:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.fusion_classifier.parameters():
                param.requires_grad = True
            for param in self.domain_classifier.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor, lambda_grl: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        class_logits, features = self.backbone(x)
        domain_logits = self.domain_classifier(self.grl(features, lambda_grl))
        return class_logits, domain_logits, features


@dataclass
class TrainStats:
    """Training epoch statistics."""

    total_loss: float
    cls_loss: float
    domain_loss: float
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


def compute_lambda_grl(epoch_idx: int, total_epochs: int) -> float:
    """DANN schedule from 0 to 1."""

    progress = float(epoch_idx) / max(total_epochs - 1, 1)
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


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
            class_logits, _, _ = model(batch_x, lambda_grl=0.0)
            loss = criterion(class_logits, batch_y)
            losses.append(float(loss.item()))
            all_preds.extend(torch.argmax(class_logits, dim=1).cpu().numpy().tolist())
            all_targets.extend(batch_y.cpu().numpy().tolist())
    accuracy = float(np.mean(np.asarray(all_preds) == np.asarray(all_targets))) if all_targets else 0.0
    return float(np.mean(losses)) if losses else 0.0, accuracy


def train_one_epoch(
    model: DomainAdversarialDualStream,
    source_loader: DataLoader,
    target_loader: DataLoader,
    val_loader: DataLoader,
    class_criterion: nn.Module,
    domain_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_domain: float,
) -> TrainStats:
    """Train one DANN epoch."""

    model.train()
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    num_steps = max(len(source_loader), len(target_loader))

    total_losses: list[float] = []
    cls_losses: list[float] = []
    dom_losses: list[float] = []
    source_preds: list[int] = []
    source_true: list[int] = []

    for _ in range(num_steps):
        (source_x, source_y), source_iter = next_batch(source_iter, source_loader)
        (target_x, _), target_iter = next_batch(target_iter, target_loader)

        source_x = source_x.to(device)
        source_y = source_y.to(device)
        target_x = target_x.to(device)

        src_class_logits, src_domain_logits, _ = model(source_x, lambda_grl=lambda_domain)
        _, tgt_domain_logits, _ = model(target_x, lambda_grl=lambda_domain)

        src_domain_labels = torch.zeros(source_x.shape[0], dtype=torch.long, device=device)
        tgt_domain_labels = torch.ones(target_x.shape[0], dtype=torch.long, device=device)

        cls_loss = class_criterion(src_class_logits, source_y)
        domain_loss = 0.5 * (
            domain_criterion(src_domain_logits, src_domain_labels)
            + domain_criterion(tgt_domain_logits, tgt_domain_labels)
        )
        loss = cls_loss + lambda_domain * domain_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_losses.append(float(loss.item()))
        cls_losses.append(float(cls_loss.item()))
        dom_losses.append(float(domain_loss.item()))
        source_preds.extend(torch.argmax(src_class_logits, dim=1).detach().cpu().numpy().tolist())
        source_true.extend(source_y.detach().cpu().numpy().tolist())

    source_acc = float(np.mean(np.asarray(source_preds) == np.asarray(source_true))) if source_true else 0.0
    val_loss, val_acc = evaluate_source_validation(model, val_loader, class_criterion, device)
    return TrainStats(
        total_loss=float(np.mean(total_losses)) if total_losses else 0.0,
        cls_loss=float(np.mean(cls_losses)) if cls_losses else 0.0,
        domain_loss=float(np.mean(dom_losses)) if dom_losses else 0.0,
        source_acc=source_acc,
        val_loss=val_loss,
        val_acc=val_acc,
    )


def main() -> None:
    """Train and evaluate a domain-adversarial dual-stream model."""

    parser = argparse.ArgumentParser(description="Domain-adversarial dual-stream EEG experiment")
    add_config_argument(parser)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/domain_adversarial_dual_stream")
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
    parser.add_argument("--lambda-domain-max", type=float, default=0.5)
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--threshold-objective", choices=["balanced_acc", "f1", "youden"], default="balanced_acc")
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
            "lambda_domain_max",
            "tune_threshold",
            "threshold_objective",
        ],
    )

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
    model = DomainAdversarialDualStream(
        num_classes=int(max(y_source.max(), y_target.max()) + 1),
        num_channels=x_source.shape[1],
        checkpoint=args.checkpoint,
        linear_probe=args.linear_probe,
    ).to(device)

    class_weight = None
    if not args.disable_class_weights:
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * np.maximum(class_counts, 1))
        class_weight = torch.tensor(class_weights, dtype=torch.float32, device=device)

    class_criterion = nn.CrossEntropyLoss(weight=class_weight)
    domain_criterion = nn.CrossEntropyLoss()
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
        lambda_domain = args.lambda_domain_max * compute_lambda_grl(epoch - 1, args.epochs)
        stats = train_one_epoch(
            model=model,
            source_loader=source_train_loader,
            target_loader=target_loader,
            val_loader=source_val_loader,
            class_criterion=class_criterion,
            domain_criterion=domain_criterion,
            optimizer=optimizer,
            device=device,
            lambda_domain=lambda_domain,
        )
        history.append(
            {
                "epoch": float(epoch),
                "lambda_domain": float(lambda_domain),
                "train_total_loss": stats.total_loss,
                "train_cls_loss": stats.cls_loss,
                "train_domain_loss": stats.domain_loss,
                "train_source_acc": stats.source_acc,
                "val_loss": stats.val_loss,
                "val_acc": stats.val_acc,
            }
        )
        print(
            f"Epoch {epoch:03d} | lambda={lambda_domain:.4f} | "
            f"train_total={stats.total_loss:.4f} cls={stats.cls_loss:.4f} dom={stats.domain_loss:.4f} "
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
            model=model.backbone,
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
        model=model.backbone,
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
        "lambda_domain_max": args.lambda_domain_max,
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

    print("\nDomain-adversarial target-domain subject-level results")
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
