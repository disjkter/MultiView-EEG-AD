"""Run single-source cross-domain baselines on processed EEG datasets."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_loso_baselines import (
    EEGDataset,
    EEGNet,
    ITransformer,
    MTDNet,
    ShallowConvNet,
    TCN,
    apply_euclidean_alignment,
    build_svm,
    choose_threshold,
    collect_subject_probabilities_svm,
    collect_subject_probabilities_torch,
    compute_metrics,
    fit_euclidean_alignment,
    load_domain,
    run_epoch,
    set_seed,
    split_subjects_for_validation,
)


def match_time_length(x: np.ndarray, target_length: int) -> np.ndarray:
    """Crop or zero-pad EEG samples to a shared temporal length."""

    current_length = x.shape[-1]
    if current_length == target_length:
        return x
    if current_length > target_length:
        return x[..., :target_length]

    padded = np.zeros((x.shape[0], x.shape[1], target_length), dtype=x.dtype)
    padded[..., :current_length] = x
    return padded


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description="Run single-source cross-domain baselines.")
    parser.add_argument("--source-dir", type=str, required=True)
    parser.add_argument("--target-dir", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        choices=["svm", "eegnet", "shallowconvnet", "mtdnet", "itransformer", "tcn"],
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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    """Run one source-to-target baseline experiment."""

    args = parse_args()
    set_seed(args.seed)

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_source, y_source, sub_source = load_domain(source_dir)
    x_target, y_target, sub_target = load_domain(target_dir)

    common_length = min(int(x_source.shape[-1]), int(x_target.shape[-1]))
    x_source = match_time_length(x_source, common_length)
    x_target = match_time_length(x_target, common_length)

    train_mask, val_mask = split_subjects_for_validation(
        y_source,
        sub_source,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    x_train = x_source[train_mask]
    y_train = y_source[train_mask]
    x_val = x_source[val_mask]
    y_val = y_source[val_mask]
    sub_val = sub_source[val_mask]

    if args.use_ea:
        r_inv_half = fit_euclidean_alignment(x_train)
        x_train = apply_euclidean_alignment(x_train, r_inv_half)
        x_val = apply_euclidean_alignment(x_val, r_inv_half)
        x_target = apply_euclidean_alignment(x_target, r_inv_half)

    device = torch.device(args.device)
    num_classes = int(len(np.unique(y_source)))
    num_channels = int(x_train.shape[1])

    if args.model == "svm":
        model = build_svm(args.seed)
        from scripts.run_loso_baselines import flatten_trials

        model.fit(flatten_trials(x_train), y_train)
        _, val_true, val_prob = collect_subject_probabilities_svm(model, x_val, y_val, sub_val)
        target_subject_ids, target_true, target_prob = collect_subject_probabilities_svm(model, x_target, y_target, sub_target)
    else:
        if args.model == "eegnet":
            model = EEGNet(
                num_channels=num_channels,
                num_classes=num_classes,
                dropout=args.dropout,
            ).to(device)
        elif args.model == "shallowconvnet":
            model = ShallowConvNet(
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
        else:
            model = TCN(
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

        _, val_true, val_prob = collect_subject_probabilities_torch(model, x_val, y_val, sub_val, args.batch_size, device)
        target_subject_ids, target_true, target_prob = collect_subject_probabilities_torch(
            model,
            x_target,
            y_target,
            sub_target,
            args.batch_size,
            device,
        )

    decision_threshold = 0.5
    if args.tune_threshold and num_classes == 2:
        decision_threshold, _ = choose_threshold(val_true, val_prob, args.threshold_objective)

    target_pred = (target_prob[:, 1] >= decision_threshold).astype(np.int64) if num_classes == 2 else np.argmax(target_prob, axis=1)
    target_metrics = compute_metrics(target_subject_ids, target_true, target_pred, target_prob)

    summary = {
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "model": args.model,
        "use_ea": bool(args.use_ea),
        "tune_threshold": bool(args.tune_threshold),
        "threshold_objective": args.threshold_objective if args.tune_threshold else None,
        "common_time_length": common_length,
        "target_metrics": target_metrics,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print("\nCross-domain target summary")
    print(f"Accuracy     : {target_metrics['accuracy']:.4f}")
    print(f"Balanced Acc : {target_metrics['balanced_accuracy']:.4f}")
    print(f"F1           : {target_metrics['f1']:.4f}")
    print(f"AUC          : {target_metrics['auc']:.4f}")
    print(f"Sensitivity  : {target_metrics['sensitivity']:.4f}")
    print(f"Specificity  : {target_metrics['specificity']:.4f}")
    print(target_metrics["report"])
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
