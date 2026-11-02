"""Run current-version TAGE ablation experiments with subject-level hold-out.

This runner replaces the early ``biofoundation_v1_best.pth`` teacher checkpoint
with the current self-supervised TAGE checkpoint used by the paper pipeline.
It uses the same subject-level train/val/test protocol as the current main
experiments, then compares ablated model variants on the exact same split.

Examples
--------
Quick smoke run on a few subjects:

    python scripts/run_current_ablation_batch.py --max-subjects 5 --epochs 1

Paper-oriented single-seed batch:

    python scripts/run_current_ablation_batch.py

Use a different current pretraining checkpoint:

    python scripts/run_current_ablation_batch.py \
        --checkpoint checkpoints/tage_timefreq_consistency_bl19_40hz.pth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ablation_model import ABLATION_VARIANTS, build_ablation_model  # noqa: E402
from scripts.run_dual_stream_loso_subject import choose_threshold, safe_auc, set_seed  # noqa: E402
from scripts.run_dual_stream_subject_split import (  # noqa: E402
    IndexedEEGDataset,
    collect_subject_probabilities,
    compute_metrics,
    fit_euclidean_alignment,
    load_domain_mmap,
    sample_indices_per_subject,
    stratified_subject_split,
    subject_labels,
    run_epoch,
)
from utils.subject_split import load_subject_split_masks  # noqa: E402


DEFAULT_CHECKPOINT = "checkpoints/tage_timefreq_consistency_adssl_1s_lead.pth"
DEFAULT_DATASETS = [
    "CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
    "ds004504_processed_19ch_1s_lead",
]
DEFAULT_VARIANTS = [
    "full",
    "no_spatial",
    "no_mscnn",
    "no_multiscale",
    "no_se",
    "cls_1layer",
    "cls_3layer",
]
DEFAULT_SEEDS = [2026]


def timestamp() -> str:
    """Return a compact local timestamp for progress logs."""

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def resolve_project_path(path_text: str | None) -> Path | None:
    """Resolve a CLI path relative to the project root."""

    if path_text is None or path_text == "":
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def write_log(log_path: Path, message: str) -> None:
    """Print and append one progress line."""

    line = f"[{timestamp()}] {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for current ablation experiments."""

    parser = argparse.ArgumentParser(description="Current TAGE ablation batch runner")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--no-checkpoint", action="store_true", help="Run random-init architecture ablation.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(ABLATION_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-root", default="outputs/current_ablation")
    parser.add_argument(
        "--split-root",
        default=None,
        help=(
            "Directory containing <dataset>_seed<seed>.json split manifests. "
            "When supplied, every variant reuses the exact saved participants."
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--use-ea", action="store_true")
    parser.add_argument("--max-ea-samples", type=int, default=5000)
    parser.add_argument("--max-train-segments-per-subject", type=int, default=250)
    parser.add_argument("--max-val-segments-per-subject", type=int, default=250)
    parser.add_argument("--max-test-segments-per-subject", type=int, default=400)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-threshold-tune", action="store_true")
    parser.add_argument("--threshold-objective", default="balanced_acc")
    parser.add_argument("--selection-metric", choices=["loss", "auc"], default="auc")
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs without training.")
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> list[dict[str, object]]:
    """Build the dataset x variant x seed job list."""

    jobs: list[dict[str, object]] = []
    for dataset in args.datasets:
        data_path = PROJECT_ROOT / dataset
        if not data_path.exists():
            print(f"[SKIP] Dataset not found: {dataset}", flush=True)
            continue
        for variant in args.variants:
            for seed in args.seeds:
                jobs.append({"dataset": dataset, "variant": variant, "seed": int(seed)})
    return jobs


def count_subjects_local(y: np.ndarray, sub_ids: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    """Count unique subjects per class for a selected split."""

    labels: dict[object, int] = {}
    for subject in np.unique(sub_ids[indices]):
        subject_indices = indices[sub_ids[indices] == subject]
        labels[subject] = int(y[subject_indices[0]])
    return {
        str(label): int(sum(value == label for value in labels.values()))
        for label in sorted(set(labels.values()))
    }


def require_two_classes_local(split_name: str, y: np.ndarray, sub_ids: np.ndarray, indices: np.ndarray) -> None:
    """Fail fast when a subject-level split contains fewer than two classes."""

    counts = count_subjects_local(y, sub_ids, indices)
    if len(counts) < 2:
        raise ValueError(
            f"{split_name} split contains fewer than two subject classes: {counts}. "
            "Increase --max-subjects or use a stratified subject-level split."
        )


def select_subject_subset(y: np.ndarray, sub_ids: np.ndarray, max_subjects: int | None, seed: int) -> np.ndarray:
    """Select a stratified subject subset for smoke tests.

    The full paper run should leave ``max_subjects`` unset. When capped, this
    keeps both classes represented instead of taking the first N subject ids.
    """

    unique_subjects = np.asarray(list(subject_labels(y, sub_ids).keys()), dtype=object)
    if max_subjects is None or max_subjects <= 0 or len(unique_subjects) <= max_subjects:
        return np.ones(len(sub_ids), dtype=bool)

    labels = subject_labels(y, sub_ids)
    rng = np.random.default_rng(seed)
    selected: list[object] = []
    labels_present = sorted(set(labels.values()))
    per_class = max(1, max_subjects // max(1, len(labels_present)))

    for label in labels_present:
        candidates = np.asarray([s for s, value in labels.items() if value == label], dtype=object)
        rng.shuffle(candidates)
        selected.extend(candidates[:per_class].tolist())

    remaining_slots = max_subjects - len(selected)
    if remaining_slots > 0:
        remaining = np.asarray([s for s in unique_subjects if s not in set(selected)], dtype=object)
        rng.shuffle(remaining)
        selected.extend(remaining[:remaining_slots].tolist())

    return np.isin(sub_ids, np.asarray(selected, dtype=object))


def run_holdout_ablation(
    *,
    variant: str,
    data_dir: str,
    checkpoint: str | None,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    train_ratio: float,
    val_ratio: float,
    patience: int,
    use_ea: bool,
    max_ea_samples: int,
    max_train_segments_per_subject: int | None,
    max_val_segments_per_subject: int | None,
    max_test_segments_per_subject: int | None,
    use_class_weights: bool,
    tune_threshold: bool,
    threshold_objective: str,
    selection_metric: str,
    device: torch.device,
    max_subjects: int | None,
    split_file: str | None,
) -> dict[str, object]:
    """Train one ablation variant with subject-level hold-out evaluation.

    Tensor shapes:
    - EEG segments: [B, C, T]
    - logits: [B, num_classes]
    - subject probabilities: [num_subjects, num_classes]
    """

    x_all, y_all, sub_ids_all = load_domain_mmap(PROJECT_ROOT / data_dir)
    x_view = x_all
    y_view = y_all
    sub_view = sub_ids_all
    if split_file:
        if max_subjects is not None and max_subjects > 0:
            raise ValueError("--max-subjects cannot be combined with --split-root.")
        train_mask, val_mask, test_mask = load_subject_split_masks(split_file, sub_view)
        selected_mask = train_mask | val_mask | test_mask
        selected_sub_ids = sub_view[selected_mask]
        selected_y = y_view[selected_mask]
        train_indices = np.flatnonzero(train_mask)
        val_indices = np.flatnonzero(val_mask)
        test_indices = np.flatnonzero(test_mask)
    else:
        subject_mask = select_subject_subset(y_all, sub_ids_all, max_subjects, seed)
        selected_indices = np.where(subject_mask)[0]
        if len(selected_indices) == 0:
            raise ValueError(f"No subjects selected for {data_dir}.")
        train_mask, val_mask, test_mask = stratified_subject_split(
            y_view[subject_mask],
            sub_view[subject_mask],
            train_ratio,
            val_ratio,
            seed,
        )
        selected_sub_ids = sub_view[subject_mask]
        selected_y = y_view[subject_mask]
        local_to_global = selected_indices
        train_indices = local_to_global[np.where(train_mask)[0]]
        val_indices = local_to_global[np.where(val_mask)[0]]
        test_indices = local_to_global[np.where(test_mask)[0]]

    train_indices = sample_indices_per_subject(
        train_indices, sub_view, max_train_segments_per_subject, seed + 11,
    )
    val_indices = sample_indices_per_subject(
        val_indices, sub_view, max_val_segments_per_subject, seed + 17,
    )
    test_indices = sample_indices_per_subject(
        test_indices, sub_view, max_test_segments_per_subject, seed + 23,
    )

    require_two_classes_local("train", y_view, sub_view, train_indices)
    require_two_classes_local("val", y_view, sub_view, val_indices)
    require_two_classes_local("test", y_view, sub_view, test_indices)

    ea_matrix = None
    if use_ea:
        ea_indices = train_indices
        if len(ea_indices) > max_ea_samples:
            rng = np.random.default_rng(seed + 31)
            ea_indices = rng.choice(ea_indices, size=max_ea_samples, replace=False)
        ea_matrix = fit_euclidean_alignment(np.asarray(x_view[ea_indices], dtype=np.float32))

    train_dataset = IndexedEEGDataset(x_view, y_view, sub_view, train_indices, ea_matrix=ea_matrix)
    val_dataset = IndexedEEGDataset(x_view, y_view, sub_view, val_indices, ea_matrix=ea_matrix)
    test_dataset = IndexedEEGDataset(x_view, y_view, sub_view, test_indices, ea_matrix=ea_matrix)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = build_ablation_model(
        variant=variant,
        num_classes=int(y_view.max() + 1),
        num_channels=int(x_view.shape[1]),
        checkpoint=checkpoint,
        device=device,
    )

    class_weight = None
    if use_class_weights:
        class_counts = np.bincount(y_view[train_indices], minlength=int(y_view.max() + 1))
        weights = len(train_indices) / (len(class_counts) * np.maximum(class_counts, 1))
        class_weight = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_state = None
    best_val_loss = float("inf")
    best_val_auc = -float("inf")
    best_epoch = 0
    patience_left = patience
    history: list[dict[str, float | int | None]] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        _, val_true_epoch, val_prob_epoch = collect_subject_probabilities(
            model, val_dataset, batch_size, device,
        )
        val_auc = safe_auc(val_true_epoch, val_prob_epoch)
        print(
            f"    Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}",
            flush=True,
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

        if selection_metric == "auc" and not np.isnan(val_auc):
            improved = val_auc > best_val_auc
        else:
            improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_epoch = epoch
            patience_left = patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("    Early stopping triggered.", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    threshold = None
    threshold_score = None
    if tune_threshold:
        _, val_true, val_prob = collect_subject_probabilities(model, val_dataset, batch_size, device)
        threshold, threshold_score = choose_threshold(val_true, val_prob, threshold_objective)

    test_subjects, test_true, test_prob = collect_subject_probabilities(
        model, test_dataset, batch_size, device,
    )
    metrics = compute_metrics(test_subjects, test_true, test_prob, threshold)

    split_summary = {
        "selected_subjects": int(len(np.unique(selected_sub_ids))),
        "selected_subjects_by_label": count_subjects_local(selected_y, selected_sub_ids, np.arange(len(selected_y))),
        "train_segments": int(len(train_indices)),
        "val_segments": int(len(val_indices)),
        "test_segments": int(len(test_indices)),
        "train_subjects_by_label": count_subjects_local(y_view, sub_view, train_indices),
        "val_subjects_by_label": count_subjects_local(y_view, sub_view, val_indices),
        "test_subjects_by_label": count_subjects_local(y_view, sub_view, test_indices),
    }

    return {
        "protocol": "stratified_subject_holdout",
        "variant": variant,
        "label": ABLATION_VARIANTS[variant]["label"],
        "data_dir": data_dir,
        "checkpoint": checkpoint,
        "split_file": split_file,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "use_ea": use_ea,
        "use_class_weights": use_class_weights,
        "tune_threshold": tune_threshold,
        "threshold_objective": threshold_objective if tune_threshold else None,
        "decision_threshold": threshold,
        "threshold_score": threshold_score,
        "selection_metric": selection_metric,
        "max_subjects": max_subjects,
        "max_train_segments_per_subject": max_train_segments_per_subject,
        "max_val_segments_per_subject": max_val_segments_per_subject,
        "max_test_segments_per_subject": max_test_segments_per_subject,
        "split": split_summary,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_auc": None if np.isnan(best_val_auc) else float(best_val_auc),
        "history": history,
        "metrics": metrics,
        "participant_predictions": {
            "subject_ids": [str(value) for value in test_subjects.tolist()],
            "y_true": [int(value) for value in test_true.tolist()],
            "probabilities": [[float(item) for item in row] for row in test_prob.tolist()],
        },
    }


def main() -> None:
    """Run the current-version ablation batch."""

    args = parse_args()
    output_root = resolve_project_path(args.output_root)
    assert output_root is not None
    log_path = output_root / "batch_progress.log"

    checkpoint_path = None if args.no_checkpoint else resolve_project_path(args.checkpoint)
    checkpoint_for_runner = None if checkpoint_path is None else str(checkpoint_path)
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Current ablation checkpoint not found: {checkpoint_path}\n"
            "Either copy the checkpoint into this project, pass --checkpoint, or use --no-checkpoint."
        )

    jobs = build_jobs(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_log(log_path, f"Starting current ablation: {len(jobs)} jobs on {device}")
    write_log(log_path, f"Checkpoint: {checkpoint_for_runner or 'random initialization'}")
    write_log(log_path, f"Datasets: {', '.join(args.datasets)}")
    write_log(log_path, f"Variants: {', '.join(args.variants)}")
    write_log(log_path, f"Seeds: {', '.join(str(s) for s in args.seeds)}")
    write_log(log_path, f"Split root: {args.split_root or 'generated deterministically per seed'}")
    write_log(log_path, f"Protocol: stratified subject-level hold-out train={args.train_ratio}, val={args.val_ratio}")
    write_log(log_path, "=" * 60)

    if args.dry_run:
        for idx, job in enumerate(jobs, start=1):
            write_log(log_path, f"[{idx}/{len(jobs)}] PLAN: {job['dataset']} / {job['variant']} / seed={job['seed']}")
        return

    done = 0
    failed = 0
    start = time.time()

    for idx, job in enumerate(jobs, start=1):
        dataset = str(job["dataset"])
        variant = str(job["variant"])
        seed = int(job["seed"])
        split_file = None
        if args.split_root:
            split_root = resolve_project_path(args.split_root)
            assert split_root is not None
            split_path = split_root / f"{dataset}_seed{seed}.json"
            if not split_path.exists():
                raise FileNotFoundError(f"Fixed split not found: {split_path}")
            split_file = str(split_path)
        tag = f"{variant}_s{seed}"
        if args.max_subjects:
            tag += f"_n{args.max_subjects}"
        exp_dir = output_root / dataset / tag
        summary_path = exp_dir / "summary.json"
        error_path = exp_dir / "error.json"
        exp_dir.mkdir(parents=True, exist_ok=True)

        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as f:
                metrics = json.load(f)["metrics"]
            write_log(
                log_path,
                f"[{idx}/{len(jobs)}] SKIP: {dataset}/{tag} "
                f"BalAcc={metrics['balanced_accuracy']:.4f} F1={metrics['f1']:.4f}",
            )
            done += 1
            continue

        job_start = time.time()
        write_log(log_path, f"[{idx}/{len(jobs)}] START: {dataset}/{tag}")
        try:
            set_seed(seed)
            result = run_holdout_ablation(
                variant=variant,
                data_dir=dataset,
                checkpoint=checkpoint_for_runner,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                patience=args.patience,
                use_ea=args.use_ea,
                max_ea_samples=args.max_ea_samples,
                max_train_segments_per_subject=args.max_train_segments_per_subject,
                max_val_segments_per_subject=args.max_val_segments_per_subject,
                max_test_segments_per_subject=args.max_test_segments_per_subject,
                use_class_weights=not args.no_class_weights,
                tune_threshold=not args.no_threshold_tune,
                threshold_objective=args.threshold_objective,
                selection_metric=args.selection_metric,
                device=device,
                max_subjects=args.max_subjects,
                split_file=split_file,
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            metrics = result["metrics"]
            done += 1
            write_log(
                log_path,
                f"[{idx}/{len(jobs)}] DONE: {dataset}/{tag} "
                f"t={time.time() - job_start:.0f}s "
                f"BalAcc={metrics['balanced_accuracy']:.4f} "
                f"F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f} "
                f"Sens={metrics['sensitivity']:.4f} Spec={metrics['specificity']:.4f}",
            )
        except Exception as exc:
            failed += 1
            with open(error_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "checkpoint": checkpoint_for_runner,
                        "split_file": split_file,
                        "error": str(exc),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            write_log(
                log_path,
                f"[{idx}/{len(jobs)}] FAIL: {dataset}/{tag} "
                f"t={time.time() - job_start:.0f}s {exc}",
            )

    write_log(log_path, "=" * 60)
    write_log(log_path, f"FINISHED: done={done}, failed={failed}, elapsed={(time.time() - start) / 3600:.2f}h")


if __name__ == "__main__":
    main()
