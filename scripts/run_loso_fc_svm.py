"""Run a functional-connectivity SVM LOSO baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import fractional_matrix_power
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_loso_baselines import (  # noqa: E402
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


def build_fc_features(x_data: np.ndarray) -> np.ndarray:
    """Extract upper-triangular Pearson FC features per trial."""

    n_trials, n_channels, _ = x_data.shape
    tri_i, tri_j = np.triu_indices(n_channels, k=1)
    features = np.zeros((n_trials, len(tri_i)), dtype=np.float32)

    for idx in range(n_trials):
        trial = x_data[idx].astype(np.float64, copy=False)
        trial = trial - trial.mean(axis=1, keepdims=True)
        denom = trial.std(axis=1, keepdims=True) + 1e-6
        trial = trial / denom
        corr = np.corrcoef(trial)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = np.clip(corr, -1.0, 1.0)
        features[idx] = corr[tri_i, tri_j].astype(np.float32)

    return features


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
    model: Pipeline,
    x_data: np.ndarray,
    y_data: np.ndarray,
    sub_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict subject-level probabilities."""

    x_fc = build_fc_features(x_data)
    prob_all = model.predict_proba(x_fc).astype(np.float32)
    return aggregate_subject_probabilities(sub_ids, y_data, prob_all)


def build_fc_svm(seed: int) -> Pipeline:
    """Build the FC-SVM pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.98, svd_solver="full", random_state=seed)),
            ("svc", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description="Run LOSO FC+SVM baseline.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--use-ea", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--threshold-objective", type=str, choices=["balanced_acc", "f1"], default="balanced_acc")
    parser.add_argument("--max-subjects", type=int, default=None)
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
    fold_summaries: list[dict[str, object]] = []
    subject_true_all: list[int] = []
    subject_pred_all: list[int] = []
    subject_prob_all: list[np.ndarray] = []
    subject_ids_eval: list[object] = []

    for fold_index, test_subject in enumerate(unique_subjects, start=1):
        print(f"\nRunning LOSO FC-SVM fold {fold_index}/{len(unique_subjects)} | subject={test_subject}")
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

        model = build_fc_svm(args.seed + fold_index)
        model.fit(build_fc_features(x_train), y_train)
        _, val_true, val_prob = collect_subject_probabilities(model, x_val, y_val, sub_val)
        _, test_true, test_prob = collect_subject_probabilities(model, x_test, y_test, sub_test)

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
        "model": "fc_svm",
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
