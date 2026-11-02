"""Sweep fixed subject-level thresholds for multi-seed LOSO summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def summarize(values: list[float]) -> dict[str, float]:
    """Return mean and population standard deviation."""

    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def compute_metrics(y_true: np.ndarray, y_prob_pos: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute binary subject-level metrics at one fixed threshold."""

    y_pred = (y_prob_pos >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob_pos)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def load_seed_metrics(root: Path, seed: int, threshold: float) -> dict[str, float]:
    """Load one seed summary and recompute metrics using a fixed threshold."""

    summary_path = root / f"seed_{seed}" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    y_true = np.asarray(metrics["y_true"], dtype=np.int64)
    y_prob = np.asarray(metrics["y_prob"], dtype=np.float32)
    return compute_metrics(y_true, y_prob[:, 1], threshold)


def main() -> None:
    """Run threshold sweep and save JSON results."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--stop", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.input_dir)
    thresholds = np.round(np.arange(args.start, args.stop + args.step / 2, args.step), 6)
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        seed_metrics = [load_seed_metrics(root, seed, float(threshold)) for seed in args.seeds]
        aggregate = {
            metric: summarize([seed_result[metric] for seed_result in seed_metrics])
            for metric in seed_metrics[0]
        }
        rows.append({"threshold": float(threshold), "metrics": aggregate, "seed_metrics": seed_metrics})

    best_balanced = max(rows, key=lambda row: row["metrics"]["balanced_accuracy"]["mean"])
    best_youden = max(
        rows,
        key=lambda row: row["metrics"]["sensitivity"]["mean"] + row["metrics"]["specificity"]["mean"] - 1,
    )
    best_f1 = max(rows, key=lambda row: row["metrics"]["f1"]["mean"])
    result = {
        "source": str(root),
        "seeds": args.seeds,
        "thresholds": rows,
        "best_balanced_accuracy": best_balanced,
        "best_youden": best_youden,
        "best_f1": best_f1,
    }

    output_path = Path(args.output) if args.output else root / "fixed_threshold_sweep.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for label, row in (
        ("best_balanced_accuracy", best_balanced),
        ("best_youden", best_youden),
        ("best_f1", best_f1),
    ):
        print(f"{label}: threshold={row['threshold']:.2f}")
        for metric in ["accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity"]:
            values = row["metrics"][metric]
            print(f"  {metric}: {values['mean']:.4f} +/- {values['std']:.4f}")
    print(f"Saved sweep to: {output_path}")


if __name__ == "__main__":
    main()
