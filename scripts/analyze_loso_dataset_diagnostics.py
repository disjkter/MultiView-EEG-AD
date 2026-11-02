"""Summarize LOSO diagnostics for one dataset run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


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


def load_subject_distribution(domain_dir: Path) -> dict[str, object]:
    """Load sample- and subject-level label distribution."""

    x_path, y_path, id_path = resolve_domain_files(domain_dir)
    x = np.load(x_path)
    y = np.load(y_path)
    sub_ids = np.load(id_path, allow_pickle=True)

    subjects = np.unique(sub_ids)
    subject_labels = {
        str(subject): int(y[np.where(sub_ids == subject)[0][0]])
        for subject in subjects
    }
    samples_per_subject = [int(np.sum(sub_ids == subject)) for subject in subjects]
    return {
        "x_shape": list(x.shape),
        "sample_label_counts": dict(Counter(int(v) for v in y.tolist())),
        "subject_label_counts": dict(Counter(subject_labels.values())),
        "num_subjects": int(len(subjects)),
        "samples_per_subject": {
            "min": int(min(samples_per_subject)),
            "max": int(max(samples_per_subject)),
            "mean": float(np.mean(samples_per_subject)),
        },
    }


def build_diagnostics(summary: dict[str, object], distribution: dict[str, object]) -> dict[str, object]:
    """Build a compact diagnostics payload from LOSO results."""

    folds = summary["folds"]
    y_true = np.asarray(summary["metrics"]["y_true"], dtype=np.int64)
    y_pred = np.asarray(summary["metrics"]["y_pred"], dtype=np.int64)
    y_prob = np.asarray(summary["metrics"]["y_prob"], dtype=np.float32)
    subject_ids = [str(v) for v in summary["metrics"]["subject_ids"]]
    thresholds = [fold.get("decision_threshold") for fold in folds if fold.get("decision_threshold") is not None]
    best_epochs = [fold.get("best_epoch") for fold in folds if fold.get("best_epoch") is not None]

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    misclassified = []
    for subject_id, yt, yp, prob in zip(subject_ids, y_true.tolist(), y_pred.tolist(), y_prob.tolist()):
        if yt != yp:
            misclassified.append(
                {
                    "subject_id": subject_id,
                    "y_true": yt,
                    "y_pred": yp,
                    "p0": float(prob[0]),
                    "p1": float(prob[1]),
                }
            )

    diagnostics = {
        "distribution": distribution,
        "run_config": {
            "data_dir": summary["data_dir"],
            "use_ea": summary.get("use_ea"),
            "linear_probe": summary.get("linear_probe"),
            "disable_class_weights": summary.get("disable_class_weights"),
            "loss_type": summary.get("loss_type"),
            "selection_metric": summary.get("selection_metric"),
            "temperature_scaling": summary.get("temperature_scaling"),
            "threshold_objective": summary.get("threshold_objective"),
        },
        "dataset_metrics": {
            "accuracy": float(summary["metrics"]["accuracy"]),
            "balanced_accuracy": float(summary["metrics"]["balanced_accuracy"]),
            "f1": float(summary["metrics"]["f1"]),
            "auc": float(summary["metrics"]["auc"]),
            "sensitivity": float(summary["metrics"]["sensitivity"]),
            "specificity": float(summary["metrics"]["specificity"]),
        },
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "threshold_stats": {
            "min": None if not thresholds else float(min(thresholds)),
            "max": None if not thresholds else float(max(thresholds)),
            "mean": None if not thresholds else float(np.mean(thresholds)),
            "top_counts": Counter(float(v) for v in thresholds).most_common(10),
        },
        "best_epoch_stats": {
            "min": None if not best_epochs else int(min(best_epochs)),
            "max": None if not best_epochs else int(max(best_epochs)),
            "mean": None if not best_epochs else float(np.mean(best_epochs)),
        },
        "misclassified_subjects": misclassified,
    }
    return diagnostics


def write_markdown(output_path: Path, diagnostics: dict[str, object]) -> None:
    """Write a human-readable markdown summary."""

    dist = diagnostics["distribution"]
    metrics = diagnostics["dataset_metrics"]
    cm = diagnostics["confusion_matrix"]
    threshold_stats = diagnostics["threshold_stats"]
    best_epoch_stats = diagnostics["best_epoch_stats"]

    lines = [
        "# LOSO Diagnostics",
        "",
        "## Distribution",
        f"- x shape: `{tuple(dist['x_shape'])}`",
        f"- subjects: `{dist['num_subjects']}`",
        f"- subject label counts: `{dist['subject_label_counts']}`",
        f"- sample label counts: `{dist['sample_label_counts']}`",
        f"- samples per subject: `{dist['samples_per_subject']}`",
        "",
        "## Metrics",
        f"- accuracy: `{metrics['accuracy']:.4f}`",
        f"- balanced_accuracy: `{metrics['balanced_accuracy']:.4f}`",
        f"- f1: `{metrics['f1']:.4f}`",
        f"- auc: `{metrics['auc']:.4f}`",
        f"- sensitivity: `{metrics['sensitivity']:.4f}`",
        f"- specificity: `{metrics['specificity']:.4f}`",
        "",
        "## Confusion Matrix",
        f"- TN: `{cm['tn']}`",
        f"- FP: `{cm['fp']}`",
        f"- FN: `{cm['fn']}`",
        f"- TP: `{cm['tp']}`",
        "",
        "## Threshold Diagnostics",
        f"- min/max/mean: `{threshold_stats['min']}` / `{threshold_stats['max']}` / `{threshold_stats['mean']}`",
        f"- top threshold counts: `{threshold_stats['top_counts']}`",
        "",
        "## Best Epoch Diagnostics",
        f"- min/max/mean best epoch: `{best_epoch_stats['min']}` / `{best_epoch_stats['max']}` / `{best_epoch_stats['mean']}`",
        "",
        "## Misclassified Subjects",
    ]

    if diagnostics["misclassified_subjects"]:
        for item in diagnostics["misclassified_subjects"][:20]:
            lines.append(
                f"- subject `{item['subject_id']}`: true=`{item['y_true']}`, pred=`{item['y_pred']}`, "
                f"p0=`{item['p0']:.4f}`, p1=`{item['p1']:.4f}`"
            )
        if len(diagnostics["misclassified_subjects"]) > 20:
            lines.append(f"- ... {len(diagnostics['misclassified_subjects']) - 20} more")
    else:
        lines.append("- none")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Entry point."""

    parser = argparse.ArgumentParser(description="Analyze LOSO summary diagnostics.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    summary_path = Path(args.summary_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    distribution = load_subject_distribution(data_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    diagnostics = build_diagnostics(summary, distribution)

    json_path = output_dir / "diagnostics.json"
    md_path = output_dir / "diagnostics.md"
    json_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, diagnostics)
    print(f"Saved diagnostics to: {json_path}")
    print(f"Saved diagnostics to: {md_path}")


if __name__ == "__main__":
    main()
