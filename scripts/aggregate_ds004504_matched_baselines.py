"""Audit and aggregate matched ds004504 reference-model experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


METRICS = ("accuracy", "balanced_accuracy", "f1", "auc")
METHODS = (
    ("SVM", "svm"),
    ("EEGNet", "eegnet"),
    ("EEG Conformer", "eegconformer"),
    ("CEEDNet", "ceednet"),
    ("TF-C", "tfc"),
    ("Proposed framework", "proposed"),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--tfc-root", type=Path, required=True)
    parser.add_argument("--branch-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--dataset", default="ds004504_processed_19ch_1s_lead")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    """Load one JSON document and fail with its path when unavailable."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required result: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def summary_path(args: argparse.Namespace, key: str, seed: int) -> Path:
    """Return the expected summary path for one method and seed."""
    if key == "tfc":
        return args.tfc_root / f"seed_{seed}" / "summary.json"
    if key == "proposed":
        return args.branch_root / args.dataset / f"full_s{seed}" / "summary.json"
    return args.baseline_root / key / f"seed_{seed}" / "summary.json"


def metric_payload(summary: dict) -> dict:
    """Return the participant-level metric payload from a run summary."""
    payload = summary.get("metrics")
    if not isinstance(payload, dict):
        raise ValueError("Run summary has no participant-level metrics object")
    return payload


def format_cell(mean: float, std: float, bold: bool) -> str:
    """Format one IEEE table cell."""
    value = f"{mean:.3f}\\pm{std:.3f}"
    return f"$\\mathbf{{{value}}}$" if bold else f"${value}$"


def main() -> None:
    """Verify exact participant identities and export aggregate artifacts."""
    args = parse_args()
    runs: list[dict] = []

    for seed in args.seeds:
        split_path = args.split_root / f"{args.dataset}_seed{seed}.json"
        split = read_json(split_path)
        expected_ids = [str(item) for item in split["test_subjects"]]
        expected_set = set(expected_ids)
        reference_ids: list[str] | None = None
        reference_labels: dict[str, int] | None = None

        for method, key in METHODS:
            path = summary_path(args, key, seed)
            summary = read_json(path)
            metrics = metric_payload(summary)
            for field, expected in {
                "max_train_segments_per_subject": 250,
                "max_val_segments_per_subject": 250,
                "max_test_segments_per_subject": 400,
                "use_ea": True,
                "tune_threshold": True,
                "threshold_objective": "balanced_acc",
            }.items():
                if summary.get(field) != expected:
                    raise ValueError(f"Protocol mismatch in {path}: {field}={summary.get(field)!r}")
            observed_ids = [str(item) for item in metrics.get("subject_ids", [])]
            if len(observed_ids) != len(expected_ids) or set(observed_ids) != expected_set:
                raise ValueError(
                    f"Test participants differ for {method}, seed {seed}: "
                    f"expected={sorted(expected_set)}, observed={sorted(observed_ids)}"
                )
            if reference_ids is None:
                reference_ids = observed_ids
            elif set(observed_ids) != set(reference_ids):
                raise ValueError(f"Methods are not participant-matched at seed {seed}")

            labels = np.asarray(metrics["y_true"], dtype=int)
            predictions = np.asarray(metrics["y_pred"], dtype=int)
            probabilities = np.asarray(metrics["y_prob"], dtype=float)
            if labels.shape != (len(observed_ids),) or predictions.shape != labels.shape:
                raise ValueError(f"Prediction lengths differ in {path}")
            if probabilities.shape != (len(observed_ids), 2) or not np.isfinite(probabilities).all():
                raise ValueError(f"Invalid binary probabilities in {path}")
            label_map = dict(zip(observed_ids, labels.tolist()))
            if reference_labels is None:
                reference_labels = label_map
            elif label_map != reference_labels:
                raise ValueError(f"Participant labels differ in {path}")
            recalculated = {
                "accuracy": accuracy_score(labels, predictions),
                "balanced_accuracy": balanced_accuracy_score(labels, predictions),
                "f1": f1_score(labels, predictions, zero_division=0),
                "auc": roc_auc_score(labels, probabilities[:, 1]),
            }

            row = {
                "method": method,
                "method_key": key,
                "dataset": args.dataset,
                "seed": seed,
                "summary_path": str(path),
                "num_test_participants": len(observed_ids),
            }
            for metric in METRICS:
                if not np.isclose(recalculated[metric], metrics[metric], atol=1e-8):
                    raise ValueError(f"Stored {metric} does not match predictions in {path}")
                row[metric] = float(recalculated[metric])
            runs.append(row)

    aggregates: list[dict] = []
    for method, key in METHODS:
        selected = [row for row in runs if row["method_key"] == key]
        aggregate = {"method": method, "method_key": key, "num_seeds": len(selected)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            aggregate[f"{metric}_mean"] = float(values.mean())
            aggregate[f"{metric}_std"] = float(values.std(ddof=1))
        aggregates.append(aggregate)

    output = {
        "dataset": args.dataset,
        "seeds": args.seeds,
        "identity_audit": "passed",
        "all_methods_use_identical_test_participants_within_each_seed": True,
        "runs": runs,
        "aggregate": aggregates,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    csv_path = args.output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0].keys()))
        writer.writeheader()
        writer.writerows(aggregates)

    best = {
        metric: max(row[f"{metric}_mean"] for row in aggregates) for metric in METRICS
    }
    tex_lines = []
    for row in aggregates:
        cells = [row["method"]]
        for metric in METRICS:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            cells.append(format_cell(mean, std, np.isclose(mean, best[metric])))
        tex_lines.append(" & ".join(cells) + r" \\")
    args.output_prefix.with_name(args.output_prefix.name + "_rows").with_suffix(".tex").write_text(
        "\n".join(tex_lines) + "\n", encoding="utf-8"
    )
    print(f"Verified {len(runs)} matched runs and wrote {args.output_prefix}.json")


if __name__ == "__main__":
    main()
