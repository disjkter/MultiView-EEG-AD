"""Quantify participant-level score location/scale shift in LODO results."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper_analysis.common import (
    cohort_z_predictions,
    iter_lodo_folds,
    parse_named_path,
    positive_logit,
    write_csv,
    write_json,
)


def load_source_predictions(paths: list[Path]) -> dict[tuple[str, str, int, str], np.ndarray]:
    """Load identity-calibrated source-validation probabilities."""

    grouped: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") != "source_validation" or row.get("calibrator") != "identity":
                    continue
                key = (row["backbone"], row["variant"], int(row["seed"]), row["target"])
                grouped[key].append(float(row["raw_probability"]))
    return {key: np.asarray(values) for key, values in grouped.items()}


def compute_shift_row(
    record: dict[str, object], source_probability: np.ndarray | None
) -> dict[str, object]:
    """Compute score shift and operating-point quantities for one fold."""

    fold = record["fold"]
    metrics = fold["metrics"]
    labels = np.asarray(metrics["y_true"], dtype=np.int64)
    probabilities = np.asarray(metrics["y_prob"], dtype=np.float64)
    logits = positive_logit(probabilities)
    threshold_ba = float(metrics["balanced_accuracy"])
    cohort_z_ba = float(
        balanced_accuracy_score(labels, cohort_z_predictions(probabilities))
    )
    row: dict[str, object] = {
        key: record[key]
        for key in ("experiment", "backbone", "variant", "method", "seed", "target")
    }
    row.update(
        {
            "target_subjects": len(labels),
            "target_prevalence": float(labels.mean()),
            "target_logit_mean": float(logits.mean()),
            "target_logit_sd": float(logits.std(ddof=0)),
            "source_threshold_ba": threshold_ba,
            "cohort_z_ba": cohort_z_ba,
            "cohort_z_gain": cohort_z_ba - threshold_ba,
            "auc": float(metrics["auc"]),
        }
    )
    if source_probability is None:
        row.update(
            {
                "source_logit_mean": "",
                "source_logit_sd": "",
                "delta_mean": "",
                "delta_sd": "",
                "score_shift_magnitude": "",
                "source_scores_available": False,
            }
        )
    else:
        source_logits = positive_logit(source_probability)
        delta_mean = float(logits.mean() - source_logits.mean())
        delta_sd = float(logits.std(ddof=0) - source_logits.std(ddof=0))
        row.update(
            {
                "source_logit_mean": float(source_logits.mean()),
                "source_logit_sd": float(source_logits.std(ddof=0)),
                "delta_mean": delta_mean,
                "delta_sd": delta_sd,
                "score_shift_magnitude": math.hypot(delta_mean, delta_sd),
                "source_scores_available": True,
            }
        )
    return row


def association_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Compute descriptive Pearson and Spearman score-shift associations."""

    complete = [row for row in rows if row["source_scores_available"]]
    pairs = (
        ("abs_delta_mean", "source_threshold_ba", lambda row: abs(float(row["delta_mean"]))),
        ("abs_delta_sd", "source_threshold_ba", lambda row: abs(float(row["delta_sd"]))),
        ("score_shift_magnitude", "cohort_z_gain", lambda row: float(row["score_shift_magnitude"])),
        ("score_shift_magnitude", "auc", lambda row: float(row["score_shift_magnitude"])),
    )
    output: list[dict[str, object]] = []
    for predictor, outcome, getter in pairs:
        x = np.asarray([getter(row) for row in complete], dtype=np.float64)
        y = np.asarray([float(row[outcome]) for row in complete], dtype=np.float64)
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            pearson_r = pearson_p = spearman_r = spearman_p = float("nan")
        else:
            pearson_r, pearson_p = pearsonr(x, y)
            spearman_r, spearman_p = spearmanr(x, y)
        output.append(
            {
                "predictor": predictor,
                "outcome": outcome,
                "n_fold_runs": len(x),
                "pearson_r": float(pearson_r),
                "pearson_p_unadjusted": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p_unadjusted": float(spearman_p),
                "inference_note": "descriptive; clustered bootstrap required for manuscript inference",
            }
        )
    return output


def main() -> None:
    """Create fold-level score-shift tables from existing predictions."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--calibration-predictions", action="append", type=Path, default=[])
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    source = load_source_predictions(args.calibration_predictions)
    rows: list[dict[str, object]] = []
    for value in args.results:
        experiment, root = parse_named_path(value)
        for record in iter_lodo_folds(root, experiment):
            key = (
                str(record["backbone"]), str(record["variant"]),
                int(record["seed"]), str(record["target"]),
            )
            embedded = record["fold"].get("source_validation_metrics")
            source_probability = source.get(key)
            if source_probability is None and embedded is not None:
                source_probability = np.asarray(embedded["y_prob"], dtype=np.float64)
            rows.append(compute_shift_row(record, source_probability))
    associations = association_rows(rows)
    prefix = args.output_prefix
    write_csv(prefix.with_name(prefix.name + "_folds.csv"), rows)
    write_csv(prefix.with_name(prefix.name + "_associations.csv"), associations)
    write_json(
        prefix.with_suffix(".json"),
        {
            "metadata": {
                "target_labels_used_for_training_or_selection": False,
                "cohort_z_uses_unlabeled_target_scores": True,
                "associations_are_descriptive": True,
            },
            "folds": rows,
            "associations": associations,
        },
    )
    available = sum(bool(row["source_scores_available"]) for row in rows)
    print(f"Saved {len(rows)} score rows; source scores available for {available}")


if __name__ == "__main__":
    main()
