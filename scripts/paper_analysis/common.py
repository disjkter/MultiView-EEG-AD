"""Shared utilities for participant-level paper analyses."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from scripts.analyze_cross_cohort_calibration import calibration_metrics


def parse_named_path(value: str) -> tuple[str, Path]:
    """Parse a command-line value formatted as NAME=PATH."""

    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value}")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise ValueError(f"Expected non-empty NAME=PATH, got {value}")
    return name.strip(), Path(raw_path.strip())


def positive_logit(probability: Iterable[float]) -> np.ndarray:
    """Convert positive-class probabilities to finite logits."""

    values = np.clip(np.asarray(list(probability), dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(values) - np.log1p(-values)


def cohort_z_predictions(probability: Iterable[float]) -> np.ndarray:
    """Return the diagnostic cohort-wise zero-threshold predictions."""

    logits = positive_logit(probability)
    scale = float(logits.std())
    standardized = (logits - logits.mean()) / (scale + 1e-8)
    return (standardized >= 0.0).astype(np.int64)


def participant_metrics(
    y_true: Iterable[int], probability: Iterable[float], threshold: float
) -> dict[str, float]:
    """Compute the paper's participant-level discrimination and reliability metrics."""

    labels = np.asarray(list(y_true), dtype=np.int64)
    probabilities = np.asarray(list(probability), dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities)),
    }
    result.update(calibration_metrics(labels, probabilities))
    return result


def iter_lodo_folds(root: Path, experiment: str) -> Iterable[dict[str, object]]:
    """Yield normalized folds from every LODO summary below a result root."""

    paths = sorted(root.rglob("lodo_summary.json"))
    if not paths:
        raise FileNotFoundError(f"No lodo_summary.json files found under {root}")
    seen: set[tuple[str, int, str]] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        variant = str(payload.get("variant", path.parent.name))
        seed = int(payload["seed"])
        for fold in payload["folds"]:
            target = str(fold["target"])
            key = (variant, seed, target)
            if key in seen:
                continue
            seen.add(key)
            yield {
                "experiment": experiment,
                "variant": variant,
                "method": str(payload["method"]),
                "backbone": str(payload.get("backbone", experiment)),
                "seed": seed,
                "target": target,
                "path": str(path),
                "fold": fold,
            }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write homogeneous rows to a UTF-8 CSV."""

    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    """Write a stable human-readable JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
