"""Utilities for saving and reusing subject-level train/val/test splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def masks_to_subject_split(
    sub_ids: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, list[str]]:
    """Convert segment masks into serializable subject id lists."""

    return {
        "train_subjects": sorted(str(subject) for subject in np.unique(sub_ids[train_mask])),
        "val_subjects": sorted(str(subject) for subject in np.unique(sub_ids[val_mask])),
        "test_subjects": sorted(str(subject) for subject in np.unique(sub_ids[test_mask])),
    }


def save_subject_split(
    path: str | Path,
    sub_ids: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    metadata: dict[str, object] | None = None,
) -> None:
    """Save subject-level split lists to JSON."""

    split = masks_to_subject_split(sub_ids, train_mask, val_mask, test_mask)
    payload = {"metadata": metadata or {}, **split}
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_subject_split_masks(path: str | Path, sub_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load subject-level split JSON and return segment masks.

    A split may explicitly list ``excluded_subjects`` for few-shot protocols.
    Unlisted subjects remain an error to prevent accidental data omission.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    train_subjects = {str(subject) for subject in payload["train_subjects"]}
    val_subjects = {str(subject) for subject in payload["val_subjects"]}
    test_subjects = {str(subject) for subject in payload["test_subjects"]}
    excluded_subjects = {str(subject) for subject in payload.get("excluded_subjects", [])}

    groups = (train_subjects, val_subjects, test_subjects, excluded_subjects)
    overlap: set[str] = set()
    for left_index, left in enumerate(groups):
        for right in groups[left_index + 1 :]:
            overlap.update(left.intersection(right))
    if overlap:
        raise ValueError(f"Subject split contains overlapping subjects: {sorted(overlap)[:5]}")

    subject_strings = np.asarray([str(subject) for subject in sub_ids], dtype=object)
    available_subjects = set(subject_strings.tolist())
    referenced_subjects = train_subjects | val_subjects | test_subjects | excluded_subjects
    unknown_subjects = sorted(referenced_subjects - available_subjects)
    if unknown_subjects:
        raise ValueError(
            "Subject split references subjects absent from the dataset. "
            f"Unknown examples: {unknown_subjects[:5]}"
        )
    train_mask = np.isin(subject_strings, np.asarray(sorted(train_subjects), dtype=object))
    val_mask = np.isin(subject_strings, np.asarray(sorted(val_subjects), dtype=object))
    test_mask = np.isin(subject_strings, np.asarray(sorted(test_subjects), dtype=object))
    excluded_mask = np.isin(
        subject_strings,
        np.asarray(sorted(excluded_subjects), dtype=object),
    )

    assigned = train_mask | val_mask | test_mask | excluded_mask
    if not np.all(assigned):
        missing = sorted(set(subject_strings[~assigned]))
        raise ValueError(f"Subject split does not assign all subjects. Missing examples: {missing[:5]}")

    return train_mask, val_mask, test_mask
