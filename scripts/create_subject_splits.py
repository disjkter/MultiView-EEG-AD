"""Create reusable stratified subject-level split files without training a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_dual_stream_subject_split import load_domain_mmap, stratified_subject_split  # noqa: E402
from utils.subject_split import save_subject_split  # noqa: E402


DEFAULT_DATASETS = [
    "CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
    "ds004504_processed_19ch_1s_v2",
    "ADFSU_processed_19ch_1s_v2",
    "ADSZ_processed_19ch_1s_v2",
]


def count_subjects(y: np.ndarray, sub_ids: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    """Count subjects per class without depending on training-script helpers."""

    indices = np.asarray(indices, dtype=np.int64)
    counts: dict[str, int] = {}
    for subject in np.unique(sub_ids[indices]):
        labels = np.unique(y[indices][sub_ids[indices] == subject])
        if len(labels) != 1:
            raise ValueError(f"Subject {subject!r} has mixed labels in the selected split: {labels.tolist()}")
        key = str(int(labels[0]))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def require_two_classes_in_split(
    split_name: str,
    y: np.ndarray,
    sub_ids: np.ndarray,
    indices: np.ndarray,
) -> None:
    """Fail fast when a generated split contains fewer than two classes."""

    counts = count_subjects(y, sub_ids, indices)
    if len(counts) < 2:
        raise ValueError(f"{split_name} split contains fewer than two subject classes: {counts}")


def main() -> None:
    """Generate deterministic split JSON files and a compact manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--output-dir", default="outputs/subject_splits")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for dataset_name in args.datasets:
        _, y, sub_ids = load_domain_mmap(Path(dataset_name))
        for seed in args.seeds:
            train_mask, val_mask, test_mask = stratified_subject_split(
                y,
                sub_ids,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                seed=seed,
            )
            split_indices = {
                "train": np.flatnonzero(train_mask),
                "val": np.flatnonzero(val_mask),
                "test": np.flatnonzero(test_mask),
            }
            for split_name, indices in split_indices.items():
                require_two_classes_in_split(split_name, y, sub_ids, indices)

            split_path = output_dir / f"{dataset_name}_seed{seed}.json"
            metadata = {
                "data_dir": dataset_name,
                "seed": seed,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
            }
            save_subject_split(
                split_path,
                sub_ids,
                train_mask,
                val_mask,
                test_mask,
                metadata=metadata,
            )
            row = {
                **metadata,
                "split_file": str(split_path),
                "segments": {name: int(len(indices)) for name, indices in split_indices.items()},
                "subjects_by_label": {
                    name: count_subjects(y, sub_ids, indices)
                    for name, indices in split_indices.items()
                },
            }
            manifest.append(row)
            print(
                f"Saved {split_path}: "
                f"train={row['subjects_by_label']['train']} "
                f"val={row['subjects_by_label']['val']} "
                f"test={row['subjects_by_label']['test']}"
            )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
