"""Convert processed x/y/sub_id NPY EEG data to the ADformer dataset layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def resolve_domain_files(domain_dir: Path) -> tuple[Path, Path, Path]:
    """Resolve x/y/sub_id npy files from a processed dataset directory."""

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


def main() -> None:
    """Convert one processed dataset into ADformer Feature/Label layout."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Processed dataset directory with x/y/sub_id npy files")
    parser.add_argument("--output-root", default="external/ADformer/dataset/128Hz", help="ADformer dataset root")
    parser.add_argument("--dataset-name", required=True, help="Output dataset folder name, e.g. CAUEEG or ADFTD-RS")
    parser.add_argument("--expected-channels", type=int, default=19, help="Expected number of EEG channels")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    dataset_dir = output_root / args.dataset_name
    feature_dir = dataset_dir / "Feature"
    label_dir = dataset_dir / "Label"
    feature_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    x_path, y_path, id_path = resolve_domain_files(input_dir)
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path).astype(np.int64, copy=False)
    sub_ids = np.load(id_path, allow_pickle=True)

    if x.ndim != 3:
        raise ValueError(f"Expected x shape [N, C, T], got {x.shape}")
    if x.shape[1] != args.expected_channels:
        raise ValueError(f"Expected {args.expected_channels} channels, got {x.shape[1]}")
    if x.shape[0] != y.shape[0] or x.shape[0] != sub_ids.shape[0]:
        raise ValueError(f"Mismatched sample counts: x={x.shape}, y={y.shape}, sub_ids={sub_ids.shape}")

    unique_subjects = sorted(set(map(str, sub_ids.tolist())))
    subject_to_numeric = {subject: index + 1 for index, subject in enumerate(unique_subjects)}
    label_rows: list[list[int]] = []

    for subject in unique_subjects:
        subject_mask = np.asarray([str(value) == subject for value in sub_ids.tolist()])
        subject_x = np.asarray(x[subject_mask], dtype=np.float32)
        subject_y = y[subject_mask]
        if len(set(map(int, subject_y.tolist()))) != 1:
            raise ValueError(f"Subject {subject} has inconsistent labels: {sorted(set(subject_y.tolist()))}")

        numeric_id = subject_to_numeric[subject]
        # ADformer expects [N, T, C] or [T, C]. Our processed data is [N, C, T].
        subject_feature = np.transpose(subject_x, (0, 2, 1)).astype(np.float32, copy=False)
        np.save(feature_dir / f"feature_{numeric_id}.npy", subject_feature)
        label_rows.append([int(subject_y[0]), int(numeric_id)])

    label_array = np.asarray(label_rows, dtype=np.int64)
    np.save(label_dir / "label.npy", label_array)

    subject_map = [
        {"original_subject_id": subject, "numeric_subject_id": subject_to_numeric[subject]}
        for subject in unique_subjects
    ]
    summary = {
        "input_dir": str(input_dir),
        "output_dataset_dir": str(dataset_dir),
        "dataset_name": args.dataset_name,
        "input_shape": list(x.shape),
        "adformer_subject_count": int(len(unique_subjects)),
        "labels": {str(int(label)): int(count) for label, count in zip(*np.unique(y, return_counts=True))},
        "subject_map": subject_map,
    }
    (dataset_dir / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
