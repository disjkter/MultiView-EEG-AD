"""Convert the four-channel MAT dataset into npy files used by experiment scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


CLASS_FILES = [
    ("normal", "normal.mat"),
    ("MCI", "MCI.mat"),
    ("AD", "AD.mat"),
]


def matlab_noisy_to_zero_based(noisy: np.ndarray, num_trials: int) -> np.ndarray:
    """Convert MATLAB noisy-trial indices to zero-based indexing."""

    noisy = np.asarray(noisy).reshape(-1)
    if noisy.size == 0:
        return np.asarray([], dtype=np.int64)

    noisy = noisy.astype(np.int64, copy=False)
    if noisy.min() >= 1 and noisy.max() <= num_trials:
        noisy = noisy - 1
    return noisy[(noisy >= 0) & (noisy < num_trials)]


def select_subjects(task: str) -> tuple[dict[str, int], set[str]]:
    """Return label mapping and allowed classes for one task."""

    if task == "multiclass":
        return {"normal": 0, "MCI": 1, "AD": 2}, {"normal", "MCI", "AD"}
    if task == "ad_vs_normal":
        return {"normal": 0, "AD": 1}, {"normal", "AD"}
    if task == "mci_vs_normal":
        return {"normal": 0, "MCI": 1}, {"normal", "MCI"}
    if task == "ad_vs_mci":
        return {"MCI": 0, "AD": 1}, {"MCI", "AD"}
    raise ValueError(f"Unsupported task: {task}")


def process_dataset(input_dir: Path, output_dir: Path, task: str) -> dict[str, object]:
    """Convert MAT files into x/y/sub_ids arrays."""

    label_map, allowed = select_subjects(task)
    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    sub_ids: list[int] = []
    subject_meta: list[dict[str, object]] = []
    next_subject_id = 0

    for class_name, file_name in CLASS_FILES:
        if class_name not in allowed:
            continue

        mat = loadmat(input_dir / file_name)[class_name]
        for subject_idx in range(mat.shape[1]):
            subject = mat[0, subject_idx]
            epoch = np.asarray(subject["epoch"], dtype=np.float32)
            odor = np.asarray(subject["odor"]).reshape(-1)
            noisy = matlab_noisy_to_zero_based(subject["noisy"], epoch.shape[2])

            keep_mask = np.ones(epoch.shape[2], dtype=bool)
            keep_mask[noisy] = False
            clean_indices = np.where(keep_mask)[0]

            clean_trials = np.transpose(epoch[:, :, clean_indices], (2, 0, 1))
            x_list.append(clean_trials)
            y_list.extend([label_map[class_name]] * clean_trials.shape[0])
            sub_ids.extend([next_subject_id] * clean_trials.shape[0])

            subject_meta.append(
                {
                    "subject_id": next_subject_id,
                    "class_name": class_name,
                    "label": label_map[class_name],
                    "source_file": file_name,
                    "source_subject_index": subject_idx,
                    "num_trials_total": int(epoch.shape[2]),
                    "num_trials_clean": int(clean_trials.shape[0]),
                    "num_trials_removed": int(len(noisy)),
                    "odor_shape": list(odor.shape),
                }
            )
            next_subject_id += 1

    if not x_list:
        raise RuntimeError(f"No subjects selected for task={task}")

    x_all = np.concatenate(x_list, axis=0).astype(np.float32)
    y_all = np.asarray(y_list, dtype=np.int64)
    sub_all = np.asarray(sub_ids, dtype=np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "x.npy", x_all)
    np.save(output_dir / "y.npy", y_all)
    np.save(output_dir / "sub_ids.npy", sub_all)

    summary = {
        "task": task,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "x_shape": list(x_all.shape),
        "num_subjects": int(len(subject_meta)),
        "num_samples": int(len(y_all)),
        "label_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(y_all, return_counts=True))},
        "subjects_per_class": {
            class_name: int(sum(1 for item in subject_meta if item["class_name"] == class_name))
            for class_name in allowed
        },
        "subject_metadata": subject_meta,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """Entry point."""

    parser = argparse.ArgumentParser(description="Process the four-channel MAT dataset.")
    parser.add_argument("--input-dir", default="four_channels_dataset")
    parser.add_argument("--output-root", default="processed_four_channels")
    parser.add_argument(
        "--task",
        default="ad_vs_normal",
        choices=["multiclass", "ad_vs_normal", "mci_vs_normal", "ad_vs_mci"],
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    output_dir = output_root / args.task
    summary = process_dataset(input_dir=input_dir, output_dir=output_dir, task=args.task)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
