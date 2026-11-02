"""Convert processed NPY EEG datasets to a LEAD-style 19ch/128Hz/1s format."""

from __future__ import annotations

import argparse
import json
from math import floor
from pathlib import Path

import numpy as np
from scipy.signal import resample


def resolve_domain_files(domain_dir: Path) -> tuple[Path, Path, Path]:
    """Resolve x/y/sub_id npy files in a processed EEG dataset directory."""

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


def disambiguate_mixed_label_subject_ids(y: np.ndarray, sub_ids: np.ndarray) -> tuple[np.ndarray, int]:
    """Prefix labels when one subject id is reused across classes."""

    label_sets: dict[str, set[int]] = {}
    for subject, label in zip(sub_ids, y):
        label_sets.setdefault(str(subject), set()).add(int(label))

    mixed_subjects = {subject for subject, labels in label_sets.items() if len(labels) > 1}
    if not mixed_subjects:
        return sub_ids, 0

    fixed_ids = np.asarray(
        [
            f"label{int(label)}__{subject}" if str(subject) in mixed_subjects else str(subject)
            for subject, label in zip(sub_ids, y)
        ],
        dtype=object,
    )
    return fixed_ids, len(mixed_subjects)


def compute_num_windows(n_times: int, window_samples: int, stride_samples: int) -> int:
    """Return the number of fixed windows that fit into one segment."""

    if n_times < window_samples:
        return 0
    return floor((n_times - window_samples) / stride_samples) + 1


def main() -> None:
    """Resample segments and slice them into 1s windows for LEAD-style experiments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory with x/y/sub_id npy files")
    parser.add_argument("--output-dir", required=True, help="Directory to write LEAD-style npy files")
    parser.add_argument("--source-sfreq", type=float, required=True, help="Original sampling rate of input segments")
    parser.add_argument("--target-sfreq", type=float, default=128.0, help="Target sampling rate")
    parser.add_argument("--window-seconds", type=float, default=1.0, help="Output window length")
    parser.add_argument("--stride-seconds", type=float, default=1.0, help="Output window stride")
    parser.add_argument("--expected-channels", type=int, default=19, help="Expected channel count")
    parser.add_argument("--batch-size", type=int, default=256, help="Conversion batch size")
    parser.add_argument(
        "--no-disambiguate-mixed-label-subjects",
        action="store_true",
        help="Do not prefix labels for reused subject ids across classes",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    mixed_subject_count = 0
    if not args.no_disambiguate_mixed_label_subjects:
        sub_ids, mixed_subject_count = disambiguate_mixed_label_subject_ids(y, sub_ids)

    source_len = int(x.shape[-1])
    target_len = int(round(source_len * float(args.target_sfreq) / float(args.source_sfreq)))
    window_samples = int(round(args.window_seconds * args.target_sfreq))
    stride_samples = int(round(args.stride_seconds * args.target_sfreq))
    if target_len <= 0 or window_samples <= 0 or stride_samples <= 0:
        raise ValueError("Target length, window samples, and stride samples must be positive")

    windows_per_segment = compute_num_windows(target_len, window_samples, stride_samples)
    if windows_per_segment <= 0:
        raise ValueError(
            f"No {window_samples}-sample windows fit after resampling to {target_len} samples"
        )

    n_segments = int(x.shape[0])
    n_windows = n_segments * windows_per_segment
    x_out_path = output_dir / "x_19ch.npy"
    y_out_path = output_dir / "y_19ch.npy"
    id_out_path = output_dir / "sub_ids_19ch.npy"

    x_out = np.lib.format.open_memmap(
        x_out_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_windows, int(x.shape[1]), window_samples),
    )
    y_out = np.empty(n_windows, dtype=np.int64)
    id_out = np.empty(n_windows, dtype=object)

    out_pos = 0
    for start in range(0, n_segments, args.batch_size):
        end = min(start + args.batch_size, n_segments)
        batch = np.asarray(x[start:end], dtype=np.float32)
        if target_len != source_len:
            batch = resample(batch, target_len, axis=-1).astype(np.float32, copy=False)

        batch_size = end - start
        for local_idx in range(batch_size):
            segment = batch[local_idx]
            label = y[start + local_idx]
            subject = sub_ids[start + local_idx]
            for win_idx in range(windows_per_segment):
                win_start = win_idx * stride_samples
                win_end = win_start + window_samples
                x_out[out_pos] = segment[:, win_start:win_end]
                y_out[out_pos] = label
                id_out[out_pos] = subject
                out_pos += 1

    x_out.flush()
    np.save(y_out_path, y_out)
    np.save(id_out_path, id_out)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_shape": list(x.shape),
        "target_shape": [int(n_windows), int(x.shape[1]), int(window_samples)],
        "source_sfreq": float(args.source_sfreq),
        "target_sfreq": float(args.target_sfreq),
        "window_seconds": float(args.window_seconds),
        "stride_seconds": float(args.stride_seconds),
        "source_length": source_len,
        "resampled_length": target_len,
        "window_samples": window_samples,
        "stride_samples": stride_samples,
        "windows_per_segment": windows_per_segment,
        "num_samples": int(n_windows),
        "num_subjects": int(len(set(map(str, id_out.tolist())))),
        "mixed_label_subject_ids_disambiguated": mixed_subject_count,
        "labels": {str(int(label)): int(count) for label, count in zip(*np.unique(y_out, return_counts=True))},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
