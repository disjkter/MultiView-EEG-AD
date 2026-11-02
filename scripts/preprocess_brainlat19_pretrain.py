"""Convert BrainLat-19 1-second EEG features into 5-second pretraining arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]


def count_windows(feature_files: list[Path], window_segments: int, stride_segments: int) -> tuple[int, dict[str, int]]:
    """Count available within-subject windows without loading all data."""

    counts: dict[str, int] = {}
    total = 0
    for path in feature_files:
        data = np.load(path, mmap_mode="r")
        n_segments = int(data.shape[0])
        n_windows = 0 if n_segments < window_segments else (n_segments - window_segments) // stride_segments + 1
        counts[path.stem] = n_windows
        total += n_windows
    return total, counts


def main() -> None:
    """Write x_19ch.npy, y_19ch.npy, sub_ids_19ch.npy, and summary.json."""

    parser = argparse.ArgumentParser(description="Preprocess BrainLat-19 for self-supervised pretraining.")
    parser.add_argument("--input-dir", default="BrainLat-19")
    parser.add_argument("--output-dir", default="BrainLat19_pretrain_19ch_5s")
    parser.add_argument("--window-segments", type=int, default=5, help="Number of 1-second segments per output sample.")
    parser.add_argument("--stride-segments", type=int, default=5, help="Stride in 1-second segments.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_files = sorted((input_dir / "Feature").glob("feature_*.npy"))
    if not feature_files:
        raise FileNotFoundError(f"No feature_*.npy files found in {input_dir / 'Feature'}")

    first = np.load(feature_files[0], mmap_mode="r")
    if first.ndim != 3 or first.shape[1:] != (128, 19):
        raise ValueError(f"Expected feature shape (N, 128, 19), got {first.shape}")

    total_windows, per_file_windows = count_windows(feature_files, args.window_segments, args.stride_segments)
    if total_windows == 0:
        raise RuntimeError("No pretraining windows were generated.")

    out_time = args.window_segments * 128
    x_path = output_dir / "x_19ch.npy"
    y_path = output_dir / "y_19ch.npy"
    sub_path = output_dir / "sub_ids_19ch.npy"
    x_out = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(total_windows, 19, out_time))
    y_out = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.int64, shape=(total_windows,))
    sub_out = np.lib.format.open_memmap(sub_path, mode="w+", dtype="<U16", shape=(total_windows,))

    cursor = 0
    for path in feature_files:
        data = np.load(path, mmap_mode="r")
        n_segments = int(data.shape[0])
        subject_id = path.stem.replace("feature_", "brainlat_")
        n_written = 0
        for start in range(0, n_segments - args.window_segments + 1, args.stride_segments):
            window = np.asarray(data[start : start + args.window_segments], dtype=np.float32)
            # Input shape is (5, 128, 19); output shape is (19, 640).
            x_out[cursor] = window.transpose(2, 0, 1).reshape(19, out_time)
            y_out[cursor] = -1
            sub_out[cursor] = subject_id
            cursor += 1
            n_written += 1
        print(f"{subject_id}: source_segments={n_segments} windows={n_written}")

    x_out.flush()
    y_out.flush()
    sub_out.flush()

    if cursor != total_windows:
        raise RuntimeError(f"Window count mismatch: wrote {cursor}, expected {total_windows}")

    summary = {
        "input_dir": str(input_dir),
        "channels": CHANNELS,
        "channel_order_source": "LEAD aligned 19-channel feature package and dataset data card",
        "window_segments": args.window_segments,
        "stride_segments": args.stride_segments,
        "source_segment_seconds": 1.0,
        "target_seconds": float(args.window_segments),
        "target_sfreq": 128,
        "num_source_files": len(feature_files),
        "num_windows": int(total_windows),
        "shape": [int(dim) for dim in x_out.shape],
        "label_value": -1,
        "per_file_windows_head": dict(list(per_file_windows.items())[:20]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
