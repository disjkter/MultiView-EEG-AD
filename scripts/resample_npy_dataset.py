"""Resample an x/y/sub_id npy EEG dataset to a fixed temporal length."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample


def load_array(path: Path, allow_pickle: bool = False) -> np.ndarray:
    """Load a numpy array from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path, allow_pickle=allow_pickle)


def main() -> None:
    """Resample x_19ch.npy and copy labels/subject IDs to a new directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory with x_19ch.npy/y_19ch.npy/sub_ids_19ch.npy")
    parser.add_argument("--output-dir", required=True, help="Directory to write the resampled dataset")
    parser.add_argument("--target-length", type=int, default=640, help="Target temporal samples per segment")
    parser.add_argument("--source-sfreq", type=float, default=None, help="Optional source sampling rate metadata")
    parser.add_argument("--target-sfreq", type=float, default=128.0, help="Target sampling rate metadata")
    parser.add_argument("--window-seconds", type=float, default=5.0, help="Window length metadata")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = load_array(input_dir / "x_19ch.npy")
    y = load_array(input_dir / "y_19ch.npy")
    sub_ids = load_array(input_dir / "sub_ids_19ch.npy", allow_pickle=True)

    if x.ndim != 3:
        raise ValueError(f"Expected x shape [N, C, T], got {x.shape}")
    if x.shape[0] != y.shape[0] or x.shape[0] != sub_ids.shape[0]:
        raise ValueError(f"Mismatched sample counts: x={x.shape}, y={y.shape}, sub_ids={sub_ids.shape}")

    x_resampled = resample(x.astype(np.float32, copy=False), args.target_length, axis=-1).astype(np.float32)

    np.save(output_dir / "x_19ch.npy", x_resampled)
    np.save(output_dir / "y_19ch.npy", y.astype(np.int64, copy=False))
    np.save(output_dir / "sub_ids_19ch.npy", sub_ids)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_shape": list(x.shape),
        "target_shape": list(x_resampled.shape),
        "source_sfreq": args.source_sfreq,
        "target_sfreq": args.target_sfreq,
        "window_seconds": args.window_seconds,
        "target_length": args.target_length,
        "num_samples": int(x_resampled.shape[0]),
        "num_subjects": int(len(set(map(str, sub_ids.tolist())))),
        "labels": {str(int(label)): int(count) for label, count in zip(*np.unique(y, return_counts=True))},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
