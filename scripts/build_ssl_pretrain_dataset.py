"""Merge multiple 19-channel npy EEG datasets for self-supervised pretraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load x/y/sub_id arrays from one dataset directory."""

    x_path = data_dir / "x_19ch.npy"
    y_path = data_dir / "y_19ch.npy"
    id_path = data_dir / "sub_ids_19ch.npy"
    if not x_path.exists() or not y_path.exists() or not id_path.exists():
        raise FileNotFoundError(f"Could not find x/y/sub_id npy files under {data_dir}")
    x = np.load(x_path).astype(np.float32, copy=False)
    y = np.load(y_path).astype(np.int64, copy=False)
    sub_ids = np.load(id_path, allow_pickle=True)
    if x.ndim != 3:
        raise ValueError(f"Expected x shape [N, C, T] under {data_dir}, got {x.shape}")
    if x.shape[0] != y.shape[0] or x.shape[0] != sub_ids.shape[0]:
        raise ValueError(f"Mismatched sample counts under {data_dir}: x={x.shape}, y={y.shape}, sub_ids={sub_ids.shape}")
    return x, y, sub_ids


def main() -> None:
    """Merge datasets with identical [C, T] shape into one pretraining directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Input dataset directories")
    parser.add_argument("--names", nargs="+", default=None, help="Optional domain names, same length as --inputs")
    parser.add_argument("--output-dir", required=True, help="Merged output directory")
    parser.add_argument("--target-sfreq", type=float, default=128.0)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    args = parser.parse_args()

    input_dirs = [Path(path) for path in args.inputs]
    names = args.names or [path.name for path in input_dirs]
    if len(names) != len(input_dirs):
        raise ValueError("--names must have the same length as --inputs")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    domains: list[dict[str, object]] = []
    expected_shape: tuple[int, int] | None = None
    for data_dir, name in zip(input_dirs, names):
        x, y, sub_ids = load_dataset(data_dir)
        shape = (int(x.shape[1]), int(x.shape[2]))
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(f"Shape mismatch for {data_dir}: expected [C,T]={expected_shape}, got {shape}")

        prefixed_ids = np.asarray([f"{name}::{sid}" for sid in sub_ids.tolist()], dtype=object)
        xs.append(x)
        ys.append(y)
        ids.append(prefixed_ids)
        domains.append(
            {
                "name": name,
                "data_dir": str(data_dir),
                "shape": list(x.shape),
                "num_samples": int(x.shape[0]),
                "num_subjects": int(len(set(map(str, sub_ids.tolist())))),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_all = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
    y_all = np.concatenate(ys, axis=0).astype(np.int64, copy=False)
    id_all = np.concatenate(ids, axis=0)

    np.save(output_dir / "x_19ch.npy", x_all)
    np.save(output_dir / "y_19ch.npy", y_all)
    np.save(output_dir / "sub_ids_19ch.npy", id_all)

    summary = {
        "output_dir": str(output_dir),
        "inputs": domains,
        "target_sfreq": args.target_sfreq,
        "window_seconds": args.window_seconds,
        "shape": list(x_all.shape),
        "num_samples": int(x_all.shape[0]),
        "num_subjects": int(len(set(map(str, id_all.tolist())))),
        "labels": {str(int(label)): int(count) for label, count in zip(*np.unique(y_all, return_counts=True))},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
