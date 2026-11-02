"""Split a large NPY array into smaller chunk files.

This is useful when uploading large EEG tensors to a server is unstable.
The split is lossless: chunks can be concatenated along axis 0 to recover
the original array.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the source .npy file.")
    parser.add_argument("--output-dir", required=True, help="Directory to write chunk .npy files.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Number of samples per chunk along axis 0.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Chunk filename prefix. Defaults to the input file stem.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    array = np.load(input_path, mmap_mode="r")
    if array.ndim < 1:
        raise ValueError(f"Expected an array with at least 1 dimension, got shape {array.shape}")

    prefix = args.prefix or input_path.stem
    num_samples = int(array.shape[0])
    chunk_size = int(args.chunk_size)
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    chunks = []
    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        chunk_name = f"{prefix}_chunk_{len(chunks):04d}.npy"
        chunk_path = output_dir / chunk_name
        # np.asarray materializes only the current slice from the memory map.
        np.save(chunk_path, np.asarray(array[start:end]))
        chunks.append(
            {
                "file": chunk_name,
                "start": start,
                "end": end,
                "shape": list(array[start:end].shape),
            }
        )
        print(f"saved {chunk_path} | samples {start}:{end}")

    manifest = {
        "source": str(input_path),
        "source_name": input_path.name,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "axis": 0,
        "chunk_size": chunk_size,
        "num_chunks": len(chunks),
        "chunks": chunks,
    }
    manifest_path = output_dir / f"{prefix}_chunks_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
