"""Merge NPY chunks created by split_npy_chunks.py into one NPY file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to *_chunks_manifest.json.")
    parser.add_argument("--output", required=True, help="Path to the merged .npy file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    chunk_dir = manifest_path.parent
    shape = tuple(manifest["shape"])
    dtype = np.dtype(manifest["dtype"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged = np.lib.format.open_memmap(output_path, mode="w+", dtype=dtype, shape=shape)
    for chunk in manifest["chunks"]:
        chunk_path = chunk_dir / chunk["file"]
        start = int(chunk["start"])
        end = int(chunk["end"])
        data = np.load(chunk_path, mmap_mode="r")
        merged[start:end] = data
        print(f"merged {chunk_path} -> samples {start}:{end}")

    merged.flush()
    print(f"saved merged array to {output_path}")


if __name__ == "__main__":
    main()
