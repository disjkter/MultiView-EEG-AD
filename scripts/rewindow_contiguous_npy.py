"""Combine contiguous fixed-length EEG windows without crossing subject boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def contiguous_runs(subject_ids: np.ndarray, labels: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs with constant subject ID and label."""

    if len(subject_ids) != len(labels):
        raise ValueError("Subject and label arrays must have equal length")
    if not len(subject_ids):
        return []
    boundaries = np.flatnonzero(
        (subject_ids[1:] != subject_ids[:-1]) | (labels[1:] != labels[:-1])
    ) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(subject_ids)]))
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def rewindow(input_dir: Path, output_dir: Path, factor: int) -> dict[str, object]:
    """Concatenate groups of adjacent windows using memory-mapped arrays."""

    if factor < 2:
        raise ValueError("factor must be at least 2")
    x = np.load(input_dir / "x_19ch.npy", mmap_mode="r")
    y = np.load(input_dir / "y_19ch.npy", mmap_mode="r")
    # Object-dtype identifier arrays cannot be memory-mapped. They are small
    # relative to EEG tensors, so load identifiers normally while keeping x/y mmap-backed.
    subjects = np.load(input_dir / "sub_ids_19ch.npy", allow_pickle=True)
    if x.ndim != 3 or x.shape[0] != len(y) or len(y) != len(subjects):
        raise ValueError("Expected aligned x [N,C,T], y [N], and subject IDs [N]")

    runs = contiguous_runs(subjects, y)
    output_count = sum((stop - start) // factor for start, stop in runs)
    if output_count == 0:
        raise RuntimeError("No complete output windows can be formed")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_x = np.lib.format.open_memmap(
        output_dir / "x_19ch.npy",
        mode="w+",
        dtype=x.dtype,
        shape=(output_count, x.shape[1], x.shape[2] * factor),
    )
    out_y = np.lib.format.open_memmap(
        output_dir / "y_19ch.npy", mode="w+", dtype=y.dtype, shape=(output_count,)
    )
    subject_width = max(len(str(value)) for value in np.unique(subjects))
    out_subjects = np.lib.format.open_memmap(
        output_dir / "sub_ids_19ch.npy",
        mode="w+",
        dtype=f"<U{subject_width}",
        shape=(output_count,),
    )

    cursor = 0
    dropped = 0
    for start, stop in runs:
        usable = ((stop - start) // factor) * factor
        dropped += stop - start - usable
        for offset in range(start, start + usable, factor):
            block = np.asarray(x[offset : offset + factor])
            out_x[cursor] = block.transpose(1, 0, 2).reshape(x.shape[1], -1)
            out_y[cursor] = y[offset]
            out_subjects[cursor] = str(subjects[offset])
            cursor += 1
    if cursor != output_count:
        raise RuntimeError(f"Wrote {cursor} windows, expected {output_count}")
    out_x.flush()
    out_y.flush()
    out_subjects.flush()

    source_summary_path = input_dir / "summary.json"
    source_summary = (
        json.loads(source_summary_path.read_text(encoding="utf-8"))
        if source_summary_path.exists()
        else {}
    )
    source_window_seconds = float(
        source_summary.get("window_seconds", source_summary.get("window_sec", 1.0))
    )
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "operation": "concatenate contiguous source windows within subject and label runs",
        "source_window_seconds": source_window_seconds,
        "window_seconds": source_window_seconds * factor,
        "stride_seconds": source_window_seconds * factor,
        "factor": factor,
        "shape": [int(value) for value in out_x.shape],
        "num_segments": output_count,
        "num_subjects": int(len(np.unique(out_subjects))),
        "dropped_source_windows_at_run_ends": int(dropped),
        "channels": source_summary.get("channels"),
        "normalization": source_summary.get("normalization", "none (downstream loader)"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    """Parse command-line arguments and run rewindowing."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--factor", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(rewindow(args.input_dir, args.output_dir, args.factor), indent=2))


if __name__ == "__main__":
    main()
