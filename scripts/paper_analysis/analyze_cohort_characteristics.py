"""Compute observable participant and EEG characteristics across cohorts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper_analysis.common import parse_named_path, write_csv, write_json
from scripts.run_dual_stream_loso_subject import resolve_domain_files


BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}


def selected_indices(subject_ids: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    """Select a bounded, deterministic number of windows per participant."""

    rng = np.random.default_rng(seed)
    output: list[int] = []
    values = np.asarray([str(value) for value in subject_ids], dtype=object)
    for subject in sorted(set(values)):
        indices = np.flatnonzero(values == subject)
        if maximum > 0 and len(indices) > maximum:
            indices = np.sort(rng.choice(indices, maximum, replace=False))
        output.extend(int(index) for index in indices)
    return np.asarray(output, dtype=np.int64)


def analyze(name: str, root: Path, maximum: int, sfreq: float, seed: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Return cohort-level and channel-level observable summaries."""

    x_path, y_path, id_path = resolve_domain_files(root)
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    subject_ids = np.load(id_path, allow_pickle=True)
    indices = selected_indices(subject_ids, maximum, seed)
    sample = np.asarray(x[indices], dtype=np.float32)
    labels_by_subject: dict[str, int] = {}
    counts = Counter(str(value) for value in subject_ids)
    for label, subject in zip(y, subject_ids):
        labels_by_subject[str(subject)] = int(label)
    class_counts = Counter(labels_by_subject.values())
    frequencies = np.fft.rfftfreq(sample.shape[-1], d=1.0 / sfreq)
    power = np.abs(np.fft.rfft(sample, axis=-1)) ** 2
    total_mask = (frequencies >= 1.0) & (frequencies <= 45.0)
    total_power = power[..., total_mask].sum(axis=-1) + 1e-12
    row: dict[str, object] = {
        "cohort": name,
        "segments": int(len(x)),
        "subjects": len(labels_by_subject),
        "controls": int(class_counts[0]),
        "ad": int(class_counts[1]),
        "ad_prevalence": float(class_counts[1] / len(labels_by_subject)),
        "window_seconds": float(x.shape[-1] / sfreq),
        "windows_per_subject_mean": float(np.mean(list(counts.values()))),
        "windows_per_subject_sd": float(np.std(list(counts.values()), ddof=0)),
        "sampled_windows": len(indices),
        "amplitude_mean": float(sample.mean()),
        "amplitude_sd": float(sample.std()),
        "amplitude_rms": float(np.sqrt(np.mean(sample.astype(np.float64) ** 2))),
    }
    for band, (low, high) in BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        row[f"relative_{band}_power"] = float(np.mean(power[..., mask].sum(axis=-1) / total_power))
    channel_rows = [
        {
            "cohort": name,
            "channel_index": channel,
            "mean": float(sample[:, channel].mean()),
            "sd": float(sample[:, channel].std()),
            "rms": float(np.sqrt(np.mean(sample[:, channel].astype(np.float64) ** 2))),
        }
        for channel in range(sample.shape[1])
    ]
    return row, channel_rows


def main() -> None:
    """Analyze available processed cohorts without altering source data."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--max-windows-per-subject", type=int, default=20)
    parser.add_argument("--sampling-rate", type=float, default=128.0)
    parser.add_argument("--seed", type=int, default=260812)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    cohorts: list[dict[str, object]] = []
    channels: list[dict[str, object]] = []
    missing: list[dict[str, str]] = []
    for value in args.dataset:
        name, root = parse_named_path(value)
        if not root.exists():
            missing.append({"cohort": name, "path": str(root)})
            continue
        cohort, channel_rows = analyze(name, root, args.max_windows_per_subject, args.sampling_rate, args.seed)
        cohorts.append(cohort)
        channels.extend(channel_rows)
        print(f"Analyzed {name}: {cohort['subjects']} participants")
    prefix = args.output_prefix
    write_csv(prefix.with_name(prefix.name + "_cohorts.csv"), cohorts)
    write_csv(prefix.with_name(prefix.name + "_channels.csv"), channels)
    write_json(prefix.with_suffix(".json"), {"cohorts": cohorts, "channels": channels, "missing": missing})


if __name__ == "__main__":
    main()
