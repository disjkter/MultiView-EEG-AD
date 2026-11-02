"""Preprocess the 40Hz Auditory Entrainment BIDS dataset into 19-channel npy files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt


TARGET_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
]
CANONICAL_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a TSV file into dictionaries."""

    with path.open("r", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj, delimiter="\t"))


def resolve_label(group: str, include_mci_as_positive: bool) -> int | None:
    """Map participant group to binary label: Normal=0, AD/MCI=1."""

    if group == "Normal":
        return 0
    if group in {"Mild AD", "Moderate AD"}:
        return 1
    if include_mci_as_positive and group == "MCI":
        return 1
    return None


def load_eeg_fdt(eeg_dir: Path, subject_id: str) -> tuple[np.ndarray, float]:
    """Load one EEGLAB .fdt file using BIDS sidecar metadata."""

    prefix = f"{subject_id}_task-40HzAuditoryEntrainment"
    json_path = eeg_dir / f"{prefix}_eeg.json"
    fdt_path = eeg_dir / f"{prefix}_eeg.fdt"
    channels_path = eeg_dir / f"{prefix}_channels.tsv"

    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    sfreq = float(metadata["SamplingFrequency"])
    channels = [row["name"] for row in read_tsv(channels_path)]
    if channels != TARGET_CHANNELS:
        missing = [ch for ch in TARGET_CHANNELS if ch not in channels]
        if missing:
            raise ValueError(f"{subject_id} is missing channels: {missing}")

    n_channels = len(channels)
    raw = np.fromfile(fdt_path, dtype="<f4")
    if raw.size % n_channels != 0:
        raise ValueError(f"Unexpected fdt size for {subject_id}: {raw.size} values")
    data = raw.reshape(-1, n_channels).T.astype(np.float32, copy=False)
    reorder_idx = [channels.index(ch) for ch in TARGET_CHANNELS]
    return data[reorder_idx], sfreq


def bandpass_filter(data: np.ndarray, sfreq: float, low_hz: float, high_hz: float) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter."""

    nyquist = sfreq / 2.0
    high = min(high_hz, nyquist - 1.0)
    sos = butter(4, [low_hz / nyquist, high / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, data, axis=-1).astype(np.float32, copy=False)


def iter_event_ranges(eeg_dir: Path, subject_id: str, condition: str, sfreq: float, n_times: int) -> list[tuple[int, int]]:
    """Return sample ranges selected by condition."""

    if condition == "all":
        return [(0, n_times)]

    events_path = eeg_dir / f"{subject_id}_task-40HzAuditoryEntrainment_events.tsv"
    ranges: list[tuple[int, int]] = []
    for row in read_tsv(events_path):
        trial_type = row["trial_type"].lower()
        if condition == "stimulus" and trial_type != "stimulus":
            continue
        if condition == "rest" and trial_type != "rest":
            continue
        start = int(round(float(row["onset"]) * sfreq))
        stop = int(round((float(row["onset"]) + float(row["duration"])) * sfreq))
        ranges.append((max(0, start), min(n_times, stop)))
    return ranges


def segment_ranges(data: np.ndarray, ranges: list[tuple[int, int]], window_samples: int, stride_samples: int) -> list[np.ndarray]:
    """Slice selected ranges into fixed-length windows."""

    segments: list[np.ndarray] = []
    for start, stop in ranges:
        if stop - start < window_samples:
            continue
        for cursor in range(start, stop - window_samples + 1, stride_samples):
            segments.append(data[:, cursor : cursor + window_samples].copy())
    return segments


def main() -> None:
    """Convert the 40Hz auditory dataset to the repository's processed format."""

    parser = argparse.ArgumentParser(description="Preprocess 40Hz Auditory Entrainment EEG data.")
    parser.add_argument("--input-dir", default="40Hz Auditory Entrainment")
    parser.add_argument("--output-dir", default="auditory40hz_processed_19ch_rest_ad_vs_normal")
    parser.add_argument("--condition", choices=["rest", "stimulus", "all"], default="rest")
    parser.add_argument("--window-samples", type=int, default=1024)
    parser.add_argument("--stride-samples", type=int, default=1024)
    parser.add_argument("--low-hz", type=float, default=1.0)
    parser.add_argument("--high-hz", type=float, default=45.0)
    parser.add_argument("--include-mci-as-positive", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    participants = read_tsv(input_dir / "participants.tsv")
    segments: list[np.ndarray] = []
    labels: list[int] = []
    subject_ids: list[str] = []
    skipped: dict[str, str] = {}

    for row in participants:
        subject_id = row["participant_id"]
        label = resolve_label(row["Group"], args.include_mci_as_positive)
        if label is None:
            skipped[subject_id] = row["Group"]
            continue

        eeg_dir = input_dir / subject_id / "eeg"
        data, sfreq = load_eeg_fdt(eeg_dir, subject_id)
        data = bandpass_filter(data, sfreq, args.low_hz, args.high_hz)
        ranges = iter_event_ranges(eeg_dir, subject_id, args.condition, sfreq, data.shape[-1])
        subject_segments = segment_ranges(data, ranges, args.window_samples, args.stride_samples)
        segments.extend(subject_segments)
        labels.extend([label] * len(subject_segments))
        subject_ids.extend([subject_id] * len(subject_segments))

    if not segments:
        raise RuntimeError("No EEG segments were generated.")

    x = np.stack(segments).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)
    sub_ids = np.asarray(subject_ids)

    np.save(output_dir / "x_19ch.npy", x)
    np.save(output_dir / "y_19ch.npy", y)
    np.save(output_dir / "sub_ids_19ch.npy", sub_ids)

    summary = {
        "input_dir": str(input_dir),
        "channels": CANONICAL_CHANNELS,
        "source_channel_names": TARGET_CHANNELS,
        "channel_aliases": {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"},
        "channel_order_source": "Per-subject BIDS channels.tsv, explicitly reordered by this script",
        "condition": args.condition,
        "window_samples": args.window_samples,
        "stride_samples": args.stride_samples,
        "low_hz": args.low_hz,
        "high_hz": args.high_hz,
        "include_mci_as_positive": args.include_mci_as_positive,
        "num_segments": int(len(y)),
        "num_subjects": int(len(np.unique(sub_ids))),
        "segments_by_label": {str(label): int((y == label).sum()) for label in sorted(np.unique(y))},
        "subjects_by_label": {
            str(label): int(len(np.unique(sub_ids[y == label]))) for label in sorted(np.unique(y))
        },
        "skipped_subjects": skipped,
        "shape": list(x.shape),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
