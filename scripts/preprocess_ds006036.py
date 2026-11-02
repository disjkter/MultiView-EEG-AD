"""Preprocess OpenNeuro ds006036 into 19-channel AD-vs-CN numpy arrays."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mne
import numpy as np


TARGET_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a TSV file into dictionaries."""

    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj, delimiter="\t"))


def resolve_binary_label(group: str) -> int | None:
    """Map ds006036 group labels to AD-vs-CN labels."""

    if group == "C":
        return 0
    if group == "A":
        return 1
    return None


def subject_set_path(input_dir: Path, subject_id: str, use_derivatives: bool) -> Path:
    """Return the EEGLAB .set path for one subject."""

    if use_derivatives:
        return (
            input_dir
            / "derivatives"
            / "eeglab"
            / subject_id
            / "eeg"
            / f"{subject_id}_task-photomark_eeg.set"
        )
    return input_dir / subject_id / "eeg" / f"{subject_id}_task-photomark_eeg.set"


def load_subject_data(set_path: Path, target_sfreq: float) -> tuple[np.ndarray, float]:
    """Load one subject and return data with shape (channels, time)."""

    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    missing = [ch for ch in TARGET_CHANNELS if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"{set_path} is missing channels: {missing}")
    raw.pick_channels(TARGET_CHANNELS, ordered=True)
    if abs(float(raw.info["sfreq"]) - target_sfreq) > 1e-6:
        raw.resample(target_sfreq, npad="auto", verbose="ERROR")
    data = raw.get_data().astype(np.float32, copy=False)
    return data, float(raw.info["sfreq"])


def segment_continuous(data: np.ndarray, window_samples: int, stride_samples: int) -> list[np.ndarray]:
    """Slice continuous EEG into fixed-length windows."""

    segments: list[np.ndarray] = []
    n_times = data.shape[-1]
    for start in range(0, n_times - window_samples + 1, stride_samples):
        segments.append(data[:, start : start + window_samples].copy())
    return segments


def main() -> None:
    """Convert ds006036 to x/y/subject npy files used by the experiments."""

    parser = argparse.ArgumentParser(description="Preprocess OpenNeuro ds006036 EEG data.")
    parser.add_argument("--input-dir", default="ds006036")
    parser.add_argument("--output-dir", default="ds006036_processed_19ch_ad_vs_cn_5s")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--stride-seconds", type=float, default=5.0)
    parser.add_argument("--target-sfreq", type=float, default=128.0)
    parser.add_argument("--use-derivatives", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    participants = read_tsv(input_dir / "participants.tsv")
    segments: list[np.ndarray] = []
    labels: list[int] = []
    subject_ids: list[str] = []
    skipped: dict[str, str] = {}

    window_samples = int(round(args.window_seconds * args.target_sfreq))
    stride_samples = int(round(args.stride_seconds * args.target_sfreq))
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("Window and stride must be positive.")

    for row in participants:
        subject_id = row["participant_id"]
        group = row["Group"]
        label = resolve_binary_label(group)
        if label is None:
            skipped[subject_id] = group
            continue

        set_path = subject_set_path(input_dir, subject_id, args.use_derivatives)
        if not set_path.exists():
            skipped[subject_id] = f"missing_file:{group}"
            continue

        data, sfreq = load_subject_data(set_path, args.target_sfreq)
        subject_segments = segment_continuous(data, window_samples, stride_samples)
        segments.extend(subject_segments)
        labels.extend([label] * len(subject_segments))
        subject_ids.extend([subject_id] * len(subject_segments))
        print(f"{subject_id}: group={group} label={label} segments={len(subject_segments)} sfreq={sfreq}")

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
        "use_derivatives": args.use_derivatives,
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "target_sfreq": args.target_sfreq,
        "window_samples": window_samples,
        "stride_samples": stride_samples,
        "num_segments": int(len(y)),
        "num_subjects": int(len(np.unique(sub_ids))),
        "segments_by_label": {str(label): int((y == label).sum()) for label in sorted(np.unique(y))},
        "subjects_by_label": {
            str(label): int(len(np.unique(sub_ids[y == label]))) for label in sorted(np.unique(y))
        },
        "label_mapping": {"0": "CN", "1": "AD"},
        "skipped_subjects": skipped,
        "shape": list(x.shape),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
