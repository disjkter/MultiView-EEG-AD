"""Build canonical 1-second AD-vs-CN windows from ds004504 derivatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]
CHANNEL_ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
GROUP_TO_LABEL = {"C": 0, "A": 1}


def canonicalize_raw_channels(raw: object) -> None:
    """Rename legacy aliases and select the canonical channel order in place."""

    rename = {
        channel: CHANNEL_ALIASES[channel.upper()]
        for channel in raw.ch_names
        if channel.upper() in CHANNEL_ALIASES
    }
    if rename:
        raw.rename_channels(rename)
    case_lookup = {channel.lower(): channel for channel in raw.ch_names}
    missing = [channel for channel in CHANNELS if channel.lower() not in case_lookup]
    if missing:
        raise ValueError(f"Missing canonical channels: {missing}; available={raw.ch_names}")
    actual_names = [case_lookup[channel.lower()] for channel in CHANNELS]
    raw.pick_channels(actual_names, ordered=True)
    raw.rename_channels({actual: canonical for actual, canonical in zip(actual_names, CHANNELS) if actual != canonical})


def segment_signal(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Split [channels, time] into non-overlapping [windows, channels, time]."""

    num_windows = signal.shape[-1] // window_size
    if num_windows == 0:
        raise ValueError(f"Signal has only {signal.shape[-1]} samples; expected at least {window_size}")
    trimmed = signal[:, : num_windows * window_size]
    return trimmed.reshape(signal.shape[0], num_windows, window_size).transpose(1, 0, 2)


def load_labels(participants_path: Path) -> dict[str, int]:
    """Load the 36 AD and 29 CN labels, excluding FTD subjects."""

    participants = pd.read_csv(participants_path, sep="\t")
    labels: dict[str, int] = {}
    for row in participants.itertuples(index=False):
        group = str(row.Group).strip().upper()
        if group in GROUP_TO_LABEL:
            labels[str(row.participant_id)] = GROUP_TO_LABEL[group]
    return labels


def preprocess(args: argparse.Namespace) -> dict[str, object]:
    """Process every expected AD/CN subject and save canonical arrays."""

    import mne

    bids_root = Path(args.bids_root)
    derivatives_dir = bids_root / "derivatives"
    labels = load_labels(bids_root / "participants.tsv")
    if len(labels) != args.expected_subjects:
        raise ValueError(f"Expected {args.expected_subjects} AD/CN labels, found {len(labels)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    window_size = int(round(args.target_sfreq * args.window_sec))
    all_x: list[np.ndarray] = []
    all_y: list[int] = []
    all_ids: list[str] = []
    subject_summary: list[dict[str, object]] = []
    failures: list[str] = []

    for subject_id, label in sorted(labels.items()):
        set_files = sorted((derivatives_dir / subject_id / "eeg").glob("*.set"))
        if len(set_files) != 1:
            failures.append(f"{subject_id}: expected one .set file, found {len(set_files)}")
            continue
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The data contains 'boundary' events.*",
                    category=RuntimeWarning,
                )
                raw = mne.io.read_raw_eeglab(set_files[0], preload=True, verbose=False)
            source_sfreq = float(raw.info["sfreq"])
            canonicalize_raw_channels(raw)
            if source_sfreq != args.target_sfreq:
                raw.resample(args.target_sfreq, verbose=False)

            boundary_annotations = [
                annotation
                for annotation in raw.annotations
                if "boundary" in str(annotation["description"]).lower()
            ]
            if boundary_annotations:
                boundary_bad = mne.Annotations(
                    onset=[float(annotation["onset"]) for annotation in boundary_annotations],
                    duration=[
                        max(float(annotation["duration"]), 1.0 / args.target_sfreq)
                        for annotation in boundary_annotations
                    ],
                    description=["BAD_boundary"] * len(boundary_annotations),
                    orig_time=raw.annotations.orig_time,
                )
                raw.set_annotations(raw.annotations + boundary_bad)

            possible_windows = int(raw.n_times // window_size)
            epochs = mne.make_fixed_length_epochs(
                raw,
                duration=args.window_sec,
                overlap=0.0,
                preload=True,
                reject_by_annotation=True,
                verbose=False,
            )
            windows = epochs.get_data().astype(np.float32, copy=False)
            if len(windows) == 0:
                raise ValueError("No valid windows remain after boundary exclusion")
        except Exception as exc:
            failures.append(f"{subject_id}: {exc!r}")
            continue

        all_x.append(windows)
        all_y.extend([label] * len(windows))
        all_ids.extend([subject_id] * len(windows))
        subject_summary.append(
            {
                "subject_id": subject_id,
                "label": label,
                "source_sfreq": source_sfreq,
                "num_windows": int(len(windows)),
                "boundary_annotations": int(len(boundary_annotations)),
                "windows_dropped_by_annotations": int(possible_windows - len(windows)),
            }
        )
        print(f"Processed {subject_id}: label={label} windows={len(windows)}")

    if failures:
        raise RuntimeError("ds004504 preprocessing was incomplete:\n" + "\n".join(failures))

    x = np.vstack(all_x).astype(np.float32, copy=False)
    y = np.asarray(all_y, dtype=np.int64)
    subject_ids = np.asarray(all_ids)
    unique_subjects = np.unique(subject_ids)
    subjects_by_label = {
        str(label): int(len(np.unique(subject_ids[y == label]))) for label in sorted(np.unique(y))
    }
    if len(unique_subjects) != args.expected_subjects or subjects_by_label != {"0": 29, "1": 36}:
        raise ValueError(
            f"Unexpected final cohort: subjects={len(unique_subjects)}, by_label={subjects_by_label}"
        )

    np.save(output_dir / "x_19ch.npy", x)
    np.save(output_dir / "y_19ch.npy", y)
    np.save(output_dir / "sub_ids_19ch.npy", subject_ids)
    summary = {
        "bids_root": str(bids_root),
        "source": "preprocessed ds004504 derivatives",
        "source_preprocessing": "0.5-45 Hz, A1-A2 reference, ASR, ICA/ICLabel",
        "additional_artifact_rejection": "none",
        "output_dir": str(output_dir),
        "shape": list(x.shape),
        "sampling_rate_hz": args.target_sfreq,
        "window_sec": args.window_sec,
        "stride_sec": args.window_sec,
        "channels": CHANNELS,
        "normalization": "none (performed by downstream loader)",
        "num_subjects": int(len(unique_subjects)),
        "segments_by_label": {
            str(label): int((y == label).sum()) for label in sorted(np.unique(y))
        },
        "subjects_by_label": subjects_by_label,
        "subject_summary": subject_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", default="eeg-data")
    parser.add_argument("--output-dir", default="ds004504_processed_19ch_1s_v2")
    parser.add_argument("--target-sfreq", type=float, default=128.0)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--expected-subjects", type=int, default=65)
    args = parser.parse_args()
    summary = preprocess(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "subject_summary"}, indent=2))


if __name__ == "__main__":
    main()
