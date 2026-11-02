"""Preprocess ADSZ EEG files with configurable window length and overlap."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt


ADSZ_CHANNELS = [
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
CLASS_MAP = {"Healthy": 0, "AD": 1}


def canonical_adsz_subject_stem(recording_stem: str) -> str:
    """Remove the eyes-closed/open marker from an ADSZ recording stem."""

    ad_match = re.fullmatch(r"e[co](\d{4})", recording_stem, flags=re.IGNORECASE)
    if ad_match:
        return ad_match.group(1)

    healthy_match = re.fullmatch(r"eeg(\d+)[co](\d+)", recording_stem, flags=re.IGNORECASE)
    if healthy_match:
        return f"eeg{healthy_match.group(1)}_{healthy_match.group(2)}"

    raise ValueError(
        f"Unrecognized ADSZ recording name {recording_stem!r}; "
        "expected AD e[c/o]NNNN or Healthy eegNN[c/o]N."
    )


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int) -> np.ndarray:
    """Apply a Butterworth band-pass filter along the time axis."""

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=-1)


def zscore_epoch(epoch: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Channel-wise z-score normalization for one EEG epoch [C, T]."""

    mean = epoch.mean(axis=1, keepdims=True)
    std = epoch.std(axis=1, keepdims=True) + eps
    return ((epoch - mean) / std).astype(np.float32)


def extract_epochs(
    file_path: Path,
    fs: float,
    lowcut: float,
    highcut: float,
    filter_order: int,
    window_size: int,
    stride: int,
    normalize: bool,
) -> np.ndarray | None:
    """Load one ADSZ .out file and slice it into fixed windows."""

    try:
        raw_data = np.loadtxt(file_path, dtype=np.float32)
        if raw_data.ndim != 2 or raw_data.shape[1] != len(ADSZ_CHANNELS):
            raise ValueError(
                f"expected [time, {len(ADSZ_CHANNELS)}], got {raw_data.shape}"
            )
        continuous = raw_data.T  # [19, T]
        filtered = bandpass_filter(continuous, lowcut, highcut, fs, filter_order)

        time_steps = filtered.shape[1]
        epochs = []
        for start in range(0, time_steps - window_size + 1, stride):
            epoch = filtered[:, start : start + window_size].copy()
            if normalize:
                epoch = zscore_epoch(epoch)
            epochs.append(epoch.astype(np.float32, copy=False))

        if not epochs:
            return None
        return np.stack(epochs, axis=0).astype(np.float32)
    except Exception as exc:
        print(f"Failed to parse {file_path.name}: {exc}")
        return None


def preprocess_adsz(args: argparse.Namespace) -> dict[str, object]:
    """Preprocess ADSZ into x/y/sub_id npy files."""

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window_size = int(round(args.window_sec * args.fs))
    stride = int(round(args.stride_sec * args.fs))
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_sec and stride_sec must produce positive sample counts.")

    all_x: list[np.ndarray] = []
    all_y: list[int] = []
    all_subject_ids: list[str] = []
    subject_summary: list[dict[str, object]] = []

    for class_name, label in CLASS_MAP.items():
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Missing class directory: {class_dir}")
            continue

        for out_file in sorted(class_dir.rglob("*.out")):
            subject_id = f"ADSZ_{class_name}_{canonical_adsz_subject_stem(out_file.stem)}"
            epochs = extract_epochs(
                out_file,
                fs=args.fs,
                lowcut=args.lowcut,
                highcut=args.highcut,
                filter_order=args.filter_order,
                window_size=window_size,
                stride=stride,
                normalize=args.normalize,
            )
            if epochs is None:
                continue

            all_x.append(epochs)
            all_y.extend([label] * len(epochs))
            all_subject_ids.extend([subject_id] * len(epochs))
            subject_summary.append(
                {
                    "subject_id": subject_id,
                    "recording_id": out_file.stem,
                    "file": str(out_file),
                    "label": label,
                    "class_name": class_name,
                    "num_epochs": int(len(epochs)),
                }
            )
    if not all_x:
        raise RuntimeError("No valid ADSZ epochs were extracted.")

    x = np.vstack(all_x).astype(np.float32)
    y = np.asarray(all_y, dtype=np.int64)
    sub_ids = np.asarray(all_subject_ids)

    unique_subjects = np.unique(sub_ids)
    if args.expected_subjects is not None and len(unique_subjects) != args.expected_subjects:
        raise ValueError(
            f"Expected {args.expected_subjects} ADSZ subjects, found {len(unique_subjects)}"
        )
    expected_per_class = args.expected_subjects // len(CLASS_MAP) if args.expected_subjects else None
    if expected_per_class is not None:
        for label in CLASS_MAP.values():
            actual = len(np.unique(sub_ids[y == label]))
            if actual != expected_per_class:
                raise ValueError(
                    f"Expected {expected_per_class} subjects for label {label}, found {actual}"
                )

    np.save(output_dir / "x_19ch.npy", x)
    np.save(output_dir / "y_19ch.npy", y)
    np.save(output_dir / "sub_ids_19ch.npy", sub_ids)

    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "fs": args.fs,
        "lowcut": args.lowcut,
        "highcut": args.highcut,
        "filter_order": args.filter_order,
        "channels": ADSZ_CHANNELS,
        "channel_order_source": "LEAD ADSZ dataset description",
        "source_sampling_rate_hz": args.fs,
        "source_preprocessing": "Dataset reports 1-30 Hz filtering and expert artifact removal",
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "window_size": window_size,
        "stride": stride,
        "shape": list(x.shape),
        "num_segments": int(len(y)),
        "num_subjects": int(len(np.unique(sub_ids))),
        "num_source_recordings": int(len(subject_summary)),
        "subject_id_definition": "Participant-level ID shared by paired eyes-closed/open recordings",
        "normalization": "per-window per-channel z-score" if args.normalize else "none (performed by downstream loader)",
        "segments_by_label": {str(label): int((y == label).sum()) for label in sorted(np.unique(y))},
        "subjects_by_label": {
            str(label): int(len(np.unique(sub_ids[y == label]))) for label in sorted(np.unique(y))
        },
        "subject_summary": subject_summary,
    }
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
    (output_dir / "summary.json").write_text(summary_text, encoding="utf-8")
    (output_dir / "preprocess_summary.json").write_text(summary_text, encoding="utf-8")
    return summary


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(description="Preprocess ADSZ with configurable windows.")
    parser.add_argument(
        "--data-dir",
        default="ADSZ/dataset/alzheimer",
    )
    parser.add_argument("--output-dir", default="ADSZ_processed_19ch_1s_v2")
    parser.add_argument("--fs", type=float, default=128.0)
    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=45.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--stride-sec", type=float, default=1.0)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--expected-subjects", type=int, default=24)
    args = parser.parse_args()

    summary = preprocess_adsz(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "subject_summary"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
