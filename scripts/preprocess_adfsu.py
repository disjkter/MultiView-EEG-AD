"""Preprocess ADFSU eyes-closed EEG into canonical 19-channel 1-second windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt


CHANNELS = [
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


def bandpass(data: np.ndarray, fs: float, lowcut: float, highcut: float, order: int) -> np.ndarray:
    """Filter continuous EEG along time; input shape is [channels, time]."""

    nyquist = 0.5 * fs
    if not 0 < lowcut < highcut < nyquist:
        raise ValueError(f"Invalid bandpass {lowcut}-{highcut} Hz for fs={fs}")
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, data, axis=-1).astype(np.float32, copy=False)


def segment_signal(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Segment a [channels, time] signal into [windows, channels, time]."""

    starts = range(0, data.shape[-1] - window_size + 1, stride)
    windows = [data[:, start : start + window_size] for start in starts]
    if not windows:
        raise ValueError(f"Signal length {data.shape[-1]} is shorter than {window_size}")
    return np.stack(windows).astype(np.float32, copy=False)


def load_subject(patient_dir: Path) -> np.ndarray:
    """Load and explicitly order one subject's channel text files."""

    channels: list[np.ndarray] = []
    for channel in CHANNELS:
        path = patient_dir / f"{channel}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing {channel} for {patient_dir}")
        values = np.loadtxt(path, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError(f"Expected one-dimensional channel file, got {values.shape}: {path}")
        channels.append(values)
    lengths = {len(values) for values in channels}
    if len(lengths) != 1:
        raise ValueError(f"Channel lengths differ for {patient_dir}: {sorted(lengths)}")
    return np.stack(channels)


def preprocess(args: argparse.Namespace) -> dict[str, object]:
    """Build the canonical ADFSU dataset and return its audit summary."""

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    window_size = int(round(args.window_sec * args.fs))
    stride = int(round(args.stride_sec * args.fs))

    all_x: list[np.ndarray] = []
    all_y: list[int] = []
    all_ids: list[str] = []
    subject_summary: list[dict[str, object]] = []
    for class_name, label in CLASS_MAP.items():
        class_dir = data_dir / class_name / "Eyes_closed"
        for patient_dir in sorted(class_dir.iterdir(), key=lambda path: path.name):
            if not patient_dir.is_dir():
                continue
            signal = load_subject(patient_dir)
            filtered = bandpass(signal, args.fs, args.lowcut, args.highcut, args.filter_order)
            windows = segment_signal(filtered, window_size, stride)
            subject_id = f"ADFSU_{class_name}_{patient_dir.name}"
            all_x.append(windows)
            all_y.extend([label] * len(windows))
            all_ids.extend([subject_id] * len(windows))
            subject_summary.append(
                {
                    "subject_id": subject_id,
                    "label": label,
                    "source_samples": int(signal.shape[-1]),
                    "num_windows": int(len(windows)),
                }
            )

    x = np.vstack(all_x).astype(np.float32, copy=False)
    y = np.asarray(all_y, dtype=np.int64)
    subject_ids = np.asarray(all_ids)
    if len(np.unique(subject_ids)) != args.expected_subjects:
        raise ValueError(
            f"Expected {args.expected_subjects} subjects, found {len(np.unique(subject_ids))}"
        )

    np.save(output_dir / "x_19ch.npy", x)
    np.save(output_dir / "y_19ch.npy", y)
    np.save(output_dir / "sub_ids_19ch.npy", subject_ids)
    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "shape": list(x.shape),
        "sampling_rate_hz": args.fs,
        "channels": CHANNELS,
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "filter_hz": [args.lowcut, args.highcut],
        "filter_order": args.filter_order,
        "normalization": "none (performed by downstream loader)",
        "subject_id_definition": "Class-prefixed source patient directory",
        "num_subjects": int(len(np.unique(subject_ids))),
        "segments_by_label": {
            str(label): int((y == label).sum()) for label in sorted(np.unique(y))
        },
        "subjects_by_label": {
            str(label): int(len(np.unique(subject_ids[y == label])))
            for label in sorted(np.unique(y))
        },
        "subject_summary": subject_summary,
    }
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
    (output_dir / "summary.json").write_text(summary_text, encoding="utf-8")
    return summary


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="ADFSU")
    parser.add_argument("--output-dir", default="ADFSU_processed_19ch_1s_v2")
    parser.add_argument("--fs", type=float, default=128.0)
    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=45.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--stride-sec", type=float, default=1.0)
    parser.add_argument("--expected-subjects", type=int, default=92)
    args = parser.parse_args()
    summary = preprocess(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "subject_summary"}, indent=2))


if __name__ == "__main__":
    main()
