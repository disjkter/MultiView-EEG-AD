"""Preprocess CAUEEG into 19-channel AD-vs-HC numpy arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np


TARGET_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T3",
    "C3",
    "CZ",
    "C4",
    "T4",
    "T5",
    "P3",
    "PZ",
    "P4",
    "T6",
    "O1",
    "O2",
]


def load_items(dataset_dir: Path, split_file: str) -> list[dict[str, object]]:
    """Load all benchmark split items from a CAUEEG split JSON."""

    payload = json.loads((dataset_dir / split_file).read_text(encoding="utf-8"))
    items: list[dict[str, object]] = []
    for split_name in ("train_split", "validation_split", "test_split"):
        items.extend(payload.get(split_name, []))
    return items


def resolve_label(item: dict[str, object]) -> int | None:
    """Map CAUEEG benchmark item to HC=0, AD=1, or None for skipped samples."""

    class_name = str(item.get("class_name", "")).lower()
    symptoms = {str(symptom).lower() for symptom in item.get("symptom", [])}
    if class_name == "normal":
        return 0
    if class_name == "dementia" and "ad" in symptoms:
        return 1
    return None


def load_edf_segment(edf_path: Path, target_sfreq: float) -> np.ndarray:
    """Load one EDF file, select 19 EEG channels, and resample."""

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    expected_names = [f"{channel}-AVG" for channel in TARGET_CHANNELS]
    missing = [name for name in expected_names if name not in raw.ch_names]
    if missing:
        raise ValueError(f"{edf_path} is missing channels: {missing}")
    raw.pick_channels(expected_names, ordered=True)
    if abs(float(raw.info["sfreq"]) - target_sfreq) > 1e-6:
        raw.resample(target_sfreq, npad="auto", verbose="ERROR")
    return raw.get_data().astype(np.float32, copy=False)


def segment_continuous(data: np.ndarray, window_samples: int, stride_samples: int) -> list[np.ndarray]:
    """Slice one continuous EEG recording into fixed-length windows."""

    segments: list[np.ndarray] = []
    n_times = data.shape[-1]
    for start in range(0, n_times - window_samples + 1, stride_samples):
        segments.append(data[:, start : start + window_samples].copy())
    return segments


def main() -> None:
    """Convert CAUEEG to x/y/subject npy files for downstream experiments."""

    parser = argparse.ArgumentParser(description="Preprocess CAUEEG AD-vs-HC EEG data.")
    parser.add_argument("--input-dir", default="caueeg-dataset")
    parser.add_argument("--output-dir", default="CAUEEG_processed_19ch_ad_vs_hc_5s")
    parser.add_argument("--split-file", default="dementia.json")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--stride-seconds", type=float, default=5.0)
    parser.add_argument("--target-sfreq", type=float, default=128.0)
    parser.add_argument("--max-records", type=int, default=None, help="Optional debug limit.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window_samples = int(round(args.window_seconds * args.target_sfreq))
    stride_samples = int(round(args.stride_seconds * args.target_sfreq))
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("Window and stride must be positive.")

    items = load_items(input_dir, args.split_file)
    skipped: dict[str, str] = {}
    selected_items: list[tuple[dict[str, object], int, Path]] = []
    for item in items:
        serial = str(item["serial"])
        label = resolve_label(item)
        if label is None:
            skipped[serial] = str(item.get("symptom", item.get("class_name", "unknown")))
            continue
        edf_path = input_dir / "signal" / "edf" / f"{serial}.edf"
        if not edf_path.exists():
            skipped[serial] = "missing_edf"
            continue
        selected_items.append((item, label, edf_path))
        if args.max_records is not None and len(selected_items) >= args.max_records:
            break

    if not selected_items:
        raise RuntimeError("No EEG segments were generated.")

    record_meta: list[tuple[str, int, Path, int]] = []
    total_segments = 0
    for _, label, edf_path in selected_items:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        n_times = int(round(raw.n_times * args.target_sfreq / float(raw.info["sfreq"])))
        n_segments = max(0, (n_times - window_samples) // stride_samples + 1)
        serial = edf_path.stem
        record_meta.append((serial, label, edf_path, n_segments))
        total_segments += n_segments

    x_path = output_dir / "x_19ch.npy"
    y_path = output_dir / "y_19ch.npy"
    sub_path = output_dir / "sub_ids_19ch.npy"
    x = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_segments, len(TARGET_CHANNELS), window_samples),
    )
    y = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.int64, shape=(total_segments,))
    sub_ids = np.lib.format.open_memmap(sub_path, mode="w+", dtype=f"<U{max(len(s) for s, _, _, _ in record_meta)}", shape=(total_segments,))

    cursor = 0
    for serial, label, edf_path, expected_segments in record_meta:
        if expected_segments == 0:
            print(f"{serial}: label={label} segments=0")
            continue
        data = load_edf_segment(edf_path, args.target_sfreq)
        record_segments = segment_continuous(data, window_samples, stride_samples)
        n_segments = len(record_segments)
        if n_segments:
            x[cursor : cursor + n_segments] = np.stack(record_segments).astype(np.float32, copy=False)
            y[cursor : cursor + n_segments] = label
            sub_ids[cursor : cursor + n_segments] = serial
            cursor += n_segments
        print(f"{serial}: label={label} segments={n_segments}")

    if cursor != total_segments:
        x.flush()
        y.flush()
        sub_ids.flush()
        raise RuntimeError(f"Segment count mismatch: wrote {cursor}, expected {total_segments}")
    x.flush()
    y.flush()
    sub_ids.flush()

    y_arr = np.asarray(y)
    sub_arr = np.asarray(sub_ids)

    summary = {
        "input_dir": str(input_dir),
        "split_file": args.split_file,
        "task": "AD-vs-HC",
        "label_mapping": {"0": "Normal/HC", "1": "AD"},
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "target_sfreq": args.target_sfreq,
        "window_samples": window_samples,
        "stride_samples": stride_samples,
        "num_segments": int(len(y_arr)),
        "num_subjects": int(len(np.unique(sub_arr))),
        "segments_by_label": {str(label): int((y_arr == label).sum()) for label in sorted(np.unique(y_arr))},
        "subjects_by_label": {
            str(label): int(len(np.unique(sub_arr[y_arr == label]))) for label in sorted(np.unique(y_arr))
        },
        "channels": TARGET_CHANNELS,
        "shape": [int(dim) for dim in x.shape],
        "num_skipped_records": len(skipped),
        "skipped_examples": dict(list(skipped.items())[:20]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
