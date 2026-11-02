"""Audit standardized EEG datasets for format, metadata, and subject integrity."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


CANONICAL_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]
CHANNEL_ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
DEFAULT_DATASETS = [
    "CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
    "ds004504_processed_19ch_1s_v2",
    "BrainLat19_pretrain_19ch_1s_lead",
    "auditory40hz_processed_19ch_1s_lead",
    "ADSZ_processed_19ch_1s_v2",
    "ADFSU_processed_19ch_1s_v2",
]


def normalize_channel(channel: object) -> str:
    """Normalize capitalization and legacy temporal-channel aliases."""

    value = str(channel).strip()
    canonical_case = {name.lower(): name for name in CANONICAL_CHANNELS}
    value = CHANNEL_ALIASES.get(value.upper(), value)
    return canonical_case.get(value.lower(), value)


def load_metadata(dataset_dir: Path) -> tuple[dict[str, object], Path | None]:
    """Load local metadata and optionally inherit metadata from its input directory."""

    summary_path = dataset_dir / "summary.json"
    if not summary_path.exists():
        return {}, None
    metadata = json.loads(summary_path.read_text(encoding="utf-8"))
    input_dir = metadata.get("input_dir") or metadata.get("data_dir")
    if "channels" not in metadata and input_dir:
        parent_summary = Path(str(input_dir)) / "summary.json"
        if parent_summary.exists():
            parent = json.loads(parent_summary.read_text(encoding="utf-8"))
            if "channels" in parent:
                metadata["channels"] = parent["channels"]
    return metadata, summary_path


def check_finite(x: np.ndarray, chunk_size: int) -> bool:
    """Check all signal values without loading the complete array into RAM."""

    for start in range(0, len(x), chunk_size):
        if not np.isfinite(np.asarray(x[start : start + chunk_size])).all():
            return False
    return True


def audit_dataset(dataset_dir: Path, chunk_size: int) -> dict[str, object]:
    """Return one structured audit record."""

    errors: list[str] = []
    warnings: list[str] = []
    x_path = dataset_dir / "x_19ch.npy"
    y_path = dataset_dir / "y_19ch.npy"
    subject_path = dataset_dir / "sub_ids_19ch.npy"
    missing = [path.name for path in (x_path, y_path, subject_path) if not path.exists()]
    if missing:
        return {"dataset": str(dataset_dir), "status": "error", "errors": [f"Missing files: {missing}"]}

    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path).astype(np.int64, copy=False)
    subject_ids = np.load(subject_path, allow_pickle=True)
    if x.ndim != 3 or tuple(x.shape[1:]) != (19, 128):
        errors.append(f"Expected [N, 19, 128], got {list(x.shape)}")
    if not (len(x) == len(y) == len(subject_ids)):
        errors.append(f"Sample counts differ: x={len(x)}, y={len(y)}, subjects={len(subject_ids)}")
    finite = check_finite(x, chunk_size)
    if not finite:
        errors.append("Signal contains NaN or Inf")

    subject_labels: dict[str, set[int]] = {}
    for subject, label in zip(subject_ids, y):
        subject_labels.setdefault(str(subject), set()).add(int(label))
    mixed = sorted(subject for subject, labels in subject_labels.items() if len(labels) > 1)
    if mixed:
        errors.append(f"{len(mixed)} subject IDs have mixed labels")

    metadata, summary_path = load_metadata(dataset_dir)
    sampling_rate = metadata.get("target_sfreq", metadata.get("sampling_rate_hz", metadata.get("fs")))
    if sampling_rate is None:
        warnings.append("Sampling-rate metadata is missing")
    elif float(sampling_rate) != 128.0:
        errors.append(f"Expected 128 Hz metadata, got {sampling_rate}")

    channels = metadata.get("channels")
    normalized_channels = None
    if channels is None:
        warnings.append("Channel names/order metadata is missing; shape alone cannot prove alignment")
    else:
        normalized_channels = [normalize_channel(channel) for channel in channels]
        if normalized_channels != CANONICAL_CHANNELS:
            errors.append(f"Channel order differs from canonical order: {normalized_channels}")

    status = "error" if errors else "warning" if warnings else "pass"
    return {
        "dataset": str(dataset_dir),
        "status": status,
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite": finite,
        "num_subjects": len(subject_labels),
        "segments_by_label": {str(k): int(v) for k, v in sorted(Counter(map(int, y)).items())},
        "subjects_by_label": {
            str(label): sum(labels == {label} for labels in subject_labels.values())
            for label in sorted(set(map(int, y)))
        },
        "sampling_rate_hz": sampling_rate,
        "channels": normalized_channels,
        "summary_file": str(summary_path) if summary_path else None,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    """Audit all requested datasets and write a reusable JSON report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--output", default="outputs/dataset_audit.json")
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()

    results = [audit_dataset(Path(dataset), args.chunk_size) for dataset in args.datasets]
    for result in results:
        print(
            f"{result['status'].upper():7s} {result['dataset']}: "
            f"shape={result.get('shape')} subjects={result.get('num_subjects')}"
        )
        for message in result.get("errors", []):
            print(f"  ERROR: {message}")
        for message in result.get("warnings", []):
            print(f"  WARN : {message}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {output_path}")
    if any(result["status"] == "error" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
