"""Add missing channel provenance to existing known aligned datasets without rewriting arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]
DATASETS = {
    "auditory40hz_processed_19ch_rest_ad_vs_normal": (
        "Per-subject BIDS channels.tsv; preprocess_40hz_auditory.py explicitly reorders channels"
    ),
    "auditory40hz_processed_19ch_1s_lead": (
        "Inherited from auditory40hz_processed_19ch_rest_ad_vs_normal"
    ),
    "BrainLat19_pretrain_19ch_5s": (
        "LEAD aligned 19-channel feature package and dataset data card"
    ),
    "BrainLat19_pretrain_19ch_1s_lead": (
        "Inherited from BrainLat19_pretrain_19ch_5s"
    ),
}
OPTIONAL_INTERMEDIATE_DIRS = {
    "auditory40hz_processed_19ch_rest_ad_vs_normal",
    "BrainLat19_pretrain_19ch_5s",
}


def main() -> None:
    """Validate channel dimension and update only summary metadata."""

    for directory_name, provenance in DATASETS.items():
        directory = Path(directory_name)
        x_path = directory / "x_19ch.npy"
        summary_path = directory / "summary.json"
        if not directory.exists():
            print(f"Skipped missing optional directory: {directory}")
            continue
        if not x_path.exists() or not summary_path.exists():
            if directory_name in OPTIONAL_INTERMEDIATE_DIRS:
                print(f"Skipped incomplete optional directory: {directory}")
                continue
            raise FileNotFoundError(f"Missing x_19ch.npy or summary.json under {directory}")
        x = np.load(x_path, mmap_mode="r")
        if x.ndim != 3 or x.shape[1] != len(CHANNELS):
            raise ValueError(f"Expected 19 channels under {directory}, got {x.shape}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        existing = summary.get("channels")
        if existing is not None and existing != CHANNELS:
            raise ValueError(f"Refusing to replace conflicting channel metadata in {summary_path}")
        summary["channels"] = CHANNELS
        summary["channel_order_source"] = provenance
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Updated {summary_path}")


if __name__ == "__main__":
    main()
