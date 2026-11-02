"""Interpolate a processed four-channel dataset onto a 19-channel 10-20 layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np


TARGET_19 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]


def get_positions(ch_names: list[str]) -> np.ndarray:
    """Get standard-1020 xyz positions for channel names."""

    montage = mne.channels.make_standard_montage("standard_1020")
    ch_pos = montage.get_positions()["ch_pos"]
    positions = []
    for name in ch_names:
        if name not in ch_pos:
            raise KeyError(f"Channel {name!r} not found in standard_1020 montage")
        positions.append(np.asarray(ch_pos[name], dtype=np.float32))
    return np.stack(positions, axis=0)


def build_interpolation_weights(source_names: list[str], target_names: list[str], power: float) -> np.ndarray:
    """Build inverse-distance weights from source to target electrode positions."""

    src_pos = get_positions(source_names)
    tgt_pos = get_positions(target_names)
    weights = np.zeros((len(target_names), len(source_names)), dtype=np.float32)

    for i, target in enumerate(tgt_pos):
        dists = np.linalg.norm(src_pos - target[None, :], axis=1)
        if np.any(dists < 1e-8):
            weights[i, np.argmin(dists)] = 1.0
        else:
            inv = 1.0 / np.power(np.maximum(dists, 1e-6), power)
            weights[i] = inv / np.sum(inv)
    return weights


def interpolate_dataset(x_4ch: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Interpolate [N, 4, T] into [N, 19, T]."""

    return np.einsum("kc,nct->nkt", weights, x_4ch, optimize=True).astype(np.float32)


def main() -> None:
    """Entry point."""

    parser = argparse.ArgumentParser(description="Interpolate 4-channel EEG to 19-channel layout.")
    parser.add_argument("--input-dir", required=True, help="Processed dataset directory containing x.npy/y.npy/sub_ids.npy")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source-channels",
        nargs=4,
        required=True,
        help="Four source channel names in standard 10-20 nomenclature, e.g. Fp1 Fp2 O1 O2",
    )
    parser.add_argument("--power", type=float, default=2.0, help="Inverse-distance weighting power")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.load(input_dir / "x.npy")
    y = np.load(input_dir / "y.npy")
    sub_ids = np.load(input_dir / "sub_ids.npy", allow_pickle=True)

    if x.shape[1] != 4:
        raise ValueError(f"Expected 4 input channels, got shape {x.shape}")

    weights = build_interpolation_weights(args.source_channels, TARGET_19, power=args.power)
    x_interp = interpolate_dataset(x, weights)

    np.save(output_dir / "x_19ch.npy", x_interp)
    np.save(output_dir / "y_19ch.npy", y.astype(np.int64))
    np.save(output_dir / "sub_ids_19ch.npy", sub_ids)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_channels": args.source_channels,
        "target_channels": TARGET_19,
        "power": args.power,
        "input_shape": list(x.shape),
        "output_shape": list(x_interp.shape),
        "weights": weights.tolist(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
