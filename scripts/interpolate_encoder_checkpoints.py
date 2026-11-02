"""Linearly interpolate compatible dual-stream encoder checkpoint tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ENCODER_PREFIXES = (
    "spatial_expert.",
    "femba.",
    "neuro_prior.",
    "prior_projector.",
    "sampling_rate_embedding.",
)


def checkpoint_state(payload: object) -> dict[str, torch.Tensor]:
    """Extract a plain tensor state dictionary."""

    if isinstance(payload, dict) and "encoder_state_dict" in payload:
        payload = payload["encoder_state_dict"]
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a state dictionary.")
    return {str(key): value for key, value in payload.items() if isinstance(value, torch.Tensor)}


def interpolate_encoder_states(
    ssl_state: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
    source_weight: float,
) -> dict[str, torch.Tensor]:
    """Blend matching encoder tensors while preserving tensor dtypes."""

    if not 0.0 <= source_weight <= 1.0:
        raise ValueError("source_weight must lie in [0, 1].")
    result: dict[str, torch.Tensor] = {}
    for key, ssl_value in ssl_state.items():
        if not key.startswith(ENCODER_PREFIXES):
            continue
        source_value = source_state.get(key)
        if source_value is None or source_value.shape != ssl_value.shape:
            continue
        if torch.is_floating_point(ssl_value):
            result[key] = (1.0 - source_weight) * ssl_value + source_weight * source_value.to(
                dtype=ssl_value.dtype
            )
        else:
            result[key] = (source_value if source_weight >= 0.5 else ssl_value).clone()
    if not result:
        raise ValueError("No compatible dual-stream encoder tensors were found.")
    return result


def main() -> None:
    """Create one interpolated encoder checkpoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssl-checkpoint", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-weight", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ssl_state = checkpoint_state(torch.load(args.ssl_checkpoint, map_location="cpu"))
    source_state = checkpoint_state(torch.load(args.source_checkpoint, map_location="cpu"))
    interpolated = interpolate_encoder_states(ssl_state, source_state, args.source_weight)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(interpolated, output)
    print(f"Saved {len(interpolated)} interpolated encoder tensors to {output}")


if __name__ == "__main__":
    main()

