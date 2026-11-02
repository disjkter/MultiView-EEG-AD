"""Pretrain GraphTemporalNet with masked node/time EEG reconstruction."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.GraphTemporalMAE import GraphTemporalMaskedAutoencoder  # noqa: E402
from utils.dual_stream_config import apply_yaml_defaults  # noqa: E402
from utils.source_weighted_sampling import (  # noqa: E402
    MultiSourceEEGDataset,
    SourceWeightedSampler,
    normalize_source_weights,
)


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_source_weights(mode: str, lengths: list[int], manual: list[float] | None) -> list[float]:
    """Resolve source-level probabilities for proportional, uniform, or manual sampling."""

    if mode == "proportional":
        weights = [float(length) for length in lengths]
    elif mode == "uniform":
        weights = [1.0 for _ in lengths]
    else:
        if manual is None or len(manual) != len(lengths):
            raise ValueError("--source-weights must match --data-dirs in manual mode.")
        weights = manual
    return normalize_source_weights(weights)


def encoder_state_dict(model: GraphTemporalMaskedAutoencoder) -> dict[str, torch.Tensor]:
    """Return downstream-compatible encoder weights without the pretraining classifier."""

    return {
        key: value.detach().cpu()
        for key, value in model.encoder.state_dict().items()
        if not key.startswith("fusion_classifier.")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/graph_temporal_mae.yaml")
    parser.add_argument("--data-dirs", nargs="+", default=None)
    parser.add_argument("--source-names", nargs="+", default=None)
    parser.add_argument("--sampling-mode", choices=["proportional", "uniform", "manual"], default="uniform")
    parser.add_argument("--source-weights", nargs="+", type=float, default=None)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/pretrain_graph_temporal_mae")
    parser.add_argument("--checkpoint-out", default="checkpoints/graph_temporal_mae.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--node-mask-ratio", type=float, default=0.3)
    parser.add_argument("--time-mask-ratio", type=float, default=0.3)
    parser.add_argument("--graph-out-dim", type=int, default=256)
    parser.add_argument("--temporal-out-dim", type=int, default=256)
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--graph-num-neighbors", type=int, default=4)
    parser.add_argument("--classifier-dropout", type=float, default=0.3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args = apply_yaml_defaults(
        parser,
        args,
        [
            "data_dirs",
            "source_names",
            "sampling_mode",
            "source_weights",
            "samples_per_epoch",
            "output_dir",
            "checkpoint_out",
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "node_mask_ratio",
            "time_mask_ratio",
            "graph_out_dim",
            "temporal_out_dim",
            "decoder_hidden_dim",
            "graph_num_neighbors",
            "classifier_dropout",
            "gradient_clip",
            "seed",
            "device",
        ],
    )
    if not args.data_dirs:
        raise ValueError("Provide at least one source directory through YAML or --data-dirs.")

    set_seed(args.seed)
    data_dirs = [Path(path) for path in args.data_dirs]
    source_names = args.source_names or [path.name for path in data_dirs]
    if len(source_names) != len(data_dirs) or len(set(source_names)) != len(source_names):
        raise ValueError("--source-names must be unique and match --data-dirs.")

    dataset = MultiSourceEEGDataset(
        [path / "x_19ch.npy" for path in data_dirs],
        source_names,
    )
    if dataset.sample_shape is None:
        raise RuntimeError("Could not determine the source EEG shape.")
    num_channels, seq_len = dataset.sample_shape
    source_weights = resolve_source_weights(
        args.sampling_mode,
        dataset.source_lengths,
        args.source_weights,
    )
    samples_per_epoch = args.samples_per_epoch or len(dataset)
    sampler = SourceWeightedSampler(
        dataset.source_lengths,
        source_weights,
        num_samples=samples_per_epoch,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = GraphTemporalMaskedAutoencoder(
        num_channels=num_channels,
        seq_len=seq_len,
        graph_out_dim=args.graph_out_dim,
        temporal_out_dim=args.temporal_out_dim,
        classifier_dropout=args.classifier_dropout,
        num_neighbors=args.graph_num_neighbors,
        decoder_hidden_dim=args.decoder_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    checkpoint_out = Path(args.checkpoint_out)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_loss = float("inf")
    best_payload: dict[str, object] | None = None
    reconstruction_example: dict[str, np.ndarray] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in loader:
            x = batch.to(device, non_blocking=True)
            output = model(
                x,
                node_mask_ratio=args.node_mask_ratio,
                time_mask_ratio=args.time_mask_ratio,
            )
            loss = model.masked_loss(output, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1

            if reconstruction_example is None:
                keep = min(8, x.shape[0])
                reconstruction_example = {
                    "target": x[:keep].detach().cpu().numpy(),
                    "masked_input": output.masked_input[:keep].detach().cpu().numpy(),
                    "reconstruction": output.reconstruction[:keep].detach().cpu().numpy(),
                    "mask": output.mask[:keep].detach().cpu().numpy(),
                }

        epoch_loss = total_loss / max(total_batches, 1)
        sampled_total = max(sum(sampler.last_domain_counts), 1)
        source_fractions = {
            name: float(count / sampled_total)
            for name, count in zip(source_names, sampler.last_domain_counts)
        }
        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "source_fractions": source_fractions,
            }
        )
        print(
            f"Epoch {epoch:03d} | masked_mse={epoch_loss:.6f} | "
            + " ".join(f"{name}={fraction:.3f}" for name, fraction in source_fractions.items())
        )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_payload = {
                "format": "graph_temporal_mae_v1",
                "encoder_state_dict": encoder_state_dict(model),
                "pretrain_config": vars(args),
                "sample_shape": [num_channels, seq_len],
                "best_loss": best_loss,
                "best_epoch": epoch,
                "source_names": source_names,
                "source_weights": source_weights,
            }

    if best_payload is None:
        raise RuntimeError("No Graph MAE checkpoint was produced.")
    torch.save(best_payload, checkpoint_out)
    if reconstruction_example is not None:
        np.savez_compressed(output_dir / "reconstruction_examples.npz", **reconstruction_example)

    summary = {
        "method": "graph_temporal_mae_v1",
        "data_dirs": [str(path) for path in data_dirs],
        "source_names": source_names,
        "source_lengths": dataset.source_lengths,
        "source_weights": source_weights,
        "sampling_mode": args.sampling_mode,
        "samples_per_epoch": samples_per_epoch,
        "sample_shape": [num_channels, seq_len],
        "node_mask_ratio": args.node_mask_ratio,
        "time_mask_ratio": args.time_mask_ratio,
        "checkpoint_out": str(checkpoint_out),
        "best_loss": best_payload["best_loss"],
        "best_epoch": best_payload["best_epoch"],
        "history": history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved checkpoint: {checkpoint_out}")
    print(f"Saved summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
