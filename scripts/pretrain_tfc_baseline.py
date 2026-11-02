"""Pretrain the external TF-C baseline on the shared unlabeled EEG corpus."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.TFCBaseline import TFCEncoder  # noqa: E402
from utils.source_weighted_sampling import MultiSourceEEGDataset  # noqa: E402
from utils.dual_stream_config import apply_yaml_defaults  # noqa: E402


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def time_augmentation(x: torch.Tensor, jitter_std: float) -> torch.Tensor:
    """Apply TF-C time-space jitter to [B, C, T] EEG windows."""

    return x + torch.randn_like(x) * jitter_std


def frequency_augmentation(
    frequency: torch.Tensor, remove_ratio: float, add_ratio: float
) -> torch.Tensor:
    """Remove and perturb randomly selected frequency coefficients."""

    keep = torch.rand_like(frequency) >= remove_ratio
    add = torch.rand_like(frequency) < add_ratio
    amplitude = frequency.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    noise = torch.rand_like(frequency) * amplitude * 0.1
    return frequency * keep + noise * add


def nt_xent(first: torch.Tensor, second: torch.Tensor, temperature: float) -> torch.Tensor:
    """Compute symmetric normalized temperature-scaled cross entropy."""

    if first.shape != second.shape:
        raise ValueError("Contrastive views must have matching shapes")
    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    representations = torch.cat([first, second], dim=0)
    logits = representations @ representations.T / temperature
    batch_size = first.shape[0]
    diagonal = torch.eye(2 * batch_size, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(diagonal, -torch.inf)
    targets = torch.arange(2 * batch_size, device=logits.device)
    targets = (targets + batch_size) % (2 * batch_size)
    return F.cross_entropy(logits, targets)


def main() -> None:
    """Run TF-C pretraining and save the best encoder checkpoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dirs", nargs="+", required=True)
    parser.add_argument("--source-names", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-out", required=True)
    parser.add_argument("--config", default="configs/tfc_baseline_jbhi.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--within-domain-weight", type=float, default=0.2)
    parser.add_argument("--jitter-std", type=float, default=0.02)
    parser.add_argument("--frequency-remove-ratio", type=float, default=0.1)
    parser.add_argument("--frequency-add-ratio", type=float, default=0.1)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args = apply_yaml_defaults(
        parser,
        args,
        [
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "temperature",
            "within_domain_weight",
            "jitter_std",
            "frequency_remove_ratio",
            "frequency_add_ratio",
            "model_dim",
            "projection_dim",
            "num_layers",
            "num_heads",
            "dropout",
            "seed",
        ],
    )

    if len(args.data_dirs) != len(args.source_names):
        raise ValueError("--data-dirs and --source-names must have the same length")
    if args.temperature <= 0 or args.within_domain_weight < 0:
        raise ValueError("Temperature must be positive and loss weights non-negative")
    for ratio in (args.frequency_remove_ratio, args.frequency_add_ratio):
        if not 0 <= ratio <= 1:
            raise ValueError("Frequency perturbation ratios must lie in [0, 1]")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    checkpoint_out = Path(args.checkpoint_out)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    x_paths = [Path(data_dir) / "x_19ch.npy" for data_dir in args.data_dirs]
    dataset = MultiSourceEEGDataset(x_paths, args.source_names)
    if dataset.sample_shape is None:
        raise RuntimeError("Could not determine the TF-C input shape")
    num_channels, seq_len = dataset.sample_shape
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TFCEncoder(
        num_channels=num_channels,
        seq_len=seq_len,
        model_dim=args.model_dim,
        projection_dim=args.projection_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"TF-C setup | device={device} sources={dataset.source_lengths} "
        f"samples={len(dataset)} batches_per_epoch={len(loader)}",
        flush=True,
    )

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "time": 0.0, "frequency": 0.0, "cross_space": 0.0}
        batches = 0
        for x_time in loader:
            x_time = x_time.to(device, non_blocking=True)
            x_frequency = model.frequency_view(x_time)
            x_time_aug = time_augmentation(x_time, args.jitter_std)
            x_frequency_aug = frequency_augmentation(
                x_frequency, args.frequency_remove_ratio, args.frequency_add_ratio
            )

            h_time, z_time, h_frequency, z_frequency = model(x_time, x_frequency)
            h_time_aug, _, h_frequency_aug, _ = model(x_time_aug, x_frequency_aug)
            time_loss = nt_xent(h_time, h_time_aug, args.temperature)
            frequency_loss = nt_xent(h_frequency, h_frequency_aug, args.temperature)
            cross_space_loss = nt_xent(z_time, z_frequency, args.temperature)
            loss = args.within_domain_weight * (time_loss + frequency_loss) + cross_space_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            totals["loss"] += float(loss.detach())
            totals["time"] += float(time_loss.detach())
            totals["frequency"] += float(frequency_loss.detach())
            totals["cross_space"] += float(cross_space_loss.detach())
            batches += 1

        if not batches:
            raise RuntimeError("No full TF-C pretraining batch was available")
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({name: value / batches for name, value in totals.items()})
        history.append(row)
        print(
            f"Epoch {epoch:03d} | loss={row['loss']:.6f} time={row['time']:.6f} "
            f"frequency={row['frequency']:.6f} cross={row['cross_space']:.6f}",
            flush=True,
        )
        if float(row["loss"]) < best_loss:
            best_loss = float(row["loss"])
            torch.save(
                {
                    "method": "TF-C",
                    "source": "https://github.com/mims-harvard/TFC-pretraining",
                    "encoder_state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "model_config": {
                        "num_channels": num_channels,
                        "seq_len": seq_len,
                        "model_dim": args.model_dim,
                        "projection_dim": args.projection_dim,
                        "num_layers": args.num_layers,
                        "num_heads": args.num_heads,
                        "dropout": args.dropout,
                    },
                },
                checkpoint_out,
            )

    summary = {
        "method": "TF-C",
        "implementation_basis": "official MIT-licensed repository adapted to 19-channel EEG",
        "official_repository": "https://github.com/mims-harvard/TFC-pretraining",
        "data_dirs": args.data_dirs,
        "source_names": args.source_names,
        "source_lengths": dataset.source_lengths,
        "sample_shape": list(dataset.sample_shape),
        "checkpoint_out": str(checkpoint_out),
        "device": str(device),
        "best_loss": best_loss,
        "config": vars(args),
        "history": history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
