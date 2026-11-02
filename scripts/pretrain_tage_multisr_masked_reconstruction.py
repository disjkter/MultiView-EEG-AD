"""Multi-sampling-rate masked reconstruction pretraining for TAGE-Net."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.Spatial_MSCNN import SpatialMSCNN_DualStream


class EEGPretrainDataset(Dataset):
    """Memory-mapped EEG dataset for reconstruction pretraining."""

    def __init__(self, x_path: Path) -> None:
        self.x = np.load(x_path, mmap_mode="r")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return normalized EEG with shape [C, T]."""

        x = np.asarray(self.x[index], dtype=np.float32)
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-6
        return torch.from_numpy((x - mean) / std)


class MultiSRReconstructionModel(nn.Module):
    """Shared TAGE encoder with one reconstruction decoder per sequence length."""

    def __init__(
        self,
        num_channels: int,
        seq_lens: list[int],
        sampling_rate_options: list[float],
        spatial_out_dim: int,
        temporal_out_dim: int,
        spatial_hidden_channels: int,
        spatial_temporal_channels: int,
        classifier_dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = SpatialMSCNN_DualStream(
            num_classes=2,
            num_channels=num_channels,
            spatial_out_dim=spatial_out_dim,
            temporal_out_dim=temporal_out_dim,
            spatial_hidden_channels=spatial_hidden_channels,
            spatial_temporal_channels=spatial_temporal_channels,
            classifier_dropout=classifier_dropout,
            use_neuro_prior=False,
            sampling_rate=sampling_rate_options[0],
            use_sampling_rate_embedding=True,
            sampling_rate_options=tuple(sampling_rate_options),
        )
        fused_dim = spatial_out_dim + temporal_out_dim
        self.decoders = nn.ModuleDict(
            {
                str(seq_len): nn.Sequential(
                    nn.Linear(fused_dim, fused_dim),
                    nn.GELU(),
                    nn.Dropout(classifier_dropout),
                    nn.Linear(fused_dim, num_channels * seq_len),
                )
                for seq_len in sorted(set(seq_lens))
            }
        )
        self.num_channels = num_channels

    def forward(self, x_masked: torch.Tensor, sampling_rate: float) -> torch.Tensor:
        """Reconstruct normalized EEG with shape [B, C, T]."""

        _, features = self.encoder(x_masked, sampling_rate=sampling_rate)
        seq_len = int(x_masked.shape[-1])
        recon = self.decoders[str(seq_len)](features)
        return recon.view(-1, self.num_channels, seq_len)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible pretraining."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_mask(x: torch.Tensor, mask_ratio: float, mode: str) -> torch.Tensor:
    """Create a binary mask with True entries indicating masked values."""

    mask = torch.zeros_like(x, dtype=torch.bool)
    batch_size, num_channels, seq_len = x.shape
    if mode in {"channel", "hybrid"}:
        n_channels = max(1, int(round(num_channels * mask_ratio)))
        for batch_idx in range(batch_size):
            idx = torch.randperm(num_channels, device=x.device)[:n_channels]
            mask[batch_idx, idx, :] = True
    if mode in {"time", "hybrid"}:
        n_times = max(1, int(round(seq_len * mask_ratio)))
        for batch_idx in range(batch_size):
            idx = torch.randperm(seq_len, device=x.device)[:n_times]
            mask[batch_idx, :, idx] = True
    return mask


def main() -> None:
    """Train one shared encoder over multiple sampling rates."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dirs", nargs="+", required=True)
    parser.add_argument("--sampling-rates", nargs="+", type=float, required=True)
    parser.add_argument("--domain-names", nargs="+", default=None)
    parser.add_argument("--output-dir", default="outputs/pretrain_tage_multisr")
    parser.add_argument("--checkpoint-out", default="checkpoints/tage_selfsup_multisr.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--mask-mode", choices=["channel", "time", "hybrid"], default="hybrid")
    parser.add_argument("--spatial-out-dim", type=int, default=256)
    parser.add_argument("--temporal-out-dim", type=int, default=256)
    parser.add_argument("--spatial-hidden-channels", type=int, default=64)
    parser.add_argument("--spatial-temporal-channels", type=int, default=128)
    parser.add_argument("--classifier-dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if len(args.data_dirs) != len(args.sampling_rates):
        raise ValueError("--data-dirs and --sampling-rates must have the same length")
    domain_names = args.domain_names or [Path(path).name for path in args.data_dirs]
    if len(domain_names) != len(args.data_dirs):
        raise ValueError("--domain-names must have the same length as --data-dirs")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    checkpoint_out = Path(args.checkpoint_out)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    loaders: list[tuple[str, float, DataLoader]] = []
    domain_summaries: list[dict[str, object]] = []
    num_channels: int | None = None
    seq_lens: list[int] = []
    for data_dir_text, domain_name, sampling_rate in zip(args.data_dirs, domain_names, args.sampling_rates):
        data_dir = Path(data_dir_text)
        x_path = data_dir / "x_19ch.npy"
        x_mem = np.load(x_path, mmap_mode="r")
        if num_channels is None:
            num_channels = int(x_mem.shape[1])
        elif int(x_mem.shape[1]) != num_channels:
            raise ValueError(f"Channel mismatch in {data_dir}: expected {num_channels}, got {x_mem.shape[1]}")
        seq_len = int(x_mem.shape[2])
        seq_lens.append(seq_len)
        domain_summaries.append(
            {
                "name": domain_name,
                "data_dir": str(data_dir),
                "sampling_rate": float(sampling_rate),
                "shape": list(x_mem.shape),
            }
        )
        del x_mem
        dataset = EEGPretrainDataset(x_path)
        loaders.append(
            (
                domain_name,
                float(sampling_rate),
                DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=torch.cuda.is_available(),
                ),
            )
        )

    if num_channels is None:
        raise RuntimeError("No pretraining datasets were loaded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiSRReconstructionModel(
        num_channels=num_channels,
        seq_lens=seq_lens,
        sampling_rate_options=sorted(set(float(rate) for rate in args.sampling_rates)),
        spatial_out_dim=args.spatial_out_dim,
        temporal_out_dim=args.temporal_out_dim,
        spatial_hidden_channels=args.spatial_hidden_channels,
        spatial_temporal_channels=args.spatial_temporal_channels,
        classifier_dropout=args.classifier_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    history: list[dict[str, object]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        domain_losses: dict[str, float] = {}
        total_loss = 0.0
        total_batches = 0
        for domain_name, sampling_rate, loader in loaders:
            running = 0.0
            batches = 0
            for batch in loader:
                x = batch.to(device, non_blocking=True)
                mask = make_mask(x, args.mask_ratio, args.mask_mode)
                x_masked = x.masked_fill(mask, 0.0)
                recon = model(x_masked, sampling_rate=sampling_rate)
                loss = criterion(recon[mask], x[mask]) if mask.any() else criterion(recon, x)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running += float(loss.detach().cpu())
                total_loss += float(loss.detach().cpu())
                batches += 1
                total_batches += 1
            domain_losses[domain_name] = running / max(batches, 1)

        mean_loss = total_loss / max(total_batches, 1)
        history.append({"epoch": epoch, "loss": mean_loss, "domain_losses": domain_losses})
        print(f"Epoch {epoch:03d} | loss={mean_loss:.6f} | " + " ".join(f"{k}={v:.6f}" for k, v in domain_losses.items()))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {
                key.replace("encoder.", "", 1): value.detach().cpu()
                for key, value in model.state_dict().items()
                if key.startswith("encoder.") and not key.startswith("encoder.fusion_classifier")
            }

    if best_state is None:
        raise RuntimeError("No checkpoint was produced.")
    torch.save(best_state, checkpoint_out)
    summary = {
        "data_dirs": args.data_dirs,
        "domains": domain_summaries,
        "sampling_rates": args.sampling_rates,
        "checkpoint_out": str(checkpoint_out),
        "best_loss": best_loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "mask_ratio": args.mask_ratio,
        "mask_mode": args.mask_mode,
        "use_sampling_rate_embedding": True,
        "history": history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
