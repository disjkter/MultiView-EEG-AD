"""Self-supervised masked reconstruction pretraining for TAGE-Net encoders."""

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
        x = np.asarray(self.x[index], dtype=np.float32)
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-6
        x = (x - mean) / std
        return torch.from_numpy(x)


class TAGEReconstructionModel(nn.Module):
    """TAGE-Net encoder with a lightweight reconstruction decoder."""

    def __init__(
        self,
        num_channels: int,
        seq_len: int,
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
        )
        fused_dim = spatial_out_dim + temporal_out_dim
        self.decoder = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(fused_dim, num_channels * seq_len),
        )
        self.num_channels = num_channels
        self.seq_len = seq_len

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        """Reconstruct normalized EEG with shape [B, C, T]."""

        _, features = self.encoder(x_masked)
        recon = self.decoder(features)
        return recon.view(-1, self.num_channels, self.seq_len)


def set_seed(seed: int) -> None:
    """Set all random seeds used in this script."""

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
    """Train masked reconstruction and save encoder-compatible weights."""

    parser = argparse.ArgumentParser(description="Pretrain TAGE-Net encoders with masked reconstruction.")
    parser.add_argument("--data-dir", default="BrainLat19_pretrain_19ch_5s")
    parser.add_argument("--output-dir", default="outputs/pretrain_tage_brainlat19")
    parser.add_argument("--checkpoint-out", default="checkpoints/tage_selfsup_brainlat19.pth")
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

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    checkpoint_out = Path(args.checkpoint_out)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    x_path = data_dir / "x_19ch.npy"
    x_mem = np.load(x_path, mmap_mode="r")
    num_channels = int(x_mem.shape[1])
    seq_len = int(x_mem.shape[2])
    del x_mem

    dataset = EEGPretrainDataset(x_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TAGEReconstructionModel(
        num_channels=num_channels,
        seq_len=seq_len,
        spatial_out_dim=args.spatial_out_dim,
        temporal_out_dim=args.temporal_out_dim,
        spatial_hidden_channels=args.spatial_hidden_channels,
        spatial_temporal_channels=args.spatial_temporal_channels,
        classifier_dropout=args.classifier_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in loader:
            x = batch.to(device, non_blocking=True)
            mask = make_mask(x, args.mask_ratio, args.mask_mode)
            x_masked = x.masked_fill(mask, 0.0)
            recon = model(x_masked)
            loss = criterion(recon[mask], x[mask]) if mask.any() else criterion(recon, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1
        mean_loss = total_loss / max(total_batches, 1)
        history.append({"epoch": epoch, "loss": mean_loss})
        print(f"Epoch {epoch:03d} | loss={mean_loss:.6f}")
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
        "data_dir": str(data_dir),
        "checkpoint_out": str(checkpoint_out),
        "best_loss": best_loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "mask_ratio": args.mask_ratio,
        "mask_mode": args.mask_mode,
        "history": history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
