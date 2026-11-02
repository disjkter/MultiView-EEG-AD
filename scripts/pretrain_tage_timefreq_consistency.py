"""Time-frequency reconstruction and temporal-spatial consistency pretraining."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.Spatial_MSCNN import SpatialMSCNN_DualStream
from utils.source_weighted_sampling import (
    MultiSourceEEGDataset,
    SourceWeightedSampler,
    load_similarity_weights,
    normalize_source_weights,
)


class TAGETimeFreqConsistencyModel(nn.Module):
    """TAGE encoder with time/frequency decoders and branch-consistency heads."""

    def __init__(
        self,
        num_channels: int,
        seq_len: int,
        spatial_out_dim: int,
        temporal_out_dim: int,
        spatial_hidden_channels: int,
        spatial_temporal_channels: int,
        classifier_dropout: float,
        projection_dim: int,
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
        freq_bins = seq_len // 2 + 1
        self.time_decoder = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(fused_dim, num_channels * seq_len),
        )
        self.freq_decoder = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(fused_dim, num_channels * freq_bins),
        )
        self.spatial_projector = nn.Sequential(
            nn.Linear(spatial_out_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.temporal_projector = nn.Sequential(
            nn.Linear(temporal_out_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.view_projector = nn.Sequential(
            nn.Linear(fused_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.freq_bins = freq_bins

    def encode_parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return spatial, temporal, and fused features."""

        feat_spatial = self.encoder.spatial_expert(x)
        feat_temporal = self.encoder.femba(x)
        fused = torch.cat([feat_spatial, feat_temporal], dim=-1)
        return feat_spatial, feat_temporal, fused

    def forward(self, x_masked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstructed signal, spectrum, and branch projections."""

        feat_spatial, feat_temporal, fused = self.encode_parts(x_masked)
        recon_time = self.time_decoder(fused).view(-1, self.num_channels, self.seq_len)
        recon_freq = self.freq_decoder(fused).view(-1, self.num_channels, self.freq_bins)
        z_spatial = self.spatial_projector(feat_spatial)
        z_temporal = self.temporal_projector(feat_temporal)
        return recon_time, recon_freq, z_spatial, z_temporal, fused

    def project_view(self, fused: torch.Tensor) -> torch.Tensor:
        """Project fused features for full-window versus crop-view consistency."""

        return self.view_projector(fused)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible pretraining."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_mask(x: torch.Tensor, mask_ratio: float, mode: str) -> torch.Tensor:
    """Create a binary mask where True marks values to reconstruct."""

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


def spectral_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Return log-normalized one-sided spectrum magnitude with shape [B, C, F]."""

    spec = torch.fft.rfft(x, dim=-1)
    mag = torch.log1p(torch.abs(spec))
    mean = mag.mean(dim=-1, keepdim=True)
    std = mag.std(dim=-1, keepdim=True) + 1e-6
    return (mag - mean) / std


def branch_consistency_loss(z_spatial: torch.Tensor, z_temporal: torch.Tensor) -> torch.Tensor:
    """Align spatial and temporal branch embeddings without negative samples."""

    z_spatial = F.normalize(z_spatial, dim=-1)
    z_temporal = F.normalize(z_temporal, dim=-1)
    return 1.0 - (z_spatial * z_temporal).sum(dim=-1).mean()


def random_temporal_crop_resize(x: torch.Tensor, min_ratio: float, max_ratio: float) -> torch.Tensor:
    """Randomly crop each sample in time and resize it back to the original length."""

    batch_size, _, seq_len = x.shape
    min_len = max(2, int(round(seq_len * min_ratio)))
    max_len = max(min_len, int(round(seq_len * max_ratio)))
    max_len = min(max_len, seq_len)
    crops = []
    for batch_idx in range(batch_size):
        crop_len = int(torch.randint(min_len, max_len + 1, (1,), device=x.device).item())
        start_max = seq_len - crop_len
        start = int(torch.randint(0, start_max + 1, (1,), device=x.device).item()) if start_max > 0 else 0
        crop = x[batch_idx : batch_idx + 1, :, start : start + crop_len]
        crop = F.interpolate(crop, size=seq_len, mode="linear", align_corners=False)
        crops.append(crop)
    return torch.cat(crops, dim=0)


def resolve_sources(args: argparse.Namespace) -> tuple[list[Path], list[str]]:
    """Resolve backward-compatible single-source and new multi-source arguments."""

    data_dirs = [Path(path) for path in args.data_dirs] if args.data_dirs else [Path(args.data_dir)]
    source_names = args.source_names or [path.name for path in data_dirs]
    if len(source_names) != len(data_dirs):
        raise ValueError("--source-names must have the same length as --data-dirs.")
    if len(set(source_names)) != len(source_names):
        raise ValueError("--source-names must be unique.")
    return data_dirs, list(source_names)


def resolve_source_weights(
    args: argparse.Namespace,
    source_names: list[str],
    source_lengths: list[int],
) -> list[float]:
    """Build source-level probabilities for the requested sampling strategy."""

    if args.sampling_mode == "proportional":
        raw_weights = [float(length) for length in source_lengths]
    elif args.sampling_mode == "uniform":
        raw_weights = [1.0 for _ in source_names]
    elif args.sampling_mode == "manual":
        if args.source_weights is None or len(args.source_weights) != len(source_names):
            raise ValueError("--source-weights must match --source-names in manual mode.")
        raw_weights = args.source_weights
    else:
        if not args.similarity_json or not args.target_name:
            raise ValueError("Similarity mode requires --similarity-json and --target-name.")
        return load_similarity_weights(
            Path(args.similarity_json),
            args.target_name,
            source_names,
            min_weight=args.min_source_weight,
            temperature=args.sampling_temperature,
        )
    return normalize_source_weights(
        raw_weights,
        min_weight=args.min_source_weight,
        temperature=args.sampling_temperature,
    )


def main() -> None:
    """Train TAGE-SSL-v2 and save encoder-compatible weights."""

    parser = argparse.ArgumentParser(description="Pretrain TAGE with time-frequency reconstruction and branch consistency.")
    parser.add_argument("--data-dir", default="BrainLat19_pretrain_19ch_5s")
    parser.add_argument("--data-dirs", nargs="+", default=None, help="Source directories for multi-source pretraining")
    parser.add_argument("--source-names", nargs="+", default=None, help="Names matching --data-dirs and similarity JSON")
    parser.add_argument(
        "--sampling-mode",
        choices=["shuffle", "proportional", "uniform", "similarity", "manual"],
        default="proportional",
        help="Sampling strategy; shuffle visits each concatenated sample once per epoch",
    )
    parser.add_argument("--source-weights", nargs="+", type=float, default=None, help="Manual source weights")
    parser.add_argument("--similarity-json", default=None, help="Output JSON from analyze_source_target_similarity.py")
    parser.add_argument("--target-name", default=None, help="Target row selected from the similarity JSON")
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--min-source-weight", type=float, default=0.0)
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=0,
        help="Number of sampled segments per epoch; 0 uses the total source size",
    )
    parser.add_argument("--output-dir", default="outputs/pretrain_tage_timefreq_consistency")
    parser.add_argument("--checkpoint-out", default="checkpoints/tage_timefreq_consistency.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--mask-mode", choices=["channel", "time", "hybrid"], default="hybrid")
    parser.add_argument("--signal-weight", type=float, default=1.0)
    parser.add_argument("--freq-weight", type=float, default=0.5)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--crop-consistency-weight", type=float, default=0.1)
    parser.add_argument("--crop-min-ratio", type=float, default=0.2)
    parser.add_argument("--crop-max-ratio", type=float, default=0.6)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--spatial-out-dim", type=int, default=256)
    parser.add_argument("--temporal-out-dim", type=int, default=256)
    parser.add_argument("--spatial-hidden-channels", type=int, default=64)
    parser.add_argument("--spatial-temporal-channels", type=int, default=128)
    parser.add_argument("--classifier-dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    for name in ("signal_weight", "freq_weight", "consistency_weight", "crop_consistency_weight"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if not any(
        weight > 0
        for weight in (
            args.signal_weight,
            args.freq_weight,
            args.consistency_weight,
            args.crop_consistency_weight,
        )
    ):
        raise ValueError("At least one pretraining objective weight must be positive.")

    set_seed(args.seed)
    data_dirs, source_names = resolve_sources(args)
    output_dir = Path(args.output_dir)
    checkpoint_out = Path(args.checkpoint_out)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    x_paths = [data_dir / "x_19ch.npy" for data_dir in data_dirs]
    dataset = MultiSourceEEGDataset(x_paths, source_names)
    if dataset.sample_shape is None:
        raise RuntimeError("Could not resolve source EEG shape.")
    num_channels, seq_len = dataset.sample_shape
    if args.sampling_mode == "shuffle":
        if args.samples_per_epoch not in (0, len(dataset)):
            raise ValueError("--sampling-mode shuffle requires --samples-per-epoch 0 or the full dataset size.")
        source_weights = normalize_source_weights(dataset.source_lengths)
        samples_per_epoch = len(dataset)
        sampler = None
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
    else:
        source_weights = resolve_source_weights(args, source_names, dataset.source_lengths)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TAGETimeFreqConsistencyModel(
        num_channels=num_channels,
        seq_len=seq_len,
        spatial_out_dim=args.spatial_out_dim,
        temporal_out_dim=args.temporal_out_dim,
        spatial_hidden_channels=args.spatial_hidden_channels,
        spatial_temporal_channels=args.spatial_temporal_channels,
        classifier_dropout=args.classifier_dropout,
        projection_dim=args.projection_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "time": 0.0, "freq": 0.0, "branch": 0.0, "crop": 0.0}
        total_batches = 0
        for batch in loader:
            x = batch.to(device, non_blocking=True)
            mask = make_mask(x, args.mask_ratio, args.mask_mode)
            x_masked = x.masked_fill(mask, 0.0)

            recon_time, recon_freq, z_spatial, z_temporal, fused_full = model(x_masked)
            zero = recon_time.new_zeros(())
            time_loss = zero
            if args.signal_weight > 0:
                time_loss = F.mse_loss(recon_time[mask], x[mask]) if mask.any() else F.mse_loss(recon_time, x)
            freq_loss = zero
            if args.freq_weight > 0:
                freq_target = spectral_magnitude(x)
                freq_loss = F.mse_loss(recon_freq, freq_target)

            branch_loss = zero
            if args.consistency_weight > 0:
                branch_loss = branch_consistency_loss(z_spatial, z_temporal)

            crop_loss = zero
            if args.crop_consistency_weight > 0:
                x_crop = random_temporal_crop_resize(x, args.crop_min_ratio, args.crop_max_ratio)
                crop_mask = make_mask(x_crop, args.mask_ratio, args.mask_mode)
                _, _, fused_crop = model.encode_parts(x_crop.masked_fill(crop_mask, 0.0))
                crop_loss = branch_consistency_loss(model.project_view(fused_full), model.project_view(fused_crop))
            loss = (
                args.signal_weight * time_loss
                + args.freq_weight * freq_loss
                + args.consistency_weight * branch_loss
                + args.crop_consistency_weight * crop_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            totals["loss"] += float(loss.detach().cpu())
            totals["time"] += float(time_loss.detach().cpu())
            totals["freq"] += float(freq_loss.detach().cpu())
            totals["branch"] += float(branch_loss.detach().cpu())
            totals["crop"] += float(crop_loss.detach().cpu())
            total_batches += 1

        row = {key: value / max(total_batches, 1) for key, value in totals.items()}
        row["epoch"] = float(epoch)
        if sampler is None:
            sampled_counts = dataset.source_lengths
        else:
            sampled_counts = sampler.last_domain_counts
        sampled_total = max(sum(sampled_counts), 1)
        for name, count in zip(source_names, sampled_counts):
            row[f"source_fraction/{name}"] = float(count / sampled_total)
        history.append(row)
        print(
            f"Epoch {epoch:03d} | loss={row['loss']:.6f} "
            f"time={row['time']:.6f} freq={row['freq']:.6f} "
            f"branch={row['branch']:.6f} crop={row['crop']:.6f}"
        )
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_state = {
                key.replace("encoder.", "", 1): value.detach().cpu()
                for key, value in model.state_dict().items()
                if key.startswith("encoder.") and not key.startswith("encoder.fusion_classifier")
            }

    if best_state is None:
        raise RuntimeError("No checkpoint was produced.")
    torch.save(best_state, checkpoint_out)
    summary = {
        "data_dirs": [str(path) for path in data_dirs],
        "source_names": source_names,
        "source_lengths": dataset.source_lengths,
        "sampling_mode": args.sampling_mode,
        "source_weights": {name: weight for name, weight in zip(source_names, source_weights)},
        "similarity_json": args.similarity_json,
        "target_name": args.target_name,
        "sampling_temperature": args.sampling_temperature,
        "min_source_weight": args.min_source_weight,
        "samples_per_epoch": samples_per_epoch,
        "checkpoint_out": str(checkpoint_out),
        "best_loss": best_loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "mask_ratio": args.mask_ratio,
        "mask_mode": args.mask_mode,
        "signal_weight": args.signal_weight,
        "freq_weight": args.freq_weight,
        "consistency_weight": args.consistency_weight,
        "crop_consistency_weight": args.crop_consistency_weight,
        "crop_min_ratio": args.crop_min_ratio,
        "crop_max_ratio": args.crop_max_ratio,
        "seed": args.seed,
        "history": history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
