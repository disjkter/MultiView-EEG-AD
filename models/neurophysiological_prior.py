"""Neurophysiological prior features for interpretable EEG modeling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


DEFAULT_19_CHANNELS = (
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
)

DEFAULT_REGION_MAP = {
    "frontal": ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"),
    "temporal": ("T3", "T4", "T5", "T6"),
    "central": ("C3", "Cz", "C4"),
    "parietal": ("P3", "Pz", "P4"),
    "occipital": ("O1", "O2"),
}

DEFAULT_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass(frozen=True)
class NeuroPriorOutput:
    """Container for interpretable EEG prior features."""

    region_power: torch.Tensor
    band_power: torch.Tensor
    connectivity: torch.Tensor
    region_names: tuple[str, ...]
    band_names: tuple[str, ...]


class NeurophysiologicalPriorLayer(nn.Module):
    """Extract region, frequency-band, and connectivity priors from EEG.

    Args:
        sampling_rate: Sampling rate in Hz.
        channel_names: Ordered EEG channel names matching input channel order.
        region_map: Mapping from brain-region names to channel names.

    Input:
        x: EEG tensor with shape [B, C, T].

    Output:
        NeuroPriorOutput containing:
        - region_power: [B, R], region-level signal energy.
        - band_power: [B, C, F], channel-wise relative band power.
        - connectivity: [B, C, C], Pearson functional connectivity.
    """

    def __init__(
        self,
        sampling_rate: float,
        channel_names: tuple[str, ...] = DEFAULT_19_CHANNELS,
        region_map: dict[str, tuple[str, ...]] | None = None,
        bands: dict[str, tuple[float, float]] | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.sampling_rate = float(sampling_rate)
        self.channel_names = tuple(channel_names)
        self.region_map = region_map or DEFAULT_REGION_MAP
        self.bands = bands or DEFAULT_BANDS
        self.eps = float(eps)

        name_to_idx = {name: idx for idx, name in enumerate(self.channel_names)}
        region_indices: list[list[int]] = []
        region_names: list[str] = []
        for region_name, channels in self.region_map.items():
            indices = [name_to_idx[ch] for ch in channels if ch in name_to_idx]
            if indices:
                region_names.append(region_name)
                region_indices.append(indices)

        if not region_indices:
            raise ValueError("region_map must contain at least one known channel.")

        self.region_names = tuple(region_names)
        self.band_names = tuple(self.bands.keys())
        self._region_indices = region_indices

    def forward(self, x: torch.Tensor) -> NeuroPriorOutput:
        """Compute prior features from EEG tensor with shape [B, C, T]."""

        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, C, T], got {tuple(x.shape)}")
        if x.shape[1] != len(self.channel_names):
            raise ValueError(
                f"Expected {len(self.channel_names)} channels, got {x.shape[1]}"
            )

        region_power = self._compute_region_power(x)
        band_power = self._compute_band_power(x)
        connectivity = self._compute_pearson_connectivity(x)
        return NeuroPriorOutput(
            region_power=region_power,
            band_power=band_power,
            connectivity=connectivity,
            region_names=self.region_names,
            band_names=self.band_names,
        )

    def _compute_region_power(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mean signal energy per neuroanatomical region."""

        powers = []
        channel_power = x.pow(2).mean(dim=-1)  # [B, C]
        for indices in self._region_indices:
            powers.append(channel_power[:, indices].mean(dim=1))
        return torch.stack(powers, dim=1)

    def _compute_band_power(self, x: torch.Tensor) -> torch.Tensor:
        """Compute channel-wise relative power in canonical EEG bands."""

        spectrum = torch.fft.rfft(x, dim=-1)
        power = spectrum.abs().pow(2)  # [B, C, Freq]
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / self.sampling_rate).to(x.device)
        total_power = power.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        band_values = []
        for low, high in self.bands.values():
            mask = (freqs >= low) & (freqs < high)
            if mask.any():
                band_values.append(power[..., mask].sum(dim=-1) / total_power.squeeze(-1))
            else:
                band_values.append(torch.zeros_like(power[..., 0]))
        return torch.stack(band_values, dim=-1)

    def _compute_pearson_connectivity(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation matrix across EEG channels."""

        centered = x - x.mean(dim=-1, keepdim=True)
        centered = centered / centered.std(dim=-1, keepdim=True).clamp_min(self.eps)
        denom = max(x.shape[-1] - 1, 1)
        corr = torch.matmul(centered, centered.transpose(1, 2)) / denom
        eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device).unsqueeze(0)
        corr = corr * (1.0 - eye) + eye
        return corr.clamp(-1.0, 1.0)
