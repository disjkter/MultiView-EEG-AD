"""Memory-efficient source-aware sampling for multi-source EEG pretraining."""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def normalize_source_weights(
    weights: Sequence[float],
    *,
    min_weight: float = 0.0,
    temperature: float = 1.0,
) -> list[float]:
    """Validate, smooth, and normalize source-level sampling weights."""

    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Source weights must be a non-empty one-dimensional sequence.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Source weights must be finite and non-negative.")
    if min_weight < 0:
        raise ValueError("min_weight must be non-negative.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    values = np.maximum(values, float(min_weight))
    if float(values.sum()) <= 0:
        raise ValueError("At least one source weight must be positive.")
    values = np.power(values, 1.0 / float(temperature))
    values /= values.sum()
    return values.tolist()


def load_similarity_weights(
    json_path: Path,
    target_name: str,
    source_names: Sequence[str],
    *,
    min_weight: float = 0.0,
    temperature: float = 1.0,
) -> list[float]:
    """Load source weights for one target from similarity-analysis JSON output."""

    rows = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of similarity rows in {json_path}.")

    weight_by_source: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict) or str(row.get("target")) != target_name:
            continue
        source = str(row.get("source"))
        if source in weight_by_source:
            raise ValueError(f"Duplicate similarity row for target={target_name}, source={source}.")
        if "sampling_weight" not in row:
            raise ValueError(f"Missing sampling_weight for target={target_name}, source={source}.")
        weight_by_source[source] = float(row["sampling_weight"])

    missing = [name for name in source_names if name not in weight_by_source]
    if missing:
        raise ValueError(
            f"Similarity weights for target={target_name} are missing sources: {', '.join(missing)}"
        )
    return normalize_source_weights(
        [weight_by_source[name] for name in source_names],
        min_weight=min_weight,
        temperature=temperature,
    )


class MultiSourceEEGDataset(Dataset):
    """Read multiple NPY arrays as one memory-mapped normalized EEG dataset."""

    def __init__(self, x_paths: Sequence[Path], source_names: Sequence[str]) -> None:
        if not x_paths:
            raise ValueError("At least one source array is required.")
        if len(x_paths) != len(source_names):
            raise ValueError("x_paths and source_names must have the same length.")

        self.source_names = list(source_names)
        self.arrays: list[np.ndarray] = []
        expected_shape: tuple[int, int] | None = None
        for path in x_paths:
            if not path.exists():
                raise FileNotFoundError(f"Could not find EEG array: {path}")
            array = np.load(path, mmap_mode="r")
            if array.ndim != 3:
                raise ValueError(f"Expected [N, C, T] array at {path}, got {array.shape}.")
            shape = (int(array.shape[1]), int(array.shape[2]))
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError(f"Source shape mismatch at {path}: expected {expected_shape}, got {shape}.")
            self.arrays.append(array)

        self.sample_shape = expected_shape
        self.source_lengths = [int(array.shape[0]) for array in self.arrays]
        if any(length <= 0 for length in self.source_lengths):
            raise ValueError("Every source must contain at least one sample.")
        self.offsets = np.cumsum([0, *self.source_lengths], dtype=np.int64).tolist()

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def close(self) -> None:
        """Close underlying memory maps, mainly for short-lived tests and tools."""

        for array in self.arrays:
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def source_index(self, index: int) -> int:
        """Return the source-domain index for a global sample index."""

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return bisect.bisect_right(self.offsets, index) - 1

    def __getitem__(self, index: int) -> torch.Tensor:
        source_idx = self.source_index(index)
        local_idx = index - self.offsets[source_idx]
        x = np.asarray(self.arrays[source_idx][local_idx], dtype=np.float32)
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-6
        return torch.from_numpy((x - mean) / std)


class SourceWeightedSampler(Sampler[int]):
    """Sample a source first, then sample an EEG segment within that source."""

    def __init__(
        self,
        source_lengths: Sequence[int],
        source_weights: Sequence[float],
        num_samples: int,
        seed: int,
    ) -> None:
        if len(source_lengths) != len(source_weights):
            raise ValueError("source_lengths and source_weights must have the same length.")
        if not source_lengths or any(int(length) <= 0 for length in source_lengths):
            raise ValueError("Source lengths must be positive.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        self.source_lengths = [int(length) for length in source_lengths]
        self.source_weights = normalize_source_weights(source_weights)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.iteration = 0
        self.offsets = torch.tensor(np.cumsum([0, *self.source_lengths[:-1]]), dtype=torch.long)
        self.last_domain_counts = [0 for _ in self.source_lengths]

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.iteration)
        self.iteration += 1

        probabilities = torch.tensor(self.source_weights, dtype=torch.double)
        domain_ids = torch.multinomial(
            probabilities,
            num_samples=self.num_samples,
            replacement=True,
            generator=generator,
        )
        indices = torch.empty(self.num_samples, dtype=torch.long)
        counts = torch.bincount(domain_ids, minlength=len(self.source_lengths))
        self.last_domain_counts = counts.tolist()
        for domain_idx, source_length in enumerate(self.source_lengths):
            mask = domain_ids == domain_idx
            count = int(mask.sum())
            if count:
                local = torch.randint(source_length, (count,), generator=generator)
                indices[mask] = self.offsets[domain_idx] + local
        return iter(indices.tolist())
