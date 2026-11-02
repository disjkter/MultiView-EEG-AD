"""Analyze source-target EEG similarity for pretraining source selection.

The script compares 19-channel 1s EEG datasets using lightweight statistics:

- band-power distribution distance
- channel-correlation matrix distance
- CORAL distance on compact per-segment features

It is intended for paper analysis, not for model training.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SOURCES = [
    "BrainLat19_pretrain_19ch_1s_lead",
    "auditory40hz_processed_19ch_1s_lead",
    "ADFSU_processed_19ch_1s_v2",
    "ADSZ_processed_19ch_1s_v2",
]
DEFAULT_TARGETS = [
    "CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
    "ds004504_processed_19ch_1s_v2",
]
DEFAULT_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass
class DatasetStats:
    """Compact dataset statistics used for source-target comparison."""

    name: str
    path: str
    num_segments_total: int
    num_segments_used: int
    num_subjects: int | None
    num_subjects_available: int | None
    band_power: np.ndarray
    corr_matrix: np.ndarray
    feature_mean: np.ndarray
    feature_cov: np.ndarray
    subset: str
    split_file: str | None


def resolve_domain_files(domain_dir: Path) -> tuple[Path, Path | None, Path | None]:
    """Resolve x/y/subject files from a processed EEG directory."""

    x_path = None
    for name in ("x_19ch.npy", "x.npy"):
        candidate = domain_dir / name
        if candidate.exists():
            x_path = candidate
            break
    if x_path is None:
        raise FileNotFoundError(f"Could not find x_19ch.npy or x.npy under {domain_dir}")

    y_path = None
    for name in ("y_19ch.npy", "y.npy"):
        candidate = domain_dir / name
        if candidate.exists():
            y_path = candidate
            break

    sub_path = None
    for name in ("sub_ids_19ch.npy", "sub_ids.npy"):
        candidate = domain_dir / name
        if candidate.exists():
            sub_path = candidate
            break

    return x_path, y_path, sub_path


def choose_indices(
    n_segments: int,
    sub_ids: np.ndarray | None,
    max_segments: int,
    max_segments_per_subject: int,
    seed: int,
    eligible_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Choose a reproducible subset, with optional per-subject balancing."""

    rng = np.random.default_rng(seed)
    candidates = (
        np.asarray(eligible_indices, dtype=np.int64)
        if eligible_indices is not None
        else np.arange(n_segments, dtype=np.int64)
    )
    if len(candidates) == 0:
        raise ValueError("No eligible EEG segments remain after applying the subject subset.")
    if sub_ids is None or max_segments_per_subject <= 0:
        size = min(max_segments, len(candidates))
        return np.sort(rng.choice(candidates, size=size, replace=False))

    capped_by_subject: list[np.ndarray] = []
    unique_subjects = np.unique(sub_ids[candidates])
    rng.shuffle(unique_subjects)
    for subject in unique_subjects:
        subject_indices = candidates[sub_ids[candidates] == subject]
        if len(subject_indices) > max_segments_per_subject:
            subject_indices = rng.choice(
                subject_indices,
                size=max_segments_per_subject,
                replace=False,
            )
        capped_by_subject.append(np.asarray(subject_indices, dtype=np.int64))

    if not capped_by_subject:
        return np.sort(candidates[:max_segments])
    total_capped = sum(len(group) for group in capped_by_subject)
    if total_capped <= max_segments:
        return np.sort(np.concatenate(capped_by_subject).astype(np.int64))

    # Give every subject an equal base quota before filling the remaining budget.
    base_quota = max(1, max_segments // len(capped_by_subject))
    selected: list[np.ndarray] = []
    leftovers: list[np.ndarray] = []
    for group in capped_by_subject:
        shuffled = rng.permutation(group)
        take = min(len(shuffled), base_quota)
        selected.append(shuffled[:take])
        leftovers.append(shuffled[take:])
    indices = np.concatenate(selected)
    if len(indices) > max_segments:
        indices = rng.choice(indices, size=max_segments, replace=False)
    elif len(indices) < max_segments:
        pool = np.concatenate([group for group in leftovers if len(group)])
        fill = rng.choice(pool, size=max_segments - len(indices), replace=False)
        indices = np.concatenate([indices, fill])
    return np.sort(indices.astype(np.int64))


def normalize_segments(x: np.ndarray) -> np.ndarray:
    """Per-segment, per-channel z-score normalization. Shape: [N, C, T]."""

    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32, copy=False)


def band_power_features(x: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Return mean log band power per channel. Shape: [C, B]."""

    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / sampling_rate)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    powers = []
    for low, high in DEFAULT_BANDS.values():
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            power = np.zeros(x.shape[:2], dtype=np.float32)
        else:
            power = spec[..., mask].mean(axis=-1)
        powers.append(np.log1p(power).astype(np.float32))
    # [N, C, B] -> [C, B]
    return np.stack(powers, axis=-1).mean(axis=0)


def average_channel_correlation(x: np.ndarray) -> np.ndarray:
    """Return average channel-correlation matrix over sampled segments."""

    corr_sum = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    valid = 0
    for sample in x:
        corr = np.corrcoef(sample)
        if np.isfinite(corr).all():
            corr_sum += corr
            valid += 1
    if valid == 0:
        return np.eye(x.shape[1], dtype=np.float32)
    return (corr_sum / valid).astype(np.float32)


def compact_features(x: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Build compact per-segment features from band power and correlations."""

    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / sampling_rate)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    band_feats = []
    for low, high in DEFAULT_BANDS.values():
        mask = (freqs >= low) & (freqs < high)
        power = spec[..., mask].mean(axis=-1) if np.any(mask) else np.zeros(x.shape[:2])
        band_feats.append(np.log1p(power))
    band_vec = np.stack(band_feats, axis=-1).reshape(x.shape[0], -1)

    upper_i, upper_j = np.triu_indices(x.shape[1], k=1)
    corr_vec = np.empty((x.shape[0], len(upper_i)), dtype=np.float32)
    for idx, sample in enumerate(x):
        corr = np.corrcoef(sample)
        corr_vec[idx] = corr[upper_i, upper_j]
    return np.concatenate([band_vec, corr_vec], axis=1).astype(np.float32)


def covariance(features: np.ndarray) -> np.ndarray:
    """Return regularized covariance matrix."""

    centered = features - features.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, len(features) - 1)
    cov += np.eye(cov.shape[0], dtype=np.float32) * 1e-4
    return cov.astype(np.float32)


def compute_stats(
    dataset_name: str,
    max_segments: int,
    max_segments_per_subject: int,
    seed: int,
    sampling_rate: float,
    allowed_subjects: set[str] | None = None,
    split_file: Path | None = None,
) -> DatasetStats:
    """Load a dataset subset and compute compact statistics."""

    domain_dir = PROJECT_ROOT / dataset_name
    x_path, _, sub_path = resolve_domain_files(domain_dir)
    x_all = np.load(x_path, mmap_mode="r")
    sub_ids = np.load(sub_path, allow_pickle=True) if sub_path is not None else None
    eligible_indices = None
    subset = "all"
    if allowed_subjects is not None:
        if sub_ids is None:
            raise ValueError(f"Dataset {dataset_name} has no subject IDs for split filtering.")
        subject_strings = np.asarray([str(subject) for subject in sub_ids], dtype=object)
        eligible_indices = np.flatnonzero(
            np.isin(subject_strings, np.asarray(sorted(allowed_subjects), dtype=object))
        )
        missing = allowed_subjects.difference(set(subject_strings[eligible_indices].tolist()))
        if missing:
            raise ValueError(
                f"Split for {dataset_name} contains unknown training subjects: {sorted(missing)[:5]}"
            )
        subset = "train_subjects"
    indices = choose_indices(
        n_segments=int(x_all.shape[0]),
        sub_ids=sub_ids,
        max_segments=max_segments,
        max_segments_per_subject=max_segments_per_subject,
        seed=seed,
        eligible_indices=eligible_indices,
    )
    x = normalize_segments(np.asarray(x_all[indices], dtype=np.float32))
    feats = compact_features(x, sampling_rate=sampling_rate)
    n_subjects = int(len(np.unique(sub_ids[indices]))) if sub_ids is not None else None
    available_indices = eligible_indices if eligible_indices is not None else np.arange(len(x_all))
    n_subjects_available = (
        int(len(np.unique(sub_ids[available_indices]))) if sub_ids is not None else None
    )
    return DatasetStats(
        name=dataset_name,
        path=str(domain_dir),
        num_segments_total=int(x_all.shape[0]),
        num_segments_used=int(len(indices)),
        num_subjects=n_subjects,
        num_subjects_available=n_subjects_available,
        band_power=band_power_features(x, sampling_rate=sampling_rate),
        corr_matrix=average_channel_correlation(x),
        feature_mean=feats.mean(axis=0),
        feature_cov=covariance(feats),
        subset=subset,
        split_file=str(split_file) if split_file is not None else None,
    )


def load_training_subjects(split_file: Path) -> set[str]:
    """Load the training-subject set from a saved subject split."""

    payload = json.loads(split_file.read_text(encoding="utf-8"))
    train_subjects = payload.get("train_subjects")
    if not isinstance(train_subjects, list) or not train_subjects:
        raise ValueError(f"Split file has no non-empty train_subjects list: {split_file}")
    return {str(subject) for subject in train_subjects}


def normalized_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Scale-aware L2 distance."""

    denom = float(np.linalg.norm(a) + np.linalg.norm(b) + 1e-8)
    return float(np.linalg.norm(a - b) / denom)


def coral_distance(source: DatasetStats, target: DatasetStats) -> float:
    """CORAL distance between compact source and target feature distributions."""

    mean_dist = np.linalg.norm(source.feature_mean - target.feature_mean) ** 2
    cov_dist = np.linalg.norm(source.feature_cov - target.feature_cov, ord="fro") ** 2
    dim = source.feature_mean.shape[0]
    return float((mean_dist + cov_dist) / (4.0 * dim * dim))


def exponential_similarity(values: list[float], temperature: float = 1.0) -> list[float]:
    """Convert distances to positive affinities using a scale-normalized exponential kernel."""

    if temperature <= 0.0:
        raise ValueError("similarity temperature must be positive")
    distances = np.asarray(values, dtype=np.float64)
    scale = float(distances.std())
    if scale < 1e-12:
        return [1.0 for _ in values]
    shifted = (distances - distances.min()) / (scale * temperature)
    return np.exp(-shifted).astype(float).tolist()


def write_outputs(rows: list[dict[str, object]], output_prefix: Path) -> None:
    """Write CSV, JSON, and Markdown outputs."""

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")

    fieldnames = [
        "target",
        "source",
        "source_segments",
        "source_subjects",
        "source_subjects_available",
        "target_segments",
        "target_subjects",
        "target_subjects_available",
        "target_subset",
        "target_split_file",
        "band_power_distance",
        "connectivity_distance",
        "coral_distance",
        "composite_distance",
        "similarity_score",
        "sampling_weight",
        "weighting_method",
        "similarity_temperature",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    lines = [
        "# Source-Target Similarity Analysis",
        "",
        "| Target | Source | Band Dist | Conn Dist | CORAL | Composite Dist | Similarity | Weight |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {target} | {source} | {band_power_distance:.4f} | "
            "{connectivity_distance:.4f} | {coral_distance:.4f} | "
            "{composite_distance:.4f} | {similarity_score:.4f} | "
            "{sampling_weight:.4f} |".format(**row)
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument(
        "--target-split-files",
        nargs="+",
        default=None,
        help="Split JSON files matching --targets; only train_subjects are used for target statistics",
    )
    parser.add_argument("--max-segments", type=int, default=5000)
    parser.add_argument("--max-segments-per-subject", type=int, default=50)
    parser.add_argument("--sampling-rate", type=float, default=128.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--similarity-temperature",
        type=float,
        default=1.0,
        help="Temperature for exponential distance-to-weight conversion; larger values give flatter weights",
    )
    parser.add_argument(
        "--output-prefix",
        default="outputs/source_target_similarity/source_target_similarity",
    )
    return parser.parse_args()


def main() -> None:
    """Run source-target similarity analysis."""

    args = parse_args()
    if args.target_split_files is not None and len(args.target_split_files) != len(args.targets):
        raise ValueError("--target-split-files must have the same length as --targets.")
    if args.target_split_files is not None:
        missing_split_files = [
            str(path)
            for value in args.target_split_files
            for path in [Path(value)]
            if not path.is_file()
        ]
        if missing_split_files:
            raise FileNotFoundError(
                "Missing target split files; generate or correct them before computing source stats: "
                + ", ".join(missing_split_files)
            )

    source_stats: dict[str, DatasetStats] = {}
    for name in dict.fromkeys(args.sources):
        print(f"Computing source stats: {name}", flush=True)
        source_stats[name] = compute_stats(
            dataset_name=name,
            max_segments=args.max_segments,
            max_segments_per_subject=args.max_segments_per_subject,
            seed=args.seed,
            sampling_rate=args.sampling_rate,
        )

    target_stats: dict[str, DatasetStats] = {}
    target_split_paths: dict[str, Path | None] = {}
    for index, name in enumerate(args.targets):
        split_file = Path(args.target_split_files[index]) if args.target_split_files else None
        train_subjects = load_training_subjects(split_file) if split_file is not None else None
        subset_label = "training subjects" if train_subjects is not None else "all subjects"
        print(f"Computing target stats: {name} ({subset_label})", flush=True)
        target_stats[name] = compute_stats(
            dataset_name=name,
            max_segments=args.max_segments,
            max_segments_per_subject=args.max_segments_per_subject,
            seed=args.seed,
            sampling_rate=args.sampling_rate,
            allowed_subjects=train_subjects,
            split_file=split_file,
        )
        target_split_paths[name] = split_file

    rows: list[dict[str, object]] = []
    for target_name in args.targets:
        target = target_stats[target_name]
        target_rows = []
        for source_name in args.sources:
            if source_name == target_name:
                continue
            source = source_stats[source_name]
            band_dist = normalized_l2(source.band_power, target.band_power)
            conn_dist = normalized_l2(source.corr_matrix, target.corr_matrix)
            coral_dist = coral_distance(source, target)
            composite = float(np.mean([band_dist, conn_dist, coral_dist]))
            target_rows.append(
                {
                    "target": target_name,
                    "source": source_name,
                    "source_segments": source.num_segments_used,
                    "source_subjects": source.num_subjects,
                    "source_subjects_available": source.num_subjects_available,
                    "target_segments": target.num_segments_used,
                    "target_subjects": target.num_subjects,
                    "target_subjects_available": target.num_subjects_available,
                    "target_subset": target.subset,
                    "target_split_file": target.split_file,
                    "band_power_distance": band_dist,
                    "connectivity_distance": conn_dist,
                    "coral_distance": coral_dist,
                    "composite_distance": composite,
                }
            )
        if not target_rows:
            raise ValueError(f"Target {target_name} has no candidate sources after self-exclusion.")
        sims = exponential_similarity(
            [float(row["composite_distance"]) for row in target_rows],
            temperature=args.similarity_temperature,
        )
        sim_sum = sum(sims) + 1e-8
        for row, sim in zip(target_rows, sims):
            row["similarity_score"] = sim
            row["sampling_weight"] = float(sim / sim_sum)
            row["weighting_method"] = "scale_normalized_exponential"
            row["similarity_temperature"] = float(args.similarity_temperature)
        rows.extend(sorted(target_rows, key=lambda item: float(item["composite_distance"])))

    write_outputs(rows, Path(args.output_prefix))


if __name__ == "__main__":
    main()
