"""Audit ADFSU for duplicates, signal artifacts, and simple-feature separability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_dual_stream_loso_subject import resolve_domain_files  # noqa: E402

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows as UTF-8 CSV."""

    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_domain(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Memory-map one processed EEG domain."""

    x_path, y_path, id_path = resolve_domain_files(path)
    x = np.load(x_path, mmap_mode="r")
    y = np.asarray(np.load(y_path, mmap_mode="r"), dtype=np.int64)
    subject_ids = np.asarray(np.load(id_path, allow_pickle=True), dtype=object)
    if len(x) != len(y) or len(y) != len(subject_ids):
        raise ValueError(f"Array length mismatch under {path}")
    return x, y, subject_ids


def subject_index(subject_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Group segment indices by participant ID."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, subject in enumerate(subject_ids):
        grouped[str(subject)].append(index)
    return {key: np.asarray(value, dtype=np.int64) for key, value in grouped.items()}


def stable_indices(indices: np.ndarray, limit: int) -> np.ndarray:
    """Select deterministic approximately uniform indices."""

    if limit <= 0 or len(indices) <= limit:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=limit, dtype=np.int64)
    return indices[positions]


def participant_features(
    data: np.ndarray,
    sampling_rate: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Compute QA, spectral, and channel summaries for [N,C,T] windows."""

    values = np.asarray(data, dtype=np.float64)
    flattened = values.reshape(-1)
    channel_axes = (0, 2)
    qa = {
        "amplitude_mean": float(flattened.mean()),
        "amplitude_std": float(flattened.std()),
        "rms": float(np.sqrt(np.mean(flattened**2))),
        "peak_to_peak": float(np.ptp(flattened)),
        "zero_rate": float(np.mean(flattened == 0.0)),
        "flat_window_rate": float(np.mean(np.std(values, axis=2) < 1e-8)),
        "clipping_rate": float(
            np.mean(np.abs(flattened) >= np.quantile(np.abs(flattened), 0.9999))
        ),
    }
    channel = {}
    for index, (mean, std) in enumerate(
        zip(values.mean(axis=channel_axes), values.std(axis=channel_axes))
    ):
        channel[f"ch{index + 1:02d}_mean"] = float(mean)
        channel[f"ch{index + 1:02d}_std"] = float(std)

    demeaned = values - values.mean(axis=2, keepdims=True)
    spectrum = np.abs(np.fft.rfft(demeaned, axis=2)) ** 2
    frequencies = np.fft.rfftfreq(values.shape[2], d=1.0 / sampling_rate)
    spectral = {}
    total_mask = (frequencies >= 0.5) & (frequencies <= min(45.0, sampling_rate / 2))
    total_power = np.maximum(spectrum[:, :, total_mask].sum(axis=2), 1e-12)
    for name, (low, high) in BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        absolute = spectrum[:, :, mask].sum(axis=2)
        spectral[f"{name}_power"] = float(absolute.mean())
        spectral[f"{name}_relative_power"] = float((absolute / total_power).mean())
    return qa, spectral, channel


def collect_qa(
    name: str,
    directory: Path,
    sampling_rate: float,
    max_windows_per_subject: int,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, list[str], list[str]]:
    """Collect participant-level features and feature column names."""

    x, y, ids = load_domain(directory)
    rows = []
    grouped = subject_index(ids)
    for subject in sorted(grouped):
        indices = grouped[subject]
        labels = np.unique(y[indices])
        if len(labels) != 1:
            raise ValueError(f"Mixed labels for {name}/{subject}: {labels.tolist()}")
        selected = stable_indices(indices, max_windows_per_subject)
        qa, spectral, channel = participant_features(x[selected], sampling_rate)
        rows.append(
            {
                "cohort": name,
                "participant_id": subject,
                "label": int(labels[0]),
                "available_windows": len(indices),
                "analyzed_windows": len(selected),
                "duration_seconds_analyzed": float(len(selected) * x.shape[2] / sampling_rate),
                **qa,
                **spectral,
                **channel,
            }
        )
    qa_columns = list(qa)
    spectral_columns = list(spectral)
    feature_columns = qa_columns + spectral_columns
    matrix = np.asarray([[float(row[key]) for key in feature_columns] for row in rows])
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    return rows, matrix, labels, qa_columns, spectral_columns


def evaluate_simple_features(
    matrix: np.ndarray,
    labels: np.ndarray,
    columns: list[str],
    groups: dict[str, list[str]],
    seeds: list[int],
    permutation_repeats: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate participant-level logistic-regression sanity baselines."""

    results = []
    permutations = []
    for group_name, selected_columns in groups.items():
        selected = np.asarray([columns.index(column) for column in selected_columns])
        features = matrix[:, selected]
        observed_seed_aucs = []
        observed_seed_bas = []
        fold_cache = []
        for seed in seeds:
            splitter = RepeatedStratifiedKFold(n_splits=4, n_repeats=1, random_state=seed)
            probabilities = np.full(len(labels), np.nan, dtype=np.float64)
            for train, test in splitter.split(features, labels):
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        class_weight="balanced", max_iter=2000, random_state=seed
                    ),
                )
                model.fit(features[train], labels[train])
                probabilities[test] = model.predict_proba(features[test])[:, 1]
            if np.isnan(probabilities).any():
                raise RuntimeError("Cross-validation did not predict every participant")
            auc = float(roc_auc_score(labels, probabilities))
            ba = float(balanced_accuracy_score(labels, probabilities >= 0.5))
            observed_seed_aucs.append(auc)
            observed_seed_bas.append(ba)
            fold_cache.append((seed, probabilities))
            results.append(
                {
                    "feature_group": group_name,
                    "seed": seed,
                    "num_participants": len(labels),
                    "num_features": features.shape[1],
                    "auc": auc,
                    "balanced_accuracy": ba,
                }
            )

        rng = np.random.default_rng(4187 + len(selected_columns))
        null_auc = []
        for repeat in range(permutation_repeats):
            permuted = rng.permutation(labels)
            seed = seeds[repeat % len(seeds)]
            splitter = RepeatedStratifiedKFold(n_splits=4, n_repeats=1, random_state=seed)
            probabilities = np.full(len(labels), np.nan, dtype=np.float64)
            for train, test in splitter.split(features, permuted):
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        class_weight="balanced", max_iter=2000, random_state=seed
                    ),
                )
                model.fit(features[train], permuted[train])
                probabilities[test] = model.predict_proba(features[test])[:, 1]
            null_auc.append(float(roc_auc_score(permuted, probabilities)))
        observed = float(np.mean(observed_seed_aucs))
        p_value = (1.0 + sum(value >= observed for value in null_auc)) / (
            permutation_repeats + 1.0
        )
        permutations.append(
            {
                "feature_group": group_name,
                "observed_auc_mean": observed,
                "observed_ba_mean": float(np.mean(observed_seed_bas)),
                "permutation_repeats": permutation_repeats,
                "null_auc_mean": float(np.mean(null_auc)),
                "null_auc_95pct": float(np.percentile(null_auc, 95)),
                "permutation_p_value": float(p_value),
            }
        )
    return results, permutations


def digest_segments(data: np.ndarray, indices: Iterable[int]) -> Counter[str]:
    """Hash selected EEG windows in a stable dtype-aware format."""

    hashes: Counter[str] = Counter()
    for index in indices:
        array = np.ascontiguousarray(data[int(index)])
        digest = hashlib.sha256(array.view(np.uint8)).hexdigest()
        hashes[digest] += 1
    return hashes


def duplicate_detail_rows(
    reference_dir: Path,
    comparison_domains: list[tuple[str, Path]],
) -> list[dict[str, object]]:
    """List exact matching processed windows with participant provenance."""

    reference_x, reference_y, reference_ids = load_domain(reference_dir)
    reference: dict[str, list[int]] = defaultdict(list)
    for index in range(len(reference_x)):
        array = np.ascontiguousarray(reference_x[index])
        reference[hashlib.sha256(array.view(np.uint8)).hexdigest()].append(index)

    details = []
    domains = [("ADFSU", reference_dir), *comparison_domains]
    for name, directory in domains:
        if not directory.exists():
            continue
        x, y, ids = load_domain(directory)
        for index in range(len(x)):
            array = np.ascontiguousarray(x[index])
            digest = hashlib.sha256(array.view(np.uint8)).hexdigest()
            matches = reference.get(digest, [])
            for reference_index in matches:
                if name == "ADFSU" and index <= reference_index:
                    continue
                details.append(
                    {
                        "comparison": f"ADFSU_vs_{name}",
                        "sha256": digest,
                        "adfsu_index": reference_index,
                        "adfsu_participant_id": str(reference_ids[reference_index]),
                        "adfsu_label": int(reference_y[reference_index]),
                        "comparison_index": index,
                        "comparison_participant_id": str(ids[index]),
                        "comparison_label": int(y[index]),
                        "labels_match": bool(reference_y[reference_index] == y[index]),
                    }
                )
    return details


def integrity_rows(
    adfsu_dir: Path,
    comparison_domains: list[tuple[str, Path]],
) -> list[dict[str, object]]:
    """Check exact processed-window duplicates within ADFSU and local cohorts."""

    adfsu_x, _, _ = load_domain(adfsu_dir)
    reference = digest_segments(adfsu_x, range(len(adfsu_x)))
    rows = [
        {
            "comparison": "ADFSU_within",
            "reference_windows": len(adfsu_x),
            "comparison_windows": len(adfsu_x),
            "matching_unique_hashes": sum(count > 1 for count in reference.values()),
            "matching_window_instances": sum(count for count in reference.values() if count > 1),
            "status": "PASS" if all(count == 1 for count in reference.values()) else "REVIEW",
        }
    ]
    reference_set = set(reference)
    for name, directory in comparison_domains:
        if not directory.exists():
            rows.append(
                {
                    "comparison": f"ADFSU_vs_{name}",
                    "reference_windows": len(adfsu_x),
                    "comparison_windows": 0,
                    "matching_unique_hashes": 0,
                    "matching_window_instances": 0,
                    "status": "NOT_AVAILABLE",
                }
            )
            continue
        x, _, _ = load_domain(directory)
        matches = 0
        matching_hashes = set()
        for index in range(len(x)):
            array = np.ascontiguousarray(x[index])
            digest = hashlib.sha256(array.view(np.uint8)).hexdigest()
            if digest in reference_set:
                matches += 1
                matching_hashes.add(digest)
        rows.append(
            {
                "comparison": f"ADFSU_vs_{name}",
                "reference_windows": len(adfsu_x),
                "comparison_windows": len(x),
                "matching_unique_hashes": len(matching_hashes),
                "matching_window_instances": matches,
                "status": "PASS" if matches == 0 else "FAIL",
            }
        )
    return rows


def reconstruct_score_rows(
    results_dir: Path,
    adfsu_dir: Path,
) -> list[dict[str, object]]:
    """Export saved ADFSU participant scores with strict ID/label verification."""

    _, y, ids = load_domain(adfsu_dir)
    grouped = subject_index(ids)
    ordered = sorted(grouped)
    expected_labels = np.asarray([int(np.unique(y[grouped[key]])[0]) for key in ordered])
    rows = []
    for path in sorted(results_dir.glob("seed*/**/lodo_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        variant = str(payload.get("variant", path.parent.name))
        seed = int(payload["seed"])
        for fold in payload["folds"]:
            if str(fold["target"]) != "ADFSU":
                continue
            metrics = fold["metrics"]
            labels = np.asarray(metrics["y_true"], dtype=np.int64)
            probabilities = np.asarray(metrics["y_prob"], dtype=np.float64)
            saved_ids = metrics.get("subject_ids")
            participant_ids = [str(value) for value in saved_ids] if saved_ids else ordered
            if participant_ids != ordered:
                raise ValueError(f"Participant order mismatch in {path}")
            if not np.array_equal(labels, expected_labels):
                raise ValueError(f"Reconstructed labels do not match saved labels in {path}")
            for participant, label, probability in zip(participant_ids, labels, probabilities):
                clipped = float(np.clip(probability, 1e-7, 1 - 1e-7))
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "target": "ADFSU",
                        "participant_id": participant,
                        "label": int(label),
                        "probability": float(probability),
                        "logit": float(np.log(clipped) - np.log1p(-clipped)),
                        "id_source": "saved" if saved_ids else "verified_reconstruction",
                    }
                )
    if not rows:
        raise FileNotFoundError(f"No saved ADFSU participant predictions under {results_dir}")
    return rows


def main() -> None:
    """Run the complete local ADFSU audit."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adfsu-dir", default="ADFSU_processed_19ch_1s_v2")
    parser.add_argument("--comparison", action="append", default=[])
    parser.add_argument(
        "--results-dir", default="outputs/domain_generalization_adfsu_confirmation"
    )
    parser.add_argument("--output-dir", default="results/adfsu_audit")
    parser.add_argument("--sampling-rate", type=float, default=128.0)
    parser.add_argument("--max-windows-per-subject", type=int, default=20)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument(
        "--skip-scores",
        action="store_true",
        help="Skip saved-prediction export when auditing a newly rebuilt target dataset.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for value in args.comparison:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH for --comparison, got {value}")
        name, path = value.split("=", 1)
        comparisons.append((name, Path(path)))

    integrity = integrity_rows(Path(args.adfsu_dir), comparisons)
    write_csv(output_dir / "file_integrity.csv", integrity)
    duplicate_details = duplicate_detail_rows(Path(args.adfsu_dir), comparisons)
    if duplicate_details:
        write_csv(output_dir / "duplicate_window_details.csv", duplicate_details)

    all_qa = []
    adfsu_matrix = adfsu_labels = None
    qa_columns: list[str] = []
    spectral_columns: list[str] = []
    for name, directory in [("ADFSU", Path(args.adfsu_dir)), *comparisons]:
        if not directory.exists():
            continue
        rows, matrix, labels, qa, spectral = collect_qa(
            name,
            directory,
            args.sampling_rate,
            args.max_windows_per_subject,
        )
        all_qa.extend(rows)
        if name == "ADFSU":
            adfsu_matrix, adfsu_labels = matrix, labels
            qa_columns, spectral_columns = qa, spectral
    write_csv(output_dir / "signal_qa_participant.csv", all_qa)

    if adfsu_matrix is None or adfsu_labels is None:
        raise RuntimeError("ADFSU participant features were not generated")
    columns = qa_columns + spectral_columns
    baseline_rows, permutation_rows = evaluate_simple_features(
        adfsu_matrix,
        adfsu_labels,
        columns,
        {
            "qa_only": qa_columns,
            "spectral_only": spectral_columns,
            "qa_plus_spectral": columns,
        },
        args.seeds,
        args.permutations,
    )
    write_csv(output_dir / "simple_baselines.csv", baseline_rows)
    write_csv(output_dir / "label_permutation.csv", permutation_rows)

    scores = []
    if not args.skip_scores:
        scores = reconstruct_score_rows(Path(args.results_dir), Path(args.adfsu_dir))
        write_csv(output_dir / "participant_scores.csv", scores)

    summary = {
        "metadata": {
            "adfsu_dir": args.adfsu_dir,
            "comparison_domains": {name: str(path) for name, path in comparisons},
            "participant_level": True,
            "max_windows_per_subject": args.max_windows_per_subject,
            "line_noise_proxy": "unavailable after 0.5-45 Hz preprocessing at 128 Hz",
            "demographic_site_device_metadata": "unavailable in processed arrays",
        },
        "integrity": integrity,
        "duplicate_detail_rows": len(duplicate_details),
        "simple_baseline": permutation_rows,
        "score_export_skipped": bool(args.skip_scores),
        "score_rows": len(scores),
        "score_variants": sorted({str(row["variant"]) for row in scores}),
        "score_seeds": sorted({int(row["seed"]) for row in scores}),
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved ADFSU audit artifacts to {output_dir}")


if __name__ == "__main__":
    main()
