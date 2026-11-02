"""Audit GraphTemporalNet source checkpoints against labeled target support sets."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.GraphTemporalNet import GraphTemporalNet  # noqa: E402
from scripts.run_dual_stream_subject_split import disambiguate_mixed_label_subject_ids  # noqa: E402
from scripts.run_dual_stream_loso_subject import resolve_domain_files, set_seed  # noqa: E402
from utils.transferability import (  # noqa: E402
    conditional_prototype_distance,
    leep_score,
    logme_score,
    rank_aggregate,
    transrate_score,
)


@dataclass(frozen=True)
class DomainData:
    """Memory-mapped EEG arrays for one labeled domain."""

    name: str
    x: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray


class AuditDataset(Dataset):
    """Normalized EEG subset used for feature and gradient extraction."""

    def __init__(self, domain: DomainData, indices: np.ndarray) -> None:
        self.domain = domain
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        idx = int(self.indices[item])
        sample = np.asarray(self.domain.x[idx], dtype=np.float32).copy()
        sample = (sample - sample.mean(axis=1, keepdims=True)) / (
            sample.std(axis=1, keepdims=True) + 1e-6
        )
        return (
            torch.from_numpy(sample.astype(np.float32, copy=False)),
            torch.tensor(int(self.domain.y[idx]), dtype=torch.long),
            str(self.domain.subject_ids[idx]),
        )


def load_domain(name: str, directory: Path) -> DomainData:
    """Load one processed EEG domain."""

    x_path, y_path, id_path = resolve_domain_files(directory)
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path).astype(np.int64, copy=False)
    subject_ids = np.load(id_path, allow_pickle=True)
    subject_ids = disambiguate_mixed_label_subject_ids(y, subject_ids)
    return DomainData(name=name, x=x, y=y, subject_ids=subject_ids)


def load_split_subjects(split_file: Path, subset: str) -> list[str]:
    """Read a named subject subset from a saved split JSON."""

    payload = json.loads(split_file.read_text(encoding="utf-8"))
    key = f"{subset}_subjects"
    subjects = payload.get(key)
    if not isinstance(subjects, list) or not subjects:
        raise ValueError(f"{split_file} has no non-empty {key}")
    return [str(subject) for subject in subjects]


def select_support_subjects(
    domain: DomainData,
    candidate_subjects: list[str],
    subjects_per_class: int,
    seed: int,
) -> list[str]:
    """Select a deterministic, class-balanced target support set."""

    subject_labels: dict[str, int] = {}
    for subject, label in zip(domain.subject_ids, domain.y):
        subject_labels[str(subject)] = int(label)
    rng = random.Random(seed)
    selected: list[str] = []
    for class_id in sorted(set(subject_labels.values())):
        candidates = [
            subject
            for subject in candidate_subjects
            if subject_labels.get(subject) == class_id
        ]
        rng.shuffle(candidates)
        if len(candidates) < subjects_per_class:
            raise ValueError(
                f"{domain.name} has {len(candidates)} candidate subjects for class {class_id}; "
                f"requested {subjects_per_class}"
            )
        selected.extend(candidates[:subjects_per_class])
    return selected


def sample_subject_indices(
    domain: DomainData,
    subjects: list[str],
    max_segments_per_subject: int,
    seed: int,
) -> np.ndarray:
    """Sample an equal maximum number of segments from each selected subject."""

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    subject_strings = np.asarray([str(subject) for subject in domain.subject_ids], dtype=object)
    for subject in subjects:
        indices = np.flatnonzero(subject_strings == subject)
        if len(indices) == 0:
            raise ValueError(f"Unknown subject {subject!r} in {domain.name}")
        if max_segments_per_subject > 0 and len(indices) > max_segments_per_subject:
            indices = rng.choice(indices, size=max_segments_per_subject, replace=False)
        selected.append(np.asarray(indices, dtype=np.int64))
    return np.sort(np.concatenate(selected))


def load_model(checkpoint: Path, config: dict[str, object], device: torch.device) -> GraphTemporalNet:
    """Construct GraphTemporalNet and load a source-trained checkpoint."""

    model = GraphTemporalNet(
        num_classes=2,
        num_channels=19,
        graph_out_dim=int(config.get("graph_out_dim", 256)),
        temporal_out_dim=int(config.get("temporal_out_dim", 256)),
        classifier_dropout=float(config.get("classifier_dropout", 0.6)),
        num_neighbors=int(config.get("graph_num_neighbors", 4)),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if isinstance(payload, dict) and "encoder_state_dict" in payload:
        payload = payload["encoder_state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload: {checkpoint}")
    state = model.state_dict()
    compatible = {
        key.removeprefix("module."): value
        for key, value in payload.items()
        if key.removeprefix("module.") in state
        and state[key.removeprefix("module.")].shape == value.shape
    }
    missing_classifier = not any(key.startswith("fusion_classifier.") for key in compatible)
    if not compatible or missing_classifier:
        raise ValueError(f"Checkpoint is not a source-trained GraphTemporalNet: {checkpoint}")
    model.load_state_dict(compatible, strict=False)
    model.eval()
    return model


def extract_representations(
    model: GraphTemporalNet,
    dataset: AuditDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Extract segment-level and subject-aggregated model representations."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    segment_graph: list[np.ndarray] = []
    segment_temporal: list[np.ndarray] = []
    segment_fused: list[np.ndarray] = []
    segment_probs: list[np.ndarray] = []
    segment_labels: list[np.ndarray] = []
    segment_subjects: list[str] = []
    with torch.no_grad():
        for batch_x, batch_y, batch_subjects in loader:
            graph, temporal, fused = model.encode(batch_x.to(device))
            logits = model.fusion_classifier(fused)
            segment_graph.append(graph.cpu().numpy())
            segment_temporal.append(temporal.cpu().numpy())
            segment_fused.append(fused.cpu().numpy())
            segment_probs.append(F.softmax(logits, dim=1).cpu().numpy())
            segment_labels.append(batch_y.numpy())
            segment_subjects.extend([str(subject) for subject in batch_subjects])

    segment_arrays = {
        "segment_graph": np.concatenate(segment_graph),
        "segment_temporal": np.concatenate(segment_temporal),
        "segment_fused": np.concatenate(segment_fused),
        "segment_probabilities": np.concatenate(segment_probs),
        "segment_labels": np.concatenate(segment_labels),
        "segment_subjects": np.asarray(segment_subjects, dtype=object),
    }
    unique_subjects = np.unique(segment_arrays["segment_subjects"])
    output: dict[str, list[np.ndarray] | np.ndarray] = {
        "subject_ids": unique_subjects,
        "labels": np.asarray(
            [
                segment_arrays["segment_labels"][
                    segment_arrays["segment_subjects"] == subject
                ][0]
                for subject in unique_subjects
            ],
            dtype=np.int64,
        ),
    }
    for key in ("graph", "temporal", "fused", "probabilities"):
        segment_key = f"segment_{key}"
        output[key] = np.stack(
            [
                segment_arrays[segment_key][
                    segment_arrays["segment_subjects"] == subject
                ].mean(axis=0)
                for subject in unique_subjects
            ]
        )
    output.update(segment_arrays)
    return {key: np.asarray(value) for key, value in output.items()}


def mean_classifier_gradient(
    model: GraphTemporalNet,
    dataset: AuditDataset,
    batch_size: int,
    max_batches: int,
    device: torch.device,
) -> np.ndarray:
    """Return the mean normalized classifier gradient over deterministic batches."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    gradients: list[np.ndarray] = []
    parameters = list(model.fusion_classifier.parameters())
    model.train(False)
    for batch_index, (batch_x, batch_y, _) in enumerate(loader):
        if batch_index >= max_batches:
            break
        model.zero_grad(set_to_none=True)
        logits, _ = model(batch_x.to(device))
        loss = F.cross_entropy(logits, batch_y.to(device))
        values = torch.autograd.grad(loss, parameters)
        vector = torch.cat([value.detach().flatten() for value in values])
        vector /= vector.norm().clamp_min(1e-12)
        gradients.append(vector.cpu().numpy())
    if not gradients:
        raise ValueError("no batches available for gradient extraction")
    mean = np.mean(gradients, axis=0)
    return mean / max(np.linalg.norm(mean), 1e-12)


def write_outputs(rows: list[dict[str, object]], metadata: dict[str, object], prefix: Path) -> None:
    """Write JSON, CSV, and Markdown audit outputs."""

    prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "rows": rows}
    prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fields = list(rows[0].keys())
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# GraphTemporal Transferability Audit",
        "",
        "| Target | Source | LogME | TransRate | LEEP | Grad Cos | Graph Dist | "
        "Temporal Dist | Fused Dist | Rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['target']} | {row['source']} | {row['logme']:.4f} | "
            f"{row['transrate']:.4f} | {row['leep']:.4f} | "
            f"{row['gradient_cosine']:.4f} | {row['graph_prototype_distance']:.4f} | "
            f"{row['temporal_prototype_distance']:.4f} | "
            f"{row['fused_prototype_distance']:.4f} | {row['rank_score']:.4f} |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run transferability audit from a YAML experiment manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transferability_audit.yaml")
    parser.add_argument("--target", default=None, help="Optional target name filter")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    seed = int(args.seed if args.seed is not None else config.get("seed", 2026))
    set_seed(seed)
    requested_device = str(config.get("device", "auto"))
    device = torch.device(
        "cuda" if requested_device == "auto" and torch.cuda.is_available() else
        "cpu" if requested_device == "auto" else requested_device
    )
    batch_size = int(config.get("batch_size", 64))
    max_segments = int(config.get("max_segments_per_subject", 20))
    gradient_batches = int(config.get("gradient_batches", 4))
    support_per_class = int(config.get("support_subjects_per_class", 4))

    sources = config.get("sources")
    targets = config.get("targets")
    if not isinstance(sources, list) or not sources:
        raise ValueError("config must define a non-empty sources list")
    if not isinstance(targets, list) or not targets:
        raise ValueError("config must define a non-empty targets list")

    rows: list[dict[str, object]] = []
    target_metadata: list[dict[str, object]] = []
    for target_spec in targets:
        target_name = str(target_spec["name"])
        if args.target is not None and target_name != args.target:
            continue
        target_domain = load_domain(target_name, Path(str(target_spec["data_dir"])))
        target_train_subjects = load_split_subjects(Path(str(target_spec["split_file"])), "train")
        support_subjects = select_support_subjects(
            target_domain, target_train_subjects, support_per_class, seed
        )
        target_indices = sample_subject_indices(
            target_domain, support_subjects, max_segments, seed + 101
        )
        target_dataset = AuditDataset(target_domain, target_indices)
        target_metadata.append(
            {
                "name": target_name,
                "support_subjects": support_subjects,
                "support_segments": int(len(target_indices)),
                "split_file": str(target_spec["split_file"]),
            }
        )
        target_rows: list[dict[str, float | str | int]] = []

        for source_index, source_spec in enumerate(sources):
            source_name = str(source_spec["name"])
            if source_name == target_name:
                continue
            source_domain = load_domain(source_name, Path(str(source_spec["data_dir"])))
            if source_spec.get("split_file"):
                source_subjects = load_split_subjects(Path(str(source_spec["split_file"])), "train")
            else:
                source_subjects = sorted({str(subject) for subject in source_domain.subject_ids})
            source_indices = sample_subject_indices(
                source_domain, source_subjects, max_segments, seed + 1000 + source_index
            )
            source_dataset = AuditDataset(source_domain, source_indices)
            model = load_model(Path(str(source_spec["checkpoint"])), config, device)
            source_repr = extract_representations(model, source_dataset, batch_size, device)
            target_repr = extract_representations(model, target_dataset, batch_size, device)
            source_gradient = mean_classifier_gradient(
                model, source_dataset, batch_size, gradient_batches, device
            )
            target_gradient = mean_classifier_gradient(
                model, target_dataset, batch_size, gradient_batches, device
            )
            target_rows.append(
                {
                    "target": target_name,
                    "source": source_name,
                    "checkpoint": str(source_spec["checkpoint"]),
                    "source_subjects": int(len(source_repr["subject_ids"])),
                    "target_support_subjects": int(len(target_repr["subject_ids"])),
                    "logme": logme_score(
                        target_repr["segment_fused"], target_repr["segment_labels"]
                    ),
                    "transrate": transrate_score(
                        target_repr["segment_fused"], target_repr["segment_labels"]
                    ),
                    "leep": leep_score(target_repr["probabilities"], target_repr["labels"]),
                    "gradient_cosine": float(source_gradient @ target_gradient),
                    "graph_prototype_distance": conditional_prototype_distance(
                        source_repr["graph"],
                        source_repr["labels"],
                        target_repr["graph"],
                        target_repr["labels"],
                    ),
                    "temporal_prototype_distance": conditional_prototype_distance(
                        source_repr["temporal"],
                        source_repr["labels"],
                        target_repr["temporal"],
                        target_repr["labels"],
                    ),
                    "fused_prototype_distance": conditional_prototype_distance(
                        source_repr["fused"],
                        source_repr["labels"],
                        target_repr["fused"],
                        target_repr["labels"],
                    ),
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        rank_scores = rank_aggregate(
            [{key: float(value) for key, value in row.items() if isinstance(value, (int, float))}
             for row in target_rows]
        )
        for row, rank_score in zip(target_rows, rank_scores):
            row["rank_score"] = rank_score
            rows.append(row)

    if not rows:
        raise ValueError("no target/source pairs matched the requested configuration")
    output_prefix = Path(
        args.output_prefix
        or str(config.get("output_prefix", "outputs/transferability_audit/audit"))
    )
    metadata = {
        "config": str(config_path),
        "seed": seed,
        "device": str(device),
        "support_subjects_per_class": support_per_class,
        "max_segments_per_subject": max_segments,
        "gradient_batches": gradient_batches,
        "targets": target_metadata,
        "test_or_validation_data_used": False,
    }
    write_outputs(rows, metadata, output_prefix)
    print(f"Saved transferability audit to {output_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
