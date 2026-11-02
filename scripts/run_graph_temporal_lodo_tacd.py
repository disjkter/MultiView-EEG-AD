"""Run strict unseen-target LODO with shared DG objectives and EEG backbones."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.GraphTemporalTACD import (  # noqa: E402
    GraphTemporalTACD,
    class_conditional_margin_alignment,
    cross_covariance_penalty,
    dann_schedule,
    multi_domain_coral_loss,
    multi_domain_mmd_loss,
)
from models.EEGNetDomainDecoupling import EEGNetDomainDecoupling  # noqa: E402
from models.ADformerDomainAdapter import ADformerDomainAdapter  # noqa: E402
from models.StandardEEGBackboneAdapter import StandardEEGBackboneAdapter  # noqa: E402
from models.CalibrationAwareDG import (  # noqa: E402
    aggregate_participant_probabilities,
    calibration_aware_dg_loss,
    participant_classification_loss,
)
from scripts.run_dual_stream_subject_split import (  # noqa: E402
    choose_threshold,
    disambiguate_mixed_label_subject_ids,
    load_domain_mmap,
    sample_indices_per_subject,
    subject_labels,
)


@dataclass
class DomainStore:
    """Memory-mapped EEG domain and its labels/subject identifiers."""

    name: str
    x: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray


@dataclass(frozen=True)
class SampleRef:
    """Reference to one segment in a domain store."""

    store_index: int
    segment_index: int
    domain_index: int


class MultiDomainDataset(Dataset):
    """Index memory-mapped segments across multiple domains."""

    def __init__(self, stores: list[DomainStore], refs: list[SampleRef]) -> None:
        self.stores = stores
        self.refs = refs

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, str]:
        ref = self.refs[index]
        store = self.stores[ref.store_index]
        sample = np.array(store.x[ref.segment_index], dtype=np.float32, copy=True)
        mean = sample.mean(axis=1, keepdims=True)
        std = sample.std(axis=1, keepdims=True) + 1e-6
        sample = (sample - mean) / std
        subject = f"{store.name}::{store.subject_ids[ref.segment_index]}"
        return (
            torch.from_numpy(sample),
            int(store.y[ref.segment_index]),
            ref.domain_index,
            subject,
        )


def set_seed(seed: int) -> None:
    """Set deterministic random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_store(name: str, path: Path) -> DomainStore:
    """Load one processed domain through NumPy memory mapping."""

    x, y, subject_ids = load_domain_mmap(path)
    subject_ids = disambiguate_mixed_label_subject_ids(y, subject_ids)
    if x.ndim != 3 or x.shape[1] != 19 or x.shape[2] < 128:
        raise ValueError(f"{name} must have shape [N, 19, T>=128], got {x.shape}")
    return DomainStore(name=name, x=x, y=y, subject_ids=subject_ids)


def capped_subjects(
    store: DomainStore,
    max_per_class: int,
    seed: int,
) -> set[str]:
    """Select at most a fixed number of subjects from each disease class."""

    labels = subject_labels(store.y, store.subject_ids)
    rng = np.random.default_rng(seed)
    selected: set[str] = set()
    for label in sorted(set(labels.values())):
        subjects = np.asarray(
            [str(subject) for subject, value in labels.items() if value == label],
            dtype=object,
        )
        rng.shuffle(subjects)
        if max_per_class > 0:
            subjects = subjects[:max_per_class]
        selected.update(str(subject) for subject in subjects)
    return selected


def split_source_indices(
    store: DomainStore,
    val_ratio: float,
    max_subjects_per_class: int,
    max_segments_per_subject: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a stratified source-only train/validation split."""

    selected = capped_subjects(store, max_subjects_per_class, seed)
    labels = subject_labels(store.y, store.subject_ids)
    rng = np.random.default_rng(seed + 17)
    train_subjects: set[str] = set()
    val_subjects: set[str] = set()
    for label in sorted(set(labels.values())):
        subjects = np.asarray(
            [
                str(subject)
                for subject, value in labels.items()
                if value == label and str(subject) in selected
            ],
            dtype=object,
        )
        rng.shuffle(subjects)
        n_val = max(1, int(round(len(subjects) * val_ratio)))
        if n_val >= len(subjects):
            n_val = len(subjects) - 1
        val_subjects.update(str(subject) for subject in subjects[:n_val])
        train_subjects.update(str(subject) for subject in subjects[n_val:])

    subject_strings = np.asarray([str(value) for value in store.subject_ids], dtype=object)
    train_indices = np.flatnonzero(np.isin(subject_strings, list(train_subjects)))
    val_indices = np.flatnonzero(np.isin(subject_strings, list(val_subjects)))
    train_indices = sample_indices_per_subject(
        train_indices, subject_strings, max_segments_per_subject, seed + 31
    )
    val_indices = sample_indices_per_subject(
        val_indices, subject_strings, max_segments_per_subject, seed + 47
    )
    return train_indices, val_indices


def split_source_indices_three_way(
    store: DomainStore,
    checkpoint_ratio: float,
    calibration_ratio: float,
    max_subjects_per_class: int,
    max_segments_per_subject: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create disjoint source train, checkpoint, and calibration splits.

    Splitting is performed independently within each source-domain class. The
    checkpoint split is used only for early stopping, while the calibration
    split is used only for threshold or post-hoc calibration fitting.
    """

    if checkpoint_ratio <= 0.0 or calibration_ratio <= 0.0:
        raise ValueError("Three-way source split ratios must be positive")
    if checkpoint_ratio + calibration_ratio >= 1.0:
        raise ValueError("Three-way source validation ratios must sum to less than one")
    selected = capped_subjects(store, max_subjects_per_class, seed)
    labels = subject_labels(store.y, store.subject_ids)
    rng = np.random.default_rng(seed + 17)
    train_subjects: set[str] = set()
    checkpoint_subjects: set[str] = set()
    calibration_subjects: set[str] = set()
    for label in sorted(set(labels.values())):
        subjects = np.asarray(
            [
                str(subject)
                for subject, value in labels.items()
                if value == label and str(subject) in selected
            ],
            dtype=object,
        )
        if len(subjects) < 5:
            raise ValueError(
                f"Three-way split requires at least five subjects in class {label} "
                f"for {store.name}, found {len(subjects)}"
            )
        rng.shuffle(subjects)
        n_checkpoint = max(1, int(round(len(subjects) * checkpoint_ratio)))
        n_calibration = max(1, int(round(len(subjects) * calibration_ratio)))
        if n_checkpoint + n_calibration >= len(subjects):
            raise ValueError(f"Three-way split leaves no training subjects for {store.name}")
        checkpoint_subjects.update(str(value) for value in subjects[:n_checkpoint])
        calibration_subjects.update(
            str(value) for value in subjects[n_checkpoint : n_checkpoint + n_calibration]
        )
        train_subjects.update(str(value) for value in subjects[n_checkpoint + n_calibration :])

    if train_subjects & checkpoint_subjects or train_subjects & calibration_subjects or checkpoint_subjects & calibration_subjects:
        raise RuntimeError("Three-way participant split overlap detected")
    subject_strings = np.asarray([str(value) for value in store.subject_ids], dtype=object)

    def sampled(subjects: set[str], offset: int) -> np.ndarray:
        indices = np.flatnonzero(np.isin(subject_strings, list(subjects)))
        return sample_indices_per_subject(
            indices, subject_strings, max_segments_per_subject, seed + offset
        )

    return sampled(train_subjects, 31), sampled(checkpoint_subjects, 47), sampled(calibration_subjects, 59)


def make_refs(
    stores: list[DomainStore],
    index_sets: list[np.ndarray],
) -> list[SampleRef]:
    """Build segment references with contiguous domain labels."""

    return [
        SampleRef(store_index, int(segment_index), store_index)
        for store_index, indices in enumerate(index_sets)
        for segment_index in indices
    ]


def domain_balanced_sampler(refs: list[SampleRef], num_domains: int) -> WeightedRandomSampler:
    """Sample each source domain with equal expected probability."""

    counts = np.bincount(
        np.asarray([ref.domain_index for ref in refs]), minlength=num_domains
    )
    weights = torch.as_tensor(
        [1.0 / counts[ref.domain_index] for ref in refs], dtype=torch.double
    )
    samples_per_epoch = int(counts.max() * num_domains)
    return WeightedRandomSampler(weights, samples_per_epoch, replacement=True)


class DomainBalancedBatchSampler(Sampler[list[int]]):
    """Yield reproducible mini-batches with equal samples from every source domain."""

    def __init__(
        self,
        refs: list[SampleRef],
        num_domains: int,
        batch_size: int,
        seed: int,
    ) -> None:
        if num_domains < 2:
            raise ValueError("At least two source domains are required")
        self.samples_per_domain = batch_size // num_domains
        if self.samples_per_domain < 2:
            raise ValueError(
                "batch_size must provide at least two samples per source domain"
            )
        self.indices = [
            np.asarray(
                [index for index, ref in enumerate(refs) if ref.domain_index == domain],
                dtype=np.int64,
            )
            for domain in range(num_domains)
        ]
        if any(values.size == 0 for values in self.indices):
            raise ValueError("Every source domain must contain at least one sample")
        self.num_batches = max(
            math.ceil(values.size / self.samples_per_domain)
            for values in self.indices
        )
        self.seed = int(seed)
        self.iteration = 0

    def __iter__(self):
        """Sample each domain independently and combine equal-sized chunks."""

        rng = np.random.default_rng(self.seed + self.iteration)
        self.iteration += 1
        sampled = [
            rng.choice(
                values,
                size=self.num_batches * self.samples_per_domain,
                replace=values.size < self.num_batches * self.samples_per_domain,
            ).reshape(self.num_batches, self.samples_per_domain)
            for values in self.indices
        ]
        for batch_index in range(self.num_batches):
            batch = np.concatenate(
                [values[batch_index] for values in sampled]
            )
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self) -> int:
        return self.num_batches


class ParticipantBalancedBatchSampler(Sampler[list[int]]):
    """Balance source cohort, class, and participant in every mini-batch."""

    def __init__(
        self,
        stores: list[DomainStore],
        refs: list[SampleRef],
        subjects_per_domain_class: int,
        windows_per_subject: int,
        seed: int,
    ) -> None:
        if subjects_per_domain_class < 1 or windows_per_subject < 1:
            raise ValueError("participant and window counts must be positive")
        cells: dict[tuple[int, int], dict[str, list[int]]] = {}
        for dataset_index, ref in enumerate(refs):
            store = stores[ref.store_index]
            label = int(store.y[ref.segment_index])
            subject = str(store.subject_ids[ref.segment_index])
            cells.setdefault((ref.domain_index, label), {}).setdefault(
                subject, []
            ).append(dataset_index)
        expected = {
            (domain, label)
            for domain in range(len(stores))
            for label in (0, 1)
        }
        missing = expected.difference(cells)
        if missing:
            raise ValueError(f"Missing domain/class participant cells: {sorted(missing)}")
        self.cells = cells
        self.subjects_per_cell = int(subjects_per_domain_class)
        self.windows_per_subject = int(windows_per_subject)
        self.num_batches = max(
            math.ceil(len(subjects) / self.subjects_per_cell)
            for subjects in cells.values()
        )
        self.seed = int(seed)
        self.iteration = 0

    def __iter__(self):
        """Sample participants first and windows second within every cell."""

        rng = np.random.default_rng(self.seed + self.iteration)
        self.iteration += 1
        for batch_index in range(self.num_batches):
            batch: list[int] = []
            for cell in sorted(self.cells):
                subject_windows = self.cells[cell]
                subjects = np.asarray(sorted(subject_windows), dtype=object)
                selected_subjects = rng.choice(
                    subjects,
                    size=self.subjects_per_cell,
                    replace=subjects.size < self.subjects_per_cell,
                )
                for subject in selected_subjects:
                    windows = np.asarray(self.cells[cell][str(subject)], dtype=np.int64)
                    selected = rng.choice(
                        windows,
                        size=self.windows_per_subject,
                        replace=windows.size < self.windows_per_subject,
                    )
                    batch.extend(int(index) for index in selected)
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    disease_criterion: nn.Module,
    domain_criterion: nn.Module,
    method: str,
    grl_scale: float,
    lambda_domain: float,
    lambda_specific: float,
    lambda_orthogonal: float,
    lambda_margin: float,
    margin_variance_weight: float,
    lambda_alignment: float,
    lambda_care_score: float,
    lambda_care_calibration: float,
    lambda_care_entropy: float,
    care_calibration_bandwidth: float,
    care_calibration_equalization_weight: float,
    device: torch.device,
) -> dict[str, float]:
    """Train one source-domain epoch."""

    model.train()
    totals: list[float] = []
    disease_losses: list[float] = []
    margin_losses: list[float] = []
    alignment_losses: list[float] = []
    care_score_losses: list[float] = []
    care_calibration_losses: list[float] = []
    care_entropy_losses: list[float] = []
    for x, y, domain, subjects in loader:
        x = x.to(device)
        y = y.to(device)
        domain = domain.to(device)
        output = model(
            x,
            grl_scale=grl_scale if method in {"dann", "tacd"} else 0.0,
        )
        if method == "participant_care":
            participant_probability, participant_y, participant_domain = (
                aggregate_participant_probabilities(
                    output["disease_logits"], y, domain, subjects
                )
            )
            disease_loss = participant_classification_loss(
                participant_probability, participant_y
            )
        else:
            participant_probability = None
            participant_y = y
            participant_domain = domain
            disease_loss = disease_criterion(output["disease_logits"], y)
        loss = disease_loss
        if method in {"dann", "tacd"}:
            loss = loss + lambda_domain * domain_criterion(
                output["domain_adversarial_logits"], domain
            )
        if method == "tacd":
            loss = loss + lambda_specific * domain_criterion(
                output["domain_specific_logits"], domain
            )
            loss = loss + lambda_orthogonal * cross_covariance_penalty(
                output["disease_feature"], output["domain_feature"]
            )
        margin_loss = class_conditional_margin_alignment(
            output["disease_logits"],
            y,
            domain,
            variance_weight=margin_variance_weight,
        )
        loss = loss + lambda_margin * margin_loss
        if method == "coral":
            alignment_loss = multi_domain_coral_loss(
                output["disease_feature"], domain
            )
        elif method == "mmd":
            alignment_loss = multi_domain_mmd_loss(
                output["disease_feature"], domain
            )
        else:
            alignment_loss = output["disease_feature"].sum() * 0.0
        loss = loss + lambda_alignment * alignment_loss
        if method == "participant_care":
            participant_logits = participant_probability.clamp_min(1e-7).log()
            care_losses = calibration_aware_dg_loss(
                participant_logits,
                participant_y,
                participant_domain,
                calibration_bandwidth=care_calibration_bandwidth,
                calibration_equalization_weight=care_calibration_equalization_weight,
            )
            care_losses["error_entropy"] = participant_logits.sum() * 0.0
        elif method == "care_dg":
            care_losses = calibration_aware_dg_loss(
                output["disease_logits"],
                y,
                domain,
                calibration_bandwidth=care_calibration_bandwidth,
                calibration_equalization_weight=care_calibration_equalization_weight,
            )
        else:
            zero = output["disease_logits"].sum() * 0.0
            care_losses = {
                "score_alignment": zero,
                "calibration": zero,
                "error_entropy": zero,
            }
        loss = loss + lambda_care_score * care_losses["score_alignment"]
        loss = loss + lambda_care_calibration * care_losses["calibration"]
        loss = loss + lambda_care_entropy * care_losses["error_entropy"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        totals.append(float(loss.item()))
        disease_losses.append(float(disease_loss.item()))
        margin_losses.append(float(margin_loss.item()))
        alignment_losses.append(float(alignment_loss.item()))
        care_score_losses.append(float(care_losses["score_alignment"].item()))
        care_calibration_losses.append(float(care_losses["calibration"].item()))
        care_entropy_losses.append(float(care_losses["error_entropy"].item()))
    return {
        "loss": float(np.mean(totals)),
        "disease_loss": float(np.mean(disease_losses)),
        "margin_loss": float(np.mean(margin_losses)),
        "alignment_loss": float(np.mean(alignment_losses)),
        "care_score_loss": float(np.mean(care_score_losses)),
        "care_calibration_loss": float(np.mean(care_calibration_losses)),
        "care_entropy_loss": float(np.mean(care_entropy_losses)),
    }


def collect_subject_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average segment probabilities into subject-level predictions."""

    model.eval()
    probabilities: dict[str, list[np.ndarray]] = {}
    labels: dict[str, int] = {}
    with torch.no_grad():
        for x, y, _, subjects in loader:
            logits = model(x.to(device))["disease_logits"]
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for subject, label, probability in zip(subjects, y.numpy(), probs):
                probabilities.setdefault(str(subject), []).append(probability)
                labels[str(subject)] = int(label)
    ordered = sorted(probabilities)
    y_true = np.asarray([labels[subject] for subject in ordered], dtype=np.int64)
    y_prob = np.stack(
        [np.mean(probabilities[subject], axis=0) for subject in ordered]
    )
    return np.asarray(ordered, dtype=object), y_true, y_prob


def binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Compute subject-level binary metrics at a fixed decision threshold."""

    y_pred = (y_prob[:, 1] >= threshold).astype(np.int64)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob[:, 1])),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob[:, 1].tolist(),
    }


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute source validation disease loss."""

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for x, y, _, _ in loader:
            logits = model(x.to(device))["disease_logits"]
            losses.append(float(criterion(logits, y.to(device)).item()))
    return float(np.mean(losses))


def load_backbone_checkpoint(
    model: nn.Module,
    checkpoint: str | None,
    device: torch.device,
) -> int:
    """Load shape-compatible weights into the selected feature backbone."""

    if not checkpoint:
        return 0
    payload = torch.load(checkpoint, map_location=device)
    state = payload.get("encoder_state_dict", payload) if isinstance(payload, dict) else payload
    target = model.backbone.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in target and target[key].shape == value.shape
    }
    model.backbone.load_state_dict(compatible, strict=False)
    return len(compatible)


def build_model(args: argparse.Namespace, num_domains: int) -> nn.Module:
    """Build a DG model while preserving one shared training contract."""

    if args.backbone == "graph_temporal":
        return GraphTemporalTACD(
            num_domains=num_domains,
            graph_out_dim=args.graph_out_dim,
            temporal_out_dim=args.temporal_out_dim,
            disease_dim=args.disease_dim,
            domain_dim=args.domain_dim,
            classifier_dropout=args.dropout,
            num_neighbors=args.graph_num_neighbors,
        )
    if args.backbone == "eegnet":
        return EEGNetDomainDecoupling(
            num_domains=num_domains,
            temporal_filters=args.eegnet_temporal_filters,
            depth_multiplier=args.eegnet_depth_multiplier,
            separable_filters=args.eegnet_separable_filters,
            temporal_kernel=args.eegnet_temporal_kernel,
            separable_kernel=args.eegnet_separable_kernel,
            disease_dim=args.disease_dim,
            domain_dim=args.domain_dim,
            classifier_dropout=args.dropout,
        )
    if args.backbone == "adformer":
        if args.method != "erm":
            raise ValueError("The contemporary ADformer baseline is evaluated with ERM only")
        return ADformerDomainAdapter(
            num_domains=num_domains,
            patch_len_list=args.adformer_patch_lengths,
            up_dim_list=args.adformer_up_dims,
            d_model=args.adformer_d_model,
            n_heads=args.adformer_num_heads,
            e_layers=args.adformer_num_layers,
            d_ff=args.adformer_ff_dim,
            dropout=args.dropout,
        )
    if args.backbone in {"shallow_convnet", "eeg_conformer"}:
        if args.method != "erm" or args.lambda_margin != 0.0:
            raise ValueError(
                f"{args.backbone} is evaluated with unmodified ERM only"
            )
        return StandardEEGBackboneAdapter(
            backbone_name=args.backbone,
            num_domains=num_domains,
            dropout=args.dropout,
            conformer_d_model=args.conformer_d_model,
            conformer_num_heads=args.conformer_num_heads,
            conformer_num_layers=args.conformer_num_layers,
            conformer_ff_dim=args.conformer_ff_dim,
        )
    raise ValueError(f"Unsupported backbone: {args.backbone}")


def run_fold(
    stores_by_name: dict[str, DomainStore],
    target_name: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, object]:
    """Train on source domains and evaluate once on the untouched target."""

    source_stores = [
        store for name, store in stores_by_name.items() if name != target_name
    ]
    target_store = stores_by_name[target_name]
    train_sets: list[np.ndarray] = []
    val_sets: list[np.ndarray] = []
    calibration_sets: list[np.ndarray] = []
    for index, store in enumerate(source_stores):
        if args.separate_selection_splits:
            train_indices, val_indices, calibration_indices = split_source_indices_three_way(
                store,
                args.checkpoint_val_ratio,
                args.calibration_val_ratio,
                args.max_source_subjects_per_class,
                args.max_segments_per_subject,
                args.seed + 100 * index,
            )
        else:
            train_indices, val_indices = split_source_indices(
                store,
                args.val_ratio,
                args.max_source_subjects_per_class,
                args.max_segments_per_subject,
                args.seed + 100 * index,
            )
            calibration_indices = val_indices
        train_sets.append(train_indices)
        val_sets.append(val_indices)
        calibration_sets.append(calibration_indices)

    target_indices = sample_indices_per_subject(
        np.arange(len(target_store.y)),
        target_store.subject_ids,
        args.max_target_segments_per_subject,
        args.seed + 9000,
    )
    train_refs = make_refs(source_stores, train_sets)
    val_refs = make_refs(source_stores, val_sets)
    calibration_refs = make_refs(source_stores, calibration_sets)
    target_refs = make_refs([target_store], [target_indices])
    train_dataset = MultiDomainDataset(source_stores, train_refs)
    val_dataset = MultiDomainDataset(source_stores, val_refs)
    calibration_dataset = MultiDomainDataset(source_stores, calibration_refs)
    target_dataset = MultiDomainDataset([target_store], target_refs)
    if args.domain_balanced_batches:
        if args.method == "participant_care":
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=ParticipantBalancedBatchSampler(
                    source_stores,
                    train_refs,
                    args.participant_subjects_per_domain_class,
                    args.participant_windows_per_subject,
                    args.seed,
                ),
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=DomainBalancedBatchSampler(
                    train_refs,
                    len(source_stores),
                    args.batch_size,
                    args.seed,
                ),
            )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=domain_balanced_sampler(train_refs, len(source_stores)),
            drop_last=False,
        )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    calibration_loader = DataLoader(
        calibration_dataset, batch_size=args.batch_size, shuffle=False
    )
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)

    device_name = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    device = torch.device(device_name)
    model = build_model(args, len(source_stores)).to(device)
    loaded_tensors = load_backbone_checkpoint(model, args.checkpoint, device)

    train_labels = np.concatenate(
        [store.y[indices] for store, indices in zip(source_stores, train_sets)]
    )
    counts = np.bincount(train_labels, minlength=2)
    class_weights = len(train_labels) / (2 * np.maximum(counts, 1))
    disease_criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pth"
    best_val_loss = float("inf")
    patience_left = args.patience
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        grl_scale = dann_schedule(epoch - 1, args.epochs)
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            disease_criterion,
            domain_criterion,
            args.method,
            grl_scale,
            args.lambda_domain,
            args.lambda_specific,
            args.lambda_orthogonal,
            args.lambda_margin,
            args.margin_variance_weight,
            args.lambda_alignment,
            args.lambda_care_score,
            args.lambda_care_calibration,
            args.lambda_care_entropy,
            args.care_calibration_bandwidth,
            args.care_calibration_equalization_weight,
            device,
        )
        val_loss = evaluate_loss(model, val_loader, disease_criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_disease_loss": train_stats["disease_loss"],
                "train_margin_loss": train_stats["margin_loss"],
                "train_alignment_loss": train_stats["alignment_loss"],
                "train_care_score_loss": train_stats["care_score_loss"],
                "train_care_calibration_loss": train_stats["care_calibration_loss"],
                "train_care_entropy_loss": train_stats["care_entropy_loss"],
                "val_loss": val_loss,
                "grl_scale": grl_scale,
            }
        )
        print(
            f"{target_name} {args.method} epoch={epoch:02d} "
            f"train={train_stats['loss']:.4f} val={val_loss:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left = args.patience
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    checkpoint_subjects, checkpoint_true, checkpoint_prob = collect_subject_probabilities(
        model, val_loader, device
    )
    calibration_subjects, calibration_true, calibration_prob = collect_subject_probabilities(
        model, calibration_loader, device
    )
    decision_threshold = 0.5
    threshold_score = None
    threshold_subjects = int(len(calibration_subjects))
    if args.tune_threshold:
        decision_threshold, threshold_score = choose_threshold(
            calibration_true, calibration_prob, args.threshold_objective
        )

    subjects, y_true, y_prob = collect_subject_probabilities(
        model, target_loader, device
    )
    metrics = binary_metrics(y_true, y_prob, decision_threshold)
    metrics["subject_ids"] = subjects.tolist()
    fixed_threshold_metrics = binary_metrics(y_true, y_prob, 0.5)
    source_validation_metrics = binary_metrics(
        calibration_true, calibration_prob, decision_threshold
    )
    source_validation_metrics["subject_ids"] = calibration_subjects.tolist()
    checkpoint_validation_metrics = binary_metrics(
        checkpoint_true, checkpoint_prob, 0.5
    )
    checkpoint_validation_metrics["subject_ids"] = checkpoint_subjects.tolist()
    summary = {
        "variant": args.variant_name or args.method,
        "method": args.method,
        "backbone": args.backbone,
        "seed": args.seed,
        "target": target_name,
        "sources": [store.name for store in source_stores],
        "target_data_used_during_training": False,
        "target_data_used_for_model_selection": False,
        "target_data_used_for_threshold_selection": False,
        "domain_balanced_batches": bool(args.domain_balanced_batches),
        "tune_threshold": bool(args.tune_threshold),
        "threshold_objective": (
            args.threshold_objective if args.tune_threshold else None
        ),
        "threshold_source": (
            "pooled_source_calibration_subjects"
            if args.separate_selection_splits and args.tune_threshold
            else "pooled_source_validation_subjects"
            if args.tune_threshold
            else "fixed_0.5"
        ),
        "separate_selection_splits": bool(args.separate_selection_splits),
        "threshold_validation_subjects": threshold_subjects,
        "decision_threshold": float(decision_threshold),
        "threshold_objective_score": threshold_score,
        "loaded_checkpoint_tensors": loaded_tensors,
        "lambda_domain": float(args.lambda_domain),
        "lambda_specific": float(args.lambda_specific),
        "lambda_orthogonal": float(args.lambda_orthogonal),
        "lambda_margin": float(args.lambda_margin),
        "margin_variance_weight": float(args.margin_variance_weight),
        "lambda_alignment": float(args.lambda_alignment),
        "lambda_care_score": float(args.lambda_care_score),
        "lambda_care_calibration": float(args.lambda_care_calibration),
        "lambda_care_entropy": float(args.lambda_care_entropy),
        "care_calibration_bandwidth": float(args.care_calibration_bandwidth),
        "care_calibration_equalization_weight": float(
            args.care_calibration_equalization_weight
        ),
        "participant_subjects_per_domain_class": int(
            args.participant_subjects_per_domain_class
        ),
        "participant_windows_per_subject": int(args.participant_windows_per_subject),
        "source_train_subjects": {
            store.name: int(len(np.unique(store.subject_ids[indices])))
            for store, indices in zip(source_stores, train_sets)
        },
        "source_val_subjects": {
            store.name: int(len(np.unique(store.subject_ids[indices])))
            for store, indices in zip(source_stores, val_sets)
        },
        "source_calibration_subjects": {
            store.name: int(len(np.unique(store.subject_ids[indices])))
            for store, indices in zip(source_stores, calibration_sets)
        },
        "source_train_subject_ids": {
            store.name: sorted(
                str(value) for value in np.unique(store.subject_ids[indices])
            )
            for store, indices in zip(source_stores, train_sets)
        },
        "source_val_subject_ids": {
            store.name: sorted(
                str(value) for value in np.unique(store.subject_ids[indices])
            )
            for store, indices in zip(source_stores, val_sets)
        },
        "source_calibration_subject_ids": {
            store.name: sorted(
                str(value) for value in np.unique(store.subject_ids[indices])
            )
            for store, indices in zip(source_stores, calibration_sets)
        },
        "max_source_subjects_per_class": int(args.max_source_subjects_per_class),
        "max_segments_per_subject": int(args.max_segments_per_subject),
        "input_samples": int(target_store.x.shape[2]),
        "target_test_subjects": int(len(subjects)),
        "best_val_loss": best_val_loss,
        "history": history,
        "source_validation_metrics": source_validation_metrics,
        "checkpoint_validation_metrics": checkpoint_validation_metrics,
        "metrics": metrics,
        "fixed_threshold_metrics": fixed_threshold_metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def parse_domains(values: list[str]) -> dict[str, Path]:
    """Parse repeated name=path domain specifications."""

    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected name=path, got {value}")
        name, path = value.split("=", 1)
        result[name] = Path(path)
    if len(result) < 3:
        raise ValueError("Strict LODO requires at least three domains")
    return result


def main() -> None:
    """Run one or all strict LODO folds."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--target", action="append", default=None)
    parser.add_argument(
        "--method",
        choices=["erm", "dann", "tacd", "coral", "mmd", "care_dg", "participant_care"],
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--variant-name",
        default=None,
        help="Stable analysis label when several variants share one training method.",
    )
    parser.add_argument(
        "--backbone",
        choices=[
            "graph_temporal",
            "eegnet",
            "adformer",
            "shallow_convnet",
            "eeg_conformer",
        ],
        default="graph_temporal",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--separate-selection-splits", action="store_true")
    parser.add_argument("--checkpoint-val-ratio", type=float, default=0.15)
    parser.add_argument("--calibration-val-ratio", type=float, default=0.15)
    parser.add_argument("--max-source-subjects-per-class", type=int, default=50)
    parser.add_argument("--max-segments-per-subject", type=int, default=20)
    parser.add_argument("--max-target-segments-per-subject", type=int, default=20)
    parser.add_argument("--graph-out-dim", type=int, default=256)
    parser.add_argument("--temporal-out-dim", type=int, default=256)
    parser.add_argument("--disease-dim", type=int, default=256)
    parser.add_argument("--domain-dim", type=int, default=128)
    parser.add_argument("--graph-num-neighbors", type=int, default=4)
    parser.add_argument("--eegnet-temporal-filters", type=int, default=8)
    parser.add_argument("--eegnet-depth-multiplier", type=int, default=2)
    parser.add_argument("--eegnet-separable-filters", type=int, default=16)
    parser.add_argument("--eegnet-temporal-kernel", type=int, default=64)
    parser.add_argument("--eegnet-separable-kernel", type=int, default=16)
    parser.add_argument("--adformer-patch-lengths", default="2")
    parser.add_argument("--adformer-up-dims", default="19")
    parser.add_argument("--adformer-d-model", type=int, default=128)
    parser.add_argument("--adformer-num-heads", type=int, default=8)
    parser.add_argument("--adformer-num-layers", type=int, default=6)
    parser.add_argument("--adformer-ff-dim", type=int, default=256)
    parser.add_argument("--conformer-d-model", type=int, default=128)
    parser.add_argument("--conformer-num-heads", type=int, default=8)
    parser.add_argument("--conformer-num-layers", type=int, default=4)
    parser.add_argument("--conformer-ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lambda-domain", type=float, default=0.2)
    parser.add_argument("--lambda-specific", type=float, default=0.1)
    parser.add_argument("--lambda-orthogonal", type=float, default=0.01)
    parser.add_argument("--lambda-margin", type=float, default=0.0)
    parser.add_argument("--margin-variance-weight", type=float, default=0.1)
    parser.add_argument("--lambda-alignment", type=float, default=1.0)
    parser.add_argument("--lambda-care-score", type=float, default=0.05)
    parser.add_argument("--lambda-care-calibration", type=float, default=0.5)
    parser.add_argument("--lambda-care-entropy", type=float, default=0.05)
    parser.add_argument("--care-calibration-bandwidth", type=float, default=0.1)
    parser.add_argument(
        "--care-calibration-equalization-weight", type=float, default=1.0
    )
    parser.add_argument("--participant-subjects-per-domain-class", type=int, default=4)
    parser.add_argument("--participant-windows-per-subject", type=int, default=2)
    parser.add_argument("--domain-balanced-batches", action="store_true")
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument(
        "--threshold-objective",
        choices=["accuracy", "balanced_acc", "f1", "precision", "specificity", "youden"],
        default="balanced_acc",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    care_values = (
        args.lambda_care_score,
        args.lambda_care_calibration,
        args.lambda_care_entropy,
        args.care_calibration_equalization_weight,
    )
    if any(value < 0.0 for value in care_values):
        parser.error("CARE-DG loss weights must be non-negative")
    if args.care_calibration_bandwidth <= 0.0:
        parser.error("--care-calibration-bandwidth must be positive")
    if args.method in {"care_dg", "participant_care"} and not args.domain_balanced_batches:
        parser.error("calibration-aware methods require --domain-balanced-batches")
    if args.participant_subjects_per_domain_class < 1:
        parser.error("--participant-subjects-per-domain-class must be positive")
    if args.participant_windows_per_subject < 1:
        parser.error("--participant-windows-per-subject must be positive")
    if args.separate_selection_splits:
        if args.checkpoint_val_ratio <= 0.0 or args.calibration_val_ratio <= 0.0:
            parser.error("separate selection split ratios must be positive")
        if args.checkpoint_val_ratio + args.calibration_val_ratio >= 1.0:
            parser.error("separate selection split ratios must sum to less than one")

    set_seed(args.seed)
    domain_paths = parse_domains(args.domains)
    stores = {
        name: load_store(name, path) for name, path in domain_paths.items()
    }
    targets = args.target or list(stores)
    unknown = set(targets).difference(stores)
    if unknown:
        raise ValueError(f"Unknown targets: {sorted(unknown)}")
    output_root = Path(args.output_dir)
    fold_summaries = [
        run_fold(stores, target, args, output_root / f"target_{target}")
        for target in targets
    ]
    aggregate = {
        metric: float(
            np.mean([fold["metrics"][metric] for fold in fold_summaries])
        )
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "f1",
            "auc",
            "sensitivity",
            "specificity",
        )
    }
    payload = {
        "variant": args.variant_name or args.method,
        "method": args.method,
        "backbone": args.backbone,
        "seed": args.seed,
        "domains": list(stores),
        "target_data_used_during_training": False,
        "folds": fold_summaries,
        "aggregate": aggregate,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "lodo_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
