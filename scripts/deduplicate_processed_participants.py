"""Create a processed EEG copy with exact duplicate participants removed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_dual_stream_loso_subject import resolve_domain_files  # noqa: E402


def participant_digest(data: np.ndarray, indices: np.ndarray) -> str:
    """Hash all ordered windows for one participant."""

    digest = hashlib.sha256()
    for index in indices:
        window = np.ascontiguousarray(data[int(index)])
        digest.update(str(window.dtype).encode("ascii"))
        digest.update(np.asarray(window.shape, dtype=np.int64).tobytes())
        digest.update(window.view(np.uint8))
    return digest.hexdigest()


def find_duplicate_groups(
    data: np.ndarray, labels: np.ndarray, subject_ids: np.ndarray
) -> list[dict[str, object]]:
    """Return exact duplicate participant groups with label validation."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, subject in enumerate(subject_ids):
        grouped[str(subject)].append(index)
    by_digest: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for subject in sorted(grouped):
        indices = np.asarray(grouped[subject], dtype=np.int64)
        unique_labels = np.unique(labels[indices])
        if len(unique_labels) != 1:
            raise ValueError(f"Mixed labels for participant {subject}")
        by_digest[participant_digest(data, indices)].append(
            (subject, int(unique_labels[0]))
        )

    result = []
    for digest, members in sorted(by_digest.items()):
        if len(members) < 2:
            continue
        member_labels = {label for _, label in members}
        if len(member_labels) != 1:
            raise ValueError(f"Duplicate signal has conflicting labels: {members}")
        ordered = sorted(subject for subject, _ in members)
        result.append(
            {
                "sha256": digest,
                "label": next(iter(member_labels)),
                "kept_participant": ordered[0],
                "removed_participants": ordered[1:],
                "group_size": len(ordered),
            }
        )
    return result


def main() -> None:
    """Write a de-duplicated processed dataset without modifying the source."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    x_path, y_path, ids_path = resolve_domain_files(input_dir)
    data = np.load(x_path, mmap_mode="r")
    labels = np.asarray(np.load(y_path, mmap_mode="r"), dtype=np.int64)
    subject_ids = np.asarray(np.load(ids_path, allow_pickle=True), dtype=object)

    duplicate_groups = find_duplicate_groups(data, labels, subject_ids)
    removed = {
        subject
        for group in duplicate_groups
        for subject in group["removed_participants"]
    }
    keep_mask = np.asarray([str(subject) not in removed for subject in subject_ids])
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "x_19ch.npy", np.asarray(data[keep_mask], dtype=np.float32))
    np.save(output_dir / "y_19ch.npy", labels[keep_mask])
    np.save(output_dir / "sub_ids_19ch.npy", subject_ids[keep_mask])

    kept_ids = np.asarray(subject_ids[keep_mask], dtype=object)
    kept_labels = labels[keep_mask]
    summary = {
        "source_dir": str(input_dir),
        "output_dir": str(output_dir),
        "operation": "exact participant-level duplicate removal",
        "source_participants": int(len(np.unique(subject_ids))),
        "output_participants": int(len(np.unique(kept_ids))),
        "removed_participants": sorted(removed),
        "duplicate_groups": duplicate_groups,
        "shape": list(data[keep_mask].shape),
        "subjects_by_label": {
            str(label): int(len(np.unique(kept_ids[kept_labels == label])))
            for label in sorted(np.unique(kept_labels))
        },
        "segments_by_label": {
            str(label): int(np.sum(kept_labels == label))
            for label in sorted(np.unique(kept_labels))
        },
        "source_files": {
            "x": str(x_path),
            "y": str(y_path),
            "subject_ids": str(ids_path),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
