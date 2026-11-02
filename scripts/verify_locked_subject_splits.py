"""Verify that locked subject splits exactly cover the current dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_dual_stream_subject_split import load_domain_mmap  # noqa: E402
from utils.subject_split import load_subject_split_masks  # noqa: E402


def main() -> None:
    """Validate split coverage and export a concise audit."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--expected-subjects", type=int, required=True)
    parser.add_argument("--expected-test-subjects", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    split_root = Path(args.split_root)
    first_split_path = split_root / f"{args.data_dir}_seed{args.seeds[0]}.json"
    first_payload = json.loads(first_split_path.read_text(encoding="utf-8"))
    locked_subjects = {
        str(subject)
        for key in ("train_subjects", "val_subjects", "test_subjects", "excluded_subjects")
        for subject in first_payload.get(key, [])
    }

    _, _, subject_ids = load_domain_mmap(Path(args.data_dir))
    unique_subjects = {str(value) for value in np.unique(subject_ids)}
    missing_subjects = sorted(locked_subjects - unique_subjects)
    extra_subjects = sorted(unique_subjects - locked_subjects)
    if len(unique_subjects) != args.expected_subjects or missing_subjects or extra_subjects:
        raise ValueError(
            f"Dataset {args.data_dir} contains {len(unique_subjects)} subjects; "
            f"the locked paper protocol requires {args.expected_subjects}. "
            f"Missing locked subjects: {missing_subjects or 'none'}. "
            f"Unexpected dataset subjects: {extra_subjects or 'none'}."
        )

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        split_path = split_root / f"{args.data_dir}_seed{seed}.json"
        train, val, test = load_subject_split_masks(split_path, subject_ids)
        counts = {
            "train": len(np.unique(subject_ids[train])),
            "val": len(np.unique(subject_ids[val])),
            "test": len(np.unique(subject_ids[test])),
        }
        if counts["test"] != args.expected_test_subjects:
            raise ValueError(
                f"Locked split {split_path} has {counts['test']} test subjects; "
                f"expected {args.expected_test_subjects}."
            )
        rows.append({"seed": seed, **counts, "split_file": str(split_path)})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "data_dir": args.data_dir,
                "dataset_subjects": len(unique_subjects),
                "expected_subjects": args.expected_subjects,
                "expected_test_subjects": args.expected_test_subjects,
                "splits": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Verified {len(rows)} locked splits for {len(unique_subjects)} subjects; saved {output}")


if __name__ == "__main__":
    main()
