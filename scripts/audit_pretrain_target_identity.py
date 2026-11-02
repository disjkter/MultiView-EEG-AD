"""Audit exact stored participant identifiers across pretraining and target EEG datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_PRETRAIN = (
    "BrainLat19_pretrain_19ch_1s_lead",
    "auditory40hz_processed_19ch_1s_lead",
    "ADSZ_processed_19ch_1s_lead",
)
DEFAULT_TARGETS = (
    "CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
    "ds004504_processed_19ch_1s_lead",
)


def load_ids(dataset_dir: Path) -> set[str]:
    """Return unique stored participant identifiers from one processed dataset."""

    path = dataset_dir / "sub_ids_19ch.npy"
    values = np.load(path, allow_pickle=True)
    return {str(value).strip() for value in values.tolist()}


def main() -> None:
    """Compare exact identifier strings and save JSON and CSV audit records."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain", nargs="+", default=list(DEFAULT_PRETRAIN))
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--output-dir", default="outputs/jbhi_revision/identity_overlap_audit")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pretrain_ids = {name: load_ids(Path(name)) for name in args.pretrain}
    target_ids = {name: load_ids(Path(name)) for name in args.targets}

    rows: list[dict[str, object]] = []
    for pretrain_name, source_ids in pretrain_ids.items():
        for target_name, evaluation_ids in target_ids.items():
            overlap = sorted(source_ids & evaluation_ids)
            rows.append(
                {
                    "pretraining_dataset": pretrain_name,
                    "target_dataset": target_name,
                    "pretraining_subjects": len(source_ids),
                    "target_subjects": len(evaluation_ids),
                    "exact_stored_id_overlap_count": len(overlap),
                    "exact_stored_id_overlaps": overlap,
                }
            )

    caveat = (
        "The audit compares exact stored identifier strings only. The public datasets do not "
        "provide globally harmonized participant identifiers, so real-person overlap across "
        "independently released cohorts cannot be independently excluded."
    )
    payload = {"comparisons": rows, "all_exact_counts_zero": all(not row["exact_stored_id_overlap_count"] for row in rows), "caveat": caveat}
    (output_dir / "identity_overlap_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "identity_overlap_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0] if key != "exact_stored_id_overlaps"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key != "exact_stored_id_overlaps"})
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
