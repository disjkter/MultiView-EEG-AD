"""Verify and aggregate TF-C participant-level results on fixed CAUEEG splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aggregate_jbhi_revision_experiments import METRICS, expected_test_ids, extract_metrics


def main() -> None:
    """Audit test identities and export TF-C mean and sample standard deviation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="outputs/jbhi_revision/tfc_baseline")
    parser.add_argument("--split-root", default="outputs/jbhi_revision/fixed_splits")
    parser.add_argument("--output-prefix", default="outputs/jbhi_revision/tfc_baseline/aggregate")
    parser.add_argument("--dataset", default="CAUEEG_processed_19ch_ad_vs_hc_1s_lead")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2026, 2036)))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for seed in args.seeds:
        summary_path = Path(args.input_root) / f"seed_{seed}" / "summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics, actual_ids = extract_metrics(payload, summary_path)
        expected_ids = expected_test_ids(Path(args.split_root), args.dataset, seed)
        exact = actual_ids == expected_ids
        if not exact:
            raise ValueError(
                f"TF-C test participants differ at seed {seed}: "
                f"missing={sorted(expected_ids-actual_ids)[:5]}, "
                f"unexpected={sorted(actual_ids-expected_ids)[:5]}"
            )
        row: dict[str, object] = {"method": "TF-C", "dataset": args.dataset, "seed": seed}
        row.update({metric: float(metrics[metric]) for metric in METRICS})
        rows.append(row)
        audits.append(
            {
                "seed": seed,
                "expected_test_n": len(expected_ids),
                "actual_test_n": len(actual_ids),
                "exact_test_id_match": exact,
                "source": str(summary_path),
            }
        )

    aggregate: dict[str, object] = {
        "method": "TF-C",
        "dataset": args.dataset,
        "num_seeds": len(rows),
        "seeds": args.seeds,
    }
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows])
        aggregate[f"{metric}_mean"] = float(values.mean())
        aggregate[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    prefix.with_suffix(".json").write_text(
        json.dumps({"runs": rows, "summary": aggregate, "identity_audit": audits}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
