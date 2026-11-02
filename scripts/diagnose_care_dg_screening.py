"""Diagnose all completed CQ-CARE-DG screening variants against one baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_care_dg_screening import apply_gate, collect_variant  # noqa: E402

DEFAULT_VARIANTS = (
    "care_dg",
    "care_no_score",
    "care_no_calibration",
    "care_no_entropy",
    "pooled_mmce",
    "multidomain_calibration",
)


def main() -> None:
    """Apply the frozen gate to each available component variant."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--care-dir", type=Path, required=True)
    parser.add_argument("--baseline-variant", default="erm_margin")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--max-discrimination-drop", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = collect_variant(args.baseline_dir, args.baseline_variant)
    results = []
    for variant in args.variants:
        try:
            proposed = collect_variant(args.care_dir, variant)
            result = apply_gate(
                proposed, baseline, args.max_discrimination_drop
            )
            result["variant"] = variant
            result["status"] = "complete"
        except (FileNotFoundError, ValueError) as error:
            result = {
                "variant": variant,
                "status": "unavailable",
                "error": str(error),
            }
        results.append(result)

    complete = [row for row in results if row["status"] == "complete"]
    ranked = sorted(
        complete,
        key=lambda row: (
            float(row["mean_proposed_minus_baseline"]["brier"]),
            float(row["mean_proposed_minus_baseline"]["nll"]),
            -float(row["mean_proposed_minus_baseline"]["auc"]),
        ),
    )
    payload = {
        "baseline_variant": args.baseline_variant,
        "ranking_rule": "Brier difference, then NLL difference, then descending AUROC difference",
        "results": results,
        "ranked_complete_variants": [row["variant"] for row in ranked],
        "any_pass": any(row["decision"] == "PASS" for row in complete),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for row in ranked:
        difference = row["mean_proposed_minus_baseline"]
        print(
            f"{row['variant']}: {row['decision']} "
            f"BA={difference['balanced_accuracy']:+.4f} "
            f"AUC={difference['auc']:+.4f} "
            f"ECE={difference['ece']:+.4f} "
            f"Brier={difference['brier']:+.4f} "
            f"NLL={difference['nll']:+.4f}"
        )
    unavailable = [row for row in results if row["status"] == "unavailable"]
    for row in unavailable:
        print(f"{row['variant']}: UNAVAILABLE ({row['error']})")


if __name__ == "__main__":
    main()
