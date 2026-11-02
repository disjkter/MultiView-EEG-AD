"""Aggregate matched-seed feature-fusion ablations and apply retention rules."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MODES = ("pre_only", "source_only", "fixed_average", "concat", "learned_scalar")
METRICS = ("accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity")


def main() -> None:
    """Read recursive summaries, write a table, and report the preset verdict."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    root = Path(args.input_root)
    records: list[dict[str, object]] = []
    for path in sorted(root.glob("seed*/**/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        mode = summary.get("fusion_mode")
        if mode not in MODES:
            continue
        row: dict[str, object] = {"seed": int(summary["seed"]), "mode": mode}
        row.update({metric: float(summary["metrics"][metric]) for metric in METRICS})
        gate = summary.get("gate_statistics") or {}
        row["gate_mean"] = gate.get("mean")
        row["gate_std"] = gate.get("std")
        records.append(row)
    if not records:
        raise FileNotFoundError(f"No stage-fusion summaries found under {root}")

    seeds_by_mode = {mode: {int(row["seed"]) for row in records if row["mode"] == mode} for mode in MODES}
    matched_seeds = sorted(set.intersection(*(seeds_by_mode[mode] for mode in MODES)))
    if not matched_seeds:
        raise ValueError("No seed has all five ablation modes.")
    matched = [row for row in records if int(row["seed"]) in matched_seeds]

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for mode in MODES:
        aggregate[mode] = {}
        mode_rows = [row for row in matched if row["mode"] == mode]
        for metric in METRICS:
            values = np.asarray([row[metric] for row in mode_rows], dtype=np.float64)
            aggregate[mode][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }

    learned = aggregate["learned_scalar"]
    comparators = ("pre_only", "source_only", "fixed_average")
    mean_ba_pass = all(learned["balanced_accuracy"]["mean"] > aggregate[m]["balanced_accuracy"]["mean"] for m in comparators)
    mean_auc_pass = all(learned["auc"]["mean"] > aggregate[m]["auc"]["mean"] for m in comparators)
    seed_wins = 0
    for seed in matched_seeds:
        lookup = {str(row["mode"]): row for row in matched if int(row["seed"]) == seed}
        if lookup["learned_scalar"]["balanced_accuracy"] > max(lookup[m]["balanced_accuracy"] for m in comparators):
            seed_wins += 1
    required_wins = int(np.ceil(len(matched_seeds) / 2))
    verdict = {
        "retain_module": bool(mean_ba_pass and mean_auc_pass and seed_wins >= required_wins),
        "rules": {
            "mean_balanced_accuracy_beats_pre_source_and_average": bool(mean_ba_pass),
            "mean_auc_beats_pre_source_and_average": bool(mean_auc_pass),
            "balanced_accuracy_seed_wins": seed_wins,
            "required_seed_wins": required_wins,
        },
    }

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "mode", *METRICS, "gate_mean", "gate_std"])
        writer.writeheader()
        writer.writerows(sorted(matched, key=lambda row: (int(row["seed"]), MODES.index(str(row["mode"])))))
    json_path = prefix.with_suffix(".json")
    json_path.write_text(
        json.dumps({"matched_seeds": matched_seeds, "aggregate": aggregate, "verdict": verdict}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(verdict, indent=2))
    print(f"Saved {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
