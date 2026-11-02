"""Compare ERM cross-cohort reliability across matched EEG backbones."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_paired_crossed_bootstrap import (
    METRICS,
    bootstrap_differences,
    compute_metrics,
    participant_keys,
)


def parse_result_specs(values: list[str]) -> dict[str, Path]:
    """Parse unique ``label=directory`` result specifications."""

    specs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected label=directory, received {value!r}")
        label, raw_path = value.split("=", 1)
        if not label or label in specs:
            raise ValueError(f"Invalid or duplicate label: {label!r}")
        specs[label] = Path(raw_path)
    return specs


def load_runs(
    specs: dict[str, Path],
) -> dict[str, dict[int, dict[str, dict[str, object]]]]:
    """Load one ERM summary per backbone, seed, and target."""

    runs: dict[str, dict[int, dict[str, dict[str, object]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for label, directory in specs.items():
        paths = sorted(directory.glob("seed*/erm/lodo_summary.json"))
        if not paths:
            paths = sorted(directory.glob("seed*/**/lodo_summary.json"))
        if not paths:
            raise FileNotFoundError(f"No LODO summaries for {label} under {directory}")
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if str(payload["method"]) != "erm":
                continue
            seed = int(payload["seed"])
            for fold in payload["folds"]:
                if any(
                    bool(fold.get(field, False))
                    for field in (
                        "target_data_used_during_training",
                        "target_data_used_for_model_selection",
                        "target_data_used_for_threshold_selection",
                    )
                ):
                    raise ValueError(f"Target-use protocol violation in {path}")
                target = str(fold["target"])
                metrics = fold["metrics"]
                labels = np.asarray(metrics["y_true"], dtype=np.int64)
                subjects, _ = participant_keys(metrics, target, labels.size)
                runs[label][seed][target] = {
                    "subjects": subjects,
                    "labels": labels,
                    "probability": np.asarray(metrics["y_prob"], dtype=np.float64),
                    "threshold": float(fold["decision_threshold"]),
                }

    reference_label = next(iter(specs))
    reference = runs[reference_label]
    if len(reference) != 10:
        raise ValueError(f"{reference_label} has {len(reference)} seeds; expected 10")
    for label in specs:
        if set(runs[label]) != set(reference):
            raise ValueError(f"Seed mismatch for {label}")
        for seed in reference:
            if set(runs[label][seed]) != set(reference[seed]):
                raise ValueError(f"Target mismatch for {label}, seed {seed}")
            for target in reference[seed]:
                expected = reference[seed][target]
                current = runs[label][seed][target]
                if not np.array_equal(expected["labels"], current["labels"]):
                    raise ValueError(
                        f"Participant label/order mismatch for {label}, seed {seed}, target {target}"
                    )
                canonical = tuple(
                    f"{target}::position_{index:04d}"
                    for index in range(len(current["labels"]))
                )
                expected["subjects"] = canonical
                current["subjects"] = canonical
    return runs


def estimate_rows(
    runs: dict[str, dict[int, dict[str, dict[str, object]]]]
) -> list[dict[str, object]]:
    """Return per-target and macro mean/SD rows over matched seeds."""

    rows: list[dict[str, object]] = []
    for label, seed_runs in sorted(runs.items()):
        targets = sorted(next(iter(seed_runs.values())))
        for target in (*targets, "macro"):
            by_seed: dict[str, list[float]] = defaultdict(list)
            for seed in sorted(seed_runs):
                target_names = targets if target == "macro" else [target]
                seed_values: dict[str, list[float]] = defaultdict(list)
                for target_name in target_names:
                    run = seed_runs[seed][target_name]
                    metrics = compute_metrics(
                        run["labels"], run["probability"], run["threshold"]
                    )
                    for metric, value in metrics.items():
                        seed_values[metric].append(value)
                for metric in METRICS:
                    by_seed[metric].append(float(np.mean(seed_values[metric])))
            row: dict[str, object] = {"backbone": label, "target": target}
            for metric in METRICS:
                values = np.asarray(by_seed[metric], dtype=np.float64)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1))
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write uniform dictionaries to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run descriptive and paired participant-preserving backbone analyses."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=260818)
    args = parser.parse_args()

    specs = parse_result_specs(args.result)
    runs = load_runs(specs)
    estimates = estimate_rows(runs)
    labels = list(specs)
    comparisons = tuple(
        (labels[index], labels[baseline])
        for index in range(len(labels))
        for baseline in range(index)
    )
    samples = bootstrap_differences(
        runs, comparisons, args.num_bootstrap, args.bootstrap_seed
    )
    macro = {
        (str(row["backbone"]), metric): float(row[f"{metric}_mean"])
        for row in estimates
        if row["target"] == "macro"
        for metric in METRICS
    }
    comparison_rows = []
    for reference, baseline in comparisons:
        for metric in METRICS:
            distribution = samples[(reference, baseline, metric)]
            lower, upper = np.percentile(distribution, [2.5, 97.5])
            comparison_rows.append(
                {
                    "reference": reference,
                    "baseline": baseline,
                    "metric": metric,
                    "reference_estimate": macro[(reference, metric)],
                    "baseline_estimate": macro[(baseline, metric)],
                    "mean_difference": macro[(reference, metric)] - macro[(baseline, metric)],
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "num_bootstrap": args.num_bootstrap,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "erm_backbone_estimates.csv", estimates)
    write_csv(args.output_dir / "erm_backbone_paired_bootstrap.csv", comparison_rows)
    payload = {
        "metadata": {
            "results": {label: str(path) for label, path in specs.items()},
            "num_seeds": 10,
            "num_bootstrap": args.num_bootstrap,
            "sampling": "shared seed indices and shared participant indices within target and class",
        },
        "estimates": estimates,
        "comparisons": comparison_rows,
    }
    (args.output_dir / "erm_backbone_analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved ERM backbone analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
