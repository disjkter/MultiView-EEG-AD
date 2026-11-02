"""Analyze subject-level calibration across cross-cohort LODO experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np


CALIBRATION_METRICS = ("brier", "nll", "ece", "mean_probability")


def parse_result_spec(value: str) -> tuple[str, Path]:
    """Parse a command-line result specification formatted as NAME=PATH."""

    if "=" not in value:
        raise argparse.ArgumentTypeError("Result must use NAME=PATH format")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Both NAME and PATH must be non-empty")
    return name.strip(), Path(raw_path.strip())


def calibration_metrics(
    y_true: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    num_bins: int = 10,
) -> dict[str, float]:
    """Compute binary probabilistic calibration metrics."""

    labels = np.asarray(y_true, dtype=np.float64)
    probabilities = np.asarray(y_prob, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("y_true and y_prob must be one-dimensional")
    if labels.shape != probabilities.shape or labels.size == 0:
        raise ValueError("y_true and y_prob must have the same non-zero length")
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    if not np.all(np.isin(labels, (0.0, 1.0))):
        raise ValueError("y_true must contain binary labels")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("y_prob must lie in [0, 1]")

    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    nll = -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    bins = reliability_bins(labels, probabilities, num_bins)
    ece = sum(row["fraction"] * abs(row["accuracy"] - row["confidence"]) for row in bins)
    return {
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "nll": float(nll),
        "ece": float(ece),
        "mean_probability": float(probabilities.mean()),
    }


def reliability_bins(
    y_true: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    num_bins: int = 10,
) -> list[dict[str, float | int]]:
    """Return non-empty equal-width reliability bins."""

    labels = np.asarray(y_true, dtype=np.float64)
    probabilities = np.asarray(y_prob, dtype=np.float64)
    if labels.shape != probabilities.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.minimum(np.digitize(probabilities, edges[1:-1]), num_bins - 1)
    rows: list[dict[str, float | int]] = []
    for bin_index in range(num_bins):
        mask = indices == bin_index
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin": bin_index,
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "count": count,
                "fraction": float(count / labels.size),
                "confidence": float(probabilities[mask].mean()),
                "accuracy": float(labels[mask].mean()),
            }
        )
    return rows


def collect_runs(
    result_specs: list[tuple[str, Path]],
    num_bins: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Collect calibration metrics and seed-level reliability bins."""

    run_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, int, str]] = set()
    for experiment, root in result_specs:
        paths = sorted(root.rglob("lodo_summary.json"))
        if not paths:
            raise FileNotFoundError(f"No lodo_summary.json files found under {root}")
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            variant = path.parent.name
            seed = int(payload["seed"])
            method = str(payload["method"])
            for fold in payload["folds"]:
                target = str(fold["target"])
                key = (experiment, variant, seed, target)
                if key in seen:
                    raise ValueError(f"Duplicate experiment result: {key}")
                seen.add(key)
                metrics = fold["metrics"]
                y_true = metrics["y_true"]
                y_prob = metrics["y_prob"]
                row: dict[str, object] = {
                    "experiment": experiment,
                    "variant": variant,
                    "method": method,
                    "seed": seed,
                    "target": target,
                    "num_subjects": len(y_true),
                    "prevalence": float(np.mean(y_true)),
                    "auc": float(metrics["auc"]),
                }
                row.update(calibration_metrics(y_true, y_prob, num_bins))
                run_rows.append(row)
                for bin_row in reliability_bins(y_true, y_prob, num_bins):
                    bin_rows.append(
                        {
                            "experiment": experiment,
                            "variant": variant,
                            "method": method,
                            "seed": seed,
                            "target": target,
                            **bin_row,
                        }
                    )
    return run_rows, bin_rows


def summarize_runs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize calibration metrics across matched random seeds."""

    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["experiment"]),
            str(row["variant"]),
            str(row["method"]),
            str(row["target"]),
        )
        groups[key].append(row)
    summaries: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        summary: dict[str, object] = {
            "experiment": key[0],
            "variant": key[1],
            "method": key[2],
            "target": key[3],
            "num_seeds": len(group),
            "num_subjects_per_seed": int(group[0]["num_subjects"]),
        }
        for metric in ("prevalence", "auc", *CALIBRATION_METRICS):
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def summarize_bins(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Average seed-level reliability points without pooling repeated subjects."""

    groups: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["experiment"]),
            str(row["variant"]),
            str(row["target"]),
            int(row["bin"]),
        )
        groups[key].append(row)
    summaries: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        confidence = [float(row["confidence"]) for row in group]
        accuracy = [float(row["accuracy"]) for row in group]
        summaries.append(
            {
                "experiment": key[0],
                "variant": key[1],
                "target": key[2],
                "bin": key[3],
                "num_seed_bins": len(group),
                "confidence_mean": mean(confidence),
                "confidence_std": stdev(confidence) if len(confidence) > 1 else 0.0,
                "accuracy_mean": mean(accuracy),
                "accuracy_std": stdev(accuracy) if len(accuracy) > 1 else 0.0,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows using a stable header."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_reliability(path: Path, rows: list[dict[str, object]]) -> None:
    """Plot compact per-target reliability diagrams for manuscript use."""

    import matplotlib.pyplot as plt

    representative = ("erm", "erm_margin", "dann")
    display_names = {"erm": "ERM", "erm_margin": "ERM+Margin", "dann": "DANN"}
    styles = {
        "erm": {"color": "0.15", "linestyle": "-", "marker": "o"},
        "erm_margin": {"color": "0.4", "linestyle": "--", "marker": "s"},
        "dann": {"color": "0.65", "linestyle": ":", "marker": "^"},
    }
    rows = [row for row in rows if str(row["variant"]) in representative]
    panels = sorted({(str(row["experiment"]), str(row["target"])) for row in rows})
    columns = min(3, len(panels))
    rows_count = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(2.35 * columns, 2.35 * rows_count),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for axis, panel in zip(axes.flat, panels):
        axis.plot([0, 1], [0, 1], color="0.55", linestyle="--", linewidth=0.8)
        present = {str(row["variant"]) for row in rows if (row["experiment"], row["target"]) == panel}
        variants = [variant for variant in representative if variant in present]
        for variant in variants:
            points = [row for row in rows if (row["experiment"], row["target"], row["variant"]) == (*panel, variant)]
            points.sort(key=lambda row: int(row["bin"]))
            axis.plot(
                [float(row["confidence_mean"]) for row in points],
                [float(row["accuracy_mean"]) for row in points],
                label=display_names[variant],
                markersize=2.8,
                linewidth=1.0,
                markerfacecolor="white" if variant != "erm" else "0.15",
                markeredgewidth=0.7,
                **styles[variant],
            )
        panel_title = panel[1] if panel[0] == "development" else f"{panel[0]}: {panel[1]}"
        axis.set_title(panel_title, fontsize=8)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(color="0.9", linewidth=0.5)
        axis.tick_params(labelsize=7)
    for axis in axes.flat[len(panels):]:
        axis.set_visible(False)
    figure.supxlabel("Mean predicted probability", fontsize=8, y=0.01)
    figure.supylabel("Observed positive fraction", fontsize=8, x=0.01)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=min(3, len(labels)),
            frameon=False,
            fontsize=7,
        )
    figure.tight_layout(rect=(0.035, 0.05, 1.0, 0.84 if handles else 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run cross-cohort calibration analysis and write tables and figures."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        action="append",
        type=parse_result_spec,
        required=True,
        metavar="NAME=PATH",
        help="Experiment label and root; may be repeated.",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--expected-seeds", type=int, default=None)
    args = parser.parse_args()

    runs, raw_bins = collect_runs(args.results, args.num_bins)
    summaries = summarize_runs(runs)
    bins = summarize_bins(raw_bins)
    if args.expected_seeds is not None:
        incomplete = [
            (row["experiment"], row["variant"], row["target"], row["num_seeds"])
            for row in summaries
            if int(row["num_seeds"]) != args.expected_seeds
        ]
        if incomplete:
            raise RuntimeError(f"Incomplete seed groups: {incomplete[:10]}")

    prefix = args.output_prefix
    write_csv(prefix.with_name(prefix.name + "_runs.csv"), runs)
    write_csv(prefix.with_name(prefix.name + "_summary.csv"), summaries)
    write_csv(prefix.with_name(prefix.name + "_reliability_bins.csv"), bins)
    payload = {
        "metadata": {
            "results": [{"name": name, "path": str(path)} for name, path in args.results],
            "num_bins": args.num_bins,
            "expected_seeds": args.expected_seeds,
            "subject_level": True,
            "reliability_aggregation": "bin within seed, then average across seeds",
        },
        "summary": summaries,
    }
    prefix.with_suffix(".json").parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_reliability(prefix.with_name(prefix.name + "_reliability"), bins)
    print(f"Analyzed {len(runs)} fold-level runs")
    print(f"Saved calibration outputs to {prefix.parent}")


if __name__ == "__main__":
    main()
