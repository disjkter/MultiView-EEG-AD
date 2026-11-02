"""Create editable IEEE-style figures for the paper-upgrade analyses."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV as dictionaries."""

    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def configure_plotting() -> None:
    """Apply compact IEEE-compatible plotting defaults."""

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save(figure: object, output: Path) -> None:
    """Save vector PDF/SVG and a 400-dpi review PNG."""

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")


def plot_score_shift(path: Path, output: Path) -> None:
    """Plot the three prespecified score-shift associations."""

    import matplotlib.pyplot as plt

    rows = [row for row in read_rows(path) if row["source_scores_available"].lower() == "true"]
    if not rows:
        raise ValueError("Score-shift plotting requires source-validation predictions")
    panels = (
        ("delta_mean", "source_threshold_ba", "|Logit mean shift|", "Source-threshold BA", True),
        ("delta_sd", "source_threshold_ba", "|Logit scale shift|", "Source-threshold BA", True),
        ("score_shift_magnitude", "cohort_z_gain", "Score-shift magnitude", "Cohort-Z BA gain", False),
    )
    targets = sorted({row["target"] for row in rows})
    markers = ["o", "s", "^"]
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 2.15))
    for axis, (x_key, y_key, x_label, y_label, absolute) in zip(axes, panels):
        for target, marker in zip(targets, markers):
            group = [row for row in rows if row["target"] == target]
            x = np.asarray([float(row[x_key]) for row in group])
            if absolute:
                x = np.abs(x)
            y = np.asarray([float(row[y_key]) for row in group])
            axis.scatter(x, y, s=11, marker=marker, facecolors="none", label=target)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(color="0.9", linewidth=0.5)
    axes[0].legend(frameon=False)
    figure.tight_layout(w_pad=1.4)
    save(figure, output)
    plt.close(figure)


def plot_prevalence(path: Path, output: Path) -> None:
    """Plot ADFSU prevalence sensitivity for reliability and threshold metrics."""

    import matplotlib.pyplot as plt

    rows = read_rows(path)
    variants = sorted({row["variant"] for row in rows})
    metrics = (("balanced_accuracy", "Balanced accuracy"), ("ece", "ECE"), ("brier", "Brier score"), ("nll", "NLL"))
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 4.2), sharex=True)
    for axis, (metric, label) in zip(axes.flat, metrics):
        for variant in variants:
            group = sorted(
                [row for row in rows if row["variant"] == variant],
                key=lambda row: float(row["requested_prevalence"]),
            )
            x = np.asarray([float(row["requested_prevalence"]) for row in group])
            y = np.asarray([float(row[f"{metric}_mean"]) for row in group])
            low = np.asarray([float(row[f"{metric}_ci_low"]) for row in group])
            high = np.asarray([float(row[f"{metric}_ci_high"]) for row in group])
            axis.plot(x, y, marker="o", markersize=3, label=variant)
            axis.fill_between(x, low, high, alpha=0.10)
        axis.set_ylabel(label)
        axis.grid(color="0.9", linewidth=0.5)
    for axis in axes[-1]:
        axis.set_xlabel("Simulated target prevalence")
    axes[0, 0].legend(frameon=False, ncol=min(3, len(variants)))
    figure.tight_layout()
    save(figure, output)
    plt.close(figure)


def plot_calibration_delta(path: Path, output: Path) -> None:
    """Plot source-fitted calibration changes as grayscale-safe point plots."""

    import matplotlib.pyplot as plt

    rows = read_rows(path)
    panels = (("delta_ece_mean", r"$\Delta$ECE"), ("delta_brier_mean", r"$\Delta$Brier"), ("delta_nll_mean", r"$\Delta$NLL"))
    group_order = [
        ("adfsu_confirmation", "ADFSU"),
        ("eegnet_development", "ADSZ"),
        ("eegnet_development", "CAUEEG"),
        ("eegnet_development", "ds004504"),
        ("graph_development", "ADSZ"),
        ("graph_development", "CAUEEG"),
        ("graph_development", "ds004504"),
    ]
    available = {(row["experiment"], row["target"]) for row in rows}
    groups = [group for group in group_order if group in available]
    labels = {
        ("adfsu_confirmation", "ADFSU"): "ADFSU (Graph)",
        ("eegnet_development", "ADSZ"): "ADSZ (EEGNet)",
        ("eegnet_development", "CAUEEG"): "CAUEEG (EEGNet)",
        ("eegnet_development", "ds004504"): "ds004504 (EEGNet)",
        ("graph_development", "ADSZ"): "ADSZ (Graph)",
        ("graph_development", "CAUEEG"): "CAUEEG (Graph)",
        ("graph_development", "ds004504"): "ds004504 (Graph)",
    }
    styles = {
        "temperature": {"marker": "o", "facecolors": "white", "edgecolors": "0.15"},
        "platt": {"marker": "s", "facecolors": "0.25", "edgecolors": "0.15"},
    }
    y = np.arange(len(groups), dtype=np.float64)
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 2.45), sharey=True)
    for axis, (metric, label) in zip(axes, panels):
        for index, calibrator in enumerate(("temperature", "platt")):
            values = []
            for experiment, target in groups:
                match = [
                    row for row in rows
                    if row["experiment"] == experiment and row["target"] == target
                    and row["calibrator"] == calibrator
                ]
                values.append(float(match[0][metric]) if match else float("nan"))
            axis.scatter(
                values,
                y + (index - 0.5) * 0.18,
                s=24,
                linewidths=0.7,
                label=calibrator.capitalize(),
                zorder=3,
                **styles[calibrator],
            )
        axis.axvline(0.0, color="0.25", linewidth=0.8, linestyle="--", zorder=1)
        axis.set_xlabel(label)
        axis.set_yticks(y, [labels[group] for group in groups])
        axis.grid(axis="x", color="0.88", linewidth=0.5, zorder=0)
        axis.tick_params(labelsize=7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False)
    figure.tight_layout(w_pad=0.8)
    save(figure, output)
    plt.close(figure)


def main() -> None:
    """Render selected paper-upgrade figures."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-shift-folds", type=Path)
    parser.add_argument("--prevalence-summary", type=Path)
    parser.add_argument("--calibration-delta-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_plotting()
    if args.score_shift_folds:
        plot_score_shift(args.score_shift_folds, args.output_dir / "score_shift_associations")
    if args.prevalence_summary:
        plot_prevalence(args.prevalence_summary, args.output_dir / "adfsu_prevalence_sensitivity")
    if args.calibration_delta_summary:
        plot_calibration_delta(args.calibration_delta_summary, args.output_dir / "source_calibration_metric_changes")
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
