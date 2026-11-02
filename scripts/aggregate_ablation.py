"""Aggregate ablation results across variants, datasets, and seeds.

Usage::

    E:\\anaconda3\\envs\\BioFoundation\\python.exe scripts/aggregate_ablation.py \
        --results-dir outputs/ablation \
        --output outputs/ablation_master_table.json

Also generates:
- outputs/ablation_master_table.pdf  (publication-ready table)
- outputs/ablation_bar_chart.pdf     (bar chart comparison)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


# ---------------------------------------------------------------------------
# Variant ordering and labels
# ---------------------------------------------------------------------------

VARIANT_ORDER = [
    "full",
    "no_spatial",
    "no_mscnn",
    "no_multiscale",
    "no_se",
    "attn_fusion",
    "gate_fusion",
    "cls_1layer",
    "cls_3layer",
    "neuro_prior",
]

VARIANT_CATEGORIES = {
    "full": "Baseline",
    "no_spatial": "Branch Ablation",
    "no_mscnn": "Branch Ablation",
    "no_multiscale": "Component Ablation",
    "no_se": "Component Ablation",
    "attn_fusion": "Fusion Strategy",
    "gate_fusion": "Fusion Strategy",
    "cls_1layer": "Classifier Depth",
    "cls_3layer": "Classifier Depth",
    "neuro_prior": "Neuro Prior",
}

# Datasets should appear in order of clinical relevance
DATASET_ORDER = ["ADSZ_processed_19ch_1s_lead", "ADFSU_processed_19ch_1s_lead", "ds004504_processed_19ch_1s_lead"]
DATASET_SHORT = {
    "ADSZ_processed_19ch_1s_lead": "ADSZ",
    "ADFSU_processed_19ch_1s_lead": "ADFSU",
    "ds004504_processed_19ch_1s_lead": "ds004504",
}

METRICS = ["accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity"]
METRIC_LABELS = {
    "accuracy": "Acc",
    "balanced_accuracy": "Bal Acc",
    "f1": "F1",
    "auc": "AUC",
    "sensitivity": "Sens",
    "specificity": "Spec",
}


def collect_results(results_dir: str) -> dict:
    """Walk the results directory and aggregate all summary.json files."""

    root = Path(results_dir)
    aggregated: dict[str, dict] = {}

    for summary_path in sorted(root.rglob("summary.json")):
        rel = summary_path.relative_to(root)
        parts = rel.parts
        # Expected: dataset/variant_sSEED/summary.json
        if len(parts) < 3:
            continue

        dataset = parts[0]
        run_tag = parts[1]  # e.g. "full_s2026"
        try:
            variant, seed_str = run_tag.rsplit("_s", 1)
            seed = int(seed_str)
        except (ValueError, IndexError):
            print(f"  [WARN] Cannot parse run_tag: {run_tag}, skipping")
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = data.get("metrics", {})
        num_subjects = data.get("num_subjects", 0)

        key = f"{dataset}::{variant}"
        if key not in aggregated:
            aggregated[key] = {
                "dataset": dataset,
                "variant": variant,
                "label": data.get("label", variant),
                "num_subjects": num_subjects,
                "seeds": {},
            }

        aggregated[key]["seeds"][seed] = {m: metrics.get(m) for m in METRICS}

    return aggregated


def build_master_table(aggregated: dict) -> dict:
    """Compute mean ± std across seeds for each variant × dataset."""

    master: dict = {"datasets": [], "variants": [], "rows": []}

    for dataset in DATASET_ORDER:
        if not any(k.startswith(f"{dataset}::") for k in aggregated):
            continue
        master["datasets"].append(dataset)

    master["variants"] = [v for v in VARIANT_ORDER if any(k.endswith(f"::{v}") for k in aggregated)]

    for dataset in master["datasets"]:
        for variant in master["variants"]:
            key = f"{dataset}::{variant}"
            entry = aggregated.get(key)
            if entry is None:
                continue

            seeds = entry["seeds"]
            row = {
                "dataset": dataset,
                "dataset_short": DATASET_SHORT.get(dataset, dataset),
                "variant": variant,
                "label": entry["label"],
                "category": VARIANT_CATEGORIES.get(variant, ""),
                "num_seeds": len(seeds),
                "num_subjects": entry["num_subjects"],
            }

            for metric in METRICS:
                vals = []
                for s, m in seeds.items():
                    v = m.get(metric)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        vals.append(float(v))
                if len(vals) >= 1:
                    row[metric] = np.mean(vals)
                    row[f"{metric}_std"] = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                else:
                    row[metric] = float("nan")
                    row[f"{metric}_std"] = float("nan")

            master["rows"].append(row)

    return master


def print_table(master: dict) -> None:
    """Print a formatted ASCII table."""

    header = f"{'Dataset':>12s} | {'Variant':>18s} | " + " | ".join(f"{METRIC_LABELS[m]:>7s}" for m in METRICS)
    sep = "-" * len(header)

    print("\nAblation Master Table")
    print(sep)
    print(header)
    print(sep)

    for row in master["rows"]:
        ds_short = DATASET_SHORT.get(row["dataset"], row["dataset"])
        metrics_str = " | ".join(f"{row.get(m, 0):7.4f}" for m in METRICS)
        print(f"{ds_short:>12s} | {row['label']:>18s} | {metrics_str}")

    # Per-variant averages across datasets
    print(f"\n{sep}")
    print(f"{'Variant':>18s} | " + " | ".join(f"{METRIC_LABELS[m]:>7s}" for m in METRICS) + " | Datasets")
    print(sep)

    for variant in master["variants"]:
        variant_rows = [r for r in master["rows"] if r["variant"] == variant]
        if not variant_rows:
            continue
        avg = {}
        for m in METRICS:
            vals = [r[m] for r in variant_rows if not np.isnan(r[m])]
            avg[m] = np.mean(vals) if vals else float("nan")
        n_ds = len(variant_rows)
        metrics_str = " | ".join(f"{avg[m]:7.4f}" for m in METRICS)
        print(f"{variant_rows[0]['label']:>18s} | {metrics_str} | {n_ds}")


def make_bar_chart(master: dict, output_pdf: str) -> None:
    """Generate a grouped bar chart showing Bal Acc across variants and datasets."""

    rows = master["rows"]
    datasets_short = [DATASET_SHORT.get(r["dataset"], r["dataset"]) for r in rows]
    unique_ds = list(dict.fromkeys(datasets_short))
    variants_labels = [r["label"] for r in rows if r["dataset"] == rows[0]["dataset"]]

    n_variants = len(variants_labels)
    n_datasets = len(unique_ds)
    x = np.arange(n_datasets)
    width = 0.8 / n_variants

    # Color map: highlight baseline in red, others in blue shades
    colors = plt.cm.tab10(np.linspace(0, 1, n_variants))

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, variant_label in enumerate(variants_labels):
        bal_accs = []
        for ds in unique_ds:
            ds_full = [d for d in DATASET_ORDER if DATASET_SHORT[d] == ds][0]
            match = [r for r in rows if r["dataset"] == ds_full and r["label"] == variant_label]
            bal_accs.append(match[0]["balanced_accuracy"] if match else 0)

        offset = (i - n_variants / 2 + 0.5) * width
        bars = ax.bar(x + offset, bal_accs, width, label=variant_label,
                       color=colors[i], edgecolor="white", linewidth=0.5)
        # Annotate baseline with values
        if variant_label == "Full Dual-Stream":
            for j, (bar, val) in enumerate(zip(bars, bal_accs)):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_ylabel("Balanced Accuracy", fontsize=12)
    ax.set_title("Ablation Study: Balanced Accuracy Across Variants and Datasets", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(unique_ds, fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, ncol=1, framealpha=0.9)
    ax.set_ylim(0.3, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar chart saved to: {output_pdf}")


def make_delta_table_pdf(master: dict, output_pdf: str) -> None:
    """Generate a publication-ready ablation table as PDF, with Δ from baseline."""

    rows = master["rows"]
    # Find baseline rows
    baseline = {r["dataset"]: r for r in rows if r["variant"] == "full"}
    if not baseline:
        print("  [WARN] No 'full' baseline found, skipping delta table.")
        return

    datasets_short = [DATASET_SHORT.get(ds, ds) for ds in DATASET_ORDER
                      if any(r["dataset"] == ds for r in rows)]
    variants_display = [r for r in rows if r["dataset"] == rows[0]["dataset"]]

    # Build table data
    col_labels = ["Variant"]
    for ds in datasets_short:
        col_labels.extend([f"{ds}", f"Δ"])
    col_labels.append("Avg Δ")  # Average delta across datasets

    table_data = []
    for vrow in variants_display:
        variant = vrow["variant"]
        label = vrow["label"]
        is_baseline = variant == "full"
        style = "\\textbf{" if is_baseline else ""
        style_end = "}" if is_baseline else ""

        row_data = [f"{style}{label}{style_end}"]
        deltas = []

        for ds_full in DATASET_ORDER:
            if ds_full not in baseline:
                continue
            match = [r for r in rows if r["dataset"] == ds_full and r["variant"] == variant]
            if not match:
                row_data.extend(["--", "--"])
                continue

            bal = match[0]["balanced_accuracy"]
            base_bal = baseline[ds_full]["balanced_accuracy"]
            delta = bal - base_bal

            if is_baseline:
                row_data.extend([f"${bal:.4f}$", "--"])
            else:
                sign = "+" if delta > 0 else ""
                row_data.extend([f"${bal:.4f}$", f"${sign}{delta:.4f}$"])
                deltas.append(delta)

        avg_delta = np.mean(deltas) if deltas else 0.0
        if is_baseline:
            row_data.append("--")
        else:
            sign = "+" if avg_delta > 0 else ""
            row_data.append(f"${sign}{avg_delta:.4f}$")

        table_data.append(row_data)

    # Create figure
    n_cols = len(col_labels)
    fig, ax = plt.subplots(figsize=(min(n_cols * 1.8, 16), len(table_data) * 0.45 + 1.5))
    ax.axis("off")
    ax.set_title("Ablation Study: Balanced Accuracy vs. Baseline (Full Dual-Stream)",
                 fontsize=13, fontweight="bold", pad=12)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.3)

    # Style
    header_color = "#2C3E50"
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#BDC3C7")
        cell.set_linewidth(0.4)
        if row_idx == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
        else:
            is_baseline_row = (row_idx - 1 == 0)
            if is_baseline_row:
                cell.set_facecolor("#FDEBD0")
                cell.set_text_props(weight="bold")
            elif "Δ" in col_labels[col_idx] if col_idx < len(col_labels) else False:
                # Delta columns: color positive/negative
                text = cell.get_text().get_text()
                if text and text != "--":
                    try:
                        val = float(text.replace("$", "").replace("+", ""))
                        if val > 0:
                            cell.set_facecolor("#D5F5E3")
                        elif val < 0:
                            cell.set_facecolor("#FADBD8")
                    except ValueError:
                        pass

    footnote = f"Δ = variant mean − baseline mean across seeds. Positive Δ (green) = improvement over baseline."
    fig.text(0.5, 0.02, footnote, ha="center", fontsize=7.5, color="#7F8C8D")

    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"  Delta table saved to: {output_pdf}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate ablation experiment results")
    parser.add_argument("--results-dir", default="outputs/ablation")
    parser.add_argument("--output", default="outputs/ablation_master_table.json")
    args = parser.parse_args()

    print("Collecting results...")
    aggregated = collect_results(args.results_dir)
    print(f"  Found {len(aggregated)} variant×dataset combinations")

    print("Building master table...")
    master = build_master_table(aggregated)
    print(f"  {len(master['rows'])} rows, {len(master['datasets'])} datasets, {len(master['variants'])} variants")

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(master, f, ensure_ascii=False, indent=2)
    print(f"  Master table saved to: {out_path}")

    # Print
    print_table(master)

    # Generate PDFs
    out_dir = out_path.parent
    bar_path = out_dir / "ablation_bar_chart.pdf"
    delta_path = out_dir / "ablation_delta_table.pdf"
    make_bar_chart(master, str(bar_path))
    make_delta_table_pdf(master, str(delta_path))

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
