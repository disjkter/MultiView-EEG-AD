"""Plot the phase-1 LOSO Monte Carlo summary table as a publication-style figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metric_pair(
    output_dir: Path,
    key: str,
    fallback_summary: str | None = None,
    prefer_fallback: bool = False,
) -> dict[str, tuple[float, float, float, float]]:
    """Load mean/std for F1 and AUC from aggregate results or a single-run fallback."""

    aggregate_path = output_dir / key / "aggregate_summary.json"
    if aggregate_path.exists() and not prefer_fallback:
        payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", payload)
        return {
            "F1": (
                float(metrics["f1"]["mean"]),
                float(metrics["f1"]["std"]),
            ),
            "AUROC": (
                float(metrics["auc"]["mean"]),
                float(metrics["auc"]["std"]),
            ),
        }

    if fallback_summary is None:
        raise FileNotFoundError(f"Missing aggregate summary and no fallback provided: {aggregate_path}")

    summary_path = output_dir / fallback_summary / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", payload)
    return {
        "F1": (float(metrics["f1"]), 0.0),
        "AUROC": (float(metrics["auc"]), 0.0),
    }


def main() -> None:
    """Render a Monte Carlo summary table for the first-layer experiments."""

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["ADSZ", "ADFSU", "ds004504"]

    rows = [
        (
            "SVM",
            {
                "ADSZ": load_metric_pair(output_dir, "mc_loso_adsz_svm"),
                "ADFSU": load_metric_pair(output_dir, "mc_loso_external_svm"),
                "ds004504": load_metric_pair(output_dir, "mc_loso_ds005408_svm", "baseline_svm_loso_ds005408"),
            },
        ),
        (
            "EEGNet",
            {
                "ADSZ": load_metric_pair(output_dir, "mc_loso_adsz_eegnet"),
                "ADFSU": load_metric_pair(output_dir, "mc_loso_external_eegnet"),
                "ds004504": load_metric_pair(output_dir, "mc_loso_ds005408_eegnet", "baseline_eegnet_loso_ds005408"),
            },
        ),
        (
            "ShallowConvNet",
            {
                "ADSZ": load_metric_pair(output_dir, "mc_loso_adsz_shallowconvnet"),
                "ADFSU": load_metric_pair(output_dir, "mc_loso_adfsu_shallowconvnet"),
                "ds004504": load_metric_pair(output_dir, "mc_loso_ds004504_shallowconvnet"),
            },
        ),
        (
            "MTDNet",
            {
                "ADSZ": load_metric_pair(output_dir, "mc_loso_adsz_mtdnet"),
                "ADFSU": load_metric_pair(output_dir, "mc_loso_adfsu_mtdnet"),
                "ds004504": load_metric_pair(output_dir, "mc_loso_ds004504_mtdnet"),
            },
        ),
        (
            "iTransformer",
            {
                "ADSZ": load_metric_pair(output_dir, "mc_loso_adsz_itransformer"),
                "ADFSU": load_metric_pair(output_dir, "mc_loso_adfsu_itransformer"),
                "ds004504": load_metric_pair(output_dir, "mc_loso_ds004504_itransformer"),
            },
        ),
        (
            "TCN",
            {
                "ADSZ": load_metric_pair(output_dir, "mc_loso_adsz_tcn"),
                "ADFSU": load_metric_pair(output_dir, "mc_loso_adfsu_tcn"),
                "ds004504": load_metric_pair(output_dir, "mc_loso_ds004504_tcn"),
            },
        ),
        (
            "TAGE-Net",
            {
                "ADSZ": load_metric_pair(output_dir, "mc_loso_adsz_dualstream", "loso_adsz_lite_auc", prefer_fallback=True),
                "ADFSU": load_metric_pair(output_dir, "mc_loso_external_dualstream"),
                "ds004504": load_metric_pair(output_dir, "mc_loso_ds005408_tage_np_dim16_aug_cw", "mc_loso_ds005408_tage_np_dim16_aug_cw/seed_2026"),
            },
        ),
        (
            "Spectrogram-CNN",
            {
                "ADSZ": load_metric_pair(output_dir, "mc_loso_adsz_spectrogram_cnn"),
                "ADFSU": load_metric_pair(output_dir, "mc_loso_external_spectrogram_cnn"),
                "ds004504": load_metric_pair(output_dir, "mc_loso_ds005408_spectrogram_cnn"),
            },
        ),
    ]

    n_rows = 3 + len(rows)
    n_cols = 1 + len(datasets) * 2

    fig, ax = plt.subplots(figsize=(13.8, 6.0), dpi=220)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.axis("off")

    header_face = "#f3f1ea"
    section_face = "#faf8f1"
    white = "#ffffff"
    line_color = "#3d3d3d"
    best_f1_color = "#c62828"
    best_auc_color = "#1565c0"
    text_color = "#111111"

    best_by_dataset: dict[str, dict[str, float]] = {}
    for dataset_name in datasets:
        best_by_dataset[dataset_name] = {
            "F1": max(values[dataset_name]["F1"][0] for _, values in rows),
            "AUROC": max(values[dataset_name]["AUROC"][0] for _, values in rows),
        }

    def draw_cell(col: float, row: float, width: float, height: float, facecolor: str = white) -> None:
        rect = plt.Rectangle((col, row), width, height, facecolor=facecolor, edgecolor="none")
        ax.add_patch(rect)

    def put_text(
        x: float,
        y: float,
        text: str,
        size: int = 10,
        weight: str = "normal",
        color: str = text_color,
        ha: str = "center",
        style: str = "normal",
    ) -> None:
        ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, ha=ha, va="center", style=style)

    # Background blocks
    for c in range(n_cols):
        draw_cell(c, n_rows - 1, 1, 1, header_face)
        draw_cell(c, n_rows - 2, 1, 1, header_face)
    draw_cell(0, n_rows - 3, n_cols, 0.8, section_face)
    for r in range(len(rows)):
        y = n_rows - 4 - r
        for c in range(n_cols):
            draw_cell(c, y, 1, 1, white)

    # Title
    put_text(n_cols / 2, n_rows + 0.35, "Phase 1: Intra-Dataset LOSO Evaluation (Monte Carlo)", size=14, weight="bold")

    # Headers
    put_text(0.5, n_rows - 0.5, "Methods", size=11, weight="bold")
    for idx, dataset_name in enumerate(datasets):
        start_col = 1 + idx * 2
        put_text(start_col + 1.0, n_rows - 0.38, dataset_name, size=11, weight="bold")
        put_text(start_col + 0.5, n_rows - 1.5, "F1 Score", size=10, weight="bold")
        put_text(start_col + 1.5, n_rows - 1.5, "AUROC", size=10, weight="bold")

    put_text(n_cols / 2, n_rows - 2.6, "Subject-Level Classification", size=10, weight="bold")

    # Data rows
    for row_idx, (method_name, values) in enumerate(rows):
        y_center = n_rows - 3.5 - row_idx
        put_text(0.08, y_center, method_name, size=10, weight="bold" if method_name == "TAGE-Net" else "normal", ha="left")
        for ds_idx, dataset_name in enumerate(datasets):
            f1_mean, f1_std = values[dataset_name]["F1"]
            auc_mean, auc_std = values[dataset_name]["AUROC"]
            f1_color = best_f1_color if abs(f1_mean - best_by_dataset[dataset_name]["F1"]) < 1e-8 else text_color
            auroc_color = best_auc_color if abs(auc_mean - best_by_dataset[dataset_name]["AUROC"]) < 1e-8 else text_color
            f1_weight = "bold" if f1_color == best_f1_color else "normal"
            auroc_weight = "bold" if auroc_color == best_auc_color else "normal"
            start_col = 1 + ds_idx * 2
            put_text(
                start_col + 0.5,
                y_center,
                f"{f1_mean * 100:.2f}$\\pm${f1_std * 100:.2f}",
                size=9,
                weight=f1_weight,
                color=f1_color,
            )
            put_text(
                start_col + 1.5,
                y_center,
                f"{auc_mean * 100:.2f}$\\pm${auc_std * 100:.2f}",
                size=9,
                weight=auroc_weight,
                color=auroc_color,
            )

    # Lines
    for y in [n_rows, n_rows - 1, n_rows - 2, n_rows - 2.8, 0]:
        ax.plot([0, n_cols], [y, y], color=line_color, lw=1.0)
    for r in range(len(rows)):
        y = n_rows - 4 - r
        ax.plot([0, n_cols], [y, y], color="#d0d0d0", lw=0.6)

    # Vertical separators
    ax.plot([1, 1], [0, n_rows], color=line_color, lw=1.0)
    for idx in range(len(datasets) + 1):
        x = 1 + idx * 2
        ax.plot([x, x], [n_rows - 2, n_rows], color=line_color, lw=1.0)
        if idx < len(datasets):
            ax.plot([x, x], [0, n_rows - 2], color="#8b8b8b", lw=0.7)
            ax.plot([x + 2, x + 2], [0, n_rows - 2], color="#8b8b8b", lw=0.7)

    put_text(2.0, -0.35, "Best F1", size=9, weight="bold", color=best_f1_color)
    put_text(3.0, -0.35, "Best AUROC", size=9, weight="bold", color=best_auc_color)

    fig.tight_layout(pad=0.8)
    png_path = output_dir / "phase1_summary_table.png"
    pdf_path = output_dir / "phase1_summary_table.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure to: {png_path}")
    print(f"Saved figure to: {pdf_path}")


if __name__ == "__main__":
    main()
