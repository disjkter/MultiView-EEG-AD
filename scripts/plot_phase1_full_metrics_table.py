"""Plot the full phase-1 metrics table as a publication-style figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = ["Acc", "F1", "AUROC"]


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load exported phase-1 CSV rows."""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def parse_mean(cell: str) -> float:
    """Extract the mean value from a `mean +/- std` cell."""

    mean_text = cell.split("+/-")[0].strip()
    return float(mean_text)


def main() -> None:
    """Render the phase-1 table with the main reported metrics only."""

    output_dir = Path("outputs")
    csv_path = output_dir / "phase1_full_metrics_table.csv"
    rows = load_rows(csv_path)

    datasets = []
    methods = []
    for row in rows:
        dataset = row["Dataset"]
        method = row["Method"]
        if dataset not in datasets:
            datasets.append(dataset)
        if method not in methods:
            methods.append(method)

    cell_lookup = {(row["Dataset"], row["Method"]): row for row in rows}

    best_values: dict[tuple[str, str], float] = {}
    for dataset in datasets:
        for metric in METRICS:
            best_values[(dataset, metric)] = max(
                parse_mean(cell_lookup[(dataset, method)][metric]) for method in methods
            )

    n_header_rows = 2
    n_rows = n_header_rows + len(methods)
    n_cols = 1 + len(datasets) * len(METRICS)

    fig, ax = plt.subplots(figsize=(20, 6.2), dpi=220)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.axis("off")

    header_face = "#f3f1ea"
    white = "#ffffff"
    line_color = "#3d3d3d"
    grid_color = "#cfcfcf"
    text_color = "#111111"
    best_color = "#c62828"

    def draw_cell(col: float, row: float, width: float, height: float, facecolor: str) -> None:
        rect = plt.Rectangle((col, row), width, height, facecolor=facecolor, edgecolor="none")
        ax.add_patch(rect)

    def put_text(
        x: float,
        y: float,
        text: str,
        size: int = 9,
        weight: str = "normal",
        color: str = text_color,
        ha: str = "center",
    ) -> None:
        ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, ha=ha, va="center")

    for c in range(n_cols):
        draw_cell(c, n_rows - 1, 1, 1, header_face)
        draw_cell(c, n_rows - 2, 1, 1, header_face)
    for r in range(len(methods)):
        y = n_rows - 3 - r
        for c in range(n_cols):
            draw_cell(c, y, 1, 1, white)

    put_text(0.5, n_rows - 0.5, "Methods", size=10, weight="bold")

    for ds_idx, dataset in enumerate(datasets):
        start_col = 1 + ds_idx * len(METRICS)
        put_text(start_col + len(METRICS) / 2, n_rows - 0.38, dataset, size=10, weight="bold")
        for metric_idx, metric in enumerate(METRICS):
            put_text(start_col + metric_idx + 0.5, n_rows - 1.5, metric, size=8, weight="bold")

    for row_idx, method in enumerate(methods):
        y_center = n_rows - 2.5 - row_idx
        method_weight = "bold" if method == "TAGE-Net" else "normal"
        put_text(0.08, y_center, method, size=9, weight=method_weight, ha="left")
        for ds_idx, dataset in enumerate(datasets):
            start_col = 1 + ds_idx * len(METRICS)
            row = cell_lookup[(dataset, method)]
            for metric_idx, metric in enumerate(METRICS):
                cell = row[metric]
                mean_value = parse_mean(cell)
                is_best = abs(mean_value - best_values[(dataset, metric)]) < 1e-8
                put_text(
                    start_col + metric_idx + 0.5,
                    y_center,
                    cell,
                    size=7.6,
                    weight="bold" if is_best else "normal",
                    color=best_color if is_best else text_color,
                )

    for y in [n_rows, n_rows - 1, n_rows - 2, 0]:
        ax.plot([0, n_cols], [y, y], color=line_color, lw=1.0)
    for r in range(len(methods)):
        y = n_rows - 3 - r
        ax.plot([0, n_cols], [y, y], color=grid_color, lw=0.6)

    ax.plot([1, 1], [0, n_rows], color=line_color, lw=1.0)
    for ds_idx in range(len(datasets) + 1):
        x = 1 + ds_idx * len(METRICS)
        ax.plot([x, x], [0, n_rows], color=line_color, lw=1.0)
        if ds_idx < len(datasets):
            for metric_idx in range(1, len(METRICS)):
                inner_x = x + metric_idx
                ax.plot([inner_x, inner_x], [0, n_rows - 1], color=grid_color, lw=0.5)

    put_text(2.2, -0.32, "Best value in each dataset/metric column", size=8, weight="bold", color=best_color)

    fig.suptitle("Phase 1: Intra-Dataset LOSO Evaluation (Main Metrics)", fontsize=16, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.02, right=0.995, top=0.88, bottom=0.10)
    png_path = output_dir / "phase1_full_metrics_table.png"
    pdf_path = output_dir / "phase1_full_metrics_table.pdf"
    single_png_path = output_dir / "phase1_full_metrics_table_singlepage.png"
    single_pdf_path = output_dir / "phase1_full_metrics_table_singlepage.pdf"
    for path in [png_path, pdf_path, single_png_path, single_pdf_path]:
        fig.savefig(path)
    print(f"Saved figure to: {png_path}")
    print(f"Saved figure to: {pdf_path}")
    print(f"Saved figure to: {single_png_path}")
    print(f"Saved figure to: {single_pdf_path}")


if __name__ == "__main__":
    main()
