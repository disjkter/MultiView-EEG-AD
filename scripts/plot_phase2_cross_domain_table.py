"""Plot the phase-2 cross-domain table as a publication-style figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = ["Acc", "F1", "AUROC"]


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load exported phase-2 CSV rows."""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def parse_mean(cell: str) -> float:
    """Extract the mean from a `mean +/- std` cell."""

    return float(cell.split("+/-")[0].strip())


def main() -> None:
    """Render the phase-2 cross-domain table."""

    output_dir = Path("outputs")
    rows = load_rows(output_dir / "phase2_cross_domain_table.csv")

    transfers: list[str] = []
    methods: list[str] = []
    for row in rows:
        if row["Transfer"] not in transfers:
            transfers.append(row["Transfer"])
        if row["Method"] not in methods:
            methods.append(row["Method"])

    cell_lookup = {(row["Transfer"], row["Method"]): row for row in rows}
    best_values: dict[tuple[str, str], float] = {}
    for transfer in transfers:
        for metric in METRICS:
            best_values[(transfer, metric)] = max(parse_mean(cell_lookup[(transfer, method)][metric]) for method in methods)

    n_header_rows = 2
    n_rows = n_header_rows + len(methods)
    n_cols = 1 + len(transfers) * len(METRICS)

    fig, ax = plt.subplots(figsize=(28, 5.6), dpi=220)
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
        ax.add_patch(plt.Rectangle((col, row), width, height, facecolor=facecolor, edgecolor="none"))

    def put_text(x: float, y: float, text: str, size: int = 9, weight: str = "normal", color: str = text_color, ha: str = "center") -> None:
        ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, ha=ha, va="center")

    for c in range(n_cols):
        draw_cell(c, n_rows - 1, 1, 1, header_face)
        draw_cell(c, n_rows - 2, 1, 1, header_face)
    for r in range(len(methods)):
        y = n_rows - 3 - r
        for c in range(n_cols):
            draw_cell(c, y, 1, 1, white)

    put_text(0.5, n_rows - 0.5, "Methods", size=10, weight="bold")
    for idx, transfer in enumerate(transfers):
        start_col = 1 + idx * len(METRICS)
        put_text(start_col + len(METRICS) / 2, n_rows - 0.38, transfer, size=9, weight="bold")
        for metric_idx, metric in enumerate(METRICS):
            put_text(start_col + metric_idx + 0.5, n_rows - 1.5, metric, size=8, weight="bold")

    for row_idx, method in enumerate(methods):
        y_center = n_rows - 2.5 - row_idx
        put_text(0.08, y_center, method, size=9, weight="bold" if method == "TAGE-Net" else "normal", ha="left")
        for idx, transfer in enumerate(transfers):
            start_col = 1 + idx * len(METRICS)
            row = cell_lookup[(transfer, method)]
            for metric_idx, metric in enumerate(METRICS):
                cell = row[metric]
                mean_value = parse_mean(cell)
                is_best = abs(mean_value - best_values[(transfer, metric)]) < 1e-8
                put_text(
                    start_col + metric_idx + 0.5,
                    y_center,
                    cell,
                    size=7.2,
                    weight="bold" if is_best else "normal",
                    color=best_color if is_best else text_color,
                )

    for y in [n_rows, n_rows - 1, n_rows - 2, 0]:
        ax.plot([0, n_cols], [y, y], color=line_color, lw=1.0)
    for r in range(len(methods)):
        ax.plot([0, n_cols], [n_rows - 3 - r, n_rows - 3 - r], color=grid_color, lw=0.6)

    ax.plot([1, 1], [0, n_rows], color=line_color, lw=1.0)
    for idx in range(len(transfers) + 1):
        x = 1 + idx * len(METRICS)
        ax.plot([x, x], [0, n_rows], color=line_color, lw=1.0)
        if idx < len(transfers):
            for metric_idx in range(1, len(METRICS)):
                inner_x = x + metric_idx
                ax.plot([inner_x, inner_x], [0, n_rows - 1], color=grid_color, lw=0.5)

    put_text(2.4, -0.32, "Best value in each transfer/metric column", size=8, weight="bold", color=best_color)

    fig.suptitle("Phase 2: Single-Source Cross-Domain Transfer (Main Metrics)", fontsize=16, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.015, right=0.995, top=0.88, bottom=0.10)
    for path in [
        output_dir / "phase2_cross_domain_table.png",
        output_dir / "phase2_cross_domain_table.pdf",
        output_dir / "phase2_cross_domain_table_singlepage.png",
        output_dir / "phase2_cross_domain_table_singlepage.pdf",
    ]:
        fig.savefig(path)
    print("Saved phase-2 figures.")


if __name__ == "__main__":
    main()
