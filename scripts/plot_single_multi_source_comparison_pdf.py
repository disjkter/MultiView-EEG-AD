"""Render a polished PDF comparing three initialization strategies on ds004504."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


DS004504_ROWS = [
    ["BrainLat19 single-source", "0.8056 +/- 0.1039", "0.7952 +/- 0.1247", "0.8434 +/- 0.0713", "0.8667 +/- 0.0356", "0.8571 +/- 0.0000", "0.7333 +/- 0.2494"],
    ["auditory40hz single-source", "0.7179 +/- 0.0725", "0.7222 +/- 0.0647", "0.7049 +/- 0.1134", "0.8254 +/- 0.0112", "0.6667 +/- 0.1782", "0.7778 +/- 0.0786"],
    ["Multi-source pretraining", "0.7949 +/- 0.0363", "0.8016 +/- 0.0312", "0.7845 +/- 0.0541", "0.8492 +/- 0.0224", "0.7143 +/- 0.1166", "0.8889 +/- 0.0786"],
]

def style_table(table, rows, highlight_map: dict[tuple[int, int], str], row_colors: list[str]) -> None:
    """Apply publication-style formatting to one matplotlib table."""

    header_color = "#263238"
    highlight_fill = "#E8F1FF"
    edge_color = "#B0BEC5"

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.55)
        if row_idx == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(0.09)
            continue

        cell.set_facecolor(row_colors[row_idx - 1])
        if col_idx == 0:
            cell.set_text_props(ha="left")

        if (row_idx - 1, col_idx) in highlight_map:
            cell.set_facecolor(highlight_fill)
            cell.set_text_props(weight="bold")


def add_panel(ax, title: str, rows: list[list[str]], highlight_map: dict[tuple[int, int], str]) -> None:
    """Add one dataset comparison panel."""

    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    columns = ["Initialization", "Acc", "BalAcc", "F1", "AUC", "Sens", "Spec"]
    row_colors = ["#F7FBFF", "#FFFDF3", "#F8FFF8"]
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.34, 0.11, 0.12, 0.11, 0.11, 0.11, 0.11],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1, 1.7)
    style_table(table, rows, highlight_map, row_colors)


def main() -> None:
    """Create the PDF table."""

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ds004504_single_vs_multi_source_initialization_comparison.pdf"

    fig, ax = plt.subplots(figsize=(16.5, 8.8))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Initialization Comparison on ds004504",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    ds004504_highlights = {
        (0, 1): "best_acc",
        (2, 2): "best_balacc",
        (0, 3): "best_f1",
        (0, 4): "best_auc",
        (0, 5): "best_sens",
        (2, 6): "best_spec",
    }
    add_panel(ax, "Target Dataset: ds004504", DS004504_ROWS, ds004504_highlights)

    note = (
        "Values are mean +/- std across 3 seeds. Blue shading marks the strongest metric in each column. "
        "BrainLat19 gives the strongest Acc/F1/AUC, while multi-source pretraining is strongest on BalAcc and Specificity."
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=9, color="#455A64")

    plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.94])
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
