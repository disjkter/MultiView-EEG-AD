"""Render the final initialization ablation summary tables as a PDF."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


DS004504_ROWS = [
    ["Random init", "0.7692 +/- 0.0628", "0.7619 +/- 0.0680", "0.8024 +/- 0.0438", "0.8095 +/- 0.0514", "0.8571 +/- 0.0000", "0.6667 +/- 0.1361"],
    ["BrainLat19 single-source", "0.8056 +/- 0.1039", "0.7952 +/- 0.1247", "0.8434 +/- 0.0713", "0.8667 +/- 0.0356", "0.8571 +/- 0.0000", "0.7333 +/- 0.2494"],
    ["auditory40hz single-source", "0.7179 +/- 0.0725", "0.7222 +/- 0.0647", "0.7049 +/- 0.1134", "0.8254 +/- 0.0112", "0.6667 +/- 0.1782", "0.7778 +/- 0.0786"],
    ["Multi-source pretraining", "0.7949 +/- 0.0363", "0.8016 +/- 0.0312", "0.7845 +/- 0.0541", "0.8492 +/- 0.0224", "0.7143 +/- 0.1166", "0.8889 +/- 0.0786"],
    ["CAUEEG source-task init", "0.7778 +/- 0.1416", "0.7524 +/- 0.1831", "0.8390 +/- 0.0771", "0.9429 +/- 0.0467", "0.9048 +/- 0.0673", "0.6000 +/- 0.4320"],
]

CAUEEG_ROWS = [
    ["Random init", "0.8744 +/- 0.0347", "0.8605 +/- 0.0398", "0.8128 +/- 0.0524", "0.9249 +/- 0.0319", "0.8188 +/- 0.0571", "0.9022 +/- 0.0266"],
    ["BrainLat19 single-source", "0.8816 +/- 0.0326", "0.8659 +/- 0.0377", "0.8216 +/- 0.0498", "0.9336 +/- 0.0202", "0.8188 +/- 0.0542", "0.9130 +/- 0.0235"],
    ["auditory40hz single-source", "0.8502 +/- 0.0304", "0.8533 +/- 0.0277", "0.7935 +/- 0.0343", "0.9309 +/- 0.0203", "0.8623 +/- 0.1070", "0.8442 +/- 0.0825"],
    ["Multi-source pretraining", "0.8841 +/- 0.0258", "0.8786 +/- 0.0271", "0.8325 +/- 0.0347", "0.9364 +/- 0.0206", "0.8623 +/- 0.0512", "0.8949 +/- 0.0369"],
    ["ds004504 source-task init", "0.8696 +/- 0.0414", "0.8696 +/- 0.0363", "0.8184 +/- 0.0518", "0.9299 +/- 0.0212", "0.8696 +/- 0.0355", "0.8696 +/- 0.0554"],
]


def style_table(table, row_colors: list[str], highlight_cells: set[tuple[int, int]]) -> None:
    """Apply publication-style table formatting."""

    header_color = "#263238"
    edge_color = "#B0BEC5"
    highlight_fill = "#E8F1FF"

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.55)
        if row_idx == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(0.08)
            continue

        cell.set_facecolor(row_colors[row_idx - 1])
        if col_idx == 0:
            cell.set_text_props(ha="left")
        if (row_idx - 1, col_idx) in highlight_cells:
            cell.set_facecolor(highlight_fill)
            cell.set_text_props(weight="bold")


def add_panel(ax, title: str, rows: list[list[str]], highlight_cells: set[tuple[int, int]]) -> None:
    """Add one formatted panel to the figure."""

    columns = ["Initialization", "Acc", "BalAcc", "F1", "AUC", "Sens", "Spec"]
    row_colors = ["#F7FBFF", "#EEF6FF", "#FFFDF3", "#F8FFF8", "#FFF6FB"]

    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.31, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.7)
    table.scale(1, 1.55)
    style_table(table, row_colors, highlight_cells)


def main() -> None:
    """Create the final ablation summary PDF."""

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "final_initialization_ablation_summary.pdf"

    fig, axes = plt.subplots(2, 1, figsize=(16.5, 11.7))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Final Initialization Ablation Summary",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    ds004504_highlights = {
        (1, 1),  # best acc
        (3, 2),  # best balacc
        (1, 3),  # best f1
        (4, 4),  # best auc
        (1, 5),  # best sens tie representative
        (4, 5),  # best sens
        (3, 6),  # best spec
    }
    caueeg_highlights = {
        (3, 1),  # best acc
        (3, 2),  # best balacc
        (3, 3),  # best f1
        (3, 4),  # best auc
        (4, 5),  # best sens
        (1, 6),  # best spec
    }

    add_panel(axes[0], "Target Dataset: ds004504", DS004504_ROWS, ds004504_highlights)
    add_panel(axes[1], "Target Dataset: CAUEEG", CAUEEG_ROWS, caueeg_highlights)

    note = (
        "Values are mean +/- std across 3 seeds. Blue cells indicate the strongest metrics per target dataset. "
        "Single-source pretraining is effective but source-dependent, while multi-source pretraining remains the most stable default prior."
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=9, color="#455A64")

    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
