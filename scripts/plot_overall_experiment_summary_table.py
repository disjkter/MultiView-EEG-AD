"""Render the current overall experiment summary table as a PDF."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROWS = [
    ["CAUEEG", "SVM", "67.39 +/- 5.06", "68.12 +/- 1.68", "58.92 +/- 1.50", "77.60 +/- 0.73", "70.29 +/- 11.27", "65.94 +/- 12.93"],
    ["CAUEEG", "iTransformer", "68.60 +/- 3.47", "69.20 +/- 2.19", "60.10 +/- 2.50", "78.44 +/- 1.83", "71.01 +/- 7.39", "67.39 +/- 7.89"],
    ["CAUEEG", "MTDNet", "79.95 +/- 5.84", "78.08 +/- 4.22", "71.08 +/- 5.40", "85.40 +/- 5.46", "72.46 +/- 6.72", "83.70 +/- 10.24"],
    ["CAUEEG", "EEGNet", "82.13 +/- 3.36", "83.33 +/- 4.03", "76.38 +/- 4.58", "90.25 +/- 3.88", "86.96 +/- 8.13", "79.71 +/- 4.38"],
    ["CAUEEG", "TCN", "84.54 +/- 2.39", "82.79 +/- 2.85", "76.94 +/- 3.75", "90.59 +/- 2.74", "77.54 +/- 6.23", "88.04 +/- 3.55"],
    ["CAUEEG", "CEEDNet", "84.54 +/- 1.23", "85.14 +/- 0.68", "78.97 +/- 0.97", "92.22 +/- 0.43", "86.96 +/- 3.55", "83.33 +/- 3.36"],
    ["CAUEEG", "Ours e50", "87.68 +/- 3.60", "86.23 +/- 0.92", "81.94 +/- 2.76", "94.53 +/- 0.32", "81.88 +/- 8.37", "90.58 +/- 9.49"],
    ["ADFSU", "SVM", "78.33 +/- 0.59", "50.65 +/- 1.27", "87.61 +/- 0.47", "66.91 +/- 4.28", "90.20 +/- 1.39", "11.11 +/- 3.93"],
    ["ADFSU", "EEGNet", "83.33 +/- 2.57", "73.04 +/- 6.13", "89.95 +/- 1.55", "82.88 +/- 2.92", "87.75 +/- 1.83", "58.33 +/- 11.79"],
    ["ADFSU", "ShallowConvNet", "89.17 +/- 2.95", "79.90 +/- 1.73", "93.56 +/- 1.88", "89.75 +/- 1.70", "93.14 +/- 3.47", "66.67 +/- 0.00"],
    ["ADFSU", "MTDNet", "53.33 +/- 3.12", "46.24 +/- 4.01", "67.19 +/- 2.88", "60.95 +/- 2.93", "56.37 +/- 3.67", "36.11 +/- 7.86"],
    ["ADFSU", "iTransformer", "83.75 +/- 1.02", "49.26 +/- 0.60", "91.15 +/- 0.60", "65.07 +/- 4.68", "98.53 +/- 1.20", "0.00 +/- 0.00"],
    ["ADFSU", "TCN", "73.75 +/- 3.54", "60.54 +/- 6.16", "83.69 +/- 2.40", "67.69 +/- 7.94", "79.41 +/- 3.60", "41.67 +/- 11.79"],
    ["ADFSU", "CEEDNet", "72.92 +/- 3.86", "63.48 +/- 3.37", "82.77 +/- 2.94", "67.32 +/- 4.32", "76.96 +/- 4.85", "50.00 +/- 6.80"],
    ["ADFSU", "Ours e50", "77.50 +/- 2.70", "69.61 +/- 8.32", "85.92 +/- 1.77", "65.20 +/- 8.82", "80.88 +/- 3.18", "58.33 +/- 18.00"],
    ["ADFSU", "Ours v1", "87.08 +/- 0.59", "56.94 +/- 1.96", "92.94 +/- 0.30", "89.38 +/- 2.53", "100.00 +/- 0.00", "13.89 +/- 3.93"],
    ["ds004504", "SVM", "70.26 +/- 2.61", "70.69 +/- 2.74", "71.30 +/- 2.36", "77.30 +/- 2.46", "66.67 +/- 2.27", "74.71 +/- 4.30"],
    ["ds004504", "EEGNet", "78.46 +/- 1.26", "78.66 +/- 1.42", "79.81 +/- 0.98", "86.21 +/- 1.57", "76.85 +/- 1.31", "80.46 +/- 3.25"],
    ["ds004504", "ShallowConvNet", "81.03 +/- 4.41", "81.19 +/- 4.53", "82.32 +/- 4.02", "90.61 +/- 0.51", "79.63 +/- 3.46", "82.76 +/- 5.63"],
    ["ds004504", "MTDNet", "76.41 +/- 1.45", "76.36 +/- 1.55", "78.31 +/- 1.22", "81.07 +/- 2.31", "76.85 +/- 1.31", "75.86 +/- 2.82"],
    ["ds004504", "iTransformer", "70.26 +/- 4.04", "70.35 +/- 4.02", "72.08 +/- 3.97", "76.09 +/- 4.78", "69.44 +/- 4.54", "71.26 +/- 4.30"],
    ["ds004504", "TCN", "78.97 +/- 1.45", "78.78 +/- 1.40", "80.92 +/- 1.46", "86.78 +/- 1.22", "80.56 +/- 2.27", "77.01 +/- 1.63"],
    ["ds004504", "CEEDNet", "81.67 +/- 4.08", "81.25 +/- 4.28", "84.50 +/- 3.42", "89.16 +/- 1.14", "83.33 +/- 3.93", "79.17 +/- 5.89"],
    ["ds004504", "Ours old", "95.90 +/- 1.45", "95.40 +/- 1.63", "96.44 +/- 1.21", "99.97 +/- 0.05", "100.00 +/- 0.00", "90.80 +/- 3.25"],
]


def main() -> None:
    """Create a PDF table for the current overall result set."""

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "overall_experiment_summary_table.pdf"

    columns = ["Dataset", "Model", "Acc", "Bal Acc", "F1", "AUC", "Sens", "Spec"]
    fig, ax = plt.subplots(figsize=(16.5, 11.7))
    ax.axis("off")
    ax.set_title("Overall Subject-Level Experiment Summary", fontsize=16, fontweight="bold", pad=18)

    table = ax.table(
        cellText=ROWS,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.09, 0.16, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    table.scale(1, 1.25)

    dataset_colors = {
        "CAUEEG": "#F8FFF8",
        "ADFSU": "#F7FBFF",
        "ds004504": "#FFFDF3",
    }
    ours_color = "#E7F0FF"
    header_color = "#263238"

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#B0BEC5")
        cell.set_linewidth(0.45)
        if row_idx == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(0.04)
            continue
        dataset = ROWS[row_idx - 1][0]
        model = ROWS[row_idx - 1][1]
        cell.set_facecolor(dataset_colors.get(dataset, "white"))
        if "Ours" in model:
            cell.set_facecolor(ours_color)
            cell.set_text_props(weight="bold")
        if col_idx in [0, 1]:
            cell.set_text_props(ha="left")

    footnote = (
        "Values are mean +/- std (%). CEEDNet rows are included where formal 3-seed LOSO results are available. "
        "Ours e50 on ADFSU is the current-checkpoint rerun; Ours v1/old rows are historical checkpoint results."
    )
    fig.text(0.5, 0.025, footnote, ha="center", fontsize=8, color="#455A64")

    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
