"""Generate a Chinese PDF for the initialization ablation tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


DS004504_ROWS = [
    ["随机初始化", "0.7692 ± 0.0628", "0.7619 ± 0.0680", "0.8024 ± 0.0438", "0.8095 ± 0.0514", "0.8571 ± 0.0000", "0.6667 ± 0.1361"],
    ["BrainLat19 单源预训练", "0.8056 ± 0.1039", "0.7952 ± 0.1247", "0.8434 ± 0.0713", "0.8667 ± 0.0356", "0.8571 ± 0.0000", "0.7333 ± 0.2494"],
    ["auditory40hz 单源预训练", "0.7179 ± 0.0725", "0.7222 ± 0.0647", "0.7049 ± 0.1134", "0.8254 ± 0.0112", "0.6667 ± 0.1782", "0.7778 ± 0.0786"],
    ["多源预训练", "0.7949 ± 0.0363", "0.8016 ± 0.0312", "0.7845 ± 0.0541", "0.8492 ± 0.0224", "0.7143 ± 0.1166", "0.8889 ± 0.0786"],
    ["多源预训练 + CAUEEG 任务初始化", "0.7778 ± 0.1416", "0.7524 ± 0.1831", "0.8390 ± 0.0771", "0.9429 ± 0.0467", "0.9048 ± 0.0673", "0.6000 ± 0.4320"],
]

CAUEEG_ROWS = [
    ["随机初始化", "0.8744 ± 0.0347", "0.8605 ± 0.0398", "0.8128 ± 0.0524", "0.9249 ± 0.0319", "0.8188 ± 0.0571", "0.9022 ± 0.0266"],
    ["BrainLat19 单源预训练", "0.8816 ± 0.0326", "0.8659 ± 0.0377", "0.8216 ± 0.0498", "0.9336 ± 0.0202", "0.8188 ± 0.0542", "0.9130 ± 0.0235"],
    ["auditory40hz 单源预训练", "0.8502 ± 0.0304", "0.8533 ± 0.0277", "0.7935 ± 0.0343", "0.9309 ± 0.0203", "0.8623 ± 0.1070", "0.8442 ± 0.0825"],
    ["多源预训练", "0.8841 ± 0.0258", "0.8786 ± 0.0271", "0.8325 ± 0.0347", "0.9364 ± 0.0206", "0.8623 ± 0.0512", "0.8949 ± 0.0369"],
    ["多源预训练 + ds004504 任务初始化", "0.8696 ± 0.0414", "0.8696 ± 0.0363", "0.8184 ± 0.0518", "0.9299 ± 0.0212", "0.8696 ± 0.0355", "0.8696 ± 0.0554"],
]


def style_table(table, row_colors: list[str]) -> None:
    """Apply simple publication-oriented styling."""

    header_color = "#263238"
    edge_color = "#B0BEC5"

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.55)
        if row_idx == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(0.085)
        else:
            cell.set_facecolor(row_colors[row_idx - 1])
            if col_idx == 0:
                cell.set_text_props(ha="left")


def add_panel(ax: plt.Axes, title: str, rows: list[list[str]]) -> None:
    """Render one table panel."""

    columns = ["初始化策略", "Acc", "BalAcc", "F1", "AUC", "Sens", "Spec"]
    row_colors = ["#F7FBFF", "#EEF6FF", "#FFFDF3", "#F8FFF8", "#FFF6FB"]

    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.34, 0.105, 0.105, 0.105, 0.105, 0.105, 0.105],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1, 1.6)
    style_table(table, row_colors)


def main() -> None:
    """Generate the Chinese initialization ablation PDF."""

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "initialization_ablation_tables_cn.pdf"

    fig, axes = plt.subplots(2, 1, figsize=(16.5, 11.7))
    fig.patch.set_facecolor("white")
    fig.suptitle("不同初始化策略的总体比较", fontsize=18, fontweight="bold", y=0.98)

    add_panel(axes[0], "表 1 目标数据集 ds004504 上不同初始化策略的总体比较", DS004504_ROWS)
    add_panel(axes[1], "表 2 目标数据集 CAUEEG 上不同初始化策略的总体比较", CAUEEG_ROWS)

    note = (
        "注：source-task 初始化并不是独立于第一阶段预训练之外的初始化方式，而是先从多源预训练编码器开始，"
        "再通过源域 AD/HC 监督训练进一步构造得到的增强迁移先验。结果以 3 个随机种子的均值 ± 标准差表示。"
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=9, color="#455A64")

    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
