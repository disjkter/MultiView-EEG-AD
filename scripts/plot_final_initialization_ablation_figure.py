"""Create a polished figure-style PDF for initialization ablation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


COLORS = {
    "Random init": "#9E9E9E",
    "BrainLat19 single-source": "#4C78A8",
    "auditory40hz single-source": "#72B7B2",
    "Multi-source pretraining": "#2E7D32",
    "CAUEEG source-task init": "#C2185B",
    "ds004504 source-task init": "#C2185B",
}


DS004504 = [
    {
        "label": "Random init",
        "balacc_mean": 0.7619,
        "balacc_std": 0.0680,
        "auc_mean": 0.8095,
        "auc_std": 0.0514,
        "sens_mean": 0.8571,
        "sens_std": 0.0000,
        "spec_mean": 0.6667,
        "spec_std": 0.1361,
    },
    {
        "label": "BrainLat19 single-source",
        "balacc_mean": 0.7952,
        "balacc_std": 0.1247,
        "auc_mean": 0.8667,
        "auc_std": 0.0356,
        "sens_mean": 0.8571,
        "sens_std": 0.0000,
        "spec_mean": 0.7333,
        "spec_std": 0.2494,
    },
    {
        "label": "auditory40hz single-source",
        "balacc_mean": 0.7222,
        "balacc_std": 0.0647,
        "auc_mean": 0.8254,
        "auc_std": 0.0112,
        "sens_mean": 0.6667,
        "sens_std": 0.1782,
        "spec_mean": 0.7778,
        "spec_std": 0.0786,
    },
    {
        "label": "Multi-source pretraining",
        "balacc_mean": 0.8016,
        "balacc_std": 0.0312,
        "auc_mean": 0.8492,
        "auc_std": 0.0224,
        "sens_mean": 0.7143,
        "sens_std": 0.1166,
        "spec_mean": 0.8889,
        "spec_std": 0.0786,
    },
    {
        "label": "CAUEEG source-task init",
        "balacc_mean": 0.7524,
        "balacc_std": 0.1831,
        "auc_mean": 0.9429,
        "auc_std": 0.0467,
        "sens_mean": 0.9048,
        "sens_std": 0.0673,
        "spec_mean": 0.6000,
        "spec_std": 0.4320,
    },
]


CAUEEG = [
    {
        "label": "Random init",
        "balacc_mean": 0.8605,
        "balacc_std": 0.0398,
        "auc_mean": 0.9249,
        "auc_std": 0.0319,
        "sens_mean": 0.8188,
        "sens_std": 0.0571,
        "spec_mean": 0.9022,
        "spec_std": 0.0266,
    },
    {
        "label": "BrainLat19 single-source",
        "balacc_mean": 0.8659,
        "balacc_std": 0.0377,
        "auc_mean": 0.9336,
        "auc_std": 0.0202,
        "sens_mean": 0.8188,
        "sens_std": 0.0542,
        "spec_mean": 0.9130,
        "spec_std": 0.0235,
    },
    {
        "label": "auditory40hz single-source",
        "balacc_mean": 0.8533,
        "balacc_std": 0.0277,
        "auc_mean": 0.9309,
        "auc_std": 0.0203,
        "sens_mean": 0.8623,
        "sens_std": 0.1070,
        "spec_mean": 0.8442,
        "spec_std": 0.0825,
    },
    {
        "label": "Multi-source pretraining",
        "balacc_mean": 0.8786,
        "balacc_std": 0.0271,
        "auc_mean": 0.9364,
        "auc_std": 0.0206,
        "sens_mean": 0.8623,
        "sens_std": 0.0512,
        "spec_mean": 0.8949,
        "spec_std": 0.0369,
    },
    {
        "label": "ds004504 source-task init",
        "balacc_mean": 0.8696,
        "balacc_std": 0.0363,
        "auc_mean": 0.9299,
        "auc_std": 0.0212,
        "sens_mean": 0.8696,
        "sens_std": 0.0355,
        "spec_mean": 0.8696,
        "spec_std": 0.0554,
    },
]


def add_panel(ax: plt.Axes, rows: list[dict[str, float]], target_name: str, metric: str, metric_label: str) -> None:
    """Render one horizontal bar panel."""

    labels = [row["label"] for row in rows]
    means = [row[f"{metric}_mean"] for row in rows]
    stds = [row[f"{metric}_std"] for row in rows]
    colors = [COLORS[label] for label in labels]

    y_positions = list(range(len(labels)))
    ax.barh(
        y_positions,
        means,
        xerr=stds,
        color=colors,
        edgecolor="#37474F",
        linewidth=0.6,
        capsize=3,
        alpha=0.95,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.45, 1.0)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_title(f"{target_name}: {metric_label}", fontsize=12, fontweight="bold")
    for idx, mean in enumerate(means):
        ax.text(min(mean + 0.01, 0.985), idx, f"{mean:.3f}", va="center", fontsize=8.5, color="#263238")


def main() -> None:
    """Generate the figure PDF."""

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "final_initialization_ablation_figure.pdf"

    fig, axes = plt.subplots(2, 4, figsize=(21, 11), sharex=False)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Initialization Ablation Across Target Datasets",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    metrics = [
        ("balacc", "Balanced Accuracy"),
        ("auc", "AUC"),
        ("sens", "Sensitivity"),
        ("spec", "Specificity"),
    ]
    for col, (metric, metric_label) in enumerate(metrics):
        add_panel(axes[0, col], DS004504, "Target = ds004504", metric, metric_label)
        add_panel(axes[1, col], CAUEEG, "Target = CAUEEG", metric, metric_label)

    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=8, label=label)
        for label, color in COLORS.items()
        if label in {
            "Random init",
            "BrainLat19 single-source",
            "auditory40hz single-source",
            "Multi-source pretraining",
            "CAUEEG source-task init",
        }
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.02),
    )
    note = (
        "Values are mean +/- std across 3 seeds. Multi-source pretraining is the most consistent "
        "default prior, while single-source pretraining remains effective but source-dependent."
    )
    fig.text(0.5, 0.055, note, ha="center", fontsize=10, color="#455A64")

    plt.tight_layout(rect=[0.03, 0.09, 0.99, 0.95])
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
