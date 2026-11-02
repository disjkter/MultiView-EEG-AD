"""Draw the actual architecture used by the current CAUEEG subject-split run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path("outputs")


def add_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#222222",
    fontsize: int = 10,
    weight: str = "normal",
) -> None:
    """Add a rounded text box."""

    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.04",
        linewidth=1.25,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color="#171717",
        linespacing=1.2,
    )


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#303030",
    rad: float = 0.0,
) -> None:
    """Add a directed connection."""

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.25,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def draw_architecture() -> None:
    """Save PNG and PDF diagrams for the current fine-tuning architecture."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16.2, 7.4))
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    fig.suptitle(
        "Current CAUEEG Subject-Level Model: Self-Supervised Dual-Stream EEG Classifier",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )

    input_c = "#E9EEF5"
    prep_c = "#E8F3EE"
    spatial_c = "#DDECCB"
    temporal_c = "#D9E7FF"
    fusion_c = "#FFE8B8"
    head_c = "#FFD9D9"
    output_c = "#ECECEC"
    note_c = "#FAFAFA"

    add_box(ax, (0.35, 3.05), 1.65, 1.0, "EEG window\n[B, 19, 640]", input_c, weight="bold")
    add_box(ax, (2.35, 3.05), 1.85, 1.0, "Euclidean\nAlignment", prep_c)
    add_box(ax, (4.55, 3.05), 1.65, 1.0, "Normalized\nEEG", input_c)
    add_arrow(ax, (2.0, 3.55), (2.35, 3.55))
    add_arrow(ax, (4.2, 3.55), (4.55, 3.55))

    add_box(ax, (6.8, 5.4), 1.9, 0.82, "SpatialExpert", spatial_c, weight="bold")
    add_box(ax, (9.1, 5.75), 1.95, 0.7, "1x1 Conv\n19 -> 64", spatial_c, fontsize=9)
    add_box(ax, (9.1, 4.88), 1.95, 0.7, "Temporal Conv\nk=15, 64 -> 128", spatial_c, fontsize=9)
    add_box(ax, (11.45, 5.22), 1.75, 0.8, "Global Avg Pool\nLinear -> 256", spatial_c, fontsize=9)

    add_box(ax, (6.8, 1.38), 1.9, 0.82, "Temporal MSCNN", temporal_c, weight="bold")
    add_box(ax, (9.1, 2.35), 1.95, 0.7, "MS Conv Block 1\nk=51/25/11 -> 64", temporal_c, fontsize=9)
    add_box(ax, (9.1, 1.48), 1.95, 0.7, "MS Conv Block 2\nk=51/25/11 -> 128", temporal_c, fontsize=9)
    add_box(ax, (9.1, 0.61), 1.95, 0.7, "MS Conv Block 3\nk=51/25/11 -> 256", temporal_c, fontsize=9)
    add_box(ax, (11.45, 1.35), 1.75, 0.8, "SE Attention\nLinear -> 256", temporal_c, fontsize=9)

    add_arrow(ax, (6.2, 3.55), (6.8, 5.81), rad=0.12)
    add_arrow(ax, (6.2, 3.55), (6.8, 1.79), rad=-0.12)
    add_arrow(ax, (8.7, 5.81), (9.1, 6.1))
    add_arrow(ax, (10.07, 5.75), (10.07, 5.58))
    add_arrow(ax, (11.05, 5.23), (11.45, 5.62))
    add_arrow(ax, (8.7, 1.79), (9.1, 2.7))
    add_arrow(ax, (10.07, 2.35), (10.07, 2.18))
    add_arrow(ax, (10.07, 1.48), (10.07, 1.31))
    add_arrow(ax, (11.05, 0.96), (11.45, 1.75))

    add_box(ax, (13.75, 3.05), 1.0, 1.0, "Concat\n512", fusion_c, weight="bold")
    add_arrow(ax, (13.2, 5.62), (13.75, 3.95), rad=-0.12)
    add_arrow(ax, (13.2, 1.75), (13.75, 3.15), rad=0.12)

    add_box(ax, (15.0, 3.05), 0.9, 1.0, "MLP\nHead", head_c, fontsize=9, weight="bold")
    add_box(ax, (14.65, 1.55), 1.6, 0.9, "Linear 512 -> 256\nBN + GELU + Dropout\nLinear 256 -> 2", head_c, fontsize=8)
    add_box(ax, (14.75, 0.25), 1.35, 0.82, "AD / HC\nLogits", output_c, weight="bold")
    add_arrow(ax, (14.75, 3.55), (15.0, 3.55))
    add_arrow(ax, (15.45, 3.05), (15.45, 2.45))
    add_arrow(ax, (15.45, 1.55), (15.45, 1.07))

    ax.text(
        0.45,
        6.65,
        "Checkpoint: checkpoints/tage_selfsup_brainlat19_e50.pth | Fine-tuning: all compatible weights are loaded and updated.",
        fontsize=9.5,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=note_c, edgecolor="#C9C9C9"),
    )
    ax.text(
        0.45,
        0.35,
        "Current command does not enable neuro-prior, GRL, or linear-probe mode. Threshold is tuned on validation subjects; test metrics are subject-level.",
        fontsize=9.5,
        color="#4A4A4A",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    png_path = OUTPUT_DIR / "caueeg_tage_ssl_e50_current_architecture.png"
    pdf_path = OUTPUT_DIR / "caueeg_tage_ssl_e50_current_architecture.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    draw_architecture()
