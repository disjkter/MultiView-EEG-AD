"""Draw the TAGE-Net architecture used in the EEG experiments."""

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
    """Add a rounded box with centered text."""

    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.045",
        linewidth=1.3,
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
        linespacing=1.25,
    )


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#333333",
    linestyle: str = "-",
    rad: float = 0.0,
) -> None:
    """Add an arrow between two points."""

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.25,
        color=color,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def draw_architecture() -> None:
    """Create and save the architecture diagram."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(17.2, 8.2))
    ax.set_xlim(0, 16.8)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    fig.suptitle(
        "TAGE-Net: Topology-Aware Generalization EEG Network",
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )

    # Palette chosen for print-friendly contrast.
    input_c = "#E8EEF6"
    prior_c = "#E7F4EF"
    spatial_c = "#DCECCB"
    temporal_c = "#D9E7FF"
    fusion_c = "#FFE8B8"
    head_c = "#FFD6D6"
    domain_c = "#E8DDF7"
    output_c = "#EDEDED"

    add_box(ax, (0.3, 3.55), 1.65, 1.1, "EEG Input\n[B, 19, T]", input_c, weight="bold")
    add_box(ax, (2.2, 3.55), 1.65, 1.1, "Channel-wise\nNormalization", input_c)
    add_box(
        ax,
        (4.3, 3.22),
        2.05,
        1.76,
        "Neurophysiological\nPrior Layer\n\nRegions + Bands\n+ Connectivity",
        prior_c,
        fontsize=9,
        weight="bold",
    )

    add_arrow(ax, (1.95, 4.1), (2.2, 4.1))
    add_arrow(ax, (3.85, 4.1), (4.3, 4.1))

    # Spatial branch.
    add_box(ax, (6.95, 5.65), 2.0, 0.9, "Spatial Expert", spatial_c, weight="bold")
    add_box(ax, (9.35, 5.95), 1.95, 0.72, "1x1 Conv\n19 -> 64", spatial_c, fontsize=9)
    add_box(ax, (9.35, 5.05), 1.95, 0.72, "Temporal Conv\nk=15, 64 -> 128", spatial_c, fontsize=9)
    add_box(ax, (11.75, 5.48), 1.65, 0.82, "GAP + FC\n256-d", spatial_c, fontsize=9)

    # Temporal branch.
    add_box(ax, (6.95, 1.7), 2.0, 0.9, "Temporal MSCNN", temporal_c, weight="bold")
    add_box(ax, (9.35, 2.45), 1.95, 0.72, "MS Block 1\nk=51/25/11, 64", temporal_c, fontsize=9)
    add_box(ax, (9.35, 1.55), 1.95, 0.72, "MS Block 2\nk=51/25/11, 128", temporal_c, fontsize=9)
    add_box(ax, (9.35, 0.65), 1.95, 0.72, "MS Block 3\nk=51/25/11, 256", temporal_c, fontsize=9)
    add_box(ax, (11.75, 1.55), 1.65, 0.82, "SE Attention\n+ FC 256-d", temporal_c, fontsize=9)

    # Shared split arrows.
    add_arrow(ax, (6.35, 4.1), (6.95, 6.1), rad=0.12)
    add_arrow(ax, (6.35, 4.1), (6.95, 2.15), rad=-0.12)
    add_arrow(ax, (8.95, 6.1), (9.35, 6.31))
    add_arrow(ax, (10.32, 5.95), (10.32, 5.77))
    add_arrow(ax, (11.3, 5.41), (11.75, 5.89))
    add_arrow(ax, (8.95, 2.15), (9.35, 2.81))
    add_arrow(ax, (10.32, 2.45), (10.32, 2.27))
    add_arrow(ax, (10.32, 1.55), (10.32, 1.37))
    add_arrow(ax, (11.3, 1.01), (11.75, 1.96))

    # Fusion and disease head.
    add_box(ax, (14.9, 3.55), 1.55, 1.1, "Concat\n512-d", fusion_c, weight="bold")
    add_box(ax, (13.75, 3.55), 0.75, 1.1, "+", fusion_c, fontsize=18, weight="bold")
    add_arrow(ax, (13.4, 5.89), (14.9, 4.65), rad=-0.12)
    add_arrow(ax, (13.4, 1.96), (14.9, 3.55), rad=0.12)

    add_box(ax, (14.9, 2.05), 1.55, 0.92, "Disease Head\nLinear-BN-GELU\nDropout-Linear", head_c, fontsize=8)
    add_box(ax, (14.9, 0.65), 1.55, 0.82, "AD / HC\nLogits", output_c, weight="bold")
    add_arrow(ax, (15.68, 3.55), (15.68, 2.97))
    add_arrow(ax, (15.68, 2.05), (15.68, 1.47))

    # Domain extension.
    add_box(ax, (14.9, 5.8), 1.55, 0.78, "GRL", domain_c, weight="bold")
    add_box(ax, (14.9, 6.82), 1.55, 0.82, "Domain Head\nSource / Target", domain_c, fontsize=8)
    add_arrow(ax, (15.68, 4.65), (15.68, 5.8), color="#6E4AA8", linestyle="--")
    add_arrow(ax, (15.68, 6.58), (15.68, 6.82), color="#6E4AA8", linestyle="--")

    ax.text(
        0.45,
        7.15,
        "Neurophysiological priors: region-level activity, canonical EEG bands, and Pearson functional connectivity.",
        fontsize=10,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FAFAFA", edgecolor="#C9C9C9"),
    )
    ax.text(
        0.45,
        0.35,
        "Training losses: L = L_cls + lambda(t) * L_domain (+ optional prototype alignment). Priors provide a basis for interpretability and medical-report generation.",
        fontsize=9.5,
        color="#4A4A4A",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png_path = OUTPUT_DIR / "tage_net_architecture.png"
    pdf_path = OUTPUT_DIR / "tage_net_architecture.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    draw_architecture()
