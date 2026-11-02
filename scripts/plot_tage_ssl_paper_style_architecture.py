"""Draw a paper-style TAGE-SSL architecture figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


OUTPUT_DIR = Path("outputs")


def box(
    ax,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    fc: str,
    ec: str = "#333333",
    fs: int = 8,
    weight: str = "normal",
    radius: float = 0.04,
    lw: float = 1.0,
) -> None:
    """Draw a rounded box with centered multiline text."""

    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + wh[0] / 2,
        xy[1] + wh[1] / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        weight=weight,
        color="#191919",
        linespacing=1.16,
    )


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#222222",
    rad: float = 0.0,
    lw: float = 1.0,
    dashed: bool = False,
) -> None:
    """Draw an arrow."""

    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            color=color,
            linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def panel(
    ax,
    xy: tuple[float, float],
    wh: tuple[float, float],
    label: str,
    title: str,
    fc: str,
    ec: str,
) -> None:
    """Draw a dashed panel with a small label and title."""

    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.04,rounding_size=0.22",
        linewidth=1.25,
        linestyle=(0, (8, 6)),
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + 0.25, xy[1] + wh[1] - 0.35, label, fontsize=9, weight="bold")
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] - 0.35, title, fontsize=9, weight="bold", ha="center")


def draw_eeg_icon(ax, x0: float, y0: float, width: float, height: float, color: str) -> None:
    """Draw small EEG traces."""

    xs = [x0 + width * i / 48 for i in range(49)]
    for row in range(4):
        baseline = y0 + height * (row + 0.7) / 4.8
        ys = [
            baseline
            + 0.055 * ((i % 6) - 2.5)
            + (0.09 if i % 11 in {0, 1} else 0.0)
            - (0.07 if i % 13 in {6, 7} else 0.0)
            for i in range(49)
        ]
        ax.plot(xs, ys, color=color, linewidth=0.75)


def draw_sample_grid(ax, x0: float, y0: float) -> None:
    """Draw a compact batch-of-samples grid."""

    for r in range(2):
        for c in range(5):
            circ = Circle((x0 + c * 0.43, y0 - r * 0.46), 0.17, facecolor="#D8F0FF", edgecolor="#222222", lw=0.8)
            ax.add_patch(circ)
            ax.text(x0 + c * 0.43, y0 - r * 0.46, str(c + 1), ha="center", va="center", fontsize=6)


def draw_contrast_grid(ax, x0: float, y0: float, title: str) -> None:
    """Draw contrastive sample boxes with positive and negative links."""

    box(ax, (x0, y0), (2.25, 0.83), title, "#F8FCFF", fs=6.6, lw=0.85)
    centers_top = []
    centers_bot = []
    for c in range(4):
        cx = x0 + 0.43 + c * 0.43
        centers_top.append((cx, y0 + 0.58))
        centers_bot.append((cx, y0 + 0.24))
        ax.add_patch(Rectangle((cx - 0.13, y0 + 0.46), 0.26, 0.22, facecolor="#D7ECFF", edgecolor="#222222", lw=0.6))
        ax.add_patch(Rectangle((cx - 0.13, y0 + 0.12), 0.26, 0.22, facecolor="#D7ECFF", edgecolor="#222222", lw=0.6))
    for i in range(4):
        ax.plot([centers_top[i][0], centers_bot[i][0]], [centers_top[i][1], centers_bot[i][1]], color="#1E8AE8", lw=0.9)
        j = (i + 1) % 4
        ax.plot([centers_top[i][0], centers_bot[j][0]], [centers_top[i][1], centers_bot[j][1]], color="#E73345", lw=0.8)


def draw_architecture() -> None:
    """Create the figure."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16.2, 9.3))
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 9.3)
    ax.axis("off")

    panel(ax, (0.15, 0.3), (5.05, 8.65), "a)", "Method Pipeline", "#F0F5FA", "#9AA4AE")
    panel(ax, (5.45, 5.15), (10.55, 3.8), "b)", "Self-Supervised Pre-training", "#FFF9EB", "#E6C87B")
    panel(ax, (5.45, 0.3), (10.55, 4.55), "c)", "TAGE-SSL Backbone Encoder", "#FFF5EA", "#E4B16F")

    # Panel a: pipeline.
    box(ax, (0.8, 7.1), (2.25, 1.0), "All EEG Datasets\nCAUEEG / BrainLat19\nADFSU / ADSZ\nAuditory-40Hz", "#F7F7F7", fs=7)
    box(ax, (0.65, 5.35), (2.55, 1.15), "Preprocessing\n19-channel alignment\n5s windows\n128 Hz resampling\nsubject IDs", "#F7F7F7", fs=7)
    box(ax, (0.65, 3.55), (2.55, 1.05), "Subject-level Split\ntrain / val / test\nno segment leakage", "#F7F7F7", fs=7)
    box(ax, (3.55, 7.15), (1.25, 0.75), "SSL\nPre-train", "#FFE8A8", fs=7, weight="bold")
    box(ax, (3.5, 5.65), (1.35, 0.78), "Fine-tune\nAD vs HC", "#DDEBFF", fs=7, weight="bold")
    box(ax, (3.45, 4.15), (1.45, 0.8), "Subject\nVoting", "#DDEBFF", fs=7)
    box(ax, (3.45, 2.55), (1.45, 0.8), "Metrics\nAcc/F1/AUC\nSen/Spec", "#FFD9D9", fs=7)
    arrow(ax, (1.93, 7.1), (1.93, 6.5))
    arrow(ax, (1.93, 5.35), (1.93, 4.6))
    arrow(ax, (3.05, 7.6), (3.55, 7.55))
    arrow(ax, (4.17, 7.15), (4.17, 6.43))
    arrow(ax, (3.2, 4.08), (3.45, 4.52))
    arrow(ax, (4.18, 5.65), (4.18, 4.95))
    arrow(ax, (4.18, 4.15), (4.18, 3.35))

    for i, fc in enumerate(["#B6E3A5", "#B6E3A5", "#F3B2B2", "#B6E3A5"]):
        ax.add_patch(Circle((3.55 + i * 0.3, 1.75), 0.11, facecolor=fc, edgecolor="#666666", lw=0.5))
    box(ax, (3.1, 1.15), (1.95, 0.45), "Subject-level output", "#F7F7F7", fs=6.6)

    # Panel b: SSL.
    ax.text(5.92, 8.02, "Batch of samples", fontsize=7, weight="bold")
    draw_sample_grid(ax, 6.15, 7.55)
    ax.text(5.88, 6.48, "Augmentation: masking / jittering", fontsize=7)
    box(ax, (8.7, 6.15), (1.05, 1.75), "Backbone\nEncoder\nf(.)", "#FFD8A8", fs=8, weight="bold")
    box(ax, (10.25, 6.15), (0.55, 1.75), "Projection\nHead\ng(.)", "#F8CCCC", fs=7)
    draw_contrast_grid(ax, 11.45, 7.15, "Sample-level contrast")
    draw_contrast_grid(ax, 11.45, 6.05, "Subject-level contrast")
    box(ax, (14.25, 6.67), (0.65, 0.52), "L", "#D8F2B8", fs=10, weight="bold")
    arrow(ax, (8.25, 7.0), (8.7, 7.0))
    arrow(ax, (9.75, 7.0), (10.25, 7.0))
    arrow(ax, (10.8, 7.0), (11.45, 7.57))
    arrow(ax, (10.8, 7.0), (11.45, 6.47))
    arrow(ax, (13.7, 7.57), (14.25, 6.93))
    arrow(ax, (13.7, 6.47), (14.25, 6.93))
    ax.text(14.95, 6.9, "L_ssl", fontsize=7)

    # Panel c: backbone.
    ax.text(5.92, 4.12, "Input EEG window", fontsize=7, weight="bold")
    draw_eeg_icon(ax, 5.75, 3.15, 1.55, 0.78, "#E18A3B")
    draw_eeg_icon(ax, 5.75, 1.35, 1.55, 0.82, "#777777")
    box(ax, (7.65, 3.25), (1.35, 0.9), "SpatialExpert\n1x1 Conv\nk=15 Conv", "#DDECCB", fs=7)
    box(ax, (7.65, 1.25), (1.35, 0.9), "Temporal MSCNN\nk=51/25/11\nSE attention", "#D9E7FF", fs=7)
    box(ax, (9.45, 3.25), (1.2, 0.9), "256-d\nspatial token", "#DDECCB", fs=7)
    box(ax, (9.45, 1.25), (1.2, 0.9), "256-d\ntemporal token", "#D9E7FF", fs=7)
    box(ax, (11.1, 2.15), (1.0, 0.85), "Concat\n512-d", "#FFE8B8", fs=8, weight="bold")
    box(ax, (12.55, 1.7), (1.75, 1.75), "Classifier Head\nLinear 512 -> 256\nBatchNorm + GELU\nDropout\nLinear 256 -> 2", "#FFD9D9", fs=7)
    box(ax, (14.75, 2.15), (0.95, 0.85), "AD / HC\nlogits", "#ECECEC", fs=8, weight="bold")
    arrow(ax, (7.3, 3.52), (7.65, 3.7))
    arrow(ax, (7.3, 1.72), (7.65, 1.7))
    arrow(ax, (9.0, 3.7), (9.45, 3.7))
    arrow(ax, (9.0, 1.7), (9.45, 1.7))
    arrow(ax, (10.65, 3.7), (11.1, 2.78), rad=-0.12)
    arrow(ax, (10.65, 1.7), (11.1, 2.37), rad=0.12)
    arrow(ax, (12.1, 2.58), (12.55, 2.58))
    arrow(ax, (14.3, 2.58), (14.75, 2.58))

    ax.text(
        5.85,
        0.72,
        "Current run: TAGE-SSL + EA, checkpoint = tage_selfsup_brainlat19_e50.pth, full fine-tuning, validation-threshold tuning.",
        fontsize=7.5,
        color="#333333",
    )

    fig.tight_layout(pad=0.4)
    png_path = OUTPUT_DIR / "tage_ssl_paper_style_architecture.png"
    pdf_path = OUTPUT_DIR / "tage_ssl_paper_style_architecture.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    draw_architecture()
