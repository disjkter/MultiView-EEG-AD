"""Generate the paper overview for the cross-cohort AD-EEG evaluation protocol."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = (
    ROOT / "docs" / "figures" / "cross_cohort_ad_eeg_evaluation_framework",
    ROOT / "paper" / "cross_cohort_ad_eeg" / "figures" / "cross_cohort_ad_eeg_evaluation_framework",
)

INK = "#303030"
BLUE = "#1F4E79"
LIGHT = "#F2F2F2"
WHITE = "#FFFFFF"


def box(ax, x: float, y: float, width: float, height: float, title: str) -> None:
    """Draw one rectangular pipeline module in normalized coordinates."""

    ax.add_patch(Rectangle((x, y), width, height, facecolor=WHITE, edgecolor=INK, linewidth=0.9))
    header_height = 0.14
    ax.add_patch(
        Rectangle(
            (x, y + height - header_height),
            width,
            header_height,
            facecolor=LIGHT,
            edgecolor="none",
        )
    )
    ax.plot([x, x + width], [y + height - header_height] * 2, color=INK, linewidth=0.65)
    ax.text(x + width / 2, y + height - 0.07, title, ha="center", va="center", fontsize=7.7, weight="bold", color=INK, linespacing=1.0)


def inner_box(ax, x: float, y: float, width: float, height: float, lines: list[str]) -> None:
    """Draw a light-gray submodule with centered editable text."""

    ax.add_patch(Rectangle((x, y), width, height, facecolor=LIGHT, edgecolor="#666666", linewidth=0.55))
    spacing = 0.048
    start = y + height / 2 + spacing * (len(lines) - 1) / 2
    for index, line in enumerate(lines):
        ax.text(x + width / 2, start - index * spacing, line, ha="center", va="center", fontsize=7.5, color=INK)


def arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    """Draw a restrained left-to-right pipeline arrow."""

    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color=INK,
            shrinkA=0,
            shrinkB=0,
        )
    )


def retrospective_box(ax, x: float, y: float, width: float, height: float) -> None:
    """Draw the secondary ADFSU sensitivity outside the primary LODO flow."""

    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=WHITE,
            edgecolor="#666666",
            linewidth=0.8,
            linestyle=(0, (4, 2)),
        )
    )
    ax.text(
        x + 0.015,
        y + height * 0.73,
        "Retrospective ADFSU Sensitivity",
        ha="left",
        va="center",
        fontsize=7.7,
        weight="bold",
        color=INK,
    )
    ax.text(
        x + 0.015,
        y + height * 0.40,
        "Overlap-controlled | 88 participants",
        ha="left",
        va="center",
        fontsize=7.0,
        color=INK,
    )
    ax.text(
        x + 0.015,
        y + height * 0.16,
        "ADSZ excluded from source training",
        ha="left",
        va="center",
        fontsize=7.0,
        color=INK,
    )


def main() -> None:
    """Render SVG, PDF, and PNG versions of the framework figure."""

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.16, 3.15))
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y, height, width = 0.37, 0.58, 0.22
    xs = [0.015, 0.265, 0.515, 0.765]

    box(ax, xs[0], y, width, height, "Development Cohorts")
    inner_box(ax, xs[0] + 0.018, 0.62, width - 0.036, 0.16, ["CAUEEG | ds004504", "ADSZ"])
    ax.text(xs[0] + width / 2, 0.515, "19 channels | 128 Hz", ha="center", va="center", fontsize=7.2, color=INK)
    ax.text(xs[0] + width / 2, 0.455, "1-s windows", ha="center", va="center", fontsize=7.2, color=INK)

    box(ax, xs[1], y, width, height, "Participant-Level LODO")
    inner_box(ax, xs[1] + 0.018, 0.63, width - 0.036, 0.15, ["Source cohorts", "participant-level", "training | validation"])
    inner_box(ax, xs[1] + 0.018, 0.51, width - 0.036, 0.10, ["Unseen target cohort", "evaluation only"])
    ax.text(xs[1] + width / 2, 0.425, "Target cohort excluded", ha="center", va="center", fontsize=7.1, color=INK)
    ax.text(xs[1] + width / 2, 0.39, "from model selection", ha="center", va="center", fontsize=7.1, color=INK)

    box(ax, xs[2], y, width, height, "Controlled Training")
    inner_box(ax, xs[2] + 0.018, 0.63, width - 0.036, 0.15, ["Backbones", "Graph-temporal", "EEGNet"])
    inner_box(ax, xs[2] + 0.018, 0.50, width - 0.036, 0.11, ["Seven objectives", "ERM and DG variants"])
    ax.text(xs[2] + width / 2, 0.415, "10 matched seeds", ha="center", va="center", fontsize=7.2, color=INK)

    box(ax, xs[3], y, width, height, "Participant-Level\nEvaluation")
    inner_box(ax, xs[3] + 0.018, 0.67, width - 0.036, 0.11, ["Window probabilities", "mean by participant"])
    inner_box(ax, xs[3] + 0.018, 0.44, width - 0.036, 0.19, ["AUROC", "Balanced accuracy", "Sensitivity | specificity", "ECE | Brier | NLL"])
    ax.text(xs[3] + width / 2, 0.415, "Worst-cohort performance", ha="center", va="center", fontsize=7.0, color=INK)

    for left, right in zip(xs[:-1], xs[1:]):
        arrow(ax, (left + width + 0.008, 0.66), (right - 0.008, 0.66))

    retrospective_box(ax, 0.265, 0.16, 0.48, 0.15)
    ax.add_patch(
        FancyArrowPatch(
            (0.745, 0.235),
            (xs[3] + width / 2, y - 0.012),
            connectionstyle="angle3,angleA=0,angleB=90",
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            linestyle=(0, (4, 2)),
            color="#666666",
        )
    )

    ax.plot([0.02, 0.985], [0.125, 0.125], color="#666666", linewidth=0.65)
    ax.text(0.02, 0.072, "Protocol controls", ha="left", va="center", fontsize=7.4, weight="bold", color=INK)
    ax.text(
        0.19,
        0.072,
        "fixed participant splits | source-only model / threshold / calibration selection | paired inference",
        ha="left",
        va="center",
        fontsize=7.1,
        color=INK,
    )

    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
        fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("Saved framework figure to " + " and ".join(str(output.parent) for output in OUTPUTS))


if __name__ == "__main__":
    main()
