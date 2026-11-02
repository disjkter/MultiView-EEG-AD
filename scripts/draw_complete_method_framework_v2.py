"""Draw the complete two-stage AD EEG method as a publication-ready vector figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "byqiu_revision"
STEM = OUT_DIR / "complete_method_framework_v2"

INK = "#253746"
BLUE = "#3C6E91"
BLUE_FILL = "#EAF2F7"
GREEN = "#5B7F5A"
GREEN_FILL = "#EDF4E9"
ORANGE = "#B97832"
ORANGE_FILL = "#FBF0E3"
RED = "#A65D5D"
RED_FILL = "#F8EDED"
GRAY = "#7B8794"
GRAY_FILL = "#F5F7F8"


def rounded_box(ax, x, y, w, h, text="", *, fc=GRAY_FILL, ec=GRAY, fs=8.2,
                lw=1.0, radius=0.05, weight="normal", color=INK, ls="-"):
    """Add a rounded module box in data coordinates."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=color, weight=weight, linespacing=1.18)
    return patch


def arrow(ax, x1, y1, x2, y2, *, text=None, color=INK, dashed=False,
          text_offset=0.12, lw=1.35):
    """Add a directed connection with an optional short label."""
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color, linestyle="--" if dashed else "-",
        shrinkA=2, shrinkB=2,
    ))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + text_offset, text,
                ha="center", va="bottom", fontsize=7.5, color=color)


def eeg_card(ax, x, y, w, h, label, *, edge=BLUE):
    """Draw a compact EEG-window icon with a label."""
    for offset in (0.16, 0.08, 0.0):
        ax.add_patch(Rectangle((x + offset, y + offset), w, h,
                               facecolor="#FFFDF5", edgecolor=edge, linewidth=0.8))
    xs = np.linspace(x + 0.10, x + w - 0.10, 80)
    for idx, phase in enumerate((0.0, 0.8, 1.6, 2.4)):
        baseline = y + 0.20 + idx * (h - 0.34) / 3
        ys = baseline + 0.035 * np.sin(np.linspace(0, 5 * np.pi, 80) + phase)
        ax.plot(xs, ys, color=GRAY, linewidth=0.5)
    ax.text(x + w / 2, y - 0.12, label, ha="center", va="top",
            fontsize=7.5, color=INK)


def encoder(ax, x, y, w, h, title, *, compact=False):
    """Draw the shared spatial-temporal encoder and return its output anchor."""
    rounded_box(ax, x, y, w, h, fc="#FCFDFC", ec=GREEN, lw=1.1, radius=0.07)
    ax.text(x + w / 2, y + h - 0.18, title, ha="center", va="top",
            fontsize=8.5, color=INK, weight="bold")
    inner_x = x + 0.16
    inner_w = w - 0.32
    branch_h = 0.45 if compact else 0.52
    top_y = y + h - 0.85
    bottom_y = y + 0.26
    top_text = "Spatial: Channel Mix  →  Conv  →  Pool"
    bottom_text = "Temporal: Multi-scale Conv  →  Pool  →  Weight"
    rounded_box(ax, inner_x, top_y, inner_w, branch_h, top_text,
                fc=BLUE_FILL, ec=BLUE, fs=7.0 if compact else 7.3, radius=0.025)
    rounded_box(ax, inner_x, bottom_y, inner_w, branch_h, bottom_text,
                fc=GREEN_FILL, ec=GREEN, fs=7.0 if compact else 7.3, radius=0.025)
    return x + w, y + h / 2


fig, ax = plt.subplots(figsize=(12.0, 5.3))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.3)
ax.axis("off")

# Stage containers establish the two-stage method at first glance.
rounded_box(ax, 0.12, 2.78, 11.76, 2.38, fc="#FBFCFD", ec=GRAY,
            lw=0.9, radius=0.10)
rounded_box(ax, 0.12, 0.12, 11.76, 2.38, fc="#FBFCFD", ec=GRAY,
            lw=0.9, radius=0.10)
ax.text(0.38, 4.92, "Stage 1  Multi-View Self-Supervised Pretraining",
        fontsize=10.5, weight="bold", color=INK)
ax.text(0.38, 2.26, "Stage 2  Participant-Aware Fine-Tuning",
        fontsize=10.5, weight="bold", color=INK)

# Stage 1: unlabeled data and windows.
rounded_box(ax, 0.38, 3.28, 1.25, 1.18, "Unlabeled EEG\n258 participants",
            fc=BLUE_FILL, ec=BLUE, fs=8.0)
ax.text(1.005, 3.07, "BrainLat19 · 40-Hz · ADSZ", ha="center", fontsize=6.8, color=GRAY)
arrow(ax, 1.66, 3.88, 2.04, 3.88)
eeg_card(ax, 2.10, 3.49, 0.74, 0.76, "19 × 128")
arrow(ax, 2.95, 3.88, 3.30, 3.88)

# Shared two-branch encoder and fusion.
encoder(ax, 3.35, 3.12, 3.05, 1.55, "Two-Branch Encoder")
arrow(ax, 6.43, 3.88, 6.77, 3.88)
rounded_box(ax, 6.82, 3.48, 0.72, 0.80, "Concat", fc=ORANGE_FILL, ec=ORANGE, fs=7.5)
arrow(ax, 7.57, 3.88, 7.91, 3.88)

# Four pretraining tasks are grouped as one method component.
rounded_box(ax, 7.96, 3.10, 2.70, 1.58, fc="#FFFCF7", ec=ORANGE, lw=1.0, radius=0.06)
ax.text(9.31, 4.48, "Self-Supervised Tasks", ha="center", va="top",
        fontsize=8.5, weight="bold", color=INK)
task_specs = [
    (8.12, 3.84, "Waveform"), (9.39, 3.84, "Spectrum"),
    (8.12, 3.28, "Branch"), (9.39, 3.28, "Temporal View"),
]
for x, y, label in task_specs:
    rounded_box(ax, x, y, 1.10, 0.38, label, fc=ORANGE_FILL, ec=ORANGE,
                fs=7.0, radius=0.018)
arrow(ax, 10.69, 3.88, 11.07, 3.88)
rounded_box(ax, 11.11, 3.50, 0.52, 0.76, r"$\mathcal{L}_{pre}$",
            fc=GRAY_FILL, ec=GRAY, fs=9.0, radius=0.025, weight="bold")

# Stage 2: participant-level supervision and aggregation.
rounded_box(ax, 0.38, 0.63, 1.25, 1.18, "Labeled EEG\nAD / HC",
            fc=BLUE_FILL, ec=BLUE, fs=8.0)
ax.text(1.005, 0.42, "Participant-level splits", ha="center", fontsize=6.8, color=GRAY)
arrow(ax, 1.66, 1.23, 2.04, 1.23)
eeg_card(ax, 2.10, 0.84, 0.74, 0.76, "Training windows")
arrow(ax, 2.95, 1.23, 3.30, 1.23)
encoder(ax, 3.35, 0.47, 3.05, 1.55, "Transferred Encoder", compact=True)

# Weight transfer is the only cross-stage arrow.
arrow(ax, 4.88, 3.10, 4.88, 2.04, text="Encoder weights", color=GREEN,
      dashed=True, text_offset=0.05, lw=1.25)

arrow(ax, 6.43, 1.23, 6.75, 1.23)
rounded_box(ax, 6.80, 0.81, 0.90, 0.84, "Classifier", fc=ORANGE_FILL, ec=ORANGE, fs=7.5)
arrow(ax, 7.73, 1.23, 8.05, 1.23)
rounded_box(ax, 8.10, 0.81, 1.04, 0.84, "Window\nprobabilities", fc=GRAY_FILL, ec=GRAY, fs=7.2)
arrow(ax, 9.17, 1.23, 9.49, 1.23)
rounded_box(ax, 9.54, 0.81, 1.10, 0.84, "Participant\naverage", fc=GREEN_FILL, ec=GREEN, fs=7.2)
arrow(ax, 10.67, 1.23, 10.99, 1.23)
rounded_box(ax, 11.04, 0.81, 0.58, 0.84, "AD\n/\nHC", fc=RED_FILL, ec=RED, fs=7.5)

fig.tight_layout(pad=0.2)
for ext in ("pdf", "svg", "png"):
    fig.savefig(STEM.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.04,
                dpi=240)
plt.close(fig)
print(STEM.with_suffix(".pdf"))
