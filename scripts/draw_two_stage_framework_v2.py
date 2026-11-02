"""Draw the two-stage paper framework as a compact vector figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "byqiu_revision" / "two_stage_framework_v2.pdf"


def box(ax, x, y, w, h, text, fc="#F7F9FC", ec="#355C7D", fs=9, lw=1.2, radius=0.035):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#1F2933")
    return patch


def arrow(ax, x1, y1, x2, y2, text=None, color="#355C7D"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.6, color=color))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.045, text, ha="center", va="bottom",
                fontsize=8.5, color="#334E68")


fig, ax = plt.subplots(figsize=(12.0, 3.4))
ax.set_xlim(0, 12)
ax.set_ylim(0, 3.4)
ax.axis("off")

# Stage containers
box(ax, 0.15, 0.25, 6.9, 2.85, "", fc="#FBFCFE", ec="#7B8794", lw=1.0, radius=0.10)
box(ax, 7.35, 0.25, 4.5, 2.85, "", fc="#FBFCFE", ec="#7B8794", lw=1.0, radius=0.10)
ax.text(0.45, 2.84, "Stage 1  Self-Supervised Pretraining", fontsize=11, weight="bold", color="#243B53")
ax.text(7.65, 2.84, "Stage 2  AD/HC Fine-Tuning", fontsize=11, weight="bold", color="#243B53")

# Stage 1 data
box(ax, 0.45, 1.08, 1.30, 1.12, "Unlabeled EEG\n258 participants", fc="#EEF4FA", ec="#2F6FA3", fs=9)
ax.text(1.10, 0.78, "BrainLat19 · 40-Hz · ADSZ", ha="center", fontsize=7.5, color="#526D82")
arrow(ax, 1.80, 1.64, 2.35, 1.64)

# Encoder
box(ax, 2.40, 0.82, 2.08, 1.65, "", fc="#F4F8F3", ec="#4E7D52", fs=9)
ax.text(3.44, 2.21, "Two-Branch Encoder", ha="center", fontsize=9.5, weight="bold", color="#294B2D")
box(ax, 2.63, 1.48, 1.62, 0.43, "Spatial Branch", fc="#E4F0E3", ec="#6B9B6E", fs=8.5, radius=0.02)
box(ax, 2.63, 0.96, 1.62, 0.43, "Temporal Branch", fc="#E4F0E3", ec="#6B9B6E", fs=8.5, radius=0.02)

arrow(ax, 4.53, 1.64, 5.02, 1.64)
box(ax, 5.08, 0.69, 1.68, 1.90, "Waveform\nSpectrum\nBranch\nTemporal View", fc="#FFF6E8", ec="#C57C24", fs=8.5)
ax.text(5.92, 0.48, "Four pretraining losses", ha="center", fontsize=7.8, color="#7C4A12")

# Transfer arrow
arrow(ax, 6.98, 1.64, 7.75, 1.64, "Pretrained encoder")

# Stage 2
box(ax, 7.82, 1.08, 1.23, 1.12, "Labeled EEG\nCAUEEG", fc="#EEF4FA", ec="#2F6FA3", fs=9)
arrow(ax, 9.10, 1.64, 9.53, 1.64)
box(ax, 9.58, 1.06, 1.18, 1.16, "Encoder\n+\nClassifier", fc="#F4F8F3", ec="#4E7D52", fs=9)
arrow(ax, 10.81, 1.64, 11.17, 1.64)
box(ax, 11.21, 1.15, 0.44, 0.98, "AD\n/\nHC", fc="#FCEFEF", ec="#A85B5B", fs=8.5, radius=0.02)
ax.text(9.58, 0.68, "Participant-level splits", ha="center", fontsize=7.8, color="#526D82")
ax.text(10.92, 0.48, "Average window probabilities", ha="center", fontsize=7.8, color="#526D82")

fig.tight_layout(pad=0.2)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print(OUT)
