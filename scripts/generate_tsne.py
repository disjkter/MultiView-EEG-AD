"""Step 2: Generate t-SNE plots from extracted features."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEAT_DIR = PROJECT_ROOT / "outputs" / "features"
FIG_DIR = PROJECT_ROOT / "figs"
FIG_DIR.mkdir(exist_ok=True)

CKPT_ORDER = ["random", "ad_ssl", "multi_src"]
CKPT_LABELS = {
    "random": "Random Init",
    "ad_ssl": "Single-Source (AD SSL)",
    "multi_src": "Multi-Source (BL19+40Hz)",
}
DS_SHORT = {"CAUEEG": "CAUEEG", "ds004504": "ds004504"}

cmap = plt.cm.tab10
colors = [cmap(2), cmap(3)]  # green=HC, red=AD


def load_features(ds_name, ckpt_name):
    path = FEAT_DIR / f"feats_{ds_name}_{ckpt_name}.npz"
    d = np.load(path)
    return d["features"], d["labels"]


for ds_name in ["CAUEEG", "ds004504"]:
    print(f"\n=== {ds_name} ===")

    # Collect features across all checkpoints
    all_feat_pairs = []
    all_labels = []
    for ckpt_name in CKPT_ORDER:
        feats, labels = load_features(ds_name, ckpt_name)
        all_feat_pairs.append((CKPT_LABELS[ckpt_name], feats, labels))
        all_labels.append(labels)
        print(f"  {CKPT_LABELS[ckpt_name]}: {feats.shape}, classes={np.bincount(labels)}")

    n_classes = len(np.unique(np.concatenate(all_labels)))
    has_both = n_classes == 2

    if not has_both:
        print(f"  [WARN] Only one class found, showing feature structure only")

    # Concatenate all features for joint t-SNE
    all_feats = np.concatenate([f for _, f, _ in all_feat_pairs], axis=0)
    print(f"  Total: {all_feats.shape[0]} samples x {all_feats.shape[1]} dims")

    # PCA 50 -> t-SNE 2
    pca = PCA(n_components=min(50, all_feats.shape[1]), random_state=42)
    all_pca = pca.fit_transform(all_feats)
    print(f"  PCA50 variance: {pca.explained_variance_ratio_.sum():.4f}")

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1500, verbose=0)
    all_tsne = tsne.fit_transform(all_pca)
    print(f"  t-SNE done: {all_tsne.shape}")

    # Split back
    n_plots = len(all_feat_pairs)
    fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4), sharey=True)
    if n_plots == 1:
        axes = [axes]

    offset = 0
    for ax_idx, (label, feats, lbls) in enumerate(all_feat_pairs):
        n = len(lbls)
        split = all_tsne[offset:offset + n]
        offset += n

        if has_both:
            for cls_idx, cls_label in enumerate(["HC", "AD"]):
                mask = lbls == cls_idx
                ax = axes[ax_idx]
                ax.scatter(split[mask, 0], split[mask, 1], c=[colors[cls_idx]],
                          label=cls_label, s=6, alpha=0.6, edgecolors="none")
        else:
            axes[ax_idx].scatter(split[:, 0], split[:, 1], c=[colors[1]],
                                s=6, alpha=0.5, edgecolors="none")

        axes[ax_idx].set_title(label, fontsize=13, fontweight="bold")
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])

    if has_both:
        axes[0].legend(loc="upper right", fontsize=10, markerscale=2.5,
                       framealpha=0.9, edgecolor="#ccc")

    fig.suptitle(f"t-SNE Visualization of Learned Representations — {DS_SHORT[ds_name]}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = FIG_DIR / f"tsne_{ds_name}.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2, transparent=True)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Also create a combined comparison figure (Random vs Multi-Source only)
    if has_both:
        fig2, axes2 = plt.subplots(1, 2, figsize=(9, 4.2))
        pairs_to_show = [all_feat_pairs[0], all_feat_pairs[2]]  # random vs multi-source
        offset = 0
        for ax_idx, (label, feats, lbls) in enumerate(all_feat_pairs):
            if label not in [CKPT_LABELS["random"], CKPT_LABELS["multi_src"]]:
                offset += len(lbls)
                continue
            n = len(lbls)
            split = all_tsne[offset:offset + n]
            offset += n

            for cls_idx, cls_label in enumerate(["HC", "AD"]):
                mask = lbls == cls_idx
                axes2[0 if label == CKPT_LABELS["random"] else 1].scatter(
                    split[mask, 0], split[mask, 1], c=[colors[cls_idx]],
                    label=cls_label, s=8, alpha=0.6, edgecolors="none")

        axes2[0].set_title(CKPT_LABELS["random"], fontsize=13, fontweight="bold")
        axes2[1].set_title(CKPT_LABELS["multi_src"], fontsize=13, fontweight="bold")
        for ax in axes2:
            ax.set_xticks([]); ax.set_yticks([])
        axes2[0].legend(loc="upper right", fontsize=10, markerscale=2.5)

        fig2.suptitle(f"t-SNE: Random vs Multi-Source Pretraining ({DS_SHORT[ds_name]})",
                      fontsize=14, fontweight="bold")
        plt.tight_layout()
        out2 = FIG_DIR / f"tsne_{ds_name}_comparison.pdf"
        fig2.savefig(out2, dpi=300, bbox_inches="tight", pad_inches=0.2, transparent=True)
        plt.close(fig2)
        print(f"  Saved comparison: {out2}")

print("\nDone! All t-SNE figures saved.")
