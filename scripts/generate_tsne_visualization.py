"""Generate t-SNE and UMAP visualizations for paper.

Compares feature distributions of different pretraining strategies
on CAUEEG and ds004504 datasets.

Outputs:
- figs/tsne_caueeg.pdf       : t-SNE of CAUEEG features (Random vs Pretrained)
- figs/tsne_ds004504.pdf     : t-SNE of ds004504 features
- figs/umap_comparison.pdf   : UMAP comparison across strategies
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.Spatial_MSCNN import SpatialMSCNN_DualStream
from models.ablation_model import build_ablation_model


# ---------- UMAP (optional, handle gracefully) ----------
try:
    import umap as _umap_module
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


FIG_DIR = PROJECT_ROOT / "figs"
FIG_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = {
    "Random Init": None,
    "AD SSL": PROJECT_ROOT / "checkpoints/tage_timefreq_consistency_adssl_1s_lead.pth",
    "Multi-Source (BL19+40Hz)": PROJECT_ROOT / "checkpoints/tage_timefreq_consistency_bl19_40hz.pth",
}

DATASETS = {
    "CAUEEG": PROJECT_ROOT / "CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
    "ds004504": PROJECT_ROOT / "ds004504_processed_19ch_1s_lead",
}

N_SUBJECTS = 40  # Subjects per dataset
N_TRIALS_PER_SUB = 30  # Max trials per subject to keep t-SNE fast
BATCH_SIZE = 64


def load_model(ckpt_path, num_classes, num_channels):
    """Load model with partial weight matching."""
    model = SpatialMSCNN_DualStream(num_classes=num_classes, num_channels=num_channels)
    if ckpt_path and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model_state = model.state_dict()
        compatible = {k: v for k, v in ckpt.items()
                      if k in model_state and model_state[k].shape == v.shape}
        model.load_state_dict(compatible, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def extract_features(model, x_data, y_data, batch_size):
    """Extract fused features (512-dim) from dual-stream model."""
    feats, lbls = [], []
    for i in range(0, len(x_data), batch_size):
        end = min(i + batch_size, len(x_data))
        bx = x_data[i:end].astype(np.float32)
        # Per-sample z-score norm
        mean = bx.mean(axis=(1, 2), keepdims=True)
        std = bx.std(axis=(1, 2), keepdims=True) + 1e-6
        bx = (bx - mean) / std
        bt = torch.from_numpy(bx).to(DEVICE)
        with torch.no_grad():
            _, f = model(bt)
        feats.append(f.cpu().numpy())
        lbls.append(y_data[i:end])
    return np.concatenate(feats, axis=0), np.concatenate(lbls, axis=0)


def subsample_trials(sub_ids, y, max_trials_per_sub):
    """Evenly subsample trials per subject."""
    mask = np.zeros(len(sub_ids), dtype=bool)
    for s in np.unique(sub_ids):
        idx = np.where(sub_ids == s)[0]
        if len(idx) > max_trials_per_sub:
            idx = np.random.RandomState(42).choice(idx, max_trials_per_sub, replace=False)
        mask[idx] = True
    return mask


def plot_tsne(features_labels_pairs, title, out_path, figsize=(9, 6)):
    """Generate t-SNE plot with multiple subplots (one per model variant)."""
    n_variants = len(features_labels_pairs)

    # First, PCA to 50 dims
    all_feats = np.concatenate([f for f, _ in features_labels_pairs], axis=0)
    pca = PCA(n_components=min(50, all_feats.shape[1]), random_state=42)
    all_pca = pca.fit_transform(all_feats)

    # Then t-SNE
    print(f"  Running t-SNE on {all_pca.shape[0]} samples x {all_pca.shape[1]} dims...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=0)
    all_tsne = tsne.fit_transform(all_pca)

    # Split back per variant
    offset = 0
    fig, axes = plt.subplots(1, n_variants, figsize=figsize, sharey=True)
    if n_variants == 1:
        axes = [axes]

    cmap = plt.cm.Set1
    colors = [cmap(0), cmap(1)]  # HC=red, AD=blue

    for ax_idx, (ax, (_, labels)) in enumerate(zip(axes, features_labels_pairs)):
        n = len(labels)
        split = all_tsne[offset:offset + n]
        offset += n

        for cls_idx, cls_label in enumerate(["HC", "AD"]):
            mask = labels == cls_idx
            ax.scatter(split[mask, 0], split[mask, 1], c=[colors[cls_idx]],
                       label=cls_label, s=8, alpha=0.5, edgecolors="none")

        var_name = CHECKPOINTS_ORDERED[ax_idx]
        ax.set_title(var_name, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=9, markerscale=2)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  -> {out_path}")


CHECKPOINTS_ORDERED = ["Random Init", "AD SSL", "Multi-Source (BL19+40Hz)"]


def main():
    print(f"Device: {DEVICE}")
    print(f"Subsampling: {N_SUBJECTS} subjects, {N_TRIALS_PER_SUB} trials/subject")

    for ds_name, ds_path in DATASETS.items():
        if not ds_path.exists():
            print(f"[SKIP] {ds_name}")
            continue

        # Load data
        x = np.load(ds_path / "x_19ch.npy", mmap_mode="r")
        y = np.load(ds_path / "y_19ch.npy").astype(np.int64)
        sub = np.load(ds_path / "sub_ids_19ch.npy", allow_pickle=True)

        # Select subjects
        unique_subs = np.unique(sub)[:N_SUBJECTS]
        mask = np.isin(sub, unique_subs)
        indices = np.where(mask)[0]
        sub_sel = sub[indices]
        y_sel = y[indices]

        # Subsample trials
        trial_mask = subsample_trials(sub_sel, y_sel, N_TRIALS_PER_SUB)
        indices = indices[trial_mask]
        y_sel = y_sel[trial_mask]
        sub_sel = sub_sel[trial_mask]

        print(f"\n{'='*60}")
        print(f"{ds_name}: {len(unique_subs)} subjects, {len(y_sel)} trials, "
              f"HC={int(np.sum(y_sel==0))}, AD={int(np.sum(y_sel==1))}")

        # Load data slice
        x_sel = np.array(x[indices], dtype=np.float32)
        num_classes = int(y.max() + 1)
        num_channels = x.shape[1]

        features_labels_pairs = []
        for ckpt_name, ckpt_path in CHECKPOINTS.items():
            print(f"  {ckpt_name}...", end=" ", flush=True)
            model = load_model(ckpt_path, num_classes, num_channels)
            feats, lbls = extract_features(model, x_sel, y_sel, BATCH_SIZE)
            features_labels_pairs.append((feats, lbls))
            print(f"features={feats.shape}")
            del model
            torch.cuda.empty_cache()

        # -------- t-SNE --------
        print(f"  Generating t-SNE plot...")
        plot_tsne(
            features_labels_pairs,
            f"t-SNE Visualization of Learned Representations — {ds_name}",
            FIG_DIR / f"tsne_{ds_name}.pdf",
            figsize=(15, 5),
        )

        # -------- UMAP (if available) --------
        if HAS_UMAP:
            print(f"  Generating UMAP plot...")
            import umap
            fig, axes = plt.subplots(1, len(CHECKPOINTS), figsize=(15, 5))
            cmap = plt.cm.Set1
            colors = [cmap(0), cmap(1)]

            for ax_idx, (ckpt_name, (feats, lbls)) in enumerate(zip(CHECKPOINTS_ORDERED, features_labels_pairs)):
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                embedding = reducer.fit_transform(feats)
                for cls_idx, cls_label in enumerate(["HC", "AD"]):
                    mask = lbls == cls_idx
                    axes[ax_idx].scatter(embedding[mask, 0], embedding[mask, 1],
                                         c=[colors[cls_idx]], label=cls_label, s=8, alpha=0.5, edgecolors="none")
                axes[ax_idx].set_title(ckpt_name, fontsize=12, fontweight="bold")
                axes[ax_idx].set_xticks([])
                axes[ax_idx].set_yticks([])
                if ax_idx == 0:
                    axes[ax_idx].legend(loc="upper right", fontsize=9, markerscale=2)

            fig.suptitle(f"UMAP Visualization of Learned Representations — {ds_name}", fontsize=14, fontweight="bold")
            plt.tight_layout()
            fig.savefig(FIG_DIR / f"umap_{ds_name}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.2)
            plt.close(fig)
            print(f"  -> {FIG_DIR / f'umap_{ds_name}.pdf'}")

    print(f"\nAll figures saved to: {FIG_DIR}")
    print("Ready for paper inclusion.")


if __name__ == "__main__":
    main()
