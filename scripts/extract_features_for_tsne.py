"""Extract features from trained dual-stream model for t-SNE/UMAP visualization.

This script loads a trained model, runs inference on CAUEEG and ds004504 data,
extracts the fused features (before the classifier), and saves them for visualization.
"""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.Spatial_MSCNN import SpatialMSCNN_DualStream


def load_checkpoint(path: str, num_classes: int, num_channels: int, device: torch.device):
    """Load a checkpoint with partial matching."""
    model = SpatialMSCNN_DualStream(num_classes=num_classes, num_channels=num_channels)
    ckpt = torch.load(path, map_location=device)
    model_state = model.state_dict()
    compatible = {k: v for k, v in ckpt.items() if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(compatible, strict=False)
    model.to(device)
    model.eval()
    return model


def extract_features(model, x_data, y_data, sub_ids, batch_size, device):
    """Extract fused features and labels, with per-sample z-score normalization."""
    features_list = []
    labels_list = []
    subject_list = []

    for i in range(0, len(x_data), batch_size):
        end = min(i + batch_size, len(x_data))
        batch_x = x_data[i:end].astype(np.float32)
        # Per-sample z-score
        mean = batch_x.mean(axis=(1, 2), keepdims=True)
        std = batch_x.std(axis=(1, 2), keepdims=True) + 1e-6
        batch_x = (batch_x - mean) / std
        batch_t = torch.from_numpy(batch_x).to(device)

        with torch.no_grad():
            _, features = model(batch_t)

        features_list.append(features.cpu().numpy())
        labels_list.append(y_data[i:end])
        subject_list.append(sub_ids[i:end])

    return (
        np.concatenate(features_list, axis=0),
        np.concatenate(labels_list, axis=0),
        np.concatenate(subject_list, axis=0),
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoints to evaluate
    checkpoints = {
        "Random Init": None,  # Fresh random model
        "AD SSL Pretrained": "checkpoints/tage_timefreq_consistency_adssl_1s_lead.pth",
        "Multi-Source Pretrained": "checkpoints/tage_timefreq_consistency_bl19_40hz.pth",
    }

    # Datasets
    datasets = {
        "CAUEEG": "CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
        "ds004504": "ds004504_processed_19ch_1s_lead",
    }

    output_dir = PROJECT_ROOT / "outputs" / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds_name, ds_dir in datasets.items():
        ds_path = PROJECT_ROOT / ds_dir
        if not ds_path.exists():
            print(f"[SKIP] {ds_name} not found at {ds_path}")
            continue

        # Load data
        x = np.load(ds_path / "x_19ch.npy", mmap_mode="r")
        y = np.load(ds_path / "y_19ch.npy").astype(np.int64)
        sub = np.load(ds_path / "sub_ids_19ch.npy", allow_pickle=True)

        # Limit to 50 subjects for manageable t-SNE
        unique_subs = np.unique(sub)[:50]
        mask = np.isin(sub, unique_subs)
        indices = np.where(mask)[0]
        x_sub = np.array(x[indices], dtype=np.float32)
        y_sub = y[indices]
        sub_sub = sub[indices]

        print(f"{ds_name}: {len(unique_subs)} subjects, {len(y_sub)} trials, labels={np.bincount(y_sub)}")

        for ckpt_name, ckpt_path in checkpoints.items():
            print(f"  Processing: {ckpt_name}...")
            if ckpt_path and not (PROJECT_ROOT / ckpt_path).exists():
                print(f"    [SKIP] checkpoint not found: {ckpt_path}")
                continue

            num_classes = int(y.max() + 1)
            num_channels = x.shape[1]

            if ckpt_path:
                model = load_checkpoint(str(PROJECT_ROOT / ckpt_path), num_classes, num_channels, device)
                print(f"    Loaded {ckpt_path}")
            else:
                model = SpatialMSCNN_DualStream(num_classes=num_classes, num_channels=num_channels)
                model.to(device)
                model.eval()
                print(f"    Random initialization")

            features, labels, subjects = extract_features(model, x_sub, y_sub, sub_sub, batch_size=64, device=device)

            # PCA to 50 dims first (faster t-SNE)
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(50, features.shape[1]))
            features_pca = pca.fit_transform(features)
            print(f"    Features: {features.shape} -> PCA50: {features_pca.shape}")

            out_path = output_dir / f"features_{ds_name}_{ckpt_name.replace(' ', '_').replace('/', '_')}.npz"
            np.savez(out_path, features=features_pca, labels=labels, subjects=subjects,
                     pca_explained=pca.explained_variance_ratio_.sum(), num_components=features.shape[1])
            print(f"    Saved: {out_path}")

            del model
            torch.cuda.empty_cache()

    print("\nAll features extracted. Ready for t-SNE visualization.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
