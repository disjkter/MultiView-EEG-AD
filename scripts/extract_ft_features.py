"""Extract features from fine-tuned models for t-SNE.
Uses the LAST FC layer features (before classifier head) which separate AD/HC.
"""
import sys, gc, json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from models.Spatial_MSCNN import SpatialMSCNN_DualStream

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SUBJECTS = 40
N_TRIALS_PER_SUB = 25
BATCH_SIZE = 32

# Fine-tuned checkpoints on CAUEEG
FT_CHECKPOINTS = {
    "Random Init (FT)": None,  # random init + train from scratch
    "Multi-Source (FT)": "outputs/caueeg_tage_timefreq_consistency_bl19_40hz_seed2026/best_model.pth",
    "AD SSL (FT)": "outputs/caueeg_tage_timefreq_consistency_adssl_seed2026/best_model.pth",
}

CKPTS_REAL = {}
for name, path in FT_CHECKPOINTS.items():
    full_path = PROJECT_ROOT / path if path else None
    if full_path and full_path.exists():
        CKPTS_REAL[name] = full_path
    elif path is None:
        CKPTS_REAL[name] = None

DATASETS = {
    "CAUEEG": "CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
}

OUT_DIR = PROJECT_ROOT / "outputs" / "features_ft"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data_light(ds_dir, n_sub, n_trials):
    x = np.load(PROJECT_ROOT / ds_dir / "x_19ch.npy", mmap_mode="r")
    y = np.load(PROJECT_ROOT / ds_dir / "y_19ch.npy").astype(np.int64)
    sub = np.load(PROJECT_ROOT / ds_dir / "sub_ids_19ch.npy", allow_pickle=True)
    uniq = np.unique(sub)[:n_sub]
    mask = np.isin(sub, uniq)
    idx_all = np.where(mask)[0]
    keep = np.zeros(len(idx_all), dtype=bool)
    rng = np.random.RandomState(42)
    for s in uniq:
        si = np.where(sub[idx_all] == s)[0]
        if len(si) > n_trials:
            si = rng.choice(si, n_trials, replace=False)
        keep[si] = True
    idx_final = idx_all[keep]
    xx = np.array(x[idx_final], dtype=np.float32)
    yy = y[idx_final]
    ss = sub[idx_final]
    return xx, yy, ss


# ---------- Feature hook for penultimate layer ----------
# The classifier is: Linear(256/512, hidden) -> BN -> GELU -> Dropout -> Linear(hidden, 2)
# We want features AFTER the last GELU, BEFORE the final Linear

class FeatureExtractor(torch.nn.Module):
    """Wrapper that returns the fused dual-stream features (512-dim / 256-dim).
    These are the raw encoder outputs BEFORE the classifier head."""

    def __init__(self, model):
        super().__init__()
        self._model = model

    def forward(self, x):
        _, fused = self._model(x)  # (logits, fused_features)
        return fused


for ds_name, ds_dir in DATASETS.items():
    print(f"\n=== {ds_name} ===")
    xx, yy, ss = load_data_light(ds_dir, N_SUBJECTS, N_TRIALS_PER_SUB)
    num_classes = int(yy.max() + 1)
    print(f"Data: {xx.shape}, HC={int(np.sum(yy==0))}, AD={int(np.sum(yy==1))}")

    xx_norm = (xx - xx.mean(axis=(1, 2), keepdims=True)) / (xx.std(axis=(1, 2), keepdims=True) + 1e-6)

    for ckpt_name, ckpt_path in CKPTS_REAL.items():
        print(f"  {ckpt_name}...", end=" ", flush=True)

        model = SpatialMSCNN_DualStream(num_classes=num_classes, num_channels=xx.shape[1])
        if ckpt_path and ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            state = model.state_dict()
            compat = {k: v for k, v in ckpt.items() if k in state and state[k].shape == v.shape}
            model.load_state_dict(compat, strict=False)
            print(f"loaded {len(compat)} wts", end=", ")

        model.to(DEVICE)
        extractor = FeatureExtractor(model)
        extractor.eval()

        feats_list = []
        for i in range(0, len(xx_norm), BATCH_SIZE):
            end = min(i + BATCH_SIZE, len(xx_norm))
            bt = torch.from_numpy(xx_norm[i:end]).to(DEVICE)
            with torch.no_grad():
                f = extractor(bt)
            feats_list.append(f.cpu().numpy())

        feats = np.concatenate(feats_list, axis=0)
        out_path = OUT_DIR / f"feats_{ds_name}_{ckpt_name.replace(' ', '_')}.npz"
        np.savez_compressed(out_path, features=feats, labels=yy, subjects=ss)
        print(f"saved {feats.shape}")

        del model, extractor, feats, feats_list
        gc.collect()
        torch.cuda.empty_cache()

print("\nDone! Fine-tuned features ready.")
