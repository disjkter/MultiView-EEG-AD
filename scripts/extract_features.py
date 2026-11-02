"""Step 1: Extract features from pretrained models.
Saves features to outputs/features/ for later t-SNE visualization.
"""
import sys, gc, json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from models.Spatial_MSCNN import SpatialMSCNN_DualStream

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SUBJECTS = 30
N_TRIALS_PER_SUB = 25
BATCH_SIZE = 32

CKPTS = {
    "random": None,
    "ad_ssl": "checkpoints/tage_timefreq_consistency_adssl_1s_lead.pth",
    "multi_src": "checkpoints/tage_timefreq_consistency_bl19_40hz.pth",
}

DATASETS = {
    "CAUEEG": "CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
    "ds004504": "ds004504_processed_19ch_1s_lead",
}

OUT_DIR = PROJECT_ROOT / "outputs" / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data_light(ds_dir, n_sub, n_trials):
    x = np.load(PROJECT_ROOT / ds_dir / "x_19ch.npy", mmap_mode="r")
    y = np.load(PROJECT_ROOT / ds_dir / "y_19ch.npy").astype(np.int64)
    sub = np.load(PROJECT_ROOT / ds_dir / "sub_ids_19ch.npy", allow_pickle=True)

    uniq = np.unique(sub)[:n_sub]
    mask = np.isin(sub, uniq)
    idx_all = np.where(mask)[0]

    # Sub-sample trials per subject — si are local indices into idx_all
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


for ds_name, ds_dir in DATASETS.items():
    if not (PROJECT_ROOT / ds_dir).exists():
        continue

    print(f"\n=== {ds_name} ===")
    xx, yy, ss = load_data_light(ds_dir, N_SUBJECTS, N_TRIALS_PER_SUB)
    num_classes = int(yy.max() + 1)
    print(f"Data: {xx.shape}, HC={int(np.sum(yy==0))}, AD={int(np.sum(yy==1))}")

    # Per-sample norm
    xx_norm = (xx - xx.mean(axis=(1, 2), keepdims=True)) / (xx.std(axis=(1, 2), keepdims=True) + 1e-6)
    print(f"Norm done")

    for ckpt_name, ckpt_path in CKPTS.items():
        print(f"  {ckpt_name}...", end=" ", flush=True)

        model = SpatialMSCNN_DualStream(num_classes=num_classes, num_channels=xx.shape[1])
        if ckpt_path and (PROJECT_ROOT / ckpt_path).exists():
            ckpt = torch.load(PROJECT_ROOT / ckpt_path, map_location=DEVICE)
            state = model.state_dict()
            compat = {k: v for k, v in ckpt.items() if k in state and state[k].shape == v.shape}
            model.load_state_dict(compat, strict=False)
            print(f"loaded {len(compat)}/{len(ckpt)} wts", end=", ")
        model.to(DEVICE)
        model.eval()

        feats_list = []
        for i in range(0, len(xx_norm), BATCH_SIZE):
            end = min(i + BATCH_SIZE, len(xx_norm))
            bt = torch.from_numpy(xx_norm[i:end]).to(DEVICE)
            with torch.no_grad():
                _, f = model(bt)
            feats_list.append(f.cpu().numpy())

        feats = np.concatenate(feats_list, axis=0)
        out_path = OUT_DIR / f"feats_{ds_name}_{ckpt_name}.npz"
        np.savez_compressed(out_path, features=feats, labels=yy, subjects=ss)
        print(f"saved {feats.shape}")

        del model, feats, feats_list
        gc.collect()
        torch.cuda.empty_cache()

print("\nAll features extracted!")
