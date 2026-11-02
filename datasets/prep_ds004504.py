import mne
import pandas as pd
import numpy as np
import os
from pathlib import Path

# ==========================================
# 🎯 配置区
# ==========================================
BIDS_ROOT = Path(r"E:\ProgramFiles\PythonProject\BioFoundation-main\eeg-data")
DERIVATIVES_DIR = BIDS_ROOT / "derivatives"
PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"
# 确保输出路径与预训练脚本一致
OUT_DIR = BIDS_ROOT.parent / "ds004504_processed" 
OUT_DIR.mkdir(exist_ok=True)

SFREQ = 256      # 与 ADFSU_processed 对齐：256Hz
EPOCH_LEN = 5.0  # 5秒切片，对应 1280 个点

def get_clean_labels():
    df = pd.read_csv(PARTICIPANTS_FILE, sep='\t')
    label_map = {}
    stats = {'AD (A)': 0, 'CN (C)': 0, 'Excluded (F)': 0}
    
    for _, row in df.iterrows():
        sid = row['participant_id']
        group = str(row['Group']).strip().upper()
        
        # 严格匹配 ds004504 的 A/C/F 标注
        if group == 'A':
            label_map[sid] = 1
            stats['AD (A)'] += 1
        elif group == 'C':
            label_map[sid] = 0
            stats['CN (C)'] += 1
        else:
            # 排除 F 组（FTD）
            stats['Excluded (F)'] += 1
            continue 
            
    print(f"📊 最终匹配结果: {stats}")
    return label_map

def process_subject(set_file):
    """标准化预处理流水线：滤波、重采样、重参考"""
    try:
        raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
        # 0.5-45Hz 带通滤波
        raw.filter(0.5, 45.0, verbose=False)
        if raw.info['sfreq'] != SFREQ:
            raw.resample(SFREQ)
        # 平均参考 (Average Reference)
        raw.set_eeg_reference('average', projection=False, verbose=False)
        # 分段并执行伪影剔除
        epochs = mne.make_fixed_length_epochs(raw, duration=EPOCH_LEN, preload=True, verbose=False)
        epochs.drop_bad(reject=dict(eeg=150e-6), verbose=False) 
        return epochs.get_data()
    except Exception as e:
        print(f"⚠️ 处理 {set_file.name} 出错: {e}")
        return None

def main():
    label_dict = get_clean_labels()
    all_x, all_y, all_ids = [], [], []

    sub_dirs = sorted([d for d in DERIVATIVES_DIR.iterdir() if d.is_dir() and d.name.startswith('sub-')])

    for sub_dir in sub_dirs:
        sid = sub_dir.name
        if sid not in label_dict: continue
        
        label = label_dict[sid]
        set_files = list((sub_dir / "eeg").glob("*.set"))
        
        for f in set_files:
            data = process_subject(f)
            if data is not None and len(data) > 0:
                all_x.append(data)
                all_y.extend([label] * len(data))
                all_ids.extend([sid] * len(data))
                print(f"✅ 已完成受试者: {sid} ({'AD' if label==1 else 'CN'})")

    # 保存为 .npy 文件供模型读取
    np.save(OUT_DIR / "x_19ch.npy", np.concatenate(all_x))
    np.save(OUT_DIR / "y_19ch.npy", np.array(all_y))
    np.save(OUT_DIR / "sub_ids_19ch.npy", np.array(all_ids))
    print(f"🚀 预处理数据已就绪，保存在: {OUT_DIR}")

if __name__ == "__main__":
    main()
