import os
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

def main():
    # --- 路径锁定 ---
    INPUT_DIR = Path(r"E:\ProgramFiles\PythonProject\BioFoundation-main\eeg-data\derivatives")
    PARTICIPANTS_FILE = Path(r"E:\ProgramFiles\PythonProject\BioFoundation-main\eeg-data\participants.tsv")
    OUTPUT_ROOT = Path(r"E:\ProgramFiles\PythonProject\BioFoundation-main\model_data_cv")
    
    TARGET_LENGTH = 1280 # 256Hz * 5s
    NUM_FOLDS = 10
    
    print(f"🚀 开始 10 折交叉验证数据准备...")

    # 1. 加载受试者清单与标签
    df = pd.read_csv(PARTICIPANTS_FILE, sep='\t')
    label_map = {'C': 0, 'A': 1, 'F': 2, 'HC': 0, 'MDD': 1, 'BD': 2}
    
    id_col = 'participant_id' if 'participant_id' in df.columns else df.columns[0]
    group_col = 'Group' if 'Group' in df.columns else 'group'
    
    # 获取所有存在的受试者文件夹并提取标签
    valid_subs = []
    valid_labels = []
    
    for _, row in df.iterrows():
        sub_id = str(row[id_col])
        if not sub_id.startswith('sub-'): sub_id = f"sub-{sub_id}"
        
        # 检查该受试者是否有对应的 EEG 数据目录
        if (INPUT_DIR / sub_id / "eeg").exists():
            group = row[group_col]
            if group in label_map:
                valid_subs.append(sub_id)
                valid_labels.append(label_map[group])

    valid_subs = np.array(valid_subs)
    valid_labels = np.array(valid_labels)
    
    print(f"📊 总计受试者: {len(valid_subs)} 名 | 类别分布: AD/FTD/HC 映射完成")

    # 2. 10 折分层划分逻辑 (按人头划分)
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2026)

    for fold, (train_idx, val_idx) in enumerate(skf.split(valid_subs, valid_labels)):
        print(f"\n🌀 正在处理第 {fold+1} / {NUM_FOLDS} 折...")
        
        fold_splits = {
            'train': valid_subs[train_idx],
            'val': valid_subs[val_idx]
        }

        # 3. 处理并保存每一折的数据
        for mode, id_list in fold_splits.items():
            x_list, y_list = [], []
            
            for sub_id in id_list:
                clean_id = sub_id.replace('sub-', '')
                # 获取该人的标签（从 df 重新获取以确保准确）
                group_label = df.loc[df[id_col].str.contains(clean_id), group_col].values[0]
                label = label_map[group_label]
                
                eeg_dir = INPUT_DIR / sub_id / "eeg"
                for set_file in eeg_dir.glob("*.set"):
                    try:
                        raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
                        raw.resample(256, verbose=False)
                        raw.pick('eeg')
                        
                        if len(raw.times) >= TARGET_LENGTH:
                            data = raw.get_data()[:, :TARGET_LENGTH]
                            x_list.append(data.astype(np.float32))
                            y_list.append(label)
                    except Exception: continue

            if x_list:
                # 文件夹结构：model_data_cv/fold_1/train/x.npy
                save_path = OUTPUT_ROOT / f"fold_{fold+1}" / mode
                save_path.mkdir(parents=True, exist_ok=True)
                np.save(save_path / "x.npy", np.stack(x_list))
                np.save(save_path / "y.npy", np.array(y_list, dtype=np.int64))
                print(f"  ✅ {mode} 导出: {len(x_list)} 样本")

    print(f"\n✨ 10 折交叉验证数据预处理全部完成！路径: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()