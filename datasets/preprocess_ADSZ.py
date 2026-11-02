import numpy as np
from scipy.signal import butter, filtfilt
from pathlib import Path

# ==========================================
# 🎛️ 核心配置区
# ==========================================
DATA_DIR = Path(r"E:\ProgramFiles\PythonProject\BioFoundation-main\ADSZ\dataset\alzheimer")
OUTPUT_DIR = Path(r"E:\ProgramFiles\PythonProject\BioFoundation-main\ADSZ_processed_19ch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 128.0 
LOW_CUT = 0.5
HIGH_CUT = 45.0
# 🌟 新增：固定时间窗口大小 (1024 步 = 8 秒)
WINDOW_SIZE = 1024  

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=-1)
    return y

def extract_out_epochs(file_path):
    """提取单个 .out 文件，并将其切割为多个固定长度的 Epoch"""
    try:
        # 1. 加载数据并转置为 (19, Time)
        raw_data = np.loadtxt(file_path)
        continuous_data = raw_data.T 
        
        # 2. 在连续长序列上进行整体带通滤波 (防止切片后边缘产生滤波伪影)
        filtered_data = butter_bandpass_filter(continuous_data, LOW_CUT, HIGH_CUT, FS)
        
        time_steps = filtered_data.shape[1]
        epochs = []
        
        # 3. 🔪 滑动窗口切片引擎
        for start_idx in range(0, time_steps - WINDOW_SIZE + 1, WINDOW_SIZE):
            # 截取固定长度的切片: Shape (19, 1024)
            epoch = filtered_data[:, start_idx : start_idx + WINDOW_SIZE].copy()
            
            # 4. Z-score 归一化 (对每一个切片独立进行，消除局部基线漂移)
            for c in range(epoch.shape[0]):
                m = epoch[c].mean(keepdims=True)
                s = epoch[c].std(keepdims=True) + 1e-8
                epoch[c] = (epoch[c] - m) / s
                
            epochs.append(epoch)
            
        if len(epochs) == 0:
            print(f"   ⚠️ 跳过 {file_path.name}: 数据太短 ({time_steps} 步)，不足一个窗口。")
            return None
            
        # 返回形状为 (N_epochs, 19, 1024) 的张量
        return np.stack(epochs, axis=0).astype(np.float32)
        
    except Exception as e:
        print(f"   ❌ 解析失败 {file_path.name}: {e}")
        return None

def main():
    print("\n" + "🚀"*15)
    print("启动 ADSZ 定长切片流水线 (Window=1024)")
    print("🚀"*15 + "\n")
    
    all_X, all_y, all_sub_ids = [], [], []
    current_sub_id = 0
    
    category_map = {
        'Healthy': 0,
        'AD': 1      
    }
    
    for category_folder, label in category_map.items():
        folder_path = DATA_DIR / category_folder
        if not folder_path.exists():
            print(f"⚠️ 找不到文件夹: {folder_path}")
            continue
            
        out_files = list(folder_path.rglob("*.out"))
        print(f"📂 在 [{category_folder}] 中发现 {len(out_files)} 个样本文件")
        
        for out_file in out_files:
            # epochs 是一个形状为 (K, 19, 1024) 的矩阵，K 是这个文件能切出的块数
            epochs = extract_out_epochs(out_file)
            
            if epochs is not None:
                all_X.append(epochs)
                # 这个受试者切出了 K 块，就有 K 个相同的标签和 ID
                num_epochs = epochs.shape[0]
                all_y.extend([label] * num_epochs)
                all_sub_ids.extend([current_sub_id] * num_epochs)
                print(f"   ✅ {out_file.name} | 切出 {num_epochs} 个标准试次 | 标签: {category_folder}")
                current_sub_id += 1

    if len(all_X) > 0:
        # 🌟 修复点：使用 vstack 将一堆 (K, 19, 1024) 的矩阵垂直堆叠成完美的 (Total_K, 19, 1024)
        final_X = np.vstack(all_X) 
        final_y = np.array(all_y, dtype=np.int64)
        final_ids = np.array(all_sub_ids, dtype=np.int64)
        
        print("\n" + "="*50)
        print("📊 全局数据清洗战报:")
        print(f"   ├─ 特征张量 X shape: {final_X.shape}")
        print(f"   ├─ AD 患者切片数: {np.sum(final_y == 1)}")
        print(f"   ├─ HC 正常切片数: {np.sum(final_y == 0)}")
        print(f"   └─ 总有效文件数量: {len(np.unique(final_ids))}")
        print("="*50)
        
        np.save(OUTPUT_DIR / "x_19ch.npy", final_X)
        np.save(OUTPUT_DIR / "y_19ch.npy", final_y)
        np.save(OUTPUT_DIR / "sub_ids_19ch.npy", final_ids)
        print(f"\n✅ 数据已安全保存至: {OUTPUT_DIR}")
    else:
        print("\n❌ 提取失败，没有获得任何有效数据。")

if __name__ == "__main__":
    main()