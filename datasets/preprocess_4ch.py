import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from pathlib import Path

# ==========================================
# 🎛️ 核心配置区 (专为 Mendeley 数据集定制)
# ==========================================
DATA_DIR = Path(r"E:\ProgramFiles\PythonProject\BioFoundation-main\four_channels_dataset")
OUTPUT_DIR = Path(r"E:\ProgramFiles\PythonProject\BioFoundation-main\four_channels_processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 核心情报：Mendeley 数据集采样率为 200Hz
FS = 200  
LOW_CUT = 0.5
HIGH_CUT = 45.0

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """带通滤波，消除基线漂移和高频噪声"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=-1)
    return y

def extract_and_process_mat(file_path, label, subject_start_id):
    """深度破译并提取 .mat 文件中的脑电数据，并剔除噪声"""
    print(f"\n📂 正在解析 {file_path.name} (标签: {'AD' if label==1 else 'HC'})")
    mat_data = sio.loadmat(file_path)
    
    var_name = 'AD' if label == 1 else 'normal'
    if var_name not in mat_data:
        var_name = [k for k in mat_data.keys() if not k.startswith('_')][0]
    
    struct_array = mat_data[var_name]
    num_subjects = struct_array.shape[1]
    
    X_list, y_list, sub_ids_list = [], [], []
    total_epochs, dropped_epochs = 0, 0
    
    for i in range(num_subjects):
        try:
            epoch_data = struct_array[0, i]['epoch']
            noisy_idx = struct_array[0, i]['noisy']
            
            if isinstance(epoch_data, np.ndarray) and epoch_data.size == 1:
                epoch_data = epoch_data[0, 0]
                
            # 调整维度为 (Epochs, Channels, Time)
            if epoch_data.ndim == 3 and epoch_data.shape[2] > epoch_data.shape[0]:
                epoch_data = epoch_data.transpose(2, 0, 1)
            else:
                continue
                
            total_epochs += epoch_data.shape[0]
            
            # 🌟 核心剔除逻辑：利用医生的标注删除 noisy 切片
            if noisy_idx.size > 0:
                # MATLAB 索引是从 1 开始的，Python 从 0 开始，必须减 1
                noisy_idx = noisy_idx.flatten() - 1
                # 过滤掉可能越界的非法索引
                noisy_idx = noisy_idx[noisy_idx < epoch_data.shape[0]]
                dropped_epochs += len(noisy_idx)
                epoch_data = np.delete(epoch_data, noisy_idx, axis=0)
                
            # 1. 带通滤波
            filtered_data = butter_bandpass_filter(epoch_data, LOW_CUT, HIGH_CUT, FS)
            
            # 2. 临床级 Z-score 归一化 (沿时间轴独立归一化，严防数据泄露)
            for e in range(len(filtered_data)):
                m = filtered_data[e].mean(axis=-1, keepdims=True)
                s = filtered_data[e].std(axis=-1, keepdims=True) + 1e-8
                filtered_data[e] = (filtered_data[e] - m) / s
                
            X_list.append(filtered_data)
            y_list.extend([label] * len(filtered_data))
            sub_ids_list.extend([subject_start_id + i] * len(filtered_data))
            
        except Exception as e:
            print(f"  ⚠️ 读取受试者 {i} 失败: {e}")
            
    print(f"  ✅ 成功提取! 原始切片: {total_epochs} | 自动剔除噪声切片: {dropped_epochs}")
    return X_list, y_list, sub_ids_list

def main():
    print("🚀 启动 Mendeley 4 通道数据集全自动化清洗流水线...")
    
    all_X, all_y, all_sub_ids = [], [], []
    current_sub_id = 0
    
    # 1. 处理正常人数据 (HC)
    normal_mat = DATA_DIR / "normal.mat"
    if normal_mat.exists():
        X, y, ids = extract_and_process_mat(normal_mat, label=0, subject_start_id=current_sub_id)
        all_X.extend(X)
        all_y.extend(y)
        all_sub_ids.extend(ids)
        if len(ids) > 0:
            current_sub_id = max(ids) + 1
    
    # 2. 处理 AD 患者数据
    ad_mat = DATA_DIR / "AD.mat"
    if ad_mat.exists():
        X, y, ids = extract_and_process_mat(ad_mat, label=1, subject_start_id=current_sub_id)
        all_X.extend(X)
        all_y.extend(y)
        all_sub_ids.extend(ids)
        
    # 3. 合并与保存
    if len(all_X) > 0:
        final_X = np.vstack(all_X).astype(np.float32)
        final_y = np.array(all_y, dtype=np.int64)
        final_ids = np.array(all_sub_ids, dtype=np.int64)
        
        print("\n" + "="*50)
        print("📊 预处理数据全景报告:")
        print(f"   ├─ 特征张量 X shape: {final_X.shape} (切片数, 通道数, 时间点)")
        print(f"   ├─ 标签张量 y shape: {final_y.shape}")
        print(f"   ├─ AD 患者切片数: {np.sum(final_y == 1)}")
        print(f"   ├─ HC 正常切片数: {np.sum(final_y == 0)}")
        print(f"   └─ 总受试者人数: {len(np.unique(final_ids))}")
        print("="*50)
        
        np.save(OUTPUT_DIR / "x_4ch.npy", final_X)
        np.save(OUTPUT_DIR / "y_4ch.npy", final_y)
        np.save(OUTPUT_DIR / "sub_ids_4ch.npy", final_ids)
        print(f"\n✅ 数据已清洗完毕并安全保存至: {OUTPUT_DIR}")
    else:
        print("\n❌ 提取失败，请检查文件是否存在。")

if __name__ == "__main__":
    main()