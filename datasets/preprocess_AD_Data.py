import scipy.io as sio
import numpy as np
from scipy.signal import butter, filtfilt
from pathlib import Path

# ==========================================
# 🎛️ 核心配置区
# ==========================================
# 国际标准的 19 通道顺序 (严格对齐你的 Teacher 模型)
# 注意：有些设备叫 T3/T4/T5/T6，有些叫 T7/T8/P7/P8，这里做了兼容
STANDARD_19_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
    'T3', 'C3', 'Cz', 'C4', 'T4', 
    'T5', 'P3', 'Pz', 'P4', 'T6', 
    'O1', 'O2'
]

# 兼容别名映射字典
CHANNEL_ALIAS = {
    'T7': 'T3', 'T8': 'T4', 
    'P7': 'T5', 'P8': 'T6'
}

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

def extract_fieldtrip_data(file_path):
    print(f"\n📂 正在解析 FieldTrip 格式数据: {file_path.name}")
    mat_data = sio.loadmat(file_path)
    data_struct = mat_data['data'][0, 0]
    
    # 1. 提取采样率
    fs = float(data_struct['fsample'][0, 0])
    print(f"   ├─ 📡 采样率 (fsample): {fs} Hz")
    
    # 2. 提取并清理通道名称
    raw_labels = data_struct['label']
    channel_names = []
    for lbl in raw_labels:
        # 处理 MATLAB 嵌套的 cell 字符串
        name = str(lbl[0][0]).strip()
        # 处理别名 (如把 T7 映射成 T3)
        name = CHANNEL_ALIAS.get(name, name)
        channel_names.append(name)
        
    print(f"   ├─ 🧠 原始数据集包含 {len(channel_names)} 个通道")
    
    # 3. 寻找 19 通道的精确索引
    channel_indices = []
    missing_channels = []
    for target_ch in STANDARD_19_CHANNELS:
        # 忽略大小写进行匹配
        matched = False
        for i, raw_ch in enumerate(channel_names):
            if target_ch.lower() == raw_ch.lower():
                channel_indices.append(i)
                matched = True
                break
        if not matched:
            missing_channels.append(target_ch)
            
    if missing_channels:
        print(f"   ❌ 致命错误: 找不到以下必须的通道: {missing_channels}")
        return None, None
        
    print(f"   ├─ ✅ 成功锁定 19 标准通道索引！")
    
    # 4. 提取 Trial 数据并进行空间降维和滤波
    # FieldTrip 的 trial 通常是 1 x N 的 cell 数组，每个 cell 是 (Channels, Time)
    trials_cell = data_struct['trial'][0]
    num_trials = len(trials_cell)
    print(f"   ├─ 🧩 发现 {num_trials} 个脑电切片 (Trials)")
    
    processed_trials = []
    
    for i in range(num_trials):
        # 提取单个切片，并只保留需要的 19 个通道
        # shape: (19, Time)
        epoch_data = trials_cell[i][channel_indices, :] 
        
        # 带通滤波
        filtered_data = butter_bandpass_filter(epoch_data, LOW_CUT, HIGH_CUT, fs)
        
        # Z-score 归一化 (沿时间轴)
        for c in range(filtered_data.shape[0]):
            m = filtered_data[c].mean(keepdims=True)
            s = filtered_data[c].std(keepdims=True) + 1e-8
            filtered_data[c] = (filtered_data[c] - m) / s
            
        processed_trials.append(filtered_data)
        
    # 将列表转换为高维张量: (Trials, Channels, Time)
    final_X = np.array(processed_trials, dtype=np.float32)
    print(f"   └─ 🎯 预处理张量构建完成: {final_X.shape}")
    
    return final_X, fs

if __name__ == "__main__":
    # 🌟 你的数据文件绝对路径 🌟
    mat_file = Path(r"E:\ProgramFiles\PythonProject\BioFoundation-main\AD_Data\preproctrials01.mat")
    
    if mat_file.exists():
        X, fs = extract_fieldtrip_data(mat_file)
        if X is not None:
            print("\n🚀 数据清洗流线测试成功，随时可以送入 19 通道 Spatial-MSCNN 模型！")
    else:
        print(f"⚠️ 找不到文件: {mat_file}")