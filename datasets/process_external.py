import os
import mne
import numpy as np
from pathlib import Path

# 忽略 MNE 加载数据时的一些常规警告
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def load_patient_folder(folder_path, target_channels, sfreq):
    """从零散的 txt 文件缝合出 MNE Raw 对象"""
    data_list = []
    
    for ch in target_channels:
        # 处理大小写兼容问题，比如文件可能是 FZ.txt, Fz.txt 或者 fz.txt
        possible_names = [f"{ch}.txt", f"{ch.upper()}.txt", f"{ch.capitalize()}.txt", f"{ch.lower()}.txt"]
        file_path = None
        for name in possible_names:
            temp_path = os.path.join(folder_path, name)
            if os.path.exists(temp_path):
                file_path = temp_path
                break
                
        if file_path:
            ch_data = np.loadtxt(file_path)
            data_list.append(ch_data)
        else:
            print(f"    ❌ 找不到通道: {ch}，请检查文件夹内容！")
            return None
            
    # 堆叠成矩阵: (19通道, 时间步)
    data_matrix = np.vstack(data_list)
    
    # ⚠️ 极度关键：临床 txt 导出一般为微伏 (uV)，需转为伏特 (V) 以适配 MNE 运算
    data_matrix = data_matrix * 1e-6 
    
    # 捏造头文件并生成 Raw 对象
    info = mne.create_info(ch_names=target_channels, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data_matrix, info, verbose=False)
    
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False, on_missing='ignore')
    except:
        pass
        
    return raw

def main():
    # ==========================================
    # 1. 📂 核心配置区 
    # ==========================================
    EXTERNAL_RAW_DIR = Path(r"E:\ProgramFiles\PythonProject\BioFoundation-main\external_raw")
    OUTPUT_DIR = Path(r"E:\ProgramFiles\PythonProject\BioFoundation-main\external_test_data")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 🌟 修正：从你上传的 txt 来看，768点对应3秒，真实采样率是 256Hz
    SFREQ = 256 
    
    TARGET_LENGTH = 1280       # 保持 5秒 窗口长度
    STRIDE = 1280 * 3          # 保持 15秒 稀疏步长
    REJECT_THRESHOLD = 500e-6  # 极值剔除
    
    TARGET_CHANNELS = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
    ]
    
    print("🚀 开始洗炼外部独立测试集 (TXT to MNE Pipeline)...")

    # ==========================================
    # 🌟 核心修复：已将 'HC/Eyes_closed' 改为 'Healthy/Eyes_closed'
    # ==========================================
    class_mapping = {'Healthy/Eyes_closed': 0, 'AD/Eyes_closed': 1}
    
    x_list, y_list, sub_ids = [], [], []

    # ==========================================
    # 2. 🌀 遍历文件夹与提取特征
    # ==========================================
    for class_name, label in class_mapping.items():
        class_dir = EXTERNAL_RAW_DIR / class_name
        if not class_dir.exists():
            print(f"⚠️ 找不到类别文件夹: {class_dir}，跳过该类别。")
            continue
            
        print(f"\n📂 正在处理 [{class_name}] 类别的数据...")
        # 遍历该类别下的所有病人文件夹
        for patient_folder in class_dir.iterdir():
            if not patient_folder.is_dir(): continue
            
            patient_id = patient_folder.name
            print(f"  👉 缝合患者: {patient_id}")
            
            # 读取 txt 并转为 Raw
            raw = load_patient_folder(patient_folder, TARGET_CHANNELS, SFREQ)
            if raw is None: continue
            
            try:
                # 信号清洗与对齐
                raw.resample(256, verbose=False) 
                raw.filter(l_freq=0.5, h_freq=45.0, fir_design='firwin', verbose=False)
                raw.notch_filter(freqs=50.0, verbose=False)
                
                data = raw.get_data()
                total_samples = data.shape[1]
                
                # ==========================================
                # 🌟 教授级补丁：智能镜像填充 (Symmetric Padding)
                # 解决数据过短的致死 Bug
                # ==========================================
                if total_samples < TARGET_LENGTH:
                    pad_length = TARGET_LENGTH - total_samples
                    # 像照镜子一样补齐缺失的时间步
                    data = np.pad(data, pad_width=((0, 0), (0, pad_length)), mode='symmetric')
                    total_samples = data.shape[1]
                
                # 滑动窗口切片 (此时长度已被强行拉到了标准的 1280 点以上)
                valid_chunks_count = 0
                for start in range(0, total_samples - TARGET_LENGTH + 1, STRIDE):
                    chunk = data[:, start : start + TARGET_LENGTH]
                    if np.max(np.abs(chunk)) < REJECT_THRESHOLD:
                        x_list.append(chunk.astype(np.float32))
                        y_list.append(label)
                        sub_ids.append(patient_id)
                        valid_chunks_count += 1
                
                if valid_chunks_count == 0:
                    print(f"    ⚠️ {patient_id} 切片全部被极值剔除拦截！")
                        
            except Exception as e:
                print(f"    ⚠️ 处理 {patient_id} 时发生错误: {e}")

    # ==========================================
    # 3. 💾 导出标准格式
    # ==========================================
    if x_list:
        np.save(OUTPUT_DIR / "x.npy", np.stack(x_list))
        np.save(OUTPUT_DIR / "y.npy", np.array(y_list, dtype=np.int64))
        np.save(OUTPUT_DIR / "sub_ids.npy", np.array(sub_ids, dtype=str))
        
        print("\n" + "="*50)
        print(f"✅ 外部测试集洗炼完成！数据已保存至: {OUTPUT_DIR}")
        print(f"   └─ 包含受试者: {len(set(sub_ids))} 名")
        print(f"   └─ 有效切片数: {len(x_list)} 个")
        print("="*50)
        print("🚀 下一步：直接运行 test_external.py 开启最残酷的泛化检验！")
    else:
        print("❌ 未能提取出任何有效数据，请检查文件路径和内部结构！")

if __name__ == "__main__":
    main()