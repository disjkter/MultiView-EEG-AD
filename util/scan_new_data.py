import scipy.io as sio
import numpy as np
from pathlib import Path

def scan_mat(file_path):
    print("\n" + "🔍"*20)
    print(f"🚀 启动新数据集深度扫描引擎: {file_path.name}")
    print("🔍"*20 + "\n")
    
    try:
        mat_data = sio.loadmat(file_path)
        print("📦 .mat 文件内包含的有效变量:")
        for key, value in mat_data.items():
            # 过滤掉 MATLAB 内部的隐藏变量
            if not key.startswith('__'):
                if isinstance(value, np.ndarray):
                    print(f"   ├─ 变量名: '{key}' | 类型: {value.dtype} | 维度 (Shape): {value.shape}")
                else:
                    print(f"   ├─ 变量名: '{key}' | 类型: {type(value)}")
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        
    print("\n✅ 扫描完毕！")

if __name__ == "__main__":
    # 已经为你替换为精准的绝对路径
    mat_file = Path(r"E:\ProgramFiles\PythonProject\BioFoundation-main\AD_Data\preproctrials01.mat") 
    
    if mat_file.exists():
        scan_mat(mat_file)
    else:
        print(f"⚠️ 找不到文件 {mat_file}，请确保路径完全正确。")