import numpy as np
from scipy.linalg import fractional_matrix_power
from pathlib import Path

def euclidean_alignment(X):
    """
    实现 T-BME 论文中的 Session-wise Euclidean Alignment (EA)
    输入 X: shape (N_trials, Channels, Time)
    输出 X_aligned: 对齐后的数据
    """
    print(f"   ├─ 正在计算 {X.shape[0]} 个切片的协方差矩阵...")
    N, C, T = X.shape
    covs = np.zeros((N, C, C))
    
    # 计算每个 trial 的协方差矩阵
    for i in range(N):
        covs[i] = np.cov(X[i])
        
    # 计算平均协方差矩阵 R-bar
    mean_cov = np.mean(covs, axis=0)
    
    # 计算 R-bar 的 -1/2 次幂 (加上极其微小的对角线正则项防止奇异矩阵)
    mean_cov += np.eye(C) * 1e-6 
    R_inv_half = fractional_matrix_power(mean_cov, -0.5)
    
    # 这里必须取实部，消除计算误差带来的虚数
    R_inv_half = np.real(R_inv_half)
    
    print(f"   ├─ 正在执行几何空间变换投影...")
    X_aligned = np.zeros_like(X)
    for i in range(N):
        # 核心：公式 (3) 的投影操作
        X_aligned[i] = R_inv_half @ X[i]
        
    return X_aligned

def main():
    # 指向你刚才清洗好的数据目录
    DATA_DIR = Path(r"E:\ProgramFiles\PythonProject\BioFoundation-main\four_channels_processed")
    X_path = DATA_DIR / "x_4ch.npy"
    y_path = DATA_DIR / "y_4ch.npy"
    id_path = DATA_DIR / "sub_ids_4ch.npy"
    
    if not X_path.exists():
        print("❌ 找不到 4 通道数据，请检查路径。")
        return
        
    print("🚀 启动顶刊级 Euclidean Alignment (EA) 几何对齐引擎...")
    X = np.load(X_path)
    
    # 执行欧几里得对齐 (EA)
    X_aligned = euclidean_alignment(X)
    
    # 保存对齐后的数据
    out_X_path = DATA_DIR / "x_4ch_EA.npy"
    np.save(out_X_path, X_aligned)
    
    print(f"✅ EA 对齐完成！")
    print(f"   └─ 对齐后张量形状: {X_aligned.shape}")
    print(f"   └─ 已保存至: {out_X_path}")

if __name__ == "__main__":
    main()