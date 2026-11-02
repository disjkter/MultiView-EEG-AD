import numpy as np
from pathlib import Path

for mode in ['train', 'val', 'test']:
    y = np.load(f"model_data/{mode}/y.npy")
    # 统计 0(C), 1(A), 2(F) 的数量
    counts = np.bincount(y, minlength=3)
    print(f"📊 {mode} 集分布 -> 健康(C): {counts[0]}, 阿尔兹海默(A): {counts[1]}, 额颞叶痴呆(F): {counts[2]}")