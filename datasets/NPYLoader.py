import torch
import numpy as np
import os
from torch.utils.data import Dataset

class NPYLoader(Dataset):
    """
    专门针对 .npy 格式优化的脑电数据加载器
    适配 HC/MDD/BD 或 C/A/F 三分类任务
    """
    def __init__(self, data_dir, mode='train'):
        # 路径示例: model_data/train/
        self.path = os.path.join(data_dir, mode)
        
        # 直接加载到内存 (88个样本约几百MB，内存完全跑得快)
        self.X = np.load(os.path.join(self.path, 'X.npy'))
        self.y = np.load(os.path.join(self.path, 'y.npy'))
        
        print(f"✅ {mode} 加载完成: {self.X.shape[0]} 样本, 形状 {self.X.shape[1:]}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # 转换成 Torch 张量
        # X 维度: (19, 1280) -> (Channels, Time)
        x = torch.from_numpy(self.X[index]).float()
        y = torch.tensor(self.y[index], dtype=torch.long)
        
        return x, y