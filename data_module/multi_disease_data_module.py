"""
Multi-Disease EEG Data Module
多疾病EEG数据模块
"""

import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Dict, Optional
import numpy as np
from collections import Counter


class MultiDiseaseDataset(torch.utils.data.Dataset):
    """多疾病EEG数据集"""
    def __init__(self, data_path: str, disease_labels: Dict[str, int], 
                 transform=None, is_train: bool = True):
        self.data_path = data_path
        self.disease_labels = disease_labels
        self.transform = transform
        self.is_train = is_train
        
        # 加载数据索引
        self.samples = self._load_samples()
        self.class_counts = self._compute_class_counts()
        
    def _load_samples(self):
        """加载样本列表"""
        samples = []
        for disease_name, label in self.disease_labels.items():
            disease_path = os.path.join(self.data_path, disease_name)
            if os.path.exists(disease_path):
                for file_name in os.listdir(disease_path):
                    if file_name.endswith('.h5'):
                        samples.append({
                            'file_path': os.path.join(disease_path, file_name),
                            'disease_label': label,
                            'disease_name': disease_name
                        })
        return samples
    
    def _compute_class_counts(self):
        """计算各类别样本数量"""
        counts = Counter(sample['disease_label'] for sample in self.samples)
        return dict(counts)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载EEG数据（假设使用HDF5格式）
        import h5py
        with h5py.File(sample['file_path'], 'r') as f:
            eeg_data = f['eeg'][:]  # 假设数据存储在'eeg'键下
            
        # 转换为tensor
        eeg_tensor = torch.from_numpy(eeg_data).float()
        
        # 应用变换
        if self.transform:
            eeg_tensor = self.transform(eeg_tensor)
            
        return {
            'eeg': eeg_tensor,
            'label': sample['disease_label'],
            'disease_name': sample['disease_name'],
            'file_path': sample['file_path']
        }


class MultiDiseaseDataModule(pl.LightningDataModule):
    """多疾病数据模块"""
    def __init__(self, name: str = "multi_disease", cfg: Dict = None, 
                 disease_labels: Dict[str, int] = None, **kwargs):
        super().__init__()
        self.name = name
        self.cfg = cfg or {}
        self.disease_labels = disease_labels or {}
        self.data_path = os.path.join(os.environ.get('DATA_PATH', ''), f"{name}_data")
        
    def prepare_data(self):
        """验证数据存在性"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
            
        # 验证每个疾病目录存在
        for disease_name in self.disease_labels.keys():
            disease_path = os.path.join(self.data_path, disease_name)
            if not os.path.exists(disease_path):
                print(f"Warning: Disease directory not found: {disease_path}")
    
    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        # 数据增强
        train_transform = self._get_train_transform()
        val_transform = self._get_val_transform()
        
        if stage == "fit" or stage is None:
            # 训练和验证数据集
            full_dataset = MultiDiseaseDataset(
                data_path=os.path.join(self.data_path, "train"),
                disease_labels=self.disease_labels,
                transform=train_transform,
                is_train=True
            )
            
            # 按比例分割
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            self.val_dataset.dataset.transform = val_transform
            
        if stage == "test" or stage is None:
            self.test_dataset = MultiDiseaseDataset(
                data_path=os.path.join(self.data_path, "test"),
                disease_labels=self.disease_labels,
                transform=val_transform,
                is_train=False
            )
    
    def _get_train_transform(self):
        """训练数据增强"""
        import torchvision.transforms as transforms
        return transforms.Compose([
            # 时间轴扰动
            lambda x: self._random_time_shift(x, max_shift=0.1),
            # 幅度扰动
            lambda x: self._random_amplitude_scale(x, scale_range=(0.8, 1.2)),
            # 高斯噪声
            lambda x: self._add_gaussian_noise(x, noise_level=0.01),
        ])
    
    def _get_val_transform(self):
        """验证数据变换"""
        return None  # 不进行增强
    
    def _random_time_shift(self, x, max_shift=0.1):
        """随机时间平移"""
        if torch.rand(1) > 0.5:
            shift = int(max_shift * x.shape[-1] * (torch.rand(1).item() - 0.5))
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)
        return x
    
    def _random_amplitude_scale(self, x, scale_range=(0.8, 1.2)):
        """随机幅度缩放"""
        if torch.rand(1) > 0.5:
            scale = torch.empty(1).uniform_(scale_range[0], scale_range[1])
            x = x * scale
        return x
    
    def _add_gaussian_noise(self, x, noise_level=0.01):
        """添加高斯噪声"""
        if torch.rand(1) > 0.5:
            noise = torch.randn_like(x) * noise_level
            x = x + noise
        return x
    
    def train_dataloader(self):
        """训练数据加载器 - 支持类别平衡采样"""
        if self.cfg.get('class_balancing') == 'weighted_sampler':
            # 计算采样权重
            labels = [sample['label'] for sample in self.train_dataset.dataset.samples]
            class_counts = Counter(labels)
            weights = [1.0 / class_counts[label] for label in labels]
            sampler = WeightedRandomSampler(weights, len(weights))
            
            return DataLoader(
                self.train_dataset,
                batch_size=self.cfg.get('batch_size', 16),
                sampler=sampler,
                num_workers=self.cfg.get('num_workers', 4),
                pin_memory=True,
                persistent_workers=True
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.cfg.get('batch_size', 16),
                shuffle=True,
                num_workers=self.cfg.get('num_workers', 4),
                pin_memory=True,
                persistent_workers=True
            )
    
    def val_dataloader(self):
        """验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.get('batch_size', 16),
            shuffle=False,
            num_workers=self.cfg.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        """测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.get('batch_size', 16),
            shuffle=False,
            num_workers=self.cfg.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True
        )