"""
LUNA Multi-Disease Data Module for HDF5 Format
基于LUNA的多疾病数据模块（HDF5格式）
"""

import os
import torch
import h5py
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Dict, Optional, List
import numpy as np
from collections import Counter
import json


class LUNAHDF5Dataset(torch.utils.data.Dataset):
    """LUNA HDF5数据集"""
    def __init__(self, hdf5_file: str, disease_labels: Dict[str, int], 
                 transform=None, is_train: bool = True):
        self.hdf5_file = hdf5_file
        self.disease_labels = disease_labels
        self.transform = transform
        self.is_train = is_train
        self.samples = []
        
        # 加载HDF5索引
        self._load_hdf5_index()
        
    def _load_hdf5_index(self):
        """从HDF5文件加载样本索引"""
        with h5py.File(self.hdf5_file, 'r') as f:
            # 假设HDF5结构为:
            # /data/disease_name/sample_id -> EEG数据
            # /labels/sample_id -> 疾病标签
            # /channel_locations/sample_id -> 通道位置
            
            for disease_name, label in self.disease_labels.items():
                if disease_name in f:
                    disease_group = f[disease_name]
                    for sample_id in disease_group.keys():
                        self.samples.append({
                            'sample_id': sample_id,
                            'disease_name': disease_name,
                            'disease_label': label,
                            'data_path': f"{disease_name}/{sample_id}"
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        with h5py.File(self.hdf5_file, 'r') as f:
            # 读取EEG数据
            eeg_data = f[sample['data_path']][:]  # [C, T]
            
            # 读取通道位置（如果存在）
            channel_loc_path = f"channel_locations/{sample['sample_id']}"
            if channel_loc_path in f:
                channel_locations = f[channel_loc_path][:]
            else:
                # 如果没有通道位置信息，生成默认位置
                C = eeg_data.shape[0]
                channel_locations = self._generate_default_positions(C)
            
            # 转换为tensor
            eeg_tensor = torch.from_numpy(eeg_data).float()
            channel_loc_tensor = torch.from_numpy(channel_locations).float()
            
            # 应用变换
            if self.transform:
                eeg_tensor = self.transform(eeg_tensor)
        
        return {
            'eeg': eeg_tensor,
            'channel_locations': channel_loc_tensor,
            'label': sample['disease_label'],
            'disease_name': sample['disease_name'],
            'sample_id': sample['sample_id']
        }
    
    def _generate_default_positions(self, num_channels: int) -> np.ndarray:
        """生成默认通道位置（10-20系统近似）"""
        # 简化的10-20系统位置生成
        positions = []
        for i in range(num_channels):
            # 生成近似球面坐标
            theta = 2 * np.pi * i / num_channels
            phi = np.pi * (i % 4) / 4
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            positions.append([x, y, z])
        return np.array(positions)


class LUNAMultiDiseaseDataModule(pl.LightningDataModule):
    """LUNA多疾病数据模块"""
    def __init__(self, name: str = "luna_multi_disease", cfg: Dict = None, 
                 disease_labels: Dict[str, int] = None, **kwargs):
        super().__init__()
        self.name = name
        self.cfg = cfg or {}
        self.disease_labels = disease_labels or {}
        self.data_path = os.path.join(os.environ.get('DATA_PATH', ''), f"{name}_data")
        
    def prepare_data(self):
        """验证HDF5数据文件存在性"""
        required_files = ["train.h5", "val.h5", "test.h5"]
        
        for file_name in required_files:
            file_path = os.path.join(self.data_path, file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"HDF5 file not found: {file_path}")
            
            # 验证HDF5文件结构
            self._validate_hdf5_structure(file_path)
    
    def _validate_hdf5_structure(self, hdf5_file: str):
        """验证HDF5文件结构"""
        try:
            with h5py.File(hdf5_file, 'r') as f:
                # 检查疾病组是否存在
                for disease_name in self.disease_labels.keys():
                    if disease_name not in f:
                        print(f"Warning: Disease group '{disease_name}' not found in {hdf5_file}")
                        
                # 检查基本结构
                expected_groups = ['channel_locations']
                for group in expected_groups:
                    if group not in f:
                        print(f"Info: Group '{group}' not found in {hdf5_file}")
                        
        except Exception as e:
            raise ValueError(f"Invalid HDF5 file structure in {hdf5_file}: {e}")
    
    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        train_transform = self._get_train_transform()
        val_transform = self._get_val_transform()
        
        if stage == "fit" or stage is None:
            # 训练和验证数据集
            train_file = os.path.join(self.data_path, "train.h5")
            full_dataset = LUNAHDF5Dataset(
                hdf5_file=train_file,
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
            test_file = os.path.join(self.data_path, "test.h5")
            self.test_dataset = LUNAHDF5Dataset(
                hdf5_file=test_file,
                disease_labels=self.disease_labels,
                transform=val_transform,
                is_train=False
            )
    
    def _get_train_transform(self):
        """训练数据增强"""
        return lambda x: self._apply_augmentations(x)
    
    def _get_val_transform(self):
        """验证数据变换"""
        return None
    
    def _apply_augmentations(self, x: torch.Tensor) -> torch.Tensor:
        """应用数据增强"""
        if torch.rand(1) > 0.5:
            # 时间轴扰动
            max_shift = int(0.1 * x.shape[-1])
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)
        
        if torch.rand(1) > 0.5:
            # 幅度缩放
            scale = torch.empty(1).uniform_(0.8, 1.2)
            x = x * scale
        
        if torch.rand(1) > 0.5:
            # 添加噪声
            noise = torch.randn_like(x) * 0.01
            x = x + noise
            
        return x
    
    def train_dataloader(self):
        """训练数据加载器"""
        if self.cfg.get('class_balancing') == 'weighted_sampler':
            # 计算类别权重
            labels = []
            with h5py.File(os.path.join(self.data_path, "train.h5"), 'r') as f:
                for disease_name, label in self.disease_labels.items():
                    if disease_name in f:
                        labels.extend([label] * len(f[disease_name]))
            
            class_counts = Counter(labels)
            weights = [1.0 / class_counts[label] for label in labels]
            sampler = WeightedRandomSampler(weights, len(weights))
            
            return DataLoader(
                self.train_dataset,
                batch_size=self.cfg.get('batch_size', 16),
                sampler=sampler,
                num_workers=self.cfg.get('num_workers', 4),
                pin_memory=True,
                persistent_workers=True,
                collate_fn=self._collate_fn
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.cfg.get('batch_size', 16),
                shuffle=True,
                num_workers=self.cfg.get('num_workers', 4),
                pin_memory=True,
                persistent_workers=True,
                collate_fn=self._collate_fn
            )
    
    def val_dataloader(self):
        """验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.get('batch_size', 16),
            shuffle=False,
            num_workers=self.cfg.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self):
        """测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.get('batch_size', 16),
            shuffle=False,
            num_workers=self.cfg.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """自定义批处理函数"""
        # 找到最大通道数和时间长度
        max_channels = max(item['eeg'].shape[0] for item in batch)
        max_time = max(item['eeg'].shape[1] for item in batch)
        
        batch_size = len(batch)
        
        # 初始化填充张量
        eeg_batch = torch.zeros(batch_size, max_channels, max_time)
        loc_batch = torch.zeros(batch_size, max_channels, 3)
        labels = torch.zeros(batch_size, dtype=torch.long)
        
        for i, item in enumerate(batch):
            C, T = item['eeg'].shape
            
            # 填充EEG数据
            eeg_batch[i, :C, :T] = item['eeg']
            
            # 填充通道位置
            loc_batch[i, :C, :] = item['channel_locations']
            
            # 标签
            labels[i] = item['label']
        
        return {
            'eeg': eeg_batch,
            'channel_locations': loc_batch,
            'labels': labels,
            'disease_names': [item['disease_name'] for item in batch],
            'sample_ids': [item['sample_id'] for item in batch]
        }


# HDF5数据转换工具
class HDF5Converter:
    """将原始数据转换为LUNA多疾病HDF5格式"""
    
    @staticmethod
    def convert_raw_to_hdf5(raw_data_dir: str, output_hdf5: str, 
                          disease_labels: Dict[str, int]):
        """
        将原始数据目录转换为HDF5格式
        """
        with h5py.File(output_hdf5, 'w') as f:
            for disease_name in disease_labels.keys():
                disease_dir = os.path.join(raw_data_dir, disease_name)
                if not os.path.exists(disease_dir):
                    continue
                    
                disease_group = f.create_group(disease_name)
                channel_loc_group = f.create_group('channel_locations')
                
                for sample_file in os.listdir(disease_dir):
                    if sample_file.endswith(('.npy', '.mat', '.edf')):
                        sample_id = os.path.splitext(sample_file)[0]
                        
                        # 加载原始数据
                        eeg_data = HDF5Converter._load_sample_data(
                            os.path.join(disease_dir, sample_file)
                        )
                        
                        # 保存EEG数据
                        disease_group.create_dataset(
                            sample_id, 
                            data=eeg_data,
                            compression='gzip',
                            compression_opts=9
                        )
                        
                        # 生成并保存通道位置
                        C = eeg_data.shape[0]
                        channel_positions = HDF5Converter._generate_channel_positions(C)
                        channel_loc_group.create_dataset(
                            sample_id,
                            data=channel_positions,
                            compression='gzip',
                            compression_opts=9
                        )
    
    @staticmethod
    def _load_sample_data(file_path: str) -> np.ndarray:
        """加载单个样本数据"""
        if file_path.endswith('.npy'):
            return np.load(file_path)
        elif file_path.endswith('.mat'):
            import scipy.io
            mat_data = scipy.io.loadmat(file_path)
            # 假设EEG数据在'mat_data'键下
            return mat_data.get('eeg_data', mat_data[list(mat_data.keys())[0]])
        else:
            # 处理EDF等其他格式
            raise NotImplementedError(f"File format not supported: {file_path}")
    
    @staticmethod
    def _generate_channel_positions(num_channels: int) -> np.ndarray:
        """生成通道位置坐标"""
        # 实现10-20系统或其他标准布局
        # 这里返回简化的三维坐标
        positions = []
        for i in range(num_channels):
            theta = 2 * np.pi * i / num_channels
            phi = np.pi * (i % 8) / 8
            x = 0.5 * np.sin(phi) * np.cos(theta)
            y = 0.5 * np.sin(phi) * np.sin(theta)
            z = 0.5 * np.cos(phi)
            positions.append([x, y, z])
        return np.array(positions)