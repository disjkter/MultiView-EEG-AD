#*----------------------------------------------------------------------------*
#* Copyright (C) 2025 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Direct from .set files loader                                    *
#*----------------------------------------------------------------------------*

import torch
import numpy as np
from pathlib import Path
import mne
import logging
from collections import deque

logger = logging.getLogger(__name__)

class EEGSetDataset(torch.utils.data.Dataset):
    """
    直接从 BIDS 格式的 eeg-data 文件夹加载 .set 文件的数据集
    
    用法：
        dataset = EEGSetDataset(
            eeg_data_path='eeg-data',
            split='train',  # 'train', 'val', 'test'
            cache_size=100,
            use_cache=True,
            normalize=True
        )
    """
    
    def __init__(self, eeg_data_path='eeg-data', split='train', 
                 cache_size=100, use_cache=True, normalize=True,
                 target_length=1280, target_sfreq=256):
        """
        Args:
            eeg_data_path: eeg-data 文件夹路径
            split: 'train', 'val', 'test'
            cache_size: 缓存大小（最近访问的样本数）
            use_cache: 是否启用缓存
            normalize: 是否标准化数据
            target_length: 目标时间步长（默认 1280，即 5 秒）
            target_sfreq: 目标采样率（默认 256 Hz）
        """
        self.eeg_data_path = Path(eeg_data_path)
        self.split = split
        self.normalize = normalize
        self.target_length = target_length
        self.target_sfreq = target_sfreq
        self.use_cache = use_cache
        self.cache_size = cache_size
        
        if self.use_cache:
            self.cache = {}
            self.cache_queue = deque(maxlen=cache_size)
        
        # 加载参与者信息
        self._load_subject_info()
        
        # 构建样本列表
        self._build_sample_list()
        
        logger.info(f"加载 {len(self.samples)} 个 {split} 集样本")
    
    def _load_subject_info(self):
        """加载参与者信息和标签"""
        participants_file = self.eeg_data_path / 'participants.tsv'
        
        if not participants_file.exists():
            raise FileNotFoundError(f"找不到 {participants_file}")
        
        self.subject_info = {}
        self.group_to_label = {'A': 0, 'F': 1, 'C': 2}  # Alzheimer, FTD, Control(Healthy)
        
        with open(participants_file, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:  # 跳过表头
                    continue
                parts = line.strip().split('\t')
                subject_id = parts[0]  # e.g., 'sub-001'
                group = parts[1] if len(parts) > 1 else 'C'
                
                self.subject_info[subject_id] = {
                    'group': group,
                    'label': self.group_to_label.get(group, 2)
                }
    
    def _build_sample_list(self):
        """构建样本列表"""
        derivatives_path = self.eeg_data_path / 'derivatives'
        
        if not derivatives_path.exists():
            raise FileNotFoundError(f"找不到 {derivatives_path}")
        
        all_samples = []
        
        # 扫描所有 sub-* 文件夹
        for subject_dir in sorted(derivatives_path.glob('sub-*')):
            subject_id = subject_dir.name
            
            if subject_id not in self.subject_info:
                logger.warning(f"跳过 {subject_id}: 未在 participants.tsv 中找到")
                continue
            
            label = self.subject_info[subject_id]['label']
            
            # 找到 EEG 文件
            eeg_dir = subject_dir / 'eeg'
            if eeg_dir.exists():
                for set_file in eeg_dir.glob('*.set'):
                    all_samples.append({
                        'file_path': str(set_file),
                        'subject_id': subject_id,
                        'label': label
                    })
        
        if not all_samples:
            raise ValueError(f"在 {derivatives_path} 中找不到任何 .set 文件")
        
        # 按比例划分 train/val/test
        # 按组别均衡划分
        grouped_samples = {}
        for sample in all_samples:
            label = sample['label']
            if label not in grouped_samples:
                grouped_samples[label] = []
            grouped_samples[label].append(sample)
        
        self.samples = []
        for label, group_samples in grouped_samples.items():
            n = len(group_samples)
            
            if self.split == 'train':
                # 60% 训练
                indices = list(range(int(n * 0.6)))
            elif self.split == 'val':
                # 20% 验证
                indices = list(range(int(n * 0.6), int(n * 0.8)))
            elif self.split == 'test':
                # 20% 测试
                indices = list(range(int(n * 0.8), n))
            else:
                raise ValueError(f"未知的 split: {self.split}")
            
            self.samples.extend([group_samples[i] for i in indices])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        # 检查缓存
        if self.use_cache and index in self.cache:
            return self.cache[index]
        
        sample = self.samples[index]
        file_path = sample['file_path']
        label = sample['label']
        
        # 读取 .set 文件
        eeg_data = self._load_set_file(file_path)
        
        # 转换为张量
        eeg_tensor = torch.from_numpy(eeg_data).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        result = (eeg_tensor, label_tensor)
        
        # 缓存
        if self.use_cache:
            self.cache[index] = result
            self.cache_queue.append(index)
        
        return result
    
    def _load_set_file(self, file_path):
        """
        加载 .set 文件并进行预处理
        
        Returns:
            (num_channels, target_length) numpy array
        """
        try:
            # 使用 MNE 读取
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
            
            # 获取采样率
            sfreq = raw.info['sfreq']
            
            # 选择仅 EEG 通道
            raw.pick_types(eeg=True)
            
            # 获取数据
            data = raw.get_data()  # (num_channels, num_samples)
            
            # 重采样到目标采样率
            if sfreq != self.target_sfreq:
                # 计算重采样因子
                resample_factor = self.target_sfreq / sfreq
                n_samples_new = int(data.shape[1] * resample_factor)
                
                # 使用 scipy 的重采样
                from scipy import signal
                indices = np.arange(n_samples_new) / resample_factor
                data = np.array([
                    np.interp(indices, np.arange(data.shape[1]), data[ch])
                    for ch in range(data.shape[0])
                ])
            
            # 高通滤波（移除直流分量）
            if data.shape[1] > 0:
                # 简单的高通滤波：减去均值
                data = data - data.mean(axis=1, keepdims=True)
            
            # 截断或填充到目标长度
            if data.shape[1] > self.target_length:
                # 截断到中间部分
                start_idx = (data.shape[1] - self.target_length) // 2
                data = data[:, start_idx:start_idx + self.target_length]
            elif data.shape[1] < self.target_length:
                # 填充（零填充）
                pad_width = self.target_length - data.shape[1]
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left
                data = np.pad(data, ((0, 0), (pad_left, pad_right)), mode='constant')
            
            # 标准化（零均值，单位方差）
            if self.normalize:
                mean = data.mean(axis=1, keepdims=True)
                std = data.std(axis=1, keepdims=True)
                std[std == 0] = 1  # 避免除零
                data = (data - mean) / std
            
            # 如果通道数不是 19，进行调整
            if data.shape[0] != 19:
                if data.shape[0] < 19:
                    # 填充通道
                    pad_width = 19 - data.shape[0]
                    data = np.vstack([data, np.zeros((pad_width, data.shape[1]))])
                else:
                    # 选择前 19 个通道
                    data = data[:19, :]
            
            return data
        
        except Exception as e:
            logger.error(f"加载 {file_path} 时出错: {e}")
            # 返回零张量作为后备
            return np.zeros((19, self.target_length))


class EEGSetDataModule:
    """
    PyTorch Lightning DataModule for direct .set file loading
    
    直接从 eeg-data 原始格式加载数据的 DataModule
    """
    
    def __init__(self, eeg_data_path='eeg-data', batch_size=16, num_workers=4, 
                 normalize=True, cache_size=100, use_cache=True):
        """
        Args:
            eeg_data_path: eeg-data 文件夹路径
            batch_size: 批次大小
            num_workers: 数据加载工作线程数
            normalize: 是否标准化
            cache_size: 缓存大小
            use_cache: 是否使用缓存
        """
        self.eeg_data_path = eeg_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.cache_size = cache_size
        self.use_cache = use_cache
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """创建数据集"""
        self.train_dataset = EEGSetDataset(
            eeg_data_path=self.eeg_data_path,
            split='train',
            cache_size=self.cache_size,
            use_cache=self.use_cache,
            normalize=self.normalize
        )
        
        self.val_dataset = EEGSetDataset(
            eeg_data_path=self.eeg_data_path,
            split='val',
            cache_size=self.cache_size,
            use_cache=self.use_cache,
            normalize=self.normalize
        )
        
        self.test_dataset = EEGSetDataset(
            eeg_data_path=self.eeg_data_path,
            split='test',
            cache_size=self.cache_size,
            use_cache=self.use_cache,
            normalize=self.normalize
        )
    
    def train_dataloader(self):
        """返回训练数据加载器"""
        if self.train_dataset is None:
            self.setup()
        
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """返回验证数据加载器"""
        if self.val_dataset is None:
            self.setup()
        
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """返回测试数据加载器"""
        if self.test_dataset is None:
            self.setup()
        
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
