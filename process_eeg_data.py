#!/usr/bin/env python
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
#* Description: Process BIDS-format EEG data from eeg-data folder             *
#*              to HDF5 format for training                                    *
#*                                                                            *
#*----------------------------------------------------------------------------*

import os
import pickle
import numpy as np
import h5py
from tqdm import tqdm
import argparse
import pandas as pd
from pathlib import Path
import json
from mne.io import read_raw_eeglab
import warnings

warnings.filterwarnings('ignore')

# 疾病标签映射
DISEASE_LABELS = {
    'A': 0,  # Alzheimer Disease
    'F': 1,  # Frontotemporal Dementia
    'C': 2,  # Healthy Control
}

DISEASE_NAMES = {
    0: 'Alzheimer',
    1: 'Frontotemporal_Dementia',
    2: 'Healthy'
}

class EEGDataProcessor:
    """处理 BIDS 格式的 EEG 数据"""
    
    def __init__(self, eeg_data_path, output_path, target_srate=256, target_duration=5):
        """
        Args:
            eeg_data_path: eeg-data 文件夹路径
            output_path: 输出文件夹路径
            target_srate: 目标采样率 (Hz)
            target_duration: 目标信号时长 (秒)
        """
        self.eeg_data_path = eeg_data_path
        self.output_path = output_path
        self.target_srate = target_srate
        self.target_samples = target_srate * target_duration
        
        # 读取参与者信息
        self.participants_df = pd.read_csv(
            os.path.join(eeg_data_path, 'participants.tsv'),
            sep='\t'
        )
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
    
    def load_eeg_file(self, filepath):
        """加载 .set 文件并返回 EEG 数据和采样率"""
        try:
            # 使用 MNE 读取 EEGLAB 格式的 .set 文件
            raw = read_raw_eeglab(filepath, preload=True, verbose=False)
            return raw.get_data(), raw.info['sfreq']
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None
    
    def preprocess_eeg(self, eeg_data, srate):
        """预处理 EEG 数据：重采样、截断、归一化"""
        if eeg_data is None:
            return None
        
        try:
            # 简单的高通滤波 (移除直流分量)
            eeg_data = eeg_data - np.mean(eeg_data, axis=1, keepdims=True)
            
            # 选择前 target_samples 个采样点
            if eeg_data.shape[1] >= self.target_samples:
                eeg_data = eeg_data[:, :int(self.target_samples)]
            else:
                # 如果数据不足，填充零
                padding = int(self.target_samples) - eeg_data.shape[1]
                eeg_data = np.pad(eeg_data, ((0, 0), (0, padding)), mode='constant')
            
            # 【重要修改】每个样本内独立标准化，避免数据泄露
            # 对每个通道分别进行标准化
            for ch in range(eeg_data.shape[0]):
                ch_mean = np.mean(eeg_data[ch])
                ch_std = np.std(eeg_data[ch]) + 1e-8
                eeg_data[ch] = (eeg_data[ch] - ch_mean) / ch_std
            
            return eeg_data.astype(np.float32)
        except Exception as e:
            print(f"Error preprocessing EEG: {e}")
            return None
    
    def process_all_data(self):
        """处理所有受试者的数据"""
        
        print("=" * 80)
        print("Processing BIDS-format EEG data from eeg-data folder")
        print("=" * 80)
        
        data_samples = []
        
        derivatives_path = os.path.join(self.eeg_data_path, 'derivatives')
        
        # 遍历所有受试者
        for _, row in tqdm(self.participants_df.iterrows(), total=len(self.participants_df)):
            participant_id = row['participant_id']
            group = row['Group']
            label = DISEASE_LABELS[group]
            
            # 构建 .set 文件路径
            eeg_file_path = os.path.join(
                derivatives_path,
                participant_id,
                'eeg',
                f'{participant_id}_task-eyesclosed_eeg.set'
            )
            
            if not os.path.exists(eeg_file_path):
                print(f"Warning: File not found - {eeg_file_path}")
                continue
            
            # 加载原始 EEG 数据
            eeg_data, srate = self.load_eeg_file(eeg_file_path)
            
            if eeg_data is None:
                print(f"Warning: Failed to load {eeg_file_path}")
                continue
            
            # 预处理
            processed_eeg = self.preprocess_eeg(eeg_data, srate)
            
            if processed_eeg is None:
                print(f"Warning: Failed to process {eeg_file_path}")
                continue
            
            # 创建数据样本
            sample = {
                'X': processed_eeg,  # (C, T) - 通道数 × 时间步数
                'y': np.array(label, dtype=np.int64),
                'participant_id': participant_id,
                'group': group,
                'age': np.array(row['Age'], dtype=np.int32),
                'gender': row['Gender'],
            }
            
            data_samples.append(sample)
        
        print(f"\nSuccessfully loaded {len(data_samples)} samples")
        
        return data_samples
    
    def split_data(self, data_samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """
        按比例划分数据集: train/val/test
        并尽可能平衡各组
        """
        np.random.seed(random_state)
        
        # 按组分别划分
        groups = {}
        for sample in data_samples:
            group = sample['group']
            if group not in groups:
                groups[group] = []
            groups[group].append(sample)
        
        print(f"\nData distribution by group:")
        for group, samples in groups.items():
            print(f"  {DISEASE_NAMES[DISEASE_LABELS[group]]}: {len(samples)} samples")
        
        train_samples = []
        val_samples = []
        test_samples = []
        
        # 在每组内部分别划分
        for group, samples in groups.items():
            indices = np.arange(len(samples))
            np.random.shuffle(indices)
            
            n_train = int(len(samples) * train_ratio)
            n_val = int(len(samples) * val_ratio)
            
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train+n_val]
            test_idx = indices[n_train+n_val:]
            
            train_samples.extend([samples[i] for i in train_idx])
            val_samples.extend([samples[i] for i in val_idx])
            test_samples.extend([samples[i] for i in test_idx])
        
        print(f"\nData split:")
        print(f"  Train: {len(train_samples)} samples")
        print(f"  Val:   {len(val_samples)} samples")
        print(f"  Test:  {len(test_samples)} samples")
        
        return {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
    
    def save_as_pickle(self, data_samples, output_dir, split_name):
        """保存为 pickle 文件（中间格式）"""
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, sample in enumerate(tqdm(data_samples, desc=f"Saving {split_name} pickle files")):
            pickle_path = os.path.join(output_dir, f'{idx:05d}.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(sample, f)
        
        print(f"Saved {len(data_samples)} samples to {output_dir}")
    
    def create_hdf5(self, data_samples, output_file, group_size=100):
        """创建 HDF5 文件"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with h5py.File(output_file, 'w') as h5f:
            group_idx = 0
            for i in tqdm(range(0, len(data_samples), group_size), desc=f"Creating {output_file}"):
                group_samples = data_samples[i:i+group_size]
                
                if not group_samples:
                    continue
                
                grp = h5f.create_group(f"data_group_{group_idx}")
                
                # 提取 X 和 y
                X_data = np.array([s['X'] for s in group_samples])
                y_data = np.array([s['y'] for s in group_samples])
                
                # 保存数据
                grp.create_dataset("X", data=X_data, compression='gzip', compression_opts=4)
                grp.create_dataset("y", data=y_data)
                
                # 保存元数据
                for j, sample in enumerate(group_samples):
                    grp[f"participant_id_{j}"] = sample['participant_id']
                    grp[f"age_{j}"] = sample['age']
                    grp[f"gender_{j}"] = sample['gender']
                
                group_idx += 1
        
        print(f"Created HDF5 file: {output_file}")
    
    def process_and_save(self, save_intermediate_pkl=False):
        """处理数据并保存为 HDF5 格式"""
        
        # 1. 加载和预处理所有数据
        print("\n[Step 1] Loading and preprocessing EEG data...")
        data_samples = self.process_all_data()
        
        if len(data_samples) == 0:
            print("Error: No data samples loaded!")
            return
        
        # 2. 划分数据集
        print("\n[Step 2] Splitting data into train/val/test...")
        splits = self.split_data(data_samples)
        
        # 3. 保存为中间 pickle 格式（可选）
        if save_intermediate_pkl:
            print("\n[Step 3] Saving intermediate pickle files...")
            for split_name, samples in splits.items():
                pkl_dir = os.path.join(self.output_path, 'processed', split_name)
                self.save_as_pickle(samples, pkl_dir, split_name)
        
        # 4. 创建最终的 HDF5 文件
        print("\n[Step 4] Creating HDF5 files...")
        for split_name, samples in splits.items():
            output_file = os.path.join(self.output_path, f'{split_name}.h5')
            self.create_hdf5(samples, output_file)
        
        # 5. 保存数据统计信息
        print("\n[Step 5] Saving data statistics...")
        stats = {
            'total_samples': len(data_samples),
            'train_samples': len(splits['train']),
            'val_samples': len(splits['val']),
            'test_samples': len(splits['test']),
            'target_srate': self.target_srate,
            'target_samples': int(self.target_samples),
            'disease_labels': DISEASE_LABELS,
            'disease_names': DISEASE_NAMES,
        }
        
        stats_file = os.path.join(self.output_path, 'data_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nData statistics saved to {stats_file}")
        
        print("\n" + "=" * 80)
        print("Data processing completed successfully!")
        print("=" * 80)
        print(f"\nOutput files:")
        print(f"  - {os.path.join(self.output_path, 'train.h5')}")
        print(f"  - {os.path.join(self.output_path, 'val.h5')}")
        print(f"  - {os.path.join(self.output_path, 'test.h5')}")
        print(f"  - {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Process BIDS-format EEG data from eeg-data folder to HDF5"
    )
    parser.add_argument(
        '--eeg_data_path',
        type=str,
        default='eeg-data',
        help='Path to eeg-data folder (default: eeg-data)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='model-data',
        help='Output path for HDF5 files (default: model-data)'
    )
    parser.add_argument(
        '--target_srate',
        type=int,
        default=256,
        help='Target sampling rate in Hz (default: 256)'
    )
    parser.add_argument(
        '--target_duration',
        type=int,
        default=5,
        help='Target signal duration in seconds (default: 5 seconds = 1280 samples at 256 Hz)'
    )
    parser.add_argument(
        '--save_pkl',
        action='store_true',
        help='Whether to save intermediate pickle files'
    )
    
    args = parser.parse_args()
    
    # 初始化处理器
    processor = EEGDataProcessor(
        eeg_data_path=args.eeg_data_path,
        output_path=args.output_path,
        target_srate=args.target_srate,
        target_duration=args.target_duration
    )
    
    # 处理并保存数据
    processor.process_and_save(save_intermediate_pkl=args.save_pkl)


if __name__ == '__main__':
    main()
