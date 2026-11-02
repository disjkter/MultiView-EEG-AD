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
#* Author: Your Name                                                          *
#*----------------------------------------------------------------------------*

import os
import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# 处理相对导入问题
try:
    # 尝试相对导入（在包内使用时）
    from ..datasets.hdf5_dataset import HDF5Loader
except ImportError:
    # 如果相对导入失败，尝试绝对导入（直接运行时）
    try:
        # 添加项目根目录到路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, project_root)
        from datasets.hdf5_dataset import HDF5Loader
    except ImportError:
        # 最后尝试直接导入（如果在正确的工作目录中）
        from datasets.hdf5_dataset import HDF5Loader


class EEGDataModule(pl.LightningDataModule):
    """
    A data module for EEG data stored in HDF5 format.
    Assumes data is organized as ${DATA_PATH}/eeg-data_data/{train,val,test}.h5
    """
    def __init__(
        self,
        name: str = "eeg-data",
        cfg: dict = None,
        train_val_split_ratio: float = 0.8,
        datasets: dict = None,
        test=None
    ):
        super().__init__()
        self.name = name
        self.cfg = cfg or {}
        self.train_val_split_ratio = train_val_split_ratio
        self.datasets = datasets or {}
        self.test_flag = test
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # Check if data exists
        data_path = os.environ.get('DATA_PATH')
        if not data_path:
            # 如果没有设置环境变量，使用项目根目录
            data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            print(f"未设置DATA_PATH环境变量，使用项目根目录: {data_path}")
        
        dataset_path = os.path.join(data_path, f"{self.name}-data")  # 使用连字符而不是下划线
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        # Verify that required directories and HDF5 files exist
        required_dirs = ["train", "val", "test"]
        for dir_name in required_dirs:
            dir_path = os.path.join(dataset_path, dir_name)
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Required directory does not exist: {dir_path}")
            
            # Check for HDF5 files in each directory
            hdf5_file = f"{dir_name}.h5"
            file_path = os.path.join(dir_path, hdf5_file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file does not exist: {file_path}")

    def setup(self, stage=None):
        data_path = os.environ.get('DATA_PATH')
        if not data_path:
            # 如果没有设置环境变量，使用项目根目录
            data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        dataset_path = os.path.join(data_path, f"{self.name}-data")  # 使用连字符而不是下划线
        
        # Load train dataset
        train_hdf5_path = os.path.join(dataset_path, "train", "train.h5")
        self.train_dataset = HDF5Loader(
            hdf5_file=train_hdf5_path,
            finetune=True  # Adjust based on your use case
        )
        
        # Load validation dataset
        val_hdf5_path = os.path.join(dataset_path, "val", "val.h5")
        self.val_dataset = HDF5Loader(
            hdf5_file=val_hdf5_path,
            finetune=True  # Adjust based on your use case
        )
        
        # Load test dataset if needed
        if self.test_flag:
            test_hdf5_path = os.path.join(dataset_path, "test", "test.h5")
            self.test_dataset = HDF5Loader(
                hdf5_file=test_hdf5_path,
                finetune=True  # Adjust based on your use case
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.get('batch_size', 32),
            num_workers=self.cfg.get('num_workers', 4),
            shuffle=True,
            pin_memory=True,
            persistent_workers=True if self.cfg.get('num_workers', 4) > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.get('batch_size', 32),
            num_workers=self.cfg.get('num_workers', 4),
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.cfg.get('num_workers', 4) > 0 else False
        )

    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.cfg.get('batch_size', 32),
                num_workers=self.cfg.get('num_workers', 4),
                shuffle=False,
                pin_memory=True,
                persistent_workers=True if self.cfg.get('num_workers', 4) > 0 else False
            )
        else:
            return None

    def predict_dataloader(self):
        return self.test_dataloader()