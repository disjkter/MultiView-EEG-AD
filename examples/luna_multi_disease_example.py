#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LUNA Multi-Disease Classification Example
基于LUNA的多疾病分类使用示例
"""

import os
import torch
from models.luna_multi_disease import MultiDiseaseLUNA
from data_module.luna_multi_disease_data_module import LUNAMultiDiseaseDataModule, HDF5Converter
from tasks.luna_multi_disease_task import LUNAMultiDiseaseTask
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def create_luna_config():
    """创建LUNA多疾病配置"""
    return {
        'model': {
            'num_channels': 19,  # 10-20系统通道数
            'encoder': {
                'embed_dim': 384,
                'depth': 12,
                'num_heads': 6,
                'mlp_ratio': 4.0,
                'use_channel_unification': True
            },
            'disease_specific_modules': {
                'num_diseases': 6,
                'embed_dim': 192,
                'depth': 4,
                'num_heads': 4,
                'channel_groups': 3
            },
            'feature_fusion': {
                'fusion_dim': 512,
                'num_queries': 8,
                'method': 'query_based_fusion',
                'dropout': 0.1
            },
            'classifier': {
                'hidden_dims': [768, 384, 192],
                'num_classes': 7,  # 健康 + 6种疾病
                'dropout': 0.3
            }
        },
        'data_module': {
            'disease_labels': {
                'healthy': 0,
                'epilepsy': 1,
                'adhd': 2,
                'autism': 3,
                'alzheimer': 4,
                'parkinson': 5,
                'depression': 6
            },
            'cfg': {
                'batch_size': 16,
                'num_workers': 4,
                'class_balancing': 'weighted_sampler',
                'hdf5_config': {
                    'chunk_size': 1000,
                    'compression': 'gzip',
                    'compression_opts': 9
                }
            }
        },
        'task': {
            'masking': {
                'patch_size': [4, 32],
                'masking_ratio': 0.4,
                'unmasked_loss_coeff': 0.1
            },
            'criterion': {
                'feature_consistency_loss': {
                    'lambda_consistency': 0.1
                },
                'position_aware_loss': {
                    'lambda_position': 0.05
                }
            }
        },
        'optimizer': {
            'optim': 'AdamW',
            'weight_decay': 0.05,
            'param_groups': {
                'encoder': 1e-5,
                'channel_unifier': 3e-5,
                'disease_modules': 5e-5,
                'classifier': 1e-4
            }
        }
    }


def convert_data_to_hdf5():
    """将原始数据转换为HDF5格式"""
    print("=== 数据格式转换 ===")
    
    # 原始数据目录结构示例:
    # raw_data/
    # ├── healthy/
    # │   ├── sample1.npy
    # │   ├── sample2.npy
    # ├── epilepsy/
    # │   ├── patient1.npy
    # │   ├── patient2.npy
    # └── ...
    
    raw_data_dir = "path/to/raw/eeg/data"  # 替换为实际路径
    output_dir = "model-data/luna_multi_disease_data"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    disease_labels = {
        'healthy': 0, 'epilepsy': 1, 'adhd': 2,
        'autism': 3, 'alzheimer': 4, 'parkinson': 5, 'depression': 6
    }
    
    # 转换训练数据
    print("转换训练数据...")
    HDF5Converter.convert_raw_to_hdf5(
        raw_data_dir=os.path.join(raw_data_dir, "train"),
        output_hdf5=os.path.join(output_dir, "train.h5"),
        disease_labels=disease_labels
    )
    
    # 转换验证数据
    print("转换验证数据...")
    HDF5Converter.convert_raw_to_hdf5(
        raw_data_dir=os.path.join(raw_data_dir, "val"),
        output_hdf5=os.path.join(output_dir, "val.h5"),
        disease_labels=disease_labels
    )
    
    # 转换测试数据
    print("转换测试数据...")
    HDF5Converter.convert_raw_to_hdf5(
        raw_data_dir=os.path.join(raw_data_dir, "test"),
        output_hdf5=os.path.join(output_dir, "test.h5"),
        disease_labels=disease_labels
    )
    
    print("✓ 数据转换完成")


def train_luna_multi_disease():
    """训练LUNA多疾病分类器"""
    print("=== LUNA多疾病分类器训练 ===")
    
    # 1. 创建配置
    config = create_luna_config()
    
    # 2. 准备数据模块
    print("准备数据模块...")
    data_module = LUNAMultiDiseaseDataModule(
        name="luna_multi_disease",
        cfg=config['data_module']['cfg'],
        disease_labels=config['data_module']['disease_labels']
    )
    
    try:
        data_module.prepare_data()
        data_module.setup()
        print("✓ 数据模块准备完成")
    except Exception as e:
        print(f"✗ 数据准备失败: {e}")
        print("请确保数据目录结构正确:")
        print("model-data/luna_multi_disease_data/")
        print("├── train.h5")
        print("├── val.h5")
        print("└── test.h5")
        return
    
    # 3. 创建训练任务
    print("创建训练任务...")
    task = LUNAMultiDiseaseTask(config)
    print("✓ 训练任务创建完成")
    
    # 4. 设置训练器
    logger = TensorBoardLogger("tb_logs", name="luna_multi_disease")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_weighted_f1",
        mode="max",
        save_top_k=3,
        filename="luna-multidisease-{epoch:02d}-{val_weighted_f1:.4f}"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_weighted_f1",
        mode="max",
        patience=15
    )
    
    trainer = pl.Trainer(
        max_epochs=120,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=16,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    
    # 5. 开始训练
    print("开始训练...")
    trainer.fit(task, data_module)
    
    # 6. 测试
    print("开始测试...")
    trainer.test(task, data_module)
    
    print("✓ 训练完成！")


def inference_example():
    """推理使用示例"""
    print("\n=== 推理示例 ===")
    
    print("使用训练好的模型进行推理:")
    print("""
    # 加载模型
    model = LUNAMultiDiseaseTask.load_from_checkpoint("path/to/best_checkpoint.ckpt")
    model.eval()
    
    # 准备EEG数据
    eeg_sample = torch.randn(1, 19, 2000)  # [B, C, T]
    channel_locations = torch.randn(1, 19, 3)  # [B, C, 3]
    
    # 推理
    with torch.no_grad():
        logits, features = model.model(eeg_sample, channel_locations=channel_locations)
        probabilities = F.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1)
    
    print(f"预测疾病类别: {prediction.item()}")
    print(f"各类别概率: {probabilities.squeeze().cpu().numpy()}")
    """)


if __name__ == "__main__":
    # 数据转换（首次运行时需要）
    # convert_data_to_hdf5()
    
    # 训练模型
    train_luna_multi_disease()
    
    # 推理示例
    inference_example()