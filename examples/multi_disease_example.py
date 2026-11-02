#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Disease EEG Classification Example
多疾病EEG分类使用示例
"""

import os
import torch
from models.multi_disease_classifier import MultiDiseaseClassifier
from data_module.multi_disease_data_module import MultiDiseaseDataModule
from tasks.multi_disease_task import MultiDiseaseClassificationTask
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def create_sample_config():
    """创建示例配置"""
    return {
        'model': {
            'input_dim': 256,
            'num_channels': 19,
            'shared_encoder': {
                'embed_dim': 256,
                'depth': 12,
                'num_heads': 8,
                'mlp_ratio': 4.0
            },
            'disease_specific_encoders': {
                'num_diseases': 5,
                'embed_dim': 128,
                'depth': 6,
                'num_heads': 4
            },
            'feature_fusion': {
                'fusion_dim': 384,
                'fusion_method': 'attention_weighted',
                'dropout': 0.1
            },
            'classifier': {
                'hidden_dims': [512, 256, 128],
                'num_classes': 6,
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
                'depression': 5
            },
            'cfg': {
                'batch_size': 12,
                'num_workers': 4,
                'class_balancing': 'weighted_sampler'
            }
        },
        'criterion': {
            'orthogonality_loss': {
                'lambda_orth': 0.1
            }
        },
        'optimizer': {
            'optim': 'AdamW',
            'param_groups': {
                'shared_encoder': 1e-5,
                'disease_specific': 5e-5,
                'classifier': 1e-4
            },
            'weight_decay': 0.01
        }
    }


def train_multi_disease_classifier():
    """训练多疾病分类器"""
    print("=== 多疾病EEG分类器训练 ===")
    
    # 1. 创建配置
    config = create_sample_config()
    
    # 2. 准备数据
    print("准备数据...")
    data_module = MultiDiseaseDataModule(
        name="multi_disease",
        cfg=config['data_module']['cfg'],
        disease_labels=config['data_module']['disease_labels']
    )
    
    try:
        data_module.prepare_data()
        data_module.setup()
        print("✓ 数据准备完成")
    except Exception as e:
        print(f"✗ 数据准备失败: {e}")
        print("请确保数据目录结构正确:")
        print("model-data/multi_disease_data/")
        print("├── train/")
        print("│   ├── healthy/*.h5")
        print("│   ├── epilepsy/*.h5")
        print("│   ├── adhd/*.h5")
        print("│   ├── autism/*.h5")
        print("│   ├── alzheimer/*.h5")
        print("│   └── depression/*.h5")
        print("└── test/")
        return
    
    # 3. 创建模型和任务
    print("创建模型...")
    task = MultiDiseaseClassificationTask(config)
    print("✓ 模型创建完成")
    
    # 4. 设置训练器
    logger = TensorBoardLogger("tb_logs", name="multi_disease_eeg")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_macro_f1",
        mode="max",
        save_top_k=3,
        filename="multi-disease-{epoch:02d}-{val_macro_f1:.4f}"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_macro_f1",
        mode="max",
        patience=15
    )
    
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=16,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    
    # 5. 开始训练
    print("开始训练...")
    trainer.fit(task, data_module)
    
    # 6. 测试
    if hasattr(data_module, 'test_dataset') and len(data_module.test_dataset) > 0:
        print("开始测试...")
        trainer.test(task, data_module)
    
    print("训练完成！")


def inference_example():
    """推理示例"""
    print("\n=== 推理示例 ===")
    
    # 加载训练好的模型
    # model = MultiDiseaseClassifier.load_from_checkpoint("path/to/checkpoint.ckpt")
    
    print("推理功能示例:")
    print("1. 单样本预测:")
    print("   result = model.predict(eeg_sample)")
    print("   print(f'预测疾病: {result.prediction}')")
    print("   print(f'置信度: {result.confidence}')")
    
    print("\n2. 批量预测:")
    print("   results = model.batch_predict(eeg_batch)")
    print("   for i, result in enumerate(results):")
    print("       print(f'样本{i}: {result.prediction} (置信度: {result.confidence:.3f})')")
    
    print("\n3. 特征可视化:")
    print("   features = model.extract_features(eeg_sample)")
    print("   plot_attention_map(features.attention_weights)")


if __name__ == "__main__":
    # 训练模型
    train_multi_disease_classifier()
    
    # 推理示例
    inference_example()