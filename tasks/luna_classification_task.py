"""
基于 LUNA backbone 的分类微调任务
用于各种 EEG 分类下游任务
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import hydra
import torch_optimizer as torch_optim
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy, Precision, Recall, AUROC,
    AveragePrecision, CohenKappa, F1Score
)
from safetensors.torch import load_file
import numpy as np


class ChannelWiseNormalize:
    """按通道进行归一化"""
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, tensor):
        with torch.no_grad():
            # tensor: (B, C, T)
            mean = tensor.mean(dim=2, keepdim=True)
            std = tensor.std(dim=2, keepdim=True)
            return (tensor - mean) / (std + self.eps)


class LUNAClassificationTask(pl.LightningModule):
    """
    基于 LUNA 的 EEG 分类微调任务
    
    支持：
    - 加载预训练的 LUNA 权重
    - Backbone 冻结/解冻/部分微调
    - 自动混合精度训练
    - 完整的评估指标（准确率、F1、AUROC等）
    - 学习率调度和优化器配置
    """
    
    def __init__(self, hparams):
        """
        初始化 LUNA 分类任务
        
        Args:
            hparams (DictConfig): Hydra 配置对象，包含：
                - model: 模型配置
                - optimizer: 优化器配置
                - scheduler: 学习率调度器配置
                - input_normalization: 输入归一化配置
                - finetuning: 微调策略配置
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # 构建模型
        self.model = hydra.utils.instantiate(self.hparams.model)
        self.num_classes = self.hparams.model.num_classes
        
        # 输入归一化
        self.normalize = False
        if (hasattr(self.hparams, 'input_normalization') and 
            self.hparams.input_normalization is not None and 
            self.hparams.input_normalization.get('normalize', False)):
            self.normalize = True
            self.normalize_fct = ChannelWiseNormalize()
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 确定分类任务类型
        if self.num_classes == 2:
            self.classification_task = "binary"
        else:
            self.classification_task = "multiclass"
        
        # 评估指标
        self._setup_metrics()
        
        # 用于存储预测和目标（计算混淆矩阵等）
        self.train_predictions = []
        self.train_targets = []
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []
    
    def _setup_metrics(self):
        """设置评估指标"""
        label_metrics = MetricCollection([
            Accuracy(task=self.classification_task, num_classes=self.num_classes, average="macro"),
            Recall(task='multiclass', num_classes=self.num_classes, average="macro"),
            Precision(task=self.classification_task, num_classes=self.num_classes, average="macro"),
            F1Score(task=self.classification_task, num_classes=self.num_classes, average="macro"),
            CohenKappa(task=self.classification_task, num_classes=self.num_classes)
        ])
        
        logit_metrics = MetricCollection([
            AUROC(task=self.classification_task, num_classes=self.num_classes, average="macro"),
            AveragePrecision(task=self.classification_task, num_classes=self.num_classes, average="macro"),
        ])
        
        self.train_label_metrics = label_metrics.clone(prefix='train_')
        self.val_label_metrics = label_metrics.clone(prefix='val_')
        self.test_label_metrics = label_metrics.clone(prefix='test_')
        
        self.train_logit_metrics = logit_metrics.clone(prefix='train_')
        self.val_logit_metrics = logit_metrics.clone(prefix='val_')
        self.test_logit_metrics = logit_metrics.clone(prefix='test_')
    
    def load_pretrained_luna_checkpoint(self, checkpoint_path: str):
        """
        加载预训练的 LUNA 权重
        
        Args:
            checkpoint_path: 预训练权重的路径（支持 .ckpt 和 .safetensors）
        """
        if checkpoint_path is None:
            print("No pretrained checkpoint provided")
            return
        
        print(f"Loading pretrained LUNA weights from: {checkpoint_path}")
        
        try:
            if checkpoint_path.endswith('.safetensors'):
                self._load_safetensors_checkpoint(checkpoint_path)
            elif checkpoint_path.endswith('.ckpt'):
                self._load_pytorch_checkpoint(checkpoint_path)
            else:
                raise ValueError(f"Unknown checkpoint format: {checkpoint_path}")
            
            print("✓ Pretrained weights loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading pretrained weights: {e}")
            raise
    
    def _load_pytorch_checkpoint(self, checkpoint_path: str):
        """加载 PyTorch 格式的检查点"""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        
        # 移除不需要的权重（分类头等）
        state_dict = {
            k: v for k, v in state_dict.items() 
            if 'decoder_head' not in k and 'classifier' not in k
        }
        
        # 加载到 backbone
        self.model.backbone.load_state_dict(state_dict, strict=False)
    
    def _load_safetensors_checkpoint(self, checkpoint_path: str):
        """加载 SafeTensors 格式的检查点"""
        state_dict = load_file(checkpoint_path)
        
        # 移除不需要的权重
        state_dict = {
            k: v for k, v in state_dict.items() 
            if 'decoder_head' not in k and 'classifier' not in k
        }
        
        self.model.backbone.load_state_dict(state_dict, strict=False)
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 优化器
        optimizer_cfg = self.hparams.optimizer
        optimizer_type = optimizer_cfg.get('_target_', 'torch.optim.AdamW')
        
        # 提取学习率和其他参数
        lr = optimizer_cfg.get('lr', 1e-3)
        
        # 根据微调策略设置不同层的学习率
        if hasattr(self.hparams, 'finetuning'):
            finetune_cfg = self.hparams.finetuning
            if finetune_cfg.get('layer_wise_lr_decay', False):
                # 层级学习率衰减
                optimizer = self._setup_layer_wise_optimizer(lr)
            else:
                optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        
        # 学习率调度器
        if hasattr(self.hparams, 'scheduler'):
            scheduler_cfg = self.hparams.scheduler
            scheduler_type = scheduler_cfg.get('_target_', 'torch.optim.lr_scheduler.CosineAnnealingWarmRestarts')
            
            if 'CosineAnnealingWarmRestarts' in scheduler_type:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=scheduler_cfg.get('T_0', 50),
                    T_mult=scheduler_cfg.get('T_mult', 1),
                    eta_min=scheduler_cfg.get('eta_min', 1e-6)
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=100,
                    eta_min=1e-6
                )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        return optimizer
    
    def _setup_layer_wise_optimizer(self, base_lr: float):
        """设置分层学习率衰减的优化器"""
        decay = self.hparams.finetuning.get('layer_wise_decay', 0.9)
        
        # 为不同层设置不同的学习率
        param_groups = []
        
        # Backbone 的不同层
        if hasattr(self.model, 'backbone'):
            for name, param in self.model.backbone.named_parameters():
                if param.requires_grad:
                    # 根据层的深度衰减学习率
                    lr = base_lr * (decay ** self._get_layer_depth(name))
                    param_groups.append({'params': [param], 'lr': lr})
        
        # 分类头（最高学习率）
        for name, param in self.model.classifier.named_parameters():
            if param.requires_grad:
                param_groups.append({'params': [param], 'lr': base_lr})
        
        optimizer = torch.optim.AdamW(param_groups)
        return optimizer
    
    @staticmethod
    def _get_layer_depth(param_name: str) -> int:
        """估算参数在网络中的深度"""
        # 简单的启发式方法
        if 'layer' in param_name:
            try:
                depth = int(param_name.split('layer')[1].split('.')[0])
                return depth
            except:
                return 0
        return 0
    
    def forward(self, batch):
        """前向传播"""
        x = batch['X'] if isinstance(batch, dict) else batch[0]
        
        # 输入归一化
        if self.normalize:
            x = self.normalize_fct(x)
        
        logits, features = self.model(x)
        return logits, features
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        x = batch['X'] if isinstance(batch, dict) else batch[0]
        y = batch['y'] if isinstance(batch, dict) else batch[1]
        
        # 输入归一化
        if self.normalize:
            x = self.normalize_fct(x)
        
        # 前向传播
        logits, features = self.model(x)
        
        # 计算损失
        loss = self.criterion(logits, y)
        
        # 计算指标
        preds = torch.argmax(logits, dim=1)
        self.train_label_metrics(preds, y)
        self.train_logit_metrics(logits, y)
        
        # 记录
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_label_metrics, on_epoch=True, prog_bar=True)
        
        # 缓存用于计算混淆矩阵
        self.train_predictions.extend(preds.cpu().numpy())
        self.train_targets.extend(y.cpu().numpy())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        x = batch['X'] if isinstance(batch, dict) else batch[0]
        y = batch['y'] if isinstance(batch, dict) else batch[1]
        
        # 输入归一化
        if self.normalize:
            x = self.normalize_fct(x)
        
        # 前向传播
        logits, features = self.model(x)
        
        # 计算损失
        loss = self.criterion(logits, y)
        
        # 计算指标
        preds = torch.argmax(logits, dim=1)
        self.val_label_metrics(preds, y)
        self.val_logit_metrics(logits, y)
        
        # 记录
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_label_metrics, on_epoch=True, prog_bar=True)
        
        # 缓存
        self.val_predictions.extend(preds.cpu().numpy())
        self.val_targets.extend(y.cpu().numpy())
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        x = batch['X'] if isinstance(batch, dict) else batch[0]
        y = batch['y'] if isinstance(batch, dict) else batch[1]
        
        # 输入归一化
        if self.normalize:
            x = self.normalize_fct(x)
        
        # 前向传播
        logits, features = self.model(x)
        
        # 计算损失
        loss = self.criterion(logits, y)
        
        # 计算指标
        preds = torch.argmax(logits, dim=1)
        self.test_label_metrics(preds, y)
        self.test_logit_metrics(logits, y)
        
        # 记录
        self.log('test_loss', loss, on_epoch=True)
        self.log_dict(self.test_label_metrics, on_epoch=True)
        
        # 缓存
        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())
        
        return loss
    
    def on_validation_epoch_end(self):
        """验证周期结束"""
        self.log_dict(self.val_logit_metrics)
        
        # 清空缓存（下一个周期会重新积累）
        self.val_predictions = []
        self.val_targets = []
    
    def on_test_epoch_end(self):
        """测试周期结束"""
        self.log_dict(self.test_logit_metrics)
        
        # 打印详细报告
        if len(self.test_predictions) > 0:
            from sklearn.metrics import classification_report, confusion_matrix
            
            preds = np.array(self.test_predictions)
            targets = np.array(self.test_targets)
            
            print("\n" + "="*80)
            print("Test Classification Report")
            print("="*80)
            print(classification_report(targets, preds))
            
            print("\nConfusion Matrix")
            print("="*80)
            print(confusion_matrix(targets, preds))
