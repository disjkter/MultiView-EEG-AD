"""
Multi-Disease Classification Task
多疾病分类训练任务
"""

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch_optimizer as torch_optim
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class MultiDiseaseClassificationTask(pl.LightningModule):
    """多疾病分类训练任务"""
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # 初始化模型
        self.model = self._initialize_model()
        
        # 损失函数
        self.criterion = self._initialize_criterion()
        
        # 评估指标缓存
        self.train_predictions = []
        self.train_targets = []
        self.val_predictions = []
        self.val_targets = []
        
        # 疾病名称映射
        self.disease_names = {v: k for k, v in self.hparams.data_module.disease_labels.items()}
        
    def _initialize_model(self):
        """初始化模型"""
        from models.multi_disease_classifier import MultiDiseaseClassifier
        return MultiDiseaseClassifier(self.hparams.model)
    
    def _initialize_criterion(self):
        """初始化损失函数"""
        from models.multi_disease_classifier import MultiTaskLoss
        return MultiTaskLoss(self.hparams.criterion)
    
    def forward(self, batch):
        """前向传播"""
        eeg_data = batch['eeg']
        labels = batch['label']
        
        logits, features = self.model(eeg_data, labels)
        return logits, features, labels
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        logits, features, labels = self.forward(batch)
        
        # 计算损失
        losses = self.criterion(logits, labels, features)
        
        # 记录损失
        self.log('train_loss', losses['total'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_classification_loss', losses['classification'], on_step=True, on_epoch=True)
        self.log('train_orthogonality_loss', losses['orthogonality'], on_step=True, on_epoch=True)
        
        # 缓存预测结果用于指标计算
        preds = torch.argmax(logits, dim=-1)
        self.train_predictions.extend(preds.cpu().numpy())
        self.train_targets.extend(labels.cpu().numpy())
        
        return losses['total']
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        logits, features, labels = self.forward(batch)
        
        # 计算损失
        losses = self.criterion(logits, labels, features)
        
        # 记录损失
        self.log('val_loss', losses['total'], on_step=False, on_epoch=True, prog_bar=True)
        
        # 缓存预测结果
        preds = torch.argmax(logits, dim=-1)
        self.val_predictions.extend(preds.cpu().numpy())
        self.val_targets.extend(labels.cpu().numpy())
        
        return losses['total']
    
    def on_train_epoch_end(self):
        """训练周期结束时计算指标"""
        if len(self.train_predictions) > 0:
            self._compute_and_log_metrics(
                self.train_predictions, 
                self.train_targets, 
                prefix='train'
            )
            # 清空缓存
            self.train_predictions.clear()
            self.train_targets.clear()
    
    def on_validation_epoch_end(self):
        """验证周期结束时计算指标"""
        if len(self.val_predictions) > 0:
            self._compute_and_log_metrics(
                self.val_predictions, 
                self.val_targets, 
                prefix='val'
            )
            # 生成混淆矩阵图
            self._log_confusion_matrix(
                self.val_predictions, 
                self.val_targets, 
                prefix='val'
            )
            # 清空缓存
            self.val_predictions.clear()
            self.val_targets.clear()
    
    def _compute_and_log_metrics(self, predictions, targets, prefix):
        """计算并记录评估指标"""
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 基础指标
        accuracy = np.mean(predictions == targets)
        self.log(f'{prefix}_accuracy', accuracy, on_epoch=True, prog_bar=True)
        
        # 每类准确率
        unique_labels = np.unique(targets)
        per_class_acc = {}
        for label in unique_labels:
            mask = targets == label
            if np.sum(mask) > 0:
                class_acc = np.mean(predictions[mask] == targets[mask])
                class_name = self.disease_names.get(label, f'class_{label}')
                self.log(f'{prefix}_{class_name}_accuracy', class_acc, on_epoch=True)
                per_class_acc[class_name] = class_acc
        
        # 宏平均F1分数
        from sklearn.metrics import f1_score, precision_score, recall_score
        macro_f1 = f1_score(targets, predictions, average='macro', zero_division=0)
        macro_precision = precision_score(targets, predictions, average='macro', zero_division=0)
        macro_recall = recall_score(targets, predictions, average='macro', zero_division=0)
        
        self.log(f'{prefix}_macro_f1', macro_f1, on_epoch=True, prog_bar=True)
        self.log(f'{prefix}_macro_precision', macro_precision, on_epoch=True)
        self.log(f'{prefix}_macro_recall', macro_recall, on_epoch=True)
        
        # 加权F1分数
        weighted_f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        self.log(f'{prefix}_weighted_f1', weighted_f1, on_epoch=True)
    
    def _log_confusion_matrix(self, predictions, targets, prefix):
        """记录混淆矩阵"""
        cm = confusion_matrix(targets, predictions)
        
        # 创建热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[self.disease_names.get(i, f'C{i}') for i in range(len(cm))],
                   yticklabels=[self.disease_names.get(i, f'C{i}') for i in range(len(cm))])
        plt.title(f'{prefix.capitalize()} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 记录到TensorBoard
        if self.logger:
            self.logger.experiment.add_figure(
                f'{prefix}_confusion_matrix', 
                plt.gcf(), 
                self.current_epoch
            )
        plt.close()
    
    def configure_optimizers(self):
        """配置优化器"""
        # 分层学习率配置
        param_groups = []
        
        # 通用编码器参数组
        shared_params = []
        for name, param in self.model.named_parameters():
            if 'shared_encoder' in name:
                shared_params.append(param)
        param_groups.append({
            'params': shared_params,
            'lr': self.hparams.optimizer.param_groups.shared_encoder,
            'weight_decay': self.hparams.optimizer.weight_decay
        })
        
        # 疾病专属编码器参数组
        disease_params = []
        for name, param in self.model.named_parameters():
            if 'disease_specific_encoder' in name:
                disease_params.append(param)
        param_groups.append({
            'params': disease_params,
            'lr': self.hparams.optimizer.param_groups.disease_specific,
            'weight_decay': self.hparams.optimizer.weight_decay
        })
        
        # 分类头参数组
        classifier_params = []
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
        param_groups.append({
            'params': classifier_params,
            'lr': self.hparams.optimizer.param_groups.classifier,
            'weight_decay': self.hparams.optimizer.weight_decay
        })
        
        # 选择优化器
        if self.hparams.optimizer.optim == 'AdamW':
            optimizer = torch.optim.AdamW(param_groups)
        elif self.hparams.optimizer.optim == 'Adam':
            optimizer = torch.optim.Adam(param_groups)
        elif self.hparams.optimizer.optim == 'SGD':
            optimizer = torch.optim.SGD(param_groups, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer.optim}")
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs,
            eta_min=1e-7
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def predict_step(self, batch, batch_idx):
        """预测步骤"""
        eeg_data = batch['eeg']
        logits, features = self.model(eeg_data)
        
        # 返回预测结果和特征
        return {
            'predictions': torch.argmax(logits, dim=-1),
            'probabilities': F.softmax(logits, dim=-1),
            'features': features,
            'sample_info': {k: v for k, v in batch.items() if k != 'eeg'}
        }