"""
LUNA Multi-Disease Classification Task
基于LUNA的多疾病分类训练任务
"""

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch_optimizer as torch_optim
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any


class LUNAMultiDiseaseTask(pl.LightningModule):
    """LUNA多疾病分类训练任务"""
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # 初始化模型
        self.model = self._initialize_model()
        
        # 初始化损失函数
        self.classification_criterion = torch.nn.CrossEntropyLoss()
        self.consistency_criterion = self._get_consistency_criterion()
        self.position_criterion = self._get_position_criterion()
        
        # 缓存预测结果
        self.train_predictions = []
        self.train_targets = []
        self.val_predictions = []
        self.val_targets = []
        
        # 疾病名称映射
        self.disease_names = {v: k for k, v in self.hparams.data_module.disease_labels.items()}
        
    def _initialize_model(self):
        """初始化LUNA多疾病模型"""
        from models.luna_multi_disease import MultiDiseaseLUNA
        return MultiDiseaseLUNA(self.hparams.model)
    
    def _get_consistency_criterion(self):
        """获取一致性损失"""
        from models.luna_multi_disease import FeatureConsistencyLoss
        return FeatureConsistencyLoss(
            self.hparams.task.criterion.feature_consistency_loss.lambda_consistency
        )
    
    def _get_position_criterion(self):
        """获取位置感知损失"""
        from models.luna_multi_disease import PositionAwareLoss
        return PositionAwareLoss(
            self.hparams.task.criterion.position_aware_loss.lambda_position
        )
    
    def forward(self, batch):
        """前向传播"""
        eeg_data = batch['eeg']
        channel_locations = batch['channel_locations']
        labels = batch['labels']
        
        # 生成掩码（LUNA特色）
        mask = self._generate_mask(eeg_data.shape)
        
        # 模型前向传播
        logits, features = self.model(
            eeg_data, 
            mask=mask,
            channel_locations=channel_locations,
            disease_ids=labels
        )
        
        return logits, features, labels, mask
    
    def _generate_mask(self, shape):
        """生成LUNA风格的块状掩码"""
        B, C, T = shape
        patch_H, patch_W = self.hparams.task.masking.patch_size
        masking_ratio = self.hparams.task.masking.masking_ratio
        
        # 计算块数
        num_patches_H = C // patch_H
        num_patches_W = T // patch_W
        total_patches = num_patches_H * num_patches_W
        num_masked = int(total_patches * masking_ratio)
        
        # 生成掩码
        mask = torch.zeros(B, C, T, dtype=torch.bool, device=self.device)
        
        for b in range(B):
            # 随机选择要掩码的块
            selected_patches = torch.randperm(total_patches)[:num_masked]
            
            for patch_idx in selected_patches:
                patch_row = patch_idx // num_patches_W
                patch_col = patch_idx % num_patches_W
                
                h_start = patch_row * patch_H
                h_end = min(h_start + patch_H, C)
                w_start = patch_col * patch_W
                w_end = min(w_start + patch_W, T)
                
                mask[b, h_start:h_end, w_start:w_end] = True
                
        return mask
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        logits, features, labels, mask = self.forward(batch)
        
        # 计算各项损失
        classification_loss = self.classification_criterion(logits, labels)
        
        # 一致性损失
        consistency_loss = self.consistency_criterion(
            features['fused_features'], labels
        )
        
        # 位置感知损失（如果有注意力分数）
        position_loss = 0.0
        if features['attention_scores'] is not None:
            # 简化的位置损失计算
            position_loss = self.position_criterion(
                features['attention_scores'], 
                self._compute_channel_distances(batch['channel_locations'])
            )
        
        # 总损失
        total_loss = (
            classification_loss +
            consistency_loss +
            position_loss
        )
        
        # 记录损失
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_classification_loss', classification_loss, on_step=True, on_epoch=True)
        self.log('train_consistency_loss', consistency_loss, on_step=True, on_epoch=True)
        self.log('train_position_loss', position_loss, on_step=True, on_epoch=True)
        
        # 缓存预测结果
        preds = torch.argmax(logits, dim=-1)
        self.train_predictions.extend(preds.cpu().numpy())
        self.train_targets.extend(labels.cpu().numpy())
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        logits, features, labels, mask = self.forward(batch)
        
        # 计算损失
        classification_loss = self.classification_criterion(logits, labels)
        
        # 总损失
        total_loss = classification_loss
        
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # 缓存预测结果
        preds = torch.argmax(logits, dim=-1)
        self.val_predictions.extend(preds.cpu().numpy())
        self.val_targets.extend(labels.cpu().numpy())
        
        return total_loss
    
    def on_train_epoch_end(self):
        """训练周期结束"""
        if len(self.train_predictions) > 0:
            self._compute_and_log_metrics(
                self.train_predictions,
                self.train_targets,
                prefix='train'
            )
            self.train_predictions.clear()
            self.train_targets.clear()
    
    def on_validation_epoch_end(self):
        """验证周期结束"""
        if len(self.val_predictions) > 0:
            self._compute_and_log_metrics(
                self.val_predictions,
                self.val_targets,
                prefix='val'
            )
            self._log_confusion_matrix(
                self.val_predictions,
                self.val_targets,
                prefix='val'
            )
            self.val_predictions.clear()
            self.val_targets.clear()
    
    def _compute_and_log_metrics(self, predictions, targets, prefix):
        """计算并记录指标"""
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 基础指标
        accuracy = np.mean(predictions == targets)
        self.log(f'{prefix}_accuracy', accuracy, on_epoch=True, prog_bar=True)
        
        # 宏平均指标
        from sklearn.metrics import f1_score, precision_score, recall_score
        macro_f1 = f1_score(targets, predictions, average='macro', zero_division=0)
        macro_precision = precision_score(targets, predictions, average='macro', zero_division=0)
        macro_recall = recall_score(targets, predictions, average='macro', zero_division=0)
        
        self.log(f'{prefix}_macro_f1', macro_f1, on_epoch=True, prog_bar=True)
        self.log(f'{prefix}_macro_precision', macro_precision, on_epoch=True)
        self.log(f'{prefix}_macro_recall', macro_recall, on_epoch=True)
        
        # 加权F1分数
        weighted_f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        self.log(f'{prefix}_weighted_f1', weighted_f1, on_epoch=True, prog_bar=True)
        
        # 每类准确率
        unique_labels = np.unique(targets)
        for label in unique_labels:
            mask = targets == label
            if np.sum(mask) > 0:
                class_acc = np.mean(predictions[mask] == targets[mask])
                class_name = self.disease_names.get(label, f'class_{label}')
                self.log(f'{prefix}_{class_name}_accuracy', class_acc, on_epoch=True)
    
    def _log_confusion_matrix(self, predictions, targets, prefix):
        """记录混淆矩阵"""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[self.disease_names.get(i, f'C{i}') for i in range(len(cm))],
                   yticklabels=[self.disease_names.get(i, f'C{i}') for i in range(len(cm))])
        plt.title(f'{prefix.capitalize()} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if self.logger:
            self.logger.experiment.add_figure(
                f'{prefix}_confusion_matrix',
                plt.gcf(),
                self.current_epoch
            )
        plt.close()
    
    def _compute_channel_distances(self, channel_locations):
        """计算通道间距离矩阵"""
        # 简化的距离计算
        B, C, _ = channel_locations.shape
        distances = torch.zeros(B, C, C, device=self.device)
        
        for b in range(B):
            locs = channel_locations[b]  # [C, 3]
            diff = locs.unsqueeze(1) - locs.unsqueeze(0)  # [C, C, 3]
            distances[b] = torch.norm(diff, dim=-1)  # [C, C]
            
        return distances
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 分层学习率配置
        param_groups = self._create_param_groups()
        
        if self.hparams.optimizer.optim == 'AdamW':
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.hparams.optimizer.weight_decay
            )
        elif self.hparams.optimizer.optim == 'Adam':
            optimizer = torch.optim.Adam(param_groups)
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
    
    def _create_param_groups(self):
        """创建参数组"""
        param_groups = []
        
        # LUNA编码器参数组
        encoder_params = []
        for name, param in self.model.named_parameters():
            if 'luna_encoder' in name:
                encoder_params.append(param)
        param_groups.append({
            'params': encoder_params,
            'lr': self.hparams.optimizer.param_groups.encoder,
            'weight_decay': self.hparams.optimizer.weight_decay
        })
        
        # 通道统一模块参数组
        channel_params = []
        for name, param in self.model.named_parameters():
            if 'channel_location_embedder' in name:
                channel_params.append(param)
        param_groups.append({
            'params': channel_params,
            'lr': self.hparams.optimizer.param_groups.channel_unifier,
            'weight_decay': self.hparams.optimizer.weight_decay
        })
        
        # 疾病专属模块参数组
        disease_params = []
        for name, param in self.model.named_parameters():
            if 'disease_processor' in name:
                disease_params.append(param)
        param_groups.append({
            'params': disease_params,
            'lr': self.hparams.optimizer.param_groups.disease_modules,
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
        
        return param_groups
    
    def predict_step(self, batch, batch_idx):
        """预测步骤"""
        eeg_data = batch['eeg']
        channel_locations = batch['channel_locations']
        
        # 推理时不需要真实的疾病ID
        logits, features = self.model(
            eeg_data,
            channel_locations=channel_locations,
            disease_ids=None  # 使用平均疾病特征
        )
        
        return {
            'predictions': torch.argmax(logits, dim=-1),
            'probabilities': F.softmax(logits, dim=-1),
            'features': features,
            'sample_info': {k: v for k, v in batch.items() if k not in ['eeg', 'channel_locations']}
        }