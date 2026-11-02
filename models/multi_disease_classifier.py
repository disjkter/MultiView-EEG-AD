"""
Multi-Disease EEG Classifier with Feature Disentanglement
多疾病EEG分类器与特征解耦模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import einops
from models.FEMBA import FEMBAEncoder
from models.modules.channel_embeddings import ChannelEmbedding
from models.modules.frequency_embedder import FrequencyEmbedder


class DiseaseSpecificEncoder(nn.Module):
    """疾病专属特征编码器"""
    def __init__(self, input_dim: int, embed_dim: int, depth: int, num_heads: int, 
                 num_diseases: int, dropout: float = 0.1):
        super().__init__()
        self.num_diseases = num_diseases
        
        # 疾病条件嵌入
        self.disease_embedding = nn.Embedding(num_diseases, embed_dim)
        
        # 专属变换网络
        self.transform_network = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=depth
            ) for _ in range(num_diseases)
        ])
        
        # 投影层
        self.projection = nn.Linear(input_dim, embed_dim)
        
    def forward(self, x: torch.Tensor, disease_ids: torch.Tensor):
        """
        Args:
            x: 输入特征 [B, C, T]
            disease_ids: 疾病ID [B]
        Returns:
            disease_features: 疾病专属特征 [B, embed_dim]
        """
        B, C, T = x.shape
        x = self.projection(x.transpose(-1, -2))  # [B, T, embed_dim]
        
        # 为每个样本选择对应的专属编码器
        disease_features = []
        for i in range(B):
            disease_idx = disease_ids[i].item()
            feature = self.transform_network[disease_idx](x[i:i+1])
            disease_features.append(feature.mean(dim=1))  # 全局池化
            
        return torch.cat(disease_features, dim=0)


class FeatureFusionModule(nn.Module):
    """特征融合模块 - 结合通用特征和疾病专属特征"""
    def __init__(self, shared_dim: int, disease_dim: int, fusion_dim: int, 
                 num_diseases: int, method: str = "attention_weighted"):
        super().__init__()
        self.method = method
        self.fusion_dim = fusion_dim
        
        if method == "attention_weighted":
            # 注意力加权融合
            self.shared_attention = nn.MultiheadAttention(
                embed_dim=fusion_dim, num_heads=8, batch_first=True
            )
            self.disease_attention = nn.MultiheadAttention(
                embed_dim=fusion_dim, num_heads=8, batch_first=True
            )
            
            # 投影层
            self.shared_proj = nn.Linear(shared_dim, fusion_dim)
            self.disease_proj = nn.Linear(disease_dim, fusion_dim)
            self.fusion_proj = nn.Linear(fusion_dim * 2, fusion_dim)
            
        elif method == "gated_fusion":
            # 门控融合
            self.gate = nn.Sequential(
                nn.Linear(shared_dim + disease_dim, fusion_dim),
                nn.Sigmoid()
            )
            self.shared_transform = nn.Linear(shared_dim, fusion_dim)
            self.disease_transform = nn.Linear(disease_dim, fusion_dim)
            
    def forward(self, shared_features: torch.Tensor, 
                disease_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shared_features: 通用特征 [B, shared_dim]
            disease_features: 疾病专属特征 [B, disease_dim]
        Returns:
            fused_features: 融合特征 [B, fusion_dim]
        """
        if self.method == "attention_weighted":
            shared_proj = self.shared_proj(shared_features).unsqueeze(1)  # [B, 1, fusion_dim]
            disease_proj = self.disease_proj(disease_features).unsqueeze(1)  # [B, 1, fusion_dim]
            
            # 交叉注意力
            shared_attended, _ = self.shared_attention(shared_proj, disease_proj, disease_proj)
            disease_attended, _ = self.disease_attention(disease_proj, shared_proj, shared_proj)
            
            # 拼接并投影
            fused = torch.cat([shared_attended.squeeze(1), disease_attended.squeeze(1)], dim=-1)
            return self.fusion_proj(fused)
            
        elif self.method == "gated_fusion":
            gate_input = torch.cat([shared_features, disease_features], dim=-1)
            gate_weight = self.gate(gate_input)
            
            shared_transformed = self.shared_transform(shared_features)
            disease_transformed = self.disease_transform(disease_features)
            
            return gate_weight * shared_transformed + (1 - gate_weight) * disease_transformed


class OrthogonalityLoss(nn.Module):
    """特征正交性损失 - 鼓励特征解耦"""
    def __init__(self, lambda_orth: float = 0.1):
        super().__init__()
        self.lambda_orth = lambda_orth
        
    def forward(self, shared_features: torch.Tensor, 
                disease_features: torch.Tensor) -> torch.Tensor:
        """
        计算特征正交性损失
        鼓励通用特征和疾病专属特征尽可能正交
        """
        # 归一化特征
        shared_norm = F.normalize(shared_features, p=2, dim=-1)
        disease_norm = F.normalize(disease_features, p=2, dim=-1)
        
        # 计算余弦相似度
        similarity = torch.abs(torch.sum(shared_norm * disease_norm, dim=-1))
        
        # 最小化相似度（最大化正交性）
        orthogonality_loss = similarity.mean()
        
        return self.lambda_orth * orthogonality_loss


class MultiDiseaseClassifier(nn.Module):
    """多疾病EEG分类器主模型"""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 通用特征编码器（共享）
        self.shared_encoder = FEMBAEncoder(
            embed_dim=config.shared_encoder.embed_dim,
            depth=config.shared_encoder.depth,
            num_heads=config.shared_encoder.num_heads,
            mlp_ratio=config.shared_encoder.mlp_ratio
        )
        
        # 疾病专属特征编码器
        self.disease_specific_encoder = DiseaseSpecificEncoder(
            input_dim=config.model.input_dim,
            embed_dim=config.disease_specific_encoders.embed_dim,
            depth=config.disease_specific_encoders.depth,
            num_heads=config.disease_specific_encoders.num_heads,
            num_diseases=config.disease_specific_encoders.num_diseases
        )
        
        # 特征融合模块
        self.feature_fusion = FeatureFusionModule(
            shared_dim=config.shared_encoder.embed_dim,
            disease_dim=config.disease_specific_encoders.embed_dim,
            fusion_dim=config.feature_fusion.fusion_dim,
            num_diseases=config.disease_specific_encoders.num_diseases,
            method=config.feature_fusion.fusion_method
        )
        
        # 分类头
        classifier_layers = []
        input_dim = config.feature_fusion.fusion_dim
        for hidden_dim in config.classifier.hidden_dims:
            classifier_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.classifier.dropout)
            ])
            input_dim = hidden_dim
        
        classifier_layers.append(nn.Linear(input_dim, config.classifier.num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        
        # 通道和频率嵌入
        self.channel_embedding = ChannelEmbedding(
            num_channels=config.model.num_channels,
            embed_dim=config.shared_encoder.embed_dim
        )
        self.frequency_embedder = FrequencyEmbedder(
            embed_dim=config.shared_encoder.embed_dim
        )
        
    def forward(self, x: torch.Tensor, disease_ids: Optional[torch.Tensor] = None):
        """
        Args:
            x: EEG信号 [B, C, T]
            disease_ids: 疾病ID [B]，推理时可为None
        Returns:
            logits: 分类logits [B, num_classes]
            features: 各层特征字典
        """
        B, C, T = x.shape
        
        # 1. 通用特征提取
        shared_features = self.shared_encoder(x)  # [B, seq_len, embed_dim]
        shared_pooled = shared_features.mean(dim=1)  # [B, embed_dim]
        
        # 2. 疾病专属特征提取
        if disease_ids is None:
            # 推理时使用所有疾病的平均特征
            disease_features = []
            dummy_ids = torch.arange(self.config.disease_specific_encoders.num_diseases, 
                                   device=x.device)
            for disease_id in dummy_ids:
                disease_ids_batch = disease_id.expand(B)
                disease_feat = self.disease_specific_encoder(x, disease_ids_batch)
                disease_features.append(disease_feat)
            disease_pooled = torch.stack(disease_features).mean(dim=0)
        else:
            disease_pooled = self.disease_specific_encoder(x, disease_ids)
        
        # 3. 特征融合
        fused_features = self.feature_fusion(shared_pooled, disease_pooled)
        
        # 4. 分类
        logits = self.classifier(fused_features)
        
        # 返回特征用于分析
        features = {
            'shared_features': shared_pooled,
            'disease_features': disease_pooled,
            'fused_features': fused_features,
            'attention_weights': None  # 可添加注意力权重
        }
        
        return logits, features
    
    def get_attention_weights(self, x: torch.Tensor):
        """获取注意力权重用于可视化"""
        with torch.no_grad():
            # 实现注意力权重提取逻辑
            pass
        return None


# 辅助损失函数
class MultiTaskLoss(nn.Module):
    """多任务损失函数组合"""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.orthogonality_loss = OrthogonalityLoss(config.criterion.orthogonality_loss.lambda_orth)
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, 
                features: Dict) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失
        """
        losses = {}
        
        # 主分类损失
        classification_loss = F.cross_entropy(logits, labels)
        losses['classification'] = classification_loss
        
        # 特征正交性损失
        orth_loss = self.orthogonality_loss(
            features['shared_features'], 
            features['disease_features']
        )
        losses['orthogonality'] = orth_loss
        
        # 总损失
        total_loss = (classification_loss + 
                     self.config.criterion.orthogonality_loss.lambda_orth * orth_loss)
        losses['total'] = total_loss
        
        return losses