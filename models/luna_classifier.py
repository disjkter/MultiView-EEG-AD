import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入你项目中的 LUNA 骨干网络 (注意大小写)
from .LUNA import LUNA

class AttentionPooling(nn.Module):
    """
    时间注意力池化层：自动寻找时间点中对脑电信号分类最重要的波段
    """
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, x):
        # x shape: (Batch, SeqLen, Features)
        w = self.attention(x)
        attn_weights = F.softmax(w, dim=1)
        weighted_features = torch.sum(x * attn_weights, dim=1)
        return weighted_features

class LUNAClassifierWithAttention(nn.Module):
    def __init__(self, num_classes, num_channels, luna_config='LUNA_base'):
        super().__init__()
        
        # --- 1. 严格对齐官方 LUNA 配置 ---
        config_map = {
            'LUNA_base': {
                'patch_size': 40,
                'embed_dim': 64,
                'num_heads': 4,      
                'depth': 6,          
                'num_queries': 4,
            },
            'LUNA_small': {
                'patch_size': 40,
                'embed_dim': 64,
                'num_heads': 4,
                'depth': 3,
                'num_queries': 4,
            }
        }
        
        params = config_map.get(luna_config, config_map['LUNA_base']).copy()
        params.pop('drop_path', None) 
        
        # --- 2. 实例化 LUNA 骨干网络 ---
        self.backbone = LUNA(**params, num_classes=0)
        
        # 🚀 绝杀修复：LUNA 的真实输出维度 = embed_dim * num_queries (64 * 4 = 256)
        self.backbone_out_dim = params['embed_dim'] * params['num_queries']
        hidden_dim = self.backbone_out_dim // 2
        
        # --- 3. 时间注意力池化 ---
        self.attention_pool = AttentionPooling(self.backbone_out_dim)
        
        # --- 4. 核心物理改造 ---
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),    
            nn.GELU(),
            nn.Dropout(0.5),             
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, Channels, SeqLen)
        B, C, T = x.shape
        
        # 自动生成虚拟的通道空间坐标
        dummy_locations = torch.zeros((B, C, 3), device=x.device)
        dummy_locations[:, :, 0] = torch.arange(C, device=x.device).unsqueeze(0)
        
        # 1. 骨干网络提取特征 
        out = self.backbone(x, channel_locations=dummy_locations)
        
        if isinstance(out, tuple):
            features = out[1]  
        else:
            features = out
        
        # 2. 注意力机制聚合时间维度特征 
        weighted_features = self.attention_pool(features)
        
        # 3. 分类器输出结果 
        logits = self.classifier(weighted_features)
        
        return logits, weighted_features