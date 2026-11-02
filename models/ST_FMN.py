import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.LUNA import LUNA
    from models.MSCNN import MSCNN
except ImportError:
    from .LUNA import LUNA
    from .MSCNN import MSCNN

class BioFoundationDualStream(nn.Module):
    # 🌟 新增 coords_path 参数
    def __init__(self, num_classes, num_channels, luna_config='LUNA_base', femba_out_dim=256, coords_path=None):
        super().__init__()
        
        config_map = {
            'LUNA_base': {'patch_size': 40, 'embed_dim': 64, 'num_heads': 4, 'depth': 6, 'num_queries': 4},
        }
        luna_params = config_map.get(luna_config, config_map['LUNA_base']).copy()
        self.luna = LUNA(**luna_params, num_classes=0)
        self.dim_luna = luna_params['embed_dim'] * luna_params['num_queries'] 
        self.femba = MSCNN(in_channels=num_channels, embed_dim=femba_out_dim) 
        
        fused_dim = self.dim_luna + femba_out_dim 
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.LayerNorm(fused_dim // 2),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(fused_dim // 2, num_classes)
        )

        # ==========================================
        # 🌟 坐标系挂载逻辑
        # ==========================================
        if coords_path and os.path.exists(coords_path):
            coords = np.load(coords_path)
            # 使用 register_buffer，这样坐标会随着模型自动存入 GPU，但不参与梯度更新
            self.register_buffer('real_channel_coords', torch.from_numpy(coords).float())
        else:
            self.real_channel_coords = None

    def forward(self, x):
        B, C, T = x.shape
        
        # ==========================================
        # 🌟 动态坐标注入
        # ==========================================
        if self.real_channel_coords is not None:
            # 将 (C, 3) 的坐标库扩展为匹配 Batch Size 的 (B, C, 3)
            locations = self.real_channel_coords.unsqueeze(0).expand(B, -1, -1)
        else:
            # 兼容模式：如果没有找到坐标文件，退化为假坐标
            locations = torch.zeros((B, C, 3), device=x.device)
            locations[:, :, 0] = torch.arange(C, device=x.device).unsqueeze(0)
            
        # 把带有灵魂的真实三维坐标传给 LUNA！
        out_luna = self.luna(x, channel_locations=locations)
        feat_luna = out_luna[1] if isinstance(out_luna, tuple) else out_luna
        
        # 维度安全锁
        if feat_luna.dim() > 2:
            feat_luna = feat_luna.flatten(start_dim=1)
        if feat_luna.shape[1] != self.dim_luna:
            feat_luna = F.adaptive_avg_pool1d(feat_luna.unsqueeze(1), self.dim_luna).squeeze(1)
            
        feat_femba = self.femba(x)
        fused_features = torch.cat([feat_luna, feat_femba], dim=-1)
        logits = self.fusion_classifier(fused_features)
        
        return logits, fused_features