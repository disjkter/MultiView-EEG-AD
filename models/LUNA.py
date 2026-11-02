import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.layers import Mlp, DropPath

# 导入你项目中的模块（请确保路径正确）
from models.modules.rope_transformer_encoder_block import RotaryTransformerBlock
from models.modules.frequency_embedder import FrequencyFeatureEmbedder
from models.modules.channel_embeddings import ChannelEmbeddings

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

def nerf_positional_encoding(coords: torch.Tensor, embed_size: int) -> torch.Tensor:
    N, C, dim = coords.shape
    device = coords.device
    freqs = embed_size // (2 * dim)
    leftover = embed_size - freqs * 2 * dim
    freq_bands = 2.0 ** torch.arange(freqs, device=device).float()
    scaled_coords = coords.unsqueeze(-1) * freq_bands.view(1, 1, 1, -1)
    sin_enc = torch.sin(scaled_coords)
    cos_enc = torch.cos(scaled_coords)
    encoded = torch.stack([sin_enc, cos_enc], dim=-1).permute(0, 1, 3, 2, 4).reshape(N, C, freqs * dim * 2)
    if leftover > 0:
        pad = torch.zeros(N, C, leftover, device=device, dtype=coords.dtype)
        encoded = torch.cat([encoded, pad], dim=-1)
    return encoded.float() # 强制转为 float32

class ClassificationHeadWithQueries(nn.Module):
    """针对小样本优化的分类头"""
    def __init__(self, embed_dim=64, num_queries=4, num_heads=4, num_classes=3, dropout=0.5):
        super().__init__()
        self.embed_dim = int(embed_dim * num_queries)
        # 使用 MultiheadAttention 聚合所有 patch 的信息
        self.decoder_attn = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(self.embed_dim)
        # 增加 Dropout 层防止过拟合
        self.decoder_ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim * 2, num_classes)
        )
        self.learned_agg = nn.Parameter(torch.randn(1, 1, self.embed_dim))

    def forward(self, x):
        B, T, D = x.shape
        queries = self.learned_agg.repeat(B, 1, 1)
        x, _ = self.decoder_attn(query=queries, key=x, value=x)
        x = self.ln(x[:, 0, :])
        return self.decoder_ffn(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, num_queries, input_embed_dim, output_embed_dim, num_heads, dropout_p=0.3):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, input_embed_dim))
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_embed_dim, num_heads=num_heads, dropout=dropout_p, batch_first=True)
        self.queries_norm = nn.LayerNorm(input_embed_dim)
        self.keys_norm = nn.LayerNorm(input_embed_dim)
        self.values_norm = nn.LayerNorm(input_embed_dim)
        # 减少 self-attn 层数从 3 降到 1，适合小样本
        self.query_self_attn = nn.TransformerEncoderLayer(input_embed_dim, nhead=num_heads, batch_first=True, norm_first=True)

    def forward(self, x):
        B_t, C, D = x.size()
        queries = self.query_embed.repeat(B_t, 1, 1)
        queries = self.queries_norm(queries)
        x_out, attn = self.cross_attention(query=queries, key=self.keys_norm(x), value=self.values_norm(x))
        x_out = self.query_self_attn(x_out)
        return x_out, attn

class PatchEmbedNetwork(nn.Module):
    """EEG 信号切片嵌入层"""
    def __init__(self, embed_dim=64, patch_size=40):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = int(embed_dim // 4)
        # 使用 1D 卷积处理时间序列
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, self.out_channels, kernel_size=(1, 15), stride=(1, 5), padding=(0, 7)),
            nn.GroupNorm(4, self.out_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((None, 4)) # 压缩时间维度到固定长度
        )
    def forward(self, x):
        # x: (B, C, T)
        x = rearrange(x, 'B C (S P) -> B (C S) P', P=self.patch_size)
        x = x.unsqueeze(1) # (B, 1, CS, P)
        x = self.proj_in(x)
        x = rearrange(x, 'B E CS D -> B CS (D E)')
        return x.float()

class LUNA(nn.Module):
    def __init__(self, patch_size=40, num_queries=4, embed_dim=64, depth=3, num_heads=4, num_classes=3, dropout=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.patch_size = patch_size
        self.num_classes = num_classes

        # 1. 输入编码
        self.patch_embed = PatchEmbedNetwork(embed_dim=embed_dim, patch_size=patch_size)
        self.freq_embed = FrequencyFeatureEmbedder(embed_dim=embed_dim, patch_size=patch_size)
        self.channel_location_embedder = nn.Linear(embed_dim, embed_dim)

        # 2. 空间信息压缩 (Cross-Attention)
        self.cross_attn = CrossAttentionBlock(num_queries=num_queries, input_embed_dim=embed_dim, output_embed_dim=embed_dim, num_heads=num_heads, dropout_p=dropout)

        # 3. 核心 Transformer 层 (降低深度)
        self.blocks = nn.ModuleList([
            RotaryTransformerBlock(dim=int(embed_dim * num_queries), num_heads=num_heads, mlp_ratio=4., qkv_bias=True, drop=dropout, drop_path=0.1)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(int(embed_dim * num_queries))

        # 4. 分类头
        self.classifier = ClassificationHeadWithQueries(embed_dim=embed_dim, num_queries=num_queries, num_classes=num_classes, dropout=dropout)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x_signal, channel_locations):
        B, C, T = x_signal.shape
        num_patches_per_channel = T // self.patch_size
        
        # 特征提取
        x_p = self.patch_embed(x_signal) 
        x_f = self.freq_embed(x_signal).float()
        x = (x_p + x_f).float()

        # 位置编码处理
        c_min, c_max = channel_locations.min(dim=1, keepdim=True)[0], channel_locations.max(dim=1, keepdim=True)[0]
        c_loc = (channel_locations - c_min) / (c_max - c_min + 1e-8)
        c_loc_emb = nerf_positional_encoding(c_loc, self.embed_dim)
        c_loc_emb = self.channel_location_embedder(c_loc_emb)

        # 融合通道位置
        x = rearrange(x, 'B (C t) D -> (B t) C D', C=C)
        c_loc_emb = c_loc_emb.repeat(num_patches_per_channel, 1, 1)
        return x + c_loc_emb

    def forward(self, x_signal, mask=None, channel_locations=None, **kwargs):
        # 确保输入是 float32
        x_signal = x_signal.float()
        channel_locations = channel_locations.float()
        
        B, C, T = x_signal.shape
        
        # 1. 准备 Token
        x = self.prepare_tokens(x_signal, channel_locations)
        
        # 2. 空间压缩 (通道 -> Queries)
        x, _ = self.cross_attn(x) 
        x = rearrange(x, '(B t) Q D -> B t (Q D)', B=B) 
        
        # 3. 时间维度建模
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # 4. 分类输出
        logits = self.classifier(x)
        return logits, x