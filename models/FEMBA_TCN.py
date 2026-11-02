import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

# ==========================================
# 核心组件 1: Chomp1d (因果剪裁)
# ==========================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# ==========================================
# 核心组件 2: TemporalBlock (TCN 残差块)
# ==========================================
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 第一层扩张卷积
        self.conv1 = nn_utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二层扩张卷积
        self.conv2 = nn_utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # 残差连接的维度匹配
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.GELU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# ==========================================
# 核心类: FEMBA_TCN (时频专家)
# ==========================================
class FEMBA_TCN(nn.Module):
    """
    针对阿尔茨海默症 EEG 长序列设计的特征提取器 (平替 Mamba)。
    输入形状: (Batch, Channels, Time)
    输出形状: (Batch, embed_dim)
    """
    def __init__(self, in_channels, embed_dim=256, num_channels=[64, 128, 256], kernel_size=3, dropout=0.2):
        super(FEMBA_TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            # 扩张率指数增长: 1, 2, 4, 8...
            dilation_size = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            
            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size, dropout=dropout
            ))

        self.tcn = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1) # 时间维度全局池化
        self.projection = nn.Linear(num_channels[-1], embed_dim)

    def forward(self, x):
        # 1. TCN 提取: (B, C, T) -> (B, Last_Ch, T)
        x = self.tcn(x)
        
        # 2. 池化压缩时间维: (B, Last_Ch, T) -> (B, Last_Ch)
        x = self.adaptive_pool(x).flatten(1)
        
        # 3. 投影到融合维度: (B, Last_Ch) -> (B, 256)
        x = self.projection(x)
        return x