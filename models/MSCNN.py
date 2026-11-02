import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 3
        rem_channels = out_channels - 2 * branch_channels
        
        # 三个频段的分支
        self.branch1 = nn.Conv1d(in_channels, branch_channels, kernel_size=51, padding=25)
        self.branch2 = nn.Conv1d(in_channels, branch_channels, kernel_size=25, padding=12)
        self.branch3 = nn.Conv1d(in_channels, rem_channels, kernel_size=11, padding=5)
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.gelu(self.bn(out))


class MSCNN(nn.Module):
    def __init__(self, in_channels, embed_dim=256):
        super().__init__()
        
        self.ms_conv1 = MultiScaleBlock(in_channels, 64)
        self.pool1 = nn.MaxPool1d(4)
        
        self.ms_conv2 = MultiScaleBlock(64, 128)
        self.pool2 = nn.MaxPool1d(4)
        
        self.ms_conv3 = MultiScaleBlock(128, 256)
        
        # SE Block 通道注意力
        self.se_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.se_fc = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 256),
            nn.Sigmoid()
        )
        
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x):
        x = self.ms_conv1(x)
        x = self.pool1(x)
        x = self.ms_conv2(x)
        x = self.pool2(x)
        x = self.ms_conv3(x)
        
        se_weight = self.se_avg_pool(x).squeeze(-1)
        se_weight = self.se_fc(se_weight).unsqueeze(-1)
        x = x * se_weight 
        
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        out = self.proj(x)
        return out