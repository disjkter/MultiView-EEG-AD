import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 🧱 纯 PyTorch 版 Mamba 核心模块 (免编译)
# ==========================================
class PureMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, d_conv=4):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, 
            out_channels=self.d_inner, 
            kernel_size=d_conv, 
            groups=self.d_inner, 
            padding=d_conv-1
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        B, L, D = x.shape
        
        # 1. 投影与门控
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        # 2. 因果一维卷积
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # 3. 状态空间参数投影 (B, C, Delta)
        x_dbl = self.x_proj(x_conv)
        delta, B_mat, C_mat = x_dbl.split([1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        
        # 4. 纯 PyTorch 选择性扫描 (Selective Scan) 替代循环
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        y = torch.zeros(B, L, self.d_inner, device=x.device)
        
        for t in range(L):
            xt = x_conv[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            dt = delta[:, t, :].unsqueeze(-1)   # (B, d_inner, 1)
            Bt = B_mat[:, t, :].unsqueeze(1)    # (B, 1, d_state)
            Ct = C_mat[:, t, :].unsqueeze(1)    # (B, 1, d_state)
            
            # 离散化状态转移
            h = h * torch.exp(-dt) + (dt * xt) * Bt
            yt = (h * Ct).sum(dim=-1)
            y[:, t, :] = yt
            
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out

# ==========================================
# 🧠 完整的 Spatial-Mamba 网络
# ==========================================
class SpatialMamba_Net(nn.Module):
    def __init__(self, num_classes=2, num_channels=19, time_steps=1024, d_model=64):
        super(SpatialMamba_Net, self).__init__()
        
        # 1. 空间流: 将 19 通道物理拓扑压缩为高维特征
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=(num_channels, 1), stride=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ELU()
        )
        
        # 2. Mamba 序列流: 捕获 1024 时间步的脑电长程依赖
        self.mamba = PureMambaBlock(d_model=d_model)
        
        # 3. 分类头
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, 19, 1024)
        x = x.unsqueeze(1) # (B, 1, 19, 1024)
        
        x = self.spatial_conv(x) # (B, d_model, 1, 1024)
        x = x.squeeze(2)         # (B, d_model, 1024)
        
        # Mamba 要求的输入格式为 (Batch, Seq_Len, d_model)
        x = x.permute(0, 2, 1)   # (B, 1024, d_model)
        
        # 穿过纯血 Mamba 核心
        out = self.mamba(x)      # (B, 1024, d_model)
        
        # 时间轴全局平均池化
        pooled = torch.mean(out, dim=1) # (B, d_model)
        
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits, pooled