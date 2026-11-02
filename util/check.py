import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

# 路径对齐
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.Spatial_MSCNN import SpatialMSCNN_DualStream

# ==========================================
# 🎯 核心参数优化
# ==========================================
DATA_DIR = PROJECT_ROOT / "ds005408"
SAVE_DIR = PROJECT_ROOT / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32   # 减小 Batch 增加随机梯度噪声，防止陷入局部最优
LR = 5e-5         # 极其稳健的学习率
EPOCHS = 60

def main():
    print(f"🔥 开始直接训练双流网络 (19CH) | 设备: {DEVICE}")

    # 1. 加载并强制标准化
    x = np.load(DATA_DIR / "x_19ch.npy").astype(np.float32)
    y = np.load(DATA_DIR / "y_19ch.npy").astype(np.int64)
    
    # 全局 Z-Score 标准化
    x = (x - np.mean(x)) / (np.std(x) + 1e-6)
    print(f"✅ 数据标准化完成。样本总数: {len(y)}")

    # 2. 划分数据集 (确保类别比例平衡)
    train_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    
    train_set = Subset(list(zip(x, y)), train_idx)
    val_set = Subset(list(zip(x, y)), val_idx)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 初始化双流模型
    model = SpatialMSCNN_DualStream(num_classes=2, num_channels=19).to(DEVICE)
    
    # 增加 L2 正则化 (weight_decay) 防止过拟合
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()

        # 验证逻辑
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                out, _ = model(bx.to(DEVICE))
                all_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                all_targets.extend(by.numpy())
        
        acc = accuracy_score(all_targets, all_preds)
        print(f"Epoch {epoch+1:02d} | LR: {scheduler.get_last_lr()[0]:.6f} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.2%}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_DIR / "biofoundation_v1_best.pth")
            print(f"⭐ 发现更优权重: {best_acc:.2%}")

    print(f"\n✅ 训练完成！最高准确率: {best_acc:.2%}")

if __name__ == "__main__":
    main()