import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pathlib import Path

# 环境注入
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from models.Spatial_MSCNN import SpatialMSCNN_DualStream
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

class FineTuneDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.X = x_data.astype(np.float32)
        self.y = y_data.astype(np.int64)
        for i in range(len(self.X)):
            m = self.X[i].mean(axis=1, keepdims=True)
            s = self.X[i].std(axis=1, keepdims=True) + 1e-8
            self.X[i] = (self.X[i] - m) / s
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.long)

def run_single_experiment(seed, x_all, y_all, ids_all, model_weights, device):
    """单次随机实验：拆分、微调并返回受试者级别准确率"""
    unique_subs = np.unique(ids_all)
    sub_labels = [y_all[np.where(ids_all == uid)[0][0]] for uid in unique_subs]
    
    # 随机拆分：20% 微调，80% 测试
    train_subs, test_subs, _, _ = train_test_split(
        unique_subs, sub_labels, test_size=0.80, stratify=sub_labels, random_state=seed
    )

    # 准备数据加载器
    train_idx = np.isin(ids_all, train_subs)
    test_idx = np.isin(ids_all, test_subs)
    
    train_loader = DataLoader(FineTuneDataset(x_all[train_idx], y_all[train_idx]), batch_size=32, shuffle=True)
    test_loader = DataLoader(FineTuneDataset(x_all[test_idx], y_all[test_idx]), batch_size=64, shuffle=False)

    # 加载原始巅峰模型
    model = SpatialMSCNN_DualStream(num_classes=2, num_channels=19).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 微调 15 个 Epoch
    model.train()
    for _ in range(15):
        for vx, vy in train_loader:
            vx, vy = vx.to(device), vy.to(device)
            optimizer.zero_grad()
            logits, _ = model(vx)
            loss = criterion(logits, vy)
            loss.backward()
            optimizer.step()

    # 盲测评估
    model.eval()
    epoch_probs, epoch_targets = [], []
    with torch.no_grad():
        for vx, vy in test_loader:
            logits, _ = model(vx.to(device))
            epoch_probs.extend(F.softmax(logits, dim=1).cpu().numpy())
            epoch_targets.extend(vy.numpy())

    # 受试者级别投票
    sub_preds, sub_targets = [], []
    ids_test = ids_all[test_idx]
    for uid in test_subs:
        idx = np.where(ids_test == uid)[0]
        patient_prob = np.mean(np.array(epoch_probs)[idx], axis=0)
        sub_preds.append(np.argmax(patient_prob))
        sub_targets.append(epoch_targets[idx[0]])
    
    return accuracy_score(sub_targets, sub_preds)

def main():
    EXTERNAL_DATA_DIR = Path(r"E:\ProgramFiles\PythonProject\BioFoundation-main\external_test_data")
    MODEL_WEIGHTS = r"E:\ProgramFiles\PythonProject\BioFoundation-main\best_model.pth"
    
    x_all = np.load(EXTERNAL_DATA_DIR / 'x.npy')
    y_all = np.load(EXTERNAL_DATA_DIR / 'y.npy')
    ids_all = np.load(EXTERNAL_DATA_DIR / 'sub_ids.npy')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 🌟 蒙特卡洛设置：运行 5 次不同种子的实验
    seeds = [42, 123, 2024, 777, 999]
    results = []

    print("\n" + "💎"*25)
    print(f"🚀 开启蒙特卡洛跨域验证 (共 {len(seeds)} 轮随机实验)")
    print("💎"*25)

    for i, seed in enumerate(seeds):
        print(f"🔄 正在运行第 {i+1} 轮 (Seed: {seed})...", end=" ", flush=True)
        acc = run_single_experiment(seed, x_all, y_all, ids_all, MODEL_WEIGHTS, device)
        results.append(acc)
        print(f"Done! Acc: {acc:.2%}")

    # 计算统计量
    mean_acc = np.mean(results)
    std_acc = np.std(results)

    print("\n" + "="*50)
    print(f"📊 最终统计报告 (Monte Carlo Results):")
    print(f"   ├─ 各轮结果: {[f'{r:.2%}' for r in results]}")
    print(f"   ├─ 平均准确率 (Mean Acc): {mean_acc:.2%}")
    print(f"   └─ 标准差 (Std Dev): ±{std_acc:.2%}")
    print("="*50)
    print("🌟 这份带误差棒的数据，就是你论文中最强有力的证明！")

if __name__ == "__main__":
    main()