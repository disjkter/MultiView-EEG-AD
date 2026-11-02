import os
import sys
import argparse
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from safetensors.torch import load_file

# 路径自动处理
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from models.luna_classifier import LUNAClassifierWithAttention

# --- 1. 数据集定义 (保持不变) ---
class NPYDataset(Dataset):
    def __init__(self, x_data, y_data, augment=False, augment_multiplier=1):
        self.X = x_data.astype(np.float32)
        self.y = y_data.astype(np.int64)
        self._normalize_all()
        self.augment = augment
        self.multiplier = augment_multiplier
        self.length = len(self.y) * augment_multiplier if augment else len(self.y)

    def _normalize_all(self):
        for i in range(len(self.X)):
            m = self.X[i].mean(axis=1, keepdims=True)
            s = self.X[i].std(axis=1, keepdims=True) + 1e-8
            self.X[i] = (self.X[i] - m) / s

    def __len__(self): return self.length

    def __getitem__(self, idx):
        real_idx = idx % len(self.y)
        x, y = self.X[real_idx].copy(), self.y[real_idx]
        if self.augment:
            if np.random.rand() > 0.5: x += np.random.normal(0, 0.005, x.shape).astype(np.float32)
            if np.random.rand() > 0.5: x = np.roll(x, np.random.randint(-3, 4), axis=-1)
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)

# --- 2. FEMBA 增强与早停逻辑 (保持不变) ---
def femba_mixup_data(x, y, alpha=0.2, device='cuda'):
    if alpha <= 0: return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam

def femba_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class EarlyStopping:
    def __init__(self, patience=20, verbose=True, delta=0):
        self.patience, self.verbose, self.delta = patience, verbose, delta
        self.counter, self.best_score, self.early_stop, self.val_acc_max = 0, None, False, 0

    def __call__(self, val_acc, model, path):
        score = float(val_acc)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: print(f'    [早停] 计数: {self.counter}/{self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, path)
            self.counter = 0

    def save_checkpoint(self, val_acc, model, path):
        if self.verbose: print(f'    ⭐ 验证集最佳准确率更新: {self.val_acc_max:.4f} --> {val_acc:.4f}')
        torch.save(model.state_dict(), path)
        self.val_acc_max = val_acc

def eval_model(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device).float(), y.to(device)
            logits, _ = model(x)
            preds.extend(logits.argmax(1).cpu().numpy()); targets.extend(y.cpu().numpy())
    return accuracy_score(targets, preds), np.array(targets), np.array(preds)

# --- 3. 核心训练函数 (单折训练) ---
def run_train_fold(args, fold_dir, output_dir, device):
    def load_npy(m): 
        return np.load(os.path.join(fold_dir, m, 'x.npy')), np.load(os.path.join(fold_dir, m, 'y.npy'))
    
    tx, ty = load_npy('train'); vx, vy = load_npy('val')
    
    # ✅ 核心修复：添加 drop_last=True 解决 BatchNorm1d 单批次崩溃问题
    train_loader = DataLoader(
        NPYDataset(tx, ty, augment=True, augment_multiplier=args.multiplier), 
        batch_size=args.batch_size, 
        shuffle=True,
        drop_last=True 
    )
    val_loader = DataLoader(NPYDataset(vx, vy), batch_size=args.batch_size)

    model = LUNAClassifierWithAttention(num_classes=int(ty.max()+1), num_channels=tx.shape[1], luna_config=args.luna_config).to(device)

    # 注入权重
    if os.path.exists(args.weight_path):
        sd = load_file(args.weight_path, device=str(device))
        md = model.state_dict()
        new_sd = {f"backbone.{k}": v for k, v in sd.items() if f"backbone.{k}" in md and md[f"backbone.{k}"].shape == v.shape}
        model.load_state_dict(new_sd, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # 开启 early_stopping 的 verbose，以便在终端看到每次更新
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(args.epochs):
        model.train()
        t_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            mx, ya, yb, lam = femba_mixup_data(x, y, args.alpha, device)
            logits, _ = model(mx)
            loss = femba_criterion(criterion, logits, ya, yb, lam)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            t_losses.append(loss.item())
        
        v_acc, _, _ = eval_model(model, val_loader, device)
        avg_loss = np.mean(t_losses)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(v_acc)
        
        # 优化打印，让你实时看到训练状态
        print(f"  └─ Epoch {epoch+1:03d} | Train Loss: {avg_loss:.4f} | Val Acc: {v_acc:.2%}")
        
        early_stopping(v_acc, model, os.path.join(output_dir, 'best_model.pth'))
        if early_stopping.early_stop: 
            print("    🛑 触发早停，结束本折训练。")
            break
    
    return early_stopping.val_acc_max, history

# --- 4. 主程序 (多折循环) ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='model_data_cv') 
    parser.add_argument('--base_out', default='outputs_cv')
    parser.add_argument('--weight_path', default="LUNA_base.safetensors")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--multiplier', type=int, default=10)
    parser.add_argument('--luna_config', default='LUNA_base')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    session_dir = os.path.join(args.base_out, timestamp)
    os.makedirs(session_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fold_dirs = sorted([os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir) if d.startswith('fold_')])
    print(f"🚀 实验配置加载完毕 | Batch Size: {args.batch_size} | 数据增强倍数: {args.multiplier}")
    print(f"🚀 检测到 {len(fold_dirs)} 折数据，开始交叉验证实验...")

    fold_results = []
    
    for fold_idx, fold_path in enumerate(fold_dirs):
        fold_num = fold_idx + 1
        fold_out = os.path.join(session_dir, f"fold_{fold_num}")
        os.makedirs(fold_out, exist_ok=True)
        
        print(f"\n==================================================")
        print(f"🌀 正在进行第 {fold_num}/{len(fold_dirs)} 折训练 (Fold Path: {fold_path})")
        print(f"==================================================")
        
        best_acc, history = run_train_fold(args, fold_path, fold_out, device)
        fold_results.append(best_acc)
        
        plt.figure(); plt.plot(history['val_acc']); plt.title(f'Fold {fold_num} Val Acc')
        plt.savefig(os.path.join(fold_out, 'curve.png')); plt.close()
        
        print(f"✅ 第 {fold_num} 折训练完毕！本折最高验证集准确率: {best_acc:.2%}")

    # --- 最终汇总 ---
    avg_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    
    summary_txt = f"""
==============================
10-Fold Cross Validation Report
==============================
Date: {timestamp}
Config: {args.luna_config}
Batch Size: {args.batch_size}

Results per Fold:
{ {f'Fold_{i+1}': f'{res:.4f}' for i, res in enumerate(fold_results)} }

------------------------------
Mean Accuracy: {avg_acc:.4%}
Std Deviation: {std_acc:.4%}
==============================
"""
    print(summary_txt)
    with open(os.path.join(session_dir, "cv_report.txt"), 'w') as f:
        f.write(summary_txt)

if __name__ == "__main__":
    main()