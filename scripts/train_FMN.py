import os
import sys
import argparse
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime
from safetensors.torch import load_file

# ==========================================
# 1. 🌟 核心修复：路径置顶策略
# 必须先处理 sys.path，再 import 自定义模型
# ==========================================
# 获取项目根目录 F:\lzy\BioFoundation-main
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print(f"🚀 系统路径初始化成功 | 项目根目录: {ROOT}")

# 路径处理完后再导入模型
try:
    from models.ST_FMN import BioFoundationDualStream
    print("✅ 成功导入模型架构：BioFoundationDualStream (LUNA + TCN)")
except ImportError as e:
    print(f"❌ 导入模型失败！请检查 models 文件夹内是否有 __init__.py")
    print(f"错误细节: {e}")
    sys.exit(1)

# --- 数据集定义 (保持不变) ---
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

# --- 增强与早停逻辑 ---
def femba_mixup_data(x, y, alpha=0.4, device='cuda'):
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

# --- 核心训练函数 (单折训练) ---
def run_train_fold(args, fold_dir, output_dir, device):
    # 🌟 数据路径容错处理
    def load_npy(m): 
        px = os.path.join(fold_dir, m, 'x.npy')
        py = os.path.join(fold_dir, m, 'y.npy')
        if not os.path.exists(px): raise FileNotFoundError(f"找不到数据: {px}")
        return np.load(px), np.load(py)
    
    try:
        tx, ty = load_npy('train'); vx, vy = load_npy('val')
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return 0, None

    train_loader = DataLoader(
        NPYDataset(tx, ty, augment=True, augment_multiplier=args.multiplier), 
        batch_size=args.batch_size, shuffle=True, drop_last=True 
    )
    val_loader = DataLoader(NPYDataset(vx, vy), batch_size=args.batch_size)

    # 实例化模型
    model = BioFoundationDualStream(
        num_classes=int(ty.max()+1), 
        num_channels=tx.shape[1], 
        luna_config=args.luna_config
    ).to(device)

    # 权重加载逻辑 (带维度安全检查)
    def safe_load(path, prefix):
        if os.path.exists(path):
            try:
                sd = load_file(path, device=str(device))
                md = model.state_dict()
                valid_sd = {f"{prefix}.{k}": v for k, v in sd.items() 
                           if f"{prefix}.{k}" in md and md[f"{prefix}.{k}"].shape == v.shape}
                model.load_state_dict(valid_sd, strict=False)
                print(f"    ✅ [{prefix}] 成功加载预训练参数: {len(valid_sd)}")
                return len(valid_sd) > 0
            except: pass
        return False

    has_luna = safe_load(args.luna_weight_path, "luna")
    has_femba = safe_load(args.femba_weight_path, "femba")

    # 冻结策略
    for name, param in model.named_parameters():
        if name.startswith("luna."): param.requires_grad = not has_luna
        elif name.startswith("femba."): param.requires_grad = not has_femba
        else: param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    history = {'train_loss': [], 'val_acc': []}

    print(f"    🚀 开始训练 | 训练集: {len(tx)} | 验证集: {len(vx)}")
    for epoch in range(args.epochs):
        model.train()
        t_losses = []
        optimizer.zero_grad() 
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            mx, ya, yb, lam = femba_mixup_data(x, y, args.alpha, device)
            logits, _ = model(mx)
            loss = femba_criterion(criterion, logits, ya, yb, lam) / args.accum_steps
            loss.backward()
            if (i + 1) % args.accum_steps == 0:
                optimizer.step(); optimizer.zero_grad()
            t_losses.append(loss.item() * args.accum_steps)
        
        v_acc, _, _ = eval_model(model, val_loader, device)
        avg_loss = np.mean(t_losses)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(v_acc)
        print(f"  └─ Epoch {epoch+1:03d} | Train Loss: {avg_loss:.4f} | Val Acc: {v_acc:.2%}")
        
        scheduler.step()
        early_stopping(v_acc, model, os.path.join(output_dir, 'best_model.pth'))
        if early_stopping.early_stop: break
    
    return early_stopping.val_acc_max, history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='model_data_cv') 
    parser.add_argument('--base_out', default='outputs_cv')
    parser.add_argument('--luna_weight_path', default=r"F:\lzy\BioFoundation-main\LUNA_base.safetensors")
    parser.add_argument('--femba_weight_path', default=r"F:\lzy\BioFoundation-main\FEMBA_base.safetensors")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--accum_steps', type=int, default=2) 
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--multiplier', type=int, default=1)
    parser.add_argument('--luna_config', default='LUNA_base')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.base_out, exist_ok=True)
    
    # 查找折目录
    fold_dirs = sorted([os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir) 
                       if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith('fold_')])
    
    if not fold_dirs:
        print(f"❌ 错误: 在 {args.data_dir} 下没找到 fold_ 开头的文件夹！")
        return

    print(f"📂 检测到 {len(fold_dirs)} 折数据，开始实验...")
    for i, fold_path in enumerate(fold_dirs):
        print(f"\n===== Fold {i+1} =====")
        out = os.path.join(args.base_out, f"fold_{i+1}")
        os.makedirs(out, exist_ok=True)
        run_train_fold(args, fold_path, out, device)

if __name__ == "__main__":
    main()