"""
简单的下游分类训练脚本，使用项目内 LUNAClassifier 和 model-data 下的 HDF5 数据。

用法示例:
  python scripts/run_luna_classification.py --data_dir model-data --epochs 10

输出：保存训练/验证准确率曲线和测试混淆矩阵到 `outputs/luna_classification`。
"""
import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


class HDF5EEGDataset(Dataset):
    """遍历 HDF5 文件中 data_group_* 的 X 数据。可选地接受外部标签数组。

    如果 HDF5 中每个 group 包含 'y'，会优先使用其内部标签；否则可通过 `labels` 参数
    传入全局标签数组（或为 None），数组按 dataset 遍历顺序排列。
    """
    def __init__(self, h5_path, labels=None):
        super().__init__()
        self.h5_path = h5_path
        if not os.path.exists(h5_path):
            raise FileNotFoundError(h5_path)

        # 收集索引 (group_name, idx_within_group)
        # index_map entries: (group_path_or_None, index_within_group, global_index)
        self.index_map = []
        global_counter = 0
        with h5py.File(self.h5_path, 'r') as f:
            # 尝试直接访问每个顶层对象，若包含 'X' 和 'y' 则作为样本来源
            for gname in f.keys():
                try:
                    grp = f[gname]
                    # 尝试以 group 的形式读取 'X'
                    X_ds = grp['X']
                    n = X_ds.shape[0]
                    if n > 0:
                        for i in range(n):
                            self.index_map.append((gname, i, global_counter))
                            global_counter += 1
                except Exception:
                    # 如果该顶层项不是 group 或不包含 X 则继续
                    continue

            # 如果没有找到任何 group 格式的数据，尝试 root 下的 X/y
            if not self.index_map:
                try:
                    if 'X' in f:
                        n = f['X'].shape[0]
                        for i in range(n):
                            self.index_map.append((None, i, global_counter))
                            global_counter += 1
                except Exception:
                    pass

            # 如果仍然为空，尝试递归搜索任意子组中包含 X 和 y 的组
            if not self.index_map:
                def find_group_with_xy(group):
                    for name, item in group.items():
                        try:
                            if 'X' in item and 'y' in item:
                                return name
                        except Exception:
                            # 不是 group/dataset
                            pass
                        # 递归搜索子组
                        if isinstance(item, h5py.Group):
                            sub = find_group_with_xy(item)
                            if sub:
                                return name + '/' + sub
                    return None

                nested = find_group_with_xy(f)
                if nested:
                    grp = f[nested]
                    n = grp['X'].shape[0]
                    for i in range(n):
                        self.index_map.append((nested, i, global_counter))
                        global_counter += 1

        # 如果没有找到样本，则抛出友好错误并列出文件顶级键以供排查
        if len(self.index_map) == 0:
            with h5py.File(self.h5_path, 'r') as f:
                keys = list(f.keys())
            raise ValueError(
                f"No samples found in HDF5 file: {self.h5_path}. Top-level keys: {keys}. "
                "Expected groups like 'data_group_*' containing 'X' and 'y', or root datasets 'X' and 'y'."
            )

        # labels handling: labels can be a numpy array, list, or path to file
        self.labels = None
        if labels is not None:
            # if labels is a path
            if isinstance(labels, str) and os.path.exists(labels):
                lb = None
                if labels.endswith('.npy'):
                    lb = np.load(labels)
                elif labels.endswith('.csv') or labels.endswith('.txt'):
                    lb = np.loadtxt(labels, dtype=int, delimiter=',')
                elif labels.endswith('.h5') or labels.endswith('.hdf5'):
                    with h5py.File(labels, 'r') as hf:
                        # try root 'y' or first dataset
                        if 'y' in hf:
                            lb = np.array(hf['y'])
                        else:
                            # try to concatenate group-level y
                            arrs = []
                            for k in hf.keys():
                                try:
                                    g = hf[k]
                                    if 'y' in g:
                                        arrs.append(np.array(g['y']))
                                except Exception:
                                    continue
                            if arrs:
                                lb = np.concatenate(arrs, axis=0)
                else:
                    raise ValueError(f"Unsupported labels file format: {labels}")

                if lb is None:
                    raise ValueError(f"Could not load labels from: {labels}")
                self.labels = np.asarray(lb)
            else:
                # assume labels is array-like
                self.labels = np.asarray(labels)

        # verify label length if provided
        if self.labels is not None:
            total = len(self.index_map)
            if len(self.labels) != total:
                raise ValueError(f"Provided labels length ({len(self.labels)}) does not match total samples ({total})")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        gname, i, global_idx = self.index_map[idx]
        with h5py.File(self.h5_path, 'r') as f:
            if gname is None:
                X = f['X'][i]
                # try to read internal y if exists, else use external labels
                if 'y' in f:
                    y = f['y'][i]
                elif self.labels is not None:
                    y = int(self.labels[global_idx])
                else:
                    raise ValueError('No label found for sample and no external labels provided')
            else:
                # support nested group paths like 'data_group_0' or 'group/sub'
                grp = f[gname]
                X = grp['X'][i]  # e.g., (C, T)
                if 'y' in grp:
                    y = grp['y'][i]
                elif self.labels is not None:
                    y = int(self.labels[global_idx])
                else:
                    raise ValueError(f'No label found in group {gname} and no external labels provided')

        X = np.asarray(X, dtype=np.float32)
        y = int(np.asarray(y))

        # Ensure shape (C, T) -> torch (C, T)
        X = torch.from_numpy(X)
        y = torch.tensor(y, dtype=torch.long)
        return X, y


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    preds_all = []
    targets_all = []
    for X, y in tqdm(loader, desc='train', leave=False):
        X = X.to(device)
        y = y.to(device)

        logits, _ = model(X)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds = logits.argmax(dim=1).cpu().numpy()
        preds_all.extend(preds)
        targets_all.extend(y.cpu().numpy())

    acc = accuracy_score(targets_all, preds_all) if len(targets_all) > 0 else 0.0
    return np.mean(losses), acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for X, y in tqdm(loader, desc='eval', leave=False):
            X = X.to(device)
            y = y.to(device)
            logits, _ = model(X)
            loss = criterion(logits, y)
            losses.append(loss.item())
            preds = logits.argmax(dim=1).cpu().numpy()
            preds_all.extend(preds)
            targets_all.extend(y.cpu().numpy())

    acc = accuracy_score(targets_all, preds_all) if len(targets_all) > 0 else 0.0
    return np.mean(losses) if losses else 0.0, acc, np.array(targets_all), np.array(preds_all)


def plot_accuracy(train_acc, val_acc, out_path):
    plt.figure()
    epochs = np.arange(1, len(train_acc) + 1)
    plt.plot(epochs, train_acc, label='train_acc')
    plt.plot(epochs, val_acc, label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(cm, classes, out_path):
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='model_data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out', type=str, default='outputs/luna_classification')
    parser.add_argument('--labels', type=str, default=None, help='Optional labels file (.npy, .csv, .h5) or leave empty to expect y in HDF5 groups')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    train_h5 = os.path.join(args.data_dir, 'train', 'train.h5')
    val_h5 = os.path.join(args.data_dir, 'val', 'val.h5')
    test_h5 = os.path.join(args.data_dir, 'test', 'test.h5')

    # try to find labels file if provided or in data_dir
    label_file = args.labels
    if label_file is None:
        # check common filenames in data_dir
        for cand in ['labels.npy', 'labels.csv', 'labels.h5']:
            p = os.path.join(args.data_dir, cand)
            if os.path.exists(p):
                label_file = p
                break

    train_ds = HDF5EEGDataset(train_h5, labels=label_file)
    val_ds = HDF5EEGDataset(val_h5, labels=label_file)
    test_ds = HDF5EEGDataset(test_h5, labels=label_file)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 导入项目内的 LUNAClassifier
    from models.luna_classifier import LUNAClassifier

    # 根据数据集推断通道数和类别数（简化：读取第一个样本）
    sample_X, sample_y = train_ds[0]
    num_channels = sample_X.shape[0]

    # 尝试从标签集合推断 num_classes
    all_train_labels = []
    for _, y in train_ds:
        all_train_labels.append(int(y))
    num_classes = int(max(all_train_labels) + 1) if all_train_labels else 2

    model = LUNAClassifier(num_classes=num_classes, num_channels=num_channels)
    device = torch.device(args.device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_accs = []
    val_accs = []

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = eval_one_epoch(model, val_loader, criterion, device)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"  train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # 保存中间模型
        torch.save(model.state_dict(), os.path.join(args.out, f'model_epoch_{epoch}.pt'))

    # 绘制准确率曲线
    plot_accuracy(train_accs, val_accs, os.path.join(args.out, 'accuracy_curve.png'))

    # 在测试集上评估并绘制混淆矩阵
    test_loss, test_acc, targets, preds = eval_one_epoch(model, test_loader, criterion, device)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")
    print(classification_report(targets, preds))
    cm = confusion_matrix(targets, preds)
    class_names = [str(i) for i in range(cm.shape[0])]
    plot_confusion_matrix(cm, class_names, os.path.join(args.out, 'confusion_matrix.png'))


if __name__ == '__main__':
    main()
