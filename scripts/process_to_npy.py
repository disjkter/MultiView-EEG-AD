"""
Process `eeg-data` into NumPy arrays per split under `model-data/` (no HDF5).

Outputs:
  model-data/train/X.npy  (N, C, T)
  model-data/train/y.npy  (N,)
  model-data/val/X.npy
  model-data/val/y.npy
  model-data/test/X.npy
  model-data/test/y.npy

Usage:
  python scripts/process_to_npy.py --eeg_path eeg-data --out_dir model-data
"""
import os
import sys
import argparse
import numpy as np

# ensure project root is on path so we can import process_eeg_data
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from process_eeg_data import EEGDataProcessor


def save_split(split_samples, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    Xs = []
    ys = []
    meta = []
    for s in split_samples:
        Xs.append(s['X'])
        ys.append(int(s['y']))
        meta.append({
            'participant_id': s.get('participant_id'),
            'age': int(s.get('age', -1)) if s.get('age') is not None else -1,
            'gender': s.get('gender')
        })

    Xs = np.stack(Xs, axis=0).astype(np.float32)
    ys = np.array(ys, dtype=np.int64)

    np.save(os.path.join(out_dir, 'X.npy'), Xs)
    np.save(os.path.join(out_dir, 'y.npy'), ys)
    # save simple metadata
    np.save(os.path.join(out_dir, 'meta.npy'), np.array(meta, dtype=object))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eeg_path', type=str, default='eeg-data')
    parser.add_argument('--out_dir', type=str, default='model_data')
    parser.add_argument('--target_srate', type=int, default=256)
    parser.add_argument('--target_duration', type=int, default=5)
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='测试集比例')
    args = parser.parse_args()

    processor = EEGDataProcessor(
        args.eeg_path,
        args.out_dir,
        target_srate=args.target_srate,
        target_duration=args.target_duration,
    )
    samples = processor.process_all_data()
    if len(samples) == 0:
        print('No samples found after processing.')
        return

    splits = processor.split_data(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    save_split(splits['train'], os.path.join(args.out_dir, 'train'))
    save_split(splits['val'], os.path.join(args.out_dir, 'val'))
    save_split(splits['test'], os.path.join(args.out_dir, 'test'))

    print('Saved npy datasets under', args.out_dir)


if __name__ == '__main__':
    main()
