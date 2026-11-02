"""
Run full pipeline: preprocess `eeg-data` into `model-data` then train LUNA classifier.

Usage:
  python scripts/run_full_pipeline.py --eeg_path eeg-data --out_dir model-data --epochs 10

This script calls the existing `process_eeg_data.EEGDataProcessor` to create HDF5 files
and then invokes `scripts/run_luna_classification.py` to train the model.
"""
import os
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eeg_path', type=str, default='eeg-data')
    parser.add_argument('--out_dir', type=str, default='model_data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # 1) Preprocess
    print('Running preprocessing...')
    cmd = [
        'python',
        'process_eeg_data.py',
        '--eeg_data_path', args.eeg_path,
        '--output_path', args.out_dir,
        '--target_duration', '5',
        '--target_srate', '256',
    ]
    # call the script in-process
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        print('Preprocessing failed; aborting.')
        return

    # 2) Train
    print('Starting training...')
    train_cmd = [
        'python', 'scripts/run_luna_classification.py',
        '--data_dir', args.out_dir,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--device', args.device,
    ]
    subprocess.run(train_cmd, check=True)


if __name__ == '__main__':
    main()
