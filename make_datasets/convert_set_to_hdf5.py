"""
#*----------------------------------------------------------------------------*
#* Copyright (C) 2025 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author: Your Name                                                          *
#*----------------------------------------------------------------------------*
"""

import os
import sys
import pickle
import numpy as np
import mne
from scipy.io import loadmat
from tqdm import tqdm
import argparse
import glob
import shutil

# 尝试导入h5py，如果失败则在需要时抛出错误
try:
    import h5py
except ImportError:
    h5py = None
    print("Warning: h5py module is not installed. HDF5 functionality will be unavailable.", file=sys.stderr)


def print_usage_examples():
    """打印使用示例"""
    print("\n使用示例:", file=sys.stderr)
    print("  # 基本用法 - 转换所有.set文件", file=sys.stderr)
    print("  python convert_set_to_hdf5.py --set_dir ../eeg-data/derivatives --output_dir ../model-data", file=sys.stderr)
    print("", file=sys.stderr)
    print("  # 指定窗口大小和标签", file=sys.stderr)
    print("  python convert_set_to_hdf5.py --set_dir ../eeg-data/derivatives --output_dir ../model-data --window_duration 10 --label 1", file=sys.stderr)
    print("", file=sys.stderr)
    print("  # 预训练模式（不包含标签）", file=sys.stderr)
    print("  python convert_set_to_hdf5.py --set_dir ../eeg-data/derivatives --output_dir ../model-data --task_type pretrain", file=sys.stderr)
    print("", file=sys.stderr)


def validate_directories(args):
    """验证输入和输出目录"""
    # 检查输入目录
    if not os.path.exists(args.set_dir):
        print(f"错误: 输入目录 '{args.set_dir}' 不存在", file=sys.stderr)
        return False
    
    if not os.path.isdir(args.set_dir):
        print(f"错误: '{args.set_dir}' 不是一个目录", file=sys.stderr)
        return False
    
    # 检查是否有.set文件（递归搜索）
    set_files = []
    for root, dirs, files in os.walk(args.set_dir):
        for file in files:
            if file.endswith('.set'):
                set_files.append(os.path.join(root, file))
    
    if not set_files:
        print(f"错误: 在目录 '{args.set_dir}' 及其子目录中没有找到 .set 文件", file=sys.stderr)
        return False
    
    print(f"找到 {len(set_files)} 个.set文件")
    
    # 检查输出目录结构
    required_dirs = ['train', 'val', 'test']
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = os.path.join(args.output_dir, dir_name)
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"错误: 输出目录 '{args.output_dir}' 中缺少以下必需的子目录:", file=sys.stderr)
        for missing_dir in missing_dirs:
            print(f"  - {missing_dir}", file=sys.stderr)
        print("\n请先创建这些目录:", file=sys.stderr)
        print(f"  mkdir -p {args.output_dir}/{{train,val,test}}", file=sys.stderr)
        return False
    
    return True


def read_simple_set_file(set_file_path):
    """
    读取简化格式的.set文件（直接包含EEG数据字段）
    返回类似MNE Raw对象的数据结构
    """
    # 加载MATLAB文件
    mat_data = loadmat(set_file_path)
    
    # 提取必要的字段
    if 'data' not in mat_data:
        raise ValueError("在.set文件中找不到'data'字段")
    
    # EEG数据 [channels x samples]
    eeg_data = mat_data['data']
    
    # 采样率
    sampling_rate = float(mat_data.get('srate', 256)[0, 0])  # 默认256Hz
    
    # 通道名称
    ch_names = None
    if 'chanlocs' in mat_data:
        chanlocs = mat_data['chanlocs']
        if chanlocs.size > 0:
            try:
                # 尝试提取通道名称
                if hasattr(chanlocs[0, 0], '_fieldnames'):
                    fieldnames = chanlocs[0, 0]._fieldnames
                    if 'labels' in fieldnames:
                        labels = chanlocs[0, 0]['labels']
                        ch_names = [str(labels[i][0][0]) for i in range(len(labels))]
                    elif 'name' in fieldnames:
                        names = chanlocs[0, 0]['name']
                        ch_names = [str(names[i][0][0]) for i in range(len(names))]
            except:
                pass
    
    # 如果没有通道名称，使用默认名称
    if ch_names is None:
        ch_names = [f'EEG{i:03d}' for i in range(eeg_data.shape[0])]
    
    # 确保通道名称数量匹配
    if len(ch_names) != eeg_data.shape[0]:
        print(f"警告: 通道名称数量({len(ch_names)})与数据通道数({eeg_data.shape[0]})不匹配，使用默认名称")
        ch_names = [f'EEG{i:03d}' for i in range(eeg_data.shape[0])]
    
    # 创建简单的类MNE对象
    class SimpleRaw:
        def __init__(self, data, ch_names, sfreq):
            self._data = data
            self.ch_names = ch_names
            self.info = {'sfreq': sfreq}
        
        def get_data(self):
            return self._data
        
        def pick_channels(self, ch_names, ordered=False):
            indices = [self.ch_names.index(ch) for ch in ch_names if ch in self.ch_names]
            self._data = self._data[indices, :]
            self.ch_names = [self.ch_names[i] for i in indices]
            return self
        
        def filter(self, l_freq, h_freq, verbose=False):
            # 简单的滤波模拟（实际应用中应该使用proper filtering）
            print(f"应用滤波: {l_freq}-{h_freq} Hz (简化实现)")
            return self
        
        def notch_filter(self, freq, verbose=False):
            print(f"应用陷波滤波: {freq} Hz (简化实现)")
            return self
        
        def resample(self, sfreq, **kwargs):
            print(f"重采样到 {sfreq} Hz (简化实现)")
            self.info['sfreq'] = sfreq
            return self
    
    return SimpleRaw(eeg_data, ch_names, sampling_rate)


def read_set_file(set_file_path):
    """
    Read .set file using EEGLAB .set/.fdt format
    Returns MNE Raw object or SimpleRaw object for simplified format
    """
    # .set files are MATLAB struct files that contain EEG struct
    # Usually accompanied by a .fdt file with the actual data
    mat_data = loadmat(set_file_path)
    
    # Extract EEG struct from loaded MATLAB data
    eeg_struct = None
    for key in mat_data:
        if key.startswith('EEG') and isinstance(mat_data[key], np.ndarray) and mat_data[key].size > 0:
            eeg_struct = mat_data[key][0, 0]  # Get the struct
            break
    
    if eeg_struct is None:
        # 如果找不到EEG结构，尝试使用简化格式读取
        print("未找到标准EEGLAB格式，尝试简化格式读取...", file=sys.stderr)
        return read_simple_set_file(set_file_path)
    
    # Extract fields from the EEG struct
    # Field names may vary depending on EEGLAB version
    data_field_names = ['data', 'dat', 'x', 'EEG']
    fs_field_names = ['srate', 'fs', 'sampFreq', 'sfreq']
    chanlocs_field_names = ['chanlocs', 'chnlocs', 'chans', 'electrodes']
    
    # Try to find data
    eeg_data = None
    for field_name in data_field_names:
        if hasattr(eeg_struct, field_name):
            eeg_data = getattr(eeg_struct, field_name)
            break
    
    # Transpose if needed (EEGLAB stores as [n_channels x n_samples])
    if eeg_data is not None and eeg_data.shape[0] < eeg_data.shape[1]:
        eeg_data = eeg_data.T
    
    # Try to find sampling rate
    sampling_rate = None
    for field_name in fs_field_names:
        if hasattr(eeg_struct, field_name):
            sampling_rate = float(getattr(eeg_struct, field_name).item())
            break
    
    # Try to find channel locations
    chanlocs = None
    for field_name in chanlocs_field_names:
        if hasattr(eeg_struct, field_name):
            chanlocs = getattr(eeg_struct, field_name)
            break
    
    # Create channel names
    if chanlocs is not None:
        if isinstance(chanlocs, np.ndarray) and len(chanlocs) > 0:
            # Handle different ways channel names could be stored
            if hasattr(chanlocs[0], 'labels') and len(chanlocs[0]['labels']) > 0:
                ch_names = [str(label[0]) if isinstance(label, np.ndarray) and len(label) > 0 
                           else str(label) for label in chanlocs[0]['labels'][0]]
            elif hasattr(chanlocs[0], 'name') and len(chanlocs[0]['name']) > 0:
                ch_names = [str(name[0]) if isinstance(name, np.ndarray) and len(name) > 0 
                           else str(name) for name in chanlocs[0]['name'][0]]
            else:
                ch_names = [f'EEG {i:03d}' for i in range(eeg_data.shape[1])]
        else:
            ch_names = [f'EEG {i:03d}' for i in range(eeg_data.shape[1])]
    else:
        ch_names = [f'EEG {i:03d}' for i in range(eeg_data.shape[1])]
    
    # Create MNE Info object
    if sampling_rate is None:
        # Default to 256 Hz if not found
        sampling_rate = 256
        print(f"Warning: Sampling rate not found in .set file, defaulting to {sampling_rate} Hz")
    
    # 确保通道名称数量与数据匹配
    if len(ch_names) != eeg_data.shape[0]:
        print(f"警告: 通道名称数量({len(ch_names)})与数据通道数({eeg_data.shape[0]})不匹配，使用默认名称")
        ch_names = [f'EEG {i:03d}' for i in range(eeg_data.shape[0])]
    
    info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types='eeg')
    
    # Create Raw object
    raw = mne.io.RawArray(eeg_data, info)
    
    return raw


def preprocess_raw(raw):
    """
    Apply standard preprocessing pipeline similar to the project's process_raw_eeg.py
    """
    # Standard channel ordering for consistency
    CH_ORDER_STANDARD = [
        'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ'
    ]
    
    # Try to rename channels to match standard format if needed
    ch_mapping = {}
    for ch_name in raw.ch_names:
        clean_name = ch_name.upper().replace('EEG ', '').replace('-REF', '').replace('-AV', '')
        if clean_name in CH_ORDER_STANDARD:
            ch_mapping[ch_name] = clean_name
        else:
            # Try partial matches
            for std_name in CH_ORDER_STANDARD:
                if std_name in clean_name or clean_name in std_name:
                    ch_mapping[ch_name] = std_name
                    break
    
    if ch_mapping:
        raw.rename_channels(ch_mapping)
    
    # Pick channels that are in the standard order
    available_chs = [ch for ch in CH_ORDER_STANDARD if ch in raw.ch_names]
    if len(available_chs) > 10:  # At least some channels available
        raw.pick_channels(available_chs, ordered=True)
    
    # Apply filtering
    raw.filter(l_freq=0.1, h_freq=75.0, verbose=False)
    raw.notch_filter(60, verbose=False)
    
    # Resample if needed
    if int(raw.info['sfreq']) != 256:
        raw.resample(256, npad="auto", n_jobs=1, verbose=False)
    
    return raw


def convert_set_to_pickle(set_file_path, output_dir, window_duration=5, label=None):
    """
    Convert a single .set file to multiple pickle files with fixed window size
    """
    try:
        # Read the .set file
        raw = read_set_file(set_file_path)
        
        # Preprocess the data
        raw = preprocess_raw(raw)
        
        # Extract data
        data = raw.get_data()  # Shape: (n_channels, n_times)
        sampling_rate = raw.info['sfreq']
        
        # Calculate window size in samples
        window_size_samples = int(window_duration * sampling_rate)
        
        # Split the data into windows and save as pickle files
        n_windows = data.shape[1] // window_size_samples
        
        file_basename = os.path.splitext(os.path.basename(set_file_path))[0]
        
        for i in range(n_windows):
            start_sample = i * window_size_samples
            end_sample = (i + 1) * window_size_samples
            
            # Extract window data
            window_data = data[:, start_sample:end_sample]
            
            # Create data dictionary
            data_dict = {"X": window_data}
            if label is not None:
                data_dict["y"] = label
            
            # Save as pickle file
            pickle_filename = f"{file_basename}_{i}.pkl"
            pickle_filepath = os.path.join(output_dir, pickle_filename)
            
            with open(pickle_filepath, 'wb') as f:
                pickle.dump(data_dict, f)
                
    except Exception as e:
        print(f"Error converting {set_file_path}: {str(e)}", file=sys.stderr)
        raise


def create_hdf5_from_pickles(source_dir, target_file, finetune=True):
    """
    Create HDF5 file from directory of pickle files
    """
    if h5py is None:
        raise ImportError("h5py module is required for creating HDF5 files. Please install it using 'pip install h5py'")
    
    files = [f for f in os.listdir(source_dir) if f.endswith('.pkl')]
    group_size = 1000  # Number of pickle files per HDF5 group
    
    with h5py.File(target_file, 'w') as h5f:
        for i in tqdm(range(0, len(files), group_size), desc=f"Creating {target_file}"):
            group_files = files[i:i+group_size]
            data_group = []
            
            for file in group_files:
                with open(os.path.join(source_dir, file), 'rb') as f:
                    try:
                        sample = pickle.load(f)
                        data_group.append(sample)
                    except (pickle.UnpicklingError, EOFError) as e:
                        print(f"Warning: Skipping corrupt pickle file {file}: {e}")
                        continue
            
            if not data_group:
                continue
            
            grp = h5f.create_group(f"data_group_{i // group_size}")
            
            try:
                X_data = np.array([s['X'] for s in data_group])
                grp.create_dataset("X", data=X_data)
            except KeyError:
                print(f"Error: 'X' key missing in data group {(i // group_size)}. Skipping group.")
                del h5f[f"data_group_{i // group_size}"]
                continue
            except Exception as e:
                print(f"Error packing X data for group {(i // group_size)}: {e}")
                del h5f[f"data_group_{i // group_size}"]
                continue

            if finetune:
                try:
                    y_data = np.array([s['y'] for s in data_group])
                    grp.create_dataset("y", data=y_data)
                except KeyError:
                    print(f"Warning: 'y' key missing in finetune mode for group {(i // group_size)}, skipping 'y' dataset.")
                except Exception as e:
                    print(f"Error packing y data for group {(i // group_size)}: {e}")


def find_all_set_files(root_dir):
    """
    递归搜索所有.sub目录中的.set文件
    """
    set_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.set'):
                set_files.append(os.path.join(root, file))
    return set_files


def main():
    parser = argparse.ArgumentParser(
        description="将.set格式的EEG文件转换为BioFoundation项目所需的HDF5格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用说明:
---------
此脚本将EEGLAB的.set文件转换为HDF5格式，用于BioFoundation项目的训练。

输入要求:
- --set_dir: 包含subject目录的根目录路径（如eeg-data/derivatives）
- 脚本会自动递归搜索所有sub-*目录中的.set文件

输出要求:
- --output_dir: 输出目录路径（如model-data，必须包含train、val、test子目录）
- 脚本会在这三个子目录中分别生成对应的HDF5文件

处理流程:
1. 递归搜索所有.set文件
2. 读取并预处理EEG数据
3. 将数据分割为训练(70%)、验证(15%)、测试(15%)集
4. 将每个集合转换为HDF5格式
5. 清理临时pickle文件
        """
    )
    
    parser.add_argument("--set_dir", required=True, 
                       help="包含subject目录的输入根目录路径（如eeg-data/derivatives）")
    parser.add_argument("--output_dir", required=True, 
                       help="输出目录路径（如model-data，必须包含train、val、test子目录）")
    parser.add_argument("--window_duration", type=int, default=5, 
                       help="窗口持续时间（秒），默认为5秒")
    parser.add_argument("--label", type=int, 
                       help="为所有样本分配的标签（可选）")
    parser.add_argument("--task_type", choices=["pretrain", "finetune"], default="finetune", 
                       help="任务类型 - 决定是否包含标签（默认：finetune）")
    
    # 如果没有提供任何参数，显示帮助信息
    if len(sys.argv) == 1:
        parser.print_help()
        print_usage_examples()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # 验证目录结构
    if not validate_directories(args):
        print_usage_examples()
        sys.exit(1)
    
    # 检查h5py是否可用
    if h5py is None and args.task_type == "finetune":
        print("错误: finetune模式需要h5py模块，但未安装", file=sys.stderr)
        print("请运行: pip install h5py", file=sys.stderr)
        sys.exit(1)
    
    # 创建临时目录用于pickle文件
    temp_pickle_dir = os.path.join(args.output_dir, "temp_pickle")
    os.makedirs(temp_pickle_dir, exist_ok=True)
    
    # 递归搜索所有.set文件
    print(f"正在搜索 {args.set_dir} 目录中的所有.set文件...")
    set_files = find_all_set_files(args.set_dir)
    print(f"找到 {len(set_files)} 个.set文件待处理")
    
    if not set_files:
        print(f"在目录 {args.set_dir} 及其子目录中未找到.set文件")
        sys.exit(1)
    
    # 处理每个.set文件
    successful_conversions = 0
    failed_conversions = 0
    
    print(f"开始处理 {len(set_files)} 个.set文件...")
    
    for i, set_file in enumerate(tqdm(set_files, desc="转换.set文件为.pkl格式")):
        try:
            convert_set_to_pickle(set_file, temp_pickle_dir, 
                                window_duration=args.window_duration, 
                                label=args.label)
            successful_conversions += 1
            if (i + 1) % 10 == 0:  # 每10个文件报告一次进度
                print(f"已处理 {i + 1}/{len(set_files)} 个文件 ({successful_conversions} 成功, {failed_conversions} 失败)")
        except Exception as e:
            print(f"处理文件失败 {set_file}: {str(e)}", file=sys.stderr)
            failed_conversions += 1
    
    print(f"\n转换完成: {successful_conversions} 个成功, {failed_conversions} 个失败")
    
    if successful_conversions == 0:
        print("没有文件被成功转换，程序退出", file=sys.stderr)
        sys.exit(1)
    
    # 从pickle文件创建HDF5文件
    splits = ['train', 'val', 'test']
    all_pickle_files = [f for f in os.listdir(temp_pickle_dir) if f.endswith('.pkl')]
    n_files = len(all_pickle_files)
    
    if n_files == 0:
        print("No pickle files were created, exiting program", file=sys.stderr)
        sys.exit(1)
    
    # 数据分割: 70% 训练, 15% 验证, 15% 测试
    train_end = int(0.7 * n_files)
    val_end = int(0.85 * n_files)
    
    split_map = {
        'train': all_pickle_files[:train_end],
        'val': all_pickle_files[train_end:val_end],
        'test': all_pickle_files[val_end:]
    }
    
    # 处理每个分割
    for split in splits:
        files_to_move = split_map[split]
        if not files_to_move:
            print(f"Warning: No files in {split} split", file=sys.stderr)
            continue
            
        split_dir = os.path.join(args.output_dir, split)
        
        # 移动文件到对应的分割目录
        for file in files_to_move:
            src = os.path.join(temp_pickle_dir, file)
            dst = os.path.join(split_dir, file)
            
            # 处理文件名冲突
            counter = 1
            original_dst = dst
            while os.path.exists(dst):
                name, ext = os.path.splitext(original_dst)
                dst = f"{name}_{counter}{ext}"
                counter += 1
            
            try:
                os.rename(src, dst)
            except Exception as e:
                print(f"Error moving file {src} -> {dst}: {e}", file=sys.stderr)
        
        # 为这个分割创建HDF5文件
        hdf5_path = os.path.join(args.output_dir, split, f"{split}.h5")
        try:
            create_hdf5_from_pickles(split_dir, hdf5_path, finetune=(args.task_type == "finetune"))
            print(f"Created HDF5 file for {split} split: {hdf5_path}")
            
            # 成功创建HDF5后，删除原始的pkl文件以节省空间
            pkl_files_to_delete = [f for f in os.listdir(split_dir) if f.endswith('.pkl')]
            for pkl_file in pkl_files_to_delete:
                pkl_path = os.path.join(split_dir, pkl_file)
                try:
                    os.remove(pkl_path)
                except Exception as e:
                    print(f"Warning: Could not delete pickle file {pkl_path}: {e}")
            
            print(f"Cleaned up {len(pkl_files_to_delete)} pickle files from {split} directory")
            
        except Exception as e:
            print(f"Error creating HDF5 file for {split} split: {e}", file=sys.stderr)
    
    # 清理临时目录
    if os.path.exists(temp_pickle_dir):
        shutil.rmtree(temp_pickle_dir)
        print("Temporary pickle files cleaned up")
    
    print(f"\nConversion completed! HDF5 files have been saved to the train/val/test subdirectories in {args.output_dir}")
    print("Output files:")
    for split in splits:
        hdf5_file = os.path.join(args.output_dir, split, f"{split}.h5")
        if os.path.exists(hdf5_file):
            print(f"  {split}: {hdf5_file}")


if __name__ == "__main__":
    main()