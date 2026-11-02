#!/usr/bin/env python3
"""
测试convert_set_to_hdf5.py脚本的示例
展示如何正确设置目录结构和运行转换
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

def create_test_structure():
    """创建测试用的目录结构"""
    # 创建临时目录
    test_dir = tempfile.mkdtemp(prefix="biofoundation_test_")
    
    # 创建输入目录结构
    input_dir = os.path.join(test_dir, "input_set_files")
    output_dir = os.path.join(test_dir, "output_hdf5")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建必需的输出子目录
    for subdir in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    print(f"已创建测试目录结构:")
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  子目录: train/, val/, test/")
    
    return test_dir, input_dir, output_dir

def show_usage_example():
    """显示使用示例"""
    print("\n=== 使用示例 ===")
    print("# 1. 首先创建输出目录结构")
    print("mkdir -p ./my_data/{train,val,test}")
    print()
    print("# 2. 运行转换脚本")
    print("python make_datasets/convert_set_to_hdf5.py \\")
    print("    --set_dir ./path/to/set/files \\")
    print("    --output_dir ./my_data \\")
    print("    --window_duration 5 \\")
    print("    --task_type finetune")
    print()
    print("# 3. 查看生成的文件")
    print("ls -la ./my_data/train/train.h5")
    print("ls -la ./my_data/val/val.h5") 
    print("ls -la ./my_data/test/test.h5")

def main():
    print("BioFoundation .set文件转换测试")
    print("=" * 50)
    
    # 显示使用说明
    show_usage_example()
    
    # 询问用户是否要创建测试结构
    response = input("\n是否要创建测试目录结构？(y/n): ").strip().lower()
    
    if response == 'y':
        test_dir, input_dir, output_dir = create_test_structure()
        print(f"\n✓ 测试环境已准备就绪!")
        print(f"\n下一步:")
        print(f"1. 将你的.set文件放入: {input_dir}")
        print(f"2. 运行命令:")
        print(f"   python make_datasets/convert_set_to_hdf5.py \\")
        print(f"       --set_dir {input_dir} \\")
        print(f"       --output_dir {output_dir}")
        print(f"3. 完成后清理测试目录:")
        print(f"   rm -rf {test_dir}")
    else:
        print("\n好的，这里是基本使用方法:")
        show_usage_example()

if __name__ == "__main__":
    main()