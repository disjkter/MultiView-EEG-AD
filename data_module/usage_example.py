"""
EEG Data Module 使用示例
展示如何配置和使用修改后的数据模块
"""

import os
import sys
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_module.eeg_data_module import EEGDataModule

def setup_data_environment():
    """设置数据环境变量和目录结构"""
    
    # 使用项目相对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(project_root, "model-data")  # 相对于项目根目录
    
    # 设置环境变量（为了兼容性）
    os.environ['DATA_PATH'] = os.path.dirname(data_root)
    
    print(f"项目根目录: {project_root}")
    print(f"数据根目录设置为: {os.path.dirname(data_root)}")
    print(f"实际数据路径: {data_root}")
    print("期望的目录结构:")
    print(f"{os.path.dirname(data_root)}/")
    print(f"└── model_data/")  # 注意：这里使用 model 而不是 eeg-data
    print(f"    ├── train/")
    print(f"    │   └── train.h5")
    print(f"    ├── val/")
    print(f"    │   └── val.h5")
    print(f"    └── test/")
    print(f"        └── test.h5")
    
    return data_root

def create_data_module():
    """创建数据模块实例"""
    
    # 创建数据模块
    data_module = EEGDataModule(
        name="model",  # 数据集名称，对应 {name}_data 目录
        cfg={
            'batch_size': 32,
            'num_workers': 4
        },
        test=True  # 启用测试数据集
    )
    
    return data_module

def demonstrate_usage():
    """演示完整的使用流程"""
    
    print("=== EEG Data Module 使用演示 ===\n")
    
    # 1. 设置环境
    data_root = setup_data_environment()
    
    # 2. 创建数据模块
    print("\n2. 创建数据模块...")
    data_module = create_data_module()
    
    # 3. 准备数据（验证文件存在性）
    print("\n3. 验证数据文件...")
    try:
        data_module.prepare_data()
        print("✓ 数据文件验证通过")
    except Exception as e:
        print(f"✗ 数据验证失败: {e}")
        print(f"\n请确保以下路径存在并包含相应的HDF5文件:")
        print(f"训练数据: {os.path.join(data_root, 'train', 'train.h5')}")
        print(f"验证数据: {os.path.join(data_root, 'val', 'val.h5')}")
        print(f"测试数据: {os.path.join(data_root, 'test', 'test.h5')}")
        return
    
    # 4. 设置数据集
    print("\n4. 加载数据集...")
    data_module.setup()
    print("✓ 数据集加载完成")
    
    # 5. 获取数据加载器
    print("\n5. 创建数据加载器...")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    print(f"✓ 训练加载器批次大小: {train_loader.batch_size}")
    print(f"✓ 验证加载器批次大小: {val_loader.batch_size}")
    print(f"✓ 测试加载器批次大小: {test_loader.batch_size if test_loader else 'N/A'}")
    
    # 6. 显示数据信息
    print("\n6. 数据集信息:")
    print(f"   训练样本数: {len(data_module.train_dataset)}")
    print(f"   验证样本数: {len(data_module.val_dataset)}")
    if data_module.test_dataset:
        print(f"   测试样本数: {len(data_module.test_dataset)}")

if __name__ == "__main__":
    demonstrate_usage()