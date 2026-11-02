import os
import sys
import torch
from safetensors.torch import load_file

# --- 1. 路径自动对齐 ---
# 获取当前脚本的父目录（BioFoundation-main）
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from models.luna_classifier import LUNAClassifierWithAttention
    print("✅ 成功导入模型类。")
except ModuleNotFoundError:
    print("❌ 依然找不到 models 文件夹，请确认 check.py 是否在 BioFoundation-main/scripts 目录下。")
    sys.exit()

# --- 2. 实例化模型并查看 Key ---
model = LUNAClassifierWithAttention(num_classes=3, num_channels=19, luna_config='LUNA_small')
model_keys = list(model.state_dict().keys())
print("\n--- 🧠 模型内部 (model.state_dict) 的前 10 个 Key ---")
for k in model_keys[:10]:
    print(f"  {k}")

# --- 3. 读取权重文件并查看 Key ---
weight_path = r"E:\ProgramFiles\PythonProject\BioFoundation-main\LUNA_base.safetensors"
if os.path.exists(weight_path):
    print(f"\n--- 💾 权重文件 (safetensors) 的前 10 个 Key ---")
    state_dict = load_file(weight_path)
    weight_keys = list(state_dict.keys())
    for k in weight_keys[:10]:
        print(f"  {k}")
else:
    print(f"\n❌ 未找到权重文件: {weight_path}")

# --- 4. 专家对比分析 ---
print("\n" + "="*40)
print("🔍 诊断结论预览：")
common = set(model_keys).intersection(set(weight_keys))
print(f"  直接匹配成功的 Key 数量: {len(common)}")
if len(common) == 0:
    print("  👉 结论：名称不匹配。请将上面两组打印的名字贴给 Gemini。")
print("="*40)