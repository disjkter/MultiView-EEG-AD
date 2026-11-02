import numpy as np
from pathlib import Path

def probe_out_file(file_path):
    print("\n" + "="*50)
    print(f"🔍 正在探测文件: {file_path.name}")
    print("="*50)
    try:
        # 1. 读取原始文本前 5 行，检查是否有隐藏的表头或通道名
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            head = [next(f) for _ in range(5)]
        print("📄 文件头 5 行预览 (检查分隔符与表头):")
        for i, line in enumerate(head):
            # 限制打印长度，防止控制台刷屏
            print(f"   [{i}] {line.strip()[:80]} ...") 

        # 2. 尝试用 Numpy 直接将文本反序列化为底层矩阵
        print("\n⏳ 尝试将文件反序列化为数值张量...")
        try:
            # 默认尝试空格/Tab分割
            data = np.loadtxt(file_path)
        except Exception:
            try:
                # 回退尝试逗号分割 (CSV格式)
                data = np.loadtxt(file_path, delimiter=',')
            except Exception as inner_e:
                print(f"   ❌ 数值加载失败，可能包含无法解析的文本表头: {inner_e}")
                return

        print(f"✅ 数据矩阵加载成功！")
        print(f"   ├─ 原始张量形状 (Shape): {data.shape}")
        if len(data.shape) == 2:
            rows, cols = data.shape
            if rows > cols:
                print(f"   ├─ 结构剖析: 【连续时间序列】 时间步(Time)={rows}, 通道数(Channels)={cols}")
            else:
                print(f"   ├─ 结构剖析: 【连续时间序列】 通道数(Channels)={rows}, 时间步(Time)={cols}")
        else:
            print(f"   ├─ 结构剖析: 未知的高维张量")

    except Exception as e:
        print(f"❌ 探测器发生严重异常: {e}")

if __name__ == "__main__":
    # 直接指向你提供的两个极具代表性的测试文件
    ad_file = Path(r"E:\ProgramFiles\PythonProject\BioFoundation-main\ADSZ\dataset\alzheimer\AD\ec0101.out")
    hc_file = Path(r"E:\ProgramFiles\PythonProject\BioFoundation-main\ADSZ\dataset\alzheimer\Healthy\eeg20c1.out")

    if ad_file.exists():
        probe_out_file(ad_file)
    else:
        print(f"⚠️ 找不到测试文件: {ad_file}")

    if hc_file.exists():
        probe_out_file(hc_file)
    else:
        print(f"⚠️ 找不到测试文件: {hc_file}")