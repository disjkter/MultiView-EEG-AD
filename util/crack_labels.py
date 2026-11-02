import scipy.io as sio
from pathlib import Path
import numpy as np

def crack_fieldtrip_metadata(file_path):
    try:
        mat_data = sio.loadmat(file_path)
        data_struct = mat_data['data'][0, 0]
        
        print(f"\n📂 正在深度解剖: {file_path.name}")
        
        # 1. 打印真实通道数和名称
        raw_labels = data_struct['label']
        channel_names = [str(lbl[0][0]).strip() for lbl in raw_labels]
        print(f"   ├─ 🧠 真实通道数: {len(channel_names)} 个")
        print(f"   ├─ 🏷️ 通道列表: {channel_names}")
        
        # 2. 挖掘 cfg 历史路径寻找标签线索
        print(f"   ├─ 🕵️‍♂️ 寻找隐藏标签线索...")
        found_clue = False
        if 'cfg' in data_struct.dtype.names:
            cfg = data_struct['cfg'][0, 0]
            for field in cfg.dtype.names:
                try:
                    val = cfg[field]
                    # 将嵌套的值转为字符串，寻找 AD/HC 等关键字
                    if isinstance(val, np.ndarray) and val.size > 0:
                        str_val = str(val[0]).upper()
                        if any(keyword in str_val for keyword in ['AD', 'HC', 'CONTROL', 'PATIENT', 'NORMAL']):
                            print(f"   │  └─ [命中线索] 字段 '{field}': {str_val}")
                            found_clue = True
                except:
                    pass
                    
        if not found_clue:
            print("   │  └─ 未在 cfg 表层发现直观的 AD/HC 字符串。")
            
    except Exception as e:
        print(f"❌ 解析 {file_path.name} 失败: {e}")

if __name__ == "__main__":
    target_dir = Path(r"E:\ProgramFiles\PythonProject\BioFoundation-main\AD_Data")
    # 获取所有的 mat 文件并排序
    mat_files = sorted(list(target_dir.rglob("*.mat")))
    
    if len(mat_files) > 0:
        print(f"✅ 共发现 {len(mat_files)} 个文件。抽取【第1个】和【最后1个】进行基因级检测...")
        # 测头尾两个文件，通常分别属于两个类别
        test_files = [mat_files[0], mat_files[-1]] 
        for f in test_files:
            crack_fieldtrip_metadata(f)
    else:
        print(f"⚠️ 找不到文件，请检查路径: {target_dir}")