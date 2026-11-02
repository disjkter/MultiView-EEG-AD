"""
=========================================================================================
BioFoundation Project: Spatial Distillation (SD) Module for Cross-Headset EEG Transfer
=========================================================================================

Reference: 
    Dingkun Liu et al., "SDDA: Spatial Distillation based Distribution Alignment 
    for Cross-Headset EEG Classification," IEEE Transactions on Biomedical Engineering, 2025[cite: 1, 5, 6].

Description:
    This module implements the Spatial Distillation (SD) mechanism to address the 
    heterogeneous cross-headset transfer problem in EEG-based BCIs[cite: 9, 10]. 
    Traditionally, transferring models from high-density medical headsets (e.g., 19 channels) 
    to portable consumer headsets (e.g., 4 channels) requires discarding non-overlapping 
    channels, causing severe loss of spatial semantic information[cite: 38]. 

Mechanism (Teacher-Student Architecture):
    - Teacher Model: A pre-trained, high-capacity model utilizing the full set of 
      source domain electrodes (19 channels). Its weights are frozen[cite: 154].
    - Student Model: A lightweight model utilizing only the common subset of electrodes 
      (4 channels) shared with the target domain[cite: 154].
    - Objective: The student model is trained to mimic the teacher's soft probability 
      distributions (Logits) via Kullback-Leibler (KL) divergence[cite: 156, 158]. 
      This allows the student to implicitly inherit global spatial semantic representations 
      from the teacher, despite having restricted physical inputs[cite: 47, 171].

Author: [你的名字/BioFoundation团队]
Date: 2026-03
=========================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 导入你的双流模型
from models.Spatial_MSCNN import SpatialMSCNN_DualStream

def distillation_loss(student_logits, teacher_logits, temperature):
    """
    实现论文中的公式 (1): 空间蒸馏的 KL 散度损失 [cite: 158]
    """
    # 将 logits 转换为平滑的对数概率和概率分布
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    
    # 计算 KL 散度，并乘以 T^2 (论文标准做法，保持梯度量级) [cite: 158]
    loss_kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return loss_kl

def train_spatial_distillation(source_x_19ch, source_y, target_channels_idx, device):
    """
    执行空间蒸馏 (Spatial Distillation) [cite: 47]
    :param source_x_19ch: 源域 19 通道数据，形状 [Batch, 19, Length]
    :param source_y: 真实标签
    :param target_channels_idx: 4 个目标通道在 19 通道中的索引列表 (如 [0, 3, 10, 15])
    """
    # 1. 初始化 19 通道的 Teacher (冻结) [cite: 154]
    teacher = SpatialMSCNN_DualStream(num_classes=2, num_channels=19).to(device)
    teacher.load_state_dict(torch.load("best_model_19ch.pth", map_location=device))
    teacher.eval() # 必须设为评估模式，不更新梯度
    
    # 2. 初始化 4 通道的 Student (待训练) [cite: 154]
    student = SpatialMSCNN_DualStream(num_classes=2, num_channels=len(target_channels_idx)).to(device)
    student.train()
    
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    criterion_ce = nn.CrossEntropyLoss()
    
    # 论文中建议的超参数：温度系数 T 和 蒸馏权重 alpha
    TEMPERATURE = 2.0  # 论文 4.B 节设定 \tau = 2 [cite: 309, 310]
    ALPHA = 1.0        # 论文 4.B 节设定 \alpha = 1 [cite: 310, 311]
    
    epochs = 50
    print("🔥 启动基于 SDDA 理论的空间知识蒸馏 (19CH -> 4CH)...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 3. 数据准备：Student 只看对应的 4 个通道 [cite: 170]
        x_19ch = source_x_19ch.to(device)
        x_4ch = x_19ch[:, target_channels_idx, :] # 提取公共通道子集
        y = source_y.to(device)
        
        # 4. 前向传播
        with torch.no_grad():
            logits_tch, _ = teacher(x_19ch) # 导师看着全貌给出完美预测
            
        logits_stu, _ = student(x_4ch)      # 学生只看 4 个通道给出预测
        
        # 5. 计算混合 Loss (公式 11 简化版，仅保留 CE + SD) [cite: 239, 240]
        # 真实标签的交叉熵损失
        loss_ce = criterion_ce(logits_stu, y)
        # 模拟导师输出的 KL 散度损失
        loss_sd = distillation_loss(logits_stu, logits_tch, TEMPERATURE)
        
        # 总损失 = 任务损失 + alpha * 蒸馏损失
        loss_total = loss_ce + ALPHA * loss_sd 
        
        # 6. 反向传播更新学生网络
        loss_total.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | CE Loss: {loss_ce.item():.4f} | Distill Loss: {loss_sd.item():.4f}")
            
    print("✅ 空间蒸馏完成！Student 模型已经继承了 19 通道的空间视野。")
    return student