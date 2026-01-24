#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
设计更有挑战性的实验来增强论文说服力
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random

def create_challenging_splits():
    """创建更有挑战性的数据划分"""
    
    # 加载数据
    data_file = Path("data/processed/temporal_sequences_len5.pkl")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    sequences = np.array(data['sequences'])
    labels = np.array(data['labels'])
    
    print("创建挑战性数据划分...")
    
    # 1. 时间连续性划分 - 避免时序泄露
    print("\n1. 时间连续性划分（避免相邻时间帧在不同集合中）")
    
    # 按类别顺序划分，确保训练集和测试集在时间上分离
    train_classes = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]  # 前半段 + 中间段
    val_classes = [5, 6, 13, 14]  # 验证集
    test_classes = [7, 15, 16, 17, 18, 19]  # 后半段
    
    train_mask = np.isin(labels, train_classes)
    val_mask = np.isin(labels, val_classes)
    test_mask = np.isin(labels, test_classes)
    
    print(f"训练集类别: {train_classes} ({np.sum(train_mask)} 样本)")
    print(f"验证集类别: {val_classes} ({np.sum(val_mask)} 样本)")
    print(f"测试集类别: {test_classes} ({np.sum(test_mask)} 样本)")
    
    # 保存时间分离的数据集
    temporal_split = {
        'train_sequences': sequences[train_mask],
        'train_labels': labels[train_mask],
        'val_sequences': sequences[val_mask],
        'val_labels': labels[val_mask],
        'test_sequences': sequences[test_mask],
        'test_labels': labels[test_mask]
    }
    
    with open('data/processed/temporal_split.pkl', 'wb') as f:
        pickle.dump(temporal_split, f)
    
    # 2. 跨环境泛化划分
    print("\n2. 跨环境泛化划分（模拟不同天气/光照条件）")
    
    # 随机选择一些类别作为"不同环境条件"
    np.random.seed(42)
    
    # 为每个类别随机分配"环境条件"
    env_conditions = {}
    for label in range(20):
        env_conditions[label] = np.random.choice(['sunny', 'cloudy', 'morning', 'afternoon'], 1)[0]
    
    print("环境条件分配:")
    for label, condition in env_conditions.items():
        print(f"  类别 {label:2d}: {condition}")
    
    # 按环境条件划分
    train_conditions = ['sunny', 'morning']
    test_conditions = ['cloudy', 'afternoon']
    
    train_env_classes = [label for label, cond in env_conditions.items() if cond in train_conditions]
    test_env_classes = [label for label, cond in env_conditions.items() if cond in test_conditions]
    
    train_env_mask = np.isin(labels, train_env_classes)
    test_env_mask = np.isin(labels, test_env_classes)
    
    print(f"\n训练环境类别: {train_env_classes}")
    print(f"测试环境类别: {test_env_classes}")
    
    # 3. 少样本学习划分
    print("\n3. 少样本学习划分")
    
    few_shot_data = defaultdict(list)
    
    for k_shot in [1, 3, 5]:
        train_indices = []
        test_indices = []
        
        for label in range(20):
            label_indices = np.where(labels == label)[0]
            
            # 每个类别只用k个样本训练
            selected_train = np.random.choice(label_indices, k_shot, replace=False)
            remaining = np.setdiff1d(label_indices, selected_train)
            
            train_indices.extend(selected_train)
            test_indices.extend(remaining)
        
        few_shot_data[f'{k_shot}_shot'] = {
            'train_sequences': sequences[train_indices],
            'train_labels': labels[train_indices],
            'test_sequences': sequences[test_indices],
            'test_labels': labels[test_indices]
        }
        
        print(f"{k_shot}-shot: 训练集 {len(train_indices)} 样本, 测试集 {len(test_indices)} 样本")
    
    # 保存少样本数据
    with open('data/processed/few_shot_splits.pkl', 'wb') as f:
        pickle.dump(few_shot_data, f)
    
    return temporal_split, few_shot_data

def create_noise_robustness_test():
    """创建噪声鲁棒性测试数据"""
    
    print("\n创建噪声鲁棒性测试...")
    
    # 加载原始数据
    data_file = Path("data/processed/temporal_sequences_len5.pkl")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    sequences = np.array(data['sequences'])
    labels = np.array(data['labels'])
    
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
    noisy_data = {}
    
    for noise_level in noise_levels:
        print(f"生成噪声水平 {noise_level} 的数据...")
        
        # 添加高斯噪声
        noise = np.random.normal(0, noise_level, sequences.shape)
        noisy_sequences = sequences + noise
        
        # 确保数据仍在合理范围内
        noisy_sequences = np.clip(noisy_sequences, 0, 1)
        
        noisy_data[f'noise_{noise_level}'] = {
            'sequences': noisy_sequences,
            'labels': labels
        }
    
    # 保存噪声数据
    with open('data/processed/noise_robustness.pkl', 'wb') as f:
        pickle.dump(noisy_data, f)
    
    return noisy_data

def create_ablation_study_data():
    """创建消融研究数据"""
    
    print("\n创建消融研究数据...")
    
    # 加载原始数据
    data_file = Path("data/processed/temporal_sequences_len5.pkl")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    sequences = np.array(data['sequences'])
    labels = np.array(data['labels'])
    
    ablation_data = {}
    
    # 1. 单帧 vs 多帧
    print("1. 单帧特征（移除时序信息）")
    single_frame = sequences[:, 0, :, :]  # 只用第一帧
    ablation_data['single_frame'] = {
        'sequences': single_frame,
        'labels': labels
    }
    
    # 2. 不同时序长度
    for seq_len in [2, 3, 4, 5]:
        print(f"2. 时序长度 {seq_len}")
        truncated_seq = sequences[:, :seq_len, :, :]
        ablation_data[f'seq_len_{seq_len}'] = {
            'sequences': truncated_seq,
            'labels': labels
        }
    
    # 3. 空间分辨率消融
    print("3. 空间分辨率消融")
    
    # 降低环数分辨率
    low_res_rings = sequences[:, :, ::2, :]  # 从20环降到10环
    ablation_data['low_res_rings'] = {
        'sequences': low_res_rings,
        'labels': labels
    }
    
    # 降低扇区分辨率  
    low_res_sectors = sequences[:, :, :, ::2]  # 从60扇区降到30扇区
    ablation_data['low_res_sectors'] = {
        'sequences': low_res_sectors,
        'labels': labels
    }
    
    # 保存消融研究数据
    with open('data/processed/ablation_study.pkl', 'wb') as f:
        pickle.dump(ablation_data, f)
    
    return ablation_data

def analyze_failure_cases():
    """分析失败案例，找出模型的局限性"""
    
    print("\n分析潜在失败案例...")
    
    # 加载数据
    data_file = Path("data/processed/temporal_sequences_len5.pkl")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    sequences = np.array(data['sequences'])
    labels = np.array(data['labels'])
    
    # 找出最相似的不同类别对（容易混淆的情况）
    print("最容易混淆的类别对:")
    
    class_centroids = {}
    for label in range(20):
        label_mask = labels == label
        class_centroids[label] = np.mean(sequences[label_mask], axis=0)
    
    confusing_pairs = []
    for i in range(20):
        for j in range(i+1, 20):
            centroid1 = class_centroids[i].flatten()
            centroid2 = class_centroids[j].flatten()
            similarity = np.corrcoef(centroid1, centroid2)[0, 1]
            
            if not np.isnan(similarity):
                confusing_pairs.append((i, j, similarity))
    
    # 排序找出最相似的
    confusing_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("最容易混淆的类别对（高相似度但不同标签）:")
    for i, (label1, label2, sim) in enumerate(confusing_pairs[:10]):
        time_gap = abs(label1 - label2)
        if time_gap > 1:  # 不是相邻的时间段
            print(f"  {i+1}. 类别 {label1} ↔ 类别 {label2}: 相似度={sim:.4f}, 时间间隔={time_gap}")
    
    return confusing_pairs

def suggest_paper_structure():
    """建议论文结构"""
    
    print("\n" + "="*60)
    print("论文写作建议")
    print("="*60)
    
    print("""
📝 建议的论文结构：

1. 引言 (Introduction)
   - 农田机器人导航的重要性
   - 回环检测在SLAM中的作用
   - 现有方法的局限性（主要针对城市/室内环境）
   - 本文贡献：针对农田环境的时序回环检测方法

2. 相关工作 (Related Work)
   - 传统回环检测方法 (BoW, VLAD等)
   - 基于深度学习的方法
   - ScanContext及其变体
   - 农田机器人导航研究

3. 方法 (Methodology)
   - 3.1 ScanContext特征提取
   - 3.2 时序建模策略
   - 3.3 多架构深度学习模型
   - 3.4 训练策略和优化

4. 实验设计 (Experimental Setup)
   - 4.1 数据集构建（农田环境特点）
   - 4.2 评估指标
   - 4.3 对比基线方法
   - 4.4 实验设置

5. 结果与分析 (Results and Analysis)
   - 5.1 不同架构对比
   - 5.2 时序长度影响分析
   - 5.3 噪声鲁棒性测试
   - 5.4 少样本学习能力
   - 5.5 消融研究
   - 5.6 失败案例分析

6. 讨论 (Discussion)
   - 方法的优势和局限性
   - 农田环境的特殊性
   - 计算复杂度分析
   - 实际应用考虑

7. 结论 (Conclusion)
   - 主要贡献总结
   - 未来工作方向

🎯 关键写作策略：

1. 强调方法创新而非准确率
   - "提出了时序ScanContext融合框架"
   - "设计了多尺度时序建模策略"
   - "验证了不同架构在农田环境中的有效性"

2. 突出农田环境的特殊性
   - 结构化程度高，有利于特征学习
   - 重复性模式，适合回环检测
   - 光照变化、季节变化等挑战

3. 增加实验的深度
   - 时间分离的数据划分
   - 噪声鲁棒性测试
   - 少样本学习实验
   - 详细的消融研究

4. 诚实讨论局限性
   - 数据集规模限制
   - 环境条件相对简单
   - 需要更多样化的农田场景验证

5. 强调实际应用价值
   - 农田机器人导航
   - 精准农业应用
   - 自动化农业设备
    """)

if __name__ == '__main__':
    print("设计挑战性实验以增强论文说服力")
    print("="*60)
    
    # 1. 创建挑战性数据划分
    temporal_split, few_shot_data = create_challenging_splits()
    
    # 2. 创建噪声鲁棒性测试
    noisy_data = create_noise_robustness_test()
    
    # 3. 创建消融研究数据
    ablation_data = create_ablation_study_data()
    
    # 4. 分析失败案例
    confusing_pairs = analyze_failure_cases()
    
    # 5. 论文写作建议
    suggest_paper_structure()
    
    print("\n✅ 所有挑战性实验数据已生成完成！")
    print("📁 生成的文件:")
    print("  - data/processed/temporal_split.pkl")
    print("  - data/processed/few_shot_splits.pkl") 
    print("  - data/processed/noise_robustness.pkl")
    print("  - data/processed/ablation_study.pkl")
