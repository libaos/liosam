#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查数据内容脚本
"""

import pickle
import numpy as np
from pathlib import Path
from collections import Counter

def check_data():
    """检查数据内容"""
    
    # 检查预处理的数据
    data_file = Path("data/processed/temporal_sequences_len5.pkl")
    
    if data_file.exists():
        print("加载数据文件...")
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        sequences = data['sequences']
        labels = data['labels']
        
        print(f"数据基本信息:")
        print(f"  序列数量: {len(sequences)}")
        print(f"  标签数量: {len(labels)}")
        
        if len(sequences) > 0:
            print(f"  序列形状: {sequences[0].shape}")
            print(f"  序列数据类型: {type(sequences[0])}")
            print(f"  序列数值范围: [{np.min(sequences[0]):.4f}, {np.max(sequences[0]):.4f}]")
        
        # 检查标签分布
        label_counts = Counter(labels)
        print(f"\n标签分布:")
        for label, count in sorted(label_counts.items()):
            print(f"  类别 {label}: {count} 个样本")
        
        print(f"\n唯一标签数量: {len(label_counts)}")
        print(f"标签范围: [{min(labels)}, {max(labels)}]")
        
        # 检查是否有数据泄露问题
        print(f"\n数据质量检查:")
        
        # 检查序列中是否有重复的ScanContext
        unique_sequences = set()
        duplicate_count = 0
        
        for i, seq in enumerate(sequences[:100]):  # 只检查前100个
            seq_hash = hash(seq.tobytes())
            if seq_hash in unique_sequences:
                duplicate_count += 1
            else:
                unique_sequences.add(seq_hash)
        
        print(f"  前100个序列中重复序列数量: {duplicate_count}")
        
        # 检查标签是否过于简单
        if len(label_counts) < 5:
            print(f"  ⚠️  警告: 只有 {len(label_counts)} 个不同的类别，可能存在数据问题")
        
        # 检查类别分布是否均匀
        counts = list(label_counts.values())
        if max(counts) / min(counts) > 10:
            print(f"  ⚠️  警告: 类别分布不均匀，最大类别样本数是最小类别的 {max(counts) / min(counts):.1f} 倍")
        
        # 检查序列内容
        print(f"\n序列内容分析:")
        sample_seq = sequences[0]
        print(f"  第一个序列的统计信息:")
        print(f"    均值: {np.mean(sample_seq):.4f}")
        print(f"    标准差: {np.std(sample_seq):.4f}")
        print(f"    零值比例: {np.mean(sample_seq == 0):.4f}")
        
        # 检查是否所有序列都很相似
        if len(sequences) >= 10:
            similarities = []
            for i in range(1, min(10, len(sequences))):
                similarity = np.corrcoef(sequences[0].flatten(), sequences[i].flatten())[0, 1]
                similarities.append(similarity)
            
            avg_similarity = np.mean(similarities)
            print(f"    与其他序列的平均相关性: {avg_similarity:.4f}")
            
            if avg_similarity > 0.9:
                print(f"    ⚠️  警告: 序列之间相关性过高 ({avg_similarity:.4f})，可能存在数据泄露")
    
    else:
        print("未找到数据文件")

def check_model_input_output():
    """检查模型的输入输出定义"""
    print("\n" + "="*60)
    print("模型输入输出定义检查")
    print("="*60)
    
    print("输入定义:")
    print("  - 输入形状: (batch_size, sequence_length, num_rings, num_sectors)")
    print("  - 具体形状: (batch_size, 5, 20, 60)")
    print("  - 数据类型: float32")
    print("  - 数值范围: ScanContext特征值 (通常0-几十)")
    
    print("\n输出定义:")
    print("  - 输出形状: (batch_size, num_classes)")
    print("  - 具体形状: (batch_size, 20)")
    print("  - 数据类型: float32 (logits)")
    print("  - 含义: 20个路径段的分类概率")
    
    print("\n标签定义:")
    print("  - 标签含义: 路径段ID (0-19)")
    print("  - 生成方式: 将完整轨迹按顺序分成20段")
    print("  - 问题分析: 这种标签生成方式可能过于简单!")
    
    print("\n潜在问题分析:")
    print("  1. 时序相关性: 相邻的时序序列可能有很高的重叠")
    print("  2. 标签简单性: 按顺序分段可能导致标签过于容易预测")
    print("  3. 数据泄露: 训练集和测试集可能包含相似的序列")
    print("  4. 任务难度: 可能不是真正的回环检测，而是轨迹位置预测")

if __name__ == '__main__':
    check_data()
    check_model_input_output()
