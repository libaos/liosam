#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析轨迹数据，检查是否真的有回环
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import glob
from collections import defaultdict

def load_gps_data():
    """加载GPS轨迹数据"""
    # 检查是否有GPS数据文件
    gps_files = glob.glob("data/raw/**/*gps*.txt", recursive=True) + \
                glob.glob("data/raw/**/*GPS*.txt", recursive=True) + \
                glob.glob("data/raw/**/*pose*.txt", recursive=True) + \
                glob.glob("data/raw/**/*trajectory*.txt", recursive=True)
    
    print(f"找到GPS相关文件: {gps_files}")
    
    if gps_files:
        # 尝试读取第一个文件
        try:
            gps_data = np.loadtxt(gps_files[0])
            print(f"GPS数据形状: {gps_data.shape}")
            print(f"GPS数据前5行:\n{gps_data[:5]}")
            return gps_data
        except Exception as e:
            print(f"读取GPS文件失败: {e}")
    
    return None

def analyze_spatial_distribution():
    """分析空间分布，检查是否有真正的回环"""
    
    # 加载序列数据
    data_file = Path("data/processed/temporal_sequences_len5.pkl")
    if not data_file.exists():
        print("未找到序列数据文件")
        return
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    sequences = data['sequences']
    labels = data['labels']
    file_paths = data.get('file_paths', [])
    
    print(f"分析 {len(sequences)} 个序列...")
    
    # 分析每个类别的序列特征
    class_features = defaultdict(list)
    
    for i, (seq, label) in enumerate(zip(sequences, labels)):
        # 计算序列的特征向量（简单的统计特征）
        feature_vector = [
            np.mean(seq),           # 均值
            np.std(seq),            # 标准差
            np.max(seq),            # 最大值
            np.min(seq),            # 最小值
            np.sum(seq > 0.5),      # 高值点数量
            np.sum(seq < 0.1),      # 低值点数量
        ]
        class_features[label].append(feature_vector)
    
    # 计算类内和类间相似性
    print("\n类别内部相似性分析:")
    class_similarities = {}
    
    for label, features in class_features.items():
        if len(features) > 1:
            features = np.array(features)
            # 计算类内平均相关性
            correlations = []
            for i in range(len(features)):
                for j in range(i+1, len(features)):
                    corr = np.corrcoef(features[i], features[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            if correlations:
                avg_corr = np.mean(correlations)
                class_similarities[label] = avg_corr
                print(f"  类别 {label:2d}: 平均相关性 = {avg_corr:.4f} ({len(features)} 个样本)")
    
    # 分析类间相似性
    print("\n类间相似性分析 (检查是否有真正的回环):")
    
    # 计算所有类别的平均特征
    class_centroids = {}
    for label, features in class_features.items():
        class_centroids[label] = np.mean(features, axis=0)
    
    # 找出最相似的类别对
    similar_pairs = []
    labels_list = list(class_centroids.keys())
    
    for i in range(len(labels_list)):
        for j in range(i+1, len(labels_list)):
            label1, label2 = labels_list[i], labels_list[j]
            centroid1, centroid2 = class_centroids[label1], class_centroids[label2]
            
            similarity = np.corrcoef(centroid1, centroid2)[0, 1]
            if not np.isnan(similarity):
                similar_pairs.append((label1, label2, similarity))
    
    # 排序并显示最相似的类别对
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("最相似的类别对 (可能的回环位置):")
    for i, (label1, label2, sim) in enumerate(similar_pairs[:10]):
        print(f"  {i+1:2d}. 类别 {label1:2d} ↔ 类别 {label2:2d}: 相似度 = {sim:.4f}")
        
        # 如果相似度很高，可能是真正的回环
        if sim > 0.8:
            print(f"      ⭐ 可能的回环: 类别{label1} 和 类别{label2}")
    
    # 检查时间上的分布
    print(f"\n时间分布分析:")
    print(f"类别按时间顺序: {sorted(set(labels))}")
    
    # 检查是否有跳跃式的相似性（真正的回环特征）
    high_similarity_pairs = [pair for pair in similar_pairs if pair[2] > 0.7]
    
    if high_similarity_pairs:
        print(f"\n发现 {len(high_similarity_pairs)} 对高相似性类别:")
        for label1, label2, sim in high_similarity_pairs:
            time_gap = abs(label1 - label2)
            print(f"  类别 {label1} ↔ 类别 {label2}: 相似度={sim:.4f}, 时间间隔={time_gap}")
            
            if time_gap > 5:  # 时间间隔较大但相似度高，可能是真回环
                print(f"    🔄 疑似真正回环: 时间间隔{time_gap}段但高度相似")
    
    return class_features, similar_pairs

def visualize_sequence_similarity():
    """可视化序列相似性矩阵"""
    
    data_file = Path("data/processed/temporal_sequences_len5.pkl")
    if not data_file.exists():
        return
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    sequences = data['sequences']
    labels = data['labels']
    
    # 随机选择一些序列计算相似性矩阵
    n_samples = min(50, len(sequences))
    indices = np.random.choice(len(sequences), n_samples, replace=False)
    
    selected_sequences = [sequences[i] for i in indices]
    selected_labels = [labels[i] for i in indices]
    
    # 计算相似性矩阵
    similarity_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            seq1 = selected_sequences[i].flatten()
            seq2 = selected_sequences[j].flatten()
            similarity = np.corrcoef(seq1, seq2)[0, 1]
            if not np.isnan(similarity):
                similarity_matrix[i, j] = similarity
    
    # 绘制相似性矩阵
    plt.figure(figsize=(12, 10))
    plt.imshow(similarity_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='相关系数')
    plt.title('序列相似性矩阵')
    plt.xlabel('序列索引')
    plt.ylabel('序列索引')
    
    # 添加标签信息
    for i in range(0, n_samples, 5):
        plt.axhline(y=i-0.5, color='red', alpha=0.3, linewidth=0.5)
        plt.axvline(x=i-0.5, color='red', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('sequence_similarity_matrix.png', dpi=150, bbox_inches='tight')
    print("相似性矩阵已保存为 sequence_similarity_matrix.png")
    
    # 分析对角线外的高相似性
    high_sim_pairs = []
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            if similarity_matrix[i, j] > 0.8:
                label_diff = abs(selected_labels[i] - selected_labels[j])
                high_sim_pairs.append((i, j, similarity_matrix[i, j], 
                                     selected_labels[i], selected_labels[j], label_diff))
    
    if high_sim_pairs:
        print(f"\n发现 {len(high_sim_pairs)} 对高相似性序列:")
        for i, j, sim, label1, label2, label_diff in high_sim_pairs:
            print(f"  序列{i}(类别{label1}) ↔ 序列{j}(类别{label2}): 相似度={sim:.4f}, 标签差={label_diff}")

def check_file_timestamps():
    """检查文件时间戳，分析轨迹的时间特性"""
    
    data_file = Path("data/processed/temporal_sequences_len5.pkl")
    if not data_file.exists():
        return
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    file_paths = data.get('file_paths', [])
    labels = data['labels']
    
    if not file_paths:
        print("没有文件路径信息")
        return
    
    print("文件路径分析:")
    print(f"总共 {len(file_paths)} 个序列")
    
    # 分析每个类别对应的文件
    class_files = defaultdict(list)
    for i, (paths, label) in enumerate(zip(file_paths, labels)):
        if isinstance(paths, list) and len(paths) > 0:
            # 提取文件名中的数字（通常是时间戳或帧号）
            first_file = Path(paths[0]).name
            class_files[label].append(first_file)
    
    print("\n每个类别的文件分布:")
    for label in sorted(class_files.keys()):
        files = class_files[label]
        print(f"类别 {label:2d}: {len(files)} 个文件")
        if len(files) <= 3:
            for f in files:
                print(f"    {f}")
        else:
            print(f"    {files[0]} ... {files[-1]}")

if __name__ == '__main__':
    print("="*60)
    print("轨迹回环分析")
    print("="*60)
    
    # 1. 尝试加载GPS数据
    gps_data = load_gps_data()
    
    # 2. 分析空间分布
    class_features, similar_pairs = analyze_spatial_distribution()
    
    # 3. 可视化相似性
    visualize_sequence_similarity()
    
    # 4. 检查文件时间戳
    check_file_timestamps()
    
    print("\n" + "="*60)
    print("分析总结")
    print("="*60)
    print("1. 如果发现高相似性但时间间隔大的类别对，说明确实存在回环")
    print("2. 如果只有时间相邻的类别相似，说明主要是时序连续性")
    print("3. 需要检查GPS轨迹数据来确认空间上的回环模式")
