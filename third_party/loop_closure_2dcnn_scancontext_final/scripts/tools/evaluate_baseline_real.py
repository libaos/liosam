#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
真实基线模型评估脚本
评估刚刚训练完成的基线模型的实际性能
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# 导入模块
from models.sc_ring_cnn import SCRingCNN
from utils.scan_context import ScanContext
import open3d as o3d

def load_trajectory_segments():
    """加载轨迹分段配置"""
    segments = []
    for i in range(1, 21):  # 20个分段
        start_seq = 2 + (i-1) * 88
        end_seq = start_seq + 87
        segments.append({
            'id': i,
            'start': start_seq,
            'end': end_seq,
            'name': f'段{i}'
        })
    return segments

def load_point_cloud_data():
    """加载点云数据"""
    data_dir = Path("data/2025-07-03-16-28-57ply提取文件3")
    
    # 获取所有PLY文件
    ply_files = sorted(list(data_dir.glob("*.ply")))
    
    print(f"📊 找到 {len(ply_files)} 个PLY文件")
    
    # 提取序列号
    sequence_numbers = []
    valid_files = []
    
    for ply_file in ply_files:
        try:
            # 从文件名提取序列号
            filename = ply_file.stem  # cloud_00002
            seq_num = int(filename.split('_')[-1])
            sequence_numbers.append(seq_num)
            valid_files.append(ply_file)
        except:
            continue
    
    print(f"✅ 成功提取 {len(sequence_numbers)} 个序列号")
    
    return valid_files, sequence_numbers

def assign_segment_labels(sequence_numbers, segments):
    """分配分段标签"""
    labels = []
    
    for seq_num in sequence_numbers:
        # 找到对应的分段
        segment_id = None
        for segment in segments:
            if segment['start'] <= seq_num <= segment['end']:
                segment_id = segment['id']
                break
        
        if segment_id is None:
            # 如果不在任何分段内，分配到最近的分段
            distances = [abs(seq_num - (seg['start'] + seg['end']) / 2) for seg in segments]
            segment_id = segments[np.argmin(distances)]['id']
        
        labels.append(segment_id)
    
    return labels

def generate_scan_context_manual(points, num_rings=20, num_sectors=60, max_range=50.0):
    """手动实现ScanContext生成"""
    # 初始化ScanContext矩阵
    sc = np.zeros((num_rings, num_sectors))
    
    if len(points) == 0:
        return sc
    
    # 计算极坐标
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # 计算距离和角度
    distances = np.sqrt(x**2 + y**2)
    angles = np.arctan2(y, x)
    
    # 过滤超出范围的点
    valid_mask = distances <= max_range
    distances = distances[valid_mask]
    angles = angles[valid_mask]
    heights = z[valid_mask]
    
    if len(distances) == 0:
        return sc
    
    # 计算环和扇区索引
    ring_indices = np.floor(distances / max_range * num_rings).astype(int)
    ring_indices = np.clip(ring_indices, 0, num_rings - 1)
    
    # 角度归一化到[0, 2π]
    angles = (angles + np.pi) % (2 * np.pi)
    sector_indices = np.floor(angles / (2 * np.pi) * num_sectors).astype(int)
    sector_indices = np.clip(sector_indices, 0, num_sectors - 1)
    
    # 填充ScanContext - 使用最大高度
    for i in range(len(ring_indices)):
        ring_idx = ring_indices[i]
        sector_idx = sector_indices[i]
        sc[ring_idx, sector_idx] = max(sc[ring_idx, sector_idx], heights[i])
    
    return sc

def extract_scan_context_features(ply_files):
    """提取ScanContext特征"""
    scan_contexts = []
    
    print("🔄 开始提取ScanContext特征...")
    
    for i, ply_file in enumerate(tqdm(ply_files, desc="提取特征")):
        try:
            # 加载点云
            pcd = o3d.io.read_point_cloud(str(ply_file))
            points = np.asarray(pcd.points)
            
            if len(points) == 0:
                print(f"⚠️  文件 {ply_file} 为空")
                sc = np.zeros((20, 60))
                scan_contexts.append(sc)
                continue
            
            # 手动实现ScanContext提取
            sc = generate_scan_context_manual(points, num_rings=20, num_sectors=60, max_range=50.0)
            scan_contexts.append(sc)
            
        except Exception as e:
            print(f"⚠️  处理文件 {ply_file} 时出错: {e}")
            # 创建零填充的ScanContext
            sc = np.zeros((20, 60))
            scan_contexts.append(sc)
    
    return np.array(scan_contexts)

def evaluate_baseline_model():
    """评估基线模型"""
    print("🚀 开始评估真实基线模型")
    print("=" * 60)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    # 加载训练好的模型
    model_path = Path("experiments/baseline_training_20250723_135737/best_baseline_model.pth")
    if not model_path.exists():
        print("❌ 找不到训练好的基线模型")
        return None
    
    print(f"📁 加载模型: {model_path}")
    
    # 创建模型
    model = SCRingCNN(descriptor_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("✅ 模型加载成功")
    
    # 1. 加载数据
    print("📋 加载轨迹分段配置...")
    segments = load_trajectory_segments()
    
    print("📁 加载点云数据...")
    ply_files, sequence_numbers = load_point_cloud_data()
    
    print("🏷️ 分配段标签...")
    labels = assign_segment_labels(sequence_numbers, segments)
    
    # 2. 提取特征
    print("🔄 提取ScanContext特征...")
    scan_contexts = extract_scan_context_features(ply_files)
    
    print(f"✅ 提取了 {scan_contexts.shape} 的ScanContext特征")
    
    # 3. 生成描述子
    print("🧠 生成描述子...")
    descriptors = []
    
    with torch.no_grad():
        for i, sc in enumerate(tqdm(scan_contexts, desc="生成描述子")):
            # 转换为张量
            sc_tensor = torch.FloatTensor(sc).unsqueeze(0).unsqueeze(0).to(device)
            
            # 前向传播
            descriptor = model(sc_tensor)
            descriptors.append(descriptor.cpu().numpy().flatten())
    
    descriptors = np.array(descriptors)
    print(f"✅ 生成了 {descriptors.shape} 的描述子")
    
    # 4. 计算距离矩阵
    print("📊 计算距离矩阵...")
    num_samples = len(descriptors)
    distance_matrix = np.zeros((num_samples, num_samples))
    
    for i in tqdm(range(num_samples), desc="计算距离"):
        for j in range(num_samples):
            # 使用欧几里得距离
            distance_matrix[i, j] = np.linalg.norm(descriptors[i] - descriptors[j])
    
    # 5. 地点识别评估
    print("🎯 进行地点识别评估...")
    
    # 创建测试集（使用100个样本）
    test_indices = np.random.choice(num_samples, min(100, num_samples), replace=False)
    
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    
    for query_idx in tqdm(test_indices, desc="地点识别测试"):
        query_label = labels[query_idx]
        
        # 获取距离（排除自己）
        distances = distance_matrix[query_idx].copy()
        distances[query_idx] = float('inf')  # 排除自己
        
        # 找到最近的邻居
        nearest_indices = np.argsort(distances)
        
        # Top-1准确率
        if labels[nearest_indices[0]] == query_label:
            top1_correct += 1
        
        # Top-5准确率
        top5_labels = [labels[idx] for idx in nearest_indices[:5]]
        if query_label in top5_labels:
            top5_correct += 1
        
        # Top-10准确率
        top10_labels = [labels[idx] for idx in nearest_indices[:10]]
        if query_label in top10_labels:
            top10_correct += 1
    
    # 计算准确率
    num_test = len(test_indices)
    top1_accuracy = top1_correct / num_test
    top5_accuracy = top5_correct / num_test
    top10_accuracy = top10_correct / num_test
    
    # 6. 分类评估
    print("🏷️ 进行分类评估...")
    
    # 使用最近邻分类
    predicted_labels = []
    true_labels = []
    
    for query_idx in test_indices:
        query_label = labels[query_idx]
        true_labels.append(query_label)
        
        # 获取距离（排除自己）
        distances = distance_matrix[query_idx].copy()
        distances[query_idx] = float('inf')
        
        # 找到最近的邻居
        nearest_idx = np.argmin(distances)
        predicted_labels.append(labels[nearest_idx])
    
    # 计算分类指标
    classification_accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    
    # 7. 描述子质量评估
    print("🔬 评估描述子质量...")
    
    # 计算类内和类间距离
    intra_class_distances = []
    inter_class_distances = []
    
    unique_labels = list(set(labels))
    
    for label in unique_labels:
        label_indices = [i for i, l in enumerate(labels) if l == label]
        
        # 类内距离
        for i in range(len(label_indices)):
            for j in range(i+1, len(label_indices)):
                idx1, idx2 = label_indices[i], label_indices[j]
                intra_class_distances.append(distance_matrix[idx1, idx2])
        
        # 类间距离
        other_indices = [i for i, l in enumerate(labels) if l != label]
        for idx1 in label_indices[:5]:  # 限制计算量
            for idx2 in other_indices[:10]:
                inter_class_distances.append(distance_matrix[idx1, idx2])
    
    avg_intra_distance = np.mean(intra_class_distances) if intra_class_distances else 0
    avg_inter_distance = np.mean(inter_class_distances) if inter_class_distances else 0
    separation_ratio = avg_inter_distance / avg_intra_distance if avg_intra_distance > 0 else 0
    
    # 8. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "model_info": {
            "model_path": str(model_path),
            "model_type": "SCRingCNN_Baseline",
            "descriptor_dim": 512,
            "evaluation_time": timestamp
        },
        "data_info": {
            "num_samples": num_samples,
            "num_test_samples": num_test,
            "num_segments": len(unique_labels),
            "scan_context_shape": list(scan_contexts.shape)
        },
        "place_recognition": {
            "top_1": float(top1_accuracy),
            "top_5": float(top5_accuracy),
            "top_10": float(top10_accuracy)
        },
        "classification": {
            "accuracy": float(classification_accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        },
        "descriptor_quality": {
            "intra_class_distance": float(avg_intra_distance),
            "inter_class_distance": float(avg_inter_distance),
            "separation_ratio": float(separation_ratio)
        },
        "training_info": {
            "final_loss": 0.0266,
            "epochs": 80,
            "best_epoch": "80"
        }
    }
    
    # 保存结果
    results_file = f"evaluation_results/baseline_evaluation_{timestamp}.json"
    os.makedirs("evaluation_results", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印结果
    print("=" * 60)
    print("🎉 基线模型评估完成!")
    print("=" * 60)
    print(f"📊 地点识别性能:")
    print(f"   Top-1准确率: {top1_accuracy:.1%} ({top1_correct}/{num_test})")
    print(f"   Top-5准确率: {top5_accuracy:.1%} ({top5_correct}/{num_test})")
    print(f"   Top-10准确率: {top10_accuracy:.1%} ({top10_correct}/{num_test})")
    print()
    print(f"🏷️ 分类性能:")
    print(f"   分类准确率: {classification_accuracy:.1%}")
    print(f"   精确率: {precision:.1%}")
    print(f"   召回率: {recall:.1%}")
    print(f"   F1分数: {f1:.3f}")
    print()
    print(f"🔬 描述子质量:")
    print(f"   类内距离: {avg_intra_distance:.4f}")
    print(f"   类间距离: {avg_inter_distance:.4f}")
    print(f"   分离度: {separation_ratio:.2f}")
    print()
    print(f"📁 结果保存在: {results_file}")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    try:
        results = evaluate_baseline_model()
        
        if results:
            print("\n🎯 真实基线性能已确认!")
            print("现在可以训练CBAM模型并进行对比了")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
