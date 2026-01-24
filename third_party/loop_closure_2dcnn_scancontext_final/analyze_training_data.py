#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析训练数据和测试数据的差异
"""

import pickle
import numpy as np
from pathlib import Path

def analyze_training_data():
    """分析训练数据"""
    print("="*60)
    print("训练数据分析")
    print("="*60)
    
    # 加载训练数据
    with open('data/processed/temporal_sequences_len5.pkl', 'rb') as f:
        data = pickle.load(f)
    
    sequences = np.array(data['sequences'])
    labels = np.array(data['labels'])
    
    print(f"序列数量: {len(sequences)}")
    print(f"标签数量: {len(labels)}")
    print(f"序列形状: {sequences.shape}")
    print(f"标签范围: {np.min(labels)} - {np.max(labels)}")
    print(f"类别数: {len(np.unique(labels))}")
    
    print(f"\n标签分布:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        percentage = count / len(labels) * 100
        print(f"  类别 {label:2d}: {count:3d} 个序列 ({percentage:5.1f}%)")
    
    # 分析ScanContext特征
    print(f"\nScanContext特征分析:")
    print(f"特征形状: {sequences.shape[1:]}")  # (5, 20, 60)
    print(f"特征范围: {np.min(sequences):.4f} - {np.max(sequences):.4f}")
    print(f"特征均值: {np.mean(sequences):.4f}")
    print(f"特征标准差: {np.std(sequences):.4f}")
    
    return sequences, labels

def analyze_test_data():
    """分析测试时的ScanContext特征"""
    print("\n" + "="*60)
    print("测试数据分析 (从rosbag实时生成)")
    print("="*60)
    
    # 这里我们需要分析实时生成的ScanContext特征
    # 由于无法直接获取，我们分析一下可能的问题
    
    print("测试数据特征:")
    print("- 数据来源: rosbag /points_raw 话题")
    print("- 点云范围: 24153 - 32996 点")
    print("- 处理方式: 实时ScanContext生成")
    print("- 预测结果: 主要是类别1,5,18")
    
    print("\n可能的问题:")
    print("1. 训练数据是从ply文件生成的ScanContext")
    print("2. 测试数据是从rosbag实时生成的ScanContext")
    print("3. 两者的预处理可能不一致")
    print("4. 点云坐标系或尺度可能不同")

def compare_scancontext_generation():
    """对比ScanContext生成方式"""
    print("\n" + "="*60)
    print("ScanContext生成方式对比")
    print("="*60)
    
    from utils.scan_context import ScanContext
    from utils.ply_reader import PLYReader
    
    sc_generator = ScanContext()
    
    # 1. 从ply文件生成ScanContext (训练方式)
    ply_files = list(Path("/mysda/shared_dir/2025.7.3/2025-07-03-16-28-57.ply").glob("*.ply"))
    if len(ply_files) > 0:
        ply_file = ply_files[0]
        print(f"测试ply文件: {ply_file.name}")
        
        points_ply = PLYReader.read_ply_file(str(ply_file))
        if points_ply is not None:
            points_ply = points_ply[:, :3]  # 只取x,y,z
            sc_ply = sc_generator.generate_scan_context(points_ply)
            
            print(f"PLY点云:")
            print(f"  点数: {len(points_ply)}")
            print(f"  坐标范围: x[{np.min(points_ply[:,0]):.2f}, {np.max(points_ply[:,0]):.2f}]")
            print(f"             y[{np.min(points_ply[:,1]):.2f}, {np.max(points_ply[:,1]):.2f}]")
            print(f"             z[{np.min(points_ply[:,2]):.2f}, {np.max(points_ply[:,2]):.2f}]")
            print(f"  ScanContext形状: {sc_ply.shape}")
            print(f"  ScanContext范围: [{np.min(sc_ply):.4f}, {np.max(sc_ply):.4f}]")
            print(f"  ScanContext均值: {np.mean(sc_ply):.4f}")
            print(f"  ScanContext标准差: {np.std(sc_ply):.4f}")
    
    # 2. 加载训练数据中的ScanContext
    print(f"\n训练数据中的ScanContext:")
    with open('data/processed/temporal_sequences_len5.pkl', 'rb') as f:
        data = pickle.load(f)
    
    sequences = np.array(data['sequences'])
    # 取第一个序列的第一帧
    sc_train = sequences[0, 0]  # (20, 60)
    
    print(f"  ScanContext形状: {sc_train.shape}")
    print(f"  ScanContext范围: [{np.min(sc_train):.4f}, {np.max(sc_train):.4f}]")
    print(f"  ScanContext均值: {np.mean(sc_train):.4f}")
    print(f"  ScanContext标准差: {np.std(sc_train):.4f}")

def main():
    """主函数"""
    
    # 分析训练数据
    sequences, labels = analyze_training_data()
    
    # 分析测试数据
    analyze_test_data()
    
    # 对比ScanContext生成
    compare_scancontext_generation()
    
    print("\n" + "="*60)
    print("问题诊断和解决方案")
    print("="*60)
    
    print("可能的问题:")
    print("1. 🔍 数据不匹配: 训练用的是ply文件，测试用的是rosbag")
    print("2. 🔍 预处理不一致: ScanContext生成参数可能不同")
    print("3. 🔍 坐标系差异: ply和rosbag的坐标系可能不同")
    print("4. 🔍 数据分布偏移: 训练和测试的特征分布不匹配")
    
    print("\n解决方案:")
    print("1. ✅ 检查ScanContext生成参数是否一致")
    print("2. ✅ 对比训练和测试的特征分布")
    print("3. ✅ 使用相同的预处理流程")
    print("4. ✅ 考虑特征归一化或标准化")

if __name__ == '__main__':
    main()
