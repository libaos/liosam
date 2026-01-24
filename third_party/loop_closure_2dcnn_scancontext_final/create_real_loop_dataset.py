#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于真实空间位置创建回环检测数据集
"""

import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from utils.ply_reader import PLYReader
from utils.scan_context import ScanContext
import glob

class RealLoopDatasetCreator:
    """真实回环数据集创建器"""
    
    def __init__(self, data_dir, min_loop_distance=50, position_threshold=5.0):
        self.data_dir = Path(data_dir)
        self.min_loop_distance = min_loop_distance  # 最小回环间隔
        self.position_threshold = position_threshold  # 位置聚类阈值
        self.sc_generator = ScanContext()
        
    def extract_positions_from_pointclouds(self):
        """从点云中提取机器人位置"""
        print("从点云中提取机器人位置...")
        
        ply_files = sorted(glob.glob(f"{self.data_dir}/*.ply"))
        positions = []
        valid_files = []
        
        for i, ply_file in enumerate(ply_files):
            if i % 50 == 0:
                print(f"处理 {i+1}/{len(ply_files)}: {Path(ply_file).name}")
            
            try:
                points = PLYReader.read_ply_file(ply_file)
                if points is not None and len(points) > 100:
                    # 使用点云的中心作为机器人位置的近似
                    center = np.mean(points[:, :3], axis=0)
                    positions.append(center[:2])  # 只用x,y坐标
                    valid_files.append(ply_file)
            except Exception as e:
                print(f"处理文件失败 {ply_file}: {e}")
                continue
        
        positions = np.array(positions)
        print(f"提取到 {len(positions)} 个有效位置")
        
        return positions, valid_files
    
    def detect_loops_with_clustering(self, positions, valid_files):
        """使用聚类方法检测回环"""
        print(f"\n使用DBSCAN聚类检测回环...")
        print(f"位置范围: x[{np.min(positions[:,0]):.2f}, {np.max(positions[:,0]):.2f}]")
        print(f"         y[{np.min(positions[:,1]):.2f}, {np.max(positions[:,1]):.2f}]")
        
        # 使用DBSCAN聚类
        clustering = DBSCAN(eps=self.position_threshold, min_samples=3)
        cluster_labels = clustering.fit_predict(positions)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"发现 {n_clusters} 个位置聚类")
        print(f"噪声点: {n_noise} 个")
        
        # 分析每个聚类
        loop_labels = []
        cluster_info = {}
        
        for i, (pos, file_path, cluster_id) in enumerate(zip(positions, valid_files, cluster_labels)):
            if cluster_id == -1:  # 噪声点
                loop_labels.append(-1)
            else:
                if cluster_id not in cluster_info:
                    cluster_info[cluster_id] = []
                cluster_info[cluster_id].append(i)
                loop_labels.append(cluster_id)
        
        # 过滤小聚类和检查回环间隔
        valid_clusters = {}
        final_labels = [-1] * len(positions)
        
        cluster_counter = 0
        for cluster_id, indices in cluster_info.items():
            if len(indices) < 3:  # 至少3个点才算一个有效聚类
                continue
            
            # 检查时间间隔
            indices = sorted(indices)
            time_gaps = []
            for j in range(1, len(indices)):
                gap = indices[j] - indices[j-1]
                time_gaps.append(gap)
            
            # 如果有足够大的时间间隔，说明是真正的回环
            if any(gap > self.min_loop_distance for gap in time_gaps):
                valid_clusters[cluster_counter] = indices
                for idx in indices:
                    final_labels[idx] = cluster_counter
                cluster_counter += 1
        
        print(f"有效回环聚类: {len(valid_clusters)}")
        for cluster_id, indices in valid_clusters.items():
            print(f"  聚类 {cluster_id}: {len(indices)} 个位置")
        
        return final_labels, valid_clusters, positions
    
    def create_loop_dataset(self, positions, valid_files, loop_labels, sequence_length=5):
        """创建回环检测数据集"""
        print(f"\n创建回环检测数据集...")
        
        sequences = []
        labels = []
        file_sequences = []
        
        # 生成ScanContext特征
        scan_contexts = []
        print("生成ScanContext特征...")
        
        for i, file_path in enumerate(valid_files):
            if i % 50 == 0:
                print(f"  生成ScanContext {i+1}/{len(valid_files)}")
            
            try:
                points = PLYReader.read_ply_file(file_path)
                if points is not None:
                    points = points[:, :3]  # 只取x,y,z
                    sc = self.sc_generator.generate_scan_context(points)
                    scan_contexts.append(sc)
                else:
                    scan_contexts.append(None)
            except Exception as e:
                print(f"生成ScanContext失败 {file_path}: {e}")
                scan_contexts.append(None)
        
        # 创建时序序列
        print("创建时序序列...")
        for i in range(len(scan_contexts) - sequence_length + 1):
            # 检查序列中的所有ScanContext都有效
            sequence_scs = scan_contexts[i:i+sequence_length]
            if all(sc is not None for sc in sequence_scs):
                # 使用序列中间帧的标签
                middle_idx = i + sequence_length // 2
                label = loop_labels[middle_idx]
                
                if label >= 0:  # 只使用有效的回环标签
                    sequence = np.stack(sequence_scs, axis=0)
                    sequences.append(sequence)
                    labels.append(label)
                    file_sequences.append([valid_files[j] for j in range(i, i+sequence_length)])
        
        print(f"创建了 {len(sequences)} 个有效序列")
        
        if len(sequences) == 0:
            print("❌ 没有创建任何有效序列")
            return None
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        # 统计标签分布
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"标签分布:")
        for label, count in zip(unique_labels, counts):
            print(f"  回环 {label}: {count} 个序列")
        
        return {
            'sequences': sequences,
            'labels': labels,
            'file_sequences': file_sequences,
            'positions': positions,
            'loop_labels': loop_labels
        }
    
    def visualize_loops(self, positions, loop_labels, save_path=None):
        """可视化回环检测结果"""
        plt.figure(figsize=(12, 8))
        
        # 绘制轨迹
        plt.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.3, linewidth=1, label='轨迹')
        
        # 绘制不同的回环聚类
        unique_labels = set(loop_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # 噪声点
                mask = np.array(loop_labels) == label
                plt.scatter(positions[mask, 0], positions[mask, 1], 
                           c='gray', s=20, alpha=0.5, label='非回环点')
            else:
                # 回环聚类
                mask = np.array(loop_labels) == label
                plt.scatter(positions[mask, 0], positions[mask, 1], 
                           c=[color], s=50, alpha=0.8, label=f'回环 {label}')
        
        plt.xlabel('X (米)')
        plt.ylabel('Y (米)')
        plt.title('回环检测结果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果保存到: {save_path}")
        
        plt.show()

def main():
    """主函数"""
    print("="*60)
    print("创建真实回环检测数据集")
    print("="*60)
    
    # 数据路径
    data_dir = "/mysda/shared_dir/2025.7.3/2025-07-03-16-28-57.ply"
    
    # 创建数据集创建器
    creator = RealLoopDatasetCreator(
        data_dir=data_dir,
        min_loop_distance=50,  # 最小50帧间隔才算回环
        position_threshold=8.0  # 8米范围内算同一位置
    )
    
    # 1. 提取位置
    positions, valid_files = creator.extract_positions_from_pointclouds()
    
    if len(positions) == 0:
        print("❌ 未能提取到任何位置信息")
        return
    
    # 2. 检测回环
    loop_labels, valid_clusters, positions = creator.detect_loops_with_clustering(positions, valid_files)
    
    # 3. 可视化回环
    creator.visualize_loops(positions, loop_labels, 'results/real_loop_detection.png')
    
    # 4. 创建数据集
    dataset = creator.create_loop_dataset(positions, valid_files, loop_labels, sequence_length=5)
    
    if dataset is not None:
        # 保存数据集
        output_path = "data/processed/real_loop_dataset.pkl"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"\n✅ 真实回环数据集已保存到: {output_path}")
        print(f"数据集统计:")
        print(f"  序列数量: {len(dataset['sequences'])}")
        print(f"  序列形状: {dataset['sequences'].shape}")
        print(f"  回环类别数: {len(np.unique(dataset['labels']))}")
    else:
        print("❌ 数据集创建失败")

if __name__ == '__main__':
    main()
