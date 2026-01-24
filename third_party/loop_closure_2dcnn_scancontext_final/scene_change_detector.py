#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于场景内容变化的轨迹分段检测器
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.scan_context import ScanContext
from utils.ply_reader import PLYReader
import glob
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine, euclidean
from scipy.signal import find_peaks
import pickle

class SceneChangeDetector:
    """场景变化检测器"""
    
    def __init__(self, similarity_threshold=0.8, min_segment_length=20):
        self.similarity_threshold = similarity_threshold
        self.min_segment_length = min_segment_length
        self.sc_generator = ScanContext()
        
    def compute_scene_features(self, data_dir):
        """计算所有帧的场景特征"""
        print("🔍 计算场景特征...")
        
        ply_files = sorted(glob.glob(f"{data_dir}/*.ply"))
        print(f"找到 {len(ply_files)} 个ply文件")
        
        features = []
        valid_indices = []
        
        for i, ply_file in enumerate(ply_files):
            if i % 100 == 0:
                print(f"  处理 {i+1}/{len(ply_files)}")
            
            try:
                points = PLYReader.read_ply_file(ply_file)
                if points is not None and len(points) > 100:
                    points = points[:, :3]
                    
                    # 计算多种场景特征
                    scene_features = self.extract_scene_features(points)
                    if scene_features is not None:
                        features.append(scene_features)
                        valid_indices.append(i)
                        
            except Exception as e:
                print(f"处理失败 {ply_file}: {e}")
                continue
        
        features = np.array(features)
        print(f"成功提取 {len(features)} 个场景特征")
        
        return features, valid_indices
    
    def extract_scene_features(self, points):
        """提取多维场景特征"""
        try:
            # 1. ScanContext特征
            sc = self.sc_generator.generate_scan_context(points)
            if sc is None:
                return None
            
            # 2. 点云统计特征
            stats_features = self.compute_point_cloud_stats(points)
            
            # 3. 空间分布特征
            spatial_features = self.compute_spatial_features(points)
            
            # 4. ScanContext统计特征
            sc_stats = self.compute_scancontext_stats(sc)
            
            # 合并所有特征
            all_features = np.concatenate([
                sc.flatten(),           # ScanContext原始特征 (1200维)
                stats_features,         # 统计特征 (10维)
                spatial_features,       # 空间特征 (15维)
                sc_stats               # ScanContext统计特征 (8维)
            ])
            
            return all_features
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None
    
    def compute_point_cloud_stats(self, points):
        """计算点云统计特征"""
        # 距离统计
        distances = np.linalg.norm(points[:, :2], axis=1)  # 到原点距离
        heights = points[:, 2]  # 高度
        
        features = [
            np.mean(distances),      # 平均距离
            np.std(distances),       # 距离标准差
            np.min(distances),       # 最小距离
            np.max(distances),       # 最大距离
            np.mean(heights),        # 平均高度
            np.std(heights),         # 高度标准差
            np.min(heights),         # 最小高度
            np.max(heights),         # 最大高度
            len(points),             # 点数
            np.mean(np.abs(heights)) # 平均绝对高度
        ]
        
        return np.array(features)
    
    def compute_spatial_features(self, points):
        """计算空间分布特征"""
        # 角度分布
        angles = np.arctan2(points[:, 1], points[:, 0])
        angle_hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
        angle_hist = angle_hist / len(points)  # 归一化
        
        # 距离分布
        distances = np.linalg.norm(points[:, :2], axis=1)
        dist_hist, _ = np.histogram(distances, bins=5, range=(0, 50))
        dist_hist = dist_hist / len(points)  # 归一化
        
        # 密度特征
        density_near = np.sum(distances < 10) / len(points)  # 近距离点密度
        density_far = np.sum(distances > 30) / len(points)   # 远距离点密度
        
        features = np.concatenate([
            angle_hist,      # 角度分布 (8维)
            dist_hist,       # 距离分布 (5维)
            [density_near, density_far]  # 密度特征 (2维)
        ])
        
        return features
    
    def compute_scancontext_stats(self, sc):
        """计算ScanContext统计特征"""
        features = [
            np.mean(sc),                    # 平均值
            np.std(sc),                     # 标准差
            np.max(sc),                     # 最大值
            np.min(sc),                     # 最小值
            np.count_nonzero(sc) / sc.size, # 非零比例
            np.mean(np.max(sc, axis=0)),    # 每列最大值的平均
            np.mean(np.max(sc, axis=1)),    # 每行最大值的平均
            np.sum(sc > 0.5) / sc.size      # 高值比例
        ]
        
        return np.array(features)
    
    def detect_scene_changes(self, features):
        """检测场景变化点"""
        print("🔍 检测场景变化...")
        
        # 1. 计算相邻帧之间的相似度
        similarities = []
        for i in range(1, len(features)):
            # 使用余弦相似度
            sim = 1 - cosine(features[i-1], features[i])
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # 2. 检测相似度的显著下降点
        # 使用滑动窗口平滑
        window_size = 5
        smoothed_sim = np.convolve(similarities, np.ones(window_size)/window_size, mode='same')
        
        # 计算相似度的负梯度（下降程度）
        gradient = -np.gradient(smoothed_sim)
        
        # 找到梯度峰值（相似度显著下降的点）
        peaks, properties = find_peaks(gradient, 
                                     height=np.std(gradient),  # 高度阈值
                                     distance=self.min_segment_length)  # 最小间距
        
        change_points = peaks + 1  # +1因为similarities比features少1个
        
        print(f"检测到 {len(change_points)} 个场景变化点")
        
        return similarities, change_points, gradient
    
    def create_segments(self, change_points, total_frames):
        """根据变化点创建分段"""
        segments = []
        
        # 添加起始点
        segment_starts = [0] + list(change_points) + [total_frames]
        
        for i in range(len(segment_starts) - 1):
            start = segment_starts[i]
            end = segment_starts[i + 1]
            segments.append((start, end))
        
        print(f"创建了 {len(segments)} 个分段:")
        for i, (start, end) in enumerate(segments):
            print(f"  段 {i}: 帧 {start:4d} - {end:4d} (长度: {end-start:3d})")
        
        return segments
    
    def cluster_scenes(self, features, n_clusters=None):
        """使用聚类方法分析场景类型"""
        print("🔍 场景聚类分析...")
        
        if n_clusters is None:
            # 自动确定最佳聚类数
            silhouette_scores = []
            K_range = range(2, min(21, len(features)//10))
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                score = silhouette_score(features, labels)
                silhouette_scores.append(score)
            
            best_k = K_range[np.argmax(silhouette_scores)]
            print(f"最佳聚类数: {best_k} (轮廓系数: {max(silhouette_scores):.3f})")
        else:
            best_k = n_clusters
        
        # 执行聚类
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # 分析聚类结果
        print("聚类结果:")
        for i in range(best_k):
            cluster_frames = np.where(cluster_labels == i)[0]
            print(f"  聚类 {i}: {len(cluster_frames)} 帧")
        
        return cluster_labels, kmeans
    
    def visualize_analysis(self, similarities, change_points, gradient, cluster_labels=None):
        """可视化分析结果"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # 1. 相似度曲线
        axes[0].plot(similarities, 'b-', alpha=0.7, label='帧间相似度')
        axes[0].axhline(y=self.similarity_threshold, color='r', linestyle='--', label='阈值')
        for cp in change_points:
            if cp < len(similarities):
                axes[0].axvline(x=cp, color='red', alpha=0.7)
        axes[0].set_ylabel('相似度')
        axes[0].set_title('帧间相似度变化')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 梯度曲线
        axes[1].plot(gradient, 'g-', alpha=0.7, label='相似度梯度')
        axes[1].plot(change_points, gradient[change_points], 'ro', markersize=8, label='变化点')
        axes[1].set_ylabel('梯度')
        axes[1].set_title('相似度梯度（场景变化检测）')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 聚类结果
        if cluster_labels is not None:
            axes[2].plot(cluster_labels, 'o-', markersize=3, alpha=0.7)
            axes[2].set_ylabel('聚类标签')
            axes[2].set_title('场景聚类结果')
            axes[2].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('帧索引')
        
        plt.tight_layout()
        plt.savefig('scene_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，不显示
        
        print("📊 分析图表已保存为 scene_analysis.png")
    
    def save_results(self, features, similarities, change_points, segments, cluster_labels, filename='scene_analysis_results.pkl'):
        """保存分析结果"""
        results = {
            'features': features,
            'similarities': similarities,
            'change_points': change_points,
            'segments': segments,
            'cluster_labels': cluster_labels,
            'similarity_threshold': self.similarity_threshold,
            'min_segment_length': self.min_segment_length
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"📁 分析结果已保存到 {filename}")

def main():
    """主函数"""
    print("🎯 基于场景内容变化的轨迹分段分析")
    print("="*60)
    
    # 数据路径
    data_dir = "/mysda/shared_dir/2025.7.3/2025-07-03-16-28-57.ply"
    
    # 创建检测器
    detector = SceneChangeDetector(similarity_threshold=0.8, min_segment_length=20)
    
    # 1. 计算场景特征
    features, valid_indices = detector.compute_scene_features(data_dir)
    
    if len(features) == 0:
        print("❌ 未能提取到有效特征")
        return
    
    print(f"✅ 特征维度: {features.shape}")
    
    # 2. 检测场景变化
    similarities, change_points, gradient = detector.detect_scene_changes(features)
    
    # 3. 创建分段
    segments = detector.create_segments(change_points, len(features))
    
    # 4. 聚类分析
    cluster_labels, kmeans = detector.cluster_scenes(features)
    
    # 5. 可视化结果
    detector.visualize_analysis(similarities, change_points, gradient, cluster_labels)
    
    # 6. 保存结果
    detector.save_results(features, similarities, change_points, segments, cluster_labels)
    
    # 7. 分析总结
    print("\n📊 分析总结:")
    print(f"总帧数: {len(features)}")
    print(f"检测到的变化点: {len(change_points)}")
    print(f"分段数量: {len(segments)}")
    print(f"聚类数量: {len(np.unique(cluster_labels))}")
    print(f"平均分段长度: {len(features) / len(segments):.1f} 帧")
    
    # 分段长度分布
    segment_lengths = [end - start for start, end in segments]
    print(f"分段长度范围: {min(segment_lengths)} - {max(segment_lengths)} 帧")
    
    return features, similarities, change_points, segments, cluster_labels

if __name__ == '__main__':
    main()
