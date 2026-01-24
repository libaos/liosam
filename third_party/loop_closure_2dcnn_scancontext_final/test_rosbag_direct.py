#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直接从rosbag的点云话题预测回环检测
"""

import bagpy
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from utils.scan_context import ScanContext
from models.temporal_models import *
from collections import deque
import pandas as pd
import struct
import warnings
warnings.filterwarnings('ignore')

class RosbagLoopDetector:
    """从rosbag直接进行回环检测"""
    
    def __init__(self, model_path=None, sequence_length=5):
        self.sequence_length = sequence_length
        self.sc_generator = ScanContext()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = Temporal2DCNN(input_shape=(sequence_length, 20, 60), num_classes=20)
        self.model = self.model.to(self.device)
        
        # 尝试加载预训练模型
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ 加载预训练模型: {model_path}")
        else:
            print("⚠️  使用随机初始化模型")
        
        self.model.eval()
        
        # 时序缓存
        self.sc_buffer = deque(maxlen=sequence_length)
        self.predictions = []
        self.confidences = []
        self.timestamps = []
        
    def parse_pointcloud_data(self, df):
        """解析点云数据DataFrame"""
        try:
            # 从DataFrame中提取x, y, z坐标
            if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
                points = df[['x', 'y', 'z']].values.astype(np.float32)
            else:
                print("DataFrame中未找到x, y, z列")
                return None

            # 过滤无效点
            valid_mask = np.isfinite(points).all(axis=1)
            points = points[valid_mask]

            if len(points) == 0:
                return None

            return points

        except Exception as e:
            print(f"点云数据解析失败: {e}")
            return None
    
    def generate_scancontext(self, points):
        """生成ScanContext特征"""
        if points is None or len(points) == 0:
            return None
        
        try:
            # 转换为ScanContext期望的格式
            sc = self.sc_generator.generate_scan_context(points)
            return sc
        except Exception as e:
            print(f"生成ScanContext失败: {e}")
            return None
    
    def predict_location(self, sc_feature, timestamp):
        """预测当前位置"""
        if sc_feature is None:
            return None, 0.0
        
        # 添加到缓存
        self.sc_buffer.append(sc_feature)
        
        # 如果缓存未满，返回None
        if len(self.sc_buffer) < self.sequence_length:
            return None, 0.0
        
        # 构建时序序列
        sequence = np.stack(list(self.sc_buffer), axis=0)  # (seq_len, 20, 60)
        sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # (1, seq_len, 20, 60)
        
        # 模型预测
        with torch.no_grad():
            outputs = self.model(sequence)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return predicted.item(), confidence.item()
    
    def process_rosbag(self, bag_path, topic_name=None):
        """处理rosbag文件"""

        print(f"处理rosbag: {bag_path}")

        try:
            # 使用bagpy读取rosbag
            bag = bagpy.bagreader(bag_path)

            # 获取所有话题信息
            print(f"rosbag中的话题:")
            for topic in bag.topic_table['Topics']:
                print(f"  {topic}")

            # 查找点云话题
            pointcloud_topics = []
            for topic in bag.topic_table['Topics']:
                if 'point' in topic.lower() or 'cloud' in topic.lower() or 'lidar' in topic.lower():
                    pointcloud_topics.append(topic)

            if not pointcloud_topics:
                print("❌ 未找到点云相关话题")
                # 尝试使用第一个话题
                if len(bag.topic_table['Topics']) > 0:
                    topic_name = bag.topic_table['Topics'][0]
                    print(f"尝试使用第一个话题: {topic_name}")
                else:
                    return None
            else:
                # 选择点云话题
                if topic_name is None:
                    topic_name = pointcloud_topics[0]
                print(f"使用点云话题: {topic_name}")

            # 读取话题数据
            print("正在读取话题数据...")
            try:
                df = bag.message_by_topic(topic_name)
                print(f"读取到 {len(df)} 条消息")
            except Exception as e:
                print(f"读取话题数据失败: {e}")
                return None

            results = {
                'timestamps': [],
                'predictions': [],
                'confidences': [],
                'sc_features': []
            }

            # 处理点云消息
            processed_count = 0
            max_process = min(500, len(df))  # 限制处理数量

            for i in range(0, max_process, 10):  # 每10个消息处理一个
                if i % 50 == 0:
                    print(f"处理第 {i+1}/{max_process} 个消息...")

                try:
                    # 这里需要根据实际的rosbag数据格式来解析
                    # 由于bagpy的限制，我们可能需要直接处理已提取的ply文件
                    print("bagpy无法直接解析PointCloud2消息，建议使用已提取的ply文件")
                    break

                except Exception as e:
                    print(f"处理消息 {i} 失败: {e}")
                    continue

            print(f"总共处理了 {processed_count} 个有效消息")

            return results

        except Exception as e:
            print(f"处理rosbag失败: {e}")
            return None
    
    def analyze_results(self, results):
        """分析预测结果"""
        
        print("\n" + "="*60)
        print("rosbag回环检测结果分析")
        print("="*60)
        
        predictions = np.array(results['predictions'])
        confidences = np.array(results['confidences'])
        timestamps = np.array(results['timestamps'])
        
        # 基本统计
        valid_predictions = predictions[predictions >= 0]
        print(f"总消息数: {len(results['timestamps'])}")
        print(f"有效预测数: {len(valid_predictions)}")
        
        if len(valid_predictions) > 0:
            print(f"预测类别范围: {np.min(valid_predictions)} - {np.max(valid_predictions)}")
            print(f"平均置信度: {np.mean(confidences):.4f}")
            print(f"置信度标准差: {np.std(confidences):.4f}")
            
            # 预测分布
            print(f"\n预测类别分布:")
            unique, counts = np.unique(valid_predictions, return_counts=True)
            for cls, count in zip(unique, counts):
                print(f"  类别 {cls:2d}: {count:3d} 次 ({count/len(valid_predictions)*100:.1f}%)")
            
            # 检测潜在回环
            print(f"\n潜在回环检测:")
            self.detect_loops_temporal(results)
        
        return results
    
    def detect_loops_temporal(self, results):
        """基于时间序列检测潜在回环"""
        
        predictions = np.array(results['predictions'])
        confidences = np.array(results['confidences'])
        timestamps = np.array(results['timestamps'])
        
        # 找出高置信度的预测
        high_conf_mask = confidences > 0.7
        high_conf_predictions = predictions[high_conf_mask]
        high_conf_timestamps = timestamps[high_conf_mask]
        
        if len(high_conf_predictions) == 0:
            print("未发现高置信度预测")
            return
        
        # 统计每个类别的出现时间
        class_times = {}
        for pred, timestamp in zip(high_conf_predictions, high_conf_timestamps):
            if pred not in class_times:
                class_times[pred] = []
            class_times[pred].append(timestamp)
        
        # 找出出现多次的类别（潜在回环）
        potential_loops = []
        for cls, times in class_times.items():
            if len(times) > 1:
                time_gaps = []
                for i in range(1, len(times)):
                    gap = times[i] - times[i-1]
                    time_gaps.append(gap)
                potential_loops.append((cls, times, time_gaps))
        
        if potential_loops:
            print(f"发现 {len(potential_loops)} 个潜在回环:")
            for cls, times, gaps in potential_loops:
                print(f"  类别 {cls}: 出现 {len(times)} 次")
                print(f"    时间间隔: {[f'{gap:.1f}s' for gap in gaps]}")
        else:
            print("未发现明显的回环模式")
    
    def visualize_results(self, results, save_dir="results/rosbag_test"):
        """可视化结果"""
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        predictions = np.array(results['predictions'])
        confidences = np.array(results['confidences'])
        timestamps = np.array(results['timestamps'])
        
        # 转换时间戳为相对时间
        if len(timestamps) > 0:
            relative_times = timestamps - timestamps[0]
        else:
            return
        
        # 1. 时间序列预测图
        plt.figure(figsize=(15, 10))
        
        plt.subplot(3, 1, 1)
        valid_mask = predictions >= 0
        plt.plot(relative_times[valid_mask], predictions[valid_mask], 'bo-', markersize=2, linewidth=0.5)
        plt.xlabel('时间 (秒)')
        plt.ylabel('预测类别')
        plt.title('回环检测预测时间序列')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 2)
        plt.plot(relative_times, confidences, 'ro-', markersize=1, linewidth=0.5)
        plt.xlabel('时间 (秒)')
        plt.ylabel('置信度')
        plt.title('预测置信度时间序列')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 3)
        if len(predictions[valid_mask]) > 0:
            plt.hist(predictions[valid_mask], bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('预测类别')
            plt.ylabel('频次')
            plt.title('预测类别分布')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/rosbag_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 可视化结果已保存到 {save_dir}")

def main():
    """主函数"""
    
    print("开始从rosbag直接进行回环检测...")
    
    # rosbag路径
    bag_path = "/mysda/shared_dir/2025.7.3/2025-07-03-16-28-57.bag"
    
    # 创建检测器
    detector = RosbagLoopDetector(
        model_path="models/saved/temporal_2d_cnn_best.pth",
        sequence_length=5
    )
    
    # 处理rosbag
    results = detector.process_rosbag(bag_path)
    
    if results and len(results['predictions']) > 0:
        # 分析结果
        detector.analyze_results(results)
        
        # 可视化
        detector.visualize_results(results)
        
        # 保存结果
        import pickle
        with open('results/rosbag_direct_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n✅ 测试完成！结果已保存到 results/rosbag_direct_results.pkl")
    else:
        print("❌ 未能处理任何数据")

if __name__ == '__main__':
    main()
