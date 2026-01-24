#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用已提取的ply文件进行回环检测，模拟rosbag的时序处理
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from utils.scan_context import ScanContext
from models.temporal_models import *
from collections import deque
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')

class SimpleLoopDetector:
    """简单的回环检测器"""
    
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
        
    def load_point_cloud(self, ply_file):
        """加载点云文件"""
        try:
            from utils.ply_reader import PLYReader
            points = PLYReader.read_ply_file(str(ply_file))
            if points is not None and len(points) > 0:
                # 只取前3列（x, y, z坐标）
                if points.shape[1] >= 3:
                    return points[:, :3]
                else:
                    return points
            return None
        except Exception as e:
            print(f"加载点云失败 {ply_file}: {e}")
            return None
    
    def generate_scancontext(self, points):
        """生成ScanContext特征"""
        if points is None or len(points) == 0:
            return None
        
        try:
            sc = self.sc_generator.generate_scan_context(points)
            return sc
        except Exception as e:
            print(f"生成ScanContext失败: {e}")
            return None
    
    def predict_location(self, sc_feature):
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
    
    def process_data_directory(self, data_dir, sample_rate=10):
        """处理数据目录，模拟rosbag的时序处理"""
        
        print(f"处理数据目录: {data_dir}")
        
        # 获取所有ply文件并按名称排序（模拟时间顺序）
        ply_files = sorted(glob.glob(f"{data_dir}/*.ply"))
        print(f"找到 {len(ply_files)} 个点云文件")
        
        if len(ply_files) == 0:
            print("未找到点云文件")
            return None
        
        # 按采样率处理文件
        sampled_files = ply_files[::sample_rate]  # 每sample_rate个文件取一个
        print(f"按 1:{sample_rate} 采样，处理 {len(sampled_files)} 个文件")
        
        results = {
            'file_indices': [],
            'file_names': [],
            'predictions': [],
            'confidences': [],
            'sc_features': []
        }
        
        # 处理每个点云文件
        for i, ply_file in enumerate(sampled_files):
            if i % 20 == 0:
                print(f"处理 {i+1}/{len(sampled_files)}: {Path(ply_file).name}")
            
            # 加载点云
            points = self.load_point_cloud(ply_file)
            if points is None:
                continue
            
            # 生成ScanContext
            sc_feature = self.generate_scancontext(points)
            if sc_feature is None:
                continue
            
            # 预测位置
            prediction, confidence = self.predict_location(sc_feature)
            
            # 保存结果
            original_index = ply_files.index(ply_file)  # 在原始文件列表中的索引
            results['file_indices'].append(original_index)
            results['file_names'].append(Path(ply_file).name)
            results['predictions'].append(prediction if prediction is not None else -1)
            results['confidences'].append(confidence)
            results['sc_features'].append(sc_feature)
        
        return results
    
    def analyze_results(self, results):
        """分析预测结果"""
        
        print("\n" + "="*60)
        print("回环检测结果分析")
        print("="*60)
        
        predictions = np.array(results['predictions'])
        confidences = np.array(results['confidences'])
        file_indices = np.array(results['file_indices'])
        
        # 基本统计
        valid_predictions = predictions[predictions >= 0]
        print(f"总文件数: {len(results['file_names'])}")
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
            self.detect_loops(results)
        
        return results
    
    def detect_loops(self, results):
        """检测潜在的回环"""
        
        predictions = np.array(results['predictions'])
        confidences = np.array(results['confidences'])
        file_indices = np.array(results['file_indices'])
        
        # 找出高置信度的重复预测
        high_conf_mask = confidences > 0.5  # 降低阈值
        high_conf_predictions = predictions[high_conf_mask]
        high_conf_indices = file_indices[high_conf_mask]
        
        if len(high_conf_predictions) == 0:
            print("未发现高置信度预测")
            return
        
        # 统计每个类别的出现位置
        class_positions = {}
        for pred, idx in zip(high_conf_predictions, high_conf_indices):
            if pred not in class_positions:
                class_positions[pred] = []
            class_positions[pred].append(idx)
        
        # 找出出现多次的类别（潜在回环）
        potential_loops = []
        for cls, positions in class_positions.items():
            if len(positions) > 1:
                gaps = []
                for i in range(1, len(positions)):
                    gap = positions[i] - positions[i-1]
                    gaps.append(gap)
                potential_loops.append((cls, positions, gaps))
        
        if potential_loops:
            print(f"发现 {len(potential_loops)} 个潜在回环:")
            for cls, positions, gaps in potential_loops:
                print(f"  类别 {cls}: 出现 {len(positions)} 次")
                print(f"    文件索引: {positions}")
                print(f"    间隔: {gaps} 个文件")
        else:
            print("未发现明显的回环模式")
    
    def visualize_results(self, results, save_dir="results/simple_test"):
        """可视化结果"""
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        predictions = np.array(results['predictions'])
        confidences = np.array(results['confidences'])
        file_indices = np.array(results['file_indices'])
        
        # 1. 预测序列图
        plt.figure(figsize=(15, 10))
        
        plt.subplot(3, 1, 1)
        valid_mask = predictions >= 0
        plt.plot(file_indices[valid_mask], predictions[valid_mask], 'bo-', markersize=3, linewidth=1)
        plt.xlabel('文件索引')
        plt.ylabel('预测类别')
        plt.title('回环检测预测序列')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 2)
        plt.plot(file_indices, confidences, 'ro-', markersize=2, linewidth=1)
        plt.xlabel('文件索引')
        plt.ylabel('置信度')
        plt.title('预测置信度序列')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 3)
        if len(predictions[valid_mask]) > 0:
            plt.hist(predictions[valid_mask], bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('预测类别')
            plt.ylabel('频次')
            plt.title('预测类别分布')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/simple_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 可视化结果已保存到 {save_dir}")

def main():
    """主函数"""
    
    print("开始简单回环检测测试...")
    
    # 数据路径
    data_dir = "/mysda/shared_dir/2025.7.3/2025-07-03-16-28-57.ply"
    
    # 创建检测器
    detector = SimpleLoopDetector(
        model_path="models/saved/quick_trained_model.pth",
        sequence_length=5
    )
    
    # 处理数据（每20个文件采样1个，加快处理速度）
    results = detector.process_data_directory(data_dir, sample_rate=20)
    
    if results and len(results['predictions']) > 0:
        # 分析结果
        detector.analyze_results(results)
        
        # 可视化
        detector.visualize_results(results)
        
        # 保存结果
        with open('results/simple_test_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n✅ 测试完成！结果已保存到 results/simple_test_results.pkl")
    else:
        print("❌ 未能处理任何数据")

if __name__ == '__main__':
    main()
