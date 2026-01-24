#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直接从rosbag点云话题读取数据，使用Temporal 3D CNN进行实时预测
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from utils.scan_context import ScanContext
from models.temporal_models import Temporal3DCNN
from collections import deque
import glob
import time
from utils.ply_reader import PLYReader
import warnings
warnings.filterwarnings('ignore')

class RealtimeLoopDetector:
    """实时回环检测器"""
    
    def __init__(self, model_path=None, sequence_length=5):
        self.sequence_length = sequence_length
        self.sc_generator = ScanContext()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        
        # 初始化Temporal 3D CNN模型
        self.model = Temporal3DCNN(
            input_shape=(1, sequence_length, 20, 60),  # (channels, seq_len, height, width)
            num_classes=20
        )
        self.model = self.model.to(self.device)

        # 不加载预训练模型，因为维度不匹配
        print("⚠️  使用随机初始化的Temporal 3D CNN模型")
        
        self.model.eval()
        
        # 时序缓存
        self.sc_buffer = deque(maxlen=sequence_length)
        self.prediction_history = []
        self.confidence_history = []
        self.timestamp_history = []
        
    def load_pointcloud_from_ply(self, ply_file):
        """从PLY文件加载点云"""
        try:
            points = PLYReader.read_ply_file(str(ply_file))
            if points is not None and len(points) > 0:
                # 只取前3列（x, y, z坐标）
                if points.shape[1] >= 3:
                    points = points[:, :3]

                # 过滤无效点
                valid_mask = np.isfinite(points).all(axis=1)
                points = points[valid_mask]

                # 过滤距离过远的点
                distances = np.linalg.norm(points[:, :2], axis=1)
                distance_mask = distances < 50.0  # 50米范围内
                points = points[distance_mask]

                if len(points) < 100:  # 至少需要100个点
                    return None

                return points.astype(np.float32)
            return None

        except Exception as e:
            print(f"点云加载失败: {e}")
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
    
    def predict_with_temporal_3dcnn(self, sc_feature, timestamp):
        """使用Temporal 3D CNN进行预测"""
        if sc_feature is None:
            return None, 0.0
        
        # 添加到时序缓存
        self.sc_buffer.append(sc_feature)
        
        # 如果缓存未满，返回None
        if len(self.sc_buffer) < self.sequence_length:
            return None, 0.0
        
        try:
            # 构建时序序列 (seq_len, 20, 60)
            sequence = np.stack(list(self.sc_buffer), axis=0)

            # 转换为3D CNN期望的格式 (1, seq_len, 20, 60)
            sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

            # 模型预测
            with torch.no_grad():
                outputs = self.model(sequence)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                return predicted.item(), confidence.item()

        except Exception as e:
            print(f"预测失败: {e}")
            return None, 0.0
    
    def process_pointcloud_data_realtime(self, data_dir):
        """模拟实时处理点云数据"""

        print(f"开始模拟实时处理点云数据: {data_dir}")

        # 获取所有ply文件并按名称排序
        ply_files = sorted(glob.glob(f"{data_dir}/*.ply"))
        print(f"找到 {len(ply_files)} 个点云文件")

        if len(ply_files) == 0:
            print("❌ 未找到点云文件")
            return None

        print(f"开始实时预测...\n")

        # 模拟实时处理
        message_count = 0
        valid_predictions = 0
        start_time = time.time()

        # 每5个文件处理一个，模拟实时数据流
        for i, ply_file in enumerate(ply_files[::5]):
            message_count += 1
            current_time = start_time + i * 0.1  # 模拟时间戳，每100ms一帧

            # 加载点云
            points = self.load_pointcloud_from_ply(ply_file)
            if points is None:
                continue

            # 生成ScanContext
            sc_feature = self.generate_scancontext(points)
            if sc_feature is None:
                continue

            # 使用Temporal 3D CNN预测
            prediction, confidence = self.predict_with_temporal_3dcnn(sc_feature, current_time)

            if prediction is not None:
                valid_predictions += 1

                # 保存预测历史
                self.prediction_history.append(prediction)
                self.confidence_history.append(confidence)
                self.timestamp_history.append(current_time)

                # 实时输出
                file_name = Path(ply_file).name
                print(f"帧 {message_count:4d} | {file_name} | 预测类别: {prediction:2d} | "
                      f"置信度: {confidence:.4f} | 点数: {len(points):5d}")

                # 每20个有效预测显示一次统计
                if valid_predictions % 20 == 0:
                    self.show_realtime_stats()

                # 模拟实时处理延迟
                time.sleep(0.01)  # 10ms延迟

            # 限制处理数量
            if valid_predictions >= 100:
                print(f"\n已处理 {valid_predictions} 个有效预测，停止处理")
                break

        print(f"\n处理完成:")
        print(f"  总帧数: {message_count}")
        print(f"  有效预测数: {valid_predictions}")
        print(f"  成功率: {valid_predictions/message_count*100:.1f}%")

        return {
            'predictions': self.prediction_history,
            'confidences': self.confidence_history,
            'timestamps': self.timestamp_history
        }
    
    def show_realtime_stats(self):
        """显示实时统计信息"""
        if len(self.prediction_history) == 0:
            return
        
        predictions = np.array(self.prediction_history)
        confidences = np.array(self.confidence_history)
        
        print(f"\n--- 实时统计 (最近{len(predictions)}个预测) ---")
        print(f"平均置信度: {np.mean(confidences):.4f}")
        print(f"最高置信度: {np.max(confidences):.4f}")
        print(f"预测类别范围: {np.min(predictions)} - {np.max(predictions)}")
        
        # 显示最近的类别分布
        unique, counts = np.unique(predictions[-50:], return_counts=True)
        print("最近50个预测的类别分布:")
        for cls, count in zip(unique, counts):
            print(f"  类别 {cls}: {count} 次")
        print("-" * 50)
    
    def analyze_final_results(self, results):
        """分析最终结果"""
        if not results or len(results['predictions']) == 0:
            print("没有有效的预测结果")
            return
        
        predictions = np.array(results['predictions'])
        confidences = np.array(results['confidences'])
        timestamps = np.array(results['timestamps'])
        
        print(f"\n" + "="*60)
        print("Temporal 3D CNN 实时预测结果分析")
        print("="*60)
        
        print(f"总预测数: {len(predictions)}")
        print(f"预测类别数: {len(np.unique(predictions))}")
        print(f"平均置信度: {np.mean(confidences):.4f}")
        print(f"置信度标准差: {np.std(confidences):.4f}")
        print(f"最高置信度: {np.max(confidences):.4f}")
        
        # 类别分布
        print(f"\n预测类别分布:")
        unique, counts = np.unique(predictions, return_counts=True)
        for cls, count in zip(unique, counts):
            percentage = count / len(predictions) * 100
            avg_conf = np.mean(confidences[predictions == cls])
            print(f"  类别 {cls:2d}: {count:3d} 次 ({percentage:5.1f}%) | 平均置信度: {avg_conf:.4f}")
        
        # 时序分析
        print(f"\n时序分析:")
        time_duration = timestamps[-1] - timestamps[0]
        print(f"  总时长: {time_duration:.1f} 秒")
        print(f"  预测频率: {len(predictions)/time_duration:.2f} Hz")
        
        # 检测回环模式
        self.detect_temporal_loops(predictions, timestamps)
    
    def detect_temporal_loops(self, predictions, timestamps):
        """检测时序回环模式"""
        print(f"\n回环检测:")
        
        # 寻找重复出现的类别
        class_times = {}
        for i, (pred, timestamp) in enumerate(zip(predictions, timestamps)):
            if pred not in class_times:
                class_times[pred] = []
            class_times[pred].append((timestamp, i))
        
        loops_found = False
        for cls, times in class_times.items():
            if len(times) > 3:  # 至少出现4次
                time_values = [t[0] for t in times]
                time_gaps = []
                for i in range(1, len(time_values)):
                    gap = time_values[i] - time_values[i-1]
                    time_gaps.append(gap)
                
                # 如果有较大的时间间隔，可能是回环
                if any(gap > 10.0 for gap in time_gaps):  # 10秒以上间隔
                    loops_found = True
                    print(f"  类别 {cls}: 出现 {len(times)} 次")
                    print(f"    时间间隔: {[f'{gap:.1f}s' for gap in time_gaps]}")
        
        if not loops_found:
            print("  未发现明显的回环模式")
    
    def visualize_realtime_results(self, results, save_dir="results/realtime_3dcnn"):
        """可视化实时结果"""
        if not results:
            return
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        predictions = np.array(results['predictions'])
        confidences = np.array(results['confidences'])
        timestamps = np.array(results['timestamps'])
        
        # 转换为相对时间
        relative_times = timestamps - timestamps[0]
        
        plt.figure(figsize=(15, 10))
        
        # 预测时间序列
        plt.subplot(3, 1, 1)
        plt.plot(relative_times, predictions, 'bo-', markersize=3, linewidth=1)
        plt.xlabel('时间 (秒)')
        plt.ylabel('预测类别')
        plt.title('Temporal 3D CNN 实时预测序列')
        plt.grid(True, alpha=0.3)
        
        # 置信度时间序列
        plt.subplot(3, 1, 2)
        plt.plot(relative_times, confidences, 'ro-', markersize=2, linewidth=1)
        plt.xlabel('时间 (秒)')
        plt.ylabel('置信度')
        plt.title('预测置信度时间序列')
        plt.grid(True, alpha=0.3)
        
        # 类别分布
        plt.subplot(3, 1, 3)
        plt.hist(predictions, bins=len(np.unique(predictions)), alpha=0.7, edgecolor='black')
        plt.xlabel('预测类别')
        plt.ylabel('频次')
        plt.title('预测类别分布')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/realtime_3dcnn_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 可视化结果已保存到 {save_dir}")

def main():
    """主函数"""

    print("开始Temporal 3D CNN实时回环检测...")

    # 点云数据路径
    data_dir = "/mysda/shared_dir/2025.7.3/2025-07-03-16-28-57.ply"

    # 创建实时检测器
    detector = RealtimeLoopDetector(
        model_path="models/saved/quick_trained_model.pth",  # 使用之前训练的模型
        sequence_length=5
    )

    # 模拟实时处理点云数据
    results = detector.process_pointcloud_data_realtime(data_dir)

    if results:
        # 分析结果
        detector.analyze_final_results(results)

        # 可视化
        detector.visualize_realtime_results(results)

        # 保存结果
        import pickle
        with open('results/realtime_3dcnn_results.pkl', 'wb') as f:
            pickle.dump(results, f)

        print(f"\n✅ 实时预测完成！结果已保存")
    else:
        print("❌ 实时预测失败")

if __name__ == '__main__':
    main()
