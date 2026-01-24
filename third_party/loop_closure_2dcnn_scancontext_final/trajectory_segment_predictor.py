#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
轨迹分段预测器 - 预测当前处于轨迹的第几段（0-19）
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from utils.scan_context import ScanContext
from models.temporal_models import Temporal3DCNN
from models.cnn_2d_models import Simple2DCNN, Enhanced2DCNN, ResNet2D
from collections import deque
import time
from utils.ply_reader import PLYReader
import warnings
warnings.filterwarnings('ignore')

# 尝试导入rosbag相关库
try:
    import rosbag
    import sensor_msgs.point_cloud2 as pc2
    ROSBAG_AVAILABLE = True
except ImportError:
    ROSBAG_AVAILABLE = False

class TrajectorySegmentPredictor:
    """轨迹分段预测器"""
    
    def __init__(self, model_path=None, sequence_length=5):
        self.sequence_length = sequence_length
        self.sc_generator = ScanContext()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🎯 轨迹分段预测器初始化")
        print(f"设备: {self.device}")
        print(f"目标: 预测轨迹段 0-19")
        
        # 初始化模型
        self.model = Temporal3DCNN(
            input_shape=(1, sequence_length, 20, 60),
            num_classes=20  # 20个轨迹段
        )
        self.model = self.model.to(self.device)
        
        # 加载模型
        if model_path and Path(model_path).exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✅ 加载预训练模型: {model_path}")
                    print(f"训练信息: 验证准确率 {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
                else:
                    self.model.load_state_dict(checkpoint)
                    print(f"✅ 加载预训练模型: {model_path}")
            except Exception as e:
                print(f"⚠️  加载模型失败: {e}")
                print("使用随机初始化模型")
        else:
            print("⚠️  使用随机初始化模型")
        
        self.model.eval()
        
        # 时序缓存
        self.sc_buffer = deque(maxlen=sequence_length)
        self.prediction_history = []
        self.confidence_history = []
        self.timestamp_history = []
        
    def pointcloud2_to_numpy(self, cloud_msg):
        """将PointCloud2消息转换为numpy数组"""
        if not ROSBAG_AVAILABLE:
            return None
            
        try:
            points_list = []
            for point in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
                points_list.append([point[0], point[1], point[2]])
            
            if len(points_list) == 0:
                return None
                
            points = np.array(points_list, dtype=np.float32)
            valid_mask = np.isfinite(points).all(axis=1)
            points = points[valid_mask]
            
            distances = np.linalg.norm(points[:, :2], axis=1)
            distance_mask = distances < 50.0
            points = points[distance_mask]
            
            if len(points) < 100:
                return None
                
            return points
            
        except Exception as e:
            print(f"点云转换失败: {e}")
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
    
    def predict_trajectory_segment(self, sc_feature, timestamp):
        """预测轨迹段"""
        if sc_feature is None:
            return None, 0.0
        
        self.sc_buffer.append(sc_feature)
        
        if len(self.sc_buffer) < self.sequence_length:
            return None, 0.0
        
        try:
            sequence = np.stack(list(self.sc_buffer), axis=0)
            sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(sequence)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                return predicted.item(), confidence.item()
                
        except Exception as e:
            print(f"预测失败: {e}")
            return None, 0.0
    
    def process_rosbag_for_trajectory_segments(self, bag_path, topic_name=None):
        """处理rosbag进行轨迹分段预测"""
        
        if not ROSBAG_AVAILABLE:
            print("❌ rosbag库不可用")
            return None
            
        try:
            bag = rosbag.Bag(bag_path, 'r')
        except Exception as e:
            print(f"无法打开rosbag文件: {e}")
            return None
        
        # 获取话题信息
        topics_info = bag.get_type_and_topic_info()[1]
        print(f"\nrosbag话题信息:")
        
        pointcloud_topics = []
        for topic, info in topics_info.items():
            print(f"  {topic}: {info.msg_type} ({info.message_count} 消息)")
            if 'PointCloud2' in info.msg_type:
                pointcloud_topics.append(topic)
        
        if not pointcloud_topics:
            print("❌ 未找到PointCloud2话题")
            bag.close()
            return None
        
        if topic_name is None:
            topic_name = pointcloud_topics[0]
        
        print(f"\n🎯 开始轨迹分段预测")
        print(f"使用话题: {topic_name}")
        print(f"目标: 预测0→1→2→...→19的轨迹段")
        print("-" * 60)
        
        # 处理消息
        total_messages = 0
        valid_predictions = 0
        start_time = time.time()
        
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            total_messages += 1
            timestamp = t.to_sec()
            
            # 转换点云
            points = self.pointcloud2_to_numpy(msg)
            if points is None:
                continue
            
            # 生成ScanContext
            sc_feature = self.generate_scancontext(points)
            if sc_feature is None:
                continue
            
            # 预测轨迹段
            predicted_segment, confidence = self.predict_trajectory_segment(sc_feature, timestamp)
            
            if predicted_segment is not None:
                valid_predictions += 1
                
                # 保存结果
                self.prediction_history.append(predicted_segment)
                self.confidence_history.append(confidence)
                self.timestamp_history.append(timestamp)
                
                # 计算期望的轨迹段（基于消息进度）
                expected_segment = int((total_messages - 1) / (1769 / 20))  # 假设总共1769个消息
                expected_segment = min(expected_segment, 19)
                
                # 实时输出
                status = "✅" if abs(predicted_segment - expected_segment) <= 2 else "❌"
                print(f"消息 {total_messages:4d} | 预测段: {predicted_segment:2d} | "
                      f"期望段: {expected_segment:2d} | 置信度: {confidence:.4f} | "
                      f"点数: {len(points):5d} {status}")
                
                # 每50个预测显示统计
                if valid_predictions % 50 == 0:
                    self.show_segment_stats(expected_segment)
            
            # 处理完整数据集
            if total_messages >= 1769:
                print(f"\n已处理完整数据集 ({total_messages} 个消息)")
                break
        
        bag.close()
        
        elapsed_time = time.time() - start_time
        print(f"\n" + "="*60)
        print("轨迹分段预测完成")
        print("="*60)
        print(f"总消息数: {total_messages}")
        print(f"有效预测数: {valid_predictions}")
        print(f"成功率: {valid_predictions/total_messages*100:.1f}%")
        print(f"处理时间: {elapsed_time:.1f}秒")
        print(f"处理频率: {valid_predictions/elapsed_time:.2f} Hz")
        
        return self.analyze_trajectory_prediction()
    
    def show_segment_stats(self, current_expected_segment):
        """显示分段统计"""
        if len(self.prediction_history) == 0:
            return
        
        predictions = np.array(self.prediction_history)
        confidences = np.array(self.confidence_history)
        
        print(f"\n--- 轨迹分段统计 (最近{len(predictions)}个预测) ---")
        print(f"当前期望段: {current_expected_segment}")
        print(f"预测段范围: {np.min(predictions)} - {np.max(predictions)}")
        print(f"平均置信度: {np.mean(confidences):.4f}")
        print(f"最高置信度: {np.max(confidences):.4f}")
        
        # 最近预测的分布
        recent_predictions = predictions[-20:] if len(predictions) > 20 else predictions
        unique, counts = np.unique(recent_predictions, return_counts=True)
        print("最近20个预测的段分布:")
        for seg, count in zip(unique, counts):
            print(f"  段 {seg}: {count} 次")
        print("-" * 50)
    
    def analyze_trajectory_prediction(self):
        """分析轨迹预测结果"""
        if len(self.prediction_history) == 0:
            return None
        
        predictions = np.array(self.prediction_history)
        confidences = np.array(self.confidence_history)
        timestamps = np.array(self.timestamp_history)
        
        print(f"\n📊 轨迹分段预测分析")
        print(f"{'='*50}")
        
        # 基本统计
        print(f"预测段数量: {len(np.unique(predictions))}")
        print(f"预测段范围: {np.min(predictions)} - {np.max(predictions)}")
        print(f"平均置信度: {np.mean(confidences):.4f}")
        print(f"置信度标准差: {np.std(confidences):.4f}")
        
        # 段分布
        print(f"\n预测段分布:")
        unique, counts = np.unique(predictions, return_counts=True)
        for seg, count in zip(unique, counts):
            percentage = count / len(predictions) * 100
            avg_conf = np.mean(confidences[predictions == seg])
            print(f"  段 {seg:2d}: {count:3d} 次 ({percentage:5.1f}%) | 平均置信度: {avg_conf:.4f}")
        
        # 时序分析
        print(f"\n时序分析:")
        if len(timestamps) > 1:
            time_duration = timestamps[-1] - timestamps[0]
            print(f"数据时长: {time_duration:.1f} 秒")
            print(f"预测频率: {len(predictions)/time_duration:.2f} Hz")
        
        # 轨迹段进展分析
        print(f"\n轨迹段进展分析:")
        segment_changes = []
        for i in range(1, len(predictions)):
            if predictions[i] != predictions[i-1]:
                segment_changes.append((i, predictions[i-1], predictions[i]))
        
        print(f"段变化次数: {len(segment_changes)}")
        if len(segment_changes) > 0:
            print("主要段变化:")
            for i, (pos, from_seg, to_seg) in enumerate(segment_changes[:10]):
                print(f"  位置 {pos}: {from_seg} → {to_seg}")
        
        # 计算准确性（如果有期望的进展）
        expected_segments = np.linspace(0, 19, len(predictions)).astype(int)
        accuracy = np.mean(np.abs(predictions - expected_segments) <= 2) * 100
        print(f"\n准确性分析:")
        print(f"容忍度±2段的准确率: {accuracy:.1f}%")
        
        return {
            'predictions': predictions.tolist(),
            'confidences': confidences.tolist(),
            'timestamps': timestamps.tolist(),
            'accuracy': accuracy,
            'segment_changes': len(segment_changes)
        }
    
    def visualize_trajectory_prediction(self, results, save_path=None):
        """可视化轨迹预测结果"""
        if not results:
            return
        
        predictions = np.array(results['predictions'])
        confidences = np.array(results['confidences'])
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. 预测段时序图
        ax1.plot(predictions, 'bo-', markersize=3, linewidth=1, label='预测段')
        expected = np.linspace(0, 19, len(predictions))
        ax1.plot(expected, 'r--', alpha=0.7, label='期望段')
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('轨迹段')
        ax1.set_title('轨迹段预测 vs 期望')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1, 20)
        
        # 2. 置信度时序图
        ax2.plot(confidences, 'go-', markersize=2, linewidth=1)
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('置信度')
        ax2.set_title('预测置信度')
        ax2.grid(True, alpha=0.3)
        
        # 3. 段分布直方图
        ax3.hist(predictions, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('预测段')
        ax3.set_ylabel('频次')
        ax3.set_title('预测段分布')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 可视化结果保存到: {save_path}")
        
        plt.show()

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python trajectory_segment_predictor.py <bag_path> [topic_name]")
        return
    
    bag_path = sys.argv[1]
    topic_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("="*60)
    print("🎯 轨迹分段预测器")
    print("="*60)
    print(f"目标: 预测轨迹段 0→1→2→...→19")
    print(f"rosbag: {bag_path}")
    
    # 创建预测器
    predictor = TrajectorySegmentPredictor(
        model_path="models/saved/balanced_trajectory_model_avg63.7.pth",
        sequence_length=5
    )
    
    # 处理rosbag
    results = predictor.process_rosbag_for_trajectory_segments(bag_path, topic_name)
    
    if results:
        # 可视化
        predictor.visualize_trajectory_prediction(results, 'results/trajectory_segment_prediction.png')
        
        # 保存结果
        import pickle
        with open('results/trajectory_segment_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n✅ 轨迹分段预测完成！")
        print(f"准确率: {results['accuracy']:.1f}%")
    else:
        print("❌ 轨迹分段预测失败")

if __name__ == '__main__':
    main()
