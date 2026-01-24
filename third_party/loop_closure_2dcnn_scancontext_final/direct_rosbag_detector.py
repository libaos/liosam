#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直接读取rosbag点云话题进行回环检测
不依赖ROS环境，使用rosbag库直接解析
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from utils.scan_context import ScanContext
from models.temporal_models import Temporal3DCNN
from collections import deque
import time
import struct
import warnings
warnings.filterwarnings('ignore')

# 尝试导入rosbag相关库
try:
    import rosbag
    import sensor_msgs.point_cloud2 as pc2
    from sensor_msgs.msg import PointCloud2
    ROSBAG_AVAILABLE = True
    print("✅ rosbag库可用")
except ImportError:
    ROSBAG_AVAILABLE = False
    print("❌ rosbag库不可用，尝试使用bagpy")
    try:
        import bagpy
        BAGPY_AVAILABLE = True
        print("✅ bagpy库可用")
    except ImportError:
        BAGPY_AVAILABLE = False
        print("❌ bagpy库也不可用")

class DirectRosbagDetector:
    """直接读取rosbag的回环检测器"""
    
    def __init__(self, model_path=None, sequence_length=5):
        self.sequence_length = sequence_length
        self.sc_generator = ScanContext()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        
        # 初始化Temporal 3D CNN模型
        self.model = Temporal3DCNN(
            input_shape=(1, sequence_length, 20, 60),
            num_classes=20
        )
        self.model = self.model.to(self.device)
        
        # 加载模型
        if model_path and Path(model_path).exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                # 检查是否是完整的checkpoint还是只有state_dict
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✅ 加载预训练模型: {model_path}")
                    print(f"模型训练信息: 验证准确率 {checkpoint.get('best_val_acc', 'N/A'):.2f}%, 测试准确率 {checkpoint.get('test_acc', 'N/A'):.2f}%")
                else:
                    self.model.load_state_dict(checkpoint)
                    print(f"✅ 加载预训练模型: {model_path}")
            except Exception as e:
                print(f"⚠️  加载模型失败: {e}")
                print("使用随机初始化模型")
        else:
            print("⚠️  使用随机初始化的Temporal 3D CNN模型")
        
        self.model.eval()
        
        # 初始化存储
        self.sc_buffer = deque(maxlen=sequence_length)
        self.prediction_history = []
        self.confidence_history = []
        self.timestamp_history = []
        self.point_count_history = []
        
        self.total_messages = 0
        self.valid_predictions = 0
        self.start_time = time.time()
    
    def pointcloud2_to_numpy(self, cloud_msg):
        """将PointCloud2消息转换为numpy数组"""
        if not ROSBAG_AVAILABLE:
            print("❌ 无法解析PointCloud2消息，需要rosbag库")
            return None
            
        try:
            points_list = []
            
            # 读取点云数据
            for point in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
                points_list.append([point[0], point[1], point[2]])
            
            if len(points_list) == 0:
                return None
                
            points = np.array(points_list, dtype=np.float32)
            
            # 过滤无效点
            valid_mask = np.isfinite(points).all(axis=1)
            points = points[valid_mask]
            
            # 过滤距离过远的点
            distances = np.linalg.norm(points[:, :2], axis=1)
            distance_mask = distances < 50.0  # 50米范围内
            points = points[distance_mask]
            
            if len(points) < 100:  # 至少需要100个点
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
        recent_predictions = predictions[-50:] if len(predictions) > 50 else predictions
        unique, counts = np.unique(recent_predictions, return_counts=True)
        print("最近预测的类别分布:")
        for cls, count in zip(unique, counts):
            print(f"  类别 {cls}: {count} 次")
        print("-" * 50)
    
    def process_rosbag_with_rosbag(self, bag_path, topic_name=None):
        """使用rosbag库处理rosbag文件"""
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
        print(f"\nrosbag中的话题:")
        
        pointcloud_topics = []
        for topic, info in topics_info.items():
            print(f"  {topic}: {info.msg_type} ({info.message_count} 消息)")
            if 'PointCloud2' in info.msg_type:
                pointcloud_topics.append(topic)
        
        if not pointcloud_topics:
            print("❌ 未找到PointCloud2话题")
            bag.close()
            return None
        
        # 选择点云话题
        if topic_name is None:
            topic_name = pointcloud_topics[0]
        
        print(f"\n使用点云话题: {topic_name}")
        print(f"开始处理...\n")
        
        # 处理消息
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            self.total_messages += 1
            timestamp = t.to_sec()
            
            # 转换点云
            points = self.pointcloud2_to_numpy(msg)
            if points is None:
                continue
            
            # 生成ScanContext
            sc_feature = self.generate_scancontext(points)
            if sc_feature is None:
                continue
            
            # 预测
            prediction, confidence = self.predict_with_temporal_3dcnn(sc_feature, timestamp)
            
            if prediction is not None:
                self.valid_predictions += 1
                
                # 保存结果
                self.prediction_history.append(prediction)
                self.confidence_history.append(confidence)
                self.timestamp_history.append(timestamp)
                self.point_count_history.append(len(points))
                
                # 实时输出
                print(f"消息 {self.total_messages:4d} | 时间: {timestamp:.2f} | 预测类别: {prediction:2d} | "
                      f"置信度: {confidence:.4f} | 点数: {len(points):5d}")
                
                # 每50个有效预测显示统计
                if self.valid_predictions % 50 == 0:
                    self.show_realtime_stats()
            
            # 限制处理数量
            if self.valid_predictions >= 200:
                print(f"\n已处理 {self.valid_predictions} 个有效预测，停止处理")
                break
        
        bag.close()
        return self.get_final_results()
    
    def process_rosbag_with_bagpy(self, bag_path):
        """使用bagpy处理rosbag文件"""
        if not BAGPY_AVAILABLE:
            print("❌ bagpy库不可用")
            return None
        
        try:
            bag = bagpy.bagreader(bag_path)
            
            # 获取话题信息
            print(f"\nrosbag中的话题:")
            for topic in bag.topic_table['Topics']:
                print(f"  {topic}")
            
            # 查找点云话题
            pointcloud_topics = []
            for topic in bag.topic_table['Topics']:
                if 'point' in topic.lower() or 'cloud' in topic.lower() or 'lidar' in topic.lower():
                    pointcloud_topics.append(topic)
            
            if not pointcloud_topics:
                print("❌ 未找到点云相关话题")
                return None
            
            topic_name = pointcloud_topics[0]
            print(f"\n尝试使用话题: {topic_name}")
            
            # bagpy无法直接解析PointCloud2，需要其他方法
            print("⚠️  bagpy无法直接解析PointCloud2消息")
            print("建议使用rosbag库或转换为其他格式")
            
            return None
            
        except Exception as e:
            print(f"bagpy处理失败: {e}")
            return None
    
    def get_final_results(self):
        """获取最终结果"""
        elapsed_time = time.time() - self.start_time
        
        print(f"\n" + "="*60)
        print("直接rosbag回环检测结果分析")
        print("="*60)
        
        if self.valid_predictions == 0:
            print("❌ 没有有效的预测结果")
            return None
        
        predictions = np.array(self.prediction_history)
        confidences = np.array(self.confidence_history)
        timestamps = np.array(self.timestamp_history)
        point_counts = np.array(self.point_count_history)
        
        print(f"总消息数: {self.total_messages}")
        print(f"有效预测数: {self.valid_predictions}")
        print(f"成功率: {self.valid_predictions/self.total_messages*100:.1f}%")
        print(f"处理时间: {elapsed_time:.1f}秒")
        print(f"处理频率: {self.valid_predictions/elapsed_time:.2f} Hz")
        
        print(f"\n预测统计:")
        print(f"预测类别数: {len(np.unique(predictions))}")
        print(f"平均置信度: {np.mean(confidences):.4f}")
        print(f"置信度标准差: {np.std(confidences):.4f}")
        print(f"最高置信度: {np.max(confidences):.4f}")
        
        # 预测分布
        print(f"\n预测类别分布:")
        unique, counts = np.unique(predictions, return_counts=True)
        for cls, count in zip(unique, counts):
            percentage = count / len(predictions) * 100
            avg_conf = np.mean(confidences[predictions == cls])
            print(f"  类别 {cls:2d}: {count:3d} 次 ({percentage:5.1f}%) | 平均置信度: {avg_conf:.4f}")
        
        # 时序分析
        if len(timestamps) > 1:
            time_duration = timestamps[-1] - timestamps[0]
            print(f"\n时序分析:")
            print(f"数据时长: {time_duration:.1f} 秒")
            print(f"数据频率: {len(predictions)/time_duration:.2f} Hz")
        
        # 点云统计
        print(f"\n点云统计:")
        print(f"平均点数: {np.mean(point_counts):.0f}")
        print(f"点数范围: {np.min(point_counts)} - {np.max(point_counts)}")
        
        return {
            'total_messages': self.total_messages,
            'valid_predictions': self.valid_predictions,
            'success_rate': self.valid_predictions/self.total_messages*100,
            'processing_time': elapsed_time,
            'processing_frequency': self.valid_predictions/elapsed_time,
            'predictions': predictions.tolist(),
            'confidences': confidences.tolist(),
            'timestamps': timestamps.tolist(),
            'point_counts': point_counts.tolist()
        }
    
    def save_results(self, results, filename="direct_rosbag_results.pkl"):
        """保存结果"""
        if results is None:
            return
            
        import pickle
        results_dir = Path("results/direct_rosbag")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"✅ 结果已保存到 {results_dir / filename}")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python direct_rosbag_detector.py <bag_path> [topic_name]")
        print("示例: python direct_rosbag_detector.py /path/to/data.bag /velodyne_points")
        return
    
    bag_path = sys.argv[1]
    topic_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(bag_path).exists():
        print(f"❌ rosbag文件不存在: {bag_path}")
        return
    
    print(f"🚀 开始处理rosbag: {bag_path}")
    
    detector = DirectRosbagDetector(
        model_path="models/saved/temporal_3dcnn_seq5_acc11.5.pth",
        sequence_length=5
    )
    
    # 尝试使用rosbag库
    if ROSBAG_AVAILABLE:
        print("使用rosbag库处理...")
        results = detector.process_rosbag_with_rosbag(bag_path, topic_name)
    elif BAGPY_AVAILABLE:
        print("使用bagpy库处理...")
        results = detector.process_rosbag_with_bagpy(bag_path)
    else:
        print("❌ 没有可用的rosbag处理库")
        print("请安装: pip install rosbag 或 pip install bagpy")
        return
    
    if results:
        detector.save_results(results)
        print("\n✅ 处理完成！")
    else:
        print("\n❌ 处理失败")

if __name__ == '__main__':
    main()
