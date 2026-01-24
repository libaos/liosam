#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS点云话题回环检测节点
支持：1. 订阅实时点云话题  2. 读取rosbag中的点云话题
"""

import rospy
import numpy as np
import torch
import torch.nn as nn
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from pathlib import Path
import matplotlib.pyplot as plt
from utils.scan_context import ScanContext
from models.temporal_models import Temporal3DCNN
from collections import deque
import time
import threading
import rosbag
import warnings
warnings.filterwarnings('ignore')

class ROSLoopDetector:
    """ROS回环检测节点"""
    
    def __init__(self, model_path=None, sequence_length=5, topic_name="/velodyne_points"):
        # ROS初始化
        rospy.init_node('loop_detector', anonymous=True)
        
        self.sequence_length = sequence_length
        self.topic_name = topic_name
        self.sc_generator = ScanContext()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        rospy.loginfo(f"使用设备: {self.device}")
        rospy.loginfo(f"监听话题: {self.topic_name}")
        
        # 初始化Temporal 3D CNN模型
        self.model = Temporal3DCNN(
            input_shape=(1, sequence_length, 20, 60),
            num_classes=20
        )
        self.model = self.model.to(self.device)
        
        # 加载模型
        if model_path and Path(model_path).exists():
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                rospy.loginfo(f"✅ 加载预训练模型: {model_path}")
            except Exception as e:
                rospy.logwarn(f"⚠️  加载模型失败: {e}")
                rospy.logwarn("使用随机初始化模型")
        else:
            rospy.logwarn("⚠️  使用随机初始化的Temporal 3D CNN模型")
        
        self.model.eval()
        
        # 时序缓存和结果存储
        self.sc_buffer = deque(maxlen=sequence_length)
        self.prediction_history = []
        self.confidence_history = []
        self.timestamp_history = []
        self.point_count_history = []
        
        # 统计信息
        self.total_messages = 0
        self.valid_predictions = 0
        self.start_time = time.time()
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 订阅点云话题
        self.subscriber = rospy.Subscriber(
            self.topic_name, 
            PointCloud2, 
            self.pointcloud_callback,
            queue_size=10
        )
        
        rospy.loginfo("🚀 回环检测节点启动完成")
        rospy.loginfo("等待点云数据...")
    
    def pointcloud2_to_numpy(self, cloud_msg):
        """将PointCloud2消息转换为numpy数组"""
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
            rospy.logerr(f"点云转换失败: {e}")
            return None
    
    def generate_scancontext(self, points):
        """生成ScanContext特征"""
        if points is None or len(points) == 0:
            return None
        
        try:
            sc = self.sc_generator.generate_scan_context(points)
            return sc
        except Exception as e:
            rospy.logerr(f"生成ScanContext失败: {e}")
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
            rospy.logerr(f"预测失败: {e}")
            return None, 0.0
    
    def pointcloud_callback(self, msg):
        """点云话题回调函数"""
        with self.lock:
            self.total_messages += 1
            timestamp = msg.header.stamp.to_sec()
            
            # 转换点云
            points = self.pointcloud2_to_numpy(msg)
            if points is None:
                return
            
            # 生成ScanContext
            sc_feature = self.generate_scancontext(points)
            if sc_feature is None:
                return
            
            # 使用Temporal 3D CNN预测
            prediction, confidence = self.predict_with_temporal_3dcnn(sc_feature, timestamp)
            
            if prediction is not None:
                self.valid_predictions += 1
                
                # 保存预测历史
                self.prediction_history.append(prediction)
                self.confidence_history.append(confidence)
                self.timestamp_history.append(timestamp)
                self.point_count_history.append(len(points))
                
                # 实时输出
                rospy.loginfo(f"消息 {self.total_messages:4d} | 预测类别: {prediction:2d} | "
                             f"置信度: {confidence:.4f} | 点数: {len(points):5d}")
                
                # 每20个有效预测显示一次统计
                if self.valid_predictions % 20 == 0:
                    self.show_realtime_stats()
    
    def show_realtime_stats(self):
        """显示实时统计信息"""
        if len(self.prediction_history) == 0:
            return
        
        predictions = np.array(self.prediction_history)
        confidences = np.array(self.confidence_history)
        
        rospy.loginfo(f"\n--- 实时统计 (最近{len(predictions)}个预测) ---")
        rospy.loginfo(f"平均置信度: {np.mean(confidences):.4f}")
        rospy.loginfo(f"最高置信度: {np.max(confidences):.4f}")
        rospy.loginfo(f"预测类别范围: {np.min(predictions)} - {np.max(predictions)}")
        
        # 显示最近的类别分布
        recent_predictions = predictions[-50:] if len(predictions) > 50 else predictions
        unique, counts = np.unique(recent_predictions, return_counts=True)
        rospy.loginfo("最近预测的类别分布:")
        for cls, count in zip(unique, counts):
            rospy.loginfo(f"  类别 {cls}: {count} 次")
        rospy.loginfo("-" * 50)
    
    def get_statistics(self):
        """获取统计信息"""
        with self.lock:
            elapsed_time = time.time() - self.start_time
            return {
                'total_messages': self.total_messages,
                'valid_predictions': self.valid_predictions,
                'success_rate': self.valid_predictions / max(self.total_messages, 1) * 100,
                'elapsed_time': elapsed_time,
                'frequency': self.valid_predictions / max(elapsed_time, 1),
                'predictions': self.prediction_history.copy(),
                'confidences': self.confidence_history.copy(),
                'timestamps': self.timestamp_history.copy(),
                'point_counts': self.point_count_history.copy()
            }
    
    def save_results(self, filename="ros_loop_detection_results.pkl"):
        """保存结果"""
        import pickle
        stats = self.get_statistics()
        
        results_dir = Path("results/ros_detection")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / filename, 'wb') as f:
            pickle.dump(stats, f)
        
        rospy.loginfo(f"✅ 结果已保存到 {results_dir / filename}")
        return stats

class ROSBagLoopDetector(ROSLoopDetector):
    """从rosbag读取点云数据的回环检测器"""
    
    def __init__(self, bag_path, model_path=None, sequence_length=5, topic_name=None):
        self.bag_path = bag_path
        self.target_topic = topic_name
        
        # 不调用父类的__init__，因为我们不需要ROS节点
        self.sequence_length = sequence_length
        self.sc_generator = ScanContext()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        print(f"处理rosbag: {bag_path}")
        
        # 初始化模型
        self.model = Temporal3DCNN(
            input_shape=(1, sequence_length, 20, 60),
            num_classes=20
        )
        self.model = self.model.to(self.device)
        
        if model_path and Path(model_path).exists():
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
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
    
    def process_rosbag(self):
        """处理rosbag文件"""
        try:
            bag = rosbag.Bag(self.bag_path, 'r')
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
        if self.target_topic is None:
            topic_name = pointcloud_topics[0]
        else:
            topic_name = self.target_topic
        
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
                print(f"消息 {self.total_messages:4d} | 预测类别: {prediction:2d} | "
                      f"置信度: {confidence:.4f} | 点数: {len(points):5d}")
                
                # 每50个有效预测显示统计
                if self.valid_predictions % 50 == 0:
                    self.show_realtime_stats()
            
            # 限制处理数量
            if self.valid_predictions >= 200:
                print(f"\n已处理 {self.valid_predictions} 个有效预测，停止处理")
                break
        
        bag.close()
        
        elapsed_time = time.time() - self.start_time
        print(f"\n处理完成:")
        print(f"  总消息数: {self.total_messages}")
        print(f"  有效预测数: {self.valid_predictions}")
        print(f"  成功率: {self.valid_predictions/self.total_messages*100:.1f}%")
        print(f"  处理时间: {elapsed_time:.1f}秒")
        print(f"  处理频率: {self.valid_predictions/elapsed_time:.2f} Hz")
        
        return self.get_statistics()

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  实时订阅: python ros_loop_detector.py subscribe [topic_name]")
        print("  处理rosbag: python ros_loop_detector.py rosbag <bag_path> [topic_name]")
        return
    
    mode = sys.argv[1]
    
    if mode == "subscribe":
        # 实时订阅模式
        topic_name = sys.argv[2] if len(sys.argv) > 2 else "/velodyne_points"
        
        detector = ROSLoopDetector(
            model_path="models/saved/quick_trained_model.pth",
            sequence_length=5,
            topic_name=topic_name
        )
        
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("收到中断信号，正在保存结果...")
            stats = detector.save_results()
            rospy.loginfo("节点关闭")
    
    elif mode == "rosbag":
        # rosbag处理模式
        if len(sys.argv) < 3:
            print("请提供rosbag文件路径")
            return
        
        bag_path = sys.argv[2]
        topic_name = sys.argv[3] if len(sys.argv) > 3 else None
        
        detector = ROSBagLoopDetector(
            bag_path=bag_path,
            model_path="models/saved/quick_trained_model.pth",
            sequence_length=5,
            topic_name=topic_name
        )
        
        results = detector.process_rosbag()
        if results:
            detector.save_results("rosbag_loop_detection_results.pkl")
    
    else:
        print("未知模式，请使用 'subscribe' 或 'rosbag'")

if __name__ == '__main__':
    main()
