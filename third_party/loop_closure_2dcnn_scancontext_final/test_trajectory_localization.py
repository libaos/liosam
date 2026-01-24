#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试轨迹定位系统
"""

import numpy as np
import torch
from trajectory_localization_system import TrajectoryLocalizationSystem
from utils.scan_context import ScanContext
import time
import matplotlib.pyplot as plt

# 尝试导入rosbag相关库
try:
    import rosbag
    import sensor_msgs.point_cloud2 as pc2
    ROSBAG_AVAILABLE = True
except ImportError:
    ROSBAG_AVAILABLE = False

class TrajectoryLocalizationTester:
    """轨迹定位测试器"""
    
    def __init__(self, model_path, database_path):
        self.localizer = TrajectoryLocalizationSystem(num_locations=20)
        self.sc_generator = ScanContext()
        
        print(f"🎯 轨迹定位测试器")
        print(f"目标: 实时识别机器人在轨迹中的位置")
        
        # 加载位置数据库
        if not self.localizer.load_location_database(database_path):
            print("❌ 位置数据库加载失败")
            return
        
        # 加载训练好的模型
        if not self.localizer.load_trained_model(model_path):
            print("❌ 定位模型加载失败")
            return
        
        print("✅ 轨迹定位系统准备就绪")
        
        # 定位历史
        self.localization_history = []
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
    
    def test_rosbag_localization(self, bag_path, topic_name=None):
        """测试rosbag的轨迹定位"""
        
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
        
        print(f"\n🎯 开始轨迹定位测试")
        print(f"使用话题: {topic_name}")
        print(f"目标: 识别机器人在轨迹中的位置 (0-{self.localizer.num_locations-1})")
        print("-" * 60)
        
        # 处理消息
        total_messages = 0
        valid_localizations = 0
        start_time = time.time()
        
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            total_messages += 1
            timestamp = t.to_sec()
            
            # 转换点云
            points = self.pointcloud2_to_numpy(msg)
            if points is None:
                continue
            
            # 生成ScanContext
            sc_feature = self.sc_generator.generate_scan_context(points)
            if sc_feature is None:
                continue
            
            # 定位
            predicted_location, confidence = self.localizer.localize_position(sc_feature)
            
            if predicted_location is not None:
                valid_localizations += 1
                
                # 保存结果
                self.localization_history.append(predicted_location)
                self.confidence_history.append(confidence)
                self.timestamp_history.append(timestamp)
                
                # 计算期望位置（基于进度）
                expected_location = int((total_messages - 1) / (1769 / self.localizer.num_locations))
                expected_location = min(expected_location, self.localizer.num_locations - 1)
                
                # 计算定位误差
                location_error = abs(predicted_location - expected_location)
                
                # 实时输出
                status = "✅" if location_error <= 2 else "❌"
                print(f"消息 {total_messages:4d} | 预测位置: {predicted_location:2d} | "
                      f"期望位置: {expected_location:2d} | 误差: {location_error:2d} | "
                      f"置信度: {confidence:.4f} | 点数: {len(points):5d} {status}")
                
                # 每50个定位显示统计
                if valid_localizations % 50 == 0:
                    self.show_localization_stats()
            
            # 处理完整数据集
            if total_messages >= 1769:
                print(f"\n已处理完整数据集 ({total_messages} 个消息)")
                break
        
        bag.close()
        
        elapsed_time = time.time() - start_time
        print(f"\n" + "="*60)
        print("轨迹定位测试完成")
        print("="*60)
        print(f"总消息数: {total_messages}")
        print(f"有效定位数: {valid_localizations}")
        print(f"成功率: {valid_localizations/total_messages*100:.1f}%")
        print(f"处理时间: {elapsed_time:.1f}秒")
        print(f"定位频率: {valid_localizations/elapsed_time:.2f} Hz")
        
        return self.analyze_localization_results()
    
    def show_localization_stats(self):
        """显示定位统计"""
        if len(self.localization_history) == 0:
            return
        
        locations = np.array(self.localization_history)
        confidences = np.array(self.confidence_history)
        
        print(f"\n--- 轨迹定位统计 (最近{len(locations)}个定位) ---")
        print(f"位置范围: {np.min(locations)} - {np.max(locations)}")
        print(f"平均置信度: {np.mean(confidences):.4f}")
        print(f"最高置信度: {np.max(confidences):.4f}")
        
        # 最近定位的分布
        recent_locations = locations[-20:] if len(locations) > 20 else locations
        unique, counts = np.unique(recent_locations, return_counts=True)
        print("最近20个定位的位置分布:")
        for loc, count in zip(unique, counts):
            print(f"  位置 {loc}: {count} 次")
        print("-" * 50)
    
    def analyze_localization_results(self):
        """分析定位结果"""
        if len(self.localization_history) == 0:
            return None
        
        locations = np.array(self.localization_history)
        confidences = np.array(self.confidence_history)
        timestamps = np.array(self.timestamp_history)
        
        print(f"\n📊 轨迹定位结果分析")
        print(f"{'='*50}")
        
        # 基本统计
        print(f"定位位置数量: {len(np.unique(locations))}")
        print(f"位置范围: {np.min(locations)} - {np.max(locations)}")
        print(f"平均置信度: {np.mean(confidences):.4f}")
        print(f"置信度标准差: {np.std(confidences):.4f}")
        
        # 位置分布
        print(f"\n位置分布:")
        unique, counts = np.unique(locations, return_counts=True)
        for loc, count in zip(unique, counts):
            percentage = count / len(locations) * 100
            avg_conf = np.mean(confidences[locations == loc])
            print(f"  位置 {loc:2d}: {count:3d} 次 ({percentage:5.1f}%) | 平均置信度: {avg_conf:.4f}")
        
        # 时序分析
        print(f"\n时序分析:")
        if len(timestamps) > 1:
            time_duration = timestamps[-1] - timestamps[0]
            print(f"数据时长: {time_duration:.1f} 秒")
            print(f"定位频率: {len(locations)/time_duration:.2f} Hz")
        
        # 位置变化分析
        print(f"\n位置变化分析:")
        location_changes = []
        for i in range(1, len(locations)):
            if locations[i] != locations[i-1]:
                location_changes.append((i, locations[i-1], locations[i]))
        
        print(f"位置变化次数: {len(location_changes)}")
        if len(location_changes) > 0:
            print("主要位置变化:")
            for i, (pos, from_loc, to_loc) in enumerate(location_changes[:10]):
                print(f"  位置 {pos}: {from_loc} → {to_loc}")
        
        # 计算定位准确性
        expected_locations = np.linspace(0, self.localizer.num_locations-1, len(locations)).astype(int)
        location_errors = np.abs(locations - expected_locations)
        
        accuracy_1 = np.mean(location_errors <= 1) * 100  # 误差≤1的准确率
        accuracy_2 = np.mean(location_errors <= 2) * 100  # 误差≤2的准确率
        accuracy_3 = np.mean(location_errors <= 3) * 100  # 误差≤3的准确率
        
        print(f"\n定位准确性分析:")
        print(f"误差≤1位置的准确率: {accuracy_1:.1f}%")
        print(f"误差≤2位置的准确率: {accuracy_2:.1f}%")
        print(f"误差≤3位置的准确率: {accuracy_3:.1f}%")
        print(f"平均位置误差: {np.mean(location_errors):.2f}")
        
        # 可视化结果
        self.visualize_localization_results(locations, confidences, expected_locations)
        
        return {
            'locations': locations.tolist(),
            'confidences': confidences.tolist(),
            'timestamps': timestamps.tolist(),
            'accuracy_1': accuracy_1,
            'accuracy_2': accuracy_2,
            'accuracy_3': accuracy_3,
            'location_changes': len(location_changes),
            'mean_error': np.mean(location_errors)
        }
    
    def visualize_localization_results(self, locations, confidences, expected_locations):
        """可视化定位结果"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # 1. 位置预测vs期望
        axes[0].plot(expected_locations, 'b-', alpha=0.7, label='期望位置')
        axes[0].plot(locations, 'r-', alpha=0.7, label='预测位置')
        axes[0].set_ylabel('位置ID')
        axes[0].set_title('轨迹定位结果：预测位置 vs 期望位置')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 定位置信度
        axes[1].plot(confidences, 'g-', alpha=0.7, label='定位置信度')
        axes[1].set_ylabel('置信度')
        axes[1].set_title('定位置信度变化')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 位置误差
        location_errors = np.abs(locations - expected_locations)
        axes[2].plot(location_errors, 'orange', alpha=0.7, label='位置误差')
        axes[2].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='误差=1')
        axes[2].axhline(y=2, color='red', linestyle='--', alpha=0.5, label='误差=2')
        axes[2].set_ylabel('位置误差')
        axes[2].set_xlabel('帧索引')
        axes[2].set_title('定位误差分析')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('trajectory_localization_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 定位结果图表已保存为 trajectory_localization_results.png")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python test_trajectory_localization.py <bag_path> [topic_name]")
        return
    
    bag_path = sys.argv[1]
    topic_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("="*60)
    print("🎯 轨迹定位系统测试")
    print("="*60)
    print(f"目标: 识别机器人在轨迹中的具体位置")
    print(f"rosbag: {bag_path}")
    
    # 模型和数据库路径
    model_path = "models/saved/trajectory_localizer_simple2dcnn_acc*.pth"
    database_path = "location_database.pkl"
    
    # 查找最新的模型文件
    import glob
    model_files = glob.glob(model_path)
    if not model_files:
        print(f"❌ 未找到训练好的模型文件")
        print(f"请先运行 trajectory_localization_system.py 训练模型")
        return
    
    model_path = sorted(model_files)[-1]  # 使用最新的模型
    print(f"使用模型: {model_path}")
    
    # 创建测试器
    tester = TrajectoryLocalizationTester(model_path, database_path)
    
    # 测试定位
    results = tester.test_rosbag_localization(bag_path, topic_name)
    
    if results:
        print(f"\n✅ 轨迹定位测试完成！")
        print(f"误差≤1位置准确率: {results['accuracy_1']:.1f}%")
        print(f"误差≤2位置准确率: {results['accuracy_2']:.1f}%")
        print(f"平均位置误差: {results['mean_error']:.2f}")
        print(f"位置变化次数: {results['location_changes']}")
        
        # 保存结果
        import pickle
        result_path = 'trajectory_localization_test_results.pkl'
        with open(result_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"结果已保存到: {result_path}")
    else:
        print("❌ 轨迹定位测试失败")

if __name__ == '__main__':
    main()
