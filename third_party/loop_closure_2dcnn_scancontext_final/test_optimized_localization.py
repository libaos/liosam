#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试优化的轨迹定位系统
"""

import numpy as np
import torch
from trajectory_localization_system import TrajectoryLocalizationSystem
from utils.scan_context import ScanContext
import time
import matplotlib.pyplot as plt
import pickle

# 尝试导入rosbag相关库
try:
    import rosbag
    import sensor_msgs.point_cloud2 as pc2
    ROSBAG_AVAILABLE = True
except ImportError:
    ROSBAG_AVAILABLE = False

class OptimizedLocalizationTester:
    """优化的轨迹定位测试器"""
    
    def __init__(self, model_path, database_path):
        self.localizer = TrajectoryLocalizationSystem(
            num_locations=20,
            adaptive_segments=True  # 启用自适应分段
        )
        self.sc_generator = ScanContext()
        
        print(f"🎯 优化的轨迹定位测试器")
        print(f"目标: 测试优化后的实时定位性能")
        
        # 加载位置数据库
        if not self.localizer.load_location_database(database_path):
            print("❌ 位置数据库加载失败")
            return
        
        # 加载训练好的模型
        if not self.localizer.load_trained_model(model_path):
            print("❌ 定位模型加载失败")
            return
        
        print("✅ 优化的轨迹定位系统准备就绪")
        
        # 定位历史和统计
        self.localization_history = []
        self.confidence_history = []
        self.timestamp_history = []
        self.processing_times = []
        
        # 性能统计
        self.high_confidence_count = 0
        self.low_confidence_count = 0
        self.temporal_smoothing_count = 0
        
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
    
    def test_optimized_localization(self, bag_path, topic_name=None):
        """测试优化的轨迹定位"""
        
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
        
        print(f"\n🎯 开始优化轨迹定位测试")
        print(f"使用话题: {topic_name}")
        print(f"优化特性: 自适应分段 + 时序平滑 + 置信度过滤")
        print("-" * 60)
        
        # 处理消息
        total_messages = 0
        valid_localizations = 0
        start_time = time.time()
        
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            total_messages += 1
            timestamp = t.to_sec()
            
            # 记录处理开始时间
            process_start = time.time()
            
            # 转换点云
            points = self.pointcloud2_to_numpy(msg)
            if points is None:
                continue
            
            # 生成ScanContext
            sc_feature = self.sc_generator.generate_scan_context(points)
            if sc_feature is None:
                continue
            
            # 优化的定位
            predicted_location, confidence = self.localizer.localize_position(sc_feature)
            
            # 记录处理时间
            process_time = time.time() - process_start
            self.processing_times.append(process_time)
            
            if predicted_location is not None:
                valid_localizations += 1
                
                # 保存结果
                self.localization_history.append(predicted_location)
                self.confidence_history.append(confidence)
                self.timestamp_history.append(timestamp)
                
                # 统计置信度
                if confidence >= self.localizer.confidence_threshold:
                    self.high_confidence_count += 1
                else:
                    self.low_confidence_count += 1
                
                # 计算期望位置（基于进度）
                progress = total_messages / 2132  # 假设总长度
                expected_location = int(progress * (self.localizer.num_locations - 1))
                
                # 计算定位误差
                location_error = abs(predicted_location - expected_location)
                
                # 实时输出（每50个显示一次）
                if valid_localizations % 50 == 0:
                    status = "✅" if location_error <= 2 else "❌"
                    avg_process_time = np.mean(self.processing_times[-50:]) * 1000
                    print(f"消息 {total_messages:4d} | 预测: {predicted_location:2d} | "
                          f"期望: {expected_location:2d} | 误差: {location_error:2d} | "
                          f"置信度: {confidence:.3f} | 处理: {avg_process_time:.1f}ms {status}")
                    
                    # 显示优化统计
                    if valid_localizations % 200 == 0:
                        self.show_optimization_stats()
            
            # 处理完整数据集
            if total_messages >= 2132:
                print(f"\n已处理完整数据集 ({total_messages} 个消息)")
                break
        
        bag.close()
        
        elapsed_time = time.time() - start_time
        print(f"\n" + "="*60)
        print("优化轨迹定位测试完成")
        print("="*60)
        print(f"总消息数: {total_messages}")
        print(f"有效定位数: {valid_localizations}")
        print(f"成功率: {valid_localizations/total_messages*100:.1f}%")
        print(f"处理时间: {elapsed_time:.1f}秒")
        print(f"定位频率: {valid_localizations/elapsed_time:.2f} Hz")
        print(f"平均处理时间: {np.mean(self.processing_times)*1000:.2f}ms")
        
        return self.analyze_optimized_results()
    
    def show_optimization_stats(self):
        """显示优化统计信息"""
        total_localizations = len(self.localization_history)
        if total_localizations == 0:
            return
        
        high_conf_rate = self.high_confidence_count / total_localizations * 100
        low_conf_rate = self.low_confidence_count / total_localizations * 100
        
        print(f"\n--- 优化性能统计 ---")
        print(f"高置信度定位: {self.high_confidence_count} ({high_conf_rate:.1f}%)")
        print(f"低置信度定位: {self.low_confidence_count} ({low_conf_rate:.1f}%)")
        print(f"平均置信度: {np.mean(self.confidence_history):.3f}")
        print(f"置信度标准差: {np.std(self.confidence_history):.3f}")
        print("-" * 30)
    
    def analyze_optimized_results(self):
        """分析优化后的定位结果"""
        if len(self.localization_history) == 0:
            return None
        
        locations = np.array(self.localization_history)
        confidences = np.array(self.confidence_history)
        timestamps = np.array(self.timestamp_history)
        
        print(f"\n📊 优化轨迹定位结果分析")
        print(f"{'='*50}")
        
        # 基本统计
        print(f"定位位置数量: {len(np.unique(locations))}")
        print(f"位置范围: {np.min(locations)} - {np.max(locations)}")
        print(f"平均置信度: {np.mean(confidences):.4f}")
        print(f"置信度标准差: {np.std(confidences):.4f}")
        
        # 优化效果分析
        high_conf_rate = self.high_confidence_count / len(locations) * 100
        print(f"\n🎯 优化效果:")
        print(f"高置信度定位率: {high_conf_rate:.1f}%")
        print(f"平均处理时间: {np.mean(self.processing_times)*1000:.2f}ms")
        print(f"处理时间标准差: {np.std(self.processing_times)*1000:.2f}ms")
        
        # 计算定位准确性
        expected_locations = np.linspace(0, self.localizer.num_locations-1, len(locations)).astype(int)
        location_errors = np.abs(locations - expected_locations)
        
        accuracy_1 = np.mean(location_errors <= 1) * 100
        accuracy_2 = np.mean(location_errors <= 2) * 100
        accuracy_3 = np.mean(location_errors <= 3) * 100
        
        print(f"\n📈 定位准确性:")
        print(f"误差≤1位置准确率: {accuracy_1:.1f}%")
        print(f"误差≤2位置准确率: {accuracy_2:.1f}%")
        print(f"误差≤3位置准确率: {accuracy_3:.1f}%")
        print(f"平均位置误差: {np.mean(location_errors):.2f}")
        
        # 时序稳定性分析
        location_changes = np.sum(np.diff(locations) != 0)
        stability_score = 1 - (location_changes / len(locations))
        
        print(f"\n🔄 时序稳定性:")
        print(f"位置变化次数: {location_changes}")
        print(f"稳定性评分: {stability_score:.3f}")
        
        # 可视化结果
        self.visualize_optimized_results(locations, confidences, expected_locations)
        
        return {
            'locations': locations.tolist(),
            'confidences': confidences.tolist(),
            'timestamps': timestamps.tolist(),
            'accuracy_1': accuracy_1,
            'accuracy_2': accuracy_2,
            'accuracy_3': accuracy_3,
            'mean_error': np.mean(location_errors),
            'high_confidence_rate': high_conf_rate,
            'stability_score': stability_score,
            'avg_processing_time': np.mean(self.processing_times),
            'location_changes': location_changes
        }
    
    def visualize_optimized_results(self, locations, confidences, expected_locations):
        """可视化优化后的定位结果"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # 1. 位置预测vs期望
        axes[0].plot(expected_locations, 'b-', alpha=0.7, label='期望位置', linewidth=2)
        axes[0].plot(locations, 'r-', alpha=0.8, label='预测位置', linewidth=1)
        axes[0].set_ylabel('位置ID')
        axes[0].set_title('优化轨迹定位结果：预测位置 vs 期望位置')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 定位置信度
        axes[1].plot(confidences, 'g-', alpha=0.7, label='定位置信度')
        axes[1].axhline(y=self.localizer.confidence_threshold, color='red', 
                       linestyle='--', alpha=0.7, label=f'置信度阈值 ({self.localizer.confidence_threshold})')
        axes[1].set_ylabel('置信度')
        axes[1].set_title('定位置信度变化（优化后）')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 位置误差
        location_errors = np.abs(locations - expected_locations)
        axes[2].plot(location_errors, 'orange', alpha=0.7, label='位置误差')
        axes[2].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='误差=1')
        axes[2].axhline(y=2, color='red', linestyle='--', alpha=0.5, label='误差=2')
        axes[2].set_ylabel('位置误差')
        axes[2].set_title('定位误差分析（优化后）')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. 处理时间
        processing_times_ms = np.array(self.processing_times) * 1000
        axes[3].plot(processing_times_ms, 'purple', alpha=0.7, label='处理时间')
        axes[3].axhline(y=np.mean(processing_times_ms), color='red', 
                       linestyle='--', alpha=0.7, label=f'平均时间 ({np.mean(processing_times_ms):.1f}ms)')
        axes[3].set_ylabel('处理时间 (ms)')
        axes[3].set_xlabel('帧索引')
        axes[3].set_title('实时处理性能')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimized_trajectory_localization_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 优化定位结果图表已保存为 optimized_trajectory_localization_results.png")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python test_optimized_localization.py <bag_path> [topic_name]")
        return
    
    bag_path = sys.argv[1]
    topic_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("="*60)
    print("🎯 优化轨迹定位系统测试")
    print("="*60)
    print(f"目标: 测试优化后的定位性能")
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
    
    model_path = sorted(model_files)[-1]
    print(f"使用模型: {model_path}")
    
    # 创建优化测试器
    tester = OptimizedLocalizationTester(model_path, database_path)
    
    # 测试优化定位
    results = tester.test_optimized_localization(bag_path, topic_name)
    
    if results:
        print(f"\n✅ 优化轨迹定位测试完成！")
        print(f"误差≤1位置准确率: {results['accuracy_1']:.1f}%")
        print(f"误差≤2位置准确率: {results['accuracy_2']:.1f}%")
        print(f"高置信度定位率: {results['high_confidence_rate']:.1f}%")
        print(f"时序稳定性评分: {results['stability_score']:.3f}")
        print(f"平均处理时间: {results['avg_processing_time']*1000:.2f}ms")
        
        # 保存结果
        result_path = 'optimized_trajectory_localization_test_results.pkl'
        with open(result_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"结果已保存到: {result_path}")
    else:
        print("❌ 优化轨迹定位测试失败")

if __name__ == '__main__':
    main()
