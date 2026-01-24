#!/usr/bin/env python3
"""
ROS节点：实时位置定位
订阅点云话题，实时输出当前位置索引
"""
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32, Float32, String
from geometry_msgs.msg import PoseStamped
import sensor_msgs.point_cloud2 as pc2
import json
from pathlib import Path
import threading
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from localize_position import PositionLocalizer

class ROSLocalizationNode:
    """ROS位置定位节点"""
    
    def __init__(self):
        """初始化ROS节点"""
        rospy.init_node('position_localizer', anonymous=True)
        
        # 获取参数
        self.model_path = rospy.get_param('~model_path', '')
        self.map_database_path = rospy.get_param('~map_database_path', '')
        self.device = rospy.get_param('~device', 'cpu')
        self.pointcloud_topic = rospy.get_param('~pointcloud_topic', '/velodyne_points')
        self.localization_rate = rospy.get_param('~localization_rate', 1.0)  # Hz
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.6)
        
        # 验证参数
        if not self.model_path or not Path(self.model_path).exists():
            rospy.logerr(f"模型文件不存在: {self.model_path}")
            return
        
        if not self.map_database_path or not Path(self.map_database_path).exists():
            rospy.logerr(f"地图数据库不存在: {self.map_database_path}")
            return
        
        # 初始化定位器
        rospy.loginfo("初始化位置定位器...")
        try:
            self.localizer = PositionLocalizer(
                self.model_path, 
                self.map_database_path, 
                self.device
            )
            rospy.loginfo("位置定位器初始化成功")
        except Exception as e:
            rospy.logerr(f"位置定位器初始化失败: {e}")
            return
        
        # 状态变量
        self.latest_pointcloud = None
        self.pointcloud_lock = threading.Lock()
        self.last_localization_time = 0
        self.current_position = -1
        self.current_confidence = 0.0
        
        # 订阅者
        self.pointcloud_sub = rospy.Subscriber(
            self.pointcloud_topic, 
            PointCloud2, 
            self.pointcloud_callback,
            queue_size=1
        )
        
        # 发布者
        self.position_pub = rospy.Publisher(
            '~current_position', 
            Int32, 
            queue_size=1
        )
        
        self.confidence_pub = rospy.Publisher(
            '~localization_confidence', 
            Float32, 
            queue_size=1
        )
        
        self.status_pub = rospy.Publisher(
            '~localization_status', 
            String, 
            queue_size=1
        )
        
        self.pose_pub = rospy.Publisher(
            '~estimated_pose', 
            PoseStamped, 
            queue_size=1
        )
        
        # 定时器
        self.localization_timer = rospy.Timer(
            rospy.Duration(1.0 / self.localization_rate),
            self.localization_callback
        )
        
        rospy.loginfo(f"ROS位置定位节点启动成功")
        rospy.loginfo(f"  - 点云话题: {self.pointcloud_topic}")
        rospy.loginfo(f"  - 定位频率: {self.localization_rate} Hz")
        rospy.loginfo(f"  - 置信度阈值: {self.confidence_threshold}")
        rospy.loginfo(f"  - 地图位置数量: {len(self.localizer.map_database)}")
    
    def pointcloud_callback(self, msg):
        """点云回调函数"""
        with self.pointcloud_lock:
            self.latest_pointcloud = msg
    
    def pointcloud_to_numpy(self, pointcloud_msg):
        """将ROS PointCloud2消息转换为numpy数组"""
        try:
            # 提取点云数据
            points_list = []
            for point in pc2.read_points(pointcloud_msg, skip_nans=True, field_names=("x", "y", "z")):
                points_list.append([point[0], point[1], point[2]])
            
            if len(points_list) == 0:
                return None
            
            points = np.array(points_list, dtype=np.float32)
            return points
            
        except Exception as e:
            rospy.logwarn(f"点云转换失败: {e}")
            return None
    
    def localization_callback(self, event):
        """定位回调函数"""
        current_time = time.time()
        
        # 检查是否有新的点云数据
        with self.pointcloud_lock:
            if self.latest_pointcloud is None:
                return
            
            pointcloud_msg = self.latest_pointcloud
        
        try:
            # 转换点云数据
            points = self.pointcloud_to_numpy(pointcloud_msg)
            if points is None or len(points) < 100:  # 至少需要100个点
                rospy.logwarn("点云数据不足，跳过定位")
                return
            
            # 执行定位
            start_time = time.time()
            result = self.localizer.localize_from_points(points, top_k=3)
            processing_time = time.time() - start_time
            
            # 更新状态
            self.current_position = result['best_position_index']
            self.current_confidence = result['best_similarity']
            
            # 发布结果
            self.publish_results(result, pointcloud_msg.header, processing_time)
            
            # 记录日志
            if result['best_similarity'] > self.confidence_threshold:
                rospy.loginfo(f"定位成功: 位置 {self.current_position}, "
                             f"置信度 {self.current_confidence:.3f}, "
                             f"耗时 {processing_time:.3f}s")
            else:
                rospy.logwarn(f"定位置信度低: 位置 {self.current_position}, "
                             f"置信度 {self.current_confidence:.3f}")
            
        except Exception as e:
            rospy.logerr(f"定位处理失败: {e}")
    
    def publish_results(self, result, header, processing_time):
        """发布定位结果"""
        # 发布位置索引
        position_msg = Int32()
        position_msg.data = result['best_position_index']
        self.position_pub.publish(position_msg)
        
        # 发布置信度
        confidence_msg = Float32()
        confidence_msg.data = result['best_similarity']
        self.confidence_pub.publish(confidence_msg)
        
        # 发布状态信息
        status_data = {
            'position_index': result['best_position_index'],
            'confidence': result['best_similarity'],
            'confidence_level': result['confidence'],
            'processing_time': processing_time,
            'map_file': result['best_map_file'],
            'top_candidates': result['top_k_candidates'][:3]
        }
        
        status_msg = String()
        status_msg.data = json.dumps(status_data)
        self.status_pub.publish(status_msg)
        
        # 发布估计位姿（简化版，只包含位置索引信息）
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = "map"
        
        # 这里可以根据位置索引设置实际的位姿信息
        # 目前只是示例，实际使用时需要根据地图信息设置
        pose_msg.pose.position.x = float(result['best_position_index'])
        pose_msg.pose.position.y = 0.0
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        
        self.pose_pub.publish(pose_msg)
    
    def get_current_position(self):
        """获取当前位置索引"""
        return self.current_position
    
    def get_current_confidence(self):
        """获取当前置信度"""
        return self.current_confidence
    
    def is_localized(self):
        """检查是否成功定位"""
        return self.current_confidence > self.confidence_threshold
    
    def run(self):
        """运行节点"""
        rospy.loginfo("位置定位节点开始运行...")
        
        # 发布节点状态
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            # 这里可以添加额外的状态检查和发布
            rate.sleep()

def main():
    try:
        node = ROSLocalizationNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("位置定位节点关闭")
    except Exception as e:
        rospy.logerr(f"节点运行失败: {e}")

if __name__ == "__main__":
    main()
