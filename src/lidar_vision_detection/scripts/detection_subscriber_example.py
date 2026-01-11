#!/usr/bin/env python3

import rospy
from lidar_vision_detection.msg import DetectedObjectArray, DetectedObject
from std_msgs.msg import Bool
import math

class DetectionSubscriber:
    def __init__(self):
        # 初始化节点
        rospy.init_node('detection_subscriber_example', anonymous=True)
        
        # 获取参数
        self.min_confidence = rospy.get_param('~min_confidence', 0.5)
        target_classes_str = rospy.get_param('~target_classes', 'person,car')
        self.target_classes = target_classes_str.split(',')
        
        rospy.loginfo(f"监测目标类别: {self.target_classes}")
        rospy.loginfo(f"最低置信度: {self.min_confidence}")
        
        # 创建订阅者
        self.detection_sub = rospy.Subscriber('/vision_detection_node/detections', 
                                            DetectedObjectArray, 
                                            self.detection_callback)
        
        # 创建发布者 - 用于与导航系统交互
        self.obstacle_detected_pub = rospy.Publisher('/obstacle_detected', Bool, queue_size=1)
        
        rospy.loginfo("检测订阅器初始化完成")
        
    def detection_callback(self, msg):
        """处理检测结果"""
        target_detections = []
        
        # 筛选符合条件的检测结果
        for obj in msg.objects:
            if obj.label in self.target_classes and obj.score >= self.min_confidence:
                target_detections.append(obj)
                
                # 计算目标距离（如果有3D位置信息）
                dist = math.sqrt(obj.pose.position.x**2 + 
                                obj.pose.position.y**2 + 
                                obj.pose.position.z**2)
                
                rospy.loginfo(f"检测到 {obj.label}，置信度: {obj.score:.2f}, 距离: {dist:.2f}m")
        
        # 发布是否检测到障碍物（可与导航系统集成）
        if target_detections:
            self.obstacle_detected_pub.publish(Bool(True))
            
            # 这里可以添加与导航系统的集成逻辑
            # 例如，发送导航目标、触发避障行为等
        else:
            self.obstacle_detected_pub.publish(Bool(False))

def main():
    detector = DetectionSubscriber()
    
    try:
        # 保持节点运行，直到手动中断
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("节点被手动中断")

if __name__ == '__main__':
    main() 