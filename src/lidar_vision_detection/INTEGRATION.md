# 与LIO-SAM的模块化集成方案

本文档描述了如何将`lidar_vision_detection`模块与`lio_sam_move_base_tutorial`项目集成，而不修改原始项目代码。

## 集成架构

系统采用模块化设计，通过ROS话题进行通信，实现松耦合集成：

![系统架构](./docs/architecture.png)

## 集成方式

### 1. 话题订阅方式

`lidar_vision_detection`模块将视觉检测结果发布到`/vision_detection_node/detections`话题，任何需要使用这些信息的节点都可以订阅此话题。

示例订阅代码 (Python):
```python
from lidar_vision_detection.msg import DetectedObjectArray

def detection_callback(msg):
    for obj in msg.objects:
        if obj.label == "person" and obj.score > 0.7:
            # 处理检测到的人
            print(f"检测到人，置信度：{obj.score}")

rospy.Subscriber("/vision_detection_node/detections", DetectedObjectArray, detection_callback)
```

示例订阅代码 (C++):
```cpp
#include "lidar_vision_detection/DetectedObjectArray.h"

void detectionCallback(const lidar_vision_detection::DetectedObjectArray::ConstPtr& msg)
{
  for (const auto& obj : msg->objects)
  {
    if (obj.label == "person" && obj.score > 0.7)
    {
      // 处理检测到的人
      ROS_INFO("检测到人，置信度：%.2f", obj.score);
    }
  }
}

ros::Subscriber sub = nh.subscribe("/vision_detection_node/detections", 1, detectionCallback);
```

### 2. 与导航系统集成

我们提供了一个简单的`/obstacle_detected`布尔话题，可以在探测到特定物体时通知导航系统：

```cpp
// 在move_base节点中订阅
void obstacleDetectedCallback(const std_msgs::Bool::ConstPtr& msg)
{
  if (msg->data)
  {
    // 检测到障碍物，可以触发特定的导航行为
    // 例如减速、停止或重新规划路径
  }
}
```

### 3. 参数配置与自定义

可通过以下方法自定义集成行为：

1. 修改launch文件中的话题映射，适配您的机器人配置
2. 调整检测参数，如置信度阈值、目标类别等
3. 创建自定义的检测结果处理节点

## 示例：对特定目标的响应

以下示例展示如何根据检测到的物体类型调整机器人行为：

```python
def detection_callback(msg):
    for obj in msg.objects:
        if obj.label == "person" and obj.score > 0.7:
            # 检测到人时减速
            publish_velocity_scale(0.5)
        elif obj.label == "car" and obj.score > 0.8:
            # 检测到汽车时停止
            publish_velocity_scale(0.0)
```

## 集成演示

运行集成演示：

```bash
roslaunch lidar_vision_detection integration_with_liosam.launch
```

这将启动视觉检测模块和一个演示接收器，展示集成工作流程。 