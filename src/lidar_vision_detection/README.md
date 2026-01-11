# Lidar Vision Detection

这个包实现了一个基于YOLO的视觉检测系统，可与LIO-SAM导航系统无缝集成。该模块可以检测摄像头中的物体，并通过ROS话题发布检测结果。

## 功能特点

- 基于OpenCV DNN模块实现YOLO目标检测
- 支持2D图像检测和3D点云融合
- 可视化检测结果，在RViz中显示
- 模块化设计，可以方便地与LIO-SAM导航系统集成

## 依赖

- ROS (Noetic/Melodic)
- OpenCV (>=4.2 推荐)
- PCL
- cv_bridge
- roscpp
- image_transport

## 使用方法

### 安装

1. 将此包克隆到您的工作空间的`src`目录
2. 运行`catkin_make`或`catkin build`来编译

### 下载YOLO模型

您需要下载YOLOv3或YOLOv4的模型文件：

```bash
mkdir -p ~/w/lio_ws/src/lidar_vision_detection/models
cd ~/w/lio_ws/src/lidar_vision_detection/models
# 下载YOLOv3配置和权重
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg
wget https://pjreddie.com/media/files/yolov3.weights
# 下载COCO类名文件
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
```

### 独立运行

单独启动视觉检测模块：

```bash
roslaunch lidar_vision_detection vision_detection.launch
```

### 与LIO-SAM集成

将视觉检测模块与LIO-SAM集成运行：

```bash
roslaunch lidar_vision_detection integration_with_liosam.launch
```

## 参数

可通过launch文件配置以下参数：

- `image_topic` - 输入图像话题，默认: `/camera/image_raw`
- `points_topic` - 输入点云话题，默认: `/velodyne_points`
- `model_path` - YOLO权重文件路径
- `config_path` - YOLO配置文件路径
- `class_names_file` - 类别名称文件路径
- `confidence_threshold` - 置信度阈值，默认: 0.5
- `nms_threshold` - 非极大值抑制阈值，默认: 0.4

## 主要话题

- 订阅:
  - `image` - 输入图像
  - `points` - 输入点云

- 发布:
  - `detections` - 检测结果 (`DetectedObjectArray` 类型)
  - `detection_markers` - 可视化标记 (`visualization_msgs/MarkerArray` 类型) 