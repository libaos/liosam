# 安装与构建指南

## 准备工作

确保您的系统已经安装了以下依赖:

- ROS (Melodic/Noetic)
- OpenCV (>=4.2推荐)
- PCL
- CUDA (可选，用于GPU加速)

## 构建步骤

1. 将项目放置在ROS工作空间的src目录下:

```bash
cd ~/catkin_ws/src  # 或您的工作空间路径
git clone https://your-repo-url/lidar_vision_detection.git
```

2. 下载YOLO模型文件:

```bash
cd ~/catkin_ws/src/lidar_vision_detection/models
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
```

3. 返回工作空间根目录并编译:

```bash
cd ~/catkin_ws
catkin_make  # 或使用 catkin build
source devel/setup.bash
```

## 与LIO-SAM集成

本项目设计为可以与LIO-SAM Move Base Tutorial项目无缝集成。

1. 确保lio_sam_move_base_tutorial项目已经在工作空间中构建:

```bash
ls ~/catkin_ws/src/lio_sam_move_base_tutorial  # 检查项目是否存在
```

2. 启动集成演示:

```bash
roslaunch lidar_vision_detection integration_with_liosam.launch
```

## 常见问题

### OpenCV版本问题

如果编译时遇到OpenCV相关错误，请检查您的OpenCV版本：

```bash
pkg-config --modversion opencv4  # 或 opencv
```

如果版本低于4.2，建议升级OpenCV或修改代码以适配较低版本。

### 找不到YOLO模型文件

确保模型文件已正确下载并放置在`models`目录中，且launch文件中的路径设置正确。

### 性能问题

如果检测速度较慢，可以考虑以下优化:

1. 使用较小的YOLO模型如YOLOv3-tiny
2. 减小输入图像尺寸
3. 如果硬件支持，启用CUDA加速 