#!/bin/bash

# 启动Gazebo仿真
echo "启动Gazebo仿真..."
roslaunch warehouse_simulation warehouse_simulation.launch &
sleep 5

# 使用LIORF进行建图
echo "启动LIORF进行建图..."
roslaunch liorf run_lio_sam_default.launch &
sleep 5

# 等待用户输入，保存地图
read -p "请在确认地图构建完成后按回车键继续，将保存点云地图..." 
rosservice call /liorf/save_map 0.2 "/LIO-SAM/map"

# 使用LIORL进行重定位
echo "启动LIORL进行重定位..."
roslaunch liorl run_liorl.launch &
sleep 5

# 将点云地图转换为2D地图
echo "将点云地图转换为2D地图..."
roslaunch pcd2pgm run.launch &
sleep 3

# 运行move_base
echo "启动move_base与psolqr_local_planner..."
roslaunch move_base_benchmark move_base_benchmark.launch

echo "导航系统已全部启动，请在RViz中设置初始位姿估计和导航目标。" 