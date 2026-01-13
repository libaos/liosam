# Packages Overview (src/)

本仓库是 ROS1 catkin 工作区：所有 catkin package 都在 `src/` 下（可多层嵌套），用 `catkin_make`/`catkin build` 统一编译。

## Entry points (推荐从这里开始)

- 仿真一键入口（无 GUI，适合远程/CI）：`roslaunch liosam_bringup orchard_sim_headless.launch`
- Gazebo 果园 world（可开 GUI）：`roslaunch pcd_gazebo_world orchard_sim.launch gui:=true`
- 果园行先验/分割：`roslaunch orchard_row_mapping orchard_row_mapping.launch`

## Core / 算法包

- `liorf`, `liorl`: LiDAR/IMU/建图相关（robot_gazebo 下的两套实现/配置）
- `orchard_row_mapping`: 果树点云分割 + 行拟合 + 先验生成
- `orchard_tree_tracker`: 果树检测/跟踪（含基础单测）
- `orchard_scancontext_fsm`: ScanContext + 状态机/模式切换
- `orchard_teb_mode_switcher`: TEB 模式切换（dynamic_reconfigure）
- `lidar_vision_detection`: LiDAR + 视觉检测融合（自定义 msg）
- `psolqr_local_planner`: PSO + LQR 局部规划器插件
- `teb_local_planner`: TEB（第三方）
- `far_planner/*`: Far planner 相关（含 RViz plugin、msgs）
- `slope_costmap_layer`: 代价地图 slope layer（依赖 `grid_map_*`，未安装时会跳过编译）

## 驱动 / 消息 / 工具包

- `bag_route_replay`: 从 bag 读取路径并驱动 move_base 的小工具
- `continuous_navigation`: waypoint/两点导航工具节点
- `pcd2pgm`: 点云→2D 栅格地图
- `visibility_graph_msg`: Far planner 的消息定义包
- `move_base_benchmark`: move_base/规划器 benchmark（local-planning-benchmark）

## 教程 / 仿真 / Demo 包

- `liosam_bringup`: 仓库级 bringup/launch 聚合入口（本次整理新增）
- `pcd_gazebo_world`: PCD→Gazebo 地形/果园 world 生成与回放
- `gazebo_world`: local-planning-benchmark 的 Gazebo world
- `scout_gazebo`, `scout_slam_nav`: 机器人仿真与导航示例
- `warehouse_simulation`: 仿真工具包
- `velodyne_description`, `velodyne_gazebo_plugins`, `velodyne_simulator`: Velodyne 仿真相关

## 非 catkin 内容（仓库级 tools/）

这些不参与 catkin 编译，但对实验/复现实验有用：

- `tools/start_navigation.sh`: 一键串起多个 launch 的脚本（历史遗留，供参考）
- `tools/planner_comparison/`: PSOLQR vs TEB 的坡道对比分析脚本与报告
- `tools/`: 其它仓库级脚本（bag 回放、评测等）
