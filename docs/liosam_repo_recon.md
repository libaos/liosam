# liosam 仓库信息盘点（repo recon）

> 扫描范围：`/mysda/w/w/lio_ws`（以根目录 catkin 工作区 `src/` 为主；`build/`、`devel/` 等视为编译产物；仓库内另有 `lio_ws/`、`学习/` 等历史/资料目录，本文仅在“备注”中提示，不逐一展开）
>
> 生成方式：只读扫描（`package.xml` / `launch` / `yaml` / 关键节点脚本与源码），不做任何重构/修复。

## 0) “出处”标注规则

- 本文每条结论尽量附 **出处**：`文件路径` + “关键片段”（原文截取）。
- 若同一结论来自多处（例如 launch + yaml + 代码三者共同决定默认值），会列出多条出处并标明“默认值来源（launch/yaml/代码）”。

---

## 1) 仓库结构（2–3 层目录树）

### 1.1 结构概览（聚焦主要代码与入口）

```text
.
├── docs/                         # 仓库文档（FAQ/Packages/…）
├── src/                          # ROS1 catkin packages（主要代码）
│   ├── liosam_bringup/           # 仓库级统一入口（smoke / headless sim）
│   ├── pcd_gazebo_world/         # PCD→Gazebo 地形/果园 world
│   ├── orchard_row_mapping/      # 分割 + 行拟合 + 先验/树图工具
│   ├── orchard_tree_tracker/     # 果树实例化 + MOT 跟踪 + 拟合
│   ├── orchard_scancontext_fsm/  # ScanContext 模式识别（/fsm/mode）
│   ├── orchard_ch5_pipeline/     # Chapter 5 端到端 launch 编排
│   ├── orchard_teb_mode_switcher/# /fsm/mode -> 动态切 teb 参数
│   ├── lio_sam_move_base_tutorial/
│   │   ├── robot_gazebo/         # 仿真/导航/里程计相关包集合（含 liorf/liorl）
│   │   ├── teb_local_planner/    # TEB（第三方）
│   │   └── local-planning-benchmark/
│   └── ...（其它导航/规划/工具包）
├── tools/                        # 仓库级脚本/分析工具
├── build/ , devel/               # catkin_make 产物（非源码）
└── visualize_trajectory.launch   # 仓库根目录的单独 launch
```

**出处（仓库定位为 ROS1 catkin 工作区）**：`README.md`
```md
# liosam (ROS1 / Gazebo Classic workspace)
...
一个面向果园场景的 ROS1（catkin）工作区，包含：
...
## Repository Layout
...
├── src/                    # ROS1 catkin packages（主要代码都在这里）
```

**出处（catkin workspace 顶层 CMakeLists）**：`src/CMakeLists.txt`
```cmake
# toplevel CMakeLists.txt for a catkin workspace
...
set(CATKIN_TOPLEVEL TRUE)
```

### 1.2 备注：仓库内的“额外目录”

- 仓库根目录存在 `lio_ws/`（看起来是另一个工作区拷贝/子工作区）以及 `学习/` 等目录，包含大量其它工程/launch（不属于本文的主线入口）。
- 本文后续的 packages/launch 清单默认 **仅统计根目录 `src/` 下的 catkin 包与其源码 launch**，并单独列出根目录的 `visualize_trajectory.launch`。

---

## 2) `src/` 下 packages 清单 + 每包用途一句话

> 统计口径：`src/**/package.xml`

### 2.1 总览

- `src/` 下共发现 **30 个** ROS1 catkin 包（含多层嵌套包）。

**出处（包都在 src 下 + 可嵌套）**：`docs/PACKAGES.md`
```md
本仓库是 ROS1 catkin 工作区：所有 catkin package 都在 `src/` 下（可多层嵌套）
```

### 2.2 包清单（按路径）

> 说明：用途句子优先引用 `package.xml` 的 `<description>`；若 `<description>` 过于笼统，会在同一条里补充一句“从代码/launch 可见的定位”（并给出处）。

- `bag_route_replay`（`src/bag_route_replay`）：把 bag 里的 `nav_msgs/Path` 转成 `move_base` goals 依次发送的回放工具。  
  出处：`src/bag_route_replay/package.xml`
  ```xml
  <description>Replay a recorded nav_msgs/Path from a rosbag as move_base goals.</description>
  ```

- `continuous_navigation`（`src/continuous_navigation`）：按 waypoint 序列连续导航的小工具包。  
  出处：`src/continuous_navigation/package.xml`
  ```xml
  <description>A ROS package for continuous navigation through a sequence of waypoints</description>
  ```

- `far_planner` 相关（位于 `src/far_planner/src/*`，多包）：
  - `boundary_handler`：Boundary Graph Extractor。  
    出处：`src/far_planner/src/boundary_handler/package.xml`
    ```xml
    <description>Boundary Graph Extractor</description>
    ```
  - `far_planner`：FAR Planner。  
    出处：`src/far_planner/src/far_planner/package.xml`
    ```xml
    <description>FAR Planner</description>
    ```
  - `graph_decoder`：Visibility Graph Decoder。  
    出处：`src/far_planner/src/graph_decoder/package.xml`
    ```xml
    <description>Visibility Graph Decoder</description>
    ```
  - `visibility_graph_msg`：Visibility Graph Message（消息定义包）。  
    出处：`src/far_planner/src/visibility_graph_msg/package.xml`
    ```xml
    <description>Visibility Graph Message</description>
    ```
  - `goalpoint_rviz_plugin`：Route Planner Goalpoint RVIZ Plugin。  
    出处：`src/far_planner/src/goalpoint_rviz_plugin/package.xml`
    ```xml
    <description>Route Planner Goalpoint RVIZ Plugin</description>
    ```
  - `teleop_rviz_plugin`：Teleop RVIZ Plugin。  
    出处：`src/far_planner/src/teleop_rviz_plugin/package.xml`
    ```xml
    <description>
        Teleop RVIZ Plugin
    </description>
    ```

- `lidar_vision_detection`（`src/lidar_vision_detection`）：与 LIO-SAM 兼容的视觉检测包（含自定义 msg）。  
  出处：`src/lidar_vision_detection/package.xml`
  ```xml
  <description>A vision-based detection package compatible with LIO-SAM</description>
  ```

- `lio_sam_move_base_tutorial` 相关（位于 `src/lio_sam_move_base_tutorial/**`，多包）：
  - `liorf`（`src/lio_sam_move_base_tutorial/robot_gazebo/liorf`）：LiDAR/IMU 里程计/建图（LIO-SAM 派生实现，topic 命名 `liorf/*`）。  
    出处（包描述较笼统）：`src/lio_sam_move_base_tutorial/robot_gazebo/liorf/package.xml`
    ```xml
    <description>Lidar Odometry</description>
    ```
    出处（建图用法）：`src/lio_sam_move_base_tutorial/readme.md`
    ```md
    2使用liorf进行建图
    roslaunch liorf run_lio_sam_default.launch
    ```
  - `liorl`（`src/lio_sam_move_base_tutorial/robot_gazebo/liorl`）：LiDAR 定位/重定位（topic 命名 `liorl/*`，并把点云转 scan 供 move_base）。  
    出处（包描述较笼统）：`src/lio_sam_move_base_tutorial/robot_gazebo/liorl/package.xml`
    ```xml
    <description>Lidar Odometry</description>
    ```
    出处（重定位用法）：`src/lio_sam_move_base_tutorial/readme.md`
    ```md
    4使用liorl进行重定位，并将/liorl/deskew/cloud_deskewed话题实时转换为scan
    roslaunch liorl run_liorl.launch
    ```
  - `pcd2pgm`：点云地图转 2D 栅格 map 的工具包。  
    出处：`src/lio_sam_move_base_tutorial/robot_gazebo/pcd2pgm/package.xml`
    ```xml
    <description>The pcd2pgm package</description>
    ```
  - `scout_gazebo`：Scout 机器人 Gazebo 仿真包。  
    出处：`src/lio_sam_move_base_tutorial/robot_gazebo/scout_gazebo/package.xml`
    ```xml
    <description>The scout_gazebo package</description>
    ```
  - `scout_slam_nav`：Scout SLAM/Nav 示例启动集合。  
    出处：`src/lio_sam_move_base_tutorial/robot_gazebo/scout_slam_nav/package.xml`
    ```xml
    <description>The scout_slam_nav package</description>
    ```
  - `warehouse_simulation`（实际目录名 `warehouse_simulation_toolkit`）：仓库/室内仿真工具包。  
    出处：`src/lio_sam_move_base_tutorial/robot_gazebo/warehouse_simulation_toolkit/package.xml`
    ```xml
    <description>The warehouse_simulation package</description>
    ```
  - `teb_local_planner`：TEB 本地规划器插件（第三方）。  
    出处：`src/lio_sam_move_base_tutorial/teb_local_planner/package.xml`
    ```xml
    <description>The teb_local_planner package implements a plugin ... Timed Elastic Band ...</description>
    ```
  - `move_base_benchmark`：local-planning-benchmark 的 move_base benchmark 包。  
    出处：`src/lio_sam_move_base_tutorial/local-planning-benchmark/move_base_benchmark/package.xml`
    ```xml
    <description>The move_base_benchmark package.</description>
    ```
  - `gazebo_world`：local-planning-benchmark 的 Gazebo world 包。  
    出处：`src/lio_sam_move_base_tutorial/local-planning-benchmark/gazebo_world/package.xml`
    ```xml
    <description>The gazebo_world package</description>
    ```
  - `velodyne_description`：Velodyne 传感器 URDF/mesh 描述。  
    出处：`src/lio_sam_move_base_tutorial/robot_gazebo/velodyne_simulator/velodyne_description/package.xml`
    ```xml
    <description>URDF and meshes describing Velodyne laser scanners.</description>
    ```
  - `velodyne_gazebo_plugins`：Velodyne Gazebo 仿真插件。  
    出处：`src/lio_sam_move_base_tutorial/robot_gazebo/velodyne_simulator/velodyne_gazebo_plugins/package.xml`
    ```xml
    <description>Gazebo plugin to provide simulated data from Velodyne laser scanners.</description>
    ```
  - `velodyne_simulator`：Velodyne 仿真组件 metapackage。  
    出处：`src/lio_sam_move_base_tutorial/robot_gazebo/velodyne_simulator/velodyne_simulator/package.xml`
    ```xml
    <description>Metapackage allowing easy installation of Velodyne simulation components.</description>
    ```

- `liosam_bringup`（`src/liosam_bringup`）：仓库级统一入口 launch 聚合（smoke/headless sim）。  
  出处：`src/liosam_bringup/package.xml`
  ```xml
  <description>Workspace-level bringup / demo launch files for the liosam catkin workspace.</description>
  ```
  出处（定位说明）：`src/liosam_bringup/README.md`
  ```md
  本包只做一件事：提供工作区级别的统一入口 `launch/`
  ```

- `orchard_ch5_pipeline`（`src/orchard_ch5_pipeline`）：论文第 5 章“端到端”管线 launch 编排（分割/聚类/ScanContext/TEB 切换/可选 move_base）。  
  出处：`src/orchard_ch5_pipeline/package.xml`
  ```xml
  <description>Chapter 5 orchard pipeline launchers (segmentation, clustering, ScanContext FSM, TEB switching).</description>
  ```

- `orchard_row_mapping`（`src/orchard_row_mapping`）：点云分割（RandLA-Net 推理）+ 果树行拟合 + 先验/树图工具。  
  出处：`src/orchard_row_mapping/package.xml`
  ```xml
  <description>Point cloud segmentation and orchard row fitting node integrating RandLA-Net inference.</description>
  ```

- `orchard_scancontext_fsm`（`src/orchard_scancontext_fsm`）：ScanContext 的直行/左/右模式识别（发布 `/fsm/mode`）。  
  出处：`src/orchard_scancontext_fsm/package.xml`
  ```xml
  <description>ScanContext-based straight/left/right mode detection (FSM helper) for orchard navigation experiments.</description>
  ```

- `orchard_teb_mode_switcher`（`src/orchard_teb_mode_switcher`）：基于 `/fsm/mode` 动态切换 TebLocalPlannerROS 参数（dynamic_reconfigure）。  
  出处：`src/orchard_teb_mode_switcher/package.xml`
  ```xml
  <description>Switch TebLocalPlannerROS parameters based on /fsm/mode (straight/left/right) via dynamic_reconfigure.</description>
  ```

- `orchard_tree_tracker`（`src/orchard_tree_tracker`）：果树实例化 + MOT 跟踪 + 在线拟合（输入为带 label 的 PointCloud2）。  
  出处：`src/orchard_tree_tracker/package.xml`
  ```xml
  <description>ROS1 (rospy) fruit-tree instancing + MOT tracking + online fitting from segmented PointCloud2.</description>
  ```

- `pcd_gazebo_world`（`src/pcd_gazebo_world`）：把 PCD 点云转换为 Gazebo 仿真环境/果园 world。  
  出处：`src/pcd_gazebo_world/package.xml`
  ```xml
  <description>将 PCD 点云转换为 Gazebo 仿真环境</description>
  ```

- `psolqr_local_planner`（`src/psolqr_local_planner`）：PSO + LQR 的轻量本地规划器插件。  
  出处：`src/psolqr_local_planner/package.xml`
  ```xml
  <description>Lightweight ROS Local Path Planner Plugin with PSO and LQR</description>
  ```

- `slope_costmap_layer`（`src/slope_costmap_layer`）：基于点云 slope 生成 cost 的 costmap layer 插件。  
  出处（插件描述）：`src/slope_costmap_layer/slope_costmap_layer_plugin.xml`
  ```xml
  <description>
        A costmap layer that uses slope information from point cloud data to generate costs.
  </description>
  ```

---

## 3) Launch 文件路径 + 推荐入口 launch + 最短运行命令

### 3.1 所有源码 launch 文件路径清单

> 口径：只列出根工作区的源码 launch（`src/**.launch`）+ 根目录 `visualize_trajectory.launch`。

- 根目录：
  - `visualize_trajectory.launch`

- `src/bag_route_replay/launch/`
  - `src/bag_route_replay/launch/replay_from_bag.launch`

- `src/continuous_navigation/launch/`
  - `src/continuous_navigation/launch/simple_two_points_navigator.launch`
  - `src/continuous_navigation/launch/waypoint_navigator.launch`

- `src/orchard_scancontext_fsm/launch/`
  - `src/orchard_scancontext_fsm/launch/ch5_from_bag.launch`
  - `src/orchard_scancontext_fsm/launch/scancontext_fsm.launch`

- `src/liosam_bringup/launch/`
  - `src/liosam_bringup/launch/orchard_sim_headless.launch`
  - `src/liosam_bringup/launch/smoke.launch`

- `src/orchard_row_mapping/launch/`
  - `src/orchard_row_mapping/launch/orchard_tree_circles.launch`
  - `src/orchard_row_mapping/launch/orchard_row_mapping.launch`
  - `src/orchard_row_mapping/launch/orchard_viz.launch`
  - `src/orchard_row_mapping/launch/orchard_row_prior.launch`
  - `src/orchard_row_mapping/launch/pcd_publisher.launch`
  - `src/orchard_row_mapping/launch/orchard_row_liorl_pretty.launch`
  - `src/orchard_row_mapping/launch/orchard_tree_map_liorl.launch`

- `src/slope_costmap_layer/launch/`
  - `src/slope_costmap_layer/launch/slope_costmap_visualization.launch`
  - `src/slope_costmap_layer/launch/trajectory_with_costmap.launch`
  - `src/slope_costmap_layer/launch/slope_move_base.launch`
  - `src/slope_costmap_layer/launch/visualize_trajectory.launch`
  - `src/slope_costmap_layer/launch/process_pointcloud.launch`
  - `src/slope_costmap_layer/launch/slope_costmap.launch`
  - `src/slope_costmap_layer/launch/process_pointcloud_color.launch`
  - `src/slope_costmap_layer/launch/complete_costmap_visualization.launch`
  - `src/slope_costmap_layer/launch/slope_normal_visualization.launch`
  - `src/slope_costmap_layer/launch/move_base_style_costmap.launch`
  - `src/slope_costmap_layer/launch/ground_slope_visualization.launch`

- `src/pcd_gazebo_world/launch/`
  - `src/pcd_gazebo_world/launch/orchard_teb_replay.launch`
  - `src/pcd_gazebo_world/launch/pcd_terrain_sim.launch`
  - `src/pcd_gazebo_world/launch/orchard_pcd_sim.launch`
  - `src/pcd_gazebo_world/launch/orchard_sim.launch`
  - `src/pcd_gazebo_world/launch/gazebo.launch`
  - `src/pcd_gazebo_world/launch/orchard_pid_replay.launch`

- `src/far_planner/src/graph_decoder/launch/`
  - `src/far_planner/src/graph_decoder/launch/decoder.launch`

- `src/far_planner/src/far_planner/launch/`
  - `src/far_planner/src/far_planner/launch/far_planner.launch`

- `src/far_planner/src/boundary_handler/launch/`
  - `src/far_planner/src/boundary_handler/launch/boundary_handler.launch`

- `src/orchard_ch5_pipeline/launch/`
  - `src/orchard_ch5_pipeline/launch/ch5_from_bag.launch`
  - `src/orchard_ch5_pipeline/launch/ch5_live_teb.launch`
  - `src/orchard_ch5_pipeline/launch/move_base_orchard_teb.launch`

- `src/lio_sam_move_base_tutorial/teb_local_planner/launch/`
  - `src/lio_sam_move_base_tutorial/teb_local_planner/launch/test_optim_node.launch`

- `src/lio_sam_move_base_tutorial/local-planning-benchmark/move_base_benchmark/launch/`
  - `src/lio_sam_move_base_tutorial/local-planning-benchmark/move_base_benchmark/launch/move_base_benchmark_orchard.launch`
  - `src/lio_sam_move_base_tutorial/local-planning-benchmark/move_base_benchmark/launch/simple_navigation_goals.launch`
  - `src/lio_sam_move_base_tutorial/local-planning-benchmark/move_base_benchmark/launch/move_base_benchmark.launch`
  - `src/lio_sam_move_base_tutorial/local-planning-benchmark/move_base_benchmark/launch/move_base_benchmark_orchard_teb.launch`
  - `src/lio_sam_move_base_tutorial/local-planning-benchmark/move_base_benchmark/launch/move_base_benchmark copy.launch`

- `src/lio_sam_move_base_tutorial/local-planning-benchmark/gazebo_world/launch/`
  - `src/lio_sam_move_base_tutorial/local-planning-benchmark/gazebo_world/launch/world_launch.launch`

- `src/lio_sam_move_base_tutorial/robot_gazebo/liorl/launch/`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/liorl/launch/run_liorl.launch`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/liorl/launch/include/module_rviz.launch`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/liorl/launch/include/module_navsat.launch`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/liorl/launch/include/module_loam.launch`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/liorl/launch/include/module_robot_state_publisher.launch`

- `src/lio_sam_move_base_tutorial/robot_gazebo/liorf/launch/`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/liorf/launch/run_lio_sam_default.launch`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/liorf/launch/include/module_rviz.launch`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/liorf/launch/include/module_navsat.launch`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/liorf/launch/include/module_loam.launch`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/liorf/launch/include/module_robot_state_publisher.launch`

- `src/lio_sam_move_base_tutorial/robot_gazebo/liorf/launch/`（注意：目录在 `liorf/` 下）
  - （如上）

- `src/lio_sam_move_base_tutorial/robot_gazebo/liorf/launch/` / `liorl/launch/` 之外的仿真/导航：
  - `src/lio_sam_move_base_tutorial/robot_gazebo/warehouse_simulation_toolkit/launch/warehouse_simulation.launch`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/scout_gazebo/launch/scout_gazebo.launch`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/scout_gazebo/launch/show_sensor.launch`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/scout_slam_nav/launch/read_map.launch`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/scout_slam_nav/launch/gmapping_demo.launch`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/scout_slam_nav/launch/nav.launch`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/scout_slam_nav/launch/scout_gmapping.launch`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/scout_slam_nav/launch/gmapping.launch`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/scout_slam_nav/launch/amcl.launch`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/scout_slam_nav/launch/move_base.launch`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/pcd2pgm/launch/run.launch`
  - `src/lio_sam_move_base_tutorial/robot_gazebo/velodyne_simulator/velodyne_description/launch/example.launch`

- `src/orchard_tree_tracker/launch/`
  - `src/orchard_tree_tracker/launch/fruit_tree_tracker_replay.launch`
  - `src/orchard_tree_tracker/launch/fruit_tree_tracker.launch`

- `src/lidar_vision_detection/launch/`
  - `src/lidar_vision_detection/launch/demo.launch`
  - `src/lidar_vision_detection/launch/vision_detection.launch`
  - `src/lidar_vision_detection/launch/integration_with_liosam.launch`

- `src/orchard_teb_mode_switcher/launch/`
  - `src/orchard_teb_mode_switcher/launch/teb_mode_switcher.launch`

### 3.2 推荐入口 launch（多套场景分别给出）

#### A) 仓库级“最小验证 / 无数据 smoke”

- **推荐入口**：`src/liosam_bringup/launch/smoke.launch`

最短运行命令：
```bash
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
roslaunch liosam_bringup smoke.launch
```

出处：`src/liosam_bringup/README.md`
```md
roslaunch liosam_bringup smoke.launch
```

#### B) Gazebo 果园仿真（headless）

- **推荐入口**：`src/liosam_bringup/launch/orchard_sim_headless.launch`

最短运行命令：
```bash
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
roslaunch liosam_bringup orchard_sim_headless.launch
```

出处：`README.md`
```md
roslaunch liosam_bringup orchard_sim_headless.launch
```

出处：`src/liosam_bringup/launch/orchard_sim_headless.launch`
```xml
<include file="$(find pcd_gazebo_world)/launch/orchard_sim.launch">
...
<arg name="gui" value="false"/>
<arg name="use_sim_time" value="true"/>
```

#### C) 论文第 5 章端到端管线（推荐主线入口）

> 这套入口把“分割 +（可选）树聚类 + ScanContext FSM +（可选）TEB 切换 +（可选）move_base/benchmark”串起来。

- **推荐入口（离线复现 / bag）**：`src/orchard_ch5_pipeline/launch/ch5_from_bag.launch`

最短运行命令（离线）：
```bash
roslaunch orchard_ch5_pipeline ch5_from_bag.launch bag:=/abs/path/to.bag rate:=1
```

出处：`src/orchard_ch5_pipeline/README.md`
```md
roslaunch orchard_ch5_pipeline ch5_from_bag.launch \
  bag:=/abs/path/to.bag rate:=1
```

- **推荐入口（在线 + move_base + TEB）**：`src/orchard_ch5_pipeline/launch/ch5_live_teb.launch`

最短运行命令（在线）：
```bash
roslaunch orchard_ch5_pipeline ch5_live_teb.launch \
  map_filename:=/abs/path/to/map.yaml \
  teb_server:=/move_base_benchmark/TebLocalPlannerROS
```

出处：`src/orchard_ch5_pipeline/README.md`
```md
roslaunch orchard_ch5_pipeline ch5_live_teb.launch \
  map_filename:=/abs/path/to/map.yaml \
  teb_server:=/move_base_benchmark/TebLocalPlannerROS
```

#### D) LIO-SAM 派生（建图/定位）在本仓库中的入口

- 建图（`liorf`）：`src/lio_sam_move_base_tutorial/robot_gazebo/liorf/launch/run_lio_sam_default.launch`  
  出处：`src/lio_sam_move_base_tutorial/readme.md`
  ```md
  roslaunch liorf run_lio_sam_default.launch
  ```

- 定位（`liorl`）：`src/lio_sam_move_base_tutorial/robot_gazebo/liorl/launch/run_liorl.launch`  
  出处：`src/lio_sam_move_base_tutorial/readme.md`
  ```md
  roslaunch liorl run_liorl.launch
  ```

---

## 4) Topics / Frames / PointCloud2 字段要求（含默认值来源）

### 4.1 命名体系一：`liorl/*`（定位/去畸变输出，供果园管线使用）

#### 4.1.1 关键订阅/发布 topics（由 “参数 → 代码订阅/发布” 决定）

**(1) 输入 topics（参数名 + 默认值来源）**

- `liorf/pointCloudTopic`：`/points_raw`（**yaml**）→ `pointCloudTopic`（**代码读取 param**）→ `ImageProjection` 订阅该点云（**代码订阅**）。  
  出处（yaml 默认）：`src/lio_sam_move_base_tutorial/robot_gazebo/liorl/config/lio_sam_my.yaml`
  ```yaml
  liorf:
    pointCloudTopic: "/points_raw"
  ```
  出处（代码读取 param 的默认值来源=代码）：`src/lio_sam_move_base_tutorial/robot_gazebo/liorl/include/utility.h`
  ```cpp
  nh.param<std::string>("liorf/pointCloudTopic", pointCloudTopic, "points_raw");
  ```
  出处（代码订阅 pointCloudTopic）：`src/lio_sam_move_base_tutorial/robot_gazebo/liorl/src/imageProjection.cpp`
  ```cpp
  subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, ...);
  ```

- `liorf/imuTopic`：`/imu/data`（**yaml**）→ `ImageProjection` 订阅 IMU（**代码**）。  
  出处（yaml 默认）：`src/lio_sam_move_base_tutorial/robot_gazebo/liorl/config/lio_sam_my.yaml`
  ```yaml
  imuTopic: "/imu/data"
  ```
  出处（代码订阅 imuTopic）：`src/lio_sam_move_base_tutorial/robot_gazebo/liorl/src/imageProjection.cpp`
  ```cpp
  subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, ...);
  ```

- `liorf/odomTopic`：`odometry/imu`（**yaml**）→ `ImageProjection` 订阅 `odomTopic+"_incremental"`（**代码**）。  
  出处（yaml 默认）：`src/lio_sam_move_base_tutorial/robot_gazebo/liorl/config/lio_sam_my.yaml`
  ```yaml
  odomTopic: "odometry/imu"
  ```
  出处（代码订阅 odomTopic+"_incremental"）：`src/lio_sam_move_base_tutorial/robot_gazebo/liorl/src/imageProjection.cpp`
  ```cpp
  subOdom = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, ...);
  ```

**(2) 核心输出 topics（代码硬编码）**

- 去畸变输出：
  - `liorl/deskew/cloud_deskewed`（PointCloud2）
  - `liorl/deskew/cloud_info`（`liorl::cloud_info`）
  出处：`src/lio_sam_move_base_tutorial/robot_gazebo/liorl/src/imageProjection.cpp`
  ```cpp
  pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2>("liorl/deskew/cloud_deskewed", 1);
  pubLaserCloudInfo = nh.advertise<liorl::cloud_info>("liorl/deskew/cloud_info", 1);
  ```

- 里程计/轨迹/地图相关（`mapOptmization.cpp`）：
  - `liorl/mapping/odometry_incremental`
  - `liorl/mapping/odometry`
  - `liorl/mapping/path`
  - `liorl/mapping/map_local`
  - `liorl/mapping/map_global`
  - `liorl/localization/global_map`（全局地图点云）
  - `liorl/save_map`（service）
  出处：`src/lio_sam_move_base_tutorial/robot_gazebo/liorl/src/mapOptmization.cpp`
  ```cpp
  pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("liorl/mapping/odometry_incremental", 1);
  pubPath = nh.advertise<nav_msgs::Path>("liorl/mapping/path", 1);
  pubGlobalMap = nh.advertise<sensor_msgs::PointCloud2>("liorl/localization/global_map", 1);
  srvSaveMap  = nh.advertiseService("liorl/save_map", ...);
  ```

#### 4.1.2 Frames（默认值来源=yaml/代码）

- 常用 TF 命名（本仓库多处默认一致）：
  - `map`（`mapFrame`）
  - `odom_est`（`odometryFrame`）
  - `base_link_est`（`baselinkFrame`/`lidarFrame`）

出处（yaml 默认）：`src/lio_sam_move_base_tutorial/robot_gazebo/liorl/config/lio_sam_my.yaml`
```yaml
liorf:
  lidarFrame: "base_link_est"
  baselinkFrame: "base_link_est"
  odometryFrame: "odom_est"
  mapFrame: "map"
```

### 4.2 命名体系二：`liorf/*`（建图侧）

> 与 `liorl` 类似，但 topic 前缀为 `liorf/`，并提供 `/liorf/save_map` 保存地图。

- 去畸变输出：
  - `liorf/deskew/cloud_deskewed`
  - `liorf/deskew/cloud_info`
  出处：`src/lio_sam_move_base_tutorial/robot_gazebo/liorf/src/imageProjection.cpp`
  ```cpp
  pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("liorf/deskew/cloud_deskewed", 1);
  pubLaserCloudInfo = nh.advertise<liorf::cloud_info> ("liorf/deskew/cloud_info", 1);
  ```

- 保存地图 service：
  - `/liorf/save_map`
  出处：`src/lio_sam_move_base_tutorial/robot_gazebo/liorf/src/mapOptmization.cpp`
  ```cpp
  srvSaveMap  = nh.advertiseService("liorf/save_map", ...);
  ```

### 4.3 果园分割与行拟合：`orchard_row_mapping` 的 topics/默认值来源

#### 4.3.1 输入点云 topic（多套命名）

- **默认（代码）**：`~pointcloud_topic := /velodyne_points`  
  出处：`src/orchard_row_mapping/scripts/orchard_segmentation_node.py`
  ```py
  self.cloud_topic = rospy.get_param("~pointcloud_topic", "/velodyne_points")
  ```

- **预设一（liorf）**：订阅 `/liorf/deskew/cloud_deskewed`（yaml）  
  出处：`src/orchard_row_mapping/config/liorf_pretty.yaml`
  ```yaml
  pointcloud_topic: "/liorf/deskew/cloud_deskewed"
  ```

- **预设二（liorl，Chapter5 推荐）**：订阅 `/liorl/deskew/cloud_deskewed`（yaml）  
  出处：`src/orchard_row_mapping/config/liorl_pretty.yaml`
  ```yaml
  pointcloud_topic: "/liorl/deskew/cloud_deskewed"
  ```

- **Chapter5 live launch 的默认 cloud_topic**：`/liorl/deskew/cloud_deskewed`（launch arg）  
  出处：`src/orchard_ch5_pipeline/launch/ch5_live_teb.launch`
  ```xml
  <arg name="cloud_topic" default="/liorl/deskew/cloud_deskewed" />
  ```

#### 4.3.2 输出 topics（node 私有话题 → 运行时实际全局话题）

`orchard_segmentation_node.py` 在 node 内用 `~` 私有话题发布；当 node 名称为 `orchard_segmentation` 时，实际话题形如：

- `/orchard_segmentation/segmented_cloud`（PointCloud2，含 `label` 字段）
- `/orchard_segmentation/tree_cloud`（PointCloud2，含 `label` 字段，且 label 恒为 tree_class）
- `/orchard_segmentation/row_markers`（MarkerArray）

出处（代码发布私有话题名）：`src/orchard_row_mapping/scripts/orchard_segmentation_node.py`
```py
self.seg_cloud_pub = rospy.Publisher("~segmented_cloud", PointCloud2, queue_size=1)
self.tree_cloud_pub = rospy.Publisher("~tree_cloud", PointCloud2, queue_size=1)
self.row_marker_pub = rospy.Publisher("~row_markers", MarkerArray, queue_size=1)
```

#### 4.3.3 PointCloud2 字段（分割输出含 label）

出处（输出 fields 定义）：`src/orchard_row_mapping/scripts/orchard_segmentation_node.py`
```py
PointField(name="x", ...),
PointField(name="y", ...),
PointField(name="z", ...),
PointField(name="rgb", ...),
PointField(name="label", ...),
```

### 4.4 果树聚类画圆：`orchard_tree_clusters_node.py`（Chapter5 默认启用）

- 默认输入：`~input_topic := /orchard_segmentation/tree_cloud`（代码默认；Chapter5 也使用该默认）  
  出处：`src/orchard_row_mapping/scripts/orchard_tree_clusters_node.py`
  ```py
  self.input_topic = str(rospy.get_param("~input_topic", "/orchard_segmentation/tree_cloud"))
  ```

- Chapter5 live 默认参数（z 截断、聚类单元大小、最少点数等）  
  出处：`src/orchard_ch5_pipeline/launch/ch5_live_teb.launch`
  ```xml
  <arg name="tree_cluster_z_min" default="0.7" />
  <arg name="tree_cluster_z_max" default="1.3" />
  <arg name="tree_cluster_cell_size" default="0.10" />
  <arg name="tree_cluster_min_points" default="40" />
  ```

### 4.5 果树实例化 + MOT 跟踪：`orchard_tree_tracker`

- 输入 PointCloud2 fields 至少包含 `x,y,z,label`，并约定 `label==0` 为果树点。  
  出处：`src/orchard_tree_tracker/README.md`
  ```md
  - `sensor_msgs/PointCloud2`：fields 至少包含 `x,y,z,label`
  - 规则：`label==0` 视为果树点，其它值忽略
  ```
  出处（代码强制检查 required fields）：`src/orchard_tree_tracker/scripts/fruit_tree_tracker_node.py`
  ```py
  required = ("x", "y", "z", str(self.label_field))
  ...
  raise KeyError(f"missing field: {name}")
  ```
  出处（label==0 过滤）：`src/orchard_tree_tracker/scripts/fruit_tree_tracker_node.py`
  ```py
  tree_mask = finite & (label_i == 0)
  ```

- 默认订阅话题：`~input_topic := /segmented_points`（代码/launch 默认），并提供 `fruit_tree_tracker_replay.launch` 的默认输入为 `/orchard_segmentation/tree_cloud`。  
  出处（代码默认）：`src/orchard_tree_tracker/scripts/fruit_tree_tracker_node.py`
  ```py
  self.input_topic = rospy.get_param("~input_topic", "/segmented_points")
  ```
  出处（launch 默认）：`src/orchard_tree_tracker/launch/fruit_tree_tracker.launch`
  ```xml
  <arg name="input_topic" default="/segmented_points" />
  ```
  出处（replay 默认）：`src/orchard_tree_tracker/launch/fruit_tree_tracker_replay.launch`
  ```xml
  <arg name="input_topic" default="/orchard_segmentation/tree_cloud" />
  ```

### 4.6 ScanContext 模式话题：`/fsm/mode`

- 默认输入点云：`/liorl/deskew/cloud_deskewed`
- 默认输出模式话题：`/fsm/mode`

出处：`src/orchard_scancontext_fsm/launch/scancontext_fsm.launch`
```xml
<arg name="cloud_topic" default="/liorl/deskew/cloud_deskewed" />
<arg name="mode_topic" default="/fsm/mode" />
```

### 4.7 move_base_benchmark/TEB 的 frames 与 odom topic（Chapter5 live 默认）

Chapter5 live 默认采用：

- `global_frame := map`
- `local_frame := odom_est`
- `base_frame := base_link_est`
- `odom_topic := liorl/mapping/odometry_incremental`

出处：`src/orchard_ch5_pipeline/launch/ch5_live_teb.launch`
```xml
<arg name="global_frame" default="map" />
<arg name="local_frame" default="odom_est" />
<arg name="base_frame" default="base_link_est" />
<arg name="odom_topic" default="liorl/mapping/odometry_incremental" />
```

出处（move_base 节点 remap odom）：`src/orchard_ch5_pipeline/launch/move_base_orchard_teb.launch`
```xml
<remap from="odom" to="$(arg odom_topic)" />
```

### 4.8 PointCloud2 字段要求汇总（ring/time/label 等）

| 模块/阶段 | 话题示例 | 必需字段 | 规则/备注 | 出处 |
| --- | --- | --- | --- | --- |
| `liorl` 原始输入点云 | `liorf/pointCloudTopic`（默认 `/points_raw`） | **必须有** `ring`；**建议有** `time` 或 `t` | 无 `ring` 会 `ros::shutdown()`；无 `time/t` 会禁用 deskew 并警告漂移 | `src/lio_sam_move_base_tutorial/robot_gazebo/liorl/src/imageProjection.cpp`（见下） |
| `orchard_row_mapping` 输入 | `~pointcloud_topic`（默认 `/velodyne_points` 或 preset 指向 `/lior*/deskew/cloud_deskewed`） | 仅用 `x,y,z` | 读取时指定 `field_names=("x","y","z")` | `src/orchard_row_mapping/scripts/orchard_segmentation_node.py` |
| `orchard_row_mapping` 输出（分割点云） | `/orchard_segmentation/segmented_cloud` | `x,y,z,rgb,label` | `label` 为 FLOAT32；`rgb` 打包为 float | `src/orchard_row_mapping/scripts/orchard_segmentation_node.py` |
| `orchard_tree_tracker` 输入 | `~input_topic`（默认 `/segmented_points`） | **至少** `x,y,z,label` | `label==0` 为树；缺字段直接抛错 | `src/orchard_tree_tracker/README.md` + `src/orchard_tree_tracker/scripts/fruit_tree_tracker_node.py` |

出处（`liorl` 对 `ring/time` 的检查）：`src/lio_sam_move_base_tutorial/robot_gazebo/liorl/src/imageProjection.cpp`
```cpp
if (currentCloudMsg.fields[i].name == "ring") { ... }
...
ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
...
if (field.name == "time" || field.name == "t") { ... }
...
ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
```

---

## 5) 树实例化 / 跟踪 / MOT / 拟合：代码位置 + 参数默认值出处

### 5.1 `orchard_tree_tracker`（实例化 + MOT + 在线拟合）

#### 5.1.1 代码位置（入口）

- 主要节点：`src/orchard_tree_tracker/scripts/fruit_tree_tracker_node.py`
- 对应 launch：
  - `src/orchard_tree_tracker/launch/fruit_tree_tracker.launch`
  - `src/orchard_tree_tracker/launch/fruit_tree_tracker_replay.launch`

出处（节点描述/定位）：`src/orchard_tree_tracker/README.md`
```md
ROS1（rospy）果树实例化 + MOT 跟踪 + 在线拟合节点。
```

#### 5.1.2 参数默认值（来源=launch vs 代码）

> 这包的“默认值”有两套：  
> - **代码默认**：`rospy.get_param("~xxx", default)`（直接跑节点脚本时生效）  
> - **launch 默认**：`fruit_tree_tracker.launch` 里 `<arg default=...>`（用 roslaunch 启动时生效）
>
> 本仓库中两者数值保持一致（便于复现实验）。

**(A) launch 默认值来源（推荐以此作为“roslaunch 默认”定义）**  
出处：`src/orchard_tree_tracker/launch/fruit_tree_tracker.launch`
```xml
<!-- ROI -->
<arg name="roi_x_max" default="10.0" />
<arg name="roi_y_min" default="-4.0" />
...
<!-- Preprocess / Instancing -->
<arg name="voxel_size" default="0.03" />
<arg name="cell_size" default="0.10" />
<arg name="grid_T" default="5" />
<!-- MOT -->
<arg name="gate_distance" default="0.30" />
<arg name="max_missed" default="10" />
<!-- Fit -->
<arg name="K" default="20" />
<arg name="ema_alpha" default="0.4" />
```

**(B) 代码默认值来源（直接运行脚本时的 fallback）**  
出处：`src/orchard_tree_tracker/scripts/fruit_tree_tracker_node.py`
```py
self.input_topic = rospy.get_param("~input_topic", "/segmented_points")
...
voxel_size=float(rospy.get_param("~voxel_size", 0.03)),
cell_size=float(rospy.get_param("~cell_size", 0.10)),
count_threshold=int(rospy.get_param("~grid_T", 5)),
gate_distance=float(rospy.get_param("~gate_distance", 0.30)),
max_missed=int(rospy.get_param("~max_missed", 10)),
window_size=int(rospy.get_param("~K", 20)),
ema_alpha=float(rospy.get_param("~ema_alpha", 0.4)),
```

#### 5.1.3 MOT/拟合逻辑的“代码锚点”

- ROI/voxel/grid/MOT/滑窗拟合的高层描述（便于检索关键逻辑）  
  出处：`src/orchard_tree_tracker/scripts/fruit_tree_tracker_node.py`
  ```py
  Pipeline (per frame):
    1) ROI crop (x/y/z ranges)
    2) Voxel downsample
    3) 2D grid connected-components instancing (XY projection)
    4) Simple MOT tracking (nearest-neighbor with gating)
    5) Sliding-window (K) online fitting + EMA smoothing
  ```

### 5.2 `orchard_row_mapping`（分割 + 行拟合 + 先验/树图）

#### 5.2.1 关键代码文件位置

- 分割 + 行拟合节点：`src/orchard_row_mapping/scripts/orchard_segmentation_node.py`
- RandLA-Net 推理封装：`src/orchard_row_mapping/orchard_row_mapping/segmentation/`（`inference.py`, `model_loader.py` 等）
- 先验行发布（基于全局地图 + TF）：`src/orchard_row_mapping/scripts/orchard_row_prior_node.py`
- 树点地图累积（tree map builder）：`src/orchard_row_mapping/scripts/orchard_tree_map_builder_node.py`
- 树圈拟合（可视化/离线）：`src/orchard_row_mapping/scripts/orchard_tree_circles_node.py`

出处（目录说明）：`src/orchard_row_mapping/README.md`
```md
├── scripts/orchard_segmentation_node.py
└── orchard_row_mapping/segmentation
    ├── inference.py
    ├── model_loader.py
```

#### 5.2.2 参数默认值来源（代码 vs config YAML）

- **代码默认**（节选）：  
  出处：`src/orchard_row_mapping/scripts/orchard_segmentation_node.py`
  ```py
  self.cloud_topic = rospy.get_param("~pointcloud_topic", "/velodyne_points")
  self.num_points = rospy.get_param("~num_points", 16384)
  self.tree_prob_threshold = rospy.get_param("~tree_prob_threshold", 0.0)
  self.min_points_per_row = rospy.get_param("~min_points_per_row", 200)
  self.use_gpu = rospy.get_param("~use_gpu", True)
  ```

- **launch 加载 YAML（推荐把 YAML 视为“运行默认”）**：  
  出处：`src/orchard_row_mapping/launch/orchard_row_mapping.launch`
  ```xml
  <arg name="config" default="$(find orchard_row_mapping)/config/default.yaml" />
  ...
  <rosparam file="$(arg config)" command="load" />
  ```

- **两套常用 topic 命名预设**（`liorf_pretty.yaml` vs `liorl_pretty.yaml`），用于把输入切到 `lior*/deskew/cloud_deskewed`：  
  出处：`src/orchard_row_mapping/config/liorf_pretty.yaml`
  ```yaml
  pointcloud_topic: "/liorf/deskew/cloud_deskewed"
  ```
  出处：`src/orchard_row_mapping/config/liorl_pretty.yaml`
  ```yaml
  pointcloud_topic: "/liorl/deskew/cloud_deskewed"
  ```

#### 5.2.3 “先验行/树图”相关 launch（参数默认值来源）

- 行先验发布入口：`src/orchard_row_mapping/launch/orchard_row_prior.launch`（默认 pcd 路径、模型文件等）  
  出处：`src/orchard_row_mapping/launch/orchard_row_prior.launch`
  ```xml
  <arg name="pcd_path" default="/mysda/w/w/lio_ws/maps/GlobalMap.pcd" />
  <arg name="row_model_file" default="$(find orchard_row_mapping)/config/row_model_pca_major.json" />
  ```

- 树点地图累积入口：`src/orchard_row_mapping/launch/orchard_tree_map_liorl.launch`（默认输出 `/mysda/w/w/lio_ws/maps/TreeMap_auto.pcd`）  
  出处：`src/orchard_row_mapping/launch/orchard_tree_map_liorl.launch`
  ```xml
  <arg name="output_pcd" default="/mysda/w/w/lio_ws/maps/TreeMap_auto.pcd" />
  <node ... type="orchard_tree_map_builder_node.py" ...>
    <param name="input_topic" value="/orchard_segmentation/tree_cloud" />
    <param name="map_frame" value="map" />
  ```

### 5.3 `orchard_tree_clusters_node.py`（流式树聚类/圆拟合）

- 代码位置：`src/orchard_row_mapping/scripts/orchard_tree_clusters_node.py`
- 关键参数默认值（节选）：输入 topic、z 截断、聚类 cell size、最少点数  
  出处：`src/orchard_row_mapping/scripts/orchard_tree_clusters_node.py`
  ```py
  self.input_topic = str(rospy.get_param("~input_topic", "/orchard_segmentation/tree_cloud"))
  self.filter_z_min = float(rospy.get_param("~z_min", 0.7))
  self.filter_z_max = float(rospy.get_param("~z_max", 1.3))
  self.cluster_cell_size = float(rospy.get_param("~cluster_cell_size", 0.10))
  self.min_points_per_cluster = int(rospy.get_param("~min_points_per_tree", 40))
  ```

### 5.4 其它“拟合/可视化”入口：树圈拟合（离线/地图级）

- 入口：`src/orchard_row_mapping/launch/orchard_tree_circles.launch`  
  默认输入 `input_topic:=/orchard_tree_map_builder/tree_map`，并可选择 `pcd_path` 直接读 PCD。  
  出处：`src/orchard_row_mapping/launch/orchard_tree_circles.launch`
  ```xml
  <arg name="input_topic" default="/orchard_tree_map_builder/tree_map" />
  <arg name="pcd_path" default="" />
  ```

---

## 6) 依赖（package.xml 摘要 + GTSAM/PCL 等版本要求）

### 6.1 ROS/平台

- 目标平台：Ubuntu 20.04 + ROS Noetic。  
  出处：`README.md`
  ```md
  [![ROS: Noetic](...)](https://wiki.ros.org/noetic)
  [![Ubuntu: 20.04](...)](https://releases.ubuntu.com/20.04/)
  ```

### 6.2 LIO（`liorf` / `liorl`）C++ 依赖要点

- `liorl` 明确要求 PCL 版本：`find_package(PCL 1.10.0 REQUIRED QUIET)`；并依赖 `GTSAM`、`small_gicp`、`GeographicLib`、`OpenCV`、`Boost timer`。  
  出处：`src/lio_sam_move_base_tutorial/robot_gazebo/liorl/CMakeLists.txt`
  ```cmake
  find_package(PCL 1.10.0 REQUIRED QUIET)
  find_package(GTSAM REQUIRED QUIET)
  find_package(small_gicp REQUIRED)
  find_package(GeographicLib REQUIRED)
  ```

- `liorf` 依赖 `PCL`、`GTSAM`、`OpenCV`、`GeographicLib`、`Boost timer`（未在 CMake 中写死 PCL 版本）。  
  出处：`src/lio_sam_move_base_tutorial/robot_gazebo/liorf/CMakeLists.txt`
  ```cmake
  find_package(PCL REQUIRED QUIET)
  find_package(GTSAM REQUIRED QUIET)
  find_package(GeographicLib REQUIRED)
  ```

> 版本备注（可能存在不一致）：`robot_gazebo/README.md` 写的是 `GTSAM 4.1`、`PCL 1.9`，而 `liorl/CMakeLists.txt` 里要求 `PCL 1.10.0`。  
> 出处：`src/lio_sam_move_base_tutorial/robot_gazebo/README.md`
```md
* GTSAM 4.1
* Eigen 3.3.7
* PCL 1.9
```

### 6.3 分割（`orchard_row_mapping`）Python/ML 依赖要点

- 代码直接 import `torch`，并默认 `~use_gpu := True`（有 CUDA 就用 GPU，否则退回 CPU）。  
  出处：`src/orchard_row_mapping/scripts/orchard_segmentation_node.py`
  ```py
  import torch
  ...
  self.use_gpu = rospy.get_param("~use_gpu", True)
  ```

- 运行环境通过 launch 注入 `PYTHONPATH` 指向 `/.venv_orchard/...`（意味着依赖很可能装在该 venv）。  
  出处：`src/orchard_row_mapping/launch/orchard_row_mapping.launch`
  ```xml
  <env name="PYTHONPATH" value="/mysda/w/w/lio_ws/.venv_orchard/lib/python3.8/site-packages:$(optenv PYTHONPATH)" />
  ```

### 6.4 其它常见 ROS 依赖（摘选）

- `orchard_teb_mode_switcher` 依赖 `dynamic_reconfigure`。  
  出处：`src/orchard_teb_mode_switcher/package.xml`
  ```xml
  <exec_depend>dynamic_reconfigure</exec_depend>
  ```

- `slope_costmap_layer` 依赖 `grid_map_*`、`costmap_2d`、`pcl_ros`。  
  出处：`src/slope_costmap_layer/package.xml`
  ```xml
  <build_depend>grid_map_ros</build_depend>
  <build_depend>grid_map_pcl</build_depend>
  <build_depend>costmap_2d</build_depend>
  ```

---

## 7) 风险点 / 常见坑（现象 -> 原因 -> 怎么查）

### 7.1 点云字段缺失（`ring` / `time` / `label`）

- 现象：启动 `liorl`/`liorf` 后直接报错退出：`Point cloud ring channel not available...`  
  原因：输入点云缺少 `ring` 字段（机械雷达线号）。  
  怎么查：`rostopic echo -n1 <pointCloudTopic>` 看 `fields`；或用 `rosmsg show sensor_msgs/PointCloud2` 理解结构。  
  出处：`src/lio_sam_move_base_tutorial/robot_gazebo/liorl/src/imageProjection.cpp`
  ```cpp
  if (currentCloudMsg.fields[i].name == "ring") { ... }
  ...
  ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
  ros::shutdown();
  ```

- 现象：系统“能跑但漂移很大”，日志提示 `deskew function disabled`  
  原因：点云缺少 per-point 相对时间字段 `time` 或 `t`，去畸变被禁用。  
  怎么查：检查 PointCloud2 `fields` 是否含 `time`/`t`；确认驱动输出的是“同一帧内相对时间（0~scan_period）”。  
  出处：`src/lio_sam_move_base_tutorial/robot_gazebo/liorl/src/imageProjection.cpp`
  ```cpp
  if (field.name == "time" || field.name == "t") { ... }
  ...
  ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
  ```

- 现象：`orchard_tree_tracker` 无输出或直接抛异常 `missing field: label`  
  原因：输入 PointCloud2 没有 `label` 字段（或字段名不同）。  
  怎么查：检查输入点云 `fields`；必要时在 launch 里设置 `label_field:=xxx`。  
  出处：`src/orchard_tree_tracker/README.md`
  ```md
  fields 至少包含 `x,y,z,label`
  ```
  出处：`src/orchard_tree_tracker/launch/fruit_tree_tracker.launch`
  ```xml
  <arg name="label_field" default="label" />
  ```

### 7.2 TF/坐标系命名不一致（`map/odom_est/base_link_est`）

- 现象：move_base 报 TF lookup 失败、costmap 不更新或机器人不动。  
  原因：本仓库多处默认 frames 为 `map`/`odom_est`/`base_link_est`，如果你的系统用的是 `odom`/`base_link` 等，需要统一 remap/参数。  
  怎么查：`rosrun tf view_frames` / RViz TF；对照 `move_base_orchard_teb.launch` 的 frame 参数。  
  出处：`src/orchard_ch5_pipeline/launch/move_base_orchard_teb.launch`
  ```xml
  <arg name="global_frame" default="map" />
  <arg name="local_frame" default="odom_est" />
  <arg name="base_frame" default="base_link_est" />
  ```

### 7.3 `liorl` 参数命名“看起来怪”（`liorf/...` 命名空间）

- 现象：你把参数写在 `liorl:` 下但程序不生效。  
  原因：`liorl` 的 `ParamServer` 实际读取的是 `liorf/xxx` 参数名（代码继承自 liorf），因此 yaml 顶层也写成了 `liorf:`。  
  怎么查：对照 `liorl/include/utility.h` 的 `nh.param("liorf/...", ...)`；确保 rosparam 加载的 key 与之一致。  
  出处：`src/lio_sam_move_base_tutorial/robot_gazebo/liorl/include/utility.h`
  ```cpp
  nh.param<std::string>("liorf/pointCloudTopic", pointCloudTopic, "points_raw");
  ```
  出处：`src/lio_sam_move_base_tutorial/robot_gazebo/liorl/config/lio_sam_my.yaml`
  ```yaml
  liorf:
    pointCloudTopic: "/points_raw"
  ```

### 7.4 分割模型文件缺失（`orchard_row_mapping`）

- 现象：启动分割节点直接报：`No valid checkpoint file found for segmentation model`  
  原因：`~model_path` 没配或默认查找路径找不到 checkpoint。  
  怎么查：查看 `~model_path` 参数；确认 checkpoint 文件存在且可读。  
  出处：`src/orchard_row_mapping/scripts/orchard_segmentation_node.py`
  ```py
  checkpoint_path = self._resolve_checkpoint(checkpoint_param)
  if checkpoint_path is None:
      raise RuntimeError("No valid checkpoint file found for segmentation model")
  ```

### 7.5 时间同步 / IMU 频率（LIO）

- 现象：轨迹抖动/漂移、转弯时表现差。  
  原因：IMU 频率太低或时间戳不同步；LIO-SAM 类方法对 IMU 质量/频率敏感。  
  怎么查：检查 IMU topic 频率、时间戳单调性；对照 README 的建议频率。  
  出处：`src/lio_sam_move_base_tutorial/robot_gazebo/liorf/README.md`
  ```md
  We use ... outputs data at 500Hz. We recommend using an IMU that gives at least a 200Hz output rate.
  ```

### 7.6 /use_sim_time 与 bag 回放

- 现象：离线回放时节点不输出/时间不动。  
  原因：没启用 `/use_sim_time` 或 bag 未带 `/clock`。  
  怎么查：`rosparam get /use_sim_time`；`rostopic echo /clock`。  
  出处：`src/orchard_scancontext_fsm/launch/ch5_from_bag.launch`
  ```xml
  <param name="/use_sim_time" value="true" />
  <node pkg="rosbag" type="play" ... args="--clock ..."/>
  ```

---

## 附录 A) 进一步阅读（仓库内已有文档入口）

- 快速启动与仓库定位：`README.md`
- 包分组与推荐入口：`docs/PACKAGES.md`
- `liosam_bringup` 的使用：`src/liosam_bringup/README.md`
- Chapter5 端到端 pipeline：`src/orchard_ch5_pipeline/README.md`
- 分割与行拟合说明：`src/orchard_row_mapping/README.md`
- 树跟踪节点说明：`src/orchard_tree_tracker/README.md`
