# FAQ / Troubleshooting

## Build 失败：缺依赖包

- 报错类似 `Could not find a package configuration file provided by "..."`
  - 先确保你已经 `source /opt/ros/noetic/setup.bash`
  - 推荐用 `rosdep` 一次性安装依赖：

```bash
sudo apt-get update
rosdep update
rosdep install --from-paths src --ignore-src -r -y --rosdistro noetic
```

> 注：当前环境未安装 `grid_map_ros/grid_map_pcl` 时，`slope_costmap_layer` 会在 CMake 配置阶段给出 WARNING 并跳过编译；安装 `ros-noetic-grid-map` 后即可启用该包。

## Gazebo 启动/渲染异常（黑屏、崩溃、远程无界面）

- 先用无界面模式验证 `gzserver` 是否正常：

```bash
source devel/setup.bash
roslaunch liosam_bringup orchard_sim_headless.launch
```

- 如果你需要 GUI：
  - 本地机器：`roslaunch pcd_gazebo_world orchard_sim.launch gui:=true`
  - 远程机器：优先 `gui:=false`，再单独在有显示的机器上开 `gzclient`（同一 `GAZEBO_MASTER_URI`）
- OpenGL/驱动问题可尝试：
  - `LIBGL_ALWAYS_SOFTWARE=1 gzclient`

## RViz 里看不到东西

- `Fixed Frame` 先试：`map`（没有就用 `odom`）
- 常用显示项：`TF`、`RobotModel`、`PointCloud2`（例如 `/velodyne_points`）、`Odometry`
- 如果 TF 树断了：
  - 先确认 `robot_state_publisher` 是否启用（见 bringup/launch 的 `enable_robot_state_publisher`）

## LiDAR/IMU 时间同步相关（定位漂、轨迹发散）

典型现象：
- 运动时点云“拉丝/扭曲”、建图重影
- IMU 融合后越跑越偏，或短时间内发散

建议检查：
- `/clock`（仿真）是否在跑、`use_sim_time` 是否一致
- LiDAR 与 IMU 的时间戳是否同一时基（ROS time / GPS time）
- 回放 bag 时是否 `--clock`，以及是否 `use_sim_time:=true`

## 点云字段缺失（deskew/预处理相关）

部分算法/预处理需要 PointCloud2 里有额外字段：
- `ring`（线束号）
- `time`/`t`（点时间，用于去畸变）

建议：
- 用 `rostopic echo -n1 /velodyne_points` 或 `rosmsg show sensor_msgs/PointCloud2` 确认 fields
- 缺字段时：优先在驱动/转换节点补齐（不要在核心算法里硬改）

## 外参/重力方向不对（IMU 外参、frame 定义）

典型现象：
- 车在平地也“倾斜”、重力方向错
- `base_link`/`imu_link` 的静态 TF 配错导致姿态异常

建议检查：
- `static_transform_publisher` / URDF 中的外参是否一致
- IMU 安装方向（坐标轴）与算法期望是否一致
