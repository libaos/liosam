# liosam (ROS1 / Gazebo Classic workspace)

[![CI](https://github.com/libaos/liosam/actions/workflows/ci.yml/badge.svg)](https://github.com/libaos/liosam/actions/workflows/ci.yml)
[![License: Mixed](https://img.shields.io/badge/license-mixed-lightgrey.svg)](LICENSES.md)
[![ROS: Noetic](https://img.shields.io/badge/ROS-Noetic-blue.svg)](https://wiki.ros.org/noetic)
[![Ubuntu: 20.04](https://img.shields.io/badge/Ubuntu-20.04-E95420.svg)](https://releases.ubuntu.com/20.04/)

一个面向果园场景的 ROS1（catkin）工作区，包含：

- Gazebo Classic 仿真（PCD→地形 mesh、果园 world 生成、机器人加载与路径回放）
- 果树点云分割与行先验（`orchard_row_mapping`）
- 导航/局部规划相关包（`teb_local_planner`、`psolqr_local_planner` 等）
- 一些实验脚本与工具（`tools/`）

> 目标：第一次进来 3 分钟跑起来（前提：你本机已装好 ROS）。

## Table of Contents

- [Quick Start (3 min)](#quick-start-3-min)
- [What’s Included](#whats-included)
- [Repository Layout](#repository-layout)
- [Environment](#environment)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Quick Start (3 min)

推荐走“正统 catkin 工作区”方式：编译 → `source devel/setup.bash` → `roslaunch`。

### 0) Prerequisites

- **Ubuntu 20.04 + ROS Noetic**（建议装 `ros-noetic-desktop-full`，自带 Gazebo Classic）
- 你能在终端里运行 `roscore` / `roslaunch`

如果你还没装 ROS：先按官方文档安装（见 [Environment](#environment)）。

### 1) Clone

```bash
git clone https://github.com/libaos/liosam.git
cd liosam
```

### 2) Source ROS + expose workspace packages

```bash
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

> 如果你是全新系统、编译报缺依赖：先跑 `rosdep install --from-paths src --ignore-src -r -y --rosdistro noetic`（见 `docs/FAQ.md`）。

### 3) Smoke test (no data, no GUI)

```bash
roslaunch liosam_bringup smoke.launch
```

### 4) (Optional) Run Gazebo orchard sim (headless)

```bash
roslaunch liosam_bringup orchard_sim_headless.launch
```

常见可选项（Gazebo）：

- `paused:=true`：启动后暂停物理仿真（更省资源）
- `world_name:=.../xxx.world`：换 world（见 `src/pcd_gazebo_world/worlds/`）
- 需要 GUI：`roslaunch pcd_gazebo_world orchard_sim.launch gui:=true enable_robot_state_publisher:=true`

### 4) View in RViz

新开一个终端：

```bash
cd liosam
source /opt/ros/noetic/setup.bash
source devel/setup.bash

rviz -d "$(rospack find scout_gazebo)/config/show_robot.rviz"
```

如果你想自己加显示项（更通用）：

- `Fixed Frame`: `map`（没有就选 `odom`）
- Add: `TF`, `RobotModel`, `PointCloud2`（常见话题：`/velodyne_points`）, `Odometry`（`/odom`）

### Troubleshooting

- `RLException: [xxx.launch] is neither a launch file...`：基本是没把工作区加到 `ROS_PACKAGE_PATH`
- RViz 里机器人不显示：确认用 `enable_robot_state_publisher:=true`
- Gazebo GUI 崩溃/黑屏：先 `gui:=false` 跑后台，再单独开 `gzclient`，或试 `LIBGL_ALWAYS_SOFTWARE=1 gzclient`

## What’s Included

工作区里包很多，这里列最常用的入口（都在 `src/` 下）：

- `liosam_bringup`: 仓库级 bringup/统一 demo 入口（入口文档：`src/liosam_bringup/README.md`）
- `pcd_gazebo_world`: PCD→Gazebo 地形、果园 world 生成、Gazebo/回放脚本（入口文档：`src/pcd_gazebo_world/README.md`）
- `orchard_row_mapping`: 点云分割（RandLA-Net）+ 果树行拟合（入口文档：`src/orchard_row_mapping/README.md`）
- `lio_sam_move_base_tutorial`: 机器人仿真/导航相关包集合（包含 `scout_gazebo`、`teb_local_planner` 等）
- `psolqr_local_planner`: 另一个局部规划器插件（PSO + LQR）
- `continuous_navigation`: waypoint/连续导航小工具

更完整的分组清单见 `docs/PACKAGES.md`。

## Repository Layout

```text
.
├── docs/                   # 常见问题/架构/说明文档
├── src/                    # ROS1 catkin packages（主要代码都在这里）
├── tools/                  # 仓库级脚本/工具
├── README.md               # 你正在看的文档
├── ROADMAP.md              # 里程碑计划
├── CHANGELOG.md            # 版本变更记录（Keep a Changelog）
└── LICENSES.md             # 多许可证说明（按包查看）
```

说明：

- 本仓库默认不会提交运行产物/大数据（如 rosbags、maps、conda/venv 等）；见 `.gitignore`
- `src/pcd_gazebo_world/maps/` 里有少量示例素材（用于 world/对比图），不等同于完整数据集

## Environment

### Recommended

- OS: Ubuntu 20.04
- ROS: Noetic (ROS1)
- Gazebo: Classic 11（随 `desktop-full`）
- Python: 3.8+

### Verified versions

目前仓库还没有完整 CI 覆盖 ROS 编译链路，下面状态以 “建议优先验证” 的方式列出：

| Component | Recommended | Status |
| --- | --- | --- |
| Ubuntu | 20.04 | ⏳待验证（欢迎反馈） |
| ROS1 | Noetic | ⏳待验证（欢迎反馈） |
| Gazebo Classic | 11 | ⏳待验证（欢迎反馈） |

如果你在其它版本跑通了，请在 issue 里带上你的环境信息，方便我们更新表格。

## Roadmap

见 `ROADMAP.md`。

## Contributing

见 `CONTRIBUTING.md`（包含提 issue/PR 的最小信息要求与调试建议）。

## License

本仓库聚合了多个 ROS 包/第三方组件，许可证可能不同；请查看 `LICENSES.md`（按包列出）。  
如果你要把本仓库开源发布，强烈建议先梳理清楚每个子目录的来源与许可证兼容性。
