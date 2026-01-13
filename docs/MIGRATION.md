# Migration Notes

本次整理尽量保持 ROS 对外接口不变；主要变更集中在“仓库归置/入口清晰/安装规则”。

## Path moves (non-catkin content)

| Old | New | Notes |
| --- | --- | --- |
| `src/start_navigation.sh` | `tools/start_navigation.sh` | `src/` 根目录只保留 catkin packages + `CMakeLists.txt` |
| `src/planner_comparison/` | `tools/planner_comparison/` | 规划器对比分析工具不参与 catkin 编译 |
| `src/c++/` | `tools/cpp_scratch/` | C++ 练习/临时代码不参与 catkin 编译 |

## Build behavior

- `slope_costmap_layer`：未安装 `grid_map_ros/grid_map_pcl` 时会在 CMake 配置阶段提示 WARNING 并跳过编译（安装 `ros-noetic-grid-map` 后恢复正常编译）。

## New entry package

- 新增 `src/liosam_bringup/`：聚合常用 demo 启动入口（见 `docs/PACKAGES.md`）。
