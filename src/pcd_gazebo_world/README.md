# pcd_gazebo_world

把 `.pcd` 点云离线转换为 Gazebo（Classic）可加载的地形 mesh，然后在 Gazebo 里作为静态模型加载。

另外也提供了一个“果园 world 生成器”：根据果树行先验（row model json）在 Gazebo 中生成一排排树干障碍物，配合仿真机器人就能像真实环境一样发 `/cmd_vel` 控制。

## 生成 mesh（从 PCD）

仓库里自带了两份示例点云（推荐用更稠密的那份）：
- `maps/manual_priors/map4_liosam_deskew_map_dense.pcd`
- `maps/manual_priors/map4_points_raw_map.pcd`

在工作空间根目录执行：

```bash
python3 src/pcd_gazebo_world/scripts/pcd_to_mesh.py \
  maps/manual_priors/map4_liosam_deskew_map_dense.pcd \
  src/pcd_gazebo_world/models/pcd_terrain/meshes/terrain \
  --voxel 0.4 --depth 8 --formats stl
```

- 第二个参数是输出 **base 路径**：会生成 `terrain.stl`（以及你指定的其它格式）。  
- `--voxel` 越大越快但细节越少；`--depth` 越大越精细但更慢/更吃内存。

## 启动 Gazebo

```bash
source devel/setup.bash
roslaunch pcd_gazebo_world gazebo.launch
```

如果你希望在该地形里直接加载可控机器人（发 `/cmd_vel` 会动）：

```bash
source devel/setup.bash
roslaunch pcd_gazebo_world pcd_terrain_sim.launch
```

如需调整地形位置/朝向：

- 修改 `src/pcd_gazebo_world/worlds/pcd_world.world` 里的 `<pose>`  
- 或修改 `src/pcd_gazebo_world/models/pcd_terrain/model.sdf` 里的 `<scale>`

## 生成果园 world（推荐用于可控仿真）

用行先验文件生成一个 `orchard_rows.world`（默认会自动找：`maps/manual_priors/map4_manual.json` 或 `src/orchard_row_mapping/config/row_model_pca_major.json`）：

```bash
python3 src/pcd_gazebo_world/scripts/generate_orchard_world.py \
  --spacing 4.0 --trunk-radius 0.15 --trunk-height 2.0 --canopy-radius 0.8
```

输出默认写到：`src/pcd_gazebo_world/worlds/orchard_rows.world`。

## 生成更像“真实果园”的树干 world（从 PCD 提取树中心）

把点云在“树干高度带”(例如 z=0.7~1.3m) 做聚类，自动提取每棵树中心，然后生成 world。

如果你已经有“只包含树点”的 PCD（更推荐，例如 `maps/map4_bin_tree_label0.pcd`），效果会更像果园。

```bash
python3 src/pcd_gazebo_world/scripts/pcd_to_orchard_world.py \
  --pcd maps/map4_bin_tree_label0.pcd \
  --row-model maps/manual_priors/map4_manual.json
```

默认会输出：
- `src/pcd_gazebo_world/worlds/orchard_from_pcd.world`
- 如果输入是树点 PCD（如 `maps/map4_bin_tree_label0.pcd`）：`maps/map4_bin_tree_label0_circles.json`
- 如果输入是稠密地图 PCD：`maps/manual_priors/map4_tree_circles.json`

### 用 rosbag 反向认证树中心（推荐）

如果你觉得 “PCD 提取的树中心”和真实 rosbag 的轨迹/点云不一致，可以用同一个 rosbag 做认证：

1) 从 rosbag 的 `/liorl/mapping/cloud_registered`（map frame）抽样提取树中心：

```bash
python3 src/pcd_gazebo_world/scripts/rosbag_registered_cloud_to_orchard_world.py \
  --bag rosbags/2025-10-29-16-05-00.bag
```

2) 用最近邻阈值过滤掉 PCD 里不被 rosbag 支持的树（并生成新的 Gazebo world）：

```bash
python3 src/pcd_gazebo_world/scripts/validate_tree_centers.py \
  --circles maps/map4_bin_tree_label0_circles.json \
  --validator rosbags/runs/tree_centers_from_cloud_registered_full_0p7_1p3.json \
  --threshold 1.0 --align 1
```

## 启动“果园 + 机器人”仿真

```bash
source devel/setup.bash
roslaunch pcd_gazebo_world orchard_sim.launch
```

- 机器人模型默认用 `scout_gazebo` 的 `base_no_realsense.xacro`（避免依赖 `realsense_ros_gazebo`）。
- 常用话题：`/cmd_vel`（控制）、`/odom`、`/imu/data`、`/velodyne_points`。

如果你用的是上面“从 PCD 提取树中心”的 world，直接启动：

```bash
source devel/setup.bash
roslaunch pcd_gazebo_world orchard_pcd_sim.launch
```

默认会加载 `src/pcd_gazebo_world/worlds/orchard_from_pcd.world`（可在启动时用 `world_name:=...` 覆盖）。

### Gazebo 端口/显示相关（常见崩溃原因）

如果你遇到 “启动 10s 左右 Gazebo 就没了/报 Address already in use” 的情况，优先排查两类问题：

- **Gazebo master 端口冲突**：用 `gazebo_master_uri:=http://localhost:11346` 换个端口启动。
- **OpenGL/驱动问题（gzclient 崩溃）**：可以先 `gui:=false` 跑后台，再单独开 `gzclient`，或用 `LIBGL_ALWAYS_SOFTWARE=1` 强制软件渲染。

本包的 `gazebo.launch` / `orchard_*` 系列 launch 都支持：

- `gazebo_master_uri:=http://localhost:11345`
- `gazebo_ip:=127.0.0.1`

### BEV 快速核对（PCD vs centers）

```bash
python3 src/pcd_gazebo_world/scripts/plot_pcd_tree_centers_bev.py \
  --pcd maps/map4_bin_tree_label0.pcd \
  --circles maps/map4_bin_tree_label0_circles.json \
  --circles2 maps/map4_bin_tree_label0_circles_validated_by_bag.json
```

## rosbag 路径回放到 Gazebo（PID / TEB）

1) 先把 rosbag 的 `/liorl/mapping/path` 导出成 JSON：

```bash
python3 src/pcd_gazebo_world/scripts/rosbag_path_to_json.py \
  --bag rosbags/2025-10-29-16-05-00.bag \
  --topic /liorl/mapping/path \
  --out src/pcd_gazebo_world/maps/runs/rosbag_path.json
```

如果你的 Path 在 `odom_est`（或其它非 `map`）坐标系，但 bag 里有对应 `/tf`（例如 `map -> odom_est`），建议直接导出到 `map` frame（Gazebo 的 `global_frame` 默认是 `map`）：

```bash
python3 src/pcd_gazebo_world/scripts/rosbag_path_to_json.py \
  --bag rosbags/2025-10-29-16-05-00.bag \
  --topic /liorl/mapping/path \
  --target-frame map --tf-topic /tf \
  --out src/pcd_gazebo_world/maps/runs/rosbag_path_map.json
```

2) PID 回放（不依赖 move_base）：

```bash
source devel/setup.bash
roslaunch pcd_gazebo_world orchard_pid_replay.launch \
  use_skid_steer:=false use_planar_move:=true \
  world_name:=$(rospack find pcd_gazebo_world)/worlds/orchard_from_pcd_validated_by_bag.world \
  path_file:=src/pcd_gazebo_world/maps/runs/rosbag_path.json
```

3) TEB 回放（通过 move_base + teb_local_planner）：

```bash
source devel/setup.bash
roslaunch pcd_gazebo_world orchard_teb_replay.launch \
  use_skid_steer:=false use_planar_move:=true \
  world_name:=$(rospack find pcd_gazebo_world)/worlds/orchard_from_pcd_validated_by_bag.world \
  path_file:=src/pcd_gazebo_world/maps/runs/rosbag_path.json
```

说明：很多 rosbag 的轨迹是“绕一圈回到起点附近”，如果只发布最后一个 goal，机器人只会走很短一段就结束。  
`orchard_teb_replay.launch` 默认启用了 `use_waypoint_goals:=true`，会把整条路径拆成一串中间 goal 依次发送，从而跑完整条路线。

如果你在 Codex/聊天终端里复制命令总被自动换行破坏，直接用脚本：

- `src/pcd_gazebo_world/tools/run_orchard_teb_server.sh`（启动仿真）
- `src/pcd_gazebo_world/tools/run_orchard_gzclient.sh`（打开 GUI）
- 如果 TEB 很快停在原地（常见报 `trajectory is not feasible`），先用：`bash src/pcd_gazebo_world/tools/run_orchard_teb_server.sh 11347 no_obstacles`

回放完成后会输出 odom 轨迹 CSV（默认写到你启动 roslaunch 时的 `$(pwd)/trajectory_data/`）。

4) 对比参考路径 vs 回放轨迹：

```bash
python3 src/pcd_gazebo_world/scripts/plot_reference_vs_replay.py \
  --reference src/pcd_gazebo_world/maps/runs/rosbag_path.json \
  --replay trajectory_data/pid_odom.csv \
  --out src/pcd_gazebo_world/maps/runs/pid_reference_vs_replay.png
```
