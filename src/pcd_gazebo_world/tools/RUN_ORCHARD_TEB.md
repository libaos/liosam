# Orchard TEB replay (copy/paste safe)

Codex/terminal 聊天窗口有时会把长命令自动换行，导致复制后 bash 解析失败。  
推荐直接跑这里的脚本（短命令，不容易被换行破坏）。

## 脚本速查（哪个是启动 Gazebo / 哪个是发 cmd_vel）

- 启动仿真后端（包含：`gzserver` + 车模型 + `move_base_benchmark` + 参考路径 goal 发布；会自动产生 `/cmd_vel`）：  
  `bash src/pcd_gazebo_world/tools/run_orchard_teb_server.sh 11347`
  - 强制不启用障碍物层（更容易“跑完整条路径”，但不会主动避树）：`bash src/pcd_gazebo_world/tools/run_orchard_teb_server.sh 11347 no_obstacles`
  - 用静态树地图（全局 planner 知道树位置；实验性，若不动优先先用前两种）：`bash src/pcd_gazebo_world/tools/run_orchard_teb_server.sh 11347 static_map`
- 启动 Gazebo 可视化 GUI（`gzclient`，只负责“看”，不负责“跑”）：  
  `bash src/pcd_gazebo_world/tools/run_orchard_gzclient.sh 11347`
- 快速自检（看 `/clock /odom /move_base_simple/goal /cmd_vel` 有没有）：  
  `bash src/pcd_gazebo_world/tools/orchard_teb_check.sh`
- 只启动仿真（不带导航，用于手动 `/cmd_vel` 测试）：  
  `bash src/pcd_gazebo_world/tools/run_orchard_sim_server.sh 11347`
- 手动发 `/cmd_vel`（测试车能不能动）：见下方 “手动控制”。

## 关于 via-points（重要）

TEB 的 `via_points`（`/move_base_benchmark/TebLocalPlannerROS/via_points`）之前会导致轨迹被“拉偏”（与 `rosbag_path.json` 偏差可到几米）。  
现在默认 **不发布 via-points**（`USE_VIA_POINTS=false`），只用 `follow_path_goals.py` 顺序发 `/move_base_simple/goal` 来复现 rosbag 路线。

如果你确实想开启 via-points（实验性）：

```bash
USE_VIA_POINTS=true bash src/pcd_gazebo_world/tools/run_orchard_teb_server.sh 11347
```

## 1) 启动仿真（gzserver + move_base + TEB）

```bash
bash src/pcd_gazebo_world/tools/run_orchard_teb_server.sh 11347
```

如果你遇到 `TebLocalPlannerROS: trajectory is not feasible` 导致车停住，优先试试：

```bash
bash src/pcd_gazebo_world/tools/run_orchard_teb_server.sh 11347 no_obstacles
```

`static_map` 用到的静态树地图在：`src/pcd_gazebo_world/maps/orchard_from_pcd_validated_by_bag/map.yaml`，
如果你更新了 circles，可以用下面脚本重新生成：

```bash
python3 src/pcd_gazebo_world/scripts/circles_to_occupancy_map.py
```

可选：指定路径 JSON（默认用 `src/pcd_gazebo_world/maps/runs/rosbag_path.json`）：

```bash
bash src/pcd_gazebo_world/tools/run_orchard_teb_server.sh 11347 src/pcd_gazebo_world/maps/runs/rosbag_path.json
```

可选：指定 world（默认用验证过的 `orchard_from_pcd_validated_by_bag.world`，树更“细”，更不容易一开局就撞上）：

```bash
bash src/pcd_gazebo_world/tools/run_orchard_teb_server.sh 11347 src/pcd_gazebo_world/maps/runs/rosbag_path.json src/pcd_gazebo_world/worlds/orchard_from_pcd_validated_by_bag.world
```

## 2) 打开 Gazebo GUI（gzclient）

另开一个终端：

```bash
bash src/pcd_gazebo_world/tools/run_orchard_gzclient.sh 11347
```

如果你的机器 OpenGL 有问题再试软件渲染（会很卡）：

```bash
bash src/pcd_gazebo_world/tools/run_orchard_gzclient.sh 11347 --software
```

## 3) 不动/没 goal 时快速自检

另开终端：

```bash
bash src/pcd_gazebo_world/tools/orchard_teb_check.sh
```

正常情况下应该能看到：
- `/clock`、`/odom` 有 publisher（`/gazebo`）
- `/move_base_simple/goal` 有消息
- `/cmd_vel` 有输出（默认 ~5Hz），Gazebo 里车会开始走

如果你之前遇到“车在原地不动但 `/move_base_simple/goal` 有消息”，现在 goal_follower 会在 `/move_base/status` 没有 active goal 时自动重发，避免启动阶段丢 goal。

如果你一直刷下面这种 WARN：
`Control loop missed its desired rate of X.0000Hz...`

这是“算不过来”的性能告警（尤其你用 `--software` 软件渲染时很常见）。  
已在脚本里把 `controller_frequency/planner_frequency`、`local_costmap` 频率降下来了；如果还卡：
- 尽量别用 `--software`（很吃 CPU）
- 关掉点云显示、把 `lidar_hz/lidar_samples` 再降一点

## 3.1) 跑一会就“乱跑”/抖动（常见原因）

如果你在 server 终端里同时看到这两行：
- `GazeboRosSkidSteerDrive Plugin ... Starting ...`
- `PlanarMovePlugin ...`

说明两个驱动插件都在抢 `/cmd_vel`，很容易出现跑一会就乱跑/抖动。  
本工程默认已经改成只开 `planar_move`（关掉 `skid_steer`）。如果你自己手动 `roslaunch`，确保带上：

```bash
roslaunch pcd_gazebo_world orchard_teb_replay.launch use_skid_steer:=false use_planar_move:=true
```

## 4) 手动控制（手动发 cmd_vel）

注意：TEB 回放在运行时，`/move_base_benchmark` 也会发布 `/cmd_vel`。  
所以要做手动 `/cmd_vel` 测试，请先只启动仿真（不带导航）：

```bash
bash src/pcd_gazebo_world/tools/run_orchard_sim_server.sh 11347
```

然后再 **另开一个终端** 手动发速度（不要在 server 的 roslaunch 终端里发；否则你按 `Ctrl+C` 会把整个 roslaunch 也停掉）：

```bash
source /mysda/w/w/lio_ws/devel/setup.bash
rostopic pub -r 5 /cmd_vel geometry_msgs/Twist '{linear: {x: 0.2}, angular: {z: 0.0}}'
```

停止发送按 `Ctrl+C`。

### 常见误区：两路 /cmd_vel 导致“乱跑”

如果你在跑 TEB 回放时又手动 `rostopic pub /cmd_vel`，会出现 **两个 publisher** 同时发 `/cmd_vel`，车就会“乱跑”。  
用下面命令确认 `/cmd_vel` 只有一个 publisher：

```bash
source /mysda/w/w/lio_ws/devel/setup.bash
rostopic info /cmd_vel
```

## 5) 最简两步启动（推荐）

开两个终端：

终端1（后台仿真 + TEB 跑路线；推荐默认模式，带静态树地图 + 点云避障）：

```bash
cd /mysda/w/w/lio_ws && bash src/pcd_gazebo_world/tools/run_orchard_teb_server.sh 11347
```

如果你发现“跑一会就撞树/卡住”，优先用 `static_map`（关点云避障，只用静态树地图+膨胀，CPU 更省）：

```bash
cd /mysda/w/w/lio_ws && bash src/pcd_gazebo_world/tools/run_orchard_teb_server.sh 11347 static_map
```

如果你只是想“尽量复现 rosbag 路径”（忽略避障、最省算力），再用 `no_obstacles`：

```bash
cd /mysda/w/w/lio_ws && bash src/pcd_gazebo_world/tools/run_orchard_teb_server.sh 11347 no_obstacles
```

终端2（打开 Gazebo 可视化）：

```bash
cd /mysda/w/w/lio_ws && bash src/pcd_gazebo_world/tools/run_orchard_gzclient.sh 11347
```

如果 GUI 黑屏/卡死，改用软件渲染：

```bash
cd /mysda/w/w/lio_ws && bash src/pcd_gazebo_world/tools/run_orchard_gzclient.sh 11347 --software
```

如果车不动/界面一闪就没了，先看终端1里有没有 `gzserver` 退出（常见是 `~/.gazebo` 写权限问题导致 exit code 255）。  
本工程脚本已把 `GAZEBO_LOG_PATH` 指到工作区的 `.gazebo/`；如果你是手动 `roslaunch`，也可以先：

```bash
export GAZEBO_LOG_PATH=/mysda/w/w/lio_ws/.gazebo
```

## 6) 轨迹对比（不依赖 matplotlib）

回放结束后（或你中途 Ctrl+C 停止），用下面命令把 **回放轨迹(/odom)** 和 **rosbag 参考路径(json)** 做对比，输出 `SVG+JSON`：

```bash
cd /mysda/w/w/lio_ws && /usr/bin/python3 src/pcd_gazebo_world/scripts/plot_reference_vs_replay.py \
  --reference src/pcd_gazebo_world/maps/runs/rosbag_path.json \
  --replay trajectory_data/teb_odom.csv \
  --out trajectory_data/reference_vs_replay.svg \
  --report trajectory_data/reference_vs_replay.json
```

说明：这个脚本只用标准库（不会触发 venv 里 numpy/matplotlib 的版本冲突）。

## 7) 一键跑一套（TEB 三种模式 + PID）

```bash
cd /mysda/w/w/lio_ws && bash src/pcd_gazebo_world/tools/run_replay_suite.sh 11347 450
```

输出会在 `trajectory_data/replay_suite_YYYYMMDD_HHMMSS/` 下，包含每种模式的 `*_odom.csv`、`*_report.json`、`*_overlay.svg`。

可选：指定参考路径 JSON（例如使用 `--target-frame map` 导出的 `rosbag_path_map.json`）：

```bash
cd /mysda/w/w/lio_ws && bash src/pcd_gazebo_world/tools/run_replay_suite.sh 11347 450 src/pcd_gazebo_world/maps/runs/rosbag_path_map.json
```
