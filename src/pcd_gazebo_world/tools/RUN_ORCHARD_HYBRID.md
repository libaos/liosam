# Orchard Hybrid Replay（直线 PID / 转弯 TEB）

目标：Gazebo 里复现 rosbag 路线时，对比三种策略：

1) **纯 PID**：直接跟随参考路径（不做规划）
2) **纯 TEB**：move_base + TEB 本地规划（用一串 waypoint goals 走完整条路线）
3) **混合**：直线用 PID（更贴参考、更稳定），转弯用 TEB（更抗约束/避障）

## 0) 参考路径

默认参考路径 JSON：
- `src/pcd_gazebo_world/maps/runs/rosbag_path.json`（frame_id=map）

## 1) 纯 PID

```bash
cd /mysda/w/w/lio_ws
bash src/pcd_gazebo_world/tools/run_orchard_pid_server.sh 11347
```

输出轨迹 CSV（可用 `RECORD_CSV` 覆盖）：
- `trajectory_data/pid_odom.csv`

## 2) 纯 TEB

直接用已有说明：
- `src/pcd_gazebo_world/tools/RUN_ORCHARD_TEB.md`

最常用启动：

```bash
cd /mysda/w/w/lio_ws
bash src/pcd_gazebo_world/tools/run_orchard_teb_server.sh 11347
```

## 3) 混合（直线 PID / 转弯 TEB）

启动（默认 `HYBRID_MODE=auto`）：

```bash
cd /mysda/w/w/lio_ws
HYBRID_MODE=auto bash src/pcd_gazebo_world/tools/run_orchard_hybrid_server.sh 11347
```

可选：强制只用某一种输出（方便做 A/B 对比）：

```bash
HYBRID_MODE=pid bash src/pcd_gazebo_world/tools/run_orchard_hybrid_server.sh 11347
HYBRID_MODE=teb bash src/pcd_gazebo_world/tools/run_orchard_hybrid_server.sh 11347
```

输出轨迹 CSV（可用 `RECORD_CSV` 覆盖）：
- `trajectory_data/hybrid_odom.csv`

### 混合切换规则（AUTO）

节点：`src/pcd_gazebo_world/scripts/hybrid_cmd_vel_mux.py`

- 在 `/reference_path_odom` 上计算 lookahead 距离内的航向变化 `abs(delta_yaw)`
- `abs(delta_yaw) >= turn_enter_yaw` → 选 TEB
- `abs(delta_yaw) <= turn_exit_yaw` → 选 PID
- 中间区间保持上一状态（带滞回 + `min_hold_s` 防抖）

需要调参时，直接 `roslaunch ... orchard_hybrid_replay.launch` 传：
- `hybrid_lookahead_dist_m`
- `hybrid_turn_enter_yaw`
- `hybrid_turn_exit_yaw`
- `hybrid_min_hold_s`

