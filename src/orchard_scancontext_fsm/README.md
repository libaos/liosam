# orchard_scancontext_fsm

基于 **ScanContext** 的“直行/左转/右转”模式识别辅助节点，面向论文第 5 章实验复现。

该包的定位很简单：订阅点云（例如 `/liorl/deskew/cloud_deskewed`），用 ScanContext 的 **列移位**估计相对 yaw 变化（在匹配距离足够小的前提下），并发布 `/fsm/mode`（`straight|left|right`）。

## 1) 只跑模式识别节点

```bash
roslaunch orchard_scancontext_fsm scancontext_fsm.launch
```

常用参数（launch args）：
- `cloud_topic`：点云话题（默认 `/liorl/deskew/cloud_deskewed`）
- `mode_topic`：输出模式话题（默认 `/fsm/mode`）
- `yaw_rate_threshold`：判定转弯的角速度阈值（rad/s）
- `consistency_n`：一致性 N（连续 N 次同一候选模式才切换）

## 2) 一键用 rosbag 复现（可选录包）

```bash
roslaunch orchard_scancontext_fsm ch5_from_bag.launch \
  bag:=/mysda/w/w/lio_ws/rosbags/2025-10-29-16-05-00.bag \
  rate:=1 \
  record:=true \
  out_bag:=/mysda/w/w/lio_ws/rosbags/runs/ch5_sc_fsm.bag
```

这个 launch 会：
- `rosbag play --clock` 播放 bag（默认播 `/liorl/deskew/cloud_deskewed /tf /liorl/mapping/path`）
- 启动 `scancontext_fsm_node.py` 发布 `/fsm/mode`
- 可选启动 `rosbag record` 把 `/fsm/mode` 和 `/liorl/mapping/path` 录到新 bag（用于 `tools/navigation_experiments/evaluate_run.py`）

如果你不方便起 `roslaunch/roscore`（或运行环境受限），也可以直接离线生成一个“带 `/fsm/mode` 的小 bag”：

```bash
python3 src/orchard_scancontext_fsm/scripts/annotate_bag_with_fsm_mode.py \
  --in-bag rosbags/2025-10-29-16-05-00.bag \
  --out-bag rosbags/runs/ch5_sc_fsm_offline.bag \
  --start 1761725100.83 --duration 300 \
  --copy-topics /liorl/mapping/path
```

## 3) 和果树分割+拟合一起跑（可视化用）

`ch5_from_bag.launch` 默认会一起启动：
- `orchard_row_mapping` 分割+行拟合：`/orchard_segmentation/tree_cloud`、`/orchard_segmentation/row_markers`
- `orchard_row_mapping` 流式聚类画圆：`/orchard_tree_clusters/tree_circles`（默认只取树干高度带 `z∈[0.7,1.3]`）
- ScanContext 模式识别：`/fsm/mode`

常用开关（launch args）：
- `enable_orchard_segmentation:=true|false`
- `enable_tree_clustering:=true|false`
- `tree_cluster_z_min` / `tree_cluster_z_max` / `tree_cluster_cell_size` / `tree_cluster_min_points`

## 4) 接到 move_base + TEB（跑真实实验）

如果你用的是 `move_base + teb_local_planner/TebLocalPlannerROS`，推荐再启动一个“TEB 参数切换器”：
订阅 `/fsm/mode`，在 `straight` 和 `left/right` 之间动态调整 TEB 的速度/加速度参数（不用重启 move_base）。

```bash
roslaunch orchard_teb_mode_switcher teb_mode_switcher.launch \
  teb_server:=/move_base/TebLocalPlannerROS
```
