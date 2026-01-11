# orchard_tree_tracker

ROS1（rospy）果树实例化 + MOT 跟踪 + 在线拟合节点。

## 输入

- `sensor_msgs/PointCloud2`：fields 至少包含 `x,y,z,label`
- 规则：`label==0` 视为果树点，其它值忽略；会过滤 `NaN`

默认订阅 `~input_topic:=/segmented_points`。

## 输出

- `visualization_msgs/MarkerArray`：`~markers`（默认 `/tree_markers`）
  - sphere：中心
  - cylinder：冠幅（直径）
  - text：`tree_id` + 高度/冠幅
- `std_msgs/String`：`~json_out`（默认 `/tree_detections_json`）
  - JSON 形如：`[{"id":1,"cx":...,"cy":...,"height":...,"crown":...,"conf":...}, ...]`
- `std_msgs/String`：`~row_fit_json`（默认 `/tree_row_fit_json`；可设置为空字符串禁用）
  - JSON 形如：`{"frame_index":0,"stamp":...,"left":{...},"right":{...}}`
- CSV：`~csv_path`（可选）
  - `timestamp,id,cx,cy,height,crown,conf,point_count`

## 参数（核心）

- ROI：`roi_x_min/max`, `roi_y_min/max`, `roi_z_min/max`
- voxel：`voxel_size`（默认 `0.03`）
- 2D 栅格：`cell_size`（默认 `0.10`），`grid_T`（默认 `5`）
- MOT：`gate_distance`（默认 `0.30`），`max_missed`（默认 `10`）
- 拟合：滑窗 `K`（默认 `20`），EMA `ema_alpha`（默认 `0.4`）
- 日志：`log_summary_interval`（默认 `1.0` 秒；设为 `0` 可关闭）
- 输出：`publish_missed`（默认 `false`；`false`=只发布本帧看见的树 `point_count>0`；`true`=也发布 missed 的 tracks）
- 行拟合：`row_fit_min_points`（默认 `3`），`row_fit_inlier_dist`（默认 `0.20`），`row_fit_iters`（默认 `2`）
  - 历史拟合（更稳）：`row_fit_history_frames`（默认 `20`，使用最近 N 帧 detections 拟合左右行），`row_fit_min_conf`（默认 `0.0`，仅用于行拟合过滤）
  - **车在运动时更推荐**：`row_fit_fixed_frame`（默认空；例如 `map` 或 `odom_est`）启用 TF 运动补偿，把历史 detections 先累积到固定坐标系再投回当前帧拟合，能显著减少“拖影/乱拟合”
    - `row_fit_fixed_frame_timeout`（默认 `0.05` 秒）：TF 查询超时，bag/机器慢时可适当调大
- 论文出图（默认关闭）：`export_dir`（默认空），`export_every_n`（默认 `0`），`export_max_frames`（默认 `0`）
  - BEV 分辨率/画布：`bev_res`（默认 `0.0` 自动），`bev_width_px`/`bev_height_px`（默认 `0`）
  - 叠加开关：`export_draw_ids`（默认 `true`），`export_draw_crowns`（默认 `true`）

## 使用（rosbag 回放）

### 一键回放（本仓库示例 bag）

只跑本仓库自带的示例 bag（已内置 bag 路径与 input_topic）：

```bash
roslaunch orchard_tree_tracker fruit_tree_tracker_replay.launch
```

### 手动回放（任意 bag）

1) 启用仿真时钟：

```bash
rosparam set use_sim_time true
```

2) 播放 bag（带 /clock）：

```bash
rosbag play --clock your.bag
```

3) 启动节点：

```bash
roslaunch orchard_tree_tracker fruit_tree_tracker.launch input_topic:=/segmented_points csv_path:=/tmp/tree_detections.csv
```

可选：如果 `label` 字段名不同，可设置 `label_field:=xxx`。

查看 JSON 输出：

```bash
rostopic echo /tree_detections_json
```

查看左右行拟合输出：

```bash
rostopic echo /tree_row_fit_json
```

## RViz 可视化

1) 打开 RViz
2) Add → `MarkerArray`
3) Topic 选择 `/tree_markers`

## 论文可复现 BEV 出图（PNG）

### 坐标系与图像方向（严格约定）

- 点云坐标：+x 前方，+y 左侧（+z 向上）
- BEV 图像方向：
  - 图像上方 = +x（越靠上越远）
  - 图像左侧 = +y（越靠左越偏左）

像素映射（基于 ROI 与 `bev_res`，单位 m/px）：
- `u = (roi_y_max - y) / bev_res`（列；左侧=+y）
- `v = (roi_x_max - x) / bev_res`（行；上方=+x）

### 一键回放 + 出图

```bash
# Terminal 1：导出目录非空即开启导出（会写 run_meta.json + frame_%06d.png）
roslaunch orchard_tree_tracker fruit_tree_tracker_replay.launch \
  export_dir:=/tmp/orchard_bev_run \
  export_every_n:=10 \
  export_max_frames:=200
```

产物（`export_dir`）：
- `run_meta.json`：记录坐标系说明与所有关键参数（首次启动写入；若已存在不会覆盖）
- `frame_000000.png`, `frame_000010.png`, ...：每张图左上角叠加 `frame_%06d.png` 与 `header.stamp.to_sec()`

### 调分辨率/画布（只改参数即可复现）

```bash
# Terminal 1：指定分辨率（m/px），图像尺寸由 ROI 自动推算
roslaunch orchard_tree_tracker fruit_tree_tracker_replay.launch \
  export_dir:=/tmp/orchard_bev_res002 \
  bev_res:=0.02 \
  export_every_n:=10
```

```bash
# Terminal 1：指定固定画布尺寸，并反推 bev_res（ROI 会落在画布左上角区域）
roslaunch orchard_tree_tracker fruit_tree_tracker_replay.launch \
  export_dir:=/tmp/orchard_bev_fixed \
  bev_width_px:=800 \
  bev_height_px:=800 \
  export_every_n:=10
```

## 不依赖 ROS 的 test_mode

```bash
python3 src/orchard_tree_tracker/scripts/fruit_tree_tracker_node.py --test_mode --verbose
```
