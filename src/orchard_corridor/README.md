# orchard_corridor (Stage 1/2/3/4/5/6)

Stage-1 pipeline for offline rosbag debugging:
- A: pointcloud_preprocess_node (ROI + voxel downsample)
- B: bev_occupancy_node (BEV OccupancyGrid)
- G: debug_export_node (PCD + BEV PNG export)

Stage-2 adds:
- C: centerline_node (distance transform + continuity constraints + Q + hold)

Stage-3 adds:
- D: boundary_from_centerline_node (scan left/right boundaries + corridor width)

Stage-4 adds:
- E: route_id_gate_node (confidence gating + stable ID)

Stage-5 adds (no move_base):
- F: controller_node (Pure Pursuit on local centerline -> /cmd_vel)

Stage-6 adds:
- Gazebo closed-loop wiring (Gazebo `/velodyne_points` -> corridor -> `/cmd_vel`) + map alignment + trajectory recording + report.

Default frame is `base_link_est`.

## Dependencies
- ROS (Noetic/Melodic)
- python3-numpy
- OpenCV (cv2) for PNG export and dilation
- tf2_ros, tf2_sensor_msgs (optional, if use_tf:=true)

## Build
```bash
cd /mysda/w/w/lio_ws
catkin_make
source devel/setup.bash
```

## Run with rosbag (Stage 1)
```bash
rosparam set use_sim_time true
roslaunch orchard_corridor debug_stage1.launch \
  points_topic:=/points_raw \
  output_frame:=base_link_est \
  launch_rviz:=true

rosbag play --clock /mysda/w/w/lio_ws/rosbags/2025-10-29-16-05-00.bag
```

Notes:
- This bag provides /points_raw and /liorl/deskew/cloud_deskewed (no /velodyne_points).
- If you want a cleaner vehicle frame, set points_topic:=/liorl/deskew/cloud_deskewed.
- This bag publishes base_link_est; set output_frame:=base_link_est and keep RViz Fixed Frame consistent.
- debug_stage1.launch publishes an optional static TF alias base_link_est -> base_link (enabled by default).

Torch/RandLA:
- The RandLA node uses the Python env pointed by `randla_env_prefix` (or `RANDLA_ENV_PREFIX`), default `/mysda/w/w/lio_ws/conda_envs/randla39`.
- To switch to another env, set `RANDLA_ENV_PREFIX=/path/to/your/env/prefix`.

## Stage 2 (centerline)
```bash
rosparam set use_sim_time true
roslaunch orchard_corridor debug_stage2.launch \
  points_topic:=/points_raw \
  launch_rviz:=true

rosbag play --clock /mysda/w/w/lio_ws/rosbags/2025-10-29-16-05-00.bag
```

## Stage 3 (boundaries)
```bash
rosparam set use_sim_time true
roslaunch orchard_corridor debug_stage3.launch \
  points_topic:=/points_raw \
  launch_rviz:=true

rosbag play --clock /mysda/w/w/lio_ws/rosbags/2025-10-29-16-05-00.bag
```

## Stage 4 (route_id gate)
```bash
rosparam set use_sim_time true
roslaunch orchard_corridor debug_stage4.launch \
  points_topic:=/points_raw \
  launch_rviz:=true

rosbag play --clock /mysda/w/w/lio_ws/rosbags/2025-10-29-16-05-00.bag
```
Notes:
- route_id_gate expects /route_id (Int32) + /route_conf (Float32). If missing, it publishes id=-1, valid=false.
- To run the built-in NN(2D-CNN)+ScanContext classifier and feed route_id_gate, set `use_nn_scancontext:=true` (and optionally `nn_cloud_topic:=/liorl/deskew/cloud_deskewed`).

## Stage 5 (controller -> /cmd_vel, no move_base)
```bash
rosparam set use_sim_time true
roslaunch orchard_corridor debug_stage5.launch \
  points_topic:=/points_raw \
  cmd_vel_topic:=/cmd_vel \
  controller_enabled:=false \
  launch_rviz:=true

rosbag play --clock /mysda/w/w/lio_ws/rosbags/2025-10-29-16-05-00.bag
```

Enable/disable:
```bash
rosservice call /corridor_controller/set_enabled "data: true"
rosservice call /corridor_controller/set_enabled "data: false"
```

Notes:
- controller_node tracks `/corridor_centerline` in the local frame (no odom/map needed).
- Ensure `/cmd_vel` has only 1 publisher to avoid conflicts.

## RandLA-Net integration
`debug_stage1.launch` can launch RandLA automatically (default use_randla:=true).
It remaps ~tree_cloud to /tree_points.

If you want to bypass RandLA and run BEV directly on ROI:
```bash
roslaunch orchard_corridor debug_stage1.launch \
  use_randla:=false \
  tree_points_topic:=/pc_roi
```

Gazebo note (no RandLA):
- Use a z-band filter to remove ground points, otherwise BEV becomes fully occupied and centerline quality collapses.
- Example:
```bash
roslaunch orchard_corridor debug_stage5.launch \
  points_topic:=/velodyne_points \
  output_frame:=base_link \
  use_randla:=false \
  tree_points_topic:=/pc_roi \
  preprocess_config:=$(find orchard_corridor)/config/pointcloud_preprocess_sim.yaml \
  bev_config:=$(find orchard_corridor)/config/bev_occupancy_sim.yaml \
  controller_enabled:=false
```

## Stage 6 (Gazebo closed-loop + report)

Prereq: reference path JSON in `map` frame (default: `src/pcd_gazebo_world/maps/runs/rosbag_path_map.json`).
If you need to regenerate it from rosbag:

```bash
python3 src/pcd_gazebo_world/scripts/rosbag_path_to_json.py \
  --bag rosbags/2025-10-29-16-05-00.bag \
  --topic /liorl/mapping/path \
  --target-frame map --tf-topic /tf \
  --out src/pcd_gazebo_world/maps/runs/rosbag_path_map.json
```

Run Stage6 (starts Gazebo GUI + corridor Stage5 + map->odom + trajectory recorder + RViz):
```bash
source devel/setup.bash
roslaunch orchard_corridor debug_stage6.launch
```

Headless / low-load run (recommended over SSH):
```bash
source devel/setup.bash
roslaunch orchard_corridor debug_stage6.launch \
  gui:=false launch_rviz:=false \
  gpu:=false lidar_samples:=220 pub_clock_frequency:=50
```

If `gzserver` dies with `bind: Address already in use`, pick another Gazebo master port:
```bash
PORT=$(python3 - <<'PY'
import random, socket
def free(p):
  s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
  try: s.bind(("0.0.0.0",p)); return True
  except OSError: return False
  finally: s.close()
for _ in range(2000):
  p=random.randint(40000,65000)
  if free(p): print(p); break
PY
)
roslaunch orchard_corridor debug_stage6.launch gazebo_master_uri:=http://127.0.0.1:$PORT
```

Enable/disable controller:
```bash
rosservice call /corridor_controller/set_enabled "data: true"
rosservice call /corridor_controller/set_enabled "data: false"
```

Auto switch (corridor ↔ PID, recommended for full loop):
```bash
source devel/setup.bash
roslaunch orchard_corridor debug_stage6.launch \
  use_cmd_mux:=true \
  use_pid_follower:=false \
  controller_enabled:=false
```

Mux tuning knobs (most useful when corridor picks the wrong branch at intersections):
- `min_mode_dwell_s` (default 2.0): minimum time between switches (reduces chattering).
- `use_heading_guard` (default true): require robot heading to be consistent with the reference path.
- `heading_enter` / `heading_exit` / `heading_hard` (rad): enter/exit/hard-fail thresholds.
- `use_corridor_path_heading_guard` (default false): require corridor centerline direction (in base frame) to match reference direction, avoids entering wrong branch early.

Outputs (written on shutdown / Ctrl-C):
- CSV: `/mysda/w/w/lio_ws/trajectory_data/closedloop_odom_map.csv` (odom transformed to `map`)

Generate the SVG+JSON report:
```bash
python3 src/pcd_gazebo_world/scripts/plot_reference_vs_replay.py \
  --reference src/pcd_gazebo_world/maps/runs/rosbag_path_map.json \
  --replay /mysda/w/w/lio_ws/trajectory_data/closedloop_odom_map.csv \
  --out /mysda/w/w/lio_ws/trajectory_data/closedloop_vs_rosbag.svg \
  --report /mysda/w/w/lio_ws/trajectory_data/closedloop_vs_rosbag_report.json
```

Common overrides:
- `world_name:=...` (default uses `orchard_from_pcd_validated_by_bag.world`)
- `reference_path_file:=...`
- `teleport_to_reference_start:=false` (if you want to spawn manually)
- `publish_map_to_odom:=false` (if you already have correct map->odom)

## Export outputs
Services:
```bash
rosservice call /debug_export/save_tree_pcd_once "{}"
rosservice call /debug_export/save_tree_bev_once "{}"
```

Files are written to output_dir (default /mysda/w/w/lio_ws/output/debug_stage1):
- tree_points_<stamp_ns>_<seq>_<counter>.pcd
- tree_bev_<stamp_ns>_<seq>_<counter>.png
- occ_bev_<stamp_ns>_<seq>_<counter>.png (if /bev_occ available)

## Topics
- /pc_roi (sensor_msgs/PointCloud2): ROI-filtered cloud
- /tree_points (sensor_msgs/PointCloud2): RandLA tree points (sampled)
- /bev_occ (nav_msgs/OccupancyGrid): BEV occupancy grid
- /bev_occ_raw (nav_msgs/OccupancyGrid): before dilation
- /corridor_centerline (nav_msgs/Path): centerline path
- /corridor_quality (std_msgs/Float32): centerline quality
- /corridor_boundary_markers (visualization_msgs/MarkerArray): left/right boundaries
- /corridor_width (std_msgs/Float32): min/mean width (see config)
- /route_id_stable (std_msgs/Int32): gated route id
- /route_id_valid (std_msgs/Bool): route id validity

## RViz
- Stage 1: rviz/debug_stage1.rviz (Fixed Frame = base_link_est)
- Stage 2: rviz/debug_stage2.rviz (Fixed Frame = base_link_est)
- Stage 3: rviz/debug_stage3.rviz (Fixed Frame = base_link_est)
