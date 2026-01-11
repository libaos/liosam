# continuous_navigation

- 两点导航（RViz 点两次 `2D Nav Goal`）：`roslaunch continuous_navigation simple_two_points_navigator.launch`
- 多航点自动巡航（从文件读航点）：`roslaunch continuous_navigation waypoint_navigator.launch num_manual_goals:=0 cycle_waypoints:=false waypoints_file:=/abs/path/to/waypoints.yaml`
- 从 rosbag 的 `nav_msgs/Path` 生成航点文件：`tools/bag_path_to_waypoints.py rosbags/xxx.bag --out src/continuous_navigation/config/waypoints_from_bag.yaml`（默认自动识别 `/lio_sam/mapping/path` 或 `/liorl/mapping/path`，也可用 `--topic` 指定）
