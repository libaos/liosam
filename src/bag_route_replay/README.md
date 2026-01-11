# bag_route_replay

把 rosbag 里记录的 `nav_msgs/Path`（例如 LIO-SAM/LIORL 的轨迹）转换成一串 `move_base` 目标点并依次发送，让机器人在现实环境里“再跑一遍”。

## 用法

1) 编译并 source：
```bash
cd /mysda/w/w/lio_ws
catkin_make
source devel/setup.bash
```

2) 确保现实系统里已经在跑定位 + 导航（有 `/move_base` action server），然后：
```bash
roslaunch bag_route_replay replay_from_bag.launch bag_path:=/path/to/xxx.bag
```

常用参数：
- `path_topic`：不填会自动找 `/lio_sam/mapping/path` 或 `/liorl/mapping/path`
- `min_dist`：航点间距（米），建议 0.5~2.0
- `start_nearest`：从离机器人最近的航点开始（默认 `true`）
- `robot_frame`：默认 `base_link_est`（按你现在工程的 TF 命名）
- `goal_frame`：不填则使用 bag 里 Path 的 `header.frame_id`；如果填了且和 bag 不同，会用当前 TF 把航点坐标从 source frame 变换到 `goal_frame`
- `transform_timeout`：等待 TF 的超时（秒）

## 注意
- 这是“按航点重新规划”，并不保证和 bag 原始轨迹逐帧完全重合。
- 建议先在仿真/空旷区域验证，确认 `cmd_vel`、急停等安全链路正常。
