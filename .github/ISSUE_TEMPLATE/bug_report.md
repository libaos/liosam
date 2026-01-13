---
name: Bug report
about: Report something that should work but doesn't
title: "[Bug] "
labels: bug
assignees: ""
---

## Summary

一句话描述问题是什么。

## Steps to reproduce

1.
2.
3.

### Command / launch file

把你实际运行的命令贴在这里（建议不要省略参数）：

```bash
# e.g.
# source /opt/ros/noetic/setup.bash
# export ROS_PACKAGE_PATH="$PWD/src:${ROS_PACKAGE_PATH}"
# roslaunch pcd_gazebo_world orchard_sim.launch enable_robot_state_publisher:=true
```

## Expected behavior

你期望发生什么？

## Actual behavior

实际发生了什么？（包含报错信息）

## Logs / screenshots

- `roslaunch`/终端输出：
- `~/.ros/log` 或 `ROS_LOG_DIR` 中的关键日志：
- Gazebo/RViz 截图（如有）：

> 提示：如果你在仓库根目录运行，日志可能在 `.ros_log/`（取决于你的启动方式）。

## Environment

请尽量填写完整（这会显著提升定位速度）：

- OS: (e.g. Ubuntu 20.04)
- ROS distro: (e.g. Noetic)
- Gazebo: (e.g. Classic 11)
- GPU / driver: (optional, but important for gzclient crash)
- Install method: (apt/source/docker)
- Repo commit: (`git rev-parse --short HEAD`)

## Additional context

你已经尝试过哪些排查？有没有 workaround？

