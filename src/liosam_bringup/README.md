# liosam_bringup

本包只做一件事：提供工作区级别的统一入口 `launch/`，把常用 demo 启动方式集中在一个地方。

## Launches

- `orchard_sim_headless.launch`: 启动 `pcd_gazebo_world/orchard_sim.launch`，默认 `gui:=false`，适合远程/无显示环境。

## Usage

```bash
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash

roslaunch liosam_bringup orchard_sim_headless.launch
```
