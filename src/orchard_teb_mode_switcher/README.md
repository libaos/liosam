# orchard_teb_mode_switcher

把 `orchard_scancontext_fsm` 输出的 `/fsm/mode`（`straight|left|right`）接到 **move_base + TebLocalPlannerROS**：
在运行时用 `dynamic_reconfigure` 切换 TEB 参数（直行/转弯两套）。

## 用法

1) 先确保你的系统里已经在跑 `move_base`（且 `base_local_planner=teb_local_planner/TebLocalPlannerROS`）。

2) 启动模式切换（默认 teb server：`/move_base/TebLocalPlannerROS`）：
```bash
roslaunch orchard_teb_mode_switcher teb_mode_switcher.launch
```

3) 确保 `/fsm/mode` 在发布（例如启动 `orchard_scancontext_fsm`）：
```bash
roslaunch orchard_scancontext_fsm scancontext_fsm.launch
```

## 参数
- `mode_topic`：默认 `/fsm/mode`
- `teb_server`：动态参数服务器名，默认 `/move_base/TebLocalPlannerROS`
- `profiles/straight`、`profiles/turn`：要写入 dynamic_reconfigure 的字段字典（你可以加更多 teb 参数）

