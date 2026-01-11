# navigation_experiments（论文实验复现模板）

目标：用**同一条参考路线**（来自 LIO‑SAM/LIORL 的 `nav_msgs/Path`）在实车上重复跑多次，比较不同“分段/FSM + 局部运动生成（PID/TEB）+ 感知退化”的效果，并产出论文可用的指标表格。

本目录提供两类工具：
- 路线准备：从参考 bag 导出 `ref_path.csv`（含弧长 s、yaw、曲率 κ）。
- 结果评估：从每次实验的运行 bag 计算整体/分段（直行/左转/右转）误差统计与成功率。

---

## 1) 建议的实验组（对比/消融）

你论文里列的 5 组可以这样落地（保持变量控制，见第 2 节）：

1. **全程 PID（tracking-only）**：不启用 TEB/move_base，仅用直行/转弯统一的 PID 跟踪同一参考路线（可加统一的安全停止层）。
2. **全程 TEB（TEB-only）**：全程使用 `move_base + TebLocalPlannerROS`，不做 FSM 分段/切换。
3. **无宏观语义（geometry switching）**：不使用“预标注直/左/右标签”，改用几何规则（曲率阈值 + 滞回 + 一致性 N）切换。
4. **无微观集合（no semantic set）**：不使用语义分割得到的树干点集，直接对原始点云（保持同样 ROI/去地面/滤波）做几何拟合/识别。
5. **一致性参数 N**：固定其他设置，比较 `N=1/3/5` 对“切换延迟 vs 抖动次数 vs 跟踪误差”的影响。

注意：`PID` 是“路径跟踪/控制”，`TEB` 是“局部规划/速度优化”。论文表述建议用“分段局部运动生成策略”而不是都叫“路径规划算法”。

---

## 2) 变量控制（确保论文对比公平）

强烈建议在论文里明确以下控制项：
- 同一地图/同一定位：LIO‑SAM/LIORL 配置固定；`map` frame 统一。
- 同一路线：同一 `ref_path.csv`（来源 bag 固定、下采样间距固定）。
- 同一硬件、同一电池/载荷、同一场地（尽量同一时段）。
- 同一速度上限/加速度限制（例如统一加 `velocity_smoother`），避免“快的自然误差大/更危险”。
- 每组每条路线 **重复 ≥ 5 次**（顺序随机化），报告均值 ± 方差/置信区间，并给出成功率。

---

## 3) 路线准备（从参考 bag 导出 ref_path.csv + 标签模板）

1) 录一条“参考路线 bag”（人工遥控或基线算法均可），要求包含 `nav_msgs/Path`：
- 常见：`/lio_sam/mapping/path` 或 `/liorl/mapping/path`

2) 导出参考路线 CSV（含弧长 s、曲率 κ）：
```bash
python3 tools/navigation_experiments/export_reference_path.py \
  rosbags/ref_route.bag \
  --min-dist 0.5 \
  --out-csv maps/ref_route/ref_path.csv \
  --out-labels maps/ref_route/ref_labels.yaml
```

3) 在 `maps/ref_route/ref_labels.yaml` 里填入你“提前标好的”直/左/右标签分段：
- 以弧长 `s` 为单位（米），写多个 `segments`：
```yaml
version: 1
default_label: straight
segments:
  - start_s: 0.0
    end_s: 32.5
    label: straight
  - start_s: 32.5
    end_s: 41.2
    label: left
  - start_s: 41.2
    end_s: 80.0
    label: straight
```

---

## 4) 每次实验建议录哪些 topic

最少（用于轨迹对齐与误差计算）：
- `nav_msgs/Path`：`/lio_sam/mapping/path` 或 `/liorl/mapping/path`（执行轨迹）

建议额外录（用于论文分析/复现）：
- `/cmd_vel`（控制输出）
- `/tf`、`/tf_static`（若你需要从 TF 重建轨迹/检查 frame）
- `/move_base/status`、`/move_base/result`（若你用 move_base）
- TEB 调试：`/move_base/TebLocalPlannerROS/teb_poses`、`.../local_plan`、`.../teb_markers`（可选）
- FSM 输出：例如 `/fsm/mode`（`std_msgs/String`），用于统计切换次数与切换延迟（第 6 节）

---

## 5) 运行与评估（单次试验）

你可以手动 `rosbag record`，也可以用本目录的录包封装脚本：
```bash
bash tools/navigation_experiments/run_recorded.sh \
  rosbags/runs/run_teb_only_01.bag \
  --topics tools/navigation_experiments/topics_minimal.txt \
  -- roslaunch bag_route_replay replay_from_bag.launch bag_path:=/abs/path/to/ref_route.bag
```

录完之后评估：
```bash
python3 tools/navigation_experiments/evaluate_run.py \
  --ref-csv maps/ref_route/ref_path.csv \
  --labels maps/ref_route/ref_labels.yaml \
  --run-bag rosbags/runs/run_teb_only_01.bag \
  --method teb_only \
  --out-dir maps/nav_eval/run_teb_only_01
```

---

## 6) 一致性参数 N（切换延迟/抖动）怎么统计

如果你的 FSM 在运行时发布 `std_msgs/String` 模式（例如 `/fsm/mode`，值为 `straight/left/right`），在录包里保留该 topic。

评估脚本支持可选的 `--mode-topic /fsm/mode`，会输出：
- `mode_switches`：模式切换次数（抖动的直接指标）
- `boundary_delay_m`：每个切换边界相对真值边界的弧长延迟（米）

---

## 7) 汇总多次实验（生成论文表格）

```bash
python3 tools/navigation_experiments/aggregate_results.py \
  --inputs maps/nav_eval/*/metrics.json \
  --out-csv maps/nav_eval/summary.csv
```

