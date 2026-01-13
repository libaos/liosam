# Roadmap

本仓库是一个聚合型 ROS1（catkin）工作区，目标是把 “能跑” 和 “好复现” 做到开箱即用。下面里程碑以**可交付物**为导向，方便按阶段推进。

## Milestone 1 — MVP（0.x）：3 分钟 Quick Start

目标：第一次进入仓库的人，在已有 ROS 环境的前提下，3 分钟内能启动一个可视化 Demo，并且知道下一步去哪看文档/提问题。

可交付物：

- README：包含 badges/TOC/Quick Start、最小故障排查、目录说明
- `.github/`：Issue/PR 模板齐全，能引导用户给出复现步骤与环境信息
- GitHub Actions：最小 CI（例如语法/格式/文档检查）可跑通，用于 build badge
- Quick Start 路径明确：以 `pcd_gazebo_world` 的 Gazebo 仿真作为默认入口（可无编译运行）

## Milestone 2 — 稳定版（1.0）：可复现构建 + 最小验证闭环

目标：在“推荐环境”下，能够稳定构建并跑通至少 1 条端到端链路（仿真/导航/回放其一），并且 CI 能覆盖关键路径。

可交付物：

- 环境矩阵：明确支持/验证的 Ubuntu/ROS/Gazebo 版本（README 里可查）
- 依赖收敛：提供 `rosdep` 安装说明与常见依赖列表（按包分组）
- 构建指南：`catkin_make` / `catkin_tools` 两种方式的推荐用法与常见坑
- CI（可选分层）：
  - 快速：lint/语法/文档
  - 构建：至少在 1 个 ROS distro 上做 `catkin_make`（允许跳过重依赖包）
- Demo 验证：提供一条“成功截图/输出”作为验收标准（例如 Gazebo world + RViz TF 正常）

## Milestone 3 — 增强（1.x）：体验提升 + 基准与数据链路

目标：让仓库更像一个可长期维护的项目，而不仅是个人工作区。

可交付物：

- 一键环境：提供 Docker/DevContainer（可选）或脚本化安装（可选）
- 可选数据集：用 Git LFS / Release / 外链的方式提供小体量样例（避免污染仓库）
- Benchmarks：对回放误差、规划效果等提供可复现评测脚本与报告格式
- 更清晰的模块边界：按“仿真/建图/定位/导航/感知”整理文档入口与示例命令
- （可选）ROS2 迁移评估：列出阻塞点与分阶段迁移策略

