# Contributing

欢迎贡献！本仓库更像一个聚合型 ROS 工作区，因此我们优先保证：

1) **新手能跑起来**（文档/示例命令/最小排错清晰）  
2) **改动可复现**（环境信息完整、最小修改、可验证）

## Before you start

- 请先读一遍 `README.md` 的 Quick Start，确认你能跑通最小 Demo。
- 如果你的改动会影响运行路径，请同步更新 README/相关包的 README。

## Filing an Issue

请使用 GitHub 的 Issue 模板，并尽量提供：

- 复现步骤（命令、launch 文件、参数）
- 预期行为 vs 实际行为
- 日志（`roslaunch` 输出、`~/.ros/log`、Gazebo/RViz 的关键报错）
- 环境信息（Ubuntu/ROS/Gazebo/显卡驱动/仓库 commit）

**提示**：你可以用下面命令快速拿到 commit：

```bash
git rev-parse --short HEAD
```

## Pull Requests

### Scope

本仓库优先接受以下类型的 PR：

- 文档完善、示例命令、目录说明、FAQ
- 工作流/模板改进（CI、Issue/PR 模板）
- 不改变整体架构的小修复（让 Quick Start 或示例能跑通）

尽量避免在一个 PR 里做大范围重构或引入大型二进制/数据文件。

### Checklist

提交 PR 前请自查：

- [ ] 描述清楚“为什么改”和“怎么验证”
- [ ] 不提交 rosbags/大地图/conda 环境等（见 `.gitignore`）
- [ ] 涉及命令/launch 的改动已更新到 README 或相应包的 README

## Local sanity checks

不依赖 ROS 的快速检查：

```bash
python3 -m compileall -q src tools
```

如果你要验证 catkin 构建（可选，依赖你的 ROS 环境与依赖安装情况）：

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
rosdep install --from-paths src --ignore-src -r -y

# choose one:
catkin_make
# or:
catkin init && catkin build
```

## Repository conventions

- 以 `src/` 为主（ROS package）；仓库级脚本放 `tools/`
- 新增 demo/入口尽量给出一条“复制即用”的命令
- 如果引入第三方代码，请保留来源信息并明确许可证（见 `LICENSES.md`）

