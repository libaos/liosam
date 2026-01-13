# Licenses (Important)

本仓库是一个聚合型 ROS1 工作区：`src/` 下包含多个来源不同的 package/第三方组件。**因此仓库整体不一定只有一种许可证**。

在使用/开源/二次分发前，请务必按包核对许可证与来源。

## How to check

对每个 ROS package（有 `package.xml` 的目录）：

1) 查看 `package.xml` 里的 `<license>` 字段  
2) 查看该目录下是否有 `LICENSE*` / `COPYING*` 文件  

你也可以用命令快速扫描：

```bash
find src -name package.xml -print
find src -iname "LICENSE*" -o -iname "COPYING*"
```

## Known license files in this repo

目前 `src/` 下已包含的许可证文件（不代表全部包都已标注完善）：

- `src/psolqr_local_planner/LICENSE`
- `src/lio_sam_move_base_tutorial/teb_local_planner/LICENSE`
- `src/lio_sam_move_base_tutorial/local-planning-benchmark/LICENSE`
- `src/lio_sam_move_base_tutorial/robot_gazebo/LICENSE`
- `src/lio_sam_move_base_tutorial/robot_gazebo/liorf/LICENSE`
- `src/lio_sam_move_base_tutorial/robot_gazebo/liorl/LICENSE`
- `src/lio_sam_move_base_tutorial/robot_gazebo/warehouse_simulation_toolkit/LICENSE`

另外也有一些包仅在 `package.xml` 里声明许可证（例如 `src/pcd_gazebo_world/package.xml`）。

## Contributing & third‑party code

- 新增或引入第三方代码时，请保留来源信息（upstream 链接/commit）并明确许可证
- 新增 ROS package 时，请在 `package.xml` 里填写 `<license>`，必要时在目录内放置 `LICENSE` 文件

