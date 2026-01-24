# orchard_row_mapping

ROS 节点：订阅原始点云 (`sensor_msgs/PointCloud2`)，调用 `noslam` 中的 RandLA-Net 分割代码，将果树点与非果树点区分开，并自动拟合出左右两条果树行（道路边界）。

## 主要特性
- 复用 `/mysda/w/w/RandLA-Net-pytorch/noslam` 的 RandLA-Net 推理代码（本包内已 vendored 所需模块）。
- 任意原始点云话题，默认 `/velodyne_points`。
- 采用 `num_points`（默认 16384）对输入点进行随机采样/重复，以满足模型输入；推理后发布彩色点云话题 `~segmented_cloud`。
- 根据被判定为果树的点，用中值法拆分左右行并做 PCA 直线拟合，输出 `~row_markers`（RViz LINE_STRIP）。

## 依赖
- ROS (Noetic/Melodic)
- `pcl_ros`, `sensor_msgs`, `visualization_msgs`
- PyTorch（与 noslam 项目相同版本即可）
- RandLA-Net 权重文件，例如 `/mysda/w/w/RandLA-Net-pytorch/noslam/checkpoints/best_model.pth`

推荐（统一环境）：
- 使用 conda 环境 `conda_envs/randla39`（推荐：`source tools/ros_py39/setup.bash`）
- 统一说明见：`docs/conda_env.md`

## GPU 说明
`orchard_segmentation_node.py` 会在 `use_gpu` 且 `torch.cuda.is_available()` 为 true 时启用 CUDA。相关 launch 已增加 `randla_env_prefix`（或环境变量 `RANDLA_ENV_PREFIX`）用于指定 Python 环境前缀，默认 `/mysda/w/w/lio_ws/conda_envs/randla39`。

## 目录说明
```
orchard_row_mapping/
├── CMakeLists.txt
├── package.xml
├── README.md
├── config/default.yaml          # launch 默认参数
├── launch/orchard_row_mapping.launch
├── scripts/orchard_segmentation_node.py
└── orchard_row_mapping/segmentation
    ├── inference.py             # 采样与推理 helper（来自 noslam）
    ├── model_loader.py          # RandLA-Net 模型加载
    └── vendor/...               # Fixed4DRandLANet 及依赖
```

## 启动示例
```bash
cd /mysda/w/w/lio_ws
catkin_make
source devel/setup.bash
roslaunch orchard_row_mapping orchard_row_mapping.launch \
  config:=`rospack find orchard_row_mapping`/config/default.yaml
```
如需自定义权重路径、点云话题等，修改 `config/default.yaml` 或在 launch 中覆盖参数。

## 更好看的行拟合（推荐用于 liorf/LIO-SAM）
如果你在跑 `liorf`，建议直接订阅去畸变点云并开启稳一点的拟合参数：
```bash
roslaunch orchard_row_mapping orchard_row_mapping.launch \
  config:=`rospack find orchard_row_mapping`/config/liorf_pretty.yaml
```

## 最稳定（有先验地图 + liorl 定位）
如果你每次都加载同一张 `GlobalMap.pcd` 并用 `liorl` 做定位，最稳的做法是：从全局地图提取“果树行先验”，运行时直接在 `map` 坐标系发布左右行线/中线（几乎不抖）。
```bash
roslaunch orchard_row_mapping orchard_row_prior.launch \
  config:=`rospack find orchard_row_mapping`/config/liorl_prior.yaml
```
默认订阅 `/liorl/localization/global_map`；也可以在 `liorl_prior.yaml` 里直接填 `pcd_path` 读取 PCD。

## 导出点云到 PCD（配合 CloudCompare 手工微调）
- 直接编辑先验地图：打开 `/mysda/w/w/lio_ws/maps/GlobalMap.pcd` → 裁剪/去地面/删杂点 → 另存为 `TreeMap_clean.pcd`。
- 想要“树点地图”（推荐）：用 `liorl` 的 TF 把实时分割到的树点累积到 `map` 坐标系，并保存成单个 PCD：
```bash
roslaunch orchard_row_mapping orchard_tree_map_liorl.launch
# bag 播放完后 Ctrl-C，会自动保存到 /mysda/w/w/lio_ws/maps/TreeMap_auto.pcd
```
- 也可以把分割后的“果树点”导出：启动 `orchard_row_liorl_pretty.launch` 后运行（本环境可能没有 `rosrun`，直接用可执行文件路径）：
```bash
mkdir -p /mysda/w/w/lio_ws/maps
/opt/ros/noetic/lib/pcl_ros/pointcloud_to_pcd \
  input:=/orchard_segmentation/tree_cloud \
  _filename:=/mysda/w/w/lio_ws/maps/tree_cloud.pcd \
  _fixed_frame:=map _binary:=true _compressed:=false
```
然后用 CloudCompare 打开 `tree_cloud.pcd` 继续裁剪/清理，再把清理后的 PCD 作为 `orchard_row_prior.launch` 的 `pcd_path` 先验即可。

## 离线导出链路（推荐：先把整条链路跑通）

如果你希望从 rosbag 一次性导出：

- 原始点云帧（raw）
- 识别树点帧（tree）
- 原始点云上色（colored：全点 + rgb/label）
- 5/10 帧合成的 map PCD（TF 对齐）
- chunk(tree) 的多算法聚类 + BEV 圆圈预览 + 对比拼图

可以直接用一键脚本（全部中文目录输出）：

```bash
python3 src/orchard_row_mapping/tools/run_bag_export_chain.py \
  --bag /mysda/w/w/lio_ws/rosbags/2025-10-29-16-05-00.bag \
  --points-topic /liorl/deskew/cloud_deskewed \
  --use-gpu \
  --enable-kmeans-merge
```

跑完后从输出根目录的 `00_导航/` 进入即可（都是软链接，不占空间）。

## 重要参数
| 参数 | 说明 |
| --- | --- |
| `pointcloud_topic` | 输入点云话题，默认 `/velodyne_points` |
| `model_path` | RandLA-Net checkpoint 路径，若留空会依次尝试包内 `checkpoints/best_model.pth`、`/mysda/w/w/RandLA-Net-pytorch/noslam/checkpoints/best_model.pth` 等 |
| `num_points` | 每次推理使用的点数（采样/重复），默认 16384 |
| `num_classes` | 模型类别数，默认 2（0=果树，1=其他） |
| `tree_class` | 表示果树的类别 id，用于行拟合，默认 0 |
| `tree_prob_threshold` | 只用高置信度果树点做拟合（0 表示不限制） |
| `min_points_per_row` | 左右行最少需要的果树点数，默认 200 |
| `split_axis` | 分割左右行时使用的轴（`x` 或 `y`），默认 `y` |
| `split_margin` | 在中线附近留空区，避免左右行混在一起（默认 0） |
| `fit_x_min`/`fit_x_max` | 只用一定前方范围内的点拟合（默认不过滤） |
| `fit_y_abs_max` | 限制左右最大距离，过滤远处干扰（默认不过滤） |
| `fit_z_min`/`fit_z_max` | 限制高度范围，过滤地面/高处噪声（默认不过滤） |
| `line_inlier_distance` | PCA 拟合后按点到线距离做一次/多次剔除（0 表示关闭） |
| `line_endpoint_percentile` | 用分位数决定线段端点，减少离群点拉长（0 表示 min/max） |
| `line_smoothing_alpha` | 线段端点做 EMA 平滑（0 表示不平滑；越大越稳但滞后） |
| `hold_last_seconds` | 短时间内点不足时保持上一帧线，减少闪烁（0 表示关闭） |
| `label_colors` | 各类别在 `~segmented_cloud` 中的颜色映射 |

## 话题
- 发布：
  - `~segmented_cloud` (`sensor_msgs/PointCloud2`): 猜测类别的彩色点云。
  - `~row_markers` (`visualization_msgs/MarkerArray`): 左/右果树行的折线，可直接在 RViz 中显示。
- 订阅：
  - `pointcloud_topic` (`sensor_msgs/PointCloud2`): 原始点云。

## 说明
- 目前推理由固定数量点构成，若输入点数远大于 `num_points`，仅有部分点参与推理；可通过增大 `num_points` 或外部下采样来覆盖更多点。
- 本包只聚焦“分割 + 果树行拟合”；后续若需引入 ScanContext 定位，可以在同一工作空间内新建节点对 `/liorl/deskew/cloud_deskewed` 等话题做描述子匹配。
