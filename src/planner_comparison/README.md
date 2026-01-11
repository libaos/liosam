# 规划器坡道性能对比分析工具

这个工具用于对比PSOLQR与TEB两种局部路径规划器在坡道环境中的性能表现。通过模拟分析，生成可视化数据和详细报告，帮助用户理解和展示PSOLQR在坡道环境中的优势。

## 功能特点

- **3D坡道地形模拟**：创建真实的坡道环境，模拟15度左右的坡度
- **轨迹规划对比**：可视化展示两种规划器在坡道上的轨迹差异
- **速度控制分析**：对比两种规划器在上下坡时的速度稳定性
- **轨迹跟踪精度**：分析不同坡度下的轨迹跟踪误差
- **能量消耗评估**：比较两种规划器在坡道环境中的能量效率
- **自动生成PDF报告**：生成包含详细分析和可视化结果的专业报告

## 目录结构

```
planner_comparison/
├── scripts/
│   ├── slope_simulation.py    # 坡道场景模拟脚本
│   └── generate_report.py     # PDF报告生成脚本
├── results/                   # 模拟结果和图表
├── run_simulation.sh          # 运行脚本
└── README.md                  # 项目说明
```

## 依赖项

- Python 3.6+
- NumPy
- Matplotlib
- SciPy
- ReportLab (用于生成PDF)

## 使用方法

1. 确保已安装所有依赖项：

```bash
pip install numpy matplotlib scipy reportlab
```

2. 运行模拟和报告生成：

```bash
cd /root/lio_ws/src/planner_comparison
./run_simulation.sh
```

3. 查看生成的报告和可视化结果：

```bash
# 如果在图形界面环境中
xdg-open report.pdf

# 查看生成的图表
ls -l results/
```

## 自定义配置

如果需要调整模拟参数，可以修改以下文件：

- `scripts/slope_simulation.py`：修改坡度角度、轨迹生成参数等
- `scripts/generate_report.py`：自定义报告内容和格式

## 输出结果

运行脚本后，将生成以下输出：

1. **可视化图表** (保存在`results/`目录):
   - 3D坡道地形模型
   - 轨迹对比图
   - 速度控制曲线
   - 轨迹跟踪误差曲线
   - 能量消耗柱状图

2. **PDF报告** (`report.pdf`):
   - 包含所有分析结果和图表的详细报告
   - 适合用于展示和汇报

## 注意事项

- 本工具基于理论模型和模拟数据，用于展示PSOLQR在坡道环境中的理论优势
- 模拟参数可根据实际需求进行调整，以更好地匹配特定场景
- 报告内容可根据演示需求进行定制 