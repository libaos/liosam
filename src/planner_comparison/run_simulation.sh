#!/bin/bash

# 坡道模拟和报告生成脚本

# 设置工作目录
WORK_DIR="/root/lio_ws/src/planner_comparison"
RESULTS_DIR="$WORK_DIR/results"
SCRIPTS_DIR="$WORK_DIR/scripts"

# 创建必要的目录
mkdir -p $RESULTS_DIR

echo "===== 坡道环境下PSOLQR与TEB规划器性能对比分析 ====="
echo ""

# 检查Python依赖
echo "检查依赖..."
pip install numpy matplotlib scipy reportlab --quiet

# 运行坡道模拟
echo "开始运行坡道场景模拟..."
python3 $SCRIPTS_DIR/slope_simulation.py
echo ""

# 生成对比报告
echo "生成性能对比报告..."
python3 $SCRIPTS_DIR/generate_report.py --results_dir $RESULTS_DIR --output $WORK_DIR/report.pdf
echo ""

echo "模拟和报告生成完成！"
echo "报告已保存至: $WORK_DIR/report.pdf"
echo "可视化结果保存在: $RESULTS_DIR/"
echo ""
echo "使用以下命令查看报告:"
echo "xdg-open $WORK_DIR/report.pdf  # 如果在图形界面环境"
echo ""
echo "使用以下命令查看可视化结果:"
echo "ls -l $RESULTS_DIR/"
