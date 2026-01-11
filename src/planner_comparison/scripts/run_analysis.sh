#!/bin/bash

# 脚本：运行PSOLQR vs TEB对比分析
# 此脚本不会修改任何现有代码，只生成独立的分析结果

# 设置颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== PSOLQR vs TEB 对比分析工具 =====${NC}"
echo "此工具将生成理论对比数据和演示文档，不会修改任何现有代码"
echo

# 检查目录结构
SCRIPT_DIR=/root/lio_ws/src/planner_comparison/scripts
RESULTS_DIR=/root/lio_ws/src/planner_comparison/results
PRESENTATION_DIR=/root/lio_ws/src/planner_comparison/presentation

# 确保目录存在
mkdir -p $RESULTS_DIR
mkdir -p $PRESENTATION_DIR

# 安装必要的依赖
echo -e "${BLUE}正在检查并安装必要的依赖...${NC}"
pip install matplotlib numpy scipy

# 运行理论分析脚本
echo -e "${BLUE}正在生成理论分析数据...${NC}"
python $SCRIPT_DIR/theoretical_analysis.py

# 生成演示文档
echo -e "${BLUE}正在生成演示文档...${NC}"
python $SCRIPT_DIR/generate_presentation.py

echo
echo -e "${GREEN}分析完成！${NC}"
echo -e "结果保存在: ${BLUE}$RESULTS_DIR${NC}"
echo -e "演示文档保存在: ${BLUE}$PRESENTATION_DIR/psolqr_presentation.pdf${NC}"
echo
echo "您可以使用以下命令查看结果:"
echo "  - 查看图像: xdg-open $RESULTS_DIR/radar_comparison.png"
echo "  - 查看演示文档: xdg-open $PRESENTATION_DIR/psolqr_presentation.pdf"
echo
echo -e "${BLUE}===== 分析结束 =====${NC}" 