# 时序回环检测系统

基于时序Scan Context的路径识别系统，实现了2D CNN和3D CNN两种方法的对比研究。

## 系统概述

本系统实现了以下核心功能：

1. **时序Scan Context生成**: 从原始点云数据生成时序Scan Context特征序列
2. **2D CNN基线模型**: 将N×H×W时序张量视为N通道的2D图像
3. **3D CNN核心模型**: 将N×H×W时序张量视为1×N×H×W的3D体数据，显式捕捉时空动态
4. **完整的训练和评估流程**: 支持模型训练、评估和性能比较

## 项目结构

```
loop_closure_clean/
├── models/
│   ├── temporal_2d_cnn.py          # 2D CNN模型实现
│   ├── temporal_3d_cnn.py          # 3D CNN模型实现
│   └── ...
├── utils/
│   ├── temporal_dataset.py         # 时序数据集类
│   ├── scan_context.py            # Scan Context生成器
│   └── ...
├── scripts/
│   ├── training/
│   │   └── train_temporal_models.py # 训练脚本
│   ├── evaluation/
│   │   └── evaluate_temporal_models.py # 评估脚本
│   └── tools/
│       └── preprocess_temporal_data.py # 数据预处理脚本
├── demo_temporal_system.py         # 完整系统演示脚本
└── README_TEMPORAL.md              # 本文件
```

## 快速开始

### 1. 环境准备

确保安装了以下依赖：

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn tqdm tensorboard
```

### 2. 数据准备

将PLY点云文件放置在以下目录结构中：

```
data/
└── raw/
    └── ply_files/
        ├── cloud_00002.ply
        ├── cloud_00005.ply
        └── ...
```

### 3. 运行完整演示

```bash
# 运行完整流程（数据预处理 + 训练 + 评估）
python demo_temporal_system.py --data_dir data --mode full --model temporal_3d_cnn --epochs 50

# 只运行数据预处理
python demo_temporal_system.py --data_dir data --mode preprocess

# 只训练模型
python demo_temporal_system.py --data_dir data --mode train --model temporal_3d_cnn --epochs 100

# 比较多个模型性能
python demo_temporal_system.py --data_dir data --mode compare --epochs 30
```

### 4. 单独使用各个组件

#### 数据预处理

```bash
python scripts/tools/preprocess_temporal_data.py \
    --data_dir data \
    --output_dir data/processed \
    --sequence_length 5 \
    --num_classes 20 \
    --visualize
```

#### 模型训练

```bash
python scripts/training/train_temporal_models.py \
    --model temporal_3d_cnn \
    --sequence_length 5 \
    --batch_size 16 \
    --epochs 100 \
    --lr 0.001
```

#### 模型评估

```bash
python scripts/evaluation/evaluate_temporal_models.py \
    --checkpoint outputs/temporal_3d_cnn_seq5/checkpoint_best.pth \
    --output_dir outputs/evaluation
```

## 模型架构

### 2D CNN基线模型

- **输入**: (batch_size, N, H, W) - N个时序帧作为通道
- **特点**: 隐式时序建模，将时间维度作为通道维度处理
- **优势**: 计算效率高，参数量少
- **局限**: 无法显式捕捉时序动态

### 3D CNN核心模型

- **输入**: (batch_size, 1, N, H, W) - 3D体数据
- **特点**: 显式时空建模，3D卷积核同时覆盖时间和空间维度
- **优势**: 能够捕捉复杂的时空模式和运动趋势
- **创新**: 专门设计用于回环检测的轻量化3D CNN架构

## 实验配置

### 数据集设置

- **路径分段**: 将完整轨迹按顺序分成20段作为类别标签
- **时序长度**: 支持3、5、7、10帧的时序序列
- **数据划分**: 70%训练，15%验证，15%测试
- **数据增强**: 随机旋转、噪声添加等

### 训练参数

- **优化器**: Adam (lr=0.001, weight_decay=1e-4)
- **损失函数**: 交叉熵损失
- **学习率调度**: ReduceLROnPlateau
- **批次大小**: 16 (可调整)
- **训练轮数**: 100 (可调整)

### 评估指标

- **准确率**: Top-1, Top-3, Top-5准确率
- **精确率和召回率**: 宏平均和加权平均
- **推理速度**: FPS (帧每秒)
- **混淆矩阵**: 详细的分类结果分析

## 实验结果

系统会自动生成以下输出：

1. **训练日志**: 训练过程的详细记录
2. **TensorBoard日志**: 可视化训练曲线
3. **评估报告**: 详细的性能指标
4. **混淆矩阵**: 分类结果可视化
5. **t-SNE可视化**: 特征空间分布
6. **模型比较报告**: 多模型性能对比

## 消融实验

### 序列长度影响

```bash
# 测试不同序列长度
for seq_len in 3 5 7 10; do
    python scripts/training/train_temporal_models.py \
        --model temporal_3d_cnn \
        --sequence_length $seq_len \
        --epochs 50
done
```

### 模型架构比较

```bash
# 比较所有模型架构
python demo_temporal_system.py --mode compare --epochs 50
```

## 可视化功能

系统提供丰富的可视化功能：

1. **Scan Context可视化**: 单帧特征图
2. **时序序列可视化**: 多帧时序特征
3. **训练曲线**: 损失和准确率变化
4. **混淆矩阵**: 分类结果热力图
5. **类别性能**: 各类别精确率/召回率
6. **特征可视化**: t-SNE降维可视化

## 高级用法

### 自定义配置文件

创建JSON配置文件来自定义训练参数：

```json
{
  "model": {
    "type": "temporal_3d_cnn",
    "params": {
      "sequence_length": 5,
      "num_rings": 20,
      "num_sectors": 60,
      "num_classes": 20,
      "dropout_rate": 0.5
    }
  },
  "data": {
    "data_dir": "data/processed",
    "sequence_length": 5,
    "batch_size": 16,
    "num_workers": 4,
    "cache_dir": "data/cache",
    "num_classes": 20
  },
  "optimizer": {
    "type": "adam",
    "lr": 0.001,
    "weight_decay": 1e-4
  },
  "training": {
    "epochs": 100,
    "log_interval": 10
  },
  "output_dir": "outputs/custom_experiment"
}
```

然后使用配置文件训练：

```bash
python scripts/training/train_temporal_models.py --config config.json
```

### 恢复训练

```bash
python scripts/training/train_temporal_models.py \
    --config config.json \
    --resume outputs/experiment/checkpoint_latest.pth
```

## 故障排除

### 常见问题

1. **内存不足**: 减小batch_size或sequence_length
2. **CUDA错误**: 检查GPU内存和CUDA版本
3. **数据加载慢**: 增加num_workers或使用缓存
4. **训练不收敛**: 调整学习率或模型架构

### 性能优化

1. **使用混合精度训练**: 减少内存使用
2. **数据预加载**: 使用缓存加速数据加载
3. **模型剪枝**: 使用轻量级模型变体
4. **批次大小调优**: 根据硬件配置调整

## 贡献指南

欢迎提交Issue和Pull Request来改进系统。

## 许可证

本项目采用MIT许可证。

## 引用

如果您在研究中使用了本系统，请引用：

```bibtex
@misc{temporal_loop_closure,
  title={Temporal Scan Context for Loop Closure Detection},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
```
