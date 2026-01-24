# 训练脚本目录

本目录包含各种模型的专门训练脚本，每个脚本都针对特定模型进行了优化。

## 📁 脚本列表

### 🎯 专门训练脚本

| 脚本名称 | 模型类型 | 特点 | 推荐用途 |
|---------|---------|------|---------|
| `train_sc_ring_cnn.py` | SCRingCNN | 环形卷积，专为ScanContext设计 | 最高精度需求 |
| `train_sc_standard_cnn.py` | SCStandardCNN | 标准卷积，对比实验 | 环形卷积对比 |
| `train_simple_cnn.py` | SimpleCNN | 简化架构，平衡性能 | 平衡性能和效率 |
| `train_simple_cnn_lite.py` | SimpleCNNLite | 轻量级，最快速度 | 资源受限环境 |

### 🔧 通用训练脚本

| 脚本名称 | 功能 | 特点 |
|---------|------|------|
| `train.py` | 通用训练 | 支持所有模型类型，配置驱动 |
| `train_with_augmentation.py` | 数据增强训练 | 支持多种数据增强方法 |

## 🚀 使用方法

### 方法1: 直接运行专门脚本

```bash
# SCRingCNN专门训练
python scripts/training/train_sc_ring_cnn.py --epochs 10 --batch_size 16

# SCStandardCNN专门训练
python scripts/training/train_sc_standard_cnn.py --epochs 10 --use_residual

# SimpleCNN专门训练
python scripts/training/train_simple_cnn.py --epochs 8 --batch_size 32

# SimpleCNNLite专门训练
python scripts/training/train_simple_cnn_lite.py --epochs 6 --fast_mode
```

### 方法2: 使用统一启动脚本

```bash
# 从项目根目录运行
python run.py train_sc_ring --epochs 10 --batch_size 16
python run.py train_sc_standard --epochs 10 --use_residual
python run.py train_simple --epochs 8 --batch_size 32
python run.py train_simple_lite --epochs 6 --fast_mode
```

## ⚙️ 参数说明

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epochs` | int | 模型相关 | 训练轮数 |
| `--batch_size` | int | 模型相关 | 批次大小 |
| `--learning_rate` | float | 0.001 | 学习率 |
| `--descriptor_dim` | int | 模型相关 | 描述子维度 |
| `--margin` | float | 1.0 | 三元组损失边界 |
| `--data_dir` | str | auto | 数据目录路径 |
| `--max_files` | int | 100 | 最大文件数量 |
| `--device` | str | cpu | 设备类型 |

### 模型特定参数

#### SCStandardCNN
- `--use_residual`: 是否使用残差连接

#### SimpleCNN
- `--dropout`: Dropout概率 (默认: 0.5)

#### SimpleCNNLite
- `--fast_mode`: 快速模式，减少评估频率

## 📊 模型对比

### 参数数量对比

| 模型 | 参数数量 | 描述子维度 | 推理速度 |
|------|---------|-----------|---------|
| SCRingCNN | 2,262,144 | 256 | 慢 |
| SCStandardCNN | 2,260,736 | 256 | 中等 |
| SimpleCNN | 290,496 | 256 | 快 |
| SimpleCNNLite | 9,376 | 128 | 最快 |

### 训练配置建议

| 模型 | 推荐epochs | 推荐batch_size | 推荐学习率 |
|------|-----------|---------------|-----------|
| SCRingCNN | 10-15 | 16 | 0.001 |
| SCStandardCNN | 10-15 | 16 | 0.001 |
| SimpleCNN | 8-12 | 32 | 0.001 |
| SimpleCNNLite | 6-10 | 64 | 0.002 |

## 🎯 使用建议

### 1. 首次使用
```bash
# 快速测试所有模型
python run.py train_simple_lite --epochs 3 --fast_mode
python run.py train_simple --epochs 5
python run.py train_sc_standard --epochs 5
python run.py train_sc_ring --epochs 5
```

### 2. 对比实验
```bash
# 环形卷积 vs 标准卷积对比
python run.py train_sc_ring --epochs 10 --batch_size 16
python run.py train_sc_standard --epochs 10 --batch_size 16
```

### 3. 性能优化
```bash
# 轻量级模型快速训练
python run.py train_simple_lite --epochs 8 --batch_size 64 --learning_rate 0.002
```

### 4. 高精度训练
```bash
# 环形卷积长时间训练
python run.py train_sc_ring --epochs 20 --batch_size 16 --learning_rate 0.0005
```

## 📝 输出文件

每个训练脚本都会生成以下文件：

### 模型文件
- 位置: `outputs/models/`
- 格式: `best_{model_type}_{timestamp}.pth`
- 内容: 模型权重、配置、最佳指标

### 日志文件
- 位置: `outputs/logs/`
- 格式: `train_{model_type}_{timestamp}.log`
- 内容: 详细的训练日志

### 结果文件
- 位置: `outputs/results/`
- 格式: `{model_type}_results_{timestamp}.json`
- 内容: 训练结果、指标、配置

## 🔍 故障排除

### 常见问题

1. **内存不足**
   - 减少 `--batch_size`
   - 减少 `--max_files`
   - 使用 SimpleCNNLite

2. **训练太慢**
   - 使用 `--fast_mode` (SimpleCNNLite)
   - 减少 `--epochs`
   - 增加 `--batch_size`

3. **精度不够**
   - 增加 `--epochs`
   - 使用 SCRingCNN
   - 调整 `--learning_rate`

4. **模型不收敛**
   - 检查数据质量
   - 调整 `--margin`
   - 降低 `--learning_rate`
