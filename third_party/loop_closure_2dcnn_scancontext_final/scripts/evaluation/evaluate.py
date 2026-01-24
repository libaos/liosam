#!/usr/bin/env python3
"""
回环检测评估脚本
"""
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from models import SCRingCNN, SCStandardCNN, SCStandardCNNLite, SimpleCNN, SimpleCNNLite
from utils import SimpleLoopClosureDataset, setup_logger, get_timestamp, calculate_metrics

def main():
    parser = argparse.ArgumentParser(description='评估回环检测模型')
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='测试数据目录路径')
    parser.add_argument('--config', type=str, default='baseline',
                       help='配置名称')
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config(args.config)
    
    # 如果指定了数据目录，覆盖配置
    if args.data_dir:
        config.DATA_DIR = Path(args.data_dir)
    
    # 设置日志
    timestamp = get_timestamp()
    log_file = config.LOG_DIR / f"evaluate_{timestamp}.log"
    logger = setup_logger('evaluate', log_file)
    
    logger.info(f"开始评估模型: {args.model}")
    logger.info(f"数据目录: {config.DATA_DIR}")
    
    # 检查模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        return
    
    # 检查数据目录
    if not config.DATA_DIR.exists():
        logger.error(f"数据目录不存在: {config.DATA_DIR}")
        return
    
    # 加载模型
    logger.info("加载模型...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 创建模型（根据配置或从checkpoint推断）
    if hasattr(config, 'MODEL_TYPE'):
        model_type = config.MODEL_TYPE
    else:
        # 从checkpoint推断模型类型
        model_type = checkpoint.get('config', {}).get('MODEL_TYPE', 'sc_ring_cnn')

    if model_type == "simple_cnn":
        model = SimpleCNN(
            num_rings=config.INPUT_HEIGHT,
            num_sectors=config.INPUT_WIDTH,
            descriptor_dim=config.DESCRIPTOR_DIM
        )
    elif model_type == "simple_cnn_lite":
        model = SimpleCNNLite(
            num_rings=config.INPUT_HEIGHT,
            num_sectors=config.INPUT_WIDTH,
            descriptor_dim=config.DESCRIPTOR_DIM
        )
    elif model_type == "sc_standard_cnn":
        model = SCStandardCNN(
            num_rings=config.INPUT_HEIGHT,
            num_sectors=config.INPUT_WIDTH,
            descriptor_dim=config.DESCRIPTOR_DIM
        )
    elif model_type == "sc_standard_cnn_lite":
        model = SCStandardCNNLite(
            num_rings=config.INPUT_HEIGHT,
            num_sectors=config.INPUT_WIDTH,
            descriptor_dim=config.DESCRIPTOR_DIM
        )
    else:  # 默认使用SCRingCNN
        model = SCRingCNN(
            num_rings=config.INPUT_HEIGHT,
            num_sectors=config.INPUT_WIDTH,
            descriptor_dim=config.DESCRIPTOR_DIM
        )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device(config.DEVICE)
    model.to(device)
    model.eval()
    
    logger.info(f"模型加载完成，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建数据集
    logger.info("加载测试数据集...")
    dataset = SimpleLoopClosureDataset(
        data_dir=config.DATA_DIR,
        cache_dir=config.CACHE_DIR,
        max_files=config.MAX_FILES
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False
    )
    
    logger.info(f"测试集大小: {len(dataset)}")
    
    # 提取特征
    logger.info("提取特征...")
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            features = model(data)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    # 合并所有特征和标签
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    logger.info(f"特征形状: {all_features.shape}")
    logger.info(f"标签形状: {all_labels.shape}")
    
    # 计算评估指标
    logger.info("计算评估指标...")
    metrics = calculate_metrics(all_features, all_labels)
    
    # 显示结果
    logger.info("\n" + "="*50)
    logger.info("评估结果:")
    logger.info("="*50)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{key:20s}: {value:.4f}")
        else:
            logger.info(f"{key:20s}: {value}")
    
    # 保存结果
    results = {
        'model_path': str(model_path),
        'data_dir': str(config.DATA_DIR),
        'dataset_size': len(dataset),
        'metrics': metrics,
        'timestamp': timestamp
    }
    
    results_path = config.RESULTS_DIR / f"evaluation_results_{timestamp}.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\n评估完成！结果保存至: {results_path}")

if __name__ == "__main__":
    main()
