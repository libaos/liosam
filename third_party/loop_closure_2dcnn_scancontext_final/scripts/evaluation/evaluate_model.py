#!/usr/bin/env python3
"""
通用模型评估脚本
支持所有模型类型的评估
"""
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import get_config
from models import SCRingCNN, SCStandardCNN, SCStandardCNNLite, SimpleCNN, SimpleCNNLite
from utils import SimpleLoopClosureDataset, setup_model_logger, get_timestamp, calculate_metrics

def load_model(model_path, device='cpu'):
    """加载模型"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 获取模型类型
    model_type = checkpoint.get('model_type', 'SCRingCNN')
    config = checkpoint.get('config', {})
    
    # 创建对应的模型
    if model_type == 'SCRingCNN':
        model = SCRingCNN(
            num_rings=config.get('num_rings', 20),
            num_sectors=config.get('num_sectors', 60),
            descriptor_dim=config.get('descriptor_dim', 256)
        )
    elif model_type == 'SCStandardCNN':
        model = SCStandardCNN(
            num_rings=config.get('num_rings', 20),
            num_sectors=config.get('num_sectors', 60),
            descriptor_dim=config.get('descriptor_dim', 256),
            use_residual=config.get('use_residual', True)
        )
    elif model_type == 'SCStandardCNNLite':
        model = SCStandardCNNLite(
            num_rings=config.get('num_rings', 20),
            num_sectors=config.get('num_sectors', 60),
            descriptor_dim=config.get('descriptor_dim', 128)
        )
    elif model_type == 'SimpleCNN':
        model = SimpleCNN(
            num_rings=config.get('num_rings', 20),
            num_sectors=config.get('num_sectors', 60),
            descriptor_dim=config.get('descriptor_dim', 256)
        )
    elif model_type == 'SimpleCNNLite':
        model = SimpleCNNLite(
            num_rings=config.get('num_rings', 20),
            num_sectors=config.get('num_sectors', 60),
            descriptor_dim=config.get('descriptor_dim', 128)
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, model_type, config

def evaluate_model(model, dataloader, device, logger, model_type):
    """评估模型性能"""
    model.eval()
    all_features = []
    all_labels = []
    
    # 计时
    start_time = time.time()
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            
            # 单批次推理计时
            batch_start = time.time()
            features = model(data)
            batch_time = time.time() - batch_start
            inference_times.append(batch_time)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    total_time = time.time() - start_time
    
    # 合并所有特征和标签
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 计算评估指标
    metrics = calculate_metrics(all_features, all_labels)
    
    # 添加性能指标
    metrics['total_inference_time'] = total_time
    metrics['avg_batch_time'] = sum(inference_times) / len(inference_times)
    metrics['samples_per_second'] = len(all_features) / total_time
    
    # 记录结果
    logger.info(f"{model_type} 评估结果:")
    logger.info("="*50)
    
    # 准确率指标
    logger.info("准确率指标:")
    for key in ['top_1', 'top_3', 'top_5', 'top_10']:
        if key in metrics:
            logger.info(f"  {key}: {metrics[key]:.4f}")
    
    # 排序质量指标
    logger.info("排序质量指标:")
    for key in ['mAP', 'MRR']:
        if key in metrics:
            logger.info(f"  {key}: {metrics[key]:.4f}")
    
    # 特征质量指标
    logger.info("特征质量指标:")
    for key in ['separation_ratio', 'classification_accuracy']:
        if key in metrics:
            logger.info(f"  {key}: {metrics[key]:.4f}")
    
    # 性能指标
    logger.info("性能指标:")
    logger.info(f"  总推理时间: {metrics['total_inference_time']:.4f}s")
    logger.info(f"  平均批次时间: {metrics['avg_batch_time']:.4f}s")
    logger.info(f"  处理速度: {metrics['samples_per_second']:.2f} samples/s")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='通用模型评估脚本')
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='测试数据目录路径')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--max_files', type=int, default=None,
                       help='最大文件数量')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备类型')
    parser.add_argument('--output', type=str, default=None,
                       help='输出结果文件路径')
    parser.add_argument('--detailed', action='store_true',
                       help='显示详细的评估信息')
    
    args = parser.parse_args()
    
    # 设置日志
    timestamp = get_timestamp()
    project_root = Path(__file__).parent.parent.parent

    # 先尝试从模型文件推断模型类型
    model_type = 'general'
    try:
        if Path(args.model).exists():
            checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
            model_type_from_checkpoint = checkpoint.get('model_type', 'general')
            if model_type_from_checkpoint in ['SCRingCNN', 'SCStandardCNN', 'SimpleCNN', 'SimpleCNNLite']:
                model_type = model_type_from_checkpoint.lower().replace('cnn', '_cnn')
    except:
        pass

    logger, log_file = setup_model_logger(
        model_type=model_type,
        script_type='evaluation',
        timestamp=timestamp,
        project_root=project_root
    )
    
    logger.info("🔍 开始模型评估")
    logger.info(f"模型文件: {args.model}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"设备: {args.device}")
    
    # 检查模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        return
    
    # 数据目录
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = project_root / "data" / "raw" / "ply_files"
    
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return
    
    # 加载模型
    logger.info("加载模型...")
    try:
        model, model_type, model_config = load_model(args.model, args.device)
        logger.info(f"模型类型: {model_type}")
        logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        if args.detailed:
            logger.info(f"模型配置: {model_config}")
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return
    
    # 创建数据集
    logger.info("加载测试数据集...")
    dataset = SimpleLoopClosureDataset(
        data_dir=data_dir,
        cache_dir=project_root / "data" / "cache",
        max_files=args.max_files
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    logger.info(f"测试集大小: {len(dataset)}")
    
    # 评估模型
    logger.info("开始评估...")
    metrics = evaluate_model(model, dataloader, torch.device(args.device), logger, model_type)
    
    # 保存结果
    results = {
        'model_path': str(model_path),
        'model_type': model_type,
        'model_config': model_config,
        'data_dir': str(data_dir),
        'dataset_size': len(dataset),
        'evaluation_config': {
            'batch_size': args.batch_size,
            'device': args.device,
            'max_files': args.max_files
        },
        'metrics': metrics,
        'timestamp': timestamp
    }
    
    # 输出文件路径
    if args.output:
        results_path = Path(args.output)
    else:
        results_dir = project_root / "outputs" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / f"evaluation_{model_type.lower()}_{timestamp}.json"
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"评估完成！结果保存至: {results_path}")
    
    # 显示总结
    logger.info("\n" + "="*60)
    logger.info("评估总结:")
    logger.info("="*60)
    logger.info(f"模型类型: {model_type}")
    logger.info(f"Top-1准确率: {metrics.get('top_1', 0):.4f}")
    logger.info(f"mAP: {metrics.get('mAP', 0):.4f}")
    logger.info(f"MRR: {metrics.get('MRR', 0):.4f}")
    logger.info(f"处理速度: {metrics.get('samples_per_second', 0):.2f} samples/s")

if __name__ == "__main__":
    main()
