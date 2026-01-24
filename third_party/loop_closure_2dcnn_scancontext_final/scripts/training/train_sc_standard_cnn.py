#!/usr/bin/env python3
"""
SCStandardCNN专门训练脚本
标准卷积模型训练，用于与环形卷积对比
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import get_config
from models import SCStandardCNN
from utils import SimpleLoopClosureDataset, setup_model_logger, get_timestamp, calculate_metrics

class TripletLoss(nn.Module):
    """三元组损失"""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()

def create_triplets(features, labels):
    """创建三元组"""
    triplets = []
    labels_np = labels.cpu().numpy()
    
    for i in range(len(features)):
        anchor_label = labels_np[i]
        
        # 找正样本（同类别，但不是自己）
        positive_indices = [j for j in range(len(features)) 
                          if labels_np[j] == anchor_label and j != i]
        if not positive_indices:
            continue
            
        # 找负样本（不同类别）
        negative_indices = [j for j in range(len(features)) 
                          if labels_np[j] != anchor_label]
        if not negative_indices:
            continue
        
        # 随机选择正负样本
        import random
        pos_idx = random.choice(positive_indices)
        neg_idx = random.choice(negative_indices)
        
        triplets.append((i, pos_idx, neg_idx))
    
    return triplets

def train_epoch(model, dataloader, criterion, optimizer, device, logger):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training SCStandardCNN")
    
    for batch_idx, (data, labels) in enumerate(progress_bar):
        data, labels = data.to(device), labels.to(device)
        
        # 前向传播
        features = model(data)
        
        # 创建三元组
        triplets = create_triplets(features, labels)
        
        if not triplets:
            continue
        
        # 计算三元组损失
        total_triplet_loss = 0
        for anchor_idx, pos_idx, neg_idx in triplets:
            anchor = features[anchor_idx:anchor_idx+1]
            positive = features[pos_idx:pos_idx+1]
            negative = features[neg_idx:neg_idx+1]
            
            triplet_loss = criterion(anchor, positive, negative)
            total_triplet_loss += triplet_loss
        
        if len(triplets) > 0:
            loss = total_triplet_loss / len(triplets)
        else:
            continue
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    logger.info(f"SCStandardCNN Training Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, dataloader, device, logger):
    """评估模型"""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc="Evaluating SCStandardCNN"):
            data = data.to(device)
            features = model(data)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    # 合并所有特征和标签
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 计算指标
    metrics = calculate_metrics(all_features, all_labels)
    
    # 记录结果
    logger.info("SCStandardCNN Evaluation Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='SCStandardCNN专门训练脚本')
    parser.add_argument('--epochs', type=int, default=20,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--descriptor_dim', type=int, default=256,
                       help='描述子维度')
    parser.add_argument('--margin', type=float, default=1.0,
                       help='三元组损失边界')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='数据目录路径')
    parser.add_argument('--max_files', type=int, default=100,
                       help='最大文件数量')
    parser.add_argument('--device', type=str, default='0',
                       help='设备类型 (cpu, 0, 1, 2, ... 或 cuda:0, cuda:1, ...)')
    parser.add_argument('--use_residual', action='store_true',
                       help='是否使用残差连接')
    
    args = parser.parse_args()
    
    # 设置日志
    timestamp = get_timestamp()
    project_root = Path(__file__).parent.parent.parent

    logger, log_file = setup_model_logger(
        model_type='sc_standard_cnn',
        script_type='training',
        timestamp=timestamp,
        project_root=project_root
    )
    
    logger.info("🚀 开始SCStandardCNN专门训练")
    logger.info(f"训练轮数: {args.epochs}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"学习率: {args.learning_rate}")
    logger.info(f"描述子维度: {args.descriptor_dim}")
    logger.info(f"使用残差连接: {args.use_residual}")
    # 设备处理
    if args.device == 'cpu':
        device = torch.device('cpu')
        logger.info(f"使用设备: CPU")
    elif args.device.isdigit():
        # 数字形式，如 '0', '1', '2'
        gpu_id = int(args.device)
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            device = torch.device(f'cuda:{gpu_id}')
            logger.info(f"使用设备: GPU {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
        else:
            device = torch.device('cpu')
            logger.warning(f"GPU {gpu_id} 不可用，使用 CPU")
    elif args.device.startswith('cuda:'):
        # cuda:0, cuda:1 形式
        if torch.cuda.is_available():
            device = torch.device(args.device)
            gpu_id = int(args.device.split(':')[1])
            logger.info(f"使用设备: {args.device} ({torch.cuda.get_device_name(gpu_id)})")
        else:
            device = torch.device('cpu')
            logger.warning(f"CUDA 不可用，使用 CPU")
    else:
        device = torch.device('cpu')
        logger.warning(f"未知设备类型 '{args.device}'，使用 CPU")

    # 数据目录
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = project_root / "data" / "raw" / "ply_files"
    
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return
    
    # 创建数据集
    logger.info("加载数据集...")
    dataset = SimpleLoopClosureDataset(
        data_dir=data_dir,
        cache_dir=project_root / "data" / "cache",
        max_files=args.max_files
    )
    
    # 分割数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    
    # 创建SCStandardCNN模型
    model = SCStandardCNN(
        num_rings=20,
        num_sectors=60,
        descriptor_dim=args.descriptor_dim,
        use_residual=args.use_residual
    )
    
    model.to(device)
    
    logger.info(f"SCStandardCNN模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建损失函数和优化器
    criterion = TripletLoss(margin=args.margin)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 训练循环
    best_top1 = 0
    results = {
        'model_type': 'SCStandardCNN',
        'experiment_name': f'sc_standard_cnn_{timestamp}',
        'config': vars(args),
        'epochs': [],
        'best_metrics': None
    }
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, logger)
        
        # 评估
        if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, val_loader, device, logger)
            
            # 保存结果
            epoch_result = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'metrics': metrics
            }
            results['epochs'].append(epoch_result)
            
            # 保存最佳模型
            if metrics['top_1'] > best_top1:
                best_top1 = metrics['top_1']
                results['best_metrics'] = metrics
                
                model_dir = project_root / "outputs" / "models"
                model_dir.mkdir(parents=True, exist_ok=True)
                
                model_path = model_dir / f"best_sc_standard_cnn_{timestamp}.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': vars(args),
                    'metrics': metrics,
                    'epoch': epoch + 1,
                    'model_type': 'SCStandardCNN'
                }, model_path)
                
                logger.info(f"保存最佳SCStandardCNN模型: {model_path}")
    
    # 保存训练结果
    results_dir = project_root / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = results_dir / f"sc_standard_cnn_results_{timestamp}.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"SCStandardCNN训练完成！结果保存至: {results_path}")
    logger.info(f"最佳Top-1准确率: {best_top1:.4f}")

if __name__ == "__main__":
    main()
