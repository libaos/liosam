#!/usr/bin/env python3
"""
回环检测训练脚本
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
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from models import SCRingCNN, SCStandardCNN, SCStandardCNNLite, SimpleCNN, SimpleCNNLite
from utils import SimpleLoopClosureDataset, setup_logger, get_timestamp, calculate_metrics

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

def create_triplets(features, labels, batch_size):
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
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (data, labels) in enumerate(progress_bar):
        data, labels = data.to(device), labels.to(device)
        
        # 前向传播
        features = model(data)
        
        # 创建三元组
        triplets = create_triplets(features, labels, len(data))
        
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
    logger.info(f"Training Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, dataloader, device, logger):
    """评估模型"""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc="Evaluating"):
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
    logger.info("Evaluation Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='训练回环检测模型')
    parser.add_argument('--config', type=str, default='baseline', 
                       help='配置名称 (baseline, quick)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='数据目录路径')
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config(args.config)
    
    # 如果指定了数据目录，覆盖配置
    if args.data_dir:
        config.DATA_DIR = Path(args.data_dir)
    
    # 设置日志
    timestamp = get_timestamp()
    log_file = config.LOG_DIR / f"train_{config.EXPERIMENT_NAME}_{timestamp}.log"
    logger = setup_logger('train', log_file)
    
    logger.info(f"开始训练 - 实验: {config.EXPERIMENT_NAME}")
    logger.info(f"配置: {args.config}")
    logger.info(f"数据目录: {config.DATA_DIR}")
    
    # 检查数据目录
    if not config.DATA_DIR.exists():
        logger.error(f"数据目录不存在: {config.DATA_DIR}")
        return
    
    # 创建数据集
    logger.info("加载数据集...")
    dataset = SimpleLoopClosureDataset(
        data_dir=config.DATA_DIR,
        cache_dir=config.CACHE_DIR,
        max_files=config.MAX_FILES
    )
    
    # 分割数据集
    train_size = int((1 - config.VAL_RATIO) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    if config.MODEL_TYPE == "simple_cnn":
        model = SimpleCNN(
            num_rings=config.INPUT_HEIGHT,
            num_sectors=config.INPUT_WIDTH,
            descriptor_dim=config.DESCRIPTOR_DIM
        )
    elif config.MODEL_TYPE == "simple_cnn_lite":
        model = SimpleCNNLite(
            num_rings=config.INPUT_HEIGHT,
            num_sectors=config.INPUT_WIDTH,
            descriptor_dim=config.DESCRIPTOR_DIM
        )
    elif config.MODEL_TYPE == "sc_standard_cnn":
        model = SCStandardCNN(
            num_rings=config.INPUT_HEIGHT,
            num_sectors=config.INPUT_WIDTH,
            descriptor_dim=config.DESCRIPTOR_DIM
        )
    elif config.MODEL_TYPE == "sc_standard_cnn_lite":
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
    
    device = torch.device(config.DEVICE)
    model.to(device)
    
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建损失函数和优化器
    criterion = TripletLoss(margin=config.MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 训练循环
    best_top1 = 0
    results = {
        'config': args.config,
        'experiment_name': config.EXPERIMENT_NAME,
        'epochs': [],
        'best_metrics': None
    }
    
    for epoch in range(config.NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, logger)
        
        # 评估
        if (epoch + 1) % config.EVAL_INTERVAL == 0 or epoch == config.NUM_EPOCHS - 1:
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
                
                model_path = config.MODEL_SAVE_DIR / f"best_model_{config.EXPERIMENT_NAME}_{timestamp}.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config.__dict__,
                    'metrics': metrics,
                    'epoch': epoch + 1
                }, model_path)
                
                logger.info(f"保存最佳模型: {model_path}")
    
    # 保存训练结果
    results_path = config.RESULTS_DIR / f"training_results_{config.EXPERIMENT_NAME}_{timestamp}.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"训练完成！结果保存至: {results_path}")
    logger.info(f"最佳Top-1准确率: {best_top1:.4f}")

if __name__ == "__main__":
    main()
