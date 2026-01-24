#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.temporal_dataset import TemporalScanContextDataset, create_temporal_dataloaders
from models.temporal_2d_cnn import Temporal2DCNN, Temporal2DCNNLite, Temporal2DCNNResNet
from models.temporal_3d_cnn import Temporal3DCNN, Temporal3DCNNLite, Temporal3DCNNDeep
from utils.logger import Logger

class TemporalModelTrainer:
    """时序模型训练器"""
    
    def __init__(self, config):
        """
        初始化训练器
        
        参数:
            config (dict): 训练配置
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志
        self.logger = Logger(self.output_dir / 'training.log')
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # 保存配置
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # 初始化模型
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 初始化优化器和损失函数
        self.optimizer = self._create_optimizer()
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = self._create_scheduler()
        
        # 初始化数据加载器
        self.train_loader, self.val_loader, self.test_loader = self._create_dataloaders()
        
        # 训练状态
        self.best_val_acc = 0.0
        self.start_epoch = 0
        
        # 加载检查点（如果存在）
        if config.get('resume_checkpoint'):
            self._load_checkpoint(config['resume_checkpoint'])
    
    def _create_model(self):
        """创建模型"""
        model_type = self.config['model']['type']
        model_params = self.config['model']['params']
        
        if model_type == 'temporal_2d_cnn':
            return Temporal2DCNN(**model_params)
        elif model_type == 'temporal_2d_cnn_lite':
            return Temporal2DCNNLite(**model_params)
        elif model_type == 'temporal_2d_cnn_resnet':
            return Temporal2DCNNResNet(**model_params)
        elif model_type == 'temporal_3d_cnn':
            return Temporal3DCNN(**model_params)
        elif model_type == 'temporal_3d_cnn_lite':
            return Temporal3DCNNLite(**model_params)
        elif model_type == 'temporal_3d_cnn_deep':
            return Temporal3DCNNDeep(**model_params)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def _create_optimizer(self):
        """创建优化器"""
        optimizer_config = self.config['optimizer']
        optimizer_type = optimizer_config['type']
        
        if optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_config = self.config.get('scheduler')
        if not scheduler_config:
            return None
        
        scheduler_type = scheduler_config['type']
        
        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config['T_max']
            )
        elif scheduler_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config['factor'],
                patience=scheduler_config['patience']
            )
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")
    
    def _create_dataloaders(self):
        """创建数据加载器"""
        data_config = self.config['data']
        
        return create_temporal_dataloaders(
            data_dir=data_config['data_dir'],
            sequence_length=data_config['sequence_length'],
            batch_size=data_config['batch_size'],
            num_workers=data_config.get('num_workers', 4),
            cache_dir=data_config.get('cache_dir'),
            num_classes=data_config.get('num_classes', 20)
        )
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 打印进度
            if batch_idx % self.config['training']['log_interval'] == 0:
                self.logger.info(
                    f'Epoch: {epoch}, Batch: {batch_idx}/{len(self.train_loader)}, '
                    f'Loss: {loss.item():.6f}, Acc: {100.*correct/total:.2f}%'
                )
        
        # 计算平均指标
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        # 记录到tensorboard
        self.writer.add_scalar('Train/Loss', avg_loss, epoch)
        self.writer.add_scalar('Train/Accuracy', accuracy, epoch)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        # 计算平均指标
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # 记录到tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', accuracy, epoch)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新检查点
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pth')
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pth')
            self.logger.info(f"保存最佳模型，验证准确率: {self.best_val_acc:.2f}%")
    
    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint['best_val_acc']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"从检查点恢复训练，epoch: {self.start_epoch}, 最佳验证准确率: {self.best_val_acc:.2f}%")
    
    def train(self):
        """开始训练"""
        self.logger.info("开始训练...")
        self.logger.info(f"模型类型: {self.config['model']['type']}")
        self.logger.info(f"训练集大小: {len(self.train_loader.dataset)}")
        self.logger.info(f"验证集大小: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            start_time = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # 保存检查点
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            self.save_checkpoint(epoch, is_best)
            
            # 打印epoch结果
            epoch_time = time.time() - start_time
            self.logger.info(
                f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                f'Time: {epoch_time:.2f}s, LR: {current_lr:.6f}'
            )
        
        self.logger.info(f"训练完成！最佳验证准确率: {self.best_val_acc:.2f}%")
        self.writer.close()


def create_default_config(model_type='temporal_3d_cnn', sequence_length=5):
    """创建默认配置"""
    config = {
        "model": {
            "type": model_type,
            "params": {
                "sequence_length": sequence_length,
                "num_rings": 20,
                "num_sectors": 60,
                "num_classes": 20,
                "dropout_rate": 0.5
            }
        },
        "data": {
            "data_dir": "data",
            "sequence_length": sequence_length,
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
        "scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 10
        },
        "training": {
            "epochs": 100,
            "log_interval": 10
        },
        "output_dir": "outputs/temporal_experiments"
    }
    return config


def main():
    parser = argparse.ArgumentParser(description='训练时序模型')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--model', type=str, default='temporal_3d_cnn',
                       choices=['temporal_2d_cnn', 'temporal_2d_cnn_lite', 'temporal_2d_cnn_resnet',
                               'temporal_3d_cnn', 'temporal_3d_cnn_lite', 'temporal_3d_cnn_deep'],
                       help='模型类型')
    parser.add_argument('--sequence_length', type=int, default=5, help='时序序列长度')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')

    args = parser.parse_args()

    # 加载或创建配置
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config(args.model, args.sequence_length)
        # 更新命令行参数
        config['data']['batch_size'] = args.batch_size
        config['training']['epochs'] = args.epochs
        config['optimizer']['lr'] = args.lr
        config['output_dir'] = f"outputs/{args.model}_seq{args.sequence_length}"

    # 添加恢复检查点路径
    if args.resume:
        config['resume_checkpoint'] = args.resume

    # 创建训练器并开始训练
    trainer = TemporalModelTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
