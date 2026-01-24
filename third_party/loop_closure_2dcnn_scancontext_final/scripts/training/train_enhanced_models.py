#!/usr/bin/env python3
"""
增强版SC Standard CNN训练脚本
测试多种注意力机制组合的性能
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import json
import time
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from models.sc_standard_enhanced_cnn import create_enhanced_model
from utils.spatial_dataset import SpatialScanContextDataset
from utils.simple_contrastive_loss import AdaptiveTripletLoss
from utils.evaluation_metrics import compute_retrieval_metrics
from utils.logger import setup_logger

class EnhancedModelTrainer:
    """增强版模型训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # 设置日志
        self.logger = setup_logger(
            f"enhanced_trainer_{config['model_type']}", 
            f"training_enhanced_{config['model_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        self.logger.info(f"🚀 初始化增强版{config['model_type']}训练器")
        self.logger.info(f"📱 使用设备: {self.device}")
        
        # 创建模型
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 创建数据加载器
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # 创建优化器和损失函数
        self.optimizer = self._create_optimizer()
        self.criterion = self._create_criterion()
        self.scheduler = self._create_scheduler()
        
        # 训练状态
        self.best_top1 = 0.0
        self.best_map = 0.0
        self.train_losses = []
        self.val_metrics = []
    
    def _create_model(self):
        """创建增强版模型"""
        model = create_enhanced_model(
            model_type=self.config['model_type'],
            input_channels=self.config['input_channels'],
            descriptor_dim=self.config['descriptor_dim'],
            reduction=self.config.get('reduction', 16),
            dropout_rate=self.config.get('dropout_rate', 0.3)
        )
        
        # 记录模型信息
        model_info = model.get_model_info()
        self.logger.info(f"🏗️ {self.config['model_type']}模型信息:")
        for key, value in model_info.items():
            self.logger.info(f"   {key}: {value}")
        
        return model
    
    def _create_data_loaders(self):
        """创建数据加载器"""
        self.logger.info("📂 创建数据加载器...")
        
        # 训练数据集
        train_dataset = SpatialScanContextDataset(
            data_dir=self.config['data_path'],
            split='train',
            max_files=self.config.get('max_files', None),
            use_augmentation=self.config.get('augment', True)
        )
        
        # 验证数据集
        val_dataset = SpatialScanContextDataset(
            data_dir=self.config['data_path'],
            split='val',
            max_files=self.config.get('max_files', None),
            use_augmentation=False
        )
        
        self.logger.info(f"📊 训练集大小: {len(train_dataset)}")
        self.logger.info(f"📊 验证集大小: {len(val_dataset)}")
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def _create_optimizer(self):
        """创建优化器"""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
    
    def _create_criterion(self):
        """创建损失函数"""
        return AdaptiveTripletLoss(
            margin=self.config.get('margin', 0.5),
            adaptive_margin=True
        )
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        return optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[self.config['epochs']//3, 2*self.config['epochs']//3],
            gamma=0.5
        )
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        self.logger.info(f"🔄 Epoch {epoch+1}/{self.config['epochs']} - 开始训练")
        
        for batch_idx, (data, labels) in enumerate(self.train_loader):
            data, labels = data.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            embeddings = self.model(data)
            loss = self.criterion(embeddings, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 记录进度
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                self.logger.info(f"   Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        self.logger.info(f"✅ Epoch {epoch+1} 训练完成, 平均损失: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        
        self.logger.info(f"🔍 Epoch {epoch+1} - 开始验证")
        self.logger.info("🔍 开始提取特征...")
        
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in self.val_loader:
                data = data.to(self.device)
                embeddings = self.model(data)
                
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels.cpu())
        
        # 合并所有特征和标签
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        self.logger.info(f"✅ 特征提取完成，共 {len(all_embeddings)} 个样本")
        
        # 计算检索指标
        metrics = compute_retrieval_metrics(all_embeddings, all_labels)
        
        # 记录指标
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"   {key}: {value:.4f}")
            else:
                self.logger.info(f"   {key}: {value}")
        
        # 保存验证指标
        self.val_metrics.append({
            'epoch': epoch,
            'metrics': metrics
        })
        
        # 更新最佳指标
        if metrics['top_1'] > self.best_top1:
            self.best_top1 = metrics['top_1']
            self._save_best_model(epoch, 'top1')
        
        if metrics['mAP'] > self.best_map:
            self.best_map = metrics['mAP']
            self._save_best_model(epoch, 'map')
        
        self.logger.info("📊 验证结果:")
        self.logger.info(f"   Top-1: {metrics['top_1']:.4f}")
        self.logger.info(f"   Top-5: {metrics['top_5']:.4f}")
        self.logger.info(f"   mAP: {metrics['mAP']:.4f}")
        
        return metrics
    
    def _save_best_model(self, epoch, metric_type):
        """保存最佳模型"""
        os.makedirs(f"outputs/enhanced_{self.config['model_type']}/models", exist_ok=True)
        
        model_path = f"outputs/enhanced_{self.config['model_type']}/models/best_{metric_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_top1': self.best_top1,
            'best_map': self.best_map,
            'config': self.config
        }, model_path)
        
        self.logger.info(f"💾 保存最佳{metric_type}模型: {model_path}")
    
    def train(self):
        """开始训练"""
        self.logger.info(f"🎯 开始{self.config['model_type']}训练 - {self.config['epochs']}轮")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            if (epoch + 1) % self.config.get('val_interval', 20) == 0:
                val_metrics = self.validate(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"📈 Epoch {epoch+1} 学习率: {current_lr:.6f}")
        
        # 最终验证
        final_metrics = self.validate(self.config['epochs'] - 1)
        
        # 计算总训练时间
        total_time = time.time() - start_time
        total_hours = total_time / 3600
        
        self.logger.info(f"🎉 {self.config['model_type']}训练完成! 总用时: {total_hours:.2f}小时")
        self.logger.info(f"🏆 最佳mAP: {self.best_map:.4f}")
        self.logger.info(f"🎯 最佳Top-1: {self.best_top1:.4f}")
        
        # 保存训练结果
        self._save_results(total_hours)
        
        return {
            'best_top1': self.best_top1,
            'best_map': self.best_map,
            'total_time': total_hours,
            'final_metrics': final_metrics
        }
    
    def _save_results(self, total_time):
        """保存训练结果"""
        os.makedirs(f"outputs/enhanced_{self.config['model_type']}/results", exist_ok=True)
        
        results = {
            'model_type': self.config['model_type'],
            'config': self.config,
            'best_map': self.best_map,
            'best_top1': self.best_top1,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'model_info': self.model.get_model_info(),
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'total_time_hours': total_time
        }
        
        results_path = f"outputs/enhanced_{self.config['model_type']}/results/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📊 保存训练结果: {results_path}")

def main():
    parser = argparse.ArgumentParser(description='增强版SC Standard CNN训练')
    parser.add_argument('--model_type', type=str, default='cbam', 
                       choices=['cbam', 'eca', 'se', 'simam', 'dual', 'triple', 'all'],
                       help='注意力机制类型')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--device', type=str, default='cpu', help='设备')
    parser.add_argument('--data_path', type=str, default='data/raw/ply_files', help='数据路径')
    parser.add_argument('--max_files', type=int, default=None, help='最大文件数')
    
    args = parser.parse_args()
    
    config = {
        'model_type': args.model_type,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'device': args.device,
        'data_path': args.data_path,
        'max_files': args.max_files,
        'input_channels': 1,
        'descriptor_dim': 256,
        'reduction': 16,
        'dropout_rate': 0.3,
        'margin': 0.5,
        'weight_decay': 1e-4,
        'num_workers': 4,
        'augment': True,
        'val_interval': 20
    }
    
    print(f"🚀 开始训练增强版{args.model_type}模型")
    print(f"📊 配置: {config}")
    
    trainer = EnhancedModelTrainer(config)
    results = trainer.train()
    
    print(f"✅ 训练完成!")
    print(f"🏆 最佳mAP: {results['best_map']:.4f}")
    print(f"🎯 最佳Top-1: {results['best_top1']:.4f}")
    print(f"⏱️ 总用时: {results['total_time']:.2f}小时")

if __name__ == "__main__":
    main()
