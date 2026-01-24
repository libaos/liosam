#!/usr/bin/env python3
"""
SCStandardSpatialCNN 300轮长期训练脚本
同时训练两个版本：仅空间注意力 vs 空间+通道注意力
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import time
from pathlib import Path
import argparse
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.sc_standard_spatial_cnn import create_sc_standard_spatial_cnn
from utils.spatial_dataset import SpatialScanContextDataset
from utils.simple_contrastive_loss import AdaptiveTripletLoss, SimpleContrastiveLoss
from utils.evaluation_metrics import evaluate_model
from utils import setup_logger, get_timestamp

class SpatialCNNTrainer:
    """空间注意力CNN训练器"""
    
    def __init__(self, config, model_name):
        """
        初始化训练器
        
        参数:
            config (dict): 训练配置
            model_name (str): 模型名称标识
        """
        self.config = config
        self.model_name = model_name
        
        # 设备配置
        if torch.cuda.is_available() and config['device'] != 'cpu':
            if config['device'].isdigit():
                self.device = torch.device(f'cuda:{config["device"]}')
            else:
                self.device = torch.device(config['device'])
        else:
            self.device = torch.device('cpu')
        
        # 设置日志
        self.logger = setup_logger(
            f'spatial_trainer_{model_name}',
            f"training_{model_name}_{get_timestamp()}.log"
        )
        
        self.logger.info(f"🚀 初始化{model_name}训练器")
        self.logger.info(f"📱 使用设备: {self.device}")
        
        # 初始化模型
        self.model = self._create_model()
        
        # 初始化数据加载器
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # 初始化损失函数和优化器
        self.criterion = self._create_criterion()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 训练状态
        self.best_map = 0.0
        self.best_top1 = 0.0
        self.train_losses = []
        self.val_metrics = []
        
        # 检查点保存间隔
        self.save_interval = config.get('save_interval', 50)  # 每50轮保存一次
        
    def _create_model(self):
        """创建模型"""
        use_channel_attention = 'channel' in self.model_name.lower()
        
        model = create_sc_standard_spatial_cnn(
            input_channels=self.config['input_channels'],
            descriptor_dim=self.config['descriptor_dim'],
            use_channel_attention=use_channel_attention
        )
        
        model = model.to(self.device)
        
        # 打印模型信息
        model_info = model.get_model_info()
        self.logger.info(f"🏗️ {self.model_name}模型信息:")
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
        
        # 数据加载器
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
    
    def _create_criterion(self):
        """创建损失函数"""
        return AdaptiveTripletLoss(
            margin=self.config.get('margin', 0.5),
            adaptive_margin=True
        )
    
    def _create_optimizer(self):
        """创建优化器"""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
    
    def _create_scheduler(self):
        """创建学习率调度器 - 适合300轮训练"""
        return optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[100, 200, 250],  # 在100, 200, 250轮时降低学习率
            gamma=0.5
        )
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        if epoch % 10 == 0:  # 每10轮详细记录
            self.logger.info(f"🔄 Epoch {epoch+1}/{self.config['epochs']} - 开始训练")
        
        for batch_idx, (scan_contexts, labels) in enumerate(self.train_loader):
            scan_contexts = scan_contexts.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            descriptors = self.model(scan_contexts)
            
            # 计算损失
            loss = self.criterion(descriptors, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 减少打印频率，避免日志过多
            if epoch % 10 == 0 and (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                self.logger.info(
                    f"   Batch {batch_idx+1}/{num_batches}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Avg Loss: {avg_loss:.4f}"
                )
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        if epoch % 10 == 0:
            self.logger.info(f"✅ Epoch {epoch+1} 训练完成, 平均损失: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self, epoch):
        """验证模型 - 减少验证频率"""
        if epoch % 20 != 0 and epoch != self.config['epochs'] - 1:  # 每20轮验证一次
            return None
            
        self.logger.info(f"🔍 Epoch {epoch+1} - 开始验证")
        
        # 评估模型
        metrics = evaluate_model(
            self.model,
            self.val_loader,
            self.device,
            top_k_list=[1, 3, 5, 10],
            logger=self.logger
        )
        
        self.val_metrics.append({'epoch': epoch, 'metrics': metrics})
        
        # 记录指标
        self.logger.info(f"📊 验证结果:")
        self.logger.info(f"   Top-1: {metrics['top_1']:.4f}")
        self.logger.info(f"   Top-5: {metrics['top_5']:.4f}")
        self.logger.info(f"   mAP: {metrics['mAP']:.4f}")
        
        return metrics
    
    def save_model(self, epoch, metrics=None, is_best=False, is_checkpoint=False):
        """保存模型"""
        timestamp = get_timestamp()
        
        # 模型状态
        model_state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'model_name': self.model_name,
            'metrics': metrics,
            'best_mAP': self.best_map,
            'best_top1': self.best_top1,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }
        
        # 保存路径
        models_dir = Path("outputs/sc_standard_spatial_cnn/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            model_path = models_dir / f"best_{self.model_name}_{timestamp}.pth"
            self.logger.info(f"💾 保存最佳模型: {model_path}")
        elif is_checkpoint:
            model_path = models_dir / f"{self.model_name}_checkpoint_epoch_{epoch+1}_{timestamp}.pth"
            self.logger.info(f"💾 保存检查点: {model_path}")
        else:
            model_path = models_dir / f"{self.model_name}_final_{timestamp}.pth"
            self.logger.info(f"💾 保存最终模型: {model_path}")
        
        torch.save(model_state, model_path)
        
        return model_path
    
    def train(self):
        """完整训练流程"""
        self.logger.info(f"🎯 开始{self.model_name}长期训练 - {self.config['epochs']}轮")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证 (每20轮)
            metrics = self.validate(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            if epoch % 50 == 0:  # 每50轮记录学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"📈 Epoch {epoch+1} 学习率: {current_lr:.6f}")
            
            # 检查是否是最佳模型
            is_best = False
            if metrics and metrics['mAP'] > self.best_map:
                self.best_map = metrics['mAP']
                is_best = True
                self.save_model(epoch, metrics, is_best=True)
                self.logger.info(f"🏆 新的最佳模型! mAP: {self.best_map:.4f}")
            
            if metrics and metrics['top_1'] > self.best_top1:
                self.best_top1 = metrics['top_1']
            
            # 定期保存检查点
            if (epoch + 1) % self.save_interval == 0:
                self.save_model(epoch, metrics, is_checkpoint=True)
        
        # 训练完成
        total_time = time.time() - start_time
        self.logger.info(f"🎉 {self.model_name}训练完成! 总用时: {total_time/3600:.2f}小时")
        self.logger.info(f"🏆 最佳mAP: {self.best_map:.4f}")
        self.logger.info(f"🎯 最佳Top-1: {self.best_top1:.4f}")
        
        # 保存最终模型
        final_model_path = self.save_model(self.config['epochs']-1, metrics)
        
        # 保存训练结果
        self.save_results()
        
        return {
            'model_name': self.model_name,
            'best_mAP': self.best_map,
            'best_top1': self.best_top1,
            'final_model_path': final_model_path,
            'total_time_hours': total_time/3600
        }
    
    def save_results(self):
        """保存训练结果"""
        timestamp = get_timestamp()
        
        results = {
            'model_name': self.model_name,
            'config': self.config,
            'best_mAP': self.best_map,
            'best_top1': self.best_top1,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'model_info': self.model.get_model_info(),
            'timestamp': timestamp,
            'total_epochs': self.config['epochs']
        }
        
        # 保存路径
        results_dir = Path("outputs/sc_standard_spatial_cnn/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = results_dir / f"{self.model_name}_300epochs_results_{timestamp}.json"
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"📊 保存训练结果: {results_path}")

def main():
    """主函数 - 训练指定版本"""
    parser = argparse.ArgumentParser(description='SCStandardSpatialCNN 300轮长期训练')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--max_files', type=int, default=600, help='最大文件数')
    parser.add_argument('--device', type=str, default='cpu', help='设备')
    parser.add_argument('--model_type', choices=['spatial_only', 'spatial_channel'], 
                       default='spatial_channel', help='模型类型')
    
    args = parser.parse_args()
    
    # 基础配置
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_files': args.max_files,
        'device': args.device,
        'data_path': 'data/raw/ply_files',
        'input_channels': 1,
        'descriptor_dim': 256,
        'margin': 0.5,
        'weight_decay': 1e-4,
        'num_workers': 4,
        'augment': True,
        'save_interval': 50
    }
    
    model_name = f"sc_standard_{args.model_type}"
    
    print(f"🚀 开始SCStandardSpatialCNN 300轮训练")
    print(f"📊 配置: {config}")
    print(f"🎯 模型类型: {model_name}")
    
    try:
        trainer = SpatialCNNTrainer(config, model_name)
        result = trainer.train()
        
        print("✅ 训练完成!")
        print(f"🏆 最佳mAP: {result['best_mAP']:.4f}")
        print(f"🎯 最佳Top-1: {result['best_top1']:.4f}")
        print(f"⏱️ 训练时间: {result['total_time_hours']:.2f}小时")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        raise

if __name__ == "__main__":
    main()
