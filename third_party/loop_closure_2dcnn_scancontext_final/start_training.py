#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
开始训练脚本
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import numpy as np
from pathlib import Path
import time
import json

# 导入模型和数据集
from models.temporal_3d_cnn import Temporal3DCNN, Temporal3DCNNLite
from models.temporal_2d_cnn import Temporal2DCNN, Temporal2DCNNLite
from utils.temporal_dataset import TemporalScanContextDataset
from utils.logger import Logger

class SimpleTrainer:
    """简单的训练器"""
    
    def __init__(self, model_type='temporal_3d_cnn', epochs=50, batch_size=8, lr=0.001):
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.output_dir = Path(f"outputs/{model_type}_training")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志
        self.logger = Logger(self.output_dir / 'training.log')
        
        print(f"训练配置:")
        print(f"  模型类型: {model_type}")
        print(f"  训练轮数: {epochs}")
        print(f"  批次大小: {batch_size}")
        print(f"  学习率: {lr}")
        print(f"  设备: {self.device}")
        print(f"  输出目录: {self.output_dir}")
    
    def create_model(self):
        """创建模型"""
        if self.model_type == 'temporal_3d_cnn':
            model = Temporal3DCNN(sequence_length=5, num_classes=20)
        elif self.model_type == 'temporal_3d_cnn_lite':
            model = Temporal3DCNNLite(sequence_length=5, num_classes=20)
        elif self.model_type == 'temporal_2d_cnn':
            model = Temporal2DCNN(sequence_length=5, num_classes=20)
        elif self.model_type == 'temporal_2d_cnn_lite':
            model = Temporal2DCNNLite(sequence_length=5, num_classes=20)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        return model.to(self.device)
    
    def create_dataset(self):
        """创建数据集"""
        
        class SimpleTemporalDataset(TemporalScanContextDataset):
            def _load_data(self):
                # 加载预处理的数据
                data_file = Path("data/processed/temporal_sequences_len5.pkl")
                if data_file.exists():
                    with open(data_file, 'rb') as f:
                        data = pickle.load(f)
                        self.sequences = data['sequences']
                        self.labels = data['labels']
                        self.file_paths = data['file_paths'] if 'file_paths' in data else []
                    print(f"加载了 {len(self.sequences)} 个样本")
                else:
                    raise FileNotFoundError("未找到预处理数据文件")
        
        # 创建数据集
        dataset = SimpleTemporalDataset(
            data_dir="data/processed",
            split='train',
            sequence_length=5,
            use_augmentation=True
        )
        
        # 数据集划分
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"数据集划分:")
        print(f"  训练集: {len(train_dataset)} 样本")
        print(f"  验证集: {len(val_dataset)} 样本")
        print(f"  测试集: {len(test_dataset)} 样本")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, model, val_loader, criterion):
        """验证一个epoch"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, model, optimizer, epoch, val_acc, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'model_type': self.model_type
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pth')
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pth')
            print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
    
    def train(self):
        """开始训练"""
        print("\n开始训练...")
        
        # 创建模型
        model = self.create_model()
        
        # 创建数据集
        train_loader, val_loader, test_loader = self.create_dataset()
        
        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        # 训练记录
        train_history = []
        val_history = []
        best_val_acc = 0.0
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
            
            # 更新学习率
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 保存检查点
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
            
            self.save_checkpoint(model, optimizer, epoch, val_acc, is_best)
            
            # 记录历史
            train_history.append({'epoch': epoch, 'loss': train_loss, 'acc': train_acc})
            val_history.append({'epoch': epoch, 'loss': val_loss, 'acc': val_acc})
            
            # 打印结果
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                  f'Time: {epoch_time:.2f}s, LR: {current_lr:.6f}')
            
            # 记录日志
            self.logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                           f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存训练历史
        history = {
            'train': train_history,
            'val': val_history,
            'best_val_acc': best_val_acc,
            'config': {
                'model_type': self.model_type,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'lr': self.lr
            }
        }
        
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n训练完成！")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        print(f"模型保存在: {self.output_dir}")
        
        return model, best_val_acc


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='开始训练时序模型')
    parser.add_argument('--model', type=str, default='temporal_3d_cnn',
                       choices=['temporal_2d_cnn', 'temporal_2d_cnn_lite', 
                               'temporal_3d_cnn', 'temporal_3d_cnn_lite'],
                       help='模型类型')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    
    args = parser.parse_args()
    
    # 创建训练器并开始训练
    trainer = SimpleTrainer(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    try:
        model, best_acc = trainer.train()
        print(f"✓ {args.model} 训练成功完成！最佳准确率: {best_acc:.2f}%")
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
