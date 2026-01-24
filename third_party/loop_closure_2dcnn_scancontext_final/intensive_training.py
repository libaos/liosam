#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
加强训练脚本 - 更大训练量和更多实验
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
from datetime import datetime

# 导入模型和数据集
from models.temporal_3d_cnn import Temporal3DCNN, Temporal3DCNNLite, Temporal3DCNNDeep
from models.temporal_2d_cnn import Temporal2DCNN, Temporal2DCNNLite, Temporal2DCNNResNet
from utils.temporal_dataset import TemporalScanContextDataset
from utils.logger import Logger

class IntensiveTrainer:
    """加强训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"outputs/intensive_training_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志
        self.logger = Logger(self.output_dir / 'intensive_training.log')
        
        print(f"🚀 加强训练配置:")
        print(f"  设备: {self.device}")
        print(f"  输出目录: {self.output_dir}")
        print(f"  训练轮数: {config['epochs']}")
        print(f"  批次大小: {config['batch_size']}")
        print(f"  学习率: {config['lr']}")
    
    def create_model(self, model_type):
        """创建模型"""
        models = {
            'temporal_3d_cnn': Temporal3DCNN(sequence_length=5, num_classes=20),
            'temporal_3d_cnn_lite': Temporal3DCNNLite(sequence_length=5, num_classes=20),
            'temporal_3d_cnn_deep': Temporal3DCNNDeep(sequence_length=5, num_classes=20),
            'temporal_2d_cnn': Temporal2DCNN(sequence_length=5, num_classes=20),
            'temporal_2d_cnn_lite': Temporal2DCNNLite(sequence_length=5, num_classes=20),
            'temporal_2d_cnn_resnet': Temporal2DCNNResNet(sequence_length=5, num_classes=20)
        }
        
        if model_type not in models:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return models[model_type].to(self.device)
    
    def create_dataset(self, use_all_sequences=True):
        """创建数据集 - 使用更多数据"""
        
        class IntensiveTemporalDataset(TemporalScanContextDataset):
            def _load_data(self):
                # 加载所有长度的序列数据
                all_sequences = []
                all_labels = []
                all_file_paths = []
                
                if use_all_sequences:
                    # 使用多种序列长度的数据
                    for seq_len in [3, 5, 7, 10]:
                        data_file = Path(f"data/processed/temporal_sequences_len{seq_len}.pkl")
                        if data_file.exists():
                            with open(data_file, 'rb') as f:
                                data = pickle.load(f)
                                # 将不同长度的序列填充或截断到统一长度5
                                for seq in data['sequences']:
                                    if seq.shape[0] < 5:
                                        # 填充
                                        padded_seq = np.zeros((5, seq.shape[1], seq.shape[2]))
                                        padded_seq[:seq.shape[0]] = seq
                                        all_sequences.append(padded_seq)
                                    elif seq.shape[0] > 5:
                                        # 截断
                                        all_sequences.append(seq[:5])
                                    else:
                                        all_sequences.append(seq)
                                
                                all_labels.extend(data['labels'])
                                all_file_paths.extend(data.get('file_paths', []))
                            
                            print(f"加载序列长度 {seq_len}: {len(data['sequences'])} 个样本")
                else:
                    # 只使用长度为5的序列
                    data_file = Path("data/processed/temporal_sequences_len5.pkl")
                    if data_file.exists():
                        with open(data_file, 'rb') as f:
                            data = pickle.load(f)
                            all_sequences = data['sequences']
                            all_labels = data['labels']
                            all_file_paths = data.get('file_paths', [])
                
                self.sequences = all_sequences
                self.labels = all_labels
                self.file_paths = all_file_paths
                print(f"总共加载了 {len(self.sequences)} 个样本")
        
        # 创建数据集
        dataset = IntensiveTemporalDataset(
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
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"数据集划分:")
        print(f"  训练集: {len(train_dataset)} 样本")
        print(f"  验证集: {len(val_dataset)} 样本")
        print(f"  测试集: {len(test_dataset)} 样本")
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, model_type):
        """训练单个模型"""
        print(f"\n🔥 开始训练 {model_type}...")
        
        # 创建模型
        model = self.create_model(model_type)
        
        # 创建数据集
        train_loader, val_loader, test_loader = self.create_dataset(
            use_all_sequences=self.config.get('use_all_sequences', True)
        )
        
        # 优化器和损失函数
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config['lr'],
            weight_decay=self.config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        # 训练记录
        train_history = []
        val_history = []
        best_val_acc = 0.0
        patience = 0
        max_patience = 30
        
        # 训练循环
        for epoch in range(self.config['epochs']):
            start_time = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, epoch
            )
            
            # 验证
            val_loss, val_acc = self.validate_epoch(
                model, val_loader, criterion
            )
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                # 保存最佳模型
                self.save_checkpoint(model, optimizer, epoch, val_acc, model_type, is_best=True)
            else:
                patience += 1
            
            # 保存最新检查点
            if epoch % 10 == 0:
                self.save_checkpoint(model, optimizer, epoch, val_acc, model_type, is_best=False)
            
            # 记录历史
            train_history.append({'epoch': epoch, 'loss': train_loss, 'acc': train_acc})
            val_history.append({'epoch': epoch, 'loss': val_loss, 'acc': val_acc})
            
            # 打印结果
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                  f'Time: {epoch_time:.2f}s, LR: {current_lr:.6f}, Patience: {patience}')
            
            # 记录日志
            self.logger.info(
                f'{model_type} Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
            )
            
            # 早停
            if patience >= max_patience:
                print(f"早停触发，在第 {epoch} 轮停止训练")
                break
        
        # 保存训练历史
        history = {
            'model_type': model_type,
            'train': train_history,
            'val': val_history,
            'best_val_acc': best_val_acc,
            'config': self.config
        }
        
        with open(self.output_dir / f'{model_type}_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n✅ {model_type} 训练完成！最佳验证准确率: {best_val_acc:.2f}%")
        return model, best_val_acc, history
    
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
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 20 == 0:
                print(f'  Batch {batch_idx:3d}/{len(train_loader)}: Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
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
    
    def save_checkpoint(self, model, optimizer, epoch, val_acc, model_type, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'model_type': model_type,
            'config': self.config
        }
        
        if is_best:
            torch.save(checkpoint, self.output_dir / f'{model_type}_best.pth')
        else:
            torch.save(checkpoint, self.output_dir / f'{model_type}_epoch_{epoch}.pth')
    
    def run_intensive_training(self):
        """运行加强训练"""
        print("🚀 开始加强训练实验...")
        
        # 要训练的模型列表
        models_to_train = [
            'temporal_3d_cnn',
            'temporal_3d_cnn_deep',
            'temporal_2d_cnn',
            'temporal_2d_cnn_resnet'
        ]
        
        results = {}
        
        for model_type in models_to_train:
            try:
                model, best_acc, history = self.train_model(model_type)
                results[model_type] = {
                    'best_val_acc': best_acc,
                    'total_epochs': len(history['train']),
                    'final_train_acc': history['train'][-1]['acc'],
                    'final_val_acc': history['val'][-1]['acc']
                }
                
                # 清理GPU内存
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"❌ 训练 {model_type} 时出错: {e}")
                continue
        
        # 保存总结果
        with open(self.output_dir / 'intensive_results_summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # 打印总结
        print("\n" + "="*80)
        print("🏆 加强训练结果总结")
        print("="*80)
        
        for model_type, result in results.items():
            print(f"{model_type:<25}: 最佳验证准确率 {result['best_val_acc']:.2f}%, "
                  f"训练轮数 {result['total_epochs']}")
        
        return results


def main():
    """主函数"""
    # 加强训练配置
    intensive_config = {
        'epochs': 150,           # 增加到150轮
        'batch_size': 16,        # 增加批次大小
        'lr': 0.001,            # 初始学习率
        'weight_decay': 1e-4,   # 权重衰减
        'use_all_sequences': True  # 使用所有序列长度的数据
    }
    
    print("🚀 启动加强训练...")
    print(f"配置: {intensive_config}")
    
    # 创建训练器
    trainer = IntensiveTrainer(intensive_config)
    
    # 开始训练
    results = trainer.run_intensive_training()
    
    print("\n✅ 加强训练完成！")


if __name__ == '__main__':
    main()
