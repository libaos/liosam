#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练轨迹进展预测模型 - 预测当前处于轨迹的第几段（0-19）
基于时间顺序的真实标签
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import time
from sklearn.metrics import accuracy_score, classification_report
from models.temporal_models import Temporal3DCNN
from utils.ply_reader import PLYReader
from utils.scan_context import ScanContext
import glob
import warnings
warnings.filterwarnings('ignore')

class TrajectoryProgressTrainer:
    """轨迹进展预测训练器"""
    
    def __init__(self, sequence_length=5, num_classes=20, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🎯 轨迹进展预测训练器")
        print(f"设备: {self.device}")
        print(f"目标: 预测轨迹进展 0→1→2→...→19")
        
        # 初始化模型
        self.model = Temporal3DCNN(
            input_shape=(1, sequence_length, 20, 60),
            num_classes=num_classes
        )
        self.model = self.model.to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        
        # 训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        print(f"✅ 模型初始化完成")
    
    def create_trajectory_progress_dataset(self, data_dir, sequence_length=5):
        """基于时间顺序创建轨迹进展数据集"""
        print(f"📂 创建轨迹进展数据集...")
        
        # 获取所有ply文件并按时间排序
        ply_files = sorted(glob.glob(f"{data_dir}/*.ply"))
        print(f"找到 {len(ply_files)} 个ply文件")
        
        if len(ply_files) == 0:
            print("❌ 未找到ply文件")
            return None, None
        
        # 生成ScanContext特征
        sc_generator = ScanContext()
        scan_contexts = []
        
        print("生成ScanContext特征...")
        for i, ply_file in enumerate(ply_files):
            if i % 100 == 0:
                print(f"  处理 {i+1}/{len(ply_files)}")
            
            try:
                points = PLYReader.read_ply_file(ply_file)
                if points is not None and len(points) > 100:
                    points = points[:, :3]  # 只取x,y,z
                    sc = sc_generator.generate_scan_context(points)
                    scan_contexts.append(sc)
                else:
                    scan_contexts.append(None)
            except Exception as e:
                print(f"处理失败 {ply_file}: {e}")
                scan_contexts.append(None)
        
        # 创建时序序列和基于时间的标签
        sequences = []
        labels = []
        
        print("创建时序序列和时间标签...")
        total_files = len(scan_contexts)
        
        for i in range(len(scan_contexts) - sequence_length + 1):
            # 检查序列中的所有ScanContext都有效
            sequence_scs = scan_contexts[i:i+sequence_length]
            if all(sc is not None for sc in sequence_scs):
                # 计算当前位置在整个轨迹中的进展
                middle_idx = i + sequence_length // 2
                progress = int((middle_idx / total_files) * self.num_classes)
                progress = min(progress, self.num_classes - 1)  # 确保不超过19
                
                sequence = np.stack(sequence_scs, axis=0)
                sequences.append(sequence)
                labels.append(progress)
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        print(f"创建了 {len(sequences)} 个序列")
        print(f"标签分布: {np.bincount(labels)}")
        
        return sequences, labels
    
    def create_data_loaders(self, sequences, labels, batch_size=16):
        """创建数据加载器"""
        print(f"🔄 创建数据加载器...")
        
        # 数据划分
        from sklearn.model_selection import train_test_split
        
        train_seq, temp_seq, train_labels, temp_labels = train_test_split(
            sequences, labels, test_size=0.4, random_state=42, stratify=labels
        )
        
        val_seq, test_seq, val_labels, test_labels = train_test_split(
            temp_seq, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        print(f"训练集: {len(train_seq)} 样本")
        print(f"验证集: {len(val_seq)} 样本")
        print(f"测试集: {len(test_seq)} 样本")
        
        # 转换为PyTorch张量
        train_tensor = torch.FloatTensor(train_seq).to(self.device)
        val_tensor = torch.FloatTensor(val_seq).to(self.device)
        test_tensor = torch.FloatTensor(test_seq).to(self.device)
        
        train_labels_tensor = torch.LongTensor(train_labels).to(self.device)
        val_labels_tensor = torch.LongTensor(val_labels).to(self.device)
        test_labels_tensor = torch.LongTensor(test_labels).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(train_tensor, train_labels_tensor)
        val_dataset = TensorDataset(val_tensor, val_labels_tensor)
        test_dataset = TensorDataset(test_tensor, test_labels_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 5 == 0:
                print(f'  批次 {batch_idx:2d}/{len(train_loader):2d} | '
                      f'损失: {loss.item():.4f} | '
                      f'准确率: {100.*correct/total:.1f}%')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50):
        """训练模型"""
        print(f"\n🎯 开始训练 (epochs={epochs})...")
        
        best_val_acc = 0
        best_model_state = None
        patience = 10
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            print(f'\nEpoch {epoch+1:2d}/{epochs:2d} | 时间: {epoch_time:.1f}s')
            print(f'训练 - 损失: {train_loss:.4f} | 准确率: {train_acc:.1f}%')
            print(f'验证 - 损失: {val_loss:.4f} | 准确率: {val_acc:.1f}%')
            print(f'学习率: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f'🎉 新的最佳验证准确率: {best_val_acc:.1f}%')
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                print(f'\n⏹️  早停触发 (patience={patience})')
                break
            
            print('-' * 50)
        
        total_time = time.time() - start_time
        print(f'\n✅ 训练完成!')
        print(f'总训练时间: {total_time:.1f}秒')
        print(f'最佳验证准确率: {best_val_acc:.1f}%')
        
        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print('✅ 已加载最佳模型权重')
        
        return best_val_acc
    
    def test(self, test_loader):
        """测试模型"""
        print(f"\n🧪 测试模型...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        test_acc = accuracy_score(all_targets, all_predictions) * 100
        print(f'测试准确率: {test_acc:.1f}%')
        
        return test_acc, all_predictions, all_targets
    
    def save_model(self, filepath, metadata=None):
        """保存模型"""
        save_dir = Path(filepath).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'sequence_length': self.sequence_length,
            'num_classes': self.num_classes,
            'learning_rate': self.learning_rate,
        }
        
        if metadata:
            save_dict.update(metadata)
        
        torch.save(save_dict, filepath)
        print(f'✅ 模型已保存到: {filepath}')

def main():
    """主函数"""
    print("="*80)
    print("🎯 轨迹进展预测模型训练")
    print("="*80)
    
    # 数据路径
    data_dir = "/mysda/shared_dir/2025.7.3/2025-07-03-16-28-57.ply"
    
    # 训练参数
    sequence_length = 5
    num_classes = 20
    learning_rate = 0.001
    batch_size = 16
    epochs = 50
    
    # 创建训练器
    trainer = TrajectoryProgressTrainer(
        sequence_length=sequence_length,
        num_classes=num_classes,
        learning_rate=learning_rate
    )
    
    # 创建数据集
    sequences, labels = trainer.create_trajectory_progress_dataset(data_dir, sequence_length)
    
    if sequences is None:
        print("❌ 数据集创建失败")
        return
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = trainer.create_data_loaders(
        sequences, labels, batch_size
    )
    
    # 训练模型
    best_val_acc = trainer.train(train_loader, val_loader, epochs)
    
    # 测试模型
    test_acc, predictions, targets = trainer.test(test_loader)
    
    # 保存模型
    model_path = f"models/saved/trajectory_progress_model_acc{test_acc:.1f}.pth"
    metadata = {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'batch_size': batch_size,
        'epochs': epochs,
        'data_type': 'trajectory_progress'
    }
    trainer.save_model(model_path, metadata)
    
    print(f"\n🎉 训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.1f}%")
    print(f"测试准确率: {test_acc:.1f}%")
    print(f"模型保存路径: {model_path}")

if __name__ == '__main__':
    main()
