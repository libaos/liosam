#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练Temporal 3D CNN模型用于回环检测
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
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from models.temporal_models import Temporal3DCNN
import warnings
warnings.filterwarnings('ignore')

class Temporal3DCNNTrainer:
    """Temporal 3D CNN训练器"""
    
    def __init__(self, sequence_length=5, num_classes=20, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 初始化Temporal 3D CNN训练器")
        print(f"设备: {self.device}")
        print(f"序列长度: {sequence_length}")
        print(f"类别数: {num_classes}")
        
        # 初始化模型
        self.model = Temporal3DCNN(
            input_shape=(1, sequence_length, 20, 60),
            num_classes=num_classes
        )
        self.model = self.model.to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        
        # 训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        print(f"✅ 模型初始化完成")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_data(self, sequence_length=5):
        """加载训练数据"""
        print(f"\n📂 加载序列长度为{sequence_length}的数据...")
        
        data_file = Path(f"data/processed/temporal_sequences_len{sequence_length}.pkl")
        if not data_file.exists():
            print(f"❌ 数据文件不存在: {data_file}")
            return None, None, None, None, None, None
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        # 提取数据
        sequences = np.array(data['sequences'])
        labels = np.array(data['labels'])
        
        print(f"原始数据形状: {sequences.shape}")
        print(f"标签数量: {len(labels)}")
        print(f"类别分布: {np.bincount(labels)}")
        
        # 数据划分
        from sklearn.model_selection import train_test_split
        
        # 先划分训练集和临时集
        train_sequences, temp_sequences, train_labels, temp_labels = train_test_split(
            sequences, labels, test_size=0.4, random_state=42, stratify=labels
        )
        
        # 再将临时集划分为验证集和测试集
        val_sequences, test_sequences, val_labels, test_labels = train_test_split(
            temp_sequences, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        print(f"\n📊 数据划分:")
        print(f"训练集: {len(train_sequences)} 样本")
        print(f"验证集: {len(val_sequences)} 样本")
        print(f"测试集: {len(test_sequences)} 样本")
        
        return train_sequences, val_sequences, test_sequences, train_labels, val_labels, test_labels
    
    def create_data_loaders(self, train_sequences, val_sequences, test_sequences, 
                           train_labels, val_labels, test_labels, batch_size=32):
        """创建数据加载器"""
        print(f"\n🔄 创建数据加载器 (batch_size={batch_size})...")
        
        # 转换为PyTorch张量
        train_tensor = torch.FloatTensor(train_sequences).to(self.device)
        val_tensor = torch.FloatTensor(val_sequences).to(self.device)
        test_tensor = torch.FloatTensor(test_sequences).to(self.device)
        
        train_labels_tensor = torch.LongTensor(train_labels).to(self.device)
        val_labels_tensor = torch.LongTensor(val_labels).to(self.device)
        test_labels_tensor = torch.LongTensor(test_labels).to(self.device)
        
        # 创建数据集
        train_dataset = TensorDataset(train_tensor, train_labels_tensor)
        val_dataset = TensorDataset(val_tensor, val_labels_tensor)
        test_dataset = TensorDataset(test_tensor, test_labels_tensor)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"✅ 数据加载器创建完成")
        print(f"训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(val_loader)}")
        print(f"测试批次数: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            # 前向传播
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'  批次 {batch_idx:3d}/{len(train_loader):3d} | '
                      f'损失: {loss.item():.4f} | '
                      f'准确率: {100.*correct/total:.2f}%')
        
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
    
    def train(self, train_loader, val_loader, epochs=100, save_best=True):
        """训练模型"""
        print(f"\n🎯 开始训练 (epochs={epochs})...")
        
        best_val_acc = 0
        best_model_state = None
        patience = 15
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
            
            print(f'\nEpoch {epoch+1:3d}/{epochs:3d} | 时间: {epoch_time:.1f}s')
            print(f'训练 - 损失: {train_loss:.4f} | 准确率: {train_acc:.2f}%')
            print(f'验证 - 损失: {val_loss:.4f} | 准确率: {val_acc:.2f}%')
            print(f'学习率: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f'🎉 新的最佳验证准确率: {best_val_acc:.2f}%')
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                print(f'\n⏹️  早停触发 (patience={patience})')
                break
            
            print('-' * 60)
        
        total_time = time.time() - start_time
        print(f'\n✅ 训练完成!')
        print(f'总训练时间: {total_time:.1f}秒')
        print(f'最佳验证准确率: {best_val_acc:.2f}%')
        
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
        total_loss = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 计算指标
        test_loss = total_loss / len(test_loader)
        test_acc = accuracy_score(all_targets, all_predictions) * 100
        
        print(f'测试损失: {test_loss:.4f}')
        print(f'测试准确率: {test_acc:.2f}%')
        
        # 详细报告
        print(f"\n📊 分类报告:")
        print(classification_report(all_targets, all_predictions))
        
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
    
    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='训练损失', color='blue')
        ax1.plot(self.val_losses, label='验证损失', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.set_title('训练和验证损失')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(self.train_accuracies, label='训练准确率', color='blue')
        ax2.plot(self.val_accuracies, label='验证准确率', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('准确率 (%)')
        ax2.set_title('训练和验证准确率')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'✅ 训练历史图已保存到: {save_path}')
        
        plt.show()

def main():
    """主函数"""
    print("="*80)
    print("🚀 Temporal 3D CNN 训练开始")
    print("="*80)
    
    # 训练参数
    sequence_length = 5
    num_classes = 20
    learning_rate = 0.001
    batch_size = 16  # 3D CNN内存占用较大，使用较小的batch size
    epochs = 100
    
    # 创建训练器
    trainer = Temporal3DCNNTrainer(
        sequence_length=sequence_length,
        num_classes=num_classes,
        learning_rate=learning_rate
    )
    
    # 加载数据
    train_seq, val_seq, test_seq, train_labels, val_labels, test_labels = trainer.load_data(sequence_length)
    
    if train_seq is None:
        print("❌ 数据加载失败")
        return
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = trainer.create_data_loaders(
        train_seq, val_seq, test_seq, train_labels, val_labels, test_labels, batch_size
    )
    
    # 训练模型
    best_val_acc = trainer.train(train_loader, val_loader, epochs)
    
    # 测试模型
    test_acc, predictions, targets = trainer.test(test_loader)
    
    # 保存模型
    model_path = f"models/saved/temporal_3dcnn_seq{sequence_length}_acc{test_acc:.1f}.pth"
    metadata = {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'batch_size': batch_size,
        'epochs': epochs
    }
    trainer.save_model(model_path, metadata)
    
    # 绘制训练历史
    plot_path = f"outputs/temporal_3dcnn_seq{sequence_length}_training_history.png"
    trainer.plot_training_history(plot_path)
    
    print(f"\n🎉 训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"模型保存路径: {model_path}")

if __name__ == '__main__':
    main()
