#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练2D CNN轨迹分段预测模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from models.cnn_2d_models import Simple2DCNN, Enhanced2DCNN, ResNet2D
from utils.ply_reader import PLYReader
from utils.scan_context import ScanContext
import glob
import warnings
warnings.filterwarnings('ignore')

class CNN2DTrajectoryTrainer:
    """2D CNN轨迹分段预测训练器"""
    
    def __init__(self, model_type='Enhanced2DCNN', num_classes=20, learning_rate=0.001):
        self.model_type = model_type
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🎯 2D CNN轨迹分段预测训练器")
        print(f"设备: {self.device}")
        print(f"模型类型: {model_type}")
        print(f"目标: 提高轨迹分段预测准确率")
        
        # 初始化模型
        if model_type == 'Simple2DCNN':
            self.model = Simple2DCNN(num_classes=num_classes)
        elif model_type == 'Enhanced2DCNN':
            self.model = Enhanced2DCNN(num_classes=num_classes)
        elif model_type == 'ResNet2D':
            self.model = ResNet2D(num_classes=num_classes)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
        
        self.model = self.model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型参数数量: {total_params:,}")
        
        # 训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.class_accuracies = []
        
        print(f"✅ 模型初始化完成")
    
    def create_2d_dataset(self, data_dir):
        """创建2D CNN数据集（单帧ScanContext）"""
        print(f"📂 创建2D CNN数据集...")
        
        # 获取所有ply文件并按时间排序
        ply_files = sorted(glob.glob(f"{data_dir}/*.ply"))
        print(f"找到 {len(ply_files)} 个ply文件")
        
        if len(ply_files) == 0:
            print("❌ 未找到ply文件")
            return None, None, None
        
        # 生成ScanContext特征
        sc_generator = ScanContext()
        scan_contexts = []
        labels = []
        
        print("生成ScanContext特征和标签...")
        total_files = len(ply_files)
        target_samples_per_class = 150  # 每个类别目标样本数
        
        for i, ply_file in enumerate(ply_files):
            if i % 100 == 0:
                print(f"  处理 {i+1}/{len(ply_files)}")
            
            try:
                points = PLYReader.read_ply_file(ply_file)
                if points is not None and len(points) > 100:
                    points = points[:, :3]  # 只取x,y,z
                    sc = sc_generator.generate_scan_context(points)
                    
                    # 计算基于时间进展的标签
                    progress = int((i / total_files) * self.num_classes)
                    progress = min(progress, self.num_classes - 1)
                    
                    scan_contexts.append(sc)
                    labels.append(progress)
                    
            except Exception as e:
                print(f"处理失败 {ply_file}: {e}")
                continue
        
        print(f"生成了 {len(scan_contexts)} 个ScanContext特征")
        
        # 平衡数据集
        print("平衡数据集...")
        balanced_contexts = []
        balanced_labels = []
        
        for class_id in range(self.num_classes):
            class_indices = [i for i, label in enumerate(labels) if label == class_id]
            
            if len(class_indices) > 0:
                if len(class_indices) < target_samples_per_class:
                    # 数据增强：重复采样
                    indices = np.random.choice(class_indices, target_samples_per_class, replace=True)
                else:
                    # 随机采样
                    indices = np.random.choice(class_indices, target_samples_per_class, replace=False)
                
                for idx in indices:
                    balanced_contexts.append(scan_contexts[idx])
                    balanced_labels.append(class_id)
                
                print(f"类别 {class_id}: {len(class_indices)} -> {len(indices)} 样本")
        
        scan_contexts = np.array(balanced_contexts)
        labels = np.array(balanced_labels)
        
        print(f"平衡后数据集: {len(scan_contexts)} 个样本")
        print(f"标签分布: {np.bincount(labels)}")
        
        # 计算类别权重
        class_weights = compute_class_weight('balanced', 
                                           classes=np.unique(labels), 
                                           y=labels)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        return scan_contexts, labels, class_weights
    
    def create_data_loaders(self, scan_contexts, labels, class_weights, batch_size=32):
        """创建数据加载器"""
        print(f"🔄 创建数据加载器...")
        
        # 数据划分
        from sklearn.model_selection import train_test_split
        
        train_contexts, temp_contexts, train_labels, temp_labels = train_test_split(
            scan_contexts, labels, test_size=0.4, random_state=42, stratify=labels
        )
        
        val_contexts, test_contexts, val_labels, test_labels = train_test_split(
            temp_contexts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        print(f"训练集: {len(train_contexts)} 样本")
        print(f"验证集: {len(val_contexts)} 样本")
        print(f"测试集: {len(test_contexts)} 样本")
        
        # 转换为PyTorch张量 (添加通道维度)
        train_tensor = torch.FloatTensor(train_contexts).unsqueeze(1).to(self.device)  # (N, 1, 20, 60)
        val_tensor = torch.FloatTensor(val_contexts).unsqueeze(1).to(self.device)
        test_tensor = torch.FloatTensor(test_contexts).unsqueeze(1).to(self.device)
        
        train_labels_tensor = torch.LongTensor(train_labels).to(self.device)
        val_labels_tensor = torch.LongTensor(val_labels).to(self.device)
        test_labels_tensor = torch.LongTensor(test_labels).to(self.device)
        
        # 创建加权采样器
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # 创建数据加载器
        train_dataset = TensorDataset(train_tensor, train_labels_tensor)
        val_dataset = TensorDataset(val_tensor, val_labels_tensor)
        test_dataset = TensorDataset(test_tensor, test_labels_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, train_loader, val_loader, class_weights, epochs=60):
        """训练模型"""
        print(f"\n🎯 开始2D CNN训练 (epochs={epochs})...")
        
        # 使用加权损失函数
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        
        best_avg_acc = 0
        best_model_state = None
        patience = 15
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc, class_accs = self.validate_with_class_accuracy(val_loader)
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.class_accuracies.append(class_accs)
            
            epoch_time = time.time() - epoch_start
            
            # 计算平均类别准确率
            avg_class_acc = np.mean([acc for acc in class_accs.values() if acc > 0])
            
            print(f'\nEpoch {epoch+1:2d}/{epochs:2d} | 时间: {epoch_time:.1f}s')
            print(f'训练 - 损失: {train_loss:.4f} | 准确率: {train_acc:.1f}%')
            print(f'验证 - 损失: {val_loss:.4f} | 准确率: {val_acc:.1f}%')
            print(f'平均类别准确率: {avg_class_acc:.1f}%')
            print(f'学习率: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 显示最差的几个类别
            sorted_classes = sorted(class_accs.items(), key=lambda x: x[1])
            print(f'最差类别: {sorted_classes[:3]}')
            
            # 保存最佳模型（基于平均类别准确率）
            if avg_class_acc > best_avg_acc:
                best_avg_acc = avg_class_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f'🎉 新的最佳平均类别准确率: {best_avg_acc:.1f}%')
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
        print(f'最佳平均类别准确率: {best_avg_acc:.1f}%')
        
        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print('✅ 已加载最佳模型权重')
        
        return best_avg_acc
    
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
            
            if batch_idx % 10 == 0:
                print(f'  批次 {batch_idx:2d}/{len(train_loader):2d} | '
                      f'损失: {loss.item():.4f} | '
                      f'准确率: {100.*correct/total:.1f}%')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_with_class_accuracy(self, val_loader):
        """验证模型并计算每个类别的准确率"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        overall_accuracy = accuracy_score(all_targets, all_predictions) * 100
        
        # 计算每个类别的准确率
        class_accuracies = {}
        for class_id in range(self.num_classes):
            class_mask = np.array(all_targets) == class_id
            if np.sum(class_mask) > 0:
                class_predictions = np.array(all_predictions)[class_mask]
                class_targets = np.array(all_targets)[class_mask]
                class_acc = accuracy_score(class_targets, class_predictions) * 100
                class_accuracies[class_id] = class_acc
            else:
                class_accuracies[class_id] = 0
        
        return avg_loss, overall_accuracy, class_accuracies
    
    def test_detailed_analysis(self, test_loader):
        """详细测试分析"""
        print(f"\n🧪 详细测试分析...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_confidences.extend(confidence.cpu().numpy())
        
        # 总体准确率
        overall_acc = accuracy_score(all_targets, all_predictions) * 100
        print(f'总体测试准确率: {overall_acc:.1f}%')
        
        # 每个类别的详细分析
        print(f"\n📊 每个类别的详细分析:")
        print(f"{'类别':<4} {'准确率':<8} {'样本数':<6} {'平均置信度':<10}")
        print("-" * 35)
        
        class_accuracies = []
        for class_id in range(self.num_classes):
            class_mask = np.array(all_targets) == class_id
            if np.sum(class_mask) > 0:
                class_predictions = np.array(all_predictions)[class_mask]
                class_targets = np.array(all_targets)[class_mask]
                class_confidences = np.array(all_confidences)[class_mask]
                
                class_acc = accuracy_score(class_targets, class_predictions) * 100
                avg_conf = np.mean(class_confidences) * 100
                sample_count = np.sum(class_mask)
                
                class_accuracies.append(class_acc)
                print(f"{class_id:2d}   {class_acc:6.1f}%   {sample_count:4d}    {avg_conf:6.1f}%")
            else:
                class_accuracies.append(0)
                print(f"{class_id:2d}   {0:6.1f}%   {0:4d}    {0:6.1f}%")
        
        # 统计分析
        avg_class_acc = np.mean([acc for acc in class_accuracies if acc > 0])
        min_class_acc = min([acc for acc in class_accuracies if acc > 0])
        max_class_acc = max(class_accuracies)
        
        print(f"\n📈 统计分析:")
        print(f"平均类别准确率: {avg_class_acc:.1f}%")
        print(f"最低类别准确率: {min_class_acc:.1f}%")
        print(f"最高类别准确率: {max_class_acc:.1f}%")
        print(f"准确率标准差: {np.std(class_accuracies):.1f}%")
        
        return overall_acc, avg_class_acc, class_accuracies
    
    def save_model(self, filepath, metadata=None):
        """保存模型"""
        save_dir = Path(filepath).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_type': self.model_type,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'class_accuracies': self.class_accuracies,
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
    print("🎯 2D CNN轨迹分段预测模型训练")
    print("="*80)
    
    # 数据路径
    data_dir = "/mysda/shared_dir/2025.7.3/2025-07-03-16-28-57.ply"
    
    # 训练参数
    model_types = ['Simple2DCNN', 'Enhanced2DCNN', 'ResNet2D']
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"训练 {model_type} 模型")
        print(f"{'='*60}")
        
        # 创建训练器
        trainer = CNN2DTrajectoryTrainer(
            model_type=model_type,
            num_classes=20,
            learning_rate=0.001
        )
        
        # 创建数据集
        scan_contexts, labels, class_weights = trainer.create_2d_dataset(data_dir)
        
        if scan_contexts is None:
            print("❌ 数据集创建失败")
            continue
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = trainer.create_data_loaders(
            scan_contexts, labels, class_weights, batch_size=32
        )
        
        # 训练模型
        best_avg_acc = trainer.train_model(train_loader, val_loader, class_weights, epochs=50)
        
        # 测试模型
        overall_acc, avg_class_acc, class_accuracies = trainer.test_detailed_analysis(test_loader)
        
        # 保存模型
        model_path = f"models/saved/{model_type.lower()}_trajectory_avg{avg_class_acc:.1f}.pth"
        metadata = {
            'best_avg_class_acc': best_avg_acc,
            'test_overall_acc': overall_acc,
            'test_avg_class_acc': avg_class_acc,
            'class_accuracies': class_accuracies,
            'data_type': '2d_cnn_trajectory'
        }
        trainer.save_model(model_path, metadata)
        
        print(f"\n🎉 {model_type} 训练完成!")
        print(f"最佳平均类别准确率: {best_avg_acc:.1f}%")
        print(f"测试总体准确率: {overall_acc:.1f}%")
        print(f"测试平均类别准确率: {avg_class_acc:.1f}%")

if __name__ == '__main__':
    main()
