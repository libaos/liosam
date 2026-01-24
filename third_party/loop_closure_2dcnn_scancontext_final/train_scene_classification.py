#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于真实场景变化的分类模型训练
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.cnn_2d_models import Simple2DCNN, Enhanced2DCNN
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class SceneClassificationTrainer:
    """基于真实场景变化的分类训练器"""
    
    def __init__(self, model_type='simple2dcnn', learning_rate=0.001):
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🎯 真实场景分类训练器")
        print(f"设备: {self.device}")
        print(f"模型类型: {model_type}")
        print(f"目标: 基于真实场景内容进行分类")
        
    def load_scene_analysis_results(self, results_file='scene_analysis_results.pkl'):
        """加载场景分析结果"""
        if not Path(results_file).exists():
            print(f"❌ 场景分析结果文件不存在: {results_file}")
            print("请先运行 scene_change_detector.py")
            return None
        
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        
        print(f"✅ 加载场景分析结果:")
        print(f"  特征维度: {results['features'].shape}")
        print(f"  聚类标签: {len(np.unique(results['cluster_labels']))} 个类别")
        print(f"  场景变化点: {len(results['change_points'])} 个")
        
        return results
    
    def create_realistic_dataset(self, results):
        """创建基于真实场景变化的数据集"""
        features = results['features']
        cluster_labels = results['cluster_labels']
        
        # 提取ScanContext特征（前1200维）
        scan_contexts = features[:, :1200].reshape(-1, 20, 60)
        
        # 使用聚类标签作为真实标签
        labels = np.array(cluster_labels)
        
        print(f"📊 真实场景数据集:")
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            percentage = count / len(labels) * 100
            print(f"  场景类别 {label}: {count:4d} 样本 ({percentage:5.1f}%)")
        
        num_classes = len(unique_labels)
        
        return scan_contexts, labels, num_classes
    
    def create_balanced_dataset(self, scan_contexts, labels, samples_per_class=500):
        """创建平衡的数据集"""
        print(f"🔄 创建平衡数据集 (每类{samples_per_class}样本)...")
        
        balanced_contexts = []
        balanced_labels = []
        
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            
            if len(label_indices) >= samples_per_class:
                # 随机采样
                selected_indices = np.random.choice(label_indices, samples_per_class, replace=False)
            else:
                # 重复采样
                selected_indices = np.random.choice(label_indices, samples_per_class, replace=True)
            
            for idx in selected_indices:
                balanced_contexts.append(scan_contexts[idx])
                balanced_labels.append(label)
            
            print(f"  类别 {label}: {len(label_indices)} -> {samples_per_class} 样本")
        
        return np.array(balanced_contexts), np.array(balanced_labels)
    
    def train_scene_classifier(self, scan_contexts, labels, num_classes):
        """训练场景分类器"""
        print(f"\n🎯 开始训练真实场景分类器...")
        
        # 创建模型
        if self.model_type == 'simple2dcnn':
            model = Simple2DCNN(num_classes=num_classes)
        elif self.model_type == 'enhanced2dcnn':
            model = Enhanced2DCNN(num_classes=num_classes)
        else:
            raise ValueError(f"未知模型类型: {self.model_type}")
        
        model = model.to(self.device)
        
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
        
        # 转换为PyTorch张量
        train_tensor = torch.FloatTensor(train_contexts).unsqueeze(1).to(self.device)
        val_tensor = torch.FloatTensor(val_contexts).unsqueeze(1).to(self.device)
        test_tensor = torch.FloatTensor(test_contexts).unsqueeze(1).to(self.device)
        
        train_labels_tensor = torch.LongTensor(train_labels).to(self.device)
        val_labels_tensor = torch.LongTensor(val_labels).to(self.device)
        test_labels_tensor = torch.LongTensor(test_labels).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(train_tensor, train_labels_tensor)
        val_dataset = TensorDataset(val_tensor, val_labels_tensor)
        test_dataset = TensorDataset(test_tensor, test_labels_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 训练设置
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        
        # 训练循环
        best_val_acc = 0
        best_model_state = None
        patience = 10
        patience_counter = 0
        
        train_losses = []
        val_accuracies = []
        
        epochs = 50
        
        for epoch in range(epochs):
            # 训练
            model.train()
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            
            # 验证
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            val_acc = 100. * val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_accuracies.append(val_acc)
            
            print(f'Epoch {epoch+1:2d}/{epochs:2d} | '
                  f'训练损失: {avg_train_loss:.4f} | '
                  f'验证准确率: {val_acc:.1f}%')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print(f'🎉 新的最佳验证准确率: {best_val_acc:.1f}%')
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                print(f'⏹️  早停触发')
                break
            
            scheduler.step()
        
        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 测试
        model.eval()
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                test_predictions.extend(predicted.cpu().numpy())
                test_targets.extend(target.cpu().numpy())
        
        test_acc = accuracy_score(test_targets, test_predictions) * 100
        
        print(f"\n✅ 训练完成!")
        print(f"最佳验证准确率: {best_val_acc:.1f}%")
        print(f"测试准确率: {test_acc:.1f}%")
        
        # 详细分析
        print(f"\n📊 详细分类报告:")
        print(classification_report(test_targets, test_predictions))
        
        # 混淆矩阵
        cm = confusion_matrix(test_targets, test_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Scene Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('scene_classification_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存模型
        model_path = f"models/saved/scene_classifier_{self.model_type}_acc{test_acc:.1f}.pth"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': self.model_type,
            'num_classes': num_classes,
            'test_accuracy': test_acc,
            'best_val_accuracy': best_val_acc,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        }, model_path)
        
        print(f"✅ 模型已保存到: {model_path}")
        
        return model, test_acc

def main():
    """主函数"""
    print("="*60)
    print("🎯 基于真实场景变化的分类模型训练")
    print("="*60)
    
    # 创建训练器
    trainer = SceneClassificationTrainer(model_type='simple2dcnn')
    
    # 加载场景分析结果
    results = trainer.load_scene_analysis_results()
    if results is None:
        return
    
    # 创建真实场景数据集
    scan_contexts, labels, num_classes = trainer.create_realistic_dataset(results)
    
    # 创建平衡数据集
    balanced_contexts, balanced_labels = trainer.create_balanced_dataset(
        scan_contexts, labels, samples_per_class=500
    )
    
    # 训练分类器
    model, test_acc = trainer.train_scene_classifier(
        balanced_contexts, balanced_labels, num_classes
    )
    
    print(f"\n🎉 真实场景分类训练完成!")
    print(f"测试准确率: {test_acc:.1f}%")
    print(f"这是基于真实场景内容的分类结果，比之前的虚假99.6%更有意义！")

if __name__ == '__main__':
    main()
