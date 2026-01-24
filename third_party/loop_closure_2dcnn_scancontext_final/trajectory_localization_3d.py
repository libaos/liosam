#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于3D CNN的轨迹定位系统
使用3D体素化点云进行位置识别
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.cnn_3d_models import Simple3DCNN, Enhanced3DCNN, ResNet3D, PointCloudVoxelizer
from utils.ply_reader import PLYReader
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle
import time

class TrajectoryLocalization3D:
    """基于3D CNN的轨迹定位系统"""
    
    def __init__(self, num_locations=20, model_type='simple3dcnn', voxel_size=(32, 32, 32)):
        self.num_locations = num_locations
        self.model_type = model_type
        self.voxel_size = voxel_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化体素化器
        self.voxelizer = PointCloudVoxelizer(
            voxel_size=voxel_size,
            point_cloud_range=[[-25, 25], [-25, 25], [-5, 5]]  # 50x50x10米范围
        )
        
        print(f"🎯 基于3D CNN的轨迹定位系统")
        print(f"设备: {self.device}")
        print(f"目标位置数: {num_locations}")
        print(f"模型类型: {model_type}")
        print(f"体素尺寸: {voxel_size}")
        print(f"目标: 基于3D体素化特征进行精确轨迹定位")
        
        # 初始化模型
        if model_type == 'simple3dcnn':
            self.model = Simple3DCNN(num_classes=num_locations, input_size=voxel_size)
        elif model_type == 'enhanced3dcnn':
            self.model = Enhanced3DCNN(num_classes=num_locations, input_size=voxel_size)
        elif model_type == 'resnet3d':
            self.model = ResNet3D(num_classes=num_locations, input_size=voxel_size)
        else:
            raise ValueError(f"未知模型类型: {model_type}")
        
        self.model = self.model.to(self.device)
        
        # 位置信息存储
        self.location_database = {}
        self.location_features = []
        self.location_labels = []
        
        # 优化参数
        self.confidence_threshold = 0.7
        self.temporal_smoothing = True
        self.location_history = []
        self.confidence_history = []
        
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def create_3d_location_database(self, data_dir, save_path='location_database_3d.pkl'):
        """创建基于3D体素的位置数据库"""
        print(f"📍 创建3D体素位置数据库...")
        
        # 获取所有ply文件
        ply_files = sorted(glob.glob(f"{data_dir}/*.ply"))
        print(f"找到 {len(ply_files)} 个ply文件")
        
        if len(ply_files) == 0:
            print("❌ 未找到ply文件")
            return False
        
        # 计算每个位置段的文件范围
        files_per_location = len(ply_files) // self.num_locations
        print(f"每个位置段包含约 {files_per_location} 个文件")
        
        location_data = {}
        all_features = []
        all_labels = []
        
        for location_id in range(self.num_locations):
            print(f"  处理位置 {location_id+1}/{self.num_locations}")
            
            # 确定这个位置的文件范围
            start_idx = location_id * files_per_location
            if location_id == self.num_locations - 1:
                end_idx = len(ply_files)
            else:
                end_idx = (location_id + 1) * files_per_location
            
            location_files = ply_files[start_idx:end_idx]
            location_features = []
            
            # 处理这个位置的所有文件
            for ply_file in location_files:
                try:
                    points = PLYReader.read_ply_file(ply_file)
                    if points is not None and len(points) > 100:
                        points = points[:, :3]  # 只使用x,y,z坐标
                        
                        # 体素化
                        voxel_grid = self.voxelizer.voxelize(points)
                        
                        if voxel_grid is not None:
                            location_features.append(voxel_grid)
                            all_features.append(voxel_grid)
                            all_labels.append(location_id)
                            
                except Exception as e:
                    print(f"    处理失败 {ply_file}: {e}")
                    continue
            
            if len(location_features) > 0:
                # 计算这个位置的代表性特征（平均值）
                representative_voxel = np.mean(location_features, axis=0)
                location_data[location_id] = {
                    'representative_voxel': representative_voxel,
                    'sample_count': len(location_features),
                    'file_range': (start_idx, end_idx),
                    'all_features': location_features
                }
                print(f"    位置 {location_id}: {len(location_features)} 个有效样本")
            else:
                print(f"    ⚠️  位置 {location_id}: 无有效样本")
        
        self.location_database = location_data
        self.location_features = np.array(all_features)
        self.location_labels = np.array(all_labels)
        
        # 保存位置数据库
        with open(save_path, 'wb') as f:
            pickle.dump({
                'location_database': self.location_database,
                'location_features': self.location_features,
                'location_labels': self.location_labels,
                'num_locations': self.num_locations,
                'voxel_size': self.voxel_size
            }, f)
        
        print(f"✅ 3D位置数据库已保存到: {save_path}")
        print(f"总样本数: {len(all_features)}")
        print(f"体素特征形状: {self.location_features.shape}")
        
        return True
    
    def load_3d_location_database(self, load_path='location_database_3d.pkl'):
        """加载3D位置数据库"""
        if not Path(load_path).exists():
            print(f"❌ 3D位置数据库文件不存在: {load_path}")
            return False
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.location_database = data['location_database']
        self.location_features = data['location_features']
        self.location_labels = data['location_labels']
        self.num_locations = data['num_locations']
        
        print(f"✅ 已加载3D位置数据库")
        print(f"位置数量: {self.num_locations}")
        print(f"总样本数: {len(self.location_features)}")
        print(f"体素特征形状: {self.location_features.shape}")
        
        return True
    
    def train_3d_localization_model(self, epochs=50, batch_size=16):
        """训练3D定位模型"""
        print(f"\n🎯 开始训练3D轨迹定位模型...")
        
        if len(self.location_features) == 0:
            print("❌ 没有训练数据，请先创建位置数据库")
            return False
        
        # 数据划分
        print("🔍 检查数据分布...")
        unique_labels, counts = np.unique(self.location_labels, return_counts=True)
        min_samples = np.min(counts)
        
        if min_samples < 3:
            print(f"⚠️  检测到样本不足的类别 (最少{min_samples}个样本)")
            print("使用随机划分而不是分层划分")
            
            X_train, X_temp, y_train, y_temp = train_test_split(
                self.location_features, self.location_labels, 
                test_size=0.4, random_state=42
            )
            
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )
        else:
            print("✅ 所有类别样本充足，使用分层划分")
            X_train, X_temp, y_train, y_temp = train_test_split(
                self.location_features, self.location_labels, 
                test_size=0.4, random_state=42, stratify=self.location_labels
            )
            
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
        
        print(f"训练集: {len(X_train)} 样本")
        print(f"验证集: {len(X_val)} 样本")
        print(f"测试集: {len(X_test)} 样本")
        
        # 转换为PyTorch张量 (添加通道维度)
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(self.device)
        
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 训练设置
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        
        # 训练循环
        best_val_acc = 0
        best_model_state = None
        patience = 10
        patience_counter = 0
        
        train_losses = []
        val_accuracies = []
        
        print(f"\n开始训练 (批次大小: {batch_size})...")
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            
            # 验证
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    output = self.model(data)
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
                best_model_state = self.model.state_dict().copy()
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
            self.model.load_state_dict(best_model_state)
        
        # 测试
        self.model.eval()
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                test_predictions.extend(predicted.cpu().numpy())
                test_targets.extend(target.cpu().numpy())
        
        test_acc = accuracy_score(test_targets, test_predictions) * 100
        
        print(f"\n✅ 3D CNN训练完成!")
        print(f"最佳验证准确率: {best_val_acc:.1f}%")
        print(f"测试准确率: {test_acc:.1f}%")
        
        # 详细分析
        print(f"\n📊 详细3D定位性能分析:")
        
        # 计算每个位置的准确率
        location_accuracies = {}
        for location_id in range(self.num_locations):
            location_mask = np.array(test_targets) == location_id
            if np.sum(location_mask) > 0:
                location_predictions = np.array(test_predictions)[location_mask]
                location_targets = np.array(test_targets)[location_mask]
                location_acc = accuracy_score(location_targets, location_predictions) * 100
                location_accuracies[location_id] = location_acc
                sample_count = np.sum(location_mask)
                print(f"  位置 {location_id:2d}: {location_acc:6.1f}% ({sample_count:2d} 样本)")
        
        avg_location_acc = np.mean(list(location_accuracies.values()))
        print(f"\n平均位置准确率: {avg_location_acc:.1f}%")
        
        # 保存模型
        model_path = f"models/saved/trajectory_localizer_3d_{self.model_type}_acc{test_acc:.1f}.pth"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'num_locations': self.num_locations,
            'voxel_size': self.voxel_size,
            'test_accuracy': test_acc,
            'best_val_accuracy': best_val_acc,
            'location_accuracies': location_accuracies,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        }, model_path)
        
        print(f"✅ 3D定位模型已保存到: {model_path}")
        
        return test_acc

def main():
    """主函数"""
    print("="*60)
    print("🎯 基于3D CNN的轨迹定位系统")
    print("="*60)
    
    # 数据路径
    data_dir = "/mysda/shared_dir/2025.7.3/2025-07-03-16-28-57.ply"
    
    # 创建3D定位系统
    localizer = TrajectoryLocalization3D(
        num_locations=20, 
        model_type='simple3dcnn',
        voxel_size=(32, 32, 32)
    )
    
    # 1. 创建3D位置数据库
    print("\n步骤1: 创建3D体素位置数据库")
    success = localizer.create_3d_location_database(data_dir)
    
    if not success:
        print("❌ 3D位置数据库创建失败")
        return
    
    # 2. 训练3D定位模型
    print("\n步骤2: 训练3D定位模型")
    test_acc = localizer.train_3d_localization_model(epochs=30, batch_size=8)
    
    print(f"\n🎉 3D轨迹定位系统训练完成!")
    print(f"定位准确率: {test_acc:.1f}%")
    print(f"系统可以识别轨迹中的 {localizer.num_locations} 个不同位置")
    print(f"✨ 3D CNN特性:")
    print(f"  - 3D体素化点云表示")
    print(f"  - 空间几何特征学习")
    print(f"  - 端到端3D特征提取")
    print(f"下次机器人来到相同区域时，可以基于3D空间特征进行定位！")

if __name__ == '__main__':
    main()
