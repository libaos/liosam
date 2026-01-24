#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D CNN模型用于轨迹定位
基于3D点云体素化表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Simple3DCNN(nn.Module):
    """简单的3D CNN模型"""
    
    def __init__(self, num_classes=20, input_size=(32, 32, 32)):
        super(Simple3DCNN, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # 3D卷积层
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool3d(2, 2)
        
        # 计算全连接层输入维度
        self._calculate_fc_input_size()
        
        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def _calculate_fc_input_size(self):
        """计算全连接层输入维度"""
        with torch.no_grad():
            x = torch.zeros(1, 1, *self.input_size)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            self.fc_input_size = x.view(1, -1).size(1)
    
    def forward(self, x):
        # 3D卷积 + 池化
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class Enhanced3DCNN(nn.Module):
    """增强的3D CNN模型"""
    
    def __init__(self, num_classes=20, input_size=(32, 32, 32)):
        super(Enhanced3DCNN, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # 第一个3D卷积块
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2)
        )
        
        # 第二个3D卷积块
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2)
        )
        
        # 第三个3D卷积块
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2)
        )
        
        # 计算全连接层输入维度
        self._calculate_fc_input_size()
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def _calculate_fc_input_size(self):
        """计算全连接层输入维度"""
        with torch.no_grad():
            x = torch.zeros(1, 1, *self.input_size)
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            self.fc_input_size = x.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.classifier(x)
        
        return x

class ResNet3D(nn.Module):
    """3D ResNet模型"""
    
    def __init__(self, num_classes=20, input_size=(32, 32, 32)):
        super(ResNet3D, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # 初始卷积
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet块
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 分类器
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """创建ResNet层"""
        layers = []
        
        # 第一个块可能需要下采样
        layers.append(ResNet3DBlock(in_channels, out_channels, stride))
        
        # 其余块
        for _ in range(1, blocks):
            layers.append(ResNet3DBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class ResNet3DBlock(nn.Module):
    """3D ResNet基本块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet3DBlock, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # 如果输入输出维度不同，需要调整
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class PointCloudVoxelizer:
    """点云体素化工具"""
    
    def __init__(self, voxel_size=(32, 32, 32), point_cloud_range=None):
        self.voxel_size = voxel_size
        if point_cloud_range is None:
            # 默认范围：x,y,z各50米，以原点为中心
            self.point_cloud_range = [[-25, 25], [-25, 25], [-5, 5]]
        else:
            self.point_cloud_range = point_cloud_range
    
    def voxelize(self, points):
        """将点云转换为体素网格"""
        if points is None or len(points) == 0:
            return np.zeros(self.voxel_size, dtype=np.float32)
        
        # 确保points是numpy数组
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        # 只使用x,y,z坐标
        if points.shape[1] > 3:
            points = points[:, :3]
        
        # 过滤超出范围的点
        valid_mask = np.ones(len(points), dtype=bool)
        for i, (min_val, max_val) in enumerate(self.point_cloud_range):
            valid_mask &= (points[:, i] >= min_val) & (points[:, i] <= max_val)
        
        points = points[valid_mask]
        
        if len(points) == 0:
            return np.zeros(self.voxel_size, dtype=np.float32)
        
        # 计算体素索引
        voxel_indices = np.zeros((len(points), 3), dtype=int)
        for i in range(3):
            min_val, max_val = self.point_cloud_range[i]
            voxel_indices[:, i] = np.floor(
                (points[:, i] - min_val) / (max_val - min_val) * self.voxel_size[i]
            ).astype(int)
            # 确保索引在有效范围内
            voxel_indices[:, i] = np.clip(voxel_indices[:, i], 0, self.voxel_size[i] - 1)
        
        # 创建体素网格
        voxel_grid = np.zeros(self.voxel_size, dtype=np.float32)
        
        # 填充体素（使用点密度）
        for idx in voxel_indices:
            voxel_grid[idx[0], idx[1], idx[2]] += 1.0
        
        # 归一化
        if voxel_grid.max() > 0:
            voxel_grid = voxel_grid / voxel_grid.max()
        
        return voxel_grid

def test_3d_models():
    """测试3D模型"""
    print("🧪 测试3D CNN模型...")
    
    # 创建测试数据
    batch_size = 2
    input_size = (32, 32, 32)
    num_classes = 20
    
    # 测试输入
    x = torch.randn(batch_size, 1, *input_size)
    
    # 测试Simple3DCNN
    print("\n1. 测试Simple3DCNN:")
    model1 = Simple3DCNN(num_classes=num_classes, input_size=input_size)
    output1 = model1(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {output1.shape}")
    print(f"   参数数量: {sum(p.numel() for p in model1.parameters()):,}")
    
    # 测试Enhanced3DCNN
    print("\n2. 测试Enhanced3DCNN:")
    model2 = Enhanced3DCNN(num_classes=num_classes, input_size=input_size)
    output2 = model2(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {output2.shape}")
    print(f"   参数数量: {sum(p.numel() for p in model2.parameters()):,}")
    
    # 测试ResNet3D
    print("\n3. 测试ResNet3D:")
    model3 = ResNet3D(num_classes=num_classes, input_size=input_size)
    output3 = model3(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {output3.shape}")
    print(f"   参数数量: {sum(p.numel() for p in model3.parameters()):,}")
    
    # 测试体素化器
    print("\n4. 测试点云体素化:")
    voxelizer = PointCloudVoxelizer(voxel_size=(32, 32, 32))
    
    # 创建测试点云
    test_points = np.random.uniform(-20, 20, (1000, 3))
    voxel_grid = voxelizer.voxelize(test_points)
    print(f"   点云形状: {test_points.shape}")
    print(f"   体素网格形状: {voxel_grid.shape}")
    print(f"   非零体素数量: {np.count_nonzero(voxel_grid)}")
    print(f"   体素值范围: [{voxel_grid.min():.3f}, {voxel_grid.max():.3f}]")
    
    print("\n✅ 所有3D模型测试通过!")

if __name__ == '__main__':
    test_3d_models()
