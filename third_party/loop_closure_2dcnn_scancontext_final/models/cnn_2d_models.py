#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2D CNN模型用于轨迹分段预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple2DCNN(nn.Module):
    """简单的2D CNN模型"""
    
    def __init__(self, input_shape=(20, 60), num_classes=20):
        super(Simple2DCNN, self).__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # 2D卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 20x60 -> 10x30
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 10x30 -> 5x15
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 5x15 -> 2x7
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 2 * 7, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 1, 20, 60)
        
        # 第一个卷积块
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # 第二个卷积块
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # 第三个卷积块
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class Enhanced2DCNN(nn.Module):
    """增强的2D CNN模型"""
    
    def __init__(self, input_shape=(20, 60), num_classes=20):
        super(Enhanced2DCNN, self).__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 20x60 -> 10x30
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 10x30 -> 5x15
        
        # 第三个卷积块
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 5x15 -> 2x7
        
        # 第四个卷积块
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))  # 2x7 -> 1x1
        
        # 全连接层
        self.fc1 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 1, 20, 60)
        
        # 第一个卷积块
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        
        # 第二个卷积块
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        
        # 第三个卷积块
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        
        # 第四个卷积块
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        
        return x

class ResNet2D(nn.Module):
    """ResNet风格的2D CNN"""
    
    def __init__(self, input_shape=(20, 60), num_classes=20):
        super(ResNet2D, self).__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # 初始卷积
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # ResNet块
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # 第一个块可能需要下采样
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # 其余块
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: (batch_size, 1, 20, 60)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入输出维度不同，需要调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out

def get_model_info(model):
    """获取模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
    }

if __name__ == '__main__':
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = {
        'Simple2DCNN': Simple2DCNN(),
        'Enhanced2DCNN': Enhanced2DCNN(),
        'ResNet2D': ResNet2D()
    }
    
    # 测试输入
    test_input = torch.randn(4, 1, 20, 60).to(device)
    
    print("2D CNN模型对比:")
    print("="*60)
    
    for name, model in models.items():
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            output = model(test_input)
        
        info = get_model_info(model)
        
        print(f"{name}:")
        print(f"  输出形状: {output.shape}")
        print(f"  参数数量: {info['total_params']:,}")
        print(f"  模型大小: {info['model_size_mb']:.2f} MB")
        print(f"  可训练参数: {info['trainable_params']:,}")
        print()
