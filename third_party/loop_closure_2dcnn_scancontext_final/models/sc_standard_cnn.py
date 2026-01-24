#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardResidualBlock(nn.Module):
    """标准残差块（不使用环形卷积）"""
    
    def __init__(self, in_channels, out_channels, downsample=None):
        super(StandardResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
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

class SCStandardCNN(nn.Module):
    """
    ScanContext标准卷积神经网络（对比版本）
    
    与SCRingCNN相同的架构，但使用标准卷积替代环形卷积
    用于对比环形卷积的效果
    """
    
    def __init__(self, in_channels=1, num_rings=20, num_sectors=60, 
                 descriptor_dim=256, use_residual=True):
        """
        初始化SCStandardCNN网络
        
        参数:
            in_channels (int): 输入通道数
            num_rings (int): ScanContext特征图的环数（R轴大小）
            num_sectors (int): ScanContext特征图的扇区数（S轴大小）
            descriptor_dim (int): 输出描述子的维度
            use_residual (bool): 是否使用残差连接
        """
        super(SCStandardCNN, self).__init__()
        
        self.in_channels = in_channels
        self.num_rings = num_rings
        self.num_sectors = num_sectors
        self.descriptor_dim = descriptor_dim
        self.use_residual = use_residual
        
        # 特征提取层
        if use_residual:
            # 使用残差块构建网络
            self.layer1 = self._make_residual_layer(in_channels, 32)
            self.layer2 = self._make_residual_layer(32, 64)
            self.layer3 = self._make_residual_layer(64, 128)
            
            # 最后一层使用自适应池化确保输出尺寸固定
            # 为128到256通道的转换创建下采样层
            downsample = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(256)
            )
            
            self.layer4 = nn.Sequential(
                StandardResidualBlock(128, 256, downsample=downsample),
                nn.AdaptiveAvgPool2d((2, 2))  # 自适应池化到固定大小
            )
        else:
            # 使用普通卷积层构建网络
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # 标准卷积
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 标准卷积
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            self.layer3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 标准卷积
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            self.layer4 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 标准卷积
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((2, 2))  # 自适应池化到固定大小
            )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, descriptor_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_residual_layer(self, in_channels, out_channels):
        """创建残差层"""
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        layers = []
        # 添加第一个残差块，可能需要下采样
        layers.append(StandardResidualBlock(in_channels, out_channels, downsample=downsample))
        # 添加第二个残差块
        layers.append(StandardResidualBlock(out_channels, out_channels))
        # 添加最大池化层进行下采样
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入ScanContext特征图，形状为 (batch_size, in_channels, num_rings, num_sectors)
        
        返回:
            torch.Tensor: 输出描述子，形状为 (batch_size, descriptor_dim)
        """
        # 特征提取
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        
        # L2归一化
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def get_feature_maps(self, x):
        """
        获取中间特征图，用于可视化和分析
        
        参数:
            x (torch.Tensor): 输入张量
        
        返回:
            dict: 包含各层特征图的字典
        """
        features = {}
        
        x = self.layer1(x)
        features['layer1'] = x
        
        x = self.layer2(x)
        features['layer2'] = x
        
        x = self.layer3(x)
        features['layer3'] = x
        
        x = self.layer4(x)
        features['layer4'] = x
        
        return features

class SCStandardCNNLite(nn.Module):
    """
    ScanContext标准卷积神经网络的轻量级版本（对比版本）
    
    与SCRingCNNLite相同的架构，但使用标准卷积替代环形卷积
    """
    
    def __init__(self, in_channels=1, num_rings=20, num_sectors=60, descriptor_dim=128):
        """
        初始化SCStandardCNNLite网络
        
        参数:
            in_channels (int): 输入通道数
            num_rings (int): ScanContext特征图的环数
            num_sectors (int): ScanContext特征图的扇区数
            descriptor_dim (int): 输出描述子的维度
        """
        super(SCStandardCNNLite, self).__init__()
        
        self.in_channels = in_channels
        self.num_rings = num_rings
        self.num_sectors = num_sectors
        self.descriptor_dim = descriptor_dim
        
        # 特征提取层
        self.features = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # 标准卷积
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二层卷积
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 标准卷积
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三层卷积
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 标准卷积
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2))  # 自适应池化到固定大小
        )
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2 * 2, descriptor_dim),
            nn.ReLU(inplace=True)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入ScanContext特征图
        
        返回:
            torch.Tensor: 输出描述子
        """
        # 特征提取
        x = self.features(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 分类器
        x = self.classifier(x)
        
        # L2归一化
        x = F.normalize(x, p=2, dim=1)
        
        return x
