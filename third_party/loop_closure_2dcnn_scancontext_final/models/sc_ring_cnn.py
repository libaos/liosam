#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from .circular_conv import CircularPadConv2d, CircularResidualBlock

class SCRingCNN(nn.Module):
    """
    ScanContext环形域卷积神经网络
    
    该网络专门设计用于处理ScanContext特征图，解决S轴（极角轴）的周期性问题。
    网络输出一个固定长度的全局描述子，用于场景识别和回环检测。
    """
    
    def __init__(self, in_channels=1, num_rings=20, num_sectors=60, 
                 descriptor_dim=256, use_residual=True):
        """
        初始化SCRingCNN网络
        
        参数:
            in_channels (int): 输入通道数
            num_rings (int): ScanContext特征图的环数（R轴大小）
            num_sectors (int): ScanContext特征图的扇区数（S轴大小）
            descriptor_dim (int): 输出描述子的维度
            use_residual (bool): 是否使用残差连接
        """
        super(SCRingCNN, self).__init__()
        
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
                CircularResidualBlock(128, 256, downsample=downsample),
                nn.AdaptiveAvgPool2d((2, 2))  # 自适应池化到固定大小
            )
        else:
            # 使用普通卷积层构建网络
            self.layer1 = nn.Sequential(
                CircularPadConv2d(in_channels, 32, kernel_size=3),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            self.layer2 = nn.Sequential(
                CircularPadConv2d(32, 64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            self.layer3 = nn.Sequential(
                CircularPadConv2d(64, 128, kernel_size=3),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            self.layer4 = nn.Sequential(
                CircularPadConv2d(128, 256, kernel_size=3),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((2, 2))  # 自适应池化到固定大小
            )
        
        # 使用固定大小的特征图
        self.feature_size = 256 * 2 * 2  # 256通道，2x2的特征图
        
        # 全连接层，将特征图转换为固定长度的描述子
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, descriptor_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _make_residual_layer(self, in_channels, out_channels):
        """
        创建残差层
        
        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            
        返回:
            layer (nn.Sequential): 残差层
        """
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        layers = []
        # 添加第一个残差块，可能需要下采样
        layers.append(CircularResidualBlock(in_channels, out_channels, downsample=downsample))
        # 添加第二个残差块
        layers.append(CircularResidualBlock(out_channels, out_channels))
        # 添加最大池化层进行下采样
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        初始化网络权重
        """
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
            x (torch.Tensor): 输入张量，形状为(batch_size, in_channels, num_rings, num_sectors)
            
        返回:
            descriptor (torch.Tensor): 输出描述子，形状为(batch_size, descriptor_dim)
        """
        # 特征提取
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 展平特征图
        x = x.view(x.size(0), -1)
        
        # 全连接层生成描述子
        descriptor = self.fc(x)
        
        # L2归一化描述子
        descriptor = F.normalize(descriptor, p=2, dim=1)
        
        return descriptor
    
class SCRingCNNLite(nn.Module):
    """
    ScanContext环形域卷积神经网络的轻量级版本
    
    该网络是SCRingCNN的简化版本，使用更少的参数，适用于资源受限的设备。
    """
    
    def __init__(self, in_channels=1, num_rings=20, num_sectors=60, descriptor_dim=128):
        """
        初始化SCRingCNNLite网络
        
        参数:
            in_channels (int): 输入通道数
            num_rings (int): ScanContext特征图的环数（R轴大小）
            num_sectors (int): ScanContext特征图的扇区数（S轴大小）
            descriptor_dim (int): 输出描述子的维度
        """
        super(SCRingCNNLite, self).__init__()
        
        self.in_channels = in_channels
        self.num_rings = num_rings
        self.num_sectors = num_sectors
        self.descriptor_dim = descriptor_dim
        
        # 特征提取层
        self.features = nn.Sequential(
            # 第一层卷积
            CircularPadConv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二层卷积
            CircularPadConv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三层卷积
            CircularPadConv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2))  # 自适应池化到固定大小
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(64 * 2 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, descriptor_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        初始化网络权重
        """
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
            x (torch.Tensor): 输入张量，形状为(batch_size, in_channels, num_rings, num_sectors)
            
        返回:
            descriptor (torch.Tensor): 输出描述子，形状为(batch_size, descriptor_dim)
        """
        # 特征提取
        x = self.features(x)
        
        # 展平特征图
        x = x.view(x.size(0), -1)
        
        # 全连接层生成描述子
        descriptor = self.fc(x)
        
        # L2归一化描述子
        descriptor = F.normalize(descriptor, p=2, dim=1)
        
        return descriptor 