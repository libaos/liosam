#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
时序模型定义，用于挑战性实验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Temporal3DCNN(nn.Module):
    """3D CNN时序模型"""
    
    def __init__(self, input_shape, num_classes=20):
        super(Temporal3DCNN, self).__init__()
        
        # input_shape: (channels, seq_len, height, width)
        channels, seq_len, height, width = input_shape
        
        self.conv3d_layers = nn.Sequential(
            nn.Conv3d(channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x: (batch_size, seq_len, height, width)
        x = x.unsqueeze(1)  # 添加通道维度: (batch_size, 1, seq_len, height, width)
        
        x = self.conv3d_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class Temporal3DCNNDeep(nn.Module):
    """深层3D CNN时序模型"""
    
    def __init__(self, input_shape, num_classes=20):
        super(Temporal3DCNNDeep, self).__init__()
        
        channels, seq_len, height, width = input_shape
        
        self.conv3d_layers = nn.Sequential(
            nn.Conv3d(channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv3d_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Temporal2DCNN(nn.Module):
    """2D CNN时序模型"""

    def __init__(self, input_shape, num_classes=20):
        super(Temporal2DCNN, self).__init__()

        if len(input_shape) == 3:
            seq_len, height, width = input_shape
        else:
            height, width = input_shape
            seq_len = 1

        self.seq_len = seq_len

        self.conv_layers = nn.Sequential(
            nn.Conv2d(seq_len, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, height, width) 或 (batch_size, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # 添加通道维度: (batch_size, 1, height, width)
        # 如果是4D且通道数与seq_len匹配，直接使用
        # x应该是: (batch_size, seq_len, height, width)

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

class ResNet2DCNN(nn.Module):
    """基于ResNet的2D CNN模型"""

    def __init__(self, input_shape, num_classes=20):
        super(ResNet2DCNN, self).__init__()

        if len(input_shape) == 3:
            seq_len, height, width = input_shape
        else:
            height, width = input_shape
            seq_len = 1

        self.seq_len = seq_len

        # 输入投影
        self.input_proj = nn.Conv2d(seq_len, 64, kernel_size=3, padding=1)

        # ResNet块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
        ])

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # (batch_size, 1, height, width)
        # x应该是: (batch_size, seq_len, height, width)

        x = self.input_proj(x)

        for block in self.res_blocks:
            x = block(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

class SimpleCNN(nn.Module):
    """简单的2D CNN模型，用于消融研究"""
    
    def __init__(self, input_shape, num_classes=20):
        super(SimpleCNN, self).__init__()
        
        if len(input_shape) == 2:
            height, width = input_shape
        else:
            height, width = input_shape[-2:]
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x: (batch_size, height, width) 或其他形状
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # 添加通道维度
        elif len(x.shape) == 4 and x.shape[1] > 1:
            # 如果是多通道，取平均
            x = x.mean(dim=1, keepdim=True)
        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
