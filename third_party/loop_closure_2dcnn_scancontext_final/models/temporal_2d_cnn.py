#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class Temporal2DCNN(nn.Module):
    """
    时序2D CNN基线模型
    
    将N x H x W的时序张量视为N通道的2D图像进行处理。
    这是一个基线模型，用于与3D CNN进行对比。
    """
    
    def __init__(self, sequence_length=5, num_rings=20, num_sectors=60, 
                 num_classes=20, dropout_rate=0.5):
        """
        初始化时序2D CNN模型
        
        参数:
            sequence_length (int): 时序序列长度N（作为输入通道数）
            num_rings (int): ScanContext特征图的环数（H维度）
            num_sectors (int): ScanContext特征图的扇区数（W维度）
            num_classes (int): 输出类别数（路径ID数量）
            dropout_rate (float): Dropout比例
        """
        super(Temporal2DCNN, self).__init__()
        
        self.sequence_length = sequence_length
        self.num_rings = num_rings
        self.num_sectors = num_sectors
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # 特征提取层
        # 输入: (batch_size, N, H, W) -> 视为 (batch_size, N, H, W) 的2D图像
        self.features = nn.Sequential(
            # 第一层卷积块
            nn.Conv2d(sequence_length, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32, H/2, W/2)
            
            # 第二层卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (64, H/4, W/4)
            
            # 第三层卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (128, H/8, W/8)
            
            # 第四层卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # (256, 4, 4)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
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
            x (torch.Tensor): 输入时序张量，形状为 (batch_size, N, H, W)
        
        返回:
            torch.Tensor: 输出logits，形状为 (batch_size, num_classes)
        """
        # 特征提取
        features = self.features(x)  # (batch_size, 256, 4, 4)
        
        # 展平
        features = features.view(features.size(0), -1)  # (batch_size, 256*4*4)
        
        # 分类
        output = self.classifier(features)  # (batch_size, num_classes)
        
        return output
    
    def get_feature_maps(self, x):
        """
        获取中间特征图，用于可视化和分析
        
        参数:
            x (torch.Tensor): 输入张量
        
        返回:
            dict: 包含各层特征图的字典
        """
        feature_maps = {}
        
        # 逐层提取特征
        x = self.features[0](x)  # Conv2d
        x = self.features[1](x)  # BatchNorm2d
        x = self.features[2](x)  # ReLU
        feature_maps['conv1'] = x
        x = self.features[3](x)  # MaxPool2d
        
        x = self.features[4](x)  # Conv2d
        x = self.features[5](x)  # BatchNorm2d
        x = self.features[6](x)  # ReLU
        feature_maps['conv2'] = x
        x = self.features[7](x)  # MaxPool2d
        
        x = self.features[8](x)  # Conv2d
        x = self.features[9](x)  # BatchNorm2d
        x = self.features[10](x)  # ReLU
        feature_maps['conv3'] = x
        x = self.features[11](x)  # MaxPool2d
        
        x = self.features[12](x)  # Conv2d
        x = self.features[13](x)  # BatchNorm2d
        x = self.features[14](x)  # ReLU
        feature_maps['conv4'] = x
        x = self.features[15](x)  # AdaptiveAvgPool2d
        
        return feature_maps


class Temporal2DCNNLite(nn.Module):
    """
    时序2D CNN轻量级模型
    
    更轻量的版本，适用于资源受限的环境。
    """
    
    def __init__(self, sequence_length=5, num_rings=20, num_sectors=60, 
                 num_classes=20, dropout_rate=0.3):
        """
        初始化轻量级时序2D CNN模型
        
        参数:
            sequence_length (int): 时序序列长度N
            num_rings (int): ScanContext特征图的环数
            num_sectors (int): ScanContext特征图的扇区数
            num_classes (int): 输出类别数
            dropout_rate (float): Dropout比例
        """
        super(Temporal2DCNNLite, self).__init__()
        
        self.sequence_length = sequence_length
        self.num_rings = num_rings
        self.num_sectors = num_sectors
        self.num_classes = num_classes
        
        # 特征提取层（更少的通道数）
        self.features = nn.Sequential(
            # 第一层卷积块
            nn.Conv2d(sequence_length, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二层卷积块
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三层卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # 分类器（更简单）
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
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
        """前向传播"""
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output


class Temporal2DCNNResNet(nn.Module):
    """
    基于ResNet的时序2D CNN模型
    
    使用残差连接提升模型性能。
    """
    
    def __init__(self, sequence_length=5, num_rings=20, num_sectors=60, 
                 num_classes=20, dropout_rate=0.5):
        """初始化ResNet版本的时序2D CNN"""
        super(Temporal2DCNNResNet, self).__init__()
        
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(sequence_length, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差块
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """创建残差层"""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class BasicBlock(nn.Module):
    """基本残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
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
