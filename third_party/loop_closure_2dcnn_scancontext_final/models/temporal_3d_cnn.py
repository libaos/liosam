#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class Temporal3DCNN(nn.Module):
    """
    时序3D CNN核心模型
    
    将N x H x W的时序张量视为1 x N x H x W的3D体数据进行处理。
    通过3D卷积核显式地捕捉时空动态，这是我们的核心创新模型。
    """
    
    def __init__(self, sequence_length=5, num_rings=20, num_sectors=60, 
                 num_classes=20, dropout_rate=0.5):
        """
        初始化时序3D CNN模型
        
        参数:
            sequence_length (int): 时序序列长度N（时间维度）
            num_rings (int): ScanContext特征图的环数（H维度）
            num_sectors (int): ScanContext特征图的扇区数（W维度）
            num_classes (int): 输出类别数（路径ID数量）
            dropout_rate (float): Dropout比例
        """
        super(Temporal3DCNN, self).__init__()
        
        self.sequence_length = sequence_length
        self.num_rings = num_rings
        self.num_sectors = num_sectors
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # 3D特征提取层
        # 输入: (batch_size, 1, N, H, W) -> 3D卷积处理时空信息
        self.features = nn.Sequential(
            # 第一层3D卷积块
            # 3D卷积核同时覆盖时间和空间维度
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # 只在空间维度下采样
            
            # 第二层3D卷积块
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            # 第三层3D卷积块
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # 时间维度也开始下采样
            
            # 第四层3D卷积块
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((2, 2, 2))  # 自适应池化到固定大小
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128 * 2 * 2 * 2, 512),
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
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
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
        # 添加通道维度: (batch_size, N, H, W) -> (batch_size, 1, N, H, W)
        x = x.unsqueeze(1)
        
        # 3D特征提取
        features = self.features(x)  # (batch_size, 128, 2, 2, 2)
        
        # 展平
        features = features.view(features.size(0), -1)  # (batch_size, 128*2*2*2)
        
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
        x = x.unsqueeze(1)  # 添加通道维度
        feature_maps = {}
        
        # 逐层提取特征
        x = self.features[0](x)  # Conv3d
        x = self.features[1](x)  # BatchNorm3d
        x = self.features[2](x)  # ReLU
        feature_maps['conv3d_1'] = x
        x = self.features[3](x)  # MaxPool3d
        
        x = self.features[4](x)  # Conv3d
        x = self.features[5](x)  # BatchNorm3d
        x = self.features[6](x)  # ReLU
        feature_maps['conv3d_2'] = x
        x = self.features[7](x)  # MaxPool3d
        
        x = self.features[8](x)  # Conv3d
        x = self.features[9](x)  # BatchNorm3d
        x = self.features[10](x)  # ReLU
        feature_maps['conv3d_3'] = x
        x = self.features[11](x)  # MaxPool3d
        
        x = self.features[12](x)  # Conv3d
        x = self.features[13](x)  # BatchNorm3d
        x = self.features[14](x)  # ReLU
        feature_maps['conv3d_4'] = x
        x = self.features[15](x)  # AdaptiveAvgPool3d
        
        return feature_maps


class Temporal3DCNNLite(nn.Module):
    """
    时序3D CNN轻量级模型
    
    更轻量的版本，适用于资源受限的环境，但仍保持3D卷积的时空建模能力。
    """
    
    def __init__(self, sequence_length=5, num_rings=20, num_sectors=60, 
                 num_classes=20, dropout_rate=0.3):
        """
        初始化轻量级时序3D CNN模型
        
        参数:
            sequence_length (int): 时序序列长度N
            num_rings (int): ScanContext特征图的环数
            num_sectors (int): ScanContext特征图的扇区数
            num_classes (int): 输出类别数
            dropout_rate (float): Dropout比例
        """
        super(Temporal3DCNNLite, self).__init__()
        
        self.sequence_length = sequence_length
        self.num_rings = num_rings
        self.num_sectors = num_sectors
        self.num_classes = num_classes
        
        # 轻量级3D特征提取层（更少的通道数和层数）
        self.features = nn.Sequential(
            # 第一层3D卷积块
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            # 第二层3D卷积块
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # 第三层3D卷积块
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((2, 2, 2))
        )
        
        # 简化的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(32 * 2 * 2 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        x = x.unsqueeze(1)  # 添加通道维度
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output


class Temporal3DCNNDeep(nn.Module):
    """
    深层时序3D CNN模型
    
    更深的网络结构，用于捕捉更复杂的时空模式。
    """
    
    def __init__(self, sequence_length=5, num_rings=20, num_sectors=60, 
                 num_classes=20, dropout_rate=0.5):
        """初始化深层时序3D CNN模型"""
        super(Temporal3DCNNDeep, self).__init__()
        
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # 更深的3D特征提取层
        self.features = nn.Sequential(
            # 第一组3D卷积块
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            # 第二组3D卷积块
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            # 第三组3D卷积块
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # 第四组3D卷积块
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((2, 2, 2))
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128 * 2 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        x = x.unsqueeze(1)  # 添加通道维度
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output
