#!/usr/bin/env python3
"""
SCStandardCNN + Spatial Attention Enhancement
基于SCStandardCNN的空间注意力增强模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    
    def __init__(self, kernel_size=7):
        """
        初始化空间注意力模块
        
        参数:
            kernel_size (int): 卷积核大小，默认7
        """
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        
        # 空间注意力卷积层
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
            
        返回:
            torch.Tensor: 注意力增强后的特征图
        """
        # 计算通道维度的平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 拼接平均值和最大值
        attention_input = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        
        # 生成空间注意力权重
        attention_weights = self.sigmoid(self.conv(attention_input))  # [B, 1, H, W]
        
        # 应用注意力权重
        return x * attention_weights

class ChannelAttention(nn.Module):
    """通道注意力模块 (可选增强)"""
    
    def __init__(self, in_channels, reduction=16):
        """
        初始化通道注意力模块
        
        参数:
            in_channels (int): 输入通道数
            reduction (int): 降维比例，默认16
        """
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享的MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征图 [B, C, H, W]
            
        返回:
            torch.Tensor: 注意力增强后的特征图
        """
        # 平均池化和最大池化
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        # 生成通道注意力权重
        attention_weights = self.sigmoid(avg_out + max_out)
        
        # 应用注意力权重
        return x * attention_weights

class SpatialEnhancedBlock(nn.Module):
    """空间注意力增强的卷积块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 use_channel_attention=False):
        """
        初始化增强卷积块
        
        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            kernel_size (int): 卷积核大小
            stride (int): 步长
            padding (int): 填充
            use_channel_attention (bool): 是否使用通道注意力
        """
        super(SpatialEnhancedBlock, self).__init__()
        
        # 标准卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 注意力模块
        self.use_channel_attention = use_channel_attention
        if use_channel_attention:
            self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        """前向传播"""
        # 标准卷积操作
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        # 应用注意力机制
        if self.use_channel_attention:
            out = self.channel_attention(out)
        out = self.spatial_attention(out)
        
        return out

class SCStandardSpatialCNN(nn.Module):
    """
    SCStandardCNN + Spatial Attention Enhancement
    基于SCStandardCNN的空间注意力增强模型
    """
    
    def __init__(self, input_channels=1, descriptor_dim=256, use_channel_attention=False):
        """
        初始化模型
        
        参数:
            input_channels (int): 输入通道数，默认1 (Scan Context)
            descriptor_dim (int): 描述子维度，默认256
            use_channel_attention (bool): 是否使用通道注意力，默认False
        """
        super(SCStandardSpatialCNN, self).__init__()
        
        self.input_channels = input_channels
        self.descriptor_dim = descriptor_dim
        self.use_channel_attention = use_channel_attention
        
        # 特征提取层 (增强版)
        self.features = nn.Sequential(
            # 第一层: 1 -> 32
            SpatialEnhancedBlock(input_channels, 32, 3, 1, 1, use_channel_attention),
            nn.MaxPool2d(2, 2),  # 降采样
            
            # 第二层: 32 -> 64
            SpatialEnhancedBlock(32, 64, 3, 1, 1, use_channel_attention),
            nn.MaxPool2d(2, 2),  # 降采样
            
            # 第三层: 64 -> 128
            SpatialEnhancedBlock(64, 128, 3, 1, 1, use_channel_attention),
            nn.MaxPool2d(2, 2),  # 降采样
            
            # 第四层: 128 -> 256
            SpatialEnhancedBlock(128, 256, 3, 1, 1, use_channel_attention),
            nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, descriptor_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
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
            x (torch.Tensor): 输入Scan Context，形状为 [B, C, H, W]
            
        返回:
            torch.Tensor: 描述子向量，形状为 [B, descriptor_dim]
        """
        # 特征提取
        features = self.features(x)  # [B, 256, 1, 1]
        
        # 展平
        features = features.view(features.size(0), -1)  # [B, 256]
        
        # 生成描述子
        descriptor = self.classifier(features)  # [B, descriptor_dim]
        
        # L2归一化
        descriptor = F.normalize(descriptor, p=2, dim=1)
        
        return descriptor
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'SCStandardSpatialCNN',
            'input_channels': self.input_channels,
            'descriptor_dim': self.descriptor_dim,
            'use_channel_attention': self.use_channel_attention,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
            'enhancement': 'Spatial Attention + Optional Channel Attention'
        }

def create_sc_standard_spatial_cnn(input_channels=1, descriptor_dim=256, use_channel_attention=False):
    """
    创建SCStandardSpatialCNN模型
    
    参数:
        input_channels (int): 输入通道数
        descriptor_dim (int): 描述子维度
        use_channel_attention (bool): 是否使用通道注意力
        
    返回:
        SCStandardSpatialCNN: 模型实例
    """
    model = SCStandardSpatialCNN(
        input_channels=input_channels,
        descriptor_dim=descriptor_dim,
        use_channel_attention=use_channel_attention
    )
    
    return model

# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = create_sc_standard_spatial_cnn(
        input_channels=1,
        descriptor_dim=256,
        use_channel_attention=True  # 启用通道注意力
    )
    
    # 打印模型信息
    info = model.get_model_info()
    print("模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试前向传播
    batch_size = 4
    height, width = 20, 60  # Scan Context典型尺寸
    
    x = torch.randn(batch_size, 1, height, width)
    print(f"\n输入形状: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        print(f"输出形状: {output.shape}")
        print(f"输出范数: {torch.norm(output, dim=1)}")  # 应该接近1（L2归一化）
