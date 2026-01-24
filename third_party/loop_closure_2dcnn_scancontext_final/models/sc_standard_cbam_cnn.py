#!/usr/bin/env python3
"""
SC Standard CBAM CNN - CBAM注意力增强的ScanContext CNN
专门用于回环检测的CBAM注意力模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """CBAM中的通道注意力模块"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """CBAM中的空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """CBAM注意力模块 - 结合通道和空间注意力"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class SCStandardCBAMCNN(nn.Module):
    """
    SC Standard CBAM CNN
    基于CBAM注意力机制的ScanContext回环检测模型
    """
    
    def __init__(self, input_channels=1, descriptor_dim=256, reduction=16, dropout_rate=0.3):
        """
        初始化SC Standard CBAM CNN
        
        参数:
            input_channels (int): 输入通道数
            descriptor_dim (int): 描述符维度
            reduction (int): 通道缩减比例
            dropout_rate (float): Dropout比例
        """
        super(SCStandardCBAMCNN, self).__init__()
        
        self.input_channels = input_channels
        self.descriptor_dim = descriptor_dim
        self.reduction = reduction
        
        # 基础卷积层
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # CBAM注意力模块
        self.cbam = CBAM(512, reduction)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, descriptor_dim)
        
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
        # 第一层卷积
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        # 第二层卷积
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # 第三层卷积
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # 第四层卷积
        x = F.relu(self.bn4(self.conv4(x)))
        
        # CBAM注意力
        x = self.cbam(x)
        
        # 全局平均池化
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # L2归一化
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
        
        return {
            'model_name': 'SCStandardCBAMCNN',
            'attention_type': 'CBAM',
            'input_channels': self.input_channels,
            'descriptor_dim': self.descriptor_dim,
            'reduction': self.reduction,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'enhancement': 'CBAM Attention (Channel + Spatial)'
        }

# 测试代码
if __name__ == "__main__":
    print("🧪 测试SC Standard CBAM CNN模型...")
    
    # 创建模型
    model = SCStandardCBAMCNN()
    
    # 获取模型信息
    model_info = model.get_model_info()
    print(f"\n📊 模型信息:")
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # 测试前向传播
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 20, 60)  # ScanContext尺寸
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"\n✅ 前向传播测试:")
    print(f"   输入形状: {input_tensor.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   输出范数: {torch.norm(output, dim=1).mean():.4f}")
    
    print(f"\n🎯 SC Standard CBAM CNN测试完成！")
