#!/usr/bin/env python3
"""
SC Standard SE CNN - SE注意力增强的ScanContext CNN
专门用于回环检测的SE注意力模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEAttention(nn.Module):
    """SE注意力模块 - 挤压激励注意力"""
    def __init__(self, in_channels, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)

class SCStandardSECNN(nn.Module):
    """
    SC Standard SE CNN
    基于SE注意力机制的ScanContext回环检测模型
    """
    
    def __init__(self, input_channels=1, descriptor_dim=256, reduction=16, dropout_rate=0.3):
        """
        初始化SC Standard SE CNN
        
        参数:
            input_channels (int): 输入通道数
            descriptor_dim (int): 描述符维度
            reduction (int): 通道缩减比例
            dropout_rate (float): Dropout比例
        """
        super(SCStandardSECNN, self).__init__()
        
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
        
        # SE注意力模块
        self.se = SEAttention(512, reduction)
        
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
        
        # SE注意力
        x = self.se(x)
        
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
            'model_name': 'SCStandardSECNN',
            'attention_type': 'SE',
            'input_channels': self.input_channels,
            'descriptor_dim': self.descriptor_dim,
            'reduction': self.reduction,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'enhancement': 'SE Attention (Squeeze-and-Excitation)'
        }

# 测试代码
if __name__ == "__main__":
    print("🧪 测试SC Standard SE CNN模型...")
    
    # 创建模型
    model = SCStandardSECNN()
    
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
    
    print(f"\n🎯 SC Standard SE CNN测试完成！")
