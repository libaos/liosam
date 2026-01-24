#!/usr/bin/env python3
"""
SC Standard SimAM CNN - SimAM注意力增强的ScanContext CNN
专门用于回环检测的SimAM注意力模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimAM(nn.Module):
    """SimAM注意力模块 - 无参数注意力机制"""
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda
    
    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        return x * self.activation(y)

class SCStandardSimAMCNN(nn.Module):
    """
    SC Standard SimAM CNN
    基于SimAM注意力机制的ScanContext回环检测模型
    """
    
    def __init__(self, input_channels=1, descriptor_dim=256, e_lambda=1e-4, dropout_rate=0.3):
        """
        初始化SC Standard SimAM CNN
        
        参数:
            input_channels (int): 输入通道数
            descriptor_dim (int): 描述符维度
            e_lambda (float): SimAM参数
            dropout_rate (float): Dropout比例
        """
        super(SCStandardSimAMCNN, self).__init__()
        
        self.input_channels = input_channels
        self.descriptor_dim = descriptor_dim
        self.e_lambda = e_lambda
        
        # 基础卷积层
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # SimAM注意力模块
        self.simam = SimAM(e_lambda)
        
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
        
        # SimAM注意力
        x = self.simam(x)
        
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
            'model_name': 'SCStandardSimAMCNN',
            'attention_type': 'SimAM',
            'input_channels': self.input_channels,
            'descriptor_dim': self.descriptor_dim,
            'e_lambda': self.e_lambda,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'enhancement': 'SimAM Attention (Simple, Parameter-Free Attention)'
        }

# 测试代码
if __name__ == "__main__":
    print("🧪 测试SC Standard SimAM CNN模型...")
    
    # 创建模型
    model = SCStandardSimAMCNN()
    
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
    
    print(f"\n🎯 SC Standard SimAM CNN测试完成！")
