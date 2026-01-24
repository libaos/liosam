#!/usr/bin/env python3
"""
SC Standard Enhanced CNN - 增强版标准ScanContext CNN
结合多种注意力机制的回环检测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==================== 注意力机制模块 ====================

class ChannelAttention(nn.Module):
    """通道注意力模块 (CBAM中的通道注意力)"""
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
    """空间注意力模块 (CBAM中的空间注意力)"""
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
    """CBAM注意力模块"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class ECAAttention(nn.Module):
    """ECA注意力模块 - 高效通道注意力"""
    def __init__(self, kernel_size=3):
        super(ECAAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.gap(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).permute(0, 2, 1)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        y = self.sigmoid(y)  # [B, 1, C]
        y = y.permute(0, 2, 1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)

class SEAttention(nn.Module):
    """SE注意力模块 - 使用卷积实现避免维度问题"""
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

class SimAM(nn.Module):
    """SimAM注意力模块 - 无参数注意力"""
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

# ==================== 增强版SC Standard CNN ====================

class SCStandardEnhancedCNN(nn.Module):
    """
    增强版SC Standard CNN
    结合多种注意力机制的回环检测模型
    """
    
    def __init__(self, input_channels=1, descriptor_dim=256, attention_types=['cbam'], 
                 reduction=16, dropout_rate=0.3):
        """
        初始化增强版SC Standard CNN
        
        参数:
            input_channels (int): 输入通道数
            descriptor_dim (int): 描述符维度
            attention_types (list): 注意力机制类型列表
            reduction (int): 通道缩减比例
            dropout_rate (float): Dropout比例
        """
        super(SCStandardEnhancedCNN, self).__init__()
        
        self.input_channels = input_channels
        self.descriptor_dim = descriptor_dim
        self.attention_types = attention_types
        
        # 基础卷积层
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # 注意力机制模块
        self.attention_modules = nn.ModuleList()
        for att_type in attention_types:
            try:
                if att_type == 'cbam':
                    self.attention_modules.append(CBAM(512, reduction))
                elif att_type == 'eca':
                    self.attention_modules.append(ECAAttention())
                elif att_type == 'se':
                    self.attention_modules.append(SEAttention(512, reduction))
                elif att_type == 'simam':
                    self.attention_modules.append(SimAM())
                else:
                    print(f"警告: 未知的注意力类型 {att_type}")
            except Exception as e:
                print(f"错误: 创建{att_type}注意力模块失败: {e}")
                # 添加一个恒等映射作为备用
                self.attention_modules.append(nn.Identity())
        
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
        
        # 应用注意力机制
        for attention_module in self.attention_modules:
            x = attention_module(x)
        
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
        
        attention_str = '+'.join(self.attention_types) if self.attention_types else 'None'
        
        return {
            'model_name': 'SCStandardEnhancedCNN',
            'input_channels': self.input_channels,
            'descriptor_dim': self.descriptor_dim,
            'attention_mechanisms': attention_str,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'enhancement': f'Enhanced with {attention_str} attention'
        }

# ==================== 模型工厂函数 ====================

def create_enhanced_model(model_type='cbam', **kwargs):
    """
    创建增强版模型
    
    参数:
        model_type (str): 模型类型
        **kwargs: 其他参数
    
    返回:
        nn.Module: 增强版模型
    """
    attention_configs = {
        'cbam': ['cbam'],
        'eca': ['eca'],
        'se': ['se'],
        'simam': ['simam'],
        'cbam_eca': ['cbam', 'eca'],
        'cbam_se': ['cbam', 'se'],
        'all': ['cbam', 'eca', 'se', 'simam'],
        'dual': ['cbam', 'eca'],
        'triple': ['cbam', 'eca', 'se']
    }
    
    if model_type not in attention_configs:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    attention_types = attention_configs[model_type]
    
    return SCStandardEnhancedCNN(
        attention_types=attention_types,
        **kwargs
    )

# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 测试不同的注意力机制组合
    test_configs = [
        ('cbam', ['cbam']),
        ('eca', ['eca']),
        ('se', ['se']),
        ('simam', ['simam']),
        ('dual', ['cbam', 'eca']),
        ('triple', ['cbam', 'eca', 'se'])
    ]
    
    print("🧪 测试增强版SC Standard CNN模型...")
    
    # 创建测试输入
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 20, 60)  # ScanContext尺寸
    
    for config_name, attention_types in test_configs:
        try:
            model = SCStandardEnhancedCNN(attention_types=attention_types)
            output = model(input_tensor)
            
            model_info = model.get_model_info()
            
            print(f"\n✅ {config_name.upper()} 配置:")
            print(f"   注意力机制: {model_info['attention_mechanisms']}")
            print(f"   参数数量: {model_info['total_parameters']:,}")
            print(f"   模型大小: {model_info['model_size_mb']:.2f} MB")
            print(f"   输入形状: {input_tensor.shape}")
            print(f"   输出形状: {output.shape}")
            
        except Exception as e:
            print(f"❌ {config_name.upper()} 配置失败: {e}")
    
    print(f"\n🎯 所有测试完成！")
