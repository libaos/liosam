#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CircularPadConv2d(nn.Module):
    """
    带有环形填充的2D卷积层
    
    该模块在S轴（极角轴）上使用环形填充，解决ScanContext特征图的周期性边界问题。
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=True):
        """
        初始化环形填充卷积层
        
        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            kernel_size (int or tuple): 卷积核大小
            stride (int or tuple): 步长
            padding (int or tuple): 填充大小，如果为None，则自动计算为(kernel_size - 1) // 2
            dilation (int or tuple): 扩张率
            groups (int): 分组卷积的组数
            bias (bool): 是否使用偏置
        """
        super(CircularPadConv2d, self).__init__()
        
        # 处理kernel_size为单个数字的情况
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
            
        # 处理stride为单个数字的情况
        if isinstance(stride, int):
            stride = (stride, stride)
            
        # 处理dilation为单个数字的情况
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
            
        # 如果padding为None，则自动计算
        if padding is None:
            # 计算需要的填充，使输出大小与输入大小相同
            padding = ((kernel_size[0] - 1) * dilation[0] // 2,
                      (kernel_size[1] - 1) * dilation[1] // 2)
        elif isinstance(padding, int):
            padding = (padding, padding)
            
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # 创建标准的2D卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=0,  # 我们将手动处理填充
                             dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为(batch_size, in_channels, height, width)
            
        返回:
            output (torch.Tensor): 输出张量，形状为(batch_size, out_channels, height', width')
        """
        # 对S轴（宽度方向，索引为3）进行环形填充
        # 对R轴（高度方向，索引为2）进行常规填充
        
        # 首先，对R轴进行常规填充（上下填充）
        x_pad_r = F.pad(x, (0, 0, self.padding[0], self.padding[0]), mode='constant', value=0)
        
        # 然后，对S轴进行环形填充（左右填充）
        # 从右侧取padding[1]列，添加到左侧
        left_pad = x_pad_r[:, :, :, -self.padding[1]:]
        # 从左侧取padding[1]列，添加到右侧
        right_pad = x_pad_r[:, :, :, :self.padding[1]]
        
        # 拼接左填充、原始数据和右填充
        x_pad = torch.cat([left_pad, x_pad_r, right_pad], dim=3)
        
        # 应用卷积
        output = self.conv(x_pad)
        
        return output

class CircularResidualBlock(nn.Module):
    """
    带有环形填充的残差块
    
    该模块使用环形填充卷积层，并包含残差连接，用于构建深层网络。
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        """
        初始化环形残差块
        
        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            kernel_size (int): 卷积核大小
            stride (int): 步长
            downsample (nn.Module): 下采样层，用于残差连接
        """
        super(CircularResidualBlock, self).__init__()
        
        # 第一个环形填充卷积层
        self.conv1 = CircularPadConv2d(in_channels, out_channels, kernel_size, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 第二个环形填充卷积层
        self.conv2 = CircularPadConv2d(out_channels, out_channels, kernel_size)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 下采样层，用于匹配残差连接的形状
        self.downsample = downsample
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量
            
        返回:
            output (torch.Tensor): 输出张量
        """
        identity = x
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class CircularConvDataAugmentation:
    """
    ScanContext特征图的数据增强

    该类实现了ScanContext特征图的数据增强方法，包括循环移位、噪声添加、强度缩放等。
    """

    @staticmethod
    def random_shift(scan_context, max_shift=None):
        """
        随机循环移位ScanContext特征图

        参数:
            scan_context (numpy.ndarray): 形状为(num_rings, num_sectors)的ScanContext特征图
            max_shift (int): 最大移位量，如果为None，则为num_sectors

        返回:
            shifted_sc (numpy.ndarray): 移位后的ScanContext特征图
            shift (int): 实际的移位量
        """
        num_sectors = scan_context.shape[1]

        if max_shift is None:
            max_shift = num_sectors

        # 生成随机移位量
        shift = np.random.randint(0, max_shift)

        # 循环移位
        shifted_sc = np.roll(scan_context, shift, axis=1)

        return shifted_sc, shift

    @staticmethod
    def add_gaussian_noise(scan_context, noise_std=0.01):
        """
        添加高斯噪声

        参数:
            scan_context (numpy.ndarray): ScanContext特征图
            noise_std (float): 噪声标准差

        返回:
            noisy_sc (numpy.ndarray): 添加噪声后的特征图
        """
        noise = np.random.normal(0, noise_std, scan_context.shape)
        noisy_sc = scan_context + noise
        # 确保值在合理范围内
        noisy_sc = np.clip(noisy_sc, 0, np.max(scan_context) * 1.2)
        return noisy_sc

    @staticmethod
    def intensity_scaling(scan_context, scale_range=(0.8, 1.2)):
        """
        强度缩放

        参数:
            scan_context (numpy.ndarray): ScanContext特征图
            scale_range (tuple): 缩放范围

        返回:
            scaled_sc (numpy.ndarray): 缩放后的特征图
        """
        scale = np.random.uniform(scale_range[0], scale_range[1])
        scaled_sc = scan_context * scale
        return scaled_sc

    @staticmethod
    def random_dropout(scan_context, dropout_prob=0.1):
        """
        随机dropout，将部分像素设为0

        参数:
            scan_context (numpy.ndarray): ScanContext特征图
            dropout_prob (float): dropout概率

        返回:
            dropout_sc (numpy.ndarray): dropout后的特征图
        """
        mask = np.random.random(scan_context.shape) > dropout_prob
        dropout_sc = scan_context * mask
        return dropout_sc

    @staticmethod
    def radial_noise(scan_context, noise_std=0.005):
        """
        径向噪声，模拟距离测量误差

        参数:
            scan_context (numpy.ndarray): ScanContext特征图
            noise_std (float): 噪声标准差

        返回:
            noisy_sc (numpy.ndarray): 添加径向噪声后的特征图
        """
        num_rings, num_sectors = scan_context.shape

        # 创建径向权重，距离越远噪声越大
        radial_weights = np.linspace(1.0, 2.0, num_rings).reshape(-1, 1)
        radial_weights = np.repeat(radial_weights, num_sectors, axis=1)

        # 生成径向噪声
        noise = np.random.normal(0, noise_std, scan_context.shape) * radial_weights
        noisy_sc = scan_context + noise
        noisy_sc = np.clip(noisy_sc, 0, np.max(scan_context) * 1.2)

        return noisy_sc

    @staticmethod
    def sector_masking(scan_context, mask_prob=0.1, mask_width=3):
        """
        扇区遮挡，模拟部分视野被遮挡的情况

        参数:
            scan_context (numpy.ndarray): ScanContext特征图
            mask_prob (float): 遮挡概率
            mask_width (int): 遮挡宽度

        返回:
            masked_sc (numpy.ndarray): 遮挡后的特征图
        """
        num_rings, num_sectors = scan_context.shape
        masked_sc = scan_context.copy()

        if np.random.random() < mask_prob:
            # 随机选择遮挡的起始扇区
            start_sector = np.random.randint(0, num_sectors)

            # 遮挡连续的几个扇区
            for i in range(mask_width):
                sector_idx = (start_sector + i) % num_sectors
                masked_sc[:, sector_idx] = 0

        return masked_sc

    @staticmethod
    def comprehensive_augmentation(scan_context, config=None):
        """
        综合数据增强，随机应用多种增强方法

        参数:
            scan_context (numpy.ndarray): ScanContext特征图
            config (dict): 增强配置参数

        返回:
            augmented_sc (numpy.ndarray): 增强后的特征图
        """
        if config is None:
            config = {
                'rotation_range': 360,
                'noise_std': 0.01,
                'dropout_prob': 0.1,
                'intensity_scale': [0.8, 1.2],
                'radial_noise_std': 0.005,
                'sector_mask_prob': 0.1,
                'sector_mask_width': 3
            }

        augmented_sc = scan_context.copy()

        # 1. 随机旋转（循环移位）
        if config.get('rotation_range', 0) > 0:
            max_shift = int(scan_context.shape[1] * config['rotation_range'] / 360)
            augmented_sc, _ = CircularConvDataAugmentation.random_shift(
                augmented_sc, max_shift=max_shift
            )

        # 2. 添加高斯噪声
        if config.get('noise_std', 0) > 0:
            augmented_sc = CircularConvDataAugmentation.add_gaussian_noise(
                augmented_sc, noise_std=config['noise_std']
            )

        # 3. 强度缩放
        if config.get('intensity_scale'):
            augmented_sc = CircularConvDataAugmentation.intensity_scaling(
                augmented_sc, scale_range=config['intensity_scale']
            )

        # 4. 随机dropout
        if config.get('dropout_prob', 0) > 0:
            augmented_sc = CircularConvDataAugmentation.random_dropout(
                augmented_sc, dropout_prob=config['dropout_prob']
            )

        # 5. 径向噪声
        if config.get('radial_noise_std', 0) > 0:
            augmented_sc = CircularConvDataAugmentation.radial_noise(
                augmented_sc, noise_std=config['radial_noise_std']
            )

        # 6. 扇区遮挡
        if config.get('sector_mask_prob', 0) > 0:
            augmented_sc = CircularConvDataAugmentation.sector_masking(
                augmented_sc,
                mask_prob=config['sector_mask_prob'],
                mask_width=config.get('sector_mask_width', 3)
            )

        return augmented_sc

    @staticmethod
    def batch_random_shift(batch_sc):
        """
        对批次的ScanContext特征图进行随机循环移位

        参数:
            batch_sc (numpy.ndarray): 形状为(batch_size, num_rings, num_sectors)的ScanContext特征图批次

        返回:
            shifted_batch (numpy.ndarray): 移位后的ScanContext特征图批次
            shifts (numpy.ndarray): 每个样本的实际移位量
        """
        batch_size = batch_sc.shape[0]
        num_sectors = batch_sc.shape[2]

        # 为每个样本生成随机移位量
        shifts = np.random.randint(0, num_sectors, size=batch_size)

        # 初始化移位后的批次
        shifted_batch = np.zeros_like(batch_sc)

        # 对每个样本进行循环移位
        for i in range(batch_size):
            shifted_batch[i] = np.roll(batch_sc[i], shifts[i], axis=1)

        return shifted_batch, shifts