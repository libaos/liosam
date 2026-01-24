#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
from pathlib import Path
import pickle
import random
from .scan_context import ScanContext
import re
from collections import defaultdict

class TemporalScanContextDataset(Dataset):
    """
    时序ScanContext特征图数据集
    
    该类用于加载和处理时序ScanContext特征图数据，支持构建N x H x W的时序输入张量。
    用于训练和测试2D CNN和3D CNN模型。
    """
    
    def __init__(self, data_dir, split='train', sequence_length=5, 
                 transform=None, use_augmentation=False,
                 cache_dir=None, num_classes=20, overlap_ratio=0.5):
        """
        初始化时序ScanContext数据集

        参数:
            data_dir (str): 数据目录路径
            split (str): 数据集划分，可选'train', 'val', 'test'
            sequence_length (int): 时序序列长度N
            transform (callable): 数据变换函数
            use_augmentation (bool): 是否使用数据增强
            cache_dir (str): 缓存目录路径，用于存储预处理的ScanContext特征图
            num_classes (int): 路径类别数量（默认20段）
            overlap_ratio (float): 序列重叠比例，用于数据增强
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.transform = transform
        self.use_augmentation = use_augmentation
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.num_classes = num_classes
        self.overlap_ratio = overlap_ratio
        
        # 初始化ScanContext生成器
        self.sc_generator = ScanContext(
            num_sectors=60, 
            num_rings=20,
            min_range=0.1,
            max_range=80.0
        )
        
        # 加载或生成数据
        self._load_data()
        
    def _load_data(self):
        """加载或生成时序数据"""
        # 检查缓存
        cache_file = None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"temporal_data_{self.split}_seq{self.sequence_length}.pkl"
            
        if cache_file and cache_file.exists():
            print(f"从缓存加载数据: {cache_file}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.sequences = data['sequences']
                self.labels = data['labels']
                self.file_paths = data['file_paths']
        else:
            print("生成时序数据...")
            self._generate_temporal_data()
            
            # 保存到缓存
            if cache_file:
                print(f"保存数据到缓存: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'sequences': self.sequences,
                        'labels': self.labels,
                        'file_paths': self.file_paths
                    }, f)
    
    def _generate_temporal_data(self):
        """生成时序数据"""
        # 获取所有PLY文件
        ply_files = sorted(glob.glob(str(self.data_dir / "raw" / "ply_files" / "*.ply")))
        
        if not ply_files:
            raise ValueError(f"在 {self.data_dir / 'raw' / 'ply_files'} 中未找到PLY文件")
        
        print(f"找到 {len(ply_files)} 个PLY文件")
        
        # 生成ScanContext特征图
        scan_contexts = []
        valid_files = []
        
        for i, ply_file in enumerate(ply_files):
            try:
                # 加载点云并生成ScanContext
                point_cloud = self.sc_generator.load_point_cloud(ply_file)
                if point_cloud is None or len(point_cloud) == 0:
                    print(f"跳过空点云文件: {ply_file}")
                    continue
                    
                sc = self.sc_generator.make_scan_context(point_cloud)
                scan_contexts.append(sc)
                valid_files.append(ply_file)
                
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(ply_files)} 个文件")
                    
            except Exception as e:
                print(f"处理文件 {ply_file} 时出错: {e}")
                continue
        
        print(f"成功生成 {len(scan_contexts)} 个ScanContext特征图")
        
        # 按顺序分成20段作为标签
        total_frames = len(scan_contexts)
        frames_per_segment = total_frames // self.num_classes
        
        # 生成时序序列
        self.sequences = []
        self.labels = []
        self.file_paths = []
        
        # 计算步长（考虑重叠）
        step_size = max(1, int(self.sequence_length * (1 - self.overlap_ratio)))
        
        for segment_id in range(self.num_classes):
            start_idx = segment_id * frames_per_segment
            end_idx = min((segment_id + 1) * frames_per_segment, total_frames)
            
            # 在每个段内生成时序序列
            for i in range(start_idx, end_idx - self.sequence_length + 1, step_size):
                sequence = []
                sequence_files = []
                
                for j in range(self.sequence_length):
                    sequence.append(scan_contexts[i + j])
                    sequence_files.append(valid_files[i + j])
                
                # 将序列堆叠成 (N, H, W) 张量
                sequence_tensor = np.stack(sequence, axis=0)  # (N, H, W)
                
                self.sequences.append(sequence_tensor)
                self.labels.append(segment_id)
                self.file_paths.append(sequence_files)
        
        print(f"生成 {len(self.sequences)} 个时序序列，每个序列长度为 {self.sequence_length}")
        
        # 数据集划分
        self._split_data()
    
    def _split_data(self):
        """划分训练集、验证集和测试集"""
        total_samples = len(self.sequences)
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        # 按比例划分：70% 训练，15% 验证，15% 测试
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)
        
        if self.split == 'train':
            selected_indices = indices[:train_size]
        elif self.split == 'val':
            selected_indices = indices[train_size:train_size + val_size]
        elif self.split == 'test':
            selected_indices = indices[train_size + val_size:]
        else:
            raise ValueError(f"不支持的数据集划分: {self.split}")
        
        # 筛选数据
        self.sequences = [self.sequences[i] for i in selected_indices]
        self.labels = [self.labels[i] for i in selected_indices]
        self.file_paths = [self.file_paths[i] for i in selected_indices]
        
        print(f"{self.split} 集包含 {len(self.sequences)} 个样本")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        sequence = self.sequences[idx]  # (N, H, W)
        label = self.labels[idx]
        
        # 转换为torch张量
        sequence = torch.FloatTensor(sequence)
        label = torch.LongTensor([label])
        
        # 应用变换
        if self.transform:
            sequence = self.transform(sequence)
        
        # 数据增强
        if self.use_augmentation and self.split == 'train':
            sequence = self._apply_augmentation(sequence)
        
        return sequence, label.squeeze()
    
    def _apply_augmentation(self, sequence):
        """应用数据增强"""
        # 随机旋转（沿扇区维度循环移位）
        if random.random() < 0.5:
            shift = random.randint(0, sequence.shape[-1] - 1)
            sequence = torch.roll(sequence, shift, dims=-1)
        
        # 添加噪声
        if random.random() < 0.3:
            noise = torch.randn_like(sequence) * 0.01
            sequence = sequence + noise
        
        return sequence
    
    def get_class_distribution(self):
        """获取类别分布"""
        from collections import Counter
        return Counter(self.labels)
    
    def get_sequence_info(self, idx):
        """获取序列信息"""
        return {
            'sequence_shape': self.sequences[idx].shape,
            'label': self.labels[idx],
            'file_paths': self.file_paths[idx]
        }


def create_temporal_dataloaders(data_dir, sequence_length=5, batch_size=32, 
                               num_workers=4, cache_dir=None, num_classes=20):
    """
    创建时序数据加载器
    
    参数:
        data_dir (str): 数据目录路径
        sequence_length (int): 时序序列长度
        batch_size (int): 批次大小
        num_workers (int): 数据加载工作进程数
        cache_dir (str): 缓存目录
        num_classes (int): 类别数量
    
    返回:
        train_loader, val_loader, test_loader: 数据加载器
    """
    # 创建数据集
    train_dataset = TemporalScanContextDataset(
        data_dir=data_dir,
        split='train',
        sequence_length=sequence_length,
        use_augmentation=True,
        cache_dir=cache_dir,
        num_classes=num_classes
    )
    
    val_dataset = TemporalScanContextDataset(
        data_dir=data_dir,
        split='val',
        sequence_length=sequence_length,
        use_augmentation=False,
        cache_dir=cache_dir,
        num_classes=num_classes
    )
    
    test_dataset = TemporalScanContextDataset(
        data_dir=data_dir,
        split='test',
        sequence_length=sequence_length,
        use_augmentation=False,
        cache_dir=cache_dir,
        num_classes=num_classes
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
