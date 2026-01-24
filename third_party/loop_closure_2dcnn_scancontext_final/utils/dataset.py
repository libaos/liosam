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
from models.circular_conv import CircularConvDataAugmentation
import re

class ScanContextDataset(Dataset):
    """
    ScanContext特征图数据集
    
    该类用于加载和处理ScanContext特征图数据，用于训练和测试环形卷积网络。
    """
    
    def __init__(self, data_dir, split='train', transform=None,
                 use_augmentation=False, max_shift=None,
                 cache_dir=None, precomputed=False, augmentation_config=None):
        """
        初始化ScanContext数据集

        参数:
            data_dir (str): 数据目录路径
            split (str): 数据集划分，可选'train', 'val', 'test'
            transform (callable): 数据变换函数
            use_augmentation (bool): 是否使用数据增强
            max_shift (int): 最大循环移位量，如果为None，则为num_sectors
            cache_dir (str): 缓存目录路径，用于存储预处理的ScanContext特征图
            precomputed (bool): 是否使用预先计算的ScanContext特征图
            augmentation_config (dict): 数据增强配置参数
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_augmentation = use_augmentation
        self.max_shift = max_shift
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.precomputed = precomputed
        self.augmentation_config = augmentation_config or {
            'rotation_range': 360,
            'noise_std': 0.01,
            'dropout_prob': 0.1,
            'intensity_scale': [0.8, 1.2],
            'radial_noise_std': 0.005,
            'sector_mask_prob': 0.1,
            'sector_mask_width': 3
        }
        
        # 获取点云文件列表
        if precomputed:
            # 使用预先计算的ScanContext特征图
            self.files = sorted(glob.glob(str(self.data_dir / f"{split}/*.npy")))
        else:
            # 使用原始点云数据
            self.files = []
            for ext in ['*.bin', '*.pcd', '*.txt', '*.xyz', '*.npy']:
                self.files.extend(sorted(glob.glob(str(self.data_dir / f"{split}/{ext}"))))
        
        # 检查文件数量
        if len(self.files) == 0:
            raise ValueError(f"在 {self.data_dir / split} 中未找到任何文件")
            
        print(f"找到 {len(self.files)} 个文件在 {self.data_dir / split} 中")
        
        # 加载或创建正样本对和负样本对
        self.positive_pairs, self.negative_pairs = self._create_sample_pairs()
        
        # 创建ScanContext生成器
        self.sc_generator = ScanContext()
        
        # 缓存已加载的ScanContext特征图
        self.sc_cache = {}
        
    def _create_sample_pairs(self):
        """
        创建正样本对和负样本对
        
        正样本对：来自同一位置（或相近位置）的两个ScanContext特征图
        负样本对：来自不同位置的两个ScanContext特征图
        
        返回:
            positive_pairs (list): 正样本对列表
            negative_pairs (list): 负样本对列表
        """
        # 检查是否存在预先生成的样本对信息
        # 尝试多个可能的路径
        possible_paths = [
            os.path.join(self.data_dir, 'pairs_info.pkl'),  # 数据目录下
            os.path.join(os.path.dirname(self.data_dir), 'pairs_info.pkl'),  # 父目录下
            '/root/w/RandLA-Net-pytorch/回环检测/data/fruit_trees_dataset_for_training/pairs_info.pkl'  # 绝对路径
        ]
        
        for pairs_info_path in possible_paths:
            print(f"尝试加载样本对信息: {pairs_info_path}")
            if os.path.exists(pairs_info_path):
                try:
                    with open(pairs_info_path, 'rb') as f:
                        pairs_info = pickle.load(f)
                        positive_pairs = pairs_info.get('positive_pairs', [])
                        negative_pairs = pairs_info.get('negative_pairs', [])
                        print(f"从 {pairs_info_path} 加载了 {len(positive_pairs)} 个正样本对和 {len(negative_pairs)} 个负样本对")
                        
                        # 检查文件路径是否存在
                        all_exist = True
                        for file1, file2 in positive_pairs + negative_pairs:
                            if not os.path.exists(file1):
                                print(f"文件不存在: {file1}")
                                all_exist = False
                            if not os.path.exists(file2):
                                print(f"文件不存在: {file2}")
                                all_exist = False
                        
                        if not all_exist:
                            print("部分文件不存在，尝试修复路径...")
                            # 尝试修复路径
                            fixed_positive_pairs = []
                            for file1, file2 in positive_pairs:
                                file1_name = os.path.basename(file1)
                                file2_name = os.path.basename(file2)
                                new_file1 = os.path.join(self.data_dir, file1_name)
                                new_file2 = os.path.join(self.data_dir, file2_name)
                                if os.path.exists(new_file1) and os.path.exists(new_file2):
                                    fixed_positive_pairs.append((new_file1, new_file2))
                            
                            fixed_negative_pairs = []
                            for file1, file2 in negative_pairs:
                                file1_name = os.path.basename(file1)
                                file2_name = os.path.basename(file2)
                                new_file1 = os.path.join(self.data_dir, file1_name)
                                new_file2 = os.path.join(self.data_dir, file2_name)
                                if os.path.exists(new_file1) and os.path.exists(new_file2):
                                    fixed_negative_pairs.append((new_file1, new_file2))
                            
                            print(f"修复后: {len(fixed_positive_pairs)} 个正样本对和 {len(fixed_negative_pairs)} 个负样本对")
                            return fixed_positive_pairs, fixed_negative_pairs
                        
                        return positive_pairs, negative_pairs
                except Exception as e:
                    print(f"加载样本对信息失败: {e}")
        
        print("未找到预先生成的样本对信息，创建新的样本对...")
        
        # 如果没有找到预先生成的样本对信息，则创建新的样本对
        # 这里需要根据实际数据集的组织方式来实现
        # 以下是一个简单的实现，假设文件名中包含位置信息
        
        # 获取所有位置ID
        location_ids = []
        for file_path in self.files:
            # 从文件名中提取位置ID
            # 假设文件名格式为：cloud_XXX_sc.npy
            file_name = os.path.basename(file_path)
            match = re.search(r'cloud_(\d+)_sc\.npy', file_name)
            if match:
                location_id = int(match.group(1))
                location_ids.append(location_id)
            else:
                # 如果文件名不符合预期格式，则使用文件索引作为位置ID
                location_ids.append(len(location_ids))
        
        # 按位置ID对文件进行分组
        location_to_files = {}
        for i, loc_id in enumerate(location_ids):
            if loc_id not in location_to_files:
                location_to_files[loc_id] = []
            location_to_files[loc_id].append(self.files[i])
        
        # 创建正样本对（ID差异小的文件对）
        positive_pairs = []
        for i, file1 in enumerate(self.files):
            id1 = location_ids[i]
            # 为每个样本生成指定数量的正样本对
            pairs_added = 0
            for j in range(i + 1, len(self.files)):
                file2 = self.files[j]
                id2 = location_ids[j]
                # 如果ID差异小于阈值，则视为正样本对
                if abs(id2 - id1) <= 10:  # 使用较大的阈值
                    positive_pairs.append((file1, file2))
                    pairs_added += 1
                    if pairs_added >= 3:  # 每个样本最多3个正样本对
                        break
        
        # 创建负样本对（ID差异大的文件对）
        negative_pairs = []
        for i, file1 in enumerate(self.files):
            id1 = location_ids[i]
            # 为每个样本生成指定数量的负样本对
            pairs_added = 0
            # 随机选择其他文件作为负样本
            candidates = []
            for j, file2 in enumerate(self.files):
                id2 = location_ids[j]
                # 如果ID差异大于阈值，则视为负样本候选
                if abs(id2 - id1) >= 30:  # 使用较大的阈值
                    candidates.append(file2)
            
            # 从候选中随机选择指定数量的负样本
            if candidates:
                selected = random.sample(candidates, min(5, len(candidates)))  # 每个样本最多5个负样本对
                for file2 in selected:
                    negative_pairs.append((file1, file2))
                    pairs_added += 1
        
        print(f"创建了 {len(positive_pairs)} 个正样本对和 {len(negative_pairs)} 个负样本对")
        
        return positive_pairs, negative_pairs
    
    def _load_scan_context(self, file_path):
        """
        加载或生成ScanContext特征图
        
        参数:
            file_path (str): 文件路径
            
        返回:
            scan_context (numpy.ndarray): ScanContext特征图
        """
        # 检查缓存
        if file_path in self.sc_cache:
            return self.sc_cache[file_path]
        
        # 检查是否有预先计算的ScanContext特征图
        if self.precomputed:
            # 直接加载预先计算的ScanContext特征图
            scan_context = np.load(file_path)
        else:
            # 从点云数据生成ScanContext特征图
            scan_context, _ = self.sc_generator.process_point_cloud_file(file_path)
            
            # 如果指定了缓存目录，则保存ScanContext特征图
            if self.cache_dir:
                # 确保缓存目录存在
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                
                # 生成缓存文件路径
                cache_file = self.cache_dir / f"{os.path.basename(file_path)}.sc.npy"
                
                # 保存ScanContext特征图
                np.save(cache_file, scan_context)
        
        # 更新缓存
        self.sc_cache[file_path] = scan_context
        
        return scan_context
    
    def __len__(self):
        """
        返回数据集大小
        
        返回:
            size (int): 数据集大小
        """
        if self.split == 'train':
            # 训练集使用正样本对和负样本对
            return len(self.positive_pairs) + len(self.negative_pairs)
        else:
            # 验证集和测试集使用单个文件
            return len(self.files)
    
    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        
        参数:
            idx (int): 样本索引
            
        返回:
            如果是训练集：
                anchor (torch.Tensor): 锚点ScanContext特征图
                positive (torch.Tensor): 正样本ScanContext特征图
                negative (torch.Tensor): 负样本ScanContext特征图
                
            如果是验证集或测试集：
                scan_context (torch.Tensor): ScanContext特征图
                file_path (str): 文件路径
        """
        if self.split == 'train':
            # 训练集使用正样本对和负样本对
            if idx < len(self.positive_pairs):
                # 正样本对
                file1, file2 = self.positive_pairs[idx]
                anchor = self._load_scan_context(file1)
                positive = self._load_scan_context(file2)
                
                # 随机选择一个负样本对
                neg_idx = random.randint(0, len(self.negative_pairs) - 1)
                file_neg1, file_neg2 = self.negative_pairs[neg_idx]
                # 随机选择其中一个文件作为负样本
                negative = self._load_scan_context(file_neg1 if random.random() < 0.5 else file_neg2)
            else:
                # 负样本对
                idx = idx - len(self.positive_pairs)
                file1, file2 = self.negative_pairs[idx]
                anchor = self._load_scan_context(file1)
                negative = self._load_scan_context(file2)
                
                # 随机选择一个正样本对
                pos_idx = random.randint(0, len(self.positive_pairs) - 1)
                file_pos1, file_pos2 = self.positive_pairs[pos_idx]
                # 随机选择其中一个文件作为正样本
                positive = self._load_scan_context(file_pos1 if random.random() < 0.5 else file_pos2)
            
            # 数据增强
            if self.use_augmentation:
                # 使用综合数据增强
                anchor = CircularConvDataAugmentation.comprehensive_augmentation(
                    anchor, self.augmentation_config
                )
                positive = CircularConvDataAugmentation.comprehensive_augmentation(
                    positive, self.augmentation_config
                )
                negative = CircularConvDataAugmentation.comprehensive_augmentation(
                    negative, self.augmentation_config
                )
            
            # 添加通道维度
            anchor = np.expand_dims(anchor, axis=0)
            positive = np.expand_dims(positive, axis=0)
            negative = np.expand_dims(negative, axis=0)
            
            # 转换为PyTorch张量
            anchor = torch.from_numpy(anchor).float()
            positive = torch.from_numpy(positive).float()
            negative = torch.from_numpy(negative).float()
            
            # 应用变换
            if self.transform:
                anchor = self.transform(anchor)
                positive = self.transform(positive)
                negative = self.transform(negative)
                
            return anchor, positive, negative
        else:
            # 验证集和测试集使用单个文件
            file_path = self.files[idx]
            scan_context = self._load_scan_context(file_path)
            
            # 添加通道维度
            scan_context = np.expand_dims(scan_context, axis=0)
            
            # 转换为PyTorch张量
            scan_context = torch.from_numpy(scan_context).float()
            
            # 应用变换
            if self.transform:
                scan_context = self.transform(scan_context)
                
            return scan_context, file_path
    
def get_dataloader(data_dir, batch_size=32, num_workers=4,
                  use_augmentation=True, cache_dir=None, precomputed=False,
                  augmentation_config=None):
    """
    获取数据加载器

    参数:
        data_dir (str): 数据目录路径
        batch_size (int): 批次大小
        num_workers (int): 数据加载的工作线程数
        use_augmentation (bool): 是否使用数据增强
        cache_dir (str): 缓存目录路径
        precomputed (bool): 是否使用预先计算的ScanContext特征图
        augmentation_config (dict): 数据增强配置参数

    返回:
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        test_loader (DataLoader): 测试数据加载器
    """
    # 创建数据集
    train_dataset = ScanContextDataset(
        data_dir=data_dir,
        split='train',
        transform=None,
        use_augmentation=use_augmentation,
        cache_dir=cache_dir,
        precomputed=precomputed,
        augmentation_config=augmentation_config
    )
    
    val_dataset = ScanContextDataset(
        data_dir=data_dir,
        split='val',
        transform=None,
        use_augmentation=False,
        cache_dir=cache_dir,
        precomputed=precomputed
    )
    
    test_dataset = ScanContextDataset(
        data_dir=data_dir,
        split='test',
        transform=None,
        use_augmentation=False,
        cache_dir=cache_dir,
        precomputed=precomputed
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
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