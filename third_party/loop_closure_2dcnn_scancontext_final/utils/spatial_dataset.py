#!/usr/bin/env python3
"""
简化的空间注意力训练数据集
专门用于SCStandardSpatialCNN训练
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import glob
import random
from .scan_context import ScanContext
from .ply_reader import PLYReader

class SpatialScanContextDataset(Dataset):
    """简化的ScanContext数据集，用于空间注意力模型训练"""
    
    def __init__(self, data_dir, split='train', split_ratio=0.8, max_files=None, 
                 use_augmentation=False, seed=42):
        """
        初始化数据集
        
        参数:
            data_dir (str): PLY文件目录
            split (str): 'train' 或 'val'
            split_ratio (float): 训练集比例
            max_files (int): 最大文件数量
            use_augmentation (bool): 是否使用数据增强
            seed (int): 随机种子
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.use_augmentation = use_augmentation
        
        # 获取所有PLY文件
        ply_files = sorted(glob.glob(str(self.data_dir / "*.ply")))
        
        if len(ply_files) == 0:
            raise ValueError(f"在 {self.data_dir} 中未找到PLY文件")
        
        # 限制文件数量
        if max_files and max_files < len(ply_files):
            ply_files = ply_files[:max_files]
        
        # 对于连续路径数据，使用时序划分而不是随机划分
        # 保持文件的时序顺序
        print(f"📍 数据集类型: 连续路径数据，使用时序划分")

        # 时序划分：前80%作为训练集，后20%作为验证集
        split_idx = int(len(ply_files) * split_ratio)

        if split == 'train':
            self.files = ply_files[:split_idx]
            print(f"📊 训练集: 路径前{split_ratio*100:.0f}% ({len(self.files)}个文件)")
        else:  # val
            self.files = ply_files[split_idx:]
            print(f"📊 验证集: 路径后{(1-split_ratio)*100:.0f}% ({len(self.files)}个文件)")
        
        if len(self.files) == 0:
            raise ValueError(f"划分后的{split}集为空")
        
        print(f"📊 {split}集: {len(self.files)} 个文件")
        
        # 创建ScanContext生成器
        self.sc_generator = ScanContext()
        
        # 创建PLY读取器
        self.ply_reader = PLYReader()

        # 创建更合理的标签策略 - 基于位置分组
        # 将连续的位置分组，每组包含多个相似位置
        self.labels = self._create_position_labels()

        print(f"📊 标签统计: 共{len(set(self.labels))}个不同标签，平均每个标签{len(self.labels)/len(set(self.labels)):.1f}个样本")

    def _create_position_labels(self):
        """
        创建基于连续路径的标签
        对于连续路径数据，相邻位置应该有相似的标签
        """
        labels = []
        group_size = 10  # 每10个连续位置为一组（增加组大小以获得更多正样本）

        for i, file_path in enumerate(self.files):
            # 从文件名中提取位置索引
            file_name = Path(file_path).stem
            try:
                # 假设文件名格式为 "cloud_NNNNN.ply"
                import re
                numbers = re.findall(r'\d+', file_name)
                if numbers:
                    position_idx = int(numbers[-1])  # 使用最后一个数字作为位置索引
                else:
                    position_idx = i  # 如果没有数字，使用文件索引
            except:
                # 如果提取失败，使用文件在列表中的索引
                position_idx = i

            # 将位置索引分组
            group_label = position_idx // group_size
            labels.append(group_label)

        # 确保标签从0开始连续
        unique_labels = sorted(set(labels))
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        labels = [label_mapping[label] for label in labels]

        print(f"📊 创建了 {len(unique_labels)} 个位置组，每组约 {group_size} 个连续位置")

        # 显示标签分布
        from collections import Counter
        label_counts = Counter(labels)
        avg_samples_per_label = len(labels) / len(unique_labels)
        print(f"📊 标签分布: 平均每组 {avg_samples_per_label:.1f} 个样本")

        return labels
        
        # 创建更合理的标签策略 - 基于位置分组
        # 将连续的位置分组，每组包含多个相似位置
        self.labels = self._create_position_labels()

        print(f"📊 标签统计: 共{len(set(self.labels))}个不同标签，平均每个标签{len(self.labels)/len(set(self.labels)):.1f}个样本")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        返回:
            scan_context (torch.Tensor): ScanContext特征图 [1, H, W]
            label (int): 标签（文件索引）
        """
        try:
            # 读取PLY文件
            ply_file = self.files[idx]
            points = self.ply_reader.read_ply_file(ply_file)
            
            # 生成ScanContext
            scan_context = self.sc_generator.generate_scan_context(points)
            
            # 数据增强
            if self.use_augmentation and self.split == 'train':
                scan_context = self._augment_scan_context(scan_context)
            
            # 转换为tensor
            scan_context = torch.from_numpy(scan_context).float()
            
            # 添加通道维度
            if len(scan_context.shape) == 2:
                scan_context = scan_context.unsqueeze(0)  # [1, H, W]
            
            # 标签
            label = self.labels[idx]
            
            return scan_context, label
            
        except Exception as e:
            print(f"❌ 读取文件失败 {self.files[idx]}: {e}")
            # 返回零数据
            return torch.zeros(1, 20, 60), 0
    
    def _augment_scan_context(self, scan_context):
        """
        简单的数据增强
        
        参数:
            scan_context (np.ndarray): 原始ScanContext
            
        返回:
            np.ndarray: 增强后的ScanContext
        """
        # 随机旋转（循环移位）
        if random.random() < 0.7:
            shift = random.randint(1, scan_context.shape[1] - 1)
            scan_context = np.roll(scan_context, shift, axis=1)
        
        # 添加噪声
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.01, scan_context.shape)
            scan_context = scan_context + noise
            scan_context = np.clip(scan_context, 0, None)
        
        # 强度缩放
        if random.random() < 0.3:
            scale = random.uniform(0.9, 1.1)
            scan_context = scan_context * scale
        
        return scan_context
    
    def get_file_info(self, idx):
        """获取文件信息"""
        return {
            'file_path': self.files[idx],
            'label': self.labels[idx],
            'split': self.split
        }

class TripletScanContextDataset(Dataset):
    """三元组ScanContext数据集，用于三元组损失训练"""
    
    def __init__(self, base_dataset, triplets_per_sample=5):
        """
        初始化三元组数据集
        
        参数:
            base_dataset: 基础数据集
            triplets_per_sample: 每个样本生成的三元组数量
        """
        self.base_dataset = base_dataset
        self.triplets_per_sample = triplets_per_sample
        
        # 按标签组织数据
        self.label_to_indices = {}
        for idx in range(len(base_dataset)):
            label = base_dataset.labels[idx]
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        # 生成三元组
        self.triplets = self._generate_triplets()
        
    def _generate_triplets(self):
        """生成三元组"""
        triplets = []
        
        for anchor_idx in range(len(self.base_dataset)):
            anchor_label = self.base_dataset.labels[anchor_idx]
            
            # 获取正样本候选（相同标签）
            positive_candidates = [idx for idx in self.label_to_indices[anchor_label] 
                                 if idx != anchor_idx]
            
            # 获取负样本候选（不同标签）
            negative_candidates = []
            for label, indices in self.label_to_indices.items():
                if label != anchor_label:
                    negative_candidates.extend(indices)
            
            # 生成三元组
            for _ in range(self.triplets_per_sample):
                if len(positive_candidates) > 0 and len(negative_candidates) > 0:
                    positive_idx = random.choice(positive_candidates)
                    negative_idx = random.choice(negative_candidates)
                    triplets.append((anchor_idx, positive_idx, negative_idx))
        
        return triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        """
        获取三元组数据
        
        返回:
            anchor, positive, negative: 三个ScanContext特征图
            labels: 对应的标签
        """
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]
        
        anchor_sc, anchor_label = self.base_dataset[anchor_idx]
        positive_sc, positive_label = self.base_dataset[positive_idx]
        negative_sc, negative_label = self.base_dataset[negative_idx]
        
        return (anchor_sc, positive_sc, negative_sc), (anchor_label, positive_label, negative_label)

# 测试代码
if __name__ == "__main__":
    # 测试数据集
    try:
        dataset = SpatialScanContextDataset(
            data_dir="data/raw/ply_files",
            split='train',
            max_files=10,
            use_augmentation=True
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        # 测试获取数据
        scan_context, label = dataset[0]
        print(f"ScanContext形状: {scan_context.shape}")
        print(f"标签: {label}")
        
        # 测试三元组数据集
        triplet_dataset = TripletScanContextDataset(dataset, triplets_per_sample=2)
        print(f"三元组数据集大小: {len(triplet_dataset)}")
        
        triplet_data, triplet_labels = triplet_dataset[0]
        print(f"三元组数据形状: {[x.shape for x in triplet_data]}")
        print(f"三元组标签: {triplet_labels}")
        
    except Exception as e:
        print(f"测试失败: {e}")
