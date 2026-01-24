#!/usr/bin/env python3
"""
三元组损失函数实现
用于回环检测训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TripletLoss(nn.Module):
    """三元组损失函数"""
    
    def __init__(self, margin=0.5, mining_strategy='hard'):
        """
        初始化三元组损失
        
        参数:
            margin (float): 边界值，默认0.5
            mining_strategy (str): 挖掘策略，'hard', 'semi_hard', 'easy'
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
        
    def forward(self, embeddings, labels):
        """
        前向传播
        
        参数:
            embeddings (torch.Tensor): 嵌入向量 [B, D]
            labels (torch.Tensor): 标签 [B]
            
        返回:
            torch.Tensor: 损失值
        """
        # 计算距离矩阵
        distances = self._compute_distance_matrix(embeddings)
        
        # 根据挖掘策略选择三元组
        if self.mining_strategy == 'hard':
            return self._hard_triplet_loss(distances, labels)
        elif self.mining_strategy == 'semi_hard':
            return self._semi_hard_triplet_loss(distances, labels)
        else:
            return self._batch_all_triplet_loss(distances, labels)
    
    def _compute_distance_matrix(self, embeddings):
        """计算距离矩阵"""
        # L2归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 计算欧几里得距离矩阵
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = F.relu(distances)
        
        # 避免数值不稳定
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)
        
        return distances
    
    def _hard_triplet_loss(self, distances, labels):
        """硬三元组挖掘损失"""
        batch_size = labels.size(0)
        
        # 创建标签掩码
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        # 对于每个anchor，找到最难的positive和negative
        # 最难的positive：同类中距离最远的
        mask_anchor_positive = labels_equal.clone()
        mask_anchor_positive = mask_anchor_positive.fill_diagonal_(False)
        
        # 最难的negative：不同类中距离最近的
        mask_anchor_negative = labels_not_equal
        
        # 计算hardest positive距离
        max_anchor_positive_dist, _ = torch.max(
            distances * mask_anchor_positive.float(), dim=1, keepdim=True
        )
        
        # 计算hardest negative距离
        min_anchor_negative_dist, _ = torch.min(
            distances + (~mask_anchor_negative).float() * 1e16, dim=1, keepdim=True
        )
        
        # 计算三元组损失
        triplet_loss = F.relu(max_anchor_positive_dist - min_anchor_negative_dist + self.margin)
        
        return triplet_loss.mean()
    
    def _semi_hard_triplet_loss(self, distances, labels):
        """半硬三元组挖掘损失"""
        batch_size = labels.size(0)
        
        # 创建标签掩码
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        # 对角线设为False（排除自己）
        labels_equal = labels_equal.fill_diagonal_(False)
        
        # 计算所有有效的三元组
        triplet_loss = []
        
        for i in range(batch_size):
            # 找到所有positive样本
            positive_mask = labels_equal[i]
            positive_distances = distances[i][positive_mask]
            
            # 找到所有negative样本
            negative_mask = labels_not_equal[i]
            negative_distances = distances[i][negative_mask]
            
            if len(positive_distances) == 0 or len(negative_distances) == 0:
                continue
            
            # 对每个positive，找semi-hard negative
            for pos_dist in positive_distances:
                # Semi-hard negative: d(a,n) > d(a,p) but d(a,n) < d(a,p) + margin
                semi_hard_negatives = negative_distances[
                    (negative_distances > pos_dist) & 
                    (negative_distances < pos_dist + self.margin)
                ]
                
                if len(semi_hard_negatives) > 0:
                    # 选择最难的semi-hard negative
                    hardest_negative = torch.min(semi_hard_negatives)
                    loss = F.relu(pos_dist - hardest_negative + self.margin)
                    triplet_loss.append(loss)
        
        if len(triplet_loss) == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        return torch.stack(triplet_loss).mean()
    
    def _batch_all_triplet_loss(self, distances, labels):
        """批量所有三元组损失"""
        batch_size = labels.size(0)
        
        # 创建标签掩码
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        # 对角线设为False
        labels_equal = labels_equal.fill_diagonal_(False)
        
        # 扩展距离矩阵用于三元组计算
        anchor_positive_dist = distances.unsqueeze(2)  # [B, B, 1]
        anchor_negative_dist = distances.unsqueeze(1)  # [B, 1, B]
        
        # 计算三元组损失
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        
        # 创建有效三元组掩码
        mask = labels_equal.unsqueeze(2) & labels_not_equal.unsqueeze(1)
        
        # 只保留有效的三元组
        triplet_loss = triplet_loss * mask.float()
        
        # 应用ReLU并计算平均值
        triplet_loss = F.relu(triplet_loss)
        
        # 计算有效三元组的数量
        num_positive_triplets = torch.sum(mask.float())
        
        if num_positive_triplets == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        return torch.sum(triplet_loss) / num_positive_triplets

class ContrastiveLoss(nn.Module):
    """对比损失函数（备选）"""
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embeddings, labels):
        """
        前向传播
        
        参数:
            embeddings (torch.Tensor): 嵌入向量 [B, D]
            labels (torch.Tensor): 标签 [B]
            
        返回:
            torch.Tensor: 损失值
        """
        batch_size = embeddings.size(0)
        
        # 计算距离矩阵
        distances = self._compute_distance_matrix(embeddings)
        
        # 创建标签掩码
        labels_equal = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels_not_equal = 1.0 - labels_equal
        
        # 对角线设为0
        labels_equal = labels_equal.fill_diagonal_(0)
        
        # 正样本对损失
        positive_loss = labels_equal * torch.pow(distances, 2)
        
        # 负样本对损失
        negative_loss = labels_not_equal * torch.pow(F.relu(self.margin - distances), 2)
        
        # 总损失
        loss = positive_loss + negative_loss
        
        # 计算有效对的数量
        num_pairs = torch.sum(labels_equal) + torch.sum(labels_not_equal)
        
        if num_pairs == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        return torch.sum(loss) / num_pairs
    
    def _compute_distance_matrix(self, embeddings):
        """计算距离矩阵"""
        # L2归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 计算欧几里得距离矩阵
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = F.relu(distances)
        
        # 避免数值不稳定
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)
        
        return distances

# 测试代码
if __name__ == "__main__":
    # 测试三元组损失
    batch_size = 8
    embedding_dim = 256
    
    embeddings = torch.randn(batch_size, embedding_dim)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    
    # 测试不同的挖掘策略
    for strategy in ['hard', 'semi_hard', 'easy']:
        criterion = TripletLoss(margin=0.5, mining_strategy=strategy)
        loss = criterion(embeddings, labels)
        print(f"{strategy} triplet loss: {loss.item():.4f}")
    
    # 测试对比损失
    contrastive_criterion = ContrastiveLoss(margin=1.0)
    contrastive_loss = contrastive_criterion(embeddings, labels)
    print(f"Contrastive loss: {contrastive_loss.item():.4f}")
