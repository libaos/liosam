#!/usr/bin/env python3
"""
简化的对比损失函数
专门用于回环检测训练，处理稀疏标签问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleContrastiveLoss(nn.Module):
    """简化的对比损失函数"""
    
    def __init__(self, margin=1.0, temperature=0.1):
        """
        初始化对比损失
        
        参数:
            margin (float): 边界值
            temperature (float): 温度参数
        """
        super(SimpleContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        
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
        
        if batch_size < 2:
            return torch.tensor(0.0, requires_grad=True)
        
        # L2归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # 创建标签掩码
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float()
        
        # 移除对角线（自己与自己的相似度）
        mask = mask - torch.eye(batch_size, device=mask.device)
        
        # 计算正样本和负样本的损失
        positive_mask = mask
        negative_mask = 1.0 - mask - torch.eye(batch_size, device=mask.device)
        
        # 正样本损失：相似度应该高
        positive_similarities = similarity_matrix * positive_mask
        positive_count = torch.sum(positive_mask)
        
        if positive_count > 0:
            positive_loss = -torch.sum(positive_similarities) / positive_count
        else:
            positive_loss = torch.tensor(0.0, device=embeddings.device)
        
        # 负样本损失：相似度应该低
        negative_similarities = similarity_matrix * negative_mask
        negative_loss = torch.sum(F.relu(negative_similarities - self.margin)) / torch.sum(negative_mask)
        
        # 总损失
        total_loss = positive_loss + negative_loss
        
        return total_loss

class InfoNCELoss(nn.Module):
    """InfoNCE损失函数（对比学习）"""
    
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        
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
        
        if batch_size < 2:
            return torch.tensor(0.0, requires_grad=True)
        
        # L2归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # 创建标签掩码
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float()
        
        # 移除对角线
        mask = mask - torch.eye(batch_size, device=mask.device)
        
        # 计算InfoNCE损失
        # 对每个样本，正样本是相同标签的样本，负样本是不同标签的样本
        losses = []
        
        for i in range(batch_size):
            # 当前样本的相似度
            similarities = similarity_matrix[i]
            
            # 正样本掩码（相同标签，排除自己）
            positive_mask = mask[i]
            
            # 如果没有正样本，跳过
            if torch.sum(positive_mask) == 0:
                continue
            
            # 负样本掩码（不同标签）
            negative_mask = 1.0 - positive_mask - torch.eye(1, batch_size, device=mask.device)[0]
            
            # 计算正样本的平均相似度
            positive_similarities = similarities * positive_mask
            positive_exp = torch.exp(positive_similarities)
            positive_sum = torch.sum(positive_exp * positive_mask)
            
            # 计算所有样本的相似度（排除自己）
            all_mask = 1.0 - torch.eye(1, batch_size, device=mask.device)[0]
            all_exp = torch.exp(similarities * all_mask)
            all_sum = torch.sum(all_exp * all_mask)
            
            # InfoNCE损失
            if all_sum > 0 and positive_sum > 0:
                loss = -torch.log(positive_sum / all_sum)
                losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        return torch.stack(losses).mean()

class AdaptiveTripletLoss(nn.Module):
    """自适应三元组损失，处理稀疏标签"""
    
    def __init__(self, margin=0.5, adaptive_margin=True):
        super(AdaptiveTripletLoss, self).__init__()
        self.margin = margin
        self.adaptive_margin = adaptive_margin
        
    def forward(self, embeddings, labels):
        """前向传播"""
        batch_size = embeddings.size(0)
        
        if batch_size < 3:
            return torch.tensor(0.0, requires_grad=True)
        
        # L2归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 计算距离矩阵
        distances = self._compute_distance_matrix(embeddings)
        
        # 创建标签掩码
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        # 移除对角线
        labels_equal = labels_equal.fill_diagonal_(False)
        
        # 收集有效的三元组
        triplet_losses = []
        
        for i in range(batch_size):
            # 找到正样本
            positive_mask = labels_equal[i]
            positive_indices = torch.where(positive_mask)[0]
            
            # 找到负样本
            negative_mask = labels_not_equal[i]
            negative_indices = torch.where(negative_mask)[0]
            
            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue
            
            # 选择最难的正样本和负样本
            positive_distances = distances[i][positive_indices]
            negative_distances = distances[i][negative_indices]
            
            # 最难的正样本（距离最远）
            hardest_positive_dist = torch.max(positive_distances)
            
            # 最难的负样本（距离最近）
            hardest_negative_dist = torch.min(negative_distances)
            
            # 自适应边界
            if self.adaptive_margin:
                # 根据正负样本距离动态调整边界
                adaptive_margin = self.margin * (1 + hardest_positive_dist.item())
            else:
                adaptive_margin = self.margin
            
            # 计算三元组损失
            loss = F.relu(hardest_positive_dist - hardest_negative_dist + adaptive_margin)
            triplet_losses.append(loss)
        
        if len(triplet_losses) == 0:
            # 如果没有有效三元组，使用简单的对比损失
            return self._fallback_contrastive_loss(embeddings, labels)
        
        return torch.stack(triplet_losses).mean()
    
    def _compute_distance_matrix(self, embeddings):
        """计算距离矩阵"""
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
    
    def _fallback_contrastive_loss(self, embeddings, labels):
        """备用对比损失"""
        contrastive_loss = SimpleContrastiveLoss(margin=self.margin)
        return contrastive_loss(embeddings, labels)

# 测试代码
if __name__ == "__main__":
    # 测试数据
    batch_size = 8
    embedding_dim = 256
    
    embeddings = torch.randn(batch_size, embedding_dim)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])  # 有重复标签
    
    print("测试损失函数:")
    
    # 测试简单对比损失
    simple_loss = SimpleContrastiveLoss()
    loss1 = simple_loss(embeddings, labels)
    print(f"Simple Contrastive Loss: {loss1.item():.4f}")
    
    # 测试InfoNCE损失
    infonce_loss = InfoNCELoss()
    loss2 = infonce_loss(embeddings, labels)
    print(f"InfoNCE Loss: {loss2.item():.4f}")
    
    # 测试自适应三元组损失
    adaptive_loss = AdaptiveTripletLoss()
    loss3 = adaptive_loss(embeddings, labels)
    print(f"Adaptive Triplet Loss: {loss3.item():.4f}")
    
    # 测试稀疏标签情况
    sparse_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])  # 每个样本唯一标签
    print(f"\n稀疏标签测试:")
    
    loss1_sparse = simple_loss(embeddings, sparse_labels)
    print(f"Simple Contrastive Loss (sparse): {loss1_sparse.item():.4f}")
    
    loss2_sparse = infonce_loss(embeddings, sparse_labels)
    print(f"InfoNCE Loss (sparse): {loss2_sparse.item():.4f}")
    
    loss3_sparse = adaptive_loss(embeddings, sparse_labels)
    print(f"Adaptive Triplet Loss (sparse): {loss3_sparse.item():.4f}")
