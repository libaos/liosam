#!/usr/bin/env python3
"""
评估指标计算
用于回环检测模型性能评估
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score
import time

def evaluate_model(model, data_loader, device, top_k_list=[1, 3, 5, 10], logger=None):
    """
    评估模型性能
    
    参数:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 设备
        top_k_list: Top-K准确率列表
        logger: 日志记录器
        
    返回:
        dict: 评估指标字典
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    if logger:
        logger.info("🔍 开始提取特征...")
    
    # 提取所有特征
    with torch.no_grad():
        for batch_idx, (scan_contexts, labels) in enumerate(data_loader):
            scan_contexts = scan_contexts.to(device)
            labels = labels.to(device)
            
            # 前向传播
            embeddings = model(scan_contexts)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            
            if logger and (batch_idx + 1) % 5 == 0:
                logger.info(f"   处理批次 {batch_idx + 1}/{len(data_loader)}")
    
    # 合并所有特征
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    if logger:
        logger.info(f"✅ 特征提取完成，共 {len(all_embeddings)} 个样本")
    
    # 计算各种指标
    metrics = {}
    
    # 计算Top-K准确率
    for k in top_k_list:
        top_k_acc = compute_top_k_accuracy(all_embeddings, all_labels, k)
        metrics[f'top_{k}'] = top_k_acc
        if logger:
            logger.info(f"   Top-{k} 准确率: {top_k_acc:.4f}")
    
    # 计算mAP
    mAP = compute_mean_average_precision(all_embeddings, all_labels)
    metrics['mAP'] = mAP
    if logger:
        logger.info(f"   mAP: {mAP:.4f}")
    
    # 计算分离比
    separation_ratio = compute_separation_ratio(all_embeddings, all_labels)
    metrics['separation_ratio'] = separation_ratio
    if logger:
        logger.info(f"   分离比: {separation_ratio:.4f}")
    
    # 计算平均距离
    intra_class_dist, inter_class_dist = compute_class_distances(all_embeddings, all_labels)
    metrics['intra_class_distance'] = intra_class_dist
    metrics['inter_class_distance'] = inter_class_dist
    if logger:
        logger.info(f"   类内平均距离: {intra_class_dist:.4f}")
        logger.info(f"   类间平均距离: {inter_class_dist:.4f}")
    
    return metrics

def compute_top_k_accuracy(embeddings, labels, k):
    """
    计算Top-K准确率
    
    参数:
        embeddings (torch.Tensor): 嵌入向量 [N, D]
        labels (torch.Tensor): 标签 [N]
        k (int): K值
        
    返回:
        float: Top-K准确率
    """
    n_samples = embeddings.size(0)
    
    # L2归一化
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # 计算相似度矩阵
    similarity_matrix = torch.matmul(embeddings, embeddings.t())
    
    correct = 0
    total = 0
    
    for i in range(n_samples):
        # 获取当前样本的相似度
        similarities = similarity_matrix[i]
        
        # 排除自己
        similarities[i] = -float('inf')
        
        # 获取Top-K最相似的样本
        _, top_k_indices = torch.topk(similarities, k)
        
        # 检查Top-K中是否有相同标签的样本
        current_label = labels[i]
        top_k_labels = labels[top_k_indices]
        
        if torch.any(top_k_labels == current_label):
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0

def compute_mean_average_precision(embeddings, labels):
    """
    计算平均精度均值 (mAP)
    
    参数:
        embeddings (torch.Tensor): 嵌入向量 [N, D]
        labels (torch.Tensor): 标签 [N]
        
    返回:
        float: mAP值
    """
    n_samples = embeddings.size(0)
    
    # L2归一化
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # 计算相似度矩阵
    similarity_matrix = torch.matmul(embeddings, embeddings.t())
    
    average_precisions = []
    
    for i in range(n_samples):
        # 获取当前样本的相似度
        similarities = similarity_matrix[i]
        
        # 排除自己
        similarities[i] = -float('inf')
        
        # 创建真实标签（相同标签为1，不同标签为0）
        current_label = labels[i]
        true_labels = (labels == current_label).float()
        true_labels[i] = 0  # 排除自己
        
        # 如果没有相同标签的样本，跳过
        if torch.sum(true_labels) == 0:
            continue
        
        # 计算AP
        similarities_np = similarities.numpy()
        true_labels_np = true_labels.numpy()
        
        try:
            ap = average_precision_score(true_labels_np, similarities_np)
            average_precisions.append(ap)
        except:
            continue
    
    return np.mean(average_precisions) if len(average_precisions) > 0 else 0.0

def compute_separation_ratio(embeddings, labels):
    """
    计算类间类内距离分离比
    
    参数:
        embeddings (torch.Tensor): 嵌入向量 [N, D]
        labels (torch.Tensor): 标签 [N]
        
    返回:
        float: 分离比 (类间距离/类内距离)
    """
    intra_dist, inter_dist = compute_class_distances(embeddings, labels)
    
    if intra_dist == 0:
        return float('inf')
    
    return inter_dist / intra_dist

def compute_class_distances(embeddings, labels):
    """
    计算类内和类间平均距离
    
    参数:
        embeddings (torch.Tensor): 嵌入向量 [N, D]
        labels (torch.Tensor): 标签 [N]
        
    返回:
        tuple: (类内平均距离, 类间平均距离)
    """
    # L2归一化
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # 计算距离矩阵
    distances = compute_distance_matrix(embeddings)
    
    # 创建标签掩码
    labels_equal = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels_not_equal = 1.0 - labels_equal
    
    # 排除对角线
    labels_equal.fill_diagonal_(0)
    labels_not_equal.fill_diagonal_(0)
    
    # 计算类内平均距离
    intra_distances = distances * labels_equal
    intra_count = torch.sum(labels_equal)
    intra_class_dist = torch.sum(intra_distances) / intra_count if intra_count > 0 else 0.0
    
    # 计算类间平均距离
    inter_distances = distances * labels_not_equal
    inter_count = torch.sum(labels_not_equal)
    inter_class_dist = torch.sum(inter_distances) / inter_count if inter_count > 0 else 0.0
    
    # 确保返回Python标量
    if hasattr(intra_class_dist, 'item'):
        intra_class_dist = intra_class_dist.item()
    if hasattr(inter_class_dist, 'item'):
        inter_class_dist = inter_class_dist.item()

    return float(intra_class_dist), float(inter_class_dist)

def compute_distance_matrix(embeddings):
    """
    计算距离矩阵
    
    参数:
        embeddings (torch.Tensor): 嵌入向量 [N, D]
        
    返回:
        torch.Tensor: 距离矩阵 [N, N]
    """
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

def compute_retrieval_metrics(embeddings, labels, distance_threshold=0.5):
    """
    计算检索相关指标
    
    参数:
        embeddings (torch.Tensor): 嵌入向量 [N, D]
        labels (torch.Tensor): 标签 [N]
        distance_threshold (float): 距离阈值
        
    返回:
        dict: 检索指标
    """
    # L2归一化
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # 计算距离矩阵
    distances = compute_distance_matrix(embeddings)
    
    # 创建真实标签矩阵
    labels_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels_matrix.fill_diagonal_(0)  # 排除自己
    
    # 基于距离阈值的预测
    predictions = (distances <= distance_threshold).float()
    predictions.fill_diagonal_(0)  # 排除自己
    
    # 计算TP, FP, TN, FN
    tp = torch.sum(predictions * labels_matrix)
    fp = torch.sum(predictions * (1 - labels_matrix))
    tn = torch.sum((1 - predictions) * (1 - labels_matrix))
    fn = torch.sum((1 - predictions) * labels_matrix)
    
    # 计算精确率、召回率、F1分数
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    
    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1_score': f1_score.item(),
        'accuracy': accuracy.item(),
        'tp': tp.item(),
        'fp': fp.item(),
        'tn': tn.item(),
        'fn': fn.item()
    }

# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    batch_size = 20
    embedding_dim = 256
    
    embeddings = torch.randn(batch_size, embedding_dim)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    
    print("测试评估指标:")
    
    # 测试Top-K准确率
    for k in [1, 3, 5]:
        top_k_acc = compute_top_k_accuracy(embeddings, labels, k)
        print(f"Top-{k} 准确率: {top_k_acc:.4f}")
    
    # 测试mAP
    mAP = compute_mean_average_precision(embeddings, labels)
    print(f"mAP: {mAP:.4f}")
    
    # 测试分离比
    separation_ratio = compute_separation_ratio(embeddings, labels)
    print(f"分离比: {separation_ratio:.4f}")
    
    # 测试类内外距离
    intra_dist, inter_dist = compute_class_distances(embeddings, labels)
    print(f"类内距离: {intra_dist:.4f}")
    print(f"类间距离: {inter_dist:.4f}")
    
    # 测试检索指标
    retrieval_metrics = compute_retrieval_metrics(embeddings, labels)
    print(f"检索指标: {retrieval_metrics}")
