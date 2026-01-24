"""
评估指标计算
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn.functional as F

def calculate_top_k_accuracy(similarities, labels, k_values=[1, 3, 5, 10]):
    """计算Top-K准确率"""
    results = {}
    
    # 获取排序后的索引
    sorted_indices = torch.argsort(similarities, dim=1, descending=True)
    
    for k in k_values:
        if k > similarities.shape[1]:
            results[f'top_{k}'] = 0.0
            continue
            
        # 获取前k个预测
        top_k_preds = sorted_indices[:, :k]
        
        # 计算准确率
        correct = 0
        for i, label in enumerate(labels):
            if label.item() in top_k_preds[i]:
                correct += 1
        
        results[f'top_{k}'] = correct / len(labels)
    
    return results

def calculate_map(similarities, labels):
    """计算平均精度(mAP)"""
    ap_scores = []
    
    for i, label in enumerate(labels):
        # 获取当前查询的相似度
        query_similarities = similarities[i]
        
        # 排序
        sorted_indices = torch.argsort(query_similarities, descending=True)
        
        # 计算精度
        precisions = []
        relevant_count = 0
        
        for j, idx in enumerate(sorted_indices):
            if labels[idx] == label:
                relevant_count += 1
                precision = relevant_count / (j + 1)
                precisions.append(precision)
        
        if precisions:
            ap_scores.append(np.mean(precisions))
        else:
            ap_scores.append(0.0)
    
    return np.mean(ap_scores)

def calculate_mrr(similarities, labels):
    """计算平均倒数排名(MRR)"""
    rr_scores = []
    
    for i, label in enumerate(labels):
        query_similarities = similarities[i]
        sorted_indices = torch.argsort(query_similarities, descending=True)
        
        # 找到第一个正确匹配的位置
        for rank, idx in enumerate(sorted_indices):
            if labels[idx] == label and idx != i:  # 排除自己
                rr_scores.append(1.0 / (rank + 1))
                break
        else:
            rr_scores.append(0.0)
    
    return np.mean(rr_scores)

def calculate_separation_ratio(features, labels):
    """计算类间距离与类内距离的比值"""
    unique_labels = torch.unique(labels)
    
    intra_distances = []
    inter_distances = []
    
    for label in unique_labels:
        mask = labels == label
        class_features = features[mask]
        
        if len(class_features) > 1:
            # 类内距离
            for i in range(len(class_features)):
                for j in range(i + 1, len(class_features)):
                    dist = F.pairwise_distance(
                        class_features[i:i+1], 
                        class_features[j:j+1]
                    ).item()
                    intra_distances.append(dist)
        
        # 类间距离
        other_features = features[~mask]
        if len(other_features) > 0:
            for class_feat in class_features:
                for other_feat in other_features:
                    dist = F.pairwise_distance(
                        class_feat.unsqueeze(0), 
                        other_feat.unsqueeze(0)
                    ).item()
                    inter_distances.append(dist)
    
    if not intra_distances or not inter_distances:
        return 1.0
    
    avg_intra = np.mean(intra_distances)
    avg_inter = np.mean(inter_distances)
    
    return avg_inter / avg_intra if avg_intra > 0 else 1.0

def calculate_metrics(features, labels, similarities=None):
    """计算所有评估指标"""
    if similarities is None:
        # 计算余弦相似度
        features_norm = F.normalize(features, p=2, dim=1)
        similarities = torch.mm(features_norm, features_norm.t())
    
    # Top-K准确率
    top_k_results = calculate_top_k_accuracy(similarities, labels)
    
    # mAP和MRR
    map_score = calculate_map(similarities, labels)
    mrr_score = calculate_mrr(similarities, labels)
    
    # 分离比
    separation_ratio = calculate_separation_ratio(features, labels)
    
    # 分类准确率（使用最相似的作为预测）
    predictions = torch.argmax(similarities, dim=1)
    classification_acc = accuracy_score(labels.cpu(), predictions.cpu())
    
    # 精确率、召回率、F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.cpu(), predictions.cpu(), average='macro', zero_division=0
    )
    
    results = {
        **top_k_results,
        'mAP': map_score,
        'MRR': mrr_score,
        'separation_ratio': separation_ratio,
        'classification_accuracy': classification_acc,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_score_macro': f1
    }
    
    return results
