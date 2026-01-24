#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单评估脚本
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
from pathlib import Path
import json
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 导入模型
from models.temporal_3d_cnn import Temporal3DCNN
from models.temporal_2d_cnn import Temporal2DCNN
from utils.temporal_dataset import TemporalScanContextDataset

def load_model(checkpoint_path, model_type):
    """加载训练好的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    if model_type == 'temporal_3d_cnn':
        model = Temporal3DCNN(sequence_length=5, num_classes=20)
    elif model_type == 'temporal_2d_cnn':
        model = Temporal2DCNN(sequence_length=5, num_classes=20)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device

def create_test_dataset():
    """创建测试数据集"""
    
    class SimpleTemporalDataset(TemporalScanContextDataset):
        def _load_data(self):
            # 加载预处理的数据
            data_file = Path("data/processed/temporal_sequences_len5.pkl")
            if data_file.exists():
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.sequences = data['sequences']
                    self.labels = data['labels']
                    self.file_paths = data['file_paths'] if 'file_paths' in data else []
                print(f"加载了 {len(self.sequences)} 个样本")
            else:
                raise FileNotFoundError("未找到预处理数据文件")
    
    # 创建数据集
    dataset = SimpleTemporalDataset(
        data_dir="data/processed",
        split='train',
        sequence_length=5,
        use_augmentation=False
    )
    
    # 数据集划分 - 使用与训练时相同的划分
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子确保一致性
    )
    
    # 创建测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print(f"测试集大小: {len(test_dataset)} 样本")
    return test_loader

def evaluate_model(model, test_loader, device, model_name):
    """评估模型性能"""
    print(f"\n评估 {model_name} 模型...")
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            # 测量推理时间
            start_time = time.time()
            output = model(data)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # 获取预测结果
            probabilities = torch.softmax(output, dim=1)
            _, predicted = output.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # 计算评估指标
    accuracy = accuracy_score(all_targets, all_predictions)
    
    # Top-k准确率
    def top_k_accuracy(y_true, y_prob, k):
        top_k_pred = np.argsort(y_prob, axis=1)[:, -k:]
        return np.mean([y_true[i] in top_k_pred[i] for i in range(len(y_true))])
    
    top1_acc = accuracy
    top3_acc = top_k_accuracy(all_targets, all_probabilities, 3)
    top5_acc = top_k_accuracy(all_targets, all_probabilities, 5)
    
    # 分类报告
    unique_classes = np.unique(np.concatenate([all_targets, all_predictions]))
    num_classes = len(unique_classes)
    class_names = [f'Path_{i:02d}' for i in unique_classes]
    report = classification_report(all_targets, all_predictions,
                                 target_names=class_names, output_dict=True,
                                 labels=unique_classes)
    
    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)
    
    # 推理性能
    avg_inference_time = np.mean(inference_times)
    fps = len(all_predictions) / sum(inference_times)
    
    # 打印结果
    print(f"\n{model_name} 模型评估结果:")
    print(f"  测试样本数: {len(all_predictions)}")
    print(f"  准确率 (Top-1): {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"  Top-3 准确率: {top3_acc:.4f} ({top3_acc*100:.2f}%)")
    print(f"  Top-5 准确率: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    print(f"  宏平均精确率: {report['macro avg']['precision']:.4f}")
    print(f"  宏平均召回率: {report['macro avg']['recall']:.4f}")
    print(f"  宏平均F1分数: {report['macro avg']['f1-score']:.4f}")
    print(f"  平均推理时间: {avg_inference_time:.4f}s")
    print(f"  推理速度: {fps:.2f} FPS")
    
    # 返回结果
    results = {
        'model_name': model_name,
        'accuracy': float(top1_acc),
        'top3_accuracy': float(top3_acc),
        'top5_accuracy': float(top5_acc),
        'macro_precision': float(report['macro avg']['precision']),
        'macro_recall': float(report['macro avg']['recall']),
        'macro_f1': float(report['macro avg']['f1-score']),
        'avg_inference_time': float(avg_inference_time),
        'fps': float(fps),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    return results

def compare_models():
    """比较两个模型的性能"""
    print("=" * 60)
    print("时序回环检测模型性能评估")
    print("=" * 60)
    
    # 创建测试数据集
    test_loader = create_test_dataset()
    
    # 评估3D CNN模型
    model_3d, device = load_model('outputs/temporal_3d_cnn_training/checkpoint_best.pth', 'temporal_3d_cnn')
    results_3d = evaluate_model(model_3d, test_loader, device, '3D CNN')
    
    # 评估2D CNN模型
    model_2d, device = load_model('outputs/temporal_2d_cnn_training/checkpoint_best.pth', 'temporal_2d_cnn')
    results_2d = evaluate_model(model_2d, test_loader, device, '2D CNN')
    
    # 对比分析
    print("\n" + "=" * 60)
    print("模型性能对比")
    print("=" * 60)
    
    print(f"{'指标':<20} {'3D CNN':<15} {'2D CNN':<15} {'差异':<15}")
    print("-" * 65)
    
    metrics = [
        ('准确率', 'accuracy'),
        ('Top-3准确率', 'top3_accuracy'),
        ('Top-5准确率', 'top5_accuracy'),
        ('宏平均F1', 'macro_f1'),
        ('推理速度(FPS)', 'fps')
    ]
    
    for metric_name, metric_key in metrics:
        val_3d = results_3d[metric_key]
        val_2d = results_2d[metric_key]
        diff = val_3d - val_2d
        
        if metric_key == 'fps':
            print(f"{metric_name:<20} {val_3d:<15.2f} {val_2d:<15.2f} {diff:<15.2f}")
        else:
            print(f"{metric_name:<20} {val_3d:<15.4f} {val_2d:<15.4f} {diff:<15.4f}")
    
    # 保存结果
    comparison_results = {
        '3d_cnn': results_3d,
        '2d_cnn': results_2d,
        'summary': {
            'better_accuracy': '3D CNN' if results_3d['accuracy'] > results_2d['accuracy'] else '2D CNN',
            'better_speed': '3D CNN' if results_3d['fps'] > results_2d['fps'] else '2D CNN',
            'accuracy_difference': results_3d['accuracy'] - results_2d['accuracy'],
            'speed_difference': results_3d['fps'] - results_2d['fps']
        }
    }
    
    with open('outputs/model_comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n详细结果已保存到: outputs/model_comparison_results.json")
    
    # 结论
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    
    if results_3d['accuracy'] > results_2d['accuracy']:
        print(f"✓ 3D CNN在准确率上优于2D CNN ({results_3d['accuracy']:.4f} vs {results_2d['accuracy']:.4f})")
    elif results_2d['accuracy'] > results_3d['accuracy']:
        print(f"✓ 2D CNN在准确率上优于3D CNN ({results_2d['accuracy']:.4f} vs {results_3d['accuracy']:.4f})")
    else:
        print("✓ 两个模型准确率相当")
    
    if results_3d['fps'] > results_2d['fps']:
        print(f"✓ 3D CNN推理速度更快 ({results_3d['fps']:.2f} vs {results_2d['fps']:.2f} FPS)")
    else:
        print(f"✓ 2D CNN推理速度更快 ({results_2d['fps']:.2f} vs {results_3d['fps']:.2f} FPS)")
    
    return comparison_results

if __name__ == '__main__':
    try:
        results = compare_models()
        print("\n✓ 评估完成！")
    except Exception as e:
        print(f"✗ 评估失败: {e}")
        import traceback
        traceback.print_exc()
