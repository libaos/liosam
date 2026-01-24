#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成论文所需的关键实验结果
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
from models.temporal_models import *
import warnings
warnings.filterwarnings('ignore')

def quick_train_and_test(model, train_data, train_labels, test_data, test_labels, epochs=30):
    """快速训练和测试模型"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 准备数据
    train_tensor = torch.FloatTensor(train_data).to(device)
    train_labels_tensor = torch.LongTensor(train_labels).to(device)
    test_tensor = torch.FloatTensor(test_data).to(device)
    
    train_dataset = TensorDataset(train_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=min(32, len(train_data)), shuffle=True)
    
    # 训练
    model.train()
    for epoch in range(epochs):
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
    
    # 测试
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = accuracy_score(test_labels, predicted.cpu().numpy())
    
    return accuracy, predicted.cpu().numpy()

def experiment_1_temporal_split():
    """实验1: 时间分离测试"""
    
    print("\n" + "="*60)
    print("实验1: 时间分离测试 - 避免时序数据泄露")
    print("="*60)
    
    # 加载数据
    with open('data/processed/temporal_split.pkl', 'rb') as f:
        data = pickle.load(f)
    
    train_sequences = data['train_sequences']
    train_labels = data['train_labels']
    test_sequences = data['test_sequences']
    test_labels = data['test_labels']
    
    print(f"训练集: {len(train_sequences)} 样本, 类别: {sorted(set(train_labels))}")
    print(f"测试集: {len(test_sequences)} 样本, 类别: {sorted(set(test_labels))}")
    
    # 重新映射标签到连续的0-N
    unique_test_labels = sorted(set(test_labels))
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_test_labels)}
    
    mapped_train_labels = [label_map.get(label, -1) for label in train_labels]
    mapped_test_labels = [label_map[label] for label in test_labels]
    
    # 只保留训练集中存在的类别
    valid_train_mask = np.array(mapped_train_labels) >= 0
    train_sequences = train_sequences[valid_train_mask]
    mapped_train_labels = np.array(mapped_train_labels)[valid_train_mask]
    
    num_classes = len(unique_test_labels)
    print(f"有效类别数: {num_classes}")
    
    # 测试不同模型
    models = [
        ("2D CNN", Temporal2DCNN((5, 20, 60), num_classes)),
        ("简单CNN", SimpleCNN((20, 60), num_classes))
    ]
    
    results = {}
    
    for model_name, model in models:
        print(f"\n测试 {model_name}:")
        try:
            accuracy, predictions = quick_train_and_test(
                model, train_sequences, mapped_train_labels, 
                test_sequences, mapped_test_labels, epochs=20
            )
            results[model_name] = accuracy
            print(f"  准确率: {accuracy:.4f}")
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            results[model_name] = 0.0
    
    return results

def experiment_2_few_shot():
    """实验2: 少样本学习"""
    
    print("\n" + "="*60)
    print("实验2: 少样本学习")
    print("="*60)
    
    # 加载数据
    with open('data/processed/few_shot_splits.pkl', 'rb') as f:
        few_shot_data = pickle.load(f)
    
    results = {}
    
    for k_shot in ['1_shot', '3_shot', '5_shot']:
        print(f"\n{k_shot} 学习:")
        
        train_sequences = few_shot_data[k_shot]['train_sequences']
        train_labels = few_shot_data[k_shot]['train_labels']
        test_sequences = few_shot_data[k_shot]['test_sequences']
        test_labels = few_shot_data[k_shot]['test_labels']
        
        print(f"  训练集: {len(train_sequences)} 样本")
        print(f"  测试集: {len(test_sequences)} 样本")
        
        # 使用简单模型
        model = SimpleCNN((20, 60), 20)
        
        try:
            accuracy, _ = quick_train_and_test(
                model, train_sequences, train_labels,
                test_sequences, test_labels, epochs=50
            )
            results[k_shot] = accuracy
            print(f"  准确率: {accuracy:.4f}")
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            results[k_shot] = 0.0
    
    return results

def experiment_3_noise_robustness():
    """实验3: 噪声鲁棒性"""
    
    print("\n" + "="*60)
    print("实验3: 噪声鲁棒性测试")
    print("="*60)
    
    # 加载原始数据用于训练
    data_file = Path("data/processed/temporal_sequences_len5.pkl")
    with open(data_file, 'rb') as f:
        original_data = pickle.load(f)
    
    sequences = original_data['sequences']
    labels = original_data['labels']
    
    # 数据划分
    n_train = int(0.8 * len(sequences))
    train_sequences = sequences[:n_train]
    train_labels = labels[:n_train]
    
    # 训练一个基础模型
    print("训练基础模型...")
    base_model = SimpleCNN((20, 60), 20)
    base_accuracy, _ = quick_train_and_test(
        base_model, train_sequences, train_labels,
        sequences[n_train:], labels[n_train:], epochs=30
    )
    print(f"基础模型准确率: {base_accuracy:.4f}")
    
    # 加载噪声数据
    with open('data/processed/noise_robustness.pkl', 'rb') as f:
        noisy_data = pickle.load(f)
    
    results = {'clean': base_accuracy}
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = base_model.to(device)
    base_model.eval()
    
    for noise_level in noise_levels:
        print(f"\n噪声水平: {noise_level}")
        
        noisy_sequences = noisy_data[f'noise_{noise_level}']['sequences']
        test_noisy = noisy_sequences[n_train:]
        test_labels = labels[n_train:]
        
        # 测试噪声数据
        test_tensor = torch.FloatTensor(test_noisy).to(device)
        
        with torch.no_grad():
            outputs = base_model(test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = accuracy_score(test_labels, predicted.cpu().numpy())
        
        results[f'noise_{noise_level}'] = accuracy
        print(f"  准确率: {accuracy:.4f}")
    
    return results

def experiment_4_ablation():
    """实验4: 消融研究"""
    
    print("\n" + "="*60)
    print("实验4: 消融研究")
    print("="*60)
    
    # 加载消融数据
    with open('data/processed/ablation_study.pkl', 'rb') as f:
        ablation_data = pickle.load(f)
    
    results = {}
    
    configs = [
        ('single_frame', '单帧特征'),
        ('seq_len_2', '时序长度2'),
        ('seq_len_3', '时序长度3'),
        ('seq_len_4', '时序长度4'),
        ('seq_len_5', '时序长度5 (完整)'),
    ]
    
    for config_name, config_desc in configs:
        print(f"\n{config_desc}:")
        
        sequences = ablation_data[config_name]['sequences']
        labels = ablation_data[config_name]['labels']
        
        print(f"  数据形状: {sequences.shape}")
        
        # 数据划分
        n_train = int(0.8 * len(sequences))
        train_sequences = sequences[:n_train]
        train_labels = labels[:n_train]
        test_sequences = sequences[n_train:]
        test_labels = labels[n_train:]
        
        # 选择合适的模型
        if len(sequences.shape) == 3:  # 单帧
            model = SimpleCNN((sequences.shape[1], sequences.shape[2]), 20)
        else:  # 多帧
            model = SimpleCNN((sequences.shape[2], sequences.shape[3]), 20)
        
        try:
            accuracy, _ = quick_train_and_test(
                model, train_sequences, train_labels,
                test_sequences, test_labels, epochs=30
            )
            results[config_name] = accuracy
            print(f"  准确率: {accuracy:.4f}")
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            results[config_name] = 0.0
    
    return results

def generate_paper_plots(all_results):
    """生成论文图表"""
    
    print("\n" + "="*60)
    print("生成论文图表")
    print("="*60)
    
    # 确保结果目录存在
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    
    # 1. 少样本学习结果
    if 'few_shot' in all_results:
        plt.figure(figsize=(10, 6))
        few_shot_results = all_results['few_shot']
        
        shots = [1, 3, 5]
        accuracies = [few_shot_results.get(f'{k}_shot', 0) for k in shots]
        
        plt.plot(shots, accuracies, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Training Samples per Class')
        plt.ylabel('Accuracy')
        plt.title('Few-Shot Learning Performance')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        for i, (shot, acc) in enumerate(zip(shots, accuracies)):
            plt.annotate(f'{acc:.3f}', (shot, acc), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig('results/figures/few_shot_learning.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 少样本学习图表已保存")
    
    # 2. 噪声鲁棒性结果
    if 'noise_robustness' in all_results:
        plt.figure(figsize=(10, 6))
        noise_results = all_results['noise_robustness']
        
        noise_levels = [0, 0.01, 0.05, 0.1, 0.2, 0.3]
        accuracies = [noise_results.get('clean' if level == 0 else f'noise_{level}', 0) 
                     for level in noise_levels]
        
        plt.plot(noise_levels, accuracies, 'ro-', linewidth=2, markersize=8)
        plt.xlabel('Noise Level (σ)')
        plt.ylabel('Accuracy')
        plt.title('Noise Robustness Analysis')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('results/figures/noise_robustness.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 噪声鲁棒性图表已保存")
    
    # 3. 消融研究结果
    if 'ablation' in all_results:
        plt.figure(figsize=(12, 6))
        ablation_results = all_results['ablation']
        
        configs = ['single_frame', 'seq_len_2', 'seq_len_3', 'seq_len_4', 'seq_len_5']
        labels = ['Single Frame', 'Seq Len 2', 'Seq Len 3', 'Seq Len 4', 'Seq Len 5']
        accuracies = [ablation_results.get(config, 0) for config in configs]
        
        bars = plt.bar(labels, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
        plt.xlabel('Configuration')
        plt.ylabel('Accuracy')
        plt.title('Ablation Study: Temporal Sequence Length')
        plt.ylim(0, 1)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/figures/ablation_study.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 消融研究图表已保存")

def main():
    """主函数"""
    
    print("开始论文实验...")
    
    # 运行所有实验
    results = {}
    
    try:
        results['temporal_split'] = experiment_1_temporal_split()
    except Exception as e:
        print(f"时间分离实验失败: {e}")
        results['temporal_split'] = {}
    
    try:
        results['few_shot'] = experiment_2_few_shot()
    except Exception as e:
        print(f"少样本学习实验失败: {e}")
        results['few_shot'] = {}
    
    try:
        results['noise_robustness'] = experiment_3_noise_robustness()
    except Exception as e:
        print(f"噪声鲁棒性实验失败: {e}")
        results['noise_robustness'] = {}
    
    try:
        results['ablation'] = experiment_4_ablation()
    except Exception as e:
        print(f"消融研究实验失败: {e}")
        results['ablation'] = {}
    
    # 保存结果
    with open('results/paper_experiments_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # 生成图表
    generate_paper_plots(results)
    
    # 打印总结
    print("\n" + "="*60)
    print("实验结果总结")
    print("="*60)
    
    for exp_name, exp_results in results.items():
        print(f"\n{exp_name}:")
        for key, value in exp_results.items():
            print(f"  {key}: {value:.4f}")
    
    print(f"\n✅ 所有实验完成！结果已保存到 results/paper_experiments_results.pkl")
    print(f"📊 图表已保存到 results/figures/ 目录")

if __name__ == '__main__':
    main()
