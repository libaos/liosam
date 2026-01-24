#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行挑战性实验，生成论文所需的结果
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
from models.temporal_models import *
import warnings
warnings.filterwarnings('ignore')

def load_model_and_test(model_class, model_name, test_data, test_labels, input_shape):
    """加载训练好的模型并测试"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    if len(input_shape) == 4:  # 3D CNN
        model = model_class(input_shape=input_shape, num_classes=20)
    else:  # 2D CNN
        model = model_class(input_shape=input_shape[1:], num_classes=20)
    
    model = model.to(device)
    
    # 尝试加载预训练模型
    model_path = f"models/saved/{model_name}_best.pth"
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 加载预训练模型: {model_path}")
    else:
        print(f"⚠️  未找到预训练模型: {model_path}，使用随机初始化")
    
    model.eval()
    
    # 准备测试数据
    test_tensor = torch.FloatTensor(test_data).to(device)
    test_labels_tensor = torch.LongTensor(test_labels).to(device)
    
    test_dataset = TensorDataset(test_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 测试
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_preds, all_labels

def run_temporal_split_experiment():
    """运行时间分离实验"""
    
    print("\n" + "="*60)
    print("1. 时间分离实验 - 避免时序数据泄露")
    print("="*60)
    
    # 加载时间分离的数据
    with open('data/processed/temporal_split.pkl', 'rb') as f:
        data = pickle.load(f)
    
    test_sequences = data['test_sequences']
    test_labels = data['test_labels']
    
    print(f"测试集: {len(test_sequences)} 个样本")
    print(f"测试类别: {sorted(set(test_labels))}")
    
    # 测试不同模型
    models_to_test = [
        (Temporal3DCNN, "temporal_3d_cnn", (1, 5, 20, 60)),
        (Temporal3DCNNDeep, "temporal_3d_cnn_deep", (1, 5, 20, 60)),
        (Temporal2DCNN, "temporal_2d_cnn", (5, 20, 60)),
        (ResNet2DCNN, "resnet_2d_cnn", (5, 20, 60))
    ]
    
    results = {}
    
    for model_class, model_name, input_shape in models_to_test:
        print(f"\n测试模型: {model_name}")
        try:
            accuracy, preds, labels = load_model_and_test(
                model_class, model_name, test_sequences, test_labels, input_shape
            )
            results[model_name] = {
                'accuracy': accuracy,
                'predictions': preds,
                'labels': labels
            }
            print(f"  时间分离测试准确率: {accuracy:.4f}")
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            results[model_name] = {'accuracy': 0.0}
    
    return results

def run_few_shot_experiment():
    """运行少样本学习实验"""
    
    print("\n" + "="*60)
    print("2. 少样本学习实验")
    print("="*60)
    
    # 加载少样本数据
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
        
        # 快速训练一个简单模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 使用简单的2D CNN
        model = Temporal2DCNN(input_shape=(20, 60), num_classes=20).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 准备数据
        train_tensor = torch.FloatTensor(train_sequences).to(device)
        train_labels_tensor = torch.LongTensor(train_labels).to(device)
        test_tensor = torch.FloatTensor(test_sequences).to(device)
        test_labels_tensor = torch.LongTensor(test_labels).to(device)
        
        train_dataset = TensorDataset(train_tensor, train_labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=min(16, len(train_sequences)), shuffle=True)
        
        # 快速训练
        model.train()
        for epoch in range(50):  # 少量epoch
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
        
        results[k_shot] = accuracy
        print(f"  准确率: {accuracy:.4f}")
    
    return results

def run_noise_robustness_experiment():
    """运行噪声鲁棒性实验"""
    
    print("\n" + "="*60)
    print("3. 噪声鲁棒性实验")
    print("="*60)
    
    # 加载噪声数据
    with open('data/processed/noise_robustness.pkl', 'rb') as f:
        noisy_data = pickle.load(f)
    
    # 使用最好的预训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Temporal2DCNN(input_shape=(20, 60), num_classes=20).to(device)
    
    # 尝试加载预训练模型
    model_path = "models/saved/temporal_2d_cnn_best.pth"
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ 加载预训练模型")
    else:
        print("⚠️  使用随机初始化模型")
    
    model.eval()
    
    results = {}
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
    
    for noise_level in noise_levels:
        print(f"\n噪声水平: {noise_level}")
        
        sequences = noisy_data[f'noise_{noise_level}']['sequences']
        labels = noisy_data[f'noise_{noise_level}']['labels']
        
        # 测试
        test_tensor = torch.FloatTensor(sequences).to(device)
        test_labels_tensor = torch.LongTensor(labels).to(device)
        
        test_dataset = TensorDataset(test_tensor, test_labels_tensor)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                outputs = model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        results[f'noise_{noise_level}'] = accuracy
        print(f"  准确率: {accuracy:.4f}")
    
    return results

def run_ablation_study():
    """运行消融研究"""
    
    print("\n" + "="*60)
    print("4. 消融研究")
    print("="*60)
    
    # 加载消融数据
    with open('data/processed/ablation_study.pkl', 'rb') as f:
        ablation_data = pickle.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    # 测试不同配置
    configs = [
        ('single_frame', '单帧特征'),
        ('seq_len_2', '时序长度2'),
        ('seq_len_3', '时序长度3'), 
        ('seq_len_4', '时序长度4'),
        ('seq_len_5', '时序长度5'),
        ('low_res_rings', '低分辨率环数'),
        ('low_res_sectors', '低分辨率扇区')
    ]
    
    for config_name, config_desc in configs:
        print(f"\n{config_desc}:")
        
        sequences = ablation_data[config_name]['sequences']
        labels = ablation_data[config_name]['labels']
        
        print(f"  数据形状: {sequences.shape}")
        
        # 根据数据形状选择合适的模型
        if len(sequences.shape) == 3:  # 单帧数据
            input_shape = sequences.shape[1:]
            model = SimpleCNN(input_shape=input_shape, num_classes=20).to(device)
        else:  # 多帧数据
            input_shape = sequences.shape[1:]
            model = Temporal2DCNN(input_shape=input_shape, num_classes=20).to(device)
        
        # 简单训练
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 数据划分
        n_train = int(0.8 * len(sequences))
        train_sequences = sequences[:n_train]
        train_labels = labels[:n_train]
        test_sequences = sequences[n_train:]
        test_labels = labels[n_train:]
        
        # 训练
        train_tensor = torch.FloatTensor(train_sequences).to(device)
        train_labels_tensor = torch.LongTensor(train_labels).to(device)
        
        train_dataset = TensorDataset(train_tensor, train_labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        model.train()
        for epoch in range(30):
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
        
        # 测试
        model.eval()
        test_tensor = torch.FloatTensor(test_sequences).to(device)
        
        with torch.no_grad():
            test_outputs = model(test_tensor)
            _, predicted = torch.max(test_outputs.data, 1)
            accuracy = accuracy_score(test_labels, predicted.cpu().numpy())
        
        results[config_name] = accuracy
        print(f"  准确率: {accuracy:.4f}")
    
    return results

def generate_paper_results():
    """生成论文所需的结果图表"""
    
    print("\n" + "="*60)
    print("生成论文结果图表")
    print("="*60)
    
    # 运行所有实验
    temporal_results = run_temporal_split_experiment()
    few_shot_results = run_few_shot_experiment()
    noise_results = run_noise_robustness_experiment()
    ablation_results = run_ablation_study()
    
    # 保存结果
    all_results = {
        'temporal_split': temporal_results,
        'few_shot': few_shot_results,
        'noise_robustness': noise_results,
        'ablation_study': ablation_results
    }
    
    with open('results/challenging_experiments_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    print("\n✅ 所有实验完成！结果已保存到 results/challenging_experiments_results.pkl")
    
    # 生成总结报告
    print("\n" + "="*60)
    print("实验结果总结")
    print("="*60)
    
    print("\n1. 时间分离实验结果:")
    for model_name, result in temporal_results.items():
        if isinstance(result, dict) and 'accuracy' in result:
            print(f"  {model_name}: {result['accuracy']:.4f}")
    
    print("\n2. 少样本学习结果:")
    for k_shot, accuracy in few_shot_results.items():
        print(f"  {k_shot}: {accuracy:.4f}")
    
    print("\n3. 噪声鲁棒性结果:")
    for noise_level, accuracy in noise_results.items():
        print(f"  {noise_level}: {accuracy:.4f}")
    
    print("\n4. 消融研究结果:")
    for config, accuracy in ablation_results.items():
        print(f"  {config}: {accuracy:.4f}")

if __name__ == '__main__':
    # 确保结果目录存在
    Path('results').mkdir(exist_ok=True)
    
    print("开始运行挑战性实验...")
    generate_paper_results()
