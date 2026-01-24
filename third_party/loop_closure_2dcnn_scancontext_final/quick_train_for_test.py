#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速训练一个模型用于测试
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from pathlib import Path
from models.temporal_models import Temporal2DCNN
from sklearn.model_selection import train_test_split

def quick_train():
    """快速训练模型"""
    
    print("快速训练模型用于测试...")
    
    # 加载数据
    data_file = Path("data/processed/temporal_sequences_len5.pkl")
    if not data_file.exists():
        print("未找到训练数据")
        return
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    sequences = np.array(data['sequences'])
    labels = np.array(data['labels'])
    
    print(f"数据形状: {sequences.shape}")
    print(f"标签数量: {len(labels)}")
    
    # 数据划分
    train_sequences, test_sequences, train_labels, test_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"训练集: {len(train_sequences)} 样本")
    print(f"测试集: {len(test_sequences)} 样本")
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Temporal2DCNN(input_shape=(5, 20, 60), num_classes=20)
    model = model.to(device)
    
    # 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 准备数据
    train_tensor = torch.FloatTensor(train_sequences).to(device)
    train_labels_tensor = torch.LongTensor(train_labels).to(device)
    test_tensor = torch.FloatTensor(test_sequences).to(device)
    test_labels_tensor = torch.LongTensor(test_labels).to(device)
    
    train_dataset = TensorDataset(train_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 快速训练
    print("开始训练...")
    model.train()
    
    for epoch in range(20):  # 只训练20个epoch
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        train_acc = 100 * correct / total
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_tensor)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_acc = 100 * (test_predicted == test_labels_tensor).sum().item() / len(test_labels_tensor)
        
        model.train()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d}: Loss: {total_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    # 保存模型
    Path("models/saved").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "models/saved/quick_trained_model.pth")
    print("✅ 模型已保存到 models/saved/quick_trained_model.pth")
    
    return model

if __name__ == '__main__':
    quick_train()
