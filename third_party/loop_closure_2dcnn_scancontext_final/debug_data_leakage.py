#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试数据泄露问题
"""

import numpy as np
import torch
from models.cnn_2d_models import Simple2DCNN
from utils.scan_context import ScanContext
from utils.ply_reader import PLYReader
import glob
from pathlib import Path

def test_data_leakage():
    """测试是否存在数据泄露"""
    
    print("🔍 数据泄露调试测试")
    print("="*50)
    
    # 1. 检查标签生成逻辑
    print("1. 标签生成逻辑检查:")
    total_files = 1769
    num_classes = 20
    
    print("训练标签生成:")
    for i in [0, 88, 177, 354, 531, 708, 885, 1062, 1239, 1416, 1593, 1769-1]:
        progress = int((i / total_files) * num_classes)
        progress = min(progress, num_classes - 1)
        print(f"  文件索引 {i:4d} -> 标签 {progress:2d}")
    
    print("\n测试标签生成:")
    for total_messages in [1, 89, 178, 355, 532, 709, 886, 1063, 1240, 1417, 1594, 1769]:
        expected_segment = int((total_messages - 1) / (1769 / 20))
        expected_segment = min(expected_segment, 19)
        print(f"  消息索引 {total_messages:4d} -> 期望段 {expected_segment:2d}")
    
    # 2. 测试随机输入
    print("\n2. 随机输入测试:")
    
    # 加载模型
    model_path = "models/saved/simple2dcnn_trajectory_avg99.5.pth"
    if not Path(model_path).exists():
        print("❌ 模型文件不存在")
        return
    
    model = Simple2DCNN(num_classes=20)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("测试随机输入的预测结果:")
    with torch.no_grad():
        for i in range(10):
            # 生成随机输入
            random_input = torch.randn(1, 1, 20, 60)
            output = model(random_input)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            print(f"  随机输入 {i+1}: 预测段 {predicted.item():2d}, 置信度 {confidence.item():.4f}")
    
    # 3. 测试固定模式输入
    print("\n3. 固定模式输入测试:")
    
    patterns = {
        "全零": torch.zeros(1, 1, 20, 60),
        "全一": torch.ones(1, 1, 20, 60),
        "对角线": torch.zeros(1, 1, 20, 60),
        "中心点": torch.zeros(1, 1, 20, 60),
        "边缘": torch.zeros(1, 1, 20, 60)
    }
    
    # 创建特定模式
    patterns["对角线"][0, 0, range(20), range(0, 60, 3)] = 1.0
    patterns["中心点"][0, 0, 10, 30] = 1.0
    patterns["边缘"][0, 0, [0, 19], :] = 1.0
    patterns["边缘"][0, 0, :, [0, 59]] = 1.0
    
    with torch.no_grad():
        for name, pattern in patterns.items():
            output = model(pattern)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            print(f"  {name:6s}: 预测段 {predicted.item():2d}, 置信度 {confidence.item():.4f}")
    
    # 4. 检查ScanContext是否包含时序信息
    print("\n4. ScanContext时序信息检查:")
    
    data_dir = "/mysda/shared_dir/2025.7.3/2025-07-03-16-28-57.ply"
    ply_files = sorted(glob.glob(f"{data_dir}/*.ply"))
    
    if len(ply_files) > 0:
        sc_generator = ScanContext()
        
        # 检查前几个和后几个文件的ScanContext
        test_indices = [0, 1, 2, len(ply_files)//2-1, len(ply_files)//2, len(ply_files)//2+1, -3, -2, -1]
        
        print("文件ScanContext统计:")
        for idx in test_indices:
            if 0 <= idx < len(ply_files) or idx < 0:
                try:
                    ply_file = ply_files[idx]
                    points = PLYReader.read_ply_file(ply_file)
                    if points is not None:
                        points = points[:, :3]
                        sc = sc_generator.generate_scan_context(points)
                        
                        if sc is not None:
                            # 计算一些统计量
                            mean_val = np.mean(sc)
                            std_val = np.std(sc)
                            max_val = np.max(sc)
                            nonzero_count = np.count_nonzero(sc)
                            
                            print(f"  文件 {idx:4d}: 均值={mean_val:.4f}, 标准差={std_val:.4f}, 最大值={max_val:.4f}, 非零={nonzero_count}")
                        
                except Exception as e:
                    print(f"  文件 {idx:4d}: 处理失败 - {e}")
    
    print("\n🔍 调试完成")

def test_shuffled_prediction():
    """测试打乱顺序后的预测"""
    print("\n5. 打乱顺序测试:")
    
    # 加载模型
    model_path = "models/saved/simple2dcnn_trajectory_avg99.5.pth"
    if not Path(model_path).exists():
        print("❌ 模型文件不存在")
        return
    
    model = Simple2DCNN(num_classes=20)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 生成一些测试数据
    data_dir = "/mysda/shared_dir/2025.7.3/2025-07-03-16-28-57.ply"
    ply_files = sorted(glob.glob(f"{data_dir}/*.ply"))
    
    if len(ply_files) < 100:
        print("❌ 文件数量不足")
        return
    
    sc_generator = ScanContext()
    
    # 选择一些文件进行测试
    test_files = ply_files[::100]  # 每100个文件选一个
    
    print("原始顺序预测:")
    original_predictions = []
    
    for i, ply_file in enumerate(test_files):
        try:
            points = PLYReader.read_ply_file(ply_file)
            if points is not None:
                points = points[:, :3]
                sc = sc_generator.generate_scan_context(points)
                
                if sc is not None:
                    sc_tensor = torch.FloatTensor(sc).unsqueeze(0).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = model(sc_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        
                        original_predictions.append(predicted.item())
                        print(f"  文件 {i*100:4d}: 预测段 {predicted.item():2d}, 置信度 {confidence.item():.4f}")
        except Exception as e:
            print(f"  文件 {i*100:4d}: 处理失败 - {e}")
    
    print(f"\n原始顺序预测结果: {original_predictions}")
    
    # 打乱顺序测试
    print("\n打乱顺序预测:")
    import random
    shuffled_files = test_files.copy()
    random.shuffle(shuffled_files)
    
    shuffled_predictions = []
    
    for i, ply_file in enumerate(shuffled_files):
        try:
            points = PLYReader.read_ply_file(ply_file)
            if points is not None:
                points = points[:, :3]
                sc = sc_generator.generate_scan_context(points)
                
                if sc is not None:
                    sc_tensor = torch.FloatTensor(sc).unsqueeze(0).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = model(sc_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        
                        shuffled_predictions.append(predicted.item())
                        print(f"  打乱 {i:4d}: 预测段 {predicted.item():2d}, 置信度 {confidence.item():.4f}")
        except Exception as e:
            print(f"  打乱 {i:4d}: 处理失败 - {e}")
    
    print(f"\n打乱顺序预测结果: {shuffled_predictions}")
    
    # 比较结果
    if len(original_predictions) == len(shuffled_predictions):
        correlation = np.corrcoef(original_predictions, shuffled_predictions)[0, 1]
        print(f"\n预测结果相关性: {correlation:.4f}")
        
        if correlation > 0.8:
            print("⚠️  高相关性表明模型可能学到了与顺序无关的真实特征")
        else:
            print("🚨 低相关性表明模型严重依赖文件顺序！")

if __name__ == '__main__':
    test_data_leakage()
    test_shuffled_prediction()
