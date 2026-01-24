#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统测试脚本

用于验证时序回环检测系统的各个组件是否正常工作
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_models():
    """测试模型是否能正常创建和前向传播"""
    print("测试模型...")
    
    # 测试参数
    batch_size = 2
    sequence_length = 5
    num_rings = 20
    num_sectors = 60
    num_classes = 20
    
    # 创建测试输入
    test_input = torch.randn(batch_size, sequence_length, num_rings, num_sectors)
    
    # 测试2D CNN模型
    try:
        from models.temporal_2d_cnn import Temporal2DCNN, Temporal2DCNNLite, Temporal2DCNNResNet
        
        print("  测试 Temporal2DCNN...")
        model_2d = Temporal2DCNN(sequence_length, num_rings, num_sectors, num_classes)
        output_2d = model_2d(test_input)
        assert output_2d.shape == (batch_size, num_classes)
        print("    ✓ Temporal2DCNN 测试通过")
        
        print("  测试 Temporal2DCNNLite...")
        model_2d_lite = Temporal2DCNNLite(sequence_length, num_rings, num_sectors, num_classes)
        output_2d_lite = model_2d_lite(test_input)
        assert output_2d_lite.shape == (batch_size, num_classes)
        print("    ✓ Temporal2DCNNLite 测试通过")
        
    except Exception as e:
        print(f"    ✗ 2D CNN模型测试失败: {e}")
    
    # 测试3D CNN模型
    try:
        from models.temporal_3d_cnn import Temporal3DCNN, Temporal3DCNNLite, Temporal3DCNNDeep
        
        print("  测试 Temporal3DCNN...")
        model_3d = Temporal3DCNN(sequence_length, num_rings, num_sectors, num_classes)
        output_3d = model_3d(test_input)
        assert output_3d.shape == (batch_size, num_classes)
        print("    ✓ Temporal3DCNN 测试通过")
        
        print("  测试 Temporal3DCNNLite...")
        model_3d_lite = Temporal3DCNNLite(sequence_length, num_rings, num_sectors, num_classes)
        output_3d_lite = model_3d_lite(test_input)
        assert output_3d_lite.shape == (batch_size, num_classes)
        print("    ✓ Temporal3DCNNLite 测试通过")
        
    except Exception as e:
        print(f"    ✗ 3D CNN模型测试失败: {e}")


def test_scan_context():
    """测试ScanContext生成器"""
    print("测试 ScanContext 生成器...")
    
    try:
        from utils.scan_context import ScanContext
        
        # 创建ScanContext生成器
        sc_generator = ScanContext(num_sectors=60, num_rings=20)
        
        # 创建测试点云数据
        num_points = 1000
        test_points = np.random.randn(num_points, 3) * 10  # 随机点云
        
        # 生成ScanContext
        scan_context = sc_generator.make_scan_context(test_points)
        
        # 验证输出形状
        assert scan_context.shape == (20, 60)
        print("    ✓ ScanContext 生成器测试通过")
        
        # 测试Ring Key生成
        ring_key = sc_generator.make_ring_key(scan_context)
        assert ring_key.shape == (20,)
        print("    ✓ Ring Key 生成测试通过")
        
        # 测试Sector Key生成
        sector_key = sc_generator.make_sector_key(scan_context)
        assert sector_key.shape == (60,)
        print("    ✓ Sector Key 生成测试通过")
        
    except Exception as e:
        print(f"    ✗ ScanContext 测试失败: {e}")


def test_temporal_dataset():
    """测试时序数据集（使用模拟数据）"""
    print("测试时序数据集...")
    
    try:
        # 创建临时测试数据
        test_data_dir = Path("test_data")
        test_data_dir.mkdir(exist_ok=True)
        
        # 创建模拟的ScanContext数据
        num_samples = 100
        sequence_length = 5
        num_rings = 20
        num_sectors = 60
        
        sequences = []
        labels = []
        
        for i in range(num_samples):
            # 创建随机时序序列
            sequence = np.random.rand(sequence_length, num_rings, num_sectors)
            sequences.append(sequence)
            labels.append(i % 20)  # 20个类别
        
        # 保存测试数据
        import pickle
        test_cache_file = test_data_dir / "test_temporal_data.pkl"
        with open(test_cache_file, 'wb') as f:
            pickle.dump({
                'sequences': sequences,
                'labels': labels,
                'file_paths': [f"test_file_{i}.ply" for i in range(num_samples)]
            }, f)
        
        # 测试数据集加载
        from utils.temporal_dataset import TemporalScanContextDataset
        
        # 创建一个简化的数据集类用于测试
        class TestTemporalDataset(TemporalScanContextDataset):
            def _load_data(self):
                with open(test_cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.sequences = data['sequences']
                    self.labels = data['labels']
                    self.file_paths = data['file_paths']
        
        dataset = TestTemporalDataset(
            data_dir=test_data_dir,
            split='train',
            sequence_length=sequence_length
        )
        
        # 测试数据集功能
        assert len(dataset) == num_samples
        
        sample_data, sample_label = dataset[0]
        assert sample_data.shape == (sequence_length, num_rings, num_sectors)
        assert isinstance(sample_label, torch.Tensor)
        
        print("    ✓ 时序数据集测试通过")
        
        # 清理测试数据
        import shutil
        shutil.rmtree(test_data_dir)
        
    except Exception as e:
        print(f"    ✗ 时序数据集测试失败: {e}")


def test_training_components():
    """测试训练组件"""
    print("测试训练组件...")
    
    try:
        from scripts.training.train_temporal_models import create_default_config
        
        # 测试配置创建
        config = create_default_config('temporal_3d_cnn', 5)
        assert config['model']['type'] == 'temporal_3d_cnn'
        assert config['model']['params']['sequence_length'] == 5
        print("    ✓ 配置创建测试通过")
        
        # 测试优化器和损失函数
        import torch.nn as nn
        import torch.optim as optim
        
        # 创建简单模型用于测试
        from models.temporal_3d_cnn import Temporal3DCNNLite
        model = Temporal3DCNNLite(sequence_length=5, num_classes=20)
        
        # 测试优化器
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 测试前向传播和反向传播
        test_input = torch.randn(2, 5, 20, 60)
        test_target = torch.randint(0, 20, (2,))
        
        optimizer.zero_grad()
        output = model(test_input)
        loss = criterion(output, test_target)
        loss.backward()
        optimizer.step()
        
        print("    ✓ 训练组件测试通过")
        
    except Exception as e:
        print(f"    ✗ 训练组件测试失败: {e}")


def test_evaluation_components():
    """测试评估组件"""
    print("测试评估组件...")
    
    try:
        from sklearn.metrics import accuracy_score, top_k_accuracy_score
        import numpy as np
        
        # 创建模拟预测结果
        num_samples = 100
        num_classes = 20
        
        true_labels = np.random.randint(0, num_classes, num_samples)
        pred_probs = np.random.rand(num_samples, num_classes)
        pred_labels = np.argmax(pred_probs, axis=1)
        
        # 测试评估指标计算
        accuracy = accuracy_score(true_labels, pred_labels)
        top5_acc = top_k_accuracy_score(true_labels, pred_probs, k=5)
        
        assert 0 <= accuracy <= 1
        assert 0 <= top5_acc <= 1
        
        print("    ✓ 评估组件测试通过")
        
    except Exception as e:
        print(f"    ✗ 评估组件测试失败: {e}")


def test_system_integration():
    """测试系统集成"""
    print("测试系统集成...")
    
    try:
        # 测试模型导入
        from models import Temporal2DCNN, Temporal3DCNN
        
        # 创建小型模型进行端到端测试
        model_2d = Temporal2DCNN(sequence_length=3, num_rings=10, num_sectors=20, num_classes=5)
        model_3d = Temporal3DCNN(sequence_length=3, num_rings=10, num_sectors=20, num_classes=5)
        
        # 创建测试数据
        test_input = torch.randn(1, 3, 10, 20)
        
        # 测试推理
        with torch.no_grad():
            output_2d = model_2d(test_input)
            output_3d = model_3d(test_input)
        
        assert output_2d.shape == (1, 5)
        assert output_3d.shape == (1, 5)
        
        print("    ✓ 系统集成测试通过")
        
    except Exception as e:
        print(f"    ✗ 系统集成测试失败: {e}")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("时序回环检测系统测试")
    print("=" * 60)
    
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print()
    
    # 运行各项测试
    test_models()
    print()
    
    test_scan_context()
    print()
    
    test_temporal_dataset()
    print()
    
    test_training_components()
    print()
    
    test_evaluation_components()
    print()
    
    test_system_integration()
    print()
    
    print("=" * 60)
    print("测试完成！")
    print("=" * 60)
    
    print("\n如果所有测试都通过，您可以开始使用系统：")
    print("1. 运行数据预处理: python demo_temporal_system.py --mode preprocess --data_dir data")
    print("2. 训练模型: python demo_temporal_system.py --mode train --model temporal_3d_cnn")
    print("3. 运行完整演示: python demo_temporal_system.py --mode full --data_dir data")


if __name__ == '__main__':
    main()
