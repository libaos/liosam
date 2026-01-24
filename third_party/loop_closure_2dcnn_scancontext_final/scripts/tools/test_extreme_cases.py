#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import SCRingCNN, SCRingCNNLite
from models.circular_conv import CircularPadConv2d, CircularConvDataAugmentation
from utils import ScanContext

class ExtremeTest:
    """
    SC-RingCNN极端情况测试类
    
    用于测试SC-RingCNN模型在各种极端条件下的表现，包括：
    1. 极端输入值（NaN、Inf）
    2. 梯度爆炸/消失
    3. 极大批次大小
    4. 极端形状变化
    5. 随机权重扰动
    """
    
    def __init__(self, output_dir="extreme_results"):
        """
        初始化极端测试
        
        参数:
            output_dir (str): 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
    
    def test_nan_inf_inputs(self):
        """
        测试NaN和Inf输入
        """
        print("\n===== 测试NaN和Inf输入 =====")
        
        # 设置参数
        num_rings = 20
        num_sectors = 60
        descriptor_dim = 256
        
        # 创建模型
        model = SCRingCNN(
            in_channels=1,
            num_rings=num_rings,
            num_sectors=num_sectors,
            descriptor_dim=descriptor_dim
        ).to(self.device)
        
        # 测试用例
        test_cases = {
            "部分NaN": torch.tensor(np.random.randn(1, 1, num_rings, num_sectors), dtype=torch.float32),
            "部分Inf": torch.tensor(np.random.randn(1, 1, num_rings, num_sectors), dtype=torch.float32),
            "全NaN": torch.tensor(np.ones((1, 1, num_rings, num_sectors)) * np.nan, dtype=torch.float32),
            "全Inf": torch.tensor(np.ones((1, 1, num_rings, num_sectors)) * np.inf, dtype=torch.float32),
            "混合NaN和Inf": torch.tensor(np.random.randn(1, 1, num_rings, num_sectors), dtype=torch.float32)
        }
        
        # 添加NaN和Inf
        test_cases["部分NaN"][:, :, :5, :5] = float('nan')
        test_cases["部分Inf"][:, :, -5:, -5:] = float('inf')
        test_cases["混合NaN和Inf"][:, :, :10, :10] = float('nan')
        test_cases["混合NaN和Inf"][:, :, -10:, -10:] = float('inf')
        
        results = {}
        
        for name, x in test_cases.items():
            try:
                print(f"\n测试 {name}")
                x = x.to(self.device)
                
                # 前向传播
                with torch.no_grad():
                    output = model(x)
                
                # 检查输出
                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()
                
                results[name] = {
                    "状态": "成功" if not (has_nan or has_inf) else "异常值",
                    "输出包含NaN": has_nan,
                    "输出包含Inf": has_inf,
                    "输出范数": float(torch.norm(output).item()) if not (has_nan or has_inf) else "N/A"
                }
                
                print(f"输出包含NaN: {has_nan}")
                print(f"输出包含Inf: {has_inf}")
                if not (has_nan or has_inf):
                    print(f"输出范数: {torch.norm(output).item()}")
                
            except Exception as e:
                results[name] = {
                    "状态": "失败",
                    "错误": str(e)
                }
                print(f"测试失败: {e}")
        
        # 保存结果
        self._save_results("nan_inf_inputs", results)
        
        return results
    
    def test_gradient_stability(self):
        """
        测试梯度稳定性
        """
        print("\n===== 测试梯度稳定性 =====")
        
        # 设置参数
        num_rings = 20
        num_sectors = 60
        descriptor_dim = 256
        
        # 创建模型
        model = SCRingCNN(
            in_channels=1,
            num_rings=num_rings,
            num_sectors=num_sectors,
            descriptor_dim=descriptor_dim
        ).to(self.device)
        
        # 创建输入和目标
        x = torch.randn(2, 1, num_rings, num_sectors).to(self.device)
        target = torch.randn(2, descriptor_dim).to(self.device)
        target = torch.nn.functional.normalize(target, p=2, dim=1)
        
        # 测试不同的学习率
        learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        epochs = 100
        results = {}
        
        for lr in learning_rates:
            try:
                print(f"\n测试学习率: {lr}")
                
                # 重置模型
                model = SCRingCNN(
                    in_channels=1,
                    num_rings=num_rings,
                    num_sectors=num_sectors,
                    descriptor_dim=descriptor_dim
                ).to(self.device)
                
                # 创建优化器
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                
                # 训练
                losses = []
                grad_norms = []
                
                for epoch in range(epochs):
                    # 前向传播
                    output = model(x)
                    output = torch.nn.functional.normalize(output, p=2, dim=1)
                    
                    # 计算损失（余弦相似度损失）
                    loss = -torch.mean(torch.sum(output * target, dim=1))
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # 计算梯度范数
                    grad_norm = 0
                    for param in model.parameters():
                        if param.grad is not None:
                            grad_norm += torch.norm(param.grad).item() ** 2
                    grad_norm = grad_norm ** 0.5
                    
                    # 更新权重
                    optimizer.step()
                    
                    # 记录损失和梯度范数
                    losses.append(loss.item())
                    grad_norms.append(grad_norm)
                    
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Grad Norm = {grad_norm:.4f}")
                
                # 检查梯度是否稳定
                is_stable = True
                for grad_norm in grad_norms:
                    if np.isnan(grad_norm) or np.isinf(grad_norm) or grad_norm > 1000:
                        is_stable = False
                        break
                
                results[f"lr_{lr}"] = {
                    "状态": "稳定" if is_stable else "不稳定",
                    "最终损失": float(losses[-1]),
                    "最大梯度范数": float(max(grad_norms)),
                    "最小梯度范数": float(min(grad_norms)),
                    "平均梯度范数": float(np.mean(grad_norms))
                }
                
                # 可视化
                plt.figure(figsize=(12, 10))
                
                plt.subplot(2, 1, 1)
                plt.plot(losses)
                plt.title(f"损失曲线 (lr={lr})")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.grid(True)
                
                plt.subplot(2, 1, 2)
                plt.plot(grad_norms)
                plt.title(f"梯度范数 (lr={lr})")
                plt.xlabel("Epoch")
                plt.ylabel("Gradient Norm")
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"gradient_stability_lr_{lr}.png"))
                plt.close()
                
            except Exception as e:
                results[f"lr_{lr}"] = {
                    "状态": "失败",
                    "错误": str(e)
                }
                print(f"测试失败: {e}")
        
        # 保存结果
        self._save_results("gradient_stability", results)
        
        return results
    
    def test_large_batch_size(self):
        """
        测试大批次大小
        """
        print("\n===== 测试大批次大小 =====")
        
        # 设置参数
        num_rings = 20
        num_sectors = 60
        descriptor_dim = 256
        
        # 创建模型
        model = SCRingCNN(
            in_channels=1,
            num_rings=num_rings,
            num_sectors=num_sectors,
            descriptor_dim=descriptor_dim
        ).to(self.device)
        
        # 测试不同的批次大小
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        results = {}
        
        for batch_size in batch_sizes:
            try:
                print(f"\n测试批次大小: {batch_size}")
                
                # 创建输入
                x = torch.randn(batch_size, 1, num_rings, num_sectors).to(self.device)
                
                # 测量内存使用和推理时间
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                
                # 预热
                with torch.no_grad():
                    _ = model(x)
                
                # 测量推理时间
                start_time = time.time()
                with torch.no_grad():
                    output = model(x)
                inference_time = time.time() - start_time
                
                # 获取内存使用
                memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                
                results[f"batch_{batch_size}"] = {
                    "状态": "成功",
                    "推理时间(ms)": float(inference_time * 1000),
                    "每样本时间(ms)": float(inference_time * 1000 / batch_size),
                    "内存使用(MB)": float(memory_used),
                    "每样本内存(MB)": float(memory_used / batch_size)
                }
                
                print(f"推理时间: {inference_time * 1000:.2f} ms")
                print(f"每样本时间: {inference_time * 1000 / batch_size:.2f} ms")
                print(f"内存使用: {memory_used:.2f} MB")
                print(f"每样本内存: {memory_used / batch_size:.2f} MB")
                
            except Exception as e:
                results[f"batch_{batch_size}"] = {
                    "状态": "失败",
                    "错误": str(e)
                }
                print(f"测试失败: {e}")
        
        # 可视化
        valid_batch_sizes = []
        inference_times = []
        memory_usages = []
        
        for batch_size in batch_sizes:
            result = results.get(f"batch_{batch_size}")
            if result and result["状态"] == "成功":
                valid_batch_sizes.append(batch_size)
                inference_times.append(result["推理时间(ms)"])
                memory_usages.append(result["内存使用(MB)"])
        
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(valid_batch_sizes, inference_times, marker='o')
        plt.title("批次大小 vs 推理时间")
        plt.xlabel("批次大小")
        plt.ylabel("推理时间 (ms)")
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(valid_batch_sizes, memory_usages, marker='o')
        plt.title("批次大小 vs 内存使用")
        plt.xlabel("批次大小")
        plt.ylabel("内存使用 (MB)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "large_batch_size.png"))
        plt.close()
        
        # 保存结果
        self._save_results("large_batch_size", results)
        
        return results
    
    def test_extreme_shapes(self):
        """
        测试极端形状
        """
        print("\n===== 测试极端形状 =====")
        
        # 测试不同的环数和扇区数组合
        shapes = [
            (5, 5),       # 极小形状
            (5, 100),     # 极不平衡形状（宽）
            (100, 5),     # 极不平衡形状（高）
            (100, 100),   # 极大形状
            (1, 60),      # 单行
            (20, 1),      # 单列
            (50, 50),     # 方形
        ]
        
        results = {}
        
        for num_rings, num_sectors in shapes:
            try:
                print(f"\n测试形状: {num_rings}环 x {num_sectors}扇区")
                
                # 创建模型
                model = SCRingCNN(
                    in_channels=1,
                    num_rings=num_rings,
                    num_sectors=num_sectors,
                    descriptor_dim=128
                ).to(self.device)
                
                # 创建输入
                x = torch.randn(1, 1, num_rings, num_sectors).to(self.device)
                
                # 前向传播
                start_time = time.time()
                with torch.no_grad():
                    output = model(x)
                inference_time = time.time() - start_time
                
                results[f"{num_rings}x{num_sectors}"] = {
                    "状态": "成功",
                    "推理时间(ms)": float(inference_time * 1000),
                    "输出形状": tuple(output.shape)
                }
                
                print(f"推理时间: {inference_time * 1000:.2f} ms")
                print(f"输出形状: {output.shape}")
                
            except Exception as e:
                results[f"{num_rings}x{num_sectors}"] = {
                    "状态": "失败",
                    "错误": str(e)
                }
                print(f"测试失败: {e}")
        
        # 保存结果
        self._save_results("extreme_shapes", results)
        
        return results
    
    def test_weight_perturbation(self):
        """
        测试权重扰动
        """
        print("\n===== 测试权重扰动 =====")
        
        # 设置参数
        num_rings = 20
        num_sectors = 60
        descriptor_dim = 256
        
        # 创建模型
        model = SCRingCNN(
            in_channels=1,
            num_rings=num_rings,
            num_sectors=num_sectors,
            descriptor_dim=descriptor_dim
        ).to(self.device)
        
        # 创建输入
        x = torch.randn(1, 1, num_rings, num_sectors).to(self.device)
        
        # 获取原始输出
        with torch.no_grad():
            original_output = model(x).cpu().numpy()
        
        # 测试不同的扰动水平
        perturbation_levels = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
        results = {}
        distances = []
        
        for level in perturbation_levels:
            try:
                print(f"\n测试扰动水平: {level}")
                
                # 复制模型
                perturbed_model = SCRingCNN(
                    in_channels=1,
                    num_rings=num_rings,
                    num_sectors=num_sectors,
                    descriptor_dim=descriptor_dim
                ).to(self.device)
                perturbed_model.load_state_dict(model.state_dict())
                
                # 扰动权重
                with torch.no_grad():
                    for param in perturbed_model.parameters():
                        noise = torch.randn_like(param) * level * param.std()
                        param.add_(noise)
                
                # 获取扰动后的输出
                with torch.no_grad():
                    perturbed_output = perturbed_model(x).cpu().numpy()
                
                # 计算输出差异
                distance = np.sqrt(np.sum((original_output - perturbed_output) ** 2))
                distances.append(distance)
                
                results[f"level_{level}"] = {
                    "状态": "成功",
                    "输出距离": float(distance),
                    "相对扰动": float(level)
                }
                
                print(f"输出距离: {distance:.6f}")
                
            except Exception as e:
                results[f"level_{level}"] = {
                    "状态": "失败",
                    "错误": str(e)
                }
                print(f"测试失败: {e}")
        
        # 可视化
        plt.figure(figsize=(10, 6))
        plt.plot(perturbation_levels, distances, marker='o')
        plt.title("权重扰动 vs 输出距离")
        plt.xlabel("扰动水平")
        plt.ylabel("输出距离")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "weight_perturbation.png"))
        plt.close()
        
        # 保存结果
        self._save_results("weight_perturbation", results)
        
        return results
    
    def run_all_tests(self):
        """
        运行所有测试
        """
        print("开始运行所有极端情况测试...")
        
        # 记录开始时间
        start_time = time.time()
        
        # 运行所有测试
        results = {
            "nan_inf_inputs": self.test_nan_inf_inputs(),
            "gradient_stability": self.test_gradient_stability(),
            "large_batch_size": self.test_large_batch_size(),
            "extreme_shapes": self.test_extreme_shapes(),
            "weight_perturbation": self.test_weight_perturbation()
        }
        
        # 记录结束时间
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n所有测试完成! 总耗时: {total_time:.2f}秒")
        
        # 保存总结果
        self._save_results("all_tests_summary", {
            "总耗时": float(total_time),
            "测试数量": len(results),
            "完成时间": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return results
    
    def _save_results(self, test_name, results):
        """
        保存测试结果
        
        参数:
            test_name (str): 测试名称
            results (dict): 测试结果
        """
        import json
        
        # 自定义JSON编码器，处理numpy类型
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating, np.bool_)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        # 保存为JSON文件
        with open(os.path.join(self.output_dir, f"{test_name}_results.json"), "w") as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)

if __name__ == "__main__":
    # 创建测试实例
    tester = ExtremeTest(output_dir="extreme_results")
    
    # 运行所有测试
    tester.run_all_tests()