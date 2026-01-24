#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import SCRingCNN, SCRingCNNLite
from models.circular_conv import CircularPadConv2d, CircularResidualBlock, CircularConvDataAugmentation
from utils import ScanContext

class RotationInvarianceTest:
    """
    SC-RingCNN旋转不变性测试类
    
    专门测试SC-RingCNN模型的旋转不变性，包括：
    1. 基本旋转不变性测试
    2. 不同旋转角度的描述子距离分析
    3. 与标准卷积网络的对比测试
    4. 数据增强对旋转不变性的影响
    5. 实际点云数据的旋转不变性测试
    """
    
    def __init__(self, output_dir="rotation_invariance_results"):
        """
        初始化旋转不变性测试
        
        参数:
            output_dir (str): 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 设置默认参数
        self.num_rings = 20
        self.num_sectors = 60
        self.descriptor_dim = 256
        
    def create_test_input(self, pattern_type="gaussian"):
        """
        创建测试输入
        
        参数:
            pattern_type (str): 模式类型，可选"gaussian"、"lines"、"blocks"
            
        返回:
            test_tensor (torch.Tensor): 测试张量
        """
        test_tensor = torch.zeros(1, 1, self.num_rings, self.num_sectors)
        
        if pattern_type == "gaussian":
            # 创建高斯模式
            for i in range(self.num_rings):
                for j in range(self.num_sectors):
                    r = ((i - self.num_rings//2) ** 2 + (j - self.num_sectors//2) ** 2) ** 0.5
                    test_tensor[0, 0, i, j] = np.exp(-r / 10)
                    
        elif pattern_type == "lines":
            # 创建线条模式
            for i in range(self.num_rings):
                test_tensor[0, 0, i, self.num_sectors//4] = 1.0
                test_tensor[0, 0, i, self.num_sectors//2] = 0.7
                test_tensor[0, 0, i, 3*self.num_sectors//4] = 0.4
                
        elif pattern_type == "blocks":
            # 创建块状模式
            test_tensor[0, 0, :self.num_rings//2, :self.num_sectors//2] = 1.0
            test_tensor[0, 0, self.num_rings//2:, self.num_sectors//2:] = 0.5
            
        else:
            raise ValueError(f"不支持的模式类型: {pattern_type}")
            
        return test_tensor
    
    def test_basic_rotation_invariance(self):
        """
        基本旋转不变性测试
        """
        print("\n===== 基本旋转不变性测试 =====")
        
        # 创建模型
        model = SCRingCNN(
            in_channels=1,
            num_rings=self.num_rings,
            num_sectors=self.num_sectors,
            descriptor_dim=self.descriptor_dim
        ).to(self.device)
        
        # 创建测试输入
        test_tensor = self.create_test_input("gaussian").to(self.device)
        
        # 获取原始描述子
        with torch.no_grad():
            original_descriptor = model(test_tensor).cpu().numpy()
        
        # 测试不同的旋转角度
        shifts = list(range(0, self.num_sectors, self.num_sectors // 12))  # 每30度一个测试点
        results = {}
        distances = []
        
        for shift in shifts:
            if shift == 0:
                continue
                
            # 循环移位输入
            shifted_tensor = torch.roll(test_tensor, shifts=shift, dims=3)
            
            # 获取旋转后的描述子
            with torch.no_grad():
                shifted_descriptor = model(shifted_tensor).cpu().numpy()
            
            # 计算描述子之间的距离
            distance = np.sqrt(np.sum((original_descriptor - shifted_descriptor) ** 2))
            distances.append(distance)
            
            results[f"旋转{shift}"] = {
                "旋转角度": float(shift * 360 / self.num_sectors),
                "描述子距离": float(distance)
            }
            
            print(f"旋转 {shift} 扇区 ({shift * 360 / self.num_sectors:.1f}°): 距离 = {distance:.6f}")
        
        # 计算平均距离和标准差
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        print(f"平均距离: {mean_distance:.6f}")
        print(f"距离标准差: {std_distance:.6f}")
        
        results["统计"] = {
            "平均距离": float(mean_distance),
            "距离标准差": float(std_distance),
            "最大距离": float(np.max(distances)),
            "最小距离": float(np.min(distances))
        }
        
        # 可视化结果
        plt.figure(figsize=(12, 10))
        
        # 绘制原始输入和旋转后的输入
        plt.subplot(2, 2, 1)
        plt.imshow(test_tensor[0, 0].cpu().numpy())
        plt.title("原始输入")
        plt.colorbar()
        
        plt.subplot(2, 2, 2)
        plt.imshow(shifted_tensor[0, 0].cpu().numpy())
        plt.title(f"旋转后的输入 ({shifts[-1] * 360 / self.num_sectors:.1f}°)")
        plt.colorbar()
        
        # 绘制距离曲线
        plt.subplot(2, 1, 2)
        plt.plot([s * 360 / self.num_sectors for s in shifts if s != 0], distances, marker='o')
        plt.axhline(y=mean_distance, color='r', linestyle='--', label=f'平均距离: {mean_distance:.4f}')
        plt.fill_between([s * 360 / self.num_sectors for s in shifts if s != 0], 
                        [mean_distance - std_distance] * len(distances), 
                        [mean_distance + std_distance] * len(distances), 
                        alpha=0.2, color='r', label=f'标准差: {std_distance:.4f}')
        plt.xlabel('旋转角度 (度)')
        plt.ylabel('描述子距离')
        plt.title('旋转角度 vs 描述子距离')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "basic_rotation_invariance.png"))
        plt.close()
        
        # 保存结果
        self._save_results("basic_rotation_invariance", results)
        
        return results
    
    def test_pattern_rotation_invariance(self):
        """
        测试不同模式的旋转不变性
        """
        print("\n===== 不同模式的旋转不变性测试 =====")
        
        # 创建模型
        model = SCRingCNN(
            in_channels=1,
            num_rings=self.num_rings,
            num_sectors=self.num_sectors,
            descriptor_dim=self.descriptor_dim
        ).to(self.device)
        
        # 测试不同的模式
        patterns = ["gaussian", "lines", "blocks"]
        results = {}
        
        for pattern in patterns:
            print(f"\n测试模式: {pattern}")
            
            # 创建测试输入
            test_tensor = self.create_test_input(pattern).to(self.device)
            
            # 获取原始描述子
            with torch.no_grad():
                original_descriptor = model(test_tensor).cpu().numpy()
            
            # 测试不同的旋转角度
            shifts = list(range(0, self.num_sectors, self.num_sectors // 12))  # 每30度一个测试点
            pattern_results = {}
            distances = []
            
            for shift in shifts:
                if shift == 0:
                    continue
                    
                # 循环移位输入
                shifted_tensor = torch.roll(test_tensor, shifts=shift, dims=3)
                
                # 获取旋转后的描述子
                with torch.no_grad():
                    shifted_descriptor = model(shifted_tensor).cpu().numpy()
                
                # 计算描述子之间的距离
                distance = np.sqrt(np.sum((original_descriptor - shifted_descriptor) ** 2))
                distances.append(distance)
                
                pattern_results[f"旋转{shift}"] = {
                    "旋转角度": float(shift * 360 / self.num_sectors),
                    "描述子距离": float(distance)
                }
            
            # 计算平均距离和标准差
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            
            print(f"平均距离: {mean_distance:.6f}")
            print(f"距离标准差: {std_distance:.6f}")
            
            pattern_results["统计"] = {
                "平均距离": float(mean_distance),
                "距离标准差": float(std_distance),
                "最大距离": float(np.max(distances)),
                "最小距离": float(np.min(distances))
            }
            
            results[pattern] = pattern_results
            
            # 可视化结果
            plt.figure(figsize=(12, 10))
            
            # 绘制原始输入和旋转后的输入
            plt.subplot(2, 2, 1)
            plt.imshow(test_tensor[0, 0].cpu().numpy())
            plt.title(f"{pattern} - 原始输入")
            plt.colorbar()
            
            plt.subplot(2, 2, 2)
            plt.imshow(shifted_tensor[0, 0].cpu().numpy())
            plt.title(f"{pattern} - 旋转后的输入 ({shifts[-1] * 360 / self.num_sectors:.1f}°)")
            plt.colorbar()
            
            # 绘制距离曲线
            plt.subplot(2, 1, 2)
            plt.plot([s * 360 / self.num_sectors for s in shifts if s != 0], distances, marker='o')
            plt.axhline(y=mean_distance, color='r', linestyle='--', label=f'平均距离: {mean_distance:.4f}')
            plt.fill_between([s * 360 / self.num_sectors for s in shifts if s != 0], 
                            [mean_distance - std_distance] * len(distances), 
                            [mean_distance + std_distance] * len(distances), 
                            alpha=0.2, color='r', label=f'标准差: {std_distance:.4f}')
            plt.xlabel('旋转角度 (度)')
            plt.ylabel('描述子距离')
            plt.title(f'{pattern} - 旋转角度 vs 描述子距离')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"pattern_rotation_invariance_{pattern}.png"))
            plt.close()
        
        # 保存结果
        self._save_results("pattern_rotation_invariance", results)
        
        return results
    
    def test_comparison_with_standard_cnn(self):
        """
        与标准卷积网络的对比测试
        """
        print("\n===== 与标准卷积网络的对比测试 =====")
        
        # 创建SC-RingCNN模型
        sc_model = SCRingCNN(
            in_channels=1,
            num_rings=self.num_rings,
            num_sectors=self.num_sectors,
            descriptor_dim=self.descriptor_dim
        ).to(self.device)
        
        # 创建标准CNN模型（使用相同的架构，但不使用环形填充）
        class StandardCNN(torch.nn.Module):
            def __init__(self, in_channels, num_rings, num_sectors, descriptor_dim):
                super(StandardCNN, self).__init__()
                
                # 使用标准卷积层
                self.layer1 = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2)
                )
                
                self.layer2 = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2)
                )
                
                self.layer3 = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2)
                )
                
                self.layer4 = torch.nn.Sequential(
                    torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.AdaptiveAvgPool2d((2, 2))
                )
                
                self.fc = torch.nn.Sequential(
                    torch.nn.Linear(256 * 2 * 2, 512),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(512, descriptor_dim)
                )
                
            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return torch.nn.functional.normalize(x, p=2, dim=1)
        
        std_model = StandardCNN(
            in_channels=1,
            num_rings=self.num_rings,
            num_sectors=self.num_sectors,
            descriptor_dim=self.descriptor_dim
        ).to(self.device)
        
        # 创建测试输入
        test_tensor = self.create_test_input("gaussian").to(self.device)
        
        # 获取原始描述子
        with torch.no_grad():
            sc_original_descriptor = sc_model(test_tensor).cpu().numpy()
            std_original_descriptor = std_model(test_tensor).cpu().numpy()
        
        # 测试不同的旋转角度
        shifts = list(range(0, self.num_sectors, self.num_sectors // 12))  # 每30度一个测试点
        results = {"SC-RingCNN": {}, "StandardCNN": {}}
        sc_distances = []
        std_distances = []
        
        for shift in shifts:
            if shift == 0:
                continue
                
            # 循环移位输入
            shifted_tensor = torch.roll(test_tensor, shifts=shift, dims=3)
            
            # 获取旋转后的描述子
            with torch.no_grad():
                sc_shifted_descriptor = sc_model(shifted_tensor).cpu().numpy()
                std_shifted_descriptor = std_model(shifted_tensor).cpu().numpy()
            
            # 计算描述子之间的距离
            sc_distance = np.sqrt(np.sum((sc_original_descriptor - sc_shifted_descriptor) ** 2))
            std_distance = np.sqrt(np.sum((std_original_descriptor - std_shifted_descriptor) ** 2))
            
            sc_distances.append(sc_distance)
            std_distances.append(std_distance)
            
            results["SC-RingCNN"][f"旋转{shift}"] = {
                "旋转角度": float(shift * 360 / self.num_sectors),
                "描述子距离": float(sc_distance)
            }
            
            results["StandardCNN"][f"旋转{shift}"] = {
                "旋转角度": float(shift * 360 / self.num_sectors),
                "描述子距离": float(std_distance)
            }
            
            print(f"旋转 {shift} 扇区 ({shift * 360 / self.num_sectors:.1f}°):")
            print(f"  SC-RingCNN 距离 = {sc_distance:.6f}")
            print(f"  StandardCNN 距离 = {std_distance:.6f}")
        
        # 计算平均距离和标准差
        sc_mean_distance = np.mean(sc_distances)
        sc_std_distance = np.std(sc_distances)
        std_mean_distance = np.mean(std_distances)
        std_std_distance = np.std(std_distances)
        
        print(f"SC-RingCNN 平均距离: {sc_mean_distance:.6f}, 标准差: {sc_std_distance:.6f}")
        print(f"StandardCNN 平均距离: {std_mean_distance:.6f}, 标准差: {std_std_distance:.6f}")
        
        results["SC-RingCNN"]["统计"] = {
            "平均距离": float(sc_mean_distance),
            "距离标准差": float(sc_std_distance),
            "最大距离": float(np.max(sc_distances)),
            "最小距离": float(np.min(sc_distances))
        }
        
        results["StandardCNN"]["统计"] = {
            "平均距离": float(std_mean_distance),
            "距离标准差": float(std_std_distance),
            "最大距离": float(np.max(std_distances)),
            "最小距离": float(np.min(std_distances))
        }
        
        # 计算改进比例
        improvement = (std_mean_distance - sc_mean_distance) / std_mean_distance * 100
        print(f"SC-RingCNN 相比 StandardCNN 的改进: {improvement:.2f}%")
        
        results["比较"] = {
            "改进比例(%)": float(improvement),
            "标准差改进比例(%)": float((std_std_distance - sc_std_distance) / std_std_distance * 100)
        }
        
        # 可视化结果
        plt.figure(figsize=(12, 10))
        
        # 绘制原始输入和旋转后的输入
        plt.subplot(2, 2, 1)
        plt.imshow(test_tensor[0, 0].cpu().numpy())
        plt.title("原始输入")
        plt.colorbar()
        
        plt.subplot(2, 2, 2)
        plt.imshow(shifted_tensor[0, 0].cpu().numpy())
        plt.title(f"旋转后的输入 ({shifts[-1] * 360 / self.num_sectors:.1f}°)")
        plt.colorbar()
        
        # 绘制距离曲线
        plt.subplot(2, 1, 2)
        x_values = [s * 360 / self.num_sectors for s in shifts if s != 0]
        plt.plot(x_values, sc_distances, marker='o', label='SC-RingCNN')
        plt.plot(x_values, std_distances, marker='s', label='StandardCNN')
        plt.axhline(y=sc_mean_distance, color='r', linestyle='--', label=f'SC-RingCNN 平均: {sc_mean_distance:.4f}')
        plt.axhline(y=std_mean_distance, color='b', linestyle='--', label=f'StandardCNN 平均: {std_mean_distance:.4f}')
        plt.xlabel('旋转角度 (度)')
        plt.ylabel('描述子距离')
        plt.title('旋转角度 vs 描述子距离')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "comparison_with_standard_cnn.png"))
        plt.close()
        
        # 保存结果
        self._save_results("comparison_with_standard_cnn", results)
        
        return results
    
    def test_data_augmentation_effect(self):
        """
        测试数据增强对旋转不变性的影响
        """
        print("\n===== 数据增强对旋转不变性的影响 =====")
        
        # 创建模型
        model = SCRingCNN(
            in_channels=1,
            num_rings=self.num_rings,
            num_sectors=self.num_sectors,
            descriptor_dim=self.descriptor_dim
        ).to(self.device)
        
        # 创建测试输入
        test_tensor = self.create_test_input("gaussian").to(self.device)
        
        # 应用数据增强
        test_tensor_np = test_tensor.cpu().numpy()[0, 0]  # 移除批次和通道维度
        augmented_tensor, shift = CircularConvDataAugmentation.random_shift(test_tensor_np)
        augmented_tensor = torch.from_numpy(augmented_tensor).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        print(f"随机移位量: {shift} ({shift * 360 / self.num_sectors:.1f}°)")
        
        # 获取原始和增强后的描述子
        with torch.no_grad():
            original_descriptor = model(test_tensor).cpu().numpy()
            augmented_descriptor = model(augmented_tensor).cpu().numpy()
        
        # 计算描述子之间的距离
        distance = np.sqrt(np.sum((original_descriptor - augmented_descriptor) ** 2))
        
        print(f"原始与增强后的描述子距离: {distance:.6f}")
        
        # 测试批量数据增强
        batch_size = 8
        batch_tensor = np.zeros((batch_size, self.num_rings, self.num_sectors))
        for i in range(batch_size):
            batch_tensor[i] = test_tensor_np  # 复制相同的测试张量
        
        # 应用批量随机移位
        augmented_batch, shifts = CircularConvDataAugmentation.batch_random_shift(batch_tensor)
        augmented_batch = torch.from_numpy(augmented_batch).unsqueeze(1).float().to(self.device)  # 添加通道维度
        
        # 获取批量增强后的描述子
        with torch.no_grad():
            batch_descriptors = model(augmented_batch).cpu().numpy()
        
        # 计算批量描述子之间的距离矩阵
        distance_matrix = np.zeros((batch_size, batch_size))
        for i in range(batch_size):
            for j in range(batch_size):
                distance_matrix[i, j] = np.sqrt(np.sum((batch_descriptors[i] - batch_descriptors[j]) ** 2))
        
        # 计算平均距离和标准差
        mean_distance = np.mean(distance_matrix)
        std_distance = np.std(distance_matrix)
        
        print(f"批量描述子平均距离: {mean_distance:.6f}")
        print(f"批量描述子距离标准差: {std_distance:.6f}")
        
        results = {
            "单样本": {
                "移位量": int(shift),
                "旋转角度": float(shift * 360 / self.num_sectors),
                "描述子距离": float(distance)
            },
            "批量样本": {
                "移位量": shifts.tolist(),
                "平均距离": float(mean_distance),
                "距离标准差": float(std_distance)
            }
        }
        
        # 可视化结果
        plt.figure(figsize=(12, 10))
        
        # 绘制原始输入和增强后的输入
        plt.subplot(2, 2, 1)
        plt.imshow(test_tensor[0, 0].cpu().numpy())
        plt.title("原始输入")
        plt.colorbar()
        
        plt.subplot(2, 2, 2)
        plt.imshow(augmented_tensor[0, 0].cpu().numpy())
        plt.title(f"增强后的输入 (移位 {shift})")
        plt.colorbar()
        
        # 绘制批量描述子距离矩阵
        plt.subplot(2, 1, 2)
        im = plt.imshow(distance_matrix)
        plt.colorbar(im)
        plt.title(f"批量描述子距离矩阵 (平均: {mean_distance:.4f}, 标准差: {std_distance:.4f})")
        plt.xlabel('样本索引')
        plt.ylabel('样本索引')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "data_augmentation_effect.png"))
        plt.close()
        
        # 保存结果
        self._save_results("data_augmentation_effect", results)
        
        return results
    
    def run_all_tests(self):
        """
        运行所有测试
        """
        print("开始运行所有旋转不变性测试...")
        
        # 记录开始时间
        start_time = time.time()
        
        # 运行所有测试
        results = {
            "basic_rotation_invariance": self.test_basic_rotation_invariance(),
            "pattern_rotation_invariance": self.test_pattern_rotation_invariance(),
            "comparison_with_standard_cnn": self.test_comparison_with_standard_cnn(),
            "data_augmentation_effect": self.test_data_augmentation_effect()
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
    tester = RotationInvarianceTest(output_dir="rotation_invariance_results")
    
    # 运行所有测试
    tester.run_all_tests() 