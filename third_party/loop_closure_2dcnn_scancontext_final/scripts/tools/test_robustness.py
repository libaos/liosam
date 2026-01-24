#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import SCRingCNN, SCRingCNNLite
from models.circular_conv import CircularPadConv2d, CircularConvDataAugmentation
from utils import ScanContext

class RobustnessTest:
    """
    SC-RingCNN鲁棒性测试类
    
    用于测试SC-RingCNN模型在各种条件下的鲁棒性，包括：
    1. 输入大小变化
    2. 旋转不变性
    3. 噪声抵抗力
    4. 边界情况处理
    5. 数值稳定性
    """
    
    def __init__(self, output_dir="robustness_results"):
        """
        初始化鲁棒性测试
        
        参数:
            output_dir (str): 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
    def test_input_size_robustness(self):
        """
        测试模型对不同输入大小的鲁棒性
        """
        print("\n===== 测试输入大小鲁棒性 =====")
        
        # 测试不同的环数和扇区数
        ring_sizes = [10, 20, 40, 60]
        sector_sizes = [30, 60, 90, 120]
        
        results = {}
        
        for num_rings in ring_sizes:
            for num_sectors in sector_sizes:
                try:
                    print(f"\n测试输入大小: {num_rings}环 x {num_sectors}扇区")
                    
                    # 创建模型
                    model = SCRingCNN(
                        in_channels=1,
                        num_rings=num_rings,
                        num_sectors=num_sectors,
                        descriptor_dim=256
                    ).to(self.device)
                    
                    # 创建随机输入
                    batch_size = 2
                    x = torch.randn(batch_size, 1, num_rings, num_sectors).to(self.device)
                    
                    # 前向传播
                    start_time = time.time()
                    with torch.no_grad():
                        output = model(x)
                    inference_time = time.time() - start_time
                    
                    # 检查输出形状
                    expected_shape = (batch_size, 256)
                    assert output.shape == expected_shape, f"输出形状错误: {output.shape} != {expected_shape}"
                    
                    # 记录结果
                    results[f"{num_rings}x{num_sectors}"] = {
                        "status": "成功",
                        "inference_time": inference_time,
                        "output_shape": tuple(output.shape)
                    }
                    
                    print(f"测试通过! 推理时间: {inference_time:.4f}秒")
                    
                except Exception as e:
                    results[f"{num_rings}x{num_sectors}"] = {
                        "status": "失败",
                        "error": str(e)
                    }
                    print(f"测试失败: {e}")
        
        # 保存结果
        self._save_results("input_size_robustness", results)
        
        return results
    
    def test_rotation_invariance(self):
        """
        测试模型的旋转不变性
        """
        print("\n===== 测试旋转不变性 =====")
        
        # 设置参数
        num_rings = 20
        num_sectors = 60
        descriptor_dim = 256
        shifts = [0, 10, 20, 30, 40, 50]  # 不同的旋转角度
        
        # 创建模型
        model = SCRingCNN(
            in_channels=1,
            num_rings=num_rings,
            num_sectors=num_sectors,
            descriptor_dim=descriptor_dim
        ).to(self.device)
        
        # 创建随机输入
        x = torch.randn(1, 1, num_rings, num_sectors).to(self.device)
        
        # 获取原始描述子
        with torch.no_grad():
            original_descriptor = model(x).cpu().numpy()
        
        results = {"原始描述子": original_descriptor.flatten().tolist()}
        distances = []
        
        # 测试不同的旋转角度
        for shift in shifts:
            if shift == 0:
                continue
                
            # 循环移位输入
            x_shifted = torch.roll(x, shifts=shift, dims=3)
            
            # 获取旋转后的描述子
            with torch.no_grad():
                shifted_descriptor = model(x_shifted).cpu().numpy()
            
            # 计算描述子之间的距离
            distance = np.sqrt(np.sum((original_descriptor - shifted_descriptor) ** 2))
            distances.append(distance)
            
            results[f"旋转{shift}"] = {
                "描述子": shifted_descriptor.flatten().tolist(),
                "距离": float(distance)
            }
            
            print(f"旋转 {shift} 扇区: 距离 = {distance:.6f}")
        
        # 计算平均距离和标准差
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        print(f"平均距离: {mean_distance:.6f}")
        print(f"距离标准差: {std_distance:.6f}")
        
        results["统计"] = {
            "平均距离": float(mean_distance),
            "距离标准差": float(std_distance)
        }
        
        # 可视化结果
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(distances)), distances)
        plt.xlabel('旋转角度索引')
        plt.ylabel('描述子距离')
        plt.title('旋转不变性测试')
        plt.savefig(os.path.join(self.output_dir, "rotation_invariance.png"))
        plt.close()
        
        # 保存结果
        self._save_results("rotation_invariance", results)
        
        return results
    
    def test_noise_robustness(self):
        """
        测试模型对噪声的鲁棒性
        """
        print("\n===== 测试噪声鲁棒性 =====")
        
        # 设置参数
        num_rings = 20
        num_sectors = 60
        descriptor_dim = 256
        noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
        
        # 创建模型
        model = SCRingCNN(
            in_channels=1,
            num_rings=num_rings,
            num_sectors=num_sectors,
            descriptor_dim=descriptor_dim
        ).to(self.device)
        
        # 创建随机输入
        x = torch.randn(1, 1, num_rings, num_sectors).to(self.device)
        
        # 获取原始描述子
        with torch.no_grad():
            original_descriptor = model(x).cpu().numpy()
        
        results = {"原始描述子": original_descriptor.flatten().tolist()}
        distances = []
        
        # 测试不同的噪声水平
        for noise_level in noise_levels:
            if noise_level == 0.0:
                continue
                
            # 添加噪声
            noise = torch.randn_like(x) * noise_level
            x_noisy = x + noise
            
            # 获取噪声后的描述子
            with torch.no_grad():
                noisy_descriptor = model(x_noisy).cpu().numpy()
            
            # 计算描述子之间的距离
            distance = np.sqrt(np.sum((original_descriptor - noisy_descriptor) ** 2))
            distances.append(distance)
            
            results[f"噪声{noise_level}"] = {
                "距离": float(distance)
            }
            
            print(f"噪声水平 {noise_level}: 距离 = {distance:.6f}")
        
        # 可视化结果
        plt.figure(figsize=(10, 6))
        plt.plot(noise_levels[1:], distances, marker='o')
        plt.xlabel('噪声水平')
        plt.ylabel('描述子距离')
        plt.title('噪声鲁棒性测试')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "noise_robustness.png"))
        plt.close()
        
        # 保存结果
        self._save_results("noise_robustness", results)
        
        return results
    
    def test_edge_cases(self):
        """
        测试边界情况
        """
        print("\n===== 测试边界情况 =====")
        
        results = {}
        
        # 测试1：极小输入
        try:
            print("\n测试极小输入")
            model = SCRingCNN(in_channels=1, num_rings=4, num_sectors=4, descriptor_dim=8).to(self.device)
            x = torch.randn(1, 1, 4, 4).to(self.device)
            with torch.no_grad():
                output = model(x)
            results["极小输入"] = {
                "状态": "成功",
                "输出形状": tuple(output.shape)
            }
            print("测试通过!")
        except Exception as e:
            results["极小输入"] = {
                "状态": "失败",
                "错误": str(e)
            }
            print(f"测试失败: {e}")
        
        # 测试2：零输入
        try:
            print("\n测试零输入")
            model = SCRingCNN(in_channels=1, num_rings=20, num_sectors=60).to(self.device)
            x = torch.zeros(1, 1, 20, 60).to(self.device)
            with torch.no_grad():
                output = model(x)
            results["零输入"] = {
                "状态": "成功",
                "输出形状": tuple(output.shape),
                "输出范数": float(torch.norm(output).item())
            }
            print("测试通过!")
        except Exception as e:
            results["零输入"] = {
                "状态": "失败",
                "错误": str(e)
            }
            print(f"测试失败: {e}")
        
        # 测试3：极大值输入
        try:
            print("\n测试极大值输入")
            model = SCRingCNN(in_channels=1, num_rings=20, num_sectors=60).to(self.device)
            x = torch.ones(1, 1, 20, 60).to(self.device) * 1e6
            with torch.no_grad():
                output = model(x)
            results["极大值输入"] = {
                "状态": "成功",
                "输出形状": tuple(output.shape),
                "输出范数": float(torch.norm(output).item())
            }
            print("测试通过!")
        except Exception as e:
            results["极大值输入"] = {
                "状态": "失败",
                "错误": str(e)
            }
            print(f"测试失败: {e}")
        
        # 测试4：多通道输入
        try:
            print("\n测试多通道输入")
            model = SCRingCNN(in_channels=3, num_rings=20, num_sectors=60).to(self.device)
            x = torch.randn(1, 3, 20, 60).to(self.device)
            with torch.no_grad():
                output = model(x)
            results["多通道输入"] = {
                "状态": "成功",
                "输出形状": tuple(output.shape)
            }
            print("测试通过!")
        except Exception as e:
            results["多通道输入"] = {
                "状态": "失败",
                "错误": str(e)
            }
            print(f"测试失败: {e}")
        
        # 保存结果
        self._save_results("edge_cases", results)
        
        return results
    
    def test_numerical_stability(self):
        """
        测试数值稳定性
        """
        print("\n===== 测试数值稳定性 =====")
        
        # 设置参数
        num_rings = 20
        num_sectors = 60
        descriptor_dim = 256
        num_iterations = 100
        
        # 创建模型
        model = SCRingCNN(
            in_channels=1,
            num_rings=num_rings,
            num_sectors=num_sectors,
            descriptor_dim=descriptor_dim
        ).to(self.device)
        
        # 固定随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 创建随机输入
        x = torch.randn(1, 1, num_rings, num_sectors).to(self.device)
        
        # 多次运行模型，检查结果是否一致
        descriptors = []
        
        for i in tqdm(range(num_iterations), desc="运行迭代"):
            with torch.no_grad():
                output = model(x)
                descriptors.append(output.cpu().numpy())
        
        # 计算描述子之间的最大差异
        descriptors = np.array(descriptors)
        max_diff = 0
        
        for i in range(1, num_iterations):
            diff = np.max(np.abs(descriptors[0] - descriptors[i]))
            max_diff = max(max_diff, diff)
        
        print(f"最大差异: {max_diff}")
        
        results = {
            "最大差异": float(max_diff),
            "是否稳定": max_diff < 1e-6
        }
        
        # 保存结果
        self._save_results("numerical_stability", results)
        
        return results
    
    def test_circular_padding(self):
        """
        测试环形填充的正确性
        """
        print("\n===== 测试环形填充 =====")
        
        # 创建一个简单的测试张量
        # 使用一个特殊模式，使得我们可以轻松验证环形填充是否正确
        test_tensor = torch.zeros(1, 1, 5, 10)
        for i in range(10):
            test_tensor[0, 0, :, i] = i
        
        # 创建环形填充卷积层
        circular_conv = CircularPadConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1
        )
        
        # 应用环形填充
        output = circular_conv(test_tensor)
        
        # 验证输出形状
        assert output.shape == test_tensor.shape, f"输出形状错误: {output.shape} != {test_tensor.shape}"
        
        # 创建一个标准卷积层进行比较
        standard_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1
        )
        
        # 将环形卷积层的权重复制到标准卷积层
        standard_conv.weight.data = circular_conv.conv.weight.data
        standard_conv.bias.data = circular_conv.conv.bias.data
        
        # 手动进行环形填充
        padded_tensor = torch.zeros(1, 1, 5, 12)
        padded_tensor[0, 0, :, 1:-1] = test_tensor[0, 0]
        padded_tensor[0, 0, :, 0] = test_tensor[0, 0, :, -1]  # 左侧填充
        padded_tensor[0, 0, :, -1] = test_tensor[0, 0, :, 0]  # 右侧填充
        
        # 应用标准卷积
        standard_output = standard_conv(padded_tensor)
        
        # 计算差异
        diff = torch.abs(output - standard_output[:, :, :, 1:-1]).max().item()
        
        print(f"环形填充与手动填充的最大差异: {diff}")
        
        results = {
            "最大差异": float(diff),
            "是否正确": diff < 1e-6
        }
        
        # 可视化结果
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.imshow(test_tensor[0, 0].detach().cpu().numpy())
        plt.title("原始张量")
        plt.colorbar()
        
        plt.subplot(3, 1, 2)
        plt.imshow(padded_tensor[0, 0].detach().cpu().numpy())
        plt.title("手动环形填充")
        plt.colorbar()
        
        plt.subplot(3, 1, 3)
        plt.imshow(output[0, 0].detach().cpu().numpy())
        plt.title("环形填充卷积输出")
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "circular_padding_test.png"))
        plt.close()
        
        # 保存结果
        self._save_results("circular_padding", results)
        
        return results
    
    def test_data_augmentation(self):
        """
        测试数据增强的正确性
        """
        print("\n===== 测试数据增强 =====")
        
        # 创建一个简单的测试张量
        test_tensor = np.zeros((5, 10))
        for i in range(10):
            test_tensor[:, i] = i
        
        # 应用随机移位
        shifted_tensor, shift = CircularConvDataAugmentation.random_shift(test_tensor)
        
        # 验证移位是否正确
        manually_shifted = np.roll(test_tensor, shift, axis=1)
        diff = np.abs(shifted_tensor - manually_shifted).max()
        
        print(f"移位量: {shift}")
        print(f"随机移位与手动移位的最大差异: {diff}")
        
        results = {
            "移位量": int(shift),
            "最大差异": float(diff),
            "是否正确": diff < 1e-6
        }
        
        # 可视化结果
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.imshow(test_tensor)
        plt.title("原始张量")
        plt.colorbar()
        
        plt.subplot(3, 1, 2)
        plt.imshow(shifted_tensor)
        plt.title(f"随机移位 (移位量: {shift})")
        plt.colorbar()
        
        plt.subplot(3, 1, 3)
        plt.imshow(manually_shifted)
        plt.title("手动移位")
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "data_augmentation_test.png"))
        plt.close()
        
        # 测试批量随机移位
        batch_size = 3
        batch_tensor = np.zeros((batch_size, 5, 10))
        for i in range(batch_size):
            for j in range(10):
                batch_tensor[i, :, j] = j + i * 10
        
        # 应用批量随机移位
        shifted_batch, shifts = CircularConvDataAugmentation.batch_random_shift(batch_tensor)
        
        # 验证每个样本的移位是否正确
        batch_diff = 0
        for i in range(batch_size):
            manually_shifted = np.roll(batch_tensor[i], shifts[i], axis=1)
            diff = np.abs(shifted_batch[i] - manually_shifted).max()
            batch_diff = max(batch_diff, diff)
        
        print(f"批量移位量: {shifts}")
        print(f"批量随机移位的最大差异: {batch_diff}")
        
        results["批量移位"] = {
            "移位量": shifts.tolist(),
            "最大差异": float(batch_diff),
            "是否正确": batch_diff < 1e-6
        }
        
        # 保存结果
        self._save_results("data_augmentation", results)
        
        return results
    
    def test_model_comparison(self):
        """
        比较SCRingCNN和SCRingCNNLite的性能
        """
        print("\n===== 比较模型性能 =====")
        
        # 设置参数
        num_rings = 20
        num_sectors = 60
        descriptor_dim = 256
        batch_size = 8
        
        # 创建输入
        x = torch.randn(batch_size, 1, num_rings, num_sectors).to(self.device)
        
        # 测试SCRingCNN
        model1 = SCRingCNN(
            in_channels=1,
            num_rings=num_rings,
            num_sectors=num_sectors,
            descriptor_dim=descriptor_dim
        ).to(self.device)
        
        # 测试SCRingCNNLite
        model2 = SCRingCNNLite(
            in_channels=1,
            num_rings=num_rings,
            num_sectors=num_sectors,
            descriptor_dim=descriptor_dim
        ).to(self.device)
        
        # 计算参数数量
        num_params1 = sum(p.numel() for p in model1.parameters())
        num_params2 = sum(p.numel() for p in model2.parameters())
        
        print(f"SCRingCNN参数数量: {num_params1}")
        print(f"SCRingCNNLite参数数量: {num_params2}")
        print(f"参数减少: {(1 - num_params2 / num_params1) * 100:.2f}%")
        
        # 测试推理时间
        num_runs = 100
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = model1(x)
                _ = model2(x)
        
        # 测量SCRingCNN推理时间
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model1(x)
        time1 = (time.time() - start_time) / num_runs
        
        # 测量SCRingCNNLite推理时间
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model2(x)
        time2 = (time.time() - start_time) / num_runs
        
        print(f"SCRingCNN平均推理时间: {time1 * 1000:.2f} ms")
        print(f"SCRingCNNLite平均推理时间: {time2 * 1000:.2f} ms")
        print(f"速度提升: {(time1 / time2):.2f}x")
        
        # 比较描述子质量
        with torch.no_grad():
            desc1 = model1(x)
            desc2 = model2(x)
        
        # 计算描述子之间的余弦相似度
        norm_desc1 = F.normalize(desc1, p=2, dim=1)
        norm_desc2 = F.normalize(desc2, p=2, dim=1)
        
        similarity_matrix = torch.mm(norm_desc1, norm_desc2.t())
        diagonal_sim = torch.diag(similarity_matrix)
        
        mean_sim = diagonal_sim.mean().item()
        min_sim = diagonal_sim.min().item()
        max_sim = diagonal_sim.max().item()
        
        print(f"描述子平均相似度: {mean_sim:.4f}")
        print(f"描述子最小相似度: {min_sim:.4f}")
        print(f"描述子最大相似度: {max_sim:.4f}")
        
        results = {
            "SCRingCNN": {
                "参数数量": int(num_params1),
                "推理时间(ms)": float(time1 * 1000)
            },
            "SCRingCNNLite": {
                "参数数量": int(num_params2),
                "推理时间(ms)": float(time2 * 1000)
            },
            "比较": {
                "参数减少(%)": float((1 - num_params2 / num_params1) * 100),
                "速度提升(x)": float(time1 / time2)
            },
            "描述子相似度": {
                "平均": float(mean_sim),
                "最小": float(min_sim),
                "最大": float(max_sim)
            }
        }
        
        # 保存结果
        self._save_results("model_comparison", results)
        
        return results
    
    def test_circular_conv_rotation_invariance(self):
        """
        专门测试环形卷积的旋转不变性
        """
        print("\n===== 测试环形卷积旋转不变性 =====")
        
        # 创建一个具有明显模式的测试张量
        test_tensor = torch.zeros(1, 1, 20, 60)
        # 在中心位置创建一个高斯形状
        for i in range(20):
            for j in range(60):
                r = ((i - 10) ** 2 + (j - 30) ** 2) ** 0.5
                test_tensor[0, 0, i, j] = torch.exp(-r / 5)
        
        # 创建环形卷积层
        conv = CircularPadConv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        ).to(self.device)
        
        # 应用卷积
        test_tensor = test_tensor.to(self.device)
        output1 = conv(test_tensor)
        
        # 创建旋转版本
        shifts = [10, 20, 30]
        results = {}
        
        for shift in shifts:
            # 循环移位输入
            shifted_tensor = torch.roll(test_tensor, shifts=shift, dims=3)
            
            # 应用卷积
            output2 = conv(shifted_tensor)
            
            # 将输出2循环移位回来进行比较
            output2_shifted_back = torch.roll(output2, shifts=-shift, dims=3)
            
            # 计算差异
            diff = torch.abs(output1 - output2_shifted_back).max().item()
            
            results[f"移位{shift}"] = {
                "最大差异": float(diff),
                "是否等价": diff < 1e-5
            }
            
            print(f"移位 {shift}: 最大差异 = {diff:.8f}")
            
            # 可视化
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(test_tensor[0, 0].cpu().detach().numpy())
            plt.title("原始输入")
            plt.colorbar()
            
            plt.subplot(1, 3, 2)
            plt.imshow(shifted_tensor[0, 0].cpu().detach().numpy())
            plt.title(f"移位{shift}后的输入")
            plt.colorbar()
            
            plt.subplot(1, 3, 3)
            plt.imshow(torch.abs(output1[0, 0] - output2_shifted_back[0, 0]).cpu().detach().numpy())
            plt.title("输出差异")
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"circular_conv_rotation_invariance_{shift}.png"))
            plt.close()
        
        # 保存结果
        self._save_results("circular_conv_rotation_invariance", results)
        
        return results
    
    def run_all_tests(self):
        """
        运行所有测试
        """
        print("开始运行所有鲁棒性测试...")
        
        # 记录开始时间
        start_time = time.time()
        
        # 运行所有测试
        results = {
            "input_size_robustness": self.test_input_size_robustness(),
            "rotation_invariance": self.test_rotation_invariance(),
            "noise_robustness": self.test_noise_robustness(),
            "edge_cases": self.test_edge_cases(),
            "numerical_stability": self.test_numerical_stability(),
            "circular_padding": self.test_circular_padding(),
            "circular_conv_rotation_invariance": self.test_circular_conv_rotation_invariance(),
            "data_augmentation": self.test_data_augmentation(),
            "model_comparison": self.test_model_comparison()
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
    tester = RobustnessTest(output_dir="robustness_results")
    
    # 运行所有测试
    tester.run_all_tests()