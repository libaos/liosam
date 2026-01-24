#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于ScanContext + CNN的轨迹定位系统
目标：识别机器人在轨迹中的具体位置
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.cnn_2d_models import Simple2DCNN, Enhanced2DCNN, ResNet2D
from utils.scan_context import ScanContext
from utils.ply_reader import PLYReader
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle
import time

class TrajectoryLocalizationSystem:
    """优化的轨迹定位系统"""

    def __init__(self, num_locations=20, model_type='simple2dcnn', adaptive_segments=True):
        self.num_locations = num_locations
        self.model_type = model_type
        self.adaptive_segments = adaptive_segments
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sc_generator = ScanContext()

        print(f"🎯 优化的轨迹定位系统")
        print(f"设备: {self.device}")
        print(f"目标位置数: {num_locations}")
        print(f"模型类型: {model_type}")
        print(f"自适应分段: {adaptive_segments}")
        print(f"目标: 基于ScanContext特征进行精确轨迹定位")

        # 初始化模型
        if model_type == 'simple2dcnn':
            self.model = Simple2DCNN(num_classes=num_locations)
        elif model_type == 'enhanced2dcnn':
            self.model = Enhanced2DCNN(num_classes=num_locations)
        elif model_type == 'resnet2d':
            self.model = ResNet2D(num_classes=num_locations)
        else:
            raise ValueError(f"未知模型类型: {model_type}")

        self.model = self.model.to(self.device)

        # 位置信息存储
        self.location_database = {}  # 存储每个位置的代表性ScanContext
        self.location_features = []  # 所有位置的特征
        self.location_labels = []    # 对应的位置标签

        # 优化参数
        self.confidence_threshold = 0.7  # 置信度阈值
        self.temporal_smoothing = True   # 时序平滑
        self.location_history = []       # 位置历史
        self.confidence_history = []     # 置信度历史
        
    def create_adaptive_location_database(self, data_dir, save_path='location_database.pkl'):
        """创建自适应位置数据库（基于场景变化）"""
        print(f"📍 创建自适应轨迹位置数据库...")

        if self.adaptive_segments:
            # 使用场景变化检测来确定分段
            from scene_change_detector import SceneChangeDetector
            detector = SceneChangeDetector(similarity_threshold=0.75, min_segment_length=15)

            # 计算场景特征
            features, valid_indices = detector.compute_scene_features(data_dir)
            if len(features) == 0:
                print("❌ 无法提取场景特征，回退到均匀分段")
                return self.create_uniform_location_database(data_dir, save_path)

            # 检测场景变化
            similarities, change_points, gradient = detector.detect_scene_changes(features)
            segments = detector.create_segments(change_points, len(features))

            print(f"🎯 基于场景变化检测到 {len(segments)} 个自然分段")

            # 如果分段数量与目标不符，调整
            if len(segments) != self.num_locations:
                print(f"⚠️  分段数量({len(segments)})与目标({self.num_locations})不符，进行调整")
                segments = self.adjust_segments(segments, self.num_locations)

            return self.create_database_from_segments(data_dir, segments, save_path)
        else:
            return self.create_uniform_location_database(data_dir, save_path)

    def create_uniform_location_database(self, data_dir, save_path='location_database.pkl'):
        """创建均匀分段的位置数据库"""
        print(f"📍 创建均匀分段位置数据库...")

        # 获取所有ply文件
        ply_files = sorted(glob.glob(f"{data_dir}/*.ply"))
        print(f"找到 {len(ply_files)} 个ply文件")

        if len(ply_files) == 0:
            print("❌ 未找到ply文件")
            return False

        # 计算每个位置段的文件范围
        files_per_location = len(ply_files) // self.num_locations
        print(f"每个位置段包含约 {files_per_location} 个文件")

        location_data = {}
        all_features = []
        all_labels = []

        for location_id in range(self.num_locations):
            print(f"  处理位置 {location_id+1}/{self.num_locations}")

            # 确定这个位置的文件范围
            start_idx = location_id * files_per_location
            if location_id == self.num_locations - 1:
                end_idx = len(ply_files)  # 最后一个位置包含剩余所有文件
            else:
                end_idx = (location_id + 1) * files_per_location

            location_files = ply_files[start_idx:end_idx]
            location_features = []

            # 处理这个位置的所有文件
            for ply_file in location_files:
                try:
                    points = PLYReader.read_ply_file(ply_file)
                    if points is not None and len(points) > 100:
                        points = points[:, :3]
                        sc = self.sc_generator.generate_scan_context(points)

                        if sc is not None:
                            location_features.append(sc)
                            all_features.append(sc)
                            all_labels.append(location_id)

                except Exception as e:
                    print(f"    处理失败 {ply_file}: {e}")
                    continue

            if len(location_features) > 0:
                # 计算这个位置的代表性特征（平均值）
                representative_sc = np.mean(location_features, axis=0)
                location_data[location_id] = {
                    'representative_sc': representative_sc,
                    'sample_count': len(location_features),
                    'file_range': (start_idx, end_idx),
                    'all_features': location_features
                }
                print(f"    位置 {location_id}: {len(location_features)} 个有效样本")
            else:
                print(f"    ⚠️  位置 {location_id}: 无有效样本")

        self.location_database = location_data
        self.location_features = np.array(all_features)
        self.location_labels = np.array(all_labels)

        # 保存位置数据库
        with open(save_path, 'wb') as f:
            pickle.dump({
                'location_database': self.location_database,
                'location_features': self.location_features,
                'location_labels': self.location_labels,
                'num_locations': self.num_locations,
                'adaptive_segments': False
            }, f)

        print(f"✅ 位置数据库已保存到: {save_path}")
        print(f"总样本数: {len(all_features)}")
        print(f"位置分布: {np.bincount(all_labels)}")

        return True

    def create_location_database(self, data_dir, save_path='location_database.pkl'):
        """创建位置数据库（兼容旧接口）"""
        return self.create_adaptive_location_database(data_dir, save_path)
    
    def load_location_database(self, load_path='location_database.pkl'):
        """加载位置数据库"""
        if not Path(load_path).exists():
            print(f"❌ 位置数据库文件不存在: {load_path}")
            return False
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.location_database = data['location_database']
        self.location_features = data['location_features']
        self.location_labels = data['location_labels']
        self.num_locations = data['num_locations']
        
        print(f"✅ 已加载位置数据库")
        print(f"位置数量: {self.num_locations}")
        print(f"总样本数: {len(self.location_features)}")
        
        return True
    
    def train_localization_model(self, epochs=50, batch_size=32):
        """训练定位模型"""
        print(f"\n🎯 开始训练轨迹定位模型...")
        
        if len(self.location_features) == 0:
            print("❌ 没有训练数据，请先创建位置数据库")
            return False
        
        # 数据划分 - 处理样本不足的类别
        print("🔍 检查数据分布...")
        unique_labels, counts = np.unique(self.location_labels, return_counts=True)
        min_samples = np.min(counts)

        if min_samples < 3:
            print(f"⚠️  检测到样本不足的类别 (最少{min_samples}个样本)")
            print("使用随机划分而不是分层划分")

            # 使用随机划分
            X_train, X_temp, y_train, y_temp = train_test_split(
                self.location_features, self.location_labels,
                test_size=0.4, random_state=42
            )

            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )
        else:
            print("✅ 所有类别样本充足，使用分层划分")
            # 使用分层划分
            X_train, X_temp, y_train, y_temp = train_test_split(
                self.location_features, self.location_labels,
                test_size=0.4, random_state=42, stratify=self.location_labels
            )

            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
        
        print(f"训练集: {len(X_train)} 样本")
        print(f"验证集: {len(X_val)} 样本")
        print(f"测试集: {len(X_test)} 样本")
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(self.device)
        
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 训练设置
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        
        # 训练循环
        best_val_acc = 0
        best_model_state = None
        patience = 10
        patience_counter = 0
        
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            
            # 验证
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    output = self.model(data)
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            val_acc = 100. * val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_accuracies.append(val_acc)
            
            print(f'Epoch {epoch+1:2d}/{epochs:2d} | '
                  f'训练损失: {avg_train_loss:.4f} | '
                  f'验证准确率: {val_acc:.1f}%')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f'🎉 新的最佳验证准确率: {best_val_acc:.1f}%')
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                print(f'⏹️  早停触发')
                break
            
            scheduler.step()
        
        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # 测试
        self.model.eval()
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                test_predictions.extend(predicted.cpu().numpy())
                test_targets.extend(target.cpu().numpy())
        
        test_acc = accuracy_score(test_targets, test_predictions) * 100
        
        print(f"\n✅ 训练完成!")
        print(f"最佳验证准确率: {best_val_acc:.1f}%")
        print(f"测试准确率: {test_acc:.1f}%")
        
        # 详细分析
        print(f"\n📊 详细定位性能分析:")
        
        # 计算每个位置的准确率
        location_accuracies = {}
        for location_id in range(self.num_locations):
            location_mask = np.array(test_targets) == location_id
            if np.sum(location_mask) > 0:
                location_predictions = np.array(test_predictions)[location_mask]
                location_targets = np.array(test_targets)[location_mask]
                location_acc = accuracy_score(location_targets, location_predictions) * 100
                location_accuracies[location_id] = location_acc
                sample_count = np.sum(location_mask)
                print(f"  位置 {location_id:2d}: {location_acc:6.1f}% ({sample_count:2d} 样本)")
        
        avg_location_acc = np.mean(list(location_accuracies.values()))
        print(f"\n平均位置准确率: {avg_location_acc:.1f}%")

        # --- BEGIN MULTI-METRIC CALCULATION ---
        print("\n📊 详细多维度指标分析 (独立测试集):")
        errors = np.abs(np.array(test_predictions) - np.array(test_targets))
        acc_err1 = np.mean(errors <= 1) * 100
        acc_err2 = np.mean(errors <= 2) * 100
        
        try:
            report = classification_report(test_targets, test_predictions, output_dict=True, zero_division=0)
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1 = report['weighted avg']['f1-score']
        except Exception:
            # Fallback for simpler sklearn versions
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(test_targets, test_predictions, average='weighted', zero_division=0)
            recall = recall_score(test_targets, test_predictions, average='weighted', zero_division=0)
            f1 = f1_score(test_targets, test_predictions, average='weighted', zero_division=0)

        print(f"   - 误差≤1 准确率          : {acc_err1:.1f}%")
        print(f"   - 误差≤2 准确率          : {acc_err2:.1f}%")
        print(f"   - 加权精确率 (Precision) : {precision:.3f}")
        print(f"   - 加权召回率 (Recall)    : {recall:.3f}")
        print(f"   - 加权F1分数 (F1-Score)   : {f1:.3f}")
        # --- END MULTI-METRIC CALCULATION ---
        
        # 保存模型
        model_path = f"models/saved/trajectory_localizer_{self.model_type}_acc{test_acc:.1f}.pth"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'num_locations': self.num_locations,
            'test_accuracy': test_acc,
            'best_val_accuracy': best_val_acc,
            'location_accuracies': location_accuracies,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        }, model_path)
        
        print(f"✅ 定位模型已保存到: {model_path}")
        
        return test_acc
    
    def localize_position(self, scan_context):
        """优化的位置定位（包含时序平滑和置信度过滤）"""
        if scan_context is None:
            return None, 0.0

        try:
            self.model.eval()
            sc_tensor = torch.FloatTensor(scan_context).unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(sc_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_location = torch.max(probabilities, 1)

                predicted_location = predicted_location.item()
                confidence = confidence.item()

                # 置信度过滤
                if confidence < self.confidence_threshold:
                    # 低置信度时，尝试使用历史信息
                    if len(self.location_history) > 0:
                        # 使用最近的高置信度位置
                        recent_high_conf = [i for i, c in enumerate(self.confidence_history[-5:]) if c >= self.confidence_threshold]
                        if recent_high_conf:
                            last_reliable_idx = recent_high_conf[-1]
                            predicted_location = self.location_history[-(5-last_reliable_idx)]
                            confidence = 0.5  # 标记为中等置信度

                # 时序平滑
                if self.temporal_smoothing and len(self.location_history) > 0:
                    predicted_location = self.apply_temporal_smoothing(predicted_location, confidence)

                # 更新历史
                self.location_history.append(predicted_location)
                self.confidence_history.append(confidence)

                # 保持历史长度
                if len(self.location_history) > 10:
                    self.location_history.pop(0)
                    self.confidence_history.pop(0)

                return predicted_location, confidence

        except Exception as e:
            print(f"定位失败: {e}")
            return None, 0.0

    def apply_temporal_smoothing(self, current_prediction, current_confidence):
        """应用时序平滑"""
        if len(self.location_history) < 2:
            return current_prediction

        # 获取最近的位置
        recent_locations = self.location_history[-3:]
        recent_confidences = self.confidence_history[-3:]

        # 如果当前预测与最近位置差异很大，且置信度不高，则进行平滑
        last_location = self.location_history[-1]
        location_diff = abs(current_prediction - last_location)

        if location_diff > 3 and current_confidence < 0.9:
            # 计算加权平均
            weights = np.array(recent_confidences + [current_confidence])
            locations = np.array(recent_locations + [current_prediction])

            weighted_location = np.average(locations, weights=weights)
            smoothed_location = int(round(weighted_location))

            # 确保在有效范围内
            smoothed_location = max(0, min(self.num_locations - 1, smoothed_location))

            return smoothed_location

        return current_prediction
    
    def load_trained_model(self, model_path):
        """加载训练好的模型"""
        if not Path(model_path).exists():
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.num_locations = checkpoint['num_locations']
            
            print(f"✅ 已加载训练好的定位模型")
            print(f"模型准确率: {checkpoint.get('test_accuracy', 'N/A'):.1f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return False

    def adjust_segments(self, segments, target_num):
        """调整分段数量以匹配目标"""
        if len(segments) == target_num:
            return segments

        if len(segments) < target_num:
            # 分段太少，需要细分
            return self.split_segments(segments, target_num)
        else:
            # 分段太多，需要合并
            return self.merge_segments(segments, target_num)

    def split_segments(self, segments, target_num):
        """细分分段"""
        print(f"🔄 细分 {len(segments)} 个分段到 {target_num} 个")

        # 找到最长的分段进行细分
        new_segments = list(segments)

        while len(new_segments) < target_num:
            # 找到最长的分段
            lengths = [(end - start, i) for i, (start, end) in enumerate(new_segments)]
            max_length, max_idx = max(lengths)

            if max_length <= 2:  # 如果最长分段也很短，停止细分
                break

            # 细分最长的分段
            start, end = new_segments[max_idx]
            mid = (start + end) // 2
            new_segments[max_idx] = (start, mid)
            new_segments.insert(max_idx + 1, (mid, end))

        return new_segments

    def merge_segments(self, segments, target_num):
        """合并分段"""
        print(f"🔄 合并 {len(segments)} 个分段到 {target_num} 个")

        new_segments = list(segments)

        while len(new_segments) > target_num:
            # 找到最短的相邻分段对进行合并
            min_combined_length = float('inf')
            merge_idx = 0

            for i in range(len(new_segments) - 1):
                start1, end1 = new_segments[i]
                start2, end2 = new_segments[i + 1]
                combined_length = (end1 - start1) + (end2 - start2)

                if combined_length < min_combined_length:
                    min_combined_length = combined_length
                    merge_idx = i

            # 合并选中的分段
            start1, end1 = new_segments[merge_idx]
            start2, end2 = new_segments[merge_idx + 1]
            new_segments[merge_idx] = (start1, end2)
            new_segments.pop(merge_idx + 1)

        return new_segments

    def create_database_from_segments(self, data_dir, segments, save_path):
        """根据分段创建位置数据库"""
        print(f"📍 根据 {len(segments)} 个分段创建位置数据库...")

        ply_files = sorted(glob.glob(f"{data_dir}/*.ply"))
        if len(ply_files) == 0:
            print("❌ 未找到ply文件")
            return False

        location_data = {}
        all_features = []
        all_labels = []

        for location_id, (start_idx, end_idx) in enumerate(segments):
            print(f"  处理位置 {location_id+1}/{len(segments)} (帧 {start_idx}-{end_idx})")

            # 确保索引在有效范围内
            start_idx = max(0, min(start_idx, len(ply_files) - 1))
            end_idx = max(start_idx + 1, min(end_idx, len(ply_files)))

            location_files = ply_files[start_idx:end_idx]
            location_features = []

            # 处理这个位置的所有文件
            for ply_file in location_files:
                try:
                    points = PLYReader.read_ply_file(ply_file)
                    if points is not None and len(points) > 100:
                        points = points[:, :3]
                        sc = self.sc_generator.generate_scan_context(points)

                        if sc is not None:
                            location_features.append(sc)
                            all_features.append(sc)
                            all_labels.append(location_id)

                except Exception as e:
                    print(f"    处理失败 {ply_file}: {e}")
                    continue

            if len(location_features) > 0:
                # 计算这个位置的代表性特征
                representative_sc = np.mean(location_features, axis=0)
                location_data[location_id] = {
                    'representative_sc': representative_sc,
                    'sample_count': len(location_features),
                    'file_range': (start_idx, end_idx),
                    'all_features': location_features
                }
                print(f"    位置 {location_id}: {len(location_features)} 个有效样本")
            else:
                print(f"    ⚠️  位置 {location_id}: 无有效样本")

        self.location_database = location_data
        self.location_features = np.array(all_features)
        self.location_labels = np.array(all_labels)

        # 更新位置数量
        self.num_locations = len(segments)

        # 保存位置数据库
        with open(save_path, 'wb') as f:
            pickle.dump({
                'location_database': self.location_database,
                'location_features': self.location_features,
                'location_labels': self.location_labels,
                'num_locations': self.num_locations,
                'adaptive_segments': True,
                'segments': segments
            }, f)

        print(f"✅ 自适应位置数据库已保存到: {save_path}")
        print(f"总样本数: {len(all_features)}")
        print(f"实际位置数: {self.num_locations}")

        return True

def main():
    """主函数"""
    print("="*60)
    print("🎯 优化的基于ScanContext + CNN的轨迹定位系统")
    print("="*60)

    # 数据路径
    data_dir = "/mysda/w/w/RandLA-Net-pytorch/回环检测/ply_files"

    # 创建优化的定位系统
    localizer = TrajectoryLocalizationSystem(
        num_locations=8,
        model_type='resnet2d',
        adaptive_segments=False  # 强制均匀分段
    )

    # 1. 创建自适应位置数据库
    print("\n步骤1: 创建自适应位置数据库")
    success = localizer.create_location_database(data_dir)

    if not success:
        print("❌ 位置数据库创建失败")
        return

    # 2. 训练定位模型
    print("\n步骤2: 训练优化的定位模型")
    test_acc = localizer.train_localization_model(epochs=50)

    print(f"\n🎉 优化的轨迹定位系统训练完成!")
    print(f"定位准确率: {test_acc:.1f}%")
    print(f"系统可以识别轨迹中的 {localizer.num_locations} 个不同位置")
    print(f"✨ 优化特性:")
    print(f"  - 自适应场景分段")
    print(f"  - 时序平滑定位")
    print(f"  - 置信度过滤")
    print(f"  - 历史信息利用")
    print(f"下次机器人来到相同区域时，可以更准确、更稳定地定位其位置！")

if __name__ == '__main__':
    main()
