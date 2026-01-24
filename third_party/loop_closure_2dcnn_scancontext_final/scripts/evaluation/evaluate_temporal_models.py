#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, top_k_accuracy_score
from sklearn.manifold import TSNE
import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.temporal_dataset import TemporalScanContextDataset, create_temporal_dataloaders
from models.temporal_2d_cnn import Temporal2DCNN, Temporal2DCNNLite, Temporal2DCNNResNet
from models.temporal_3d_cnn import Temporal3DCNN, Temporal3DCNNLite, Temporal3DCNNDeep
from utils.logger import Logger

class TemporalModelEvaluator:
    """时序模型评估器"""
    
    def __init__(self, config, checkpoint_path):
        """
        初始化评估器
        
        参数:
            config (dict): 模型配置
            checkpoint_path (str): 模型检查点路径
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path(config.get('output_dir', 'outputs/evaluation'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志
        self.logger = Logger(self.output_dir / 'evaluation.log')
        
        # 加载模型
        self.model = self._load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 初始化数据加载器
        self.train_loader, self.val_loader, self.test_loader = self._create_dataloaders()
        
        # 类别名称
        self.class_names = [f'Path_{i:02d}' for i in range(config['data']['num_classes'])]
    
    def _load_model(self, checkpoint_path):
        """加载模型"""
        # 创建模型
        model_type = self.config['model']['type']
        model_params = self.config['model']['params']
        
        if model_type == 'temporal_2d_cnn':
            model = Temporal2DCNN(**model_params)
        elif model_type == 'temporal_2d_cnn_lite':
            model = Temporal2DCNNLite(**model_params)
        elif model_type == 'temporal_2d_cnn_resnet':
            model = Temporal2DCNNResNet(**model_params)
        elif model_type == 'temporal_3d_cnn':
            model = Temporal3DCNN(**model_params)
        elif model_type == 'temporal_3d_cnn_lite':
            model = Temporal3DCNNLite(**model_params)
        elif model_type == 'temporal_3d_cnn_deep':
            model = Temporal3DCNNDeep(**model_params)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.logger.info(f"成功加载模型: {model_type}")
        self.logger.info(f"检查点路径: {checkpoint_path}")
        
        return model
    
    def _create_dataloaders(self):
        """创建数据加载器"""
        data_config = self.config['data']
        
        return create_temporal_dataloaders(
            data_dir=data_config['data_dir'],
            sequence_length=data_config['sequence_length'],
            batch_size=data_config.get('batch_size', 32),
            num_workers=data_config.get('num_workers', 4),
            cache_dir=data_config.get('cache_dir'),
            num_classes=data_config.get('num_classes', 20)
        )
    
    def evaluate_dataset(self, dataloader, dataset_name='Test'):
        """评估数据集"""
        self.logger.info(f"评估 {dataset_name} 数据集...")
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                # 测量推理时间
                start_time = time.time()
                output = self.model(data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # 获取预测结果
                probabilities = torch.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    self.logger.info(f"已处理 {batch_idx + 1}/{len(dataloader)} 个批次")
        
        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # 计算评估指标
        metrics = self._calculate_metrics(all_targets, all_predictions, all_probabilities)
        metrics['avg_inference_time'] = np.mean(inference_times)
        metrics['fps'] = len(all_predictions) / sum(inference_times)
        
        # 打印结果
        self._print_metrics(metrics, dataset_name)
        
        # 保存详细结果
        results = {
            'predictions': all_predictions.tolist(),
            'targets': all_targets.tolist(),
            'probabilities': all_probabilities.tolist(),
            'metrics': metrics
        }
        
        with open(self.output_dir / f'{dataset_name.lower()}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _calculate_metrics(self, targets, predictions, probabilities):
        """计算评估指标"""
        # 基本准确率
        accuracy = np.mean(targets == predictions)
        
        # Top-k准确率
        top1_acc = top_k_accuracy_score(targets, probabilities, k=1)
        top3_acc = top_k_accuracy_score(targets, probabilities, k=3)
        top5_acc = top_k_accuracy_score(targets, probabilities, k=5)
        
        # 每类准确率和召回率
        report = classification_report(targets, predictions, target_names=self.class_names, output_dict=True)
        
        # 混淆矩阵
        cm = confusion_matrix(targets, predictions)
        
        metrics = {
            'accuracy': float(accuracy),
            'top1_accuracy': float(top1_acc),
            'top3_accuracy': float(top3_acc),
            'top5_accuracy': float(top5_acc),
            'macro_precision': float(report['macro avg']['precision']),
            'macro_recall': float(report['macro avg']['recall']),
            'macro_f1': float(report['macro avg']['f1-score']),
            'weighted_precision': float(report['weighted avg']['precision']),
            'weighted_recall': float(report['weighted avg']['recall']),
            'weighted_f1': float(report['weighted avg']['f1-score']),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        return metrics
    
    def _print_metrics(self, metrics, dataset_name):
        """打印评估指标"""
        self.logger.info(f"\n{dataset_name} 数据集评估结果:")
        self.logger.info(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        self.logger.info(f"Top-1 准确率: {metrics['top1_accuracy']:.4f}")
        self.logger.info(f"Top-3 准确率: {metrics['top3_accuracy']:.4f}")
        self.logger.info(f"Top-5 准确率: {metrics['top5_accuracy']:.4f}")
        self.logger.info(f"宏平均精确率: {metrics['macro_precision']:.4f}")
        self.logger.info(f"宏平均召回率: {metrics['macro_recall']:.4f}")
        self.logger.info(f"宏平均F1分数: {metrics['macro_f1']:.4f}")
        self.logger.info(f"平均推理时间: {metrics['avg_inference_time']:.4f}s")
        self.logger.info(f"推理速度: {metrics['fps']:.2f} FPS")
    
    def plot_confusion_matrix(self, results, dataset_name='Test'):
        """绘制混淆矩阵"""
        cm = np.array(results['metrics']['confusion_matrix'])
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'{dataset_name} Dataset - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{dataset_name.lower()}_confusion_matrix.png', dpi=300)
        plt.close()
        
        self.logger.info(f"混淆矩阵已保存: {dataset_name.lower()}_confusion_matrix.png")
    
    def plot_class_performance(self, results, dataset_name='Test'):
        """绘制各类别性能"""
        report = results['metrics']['classification_report']
        
        # 提取各类别指标
        classes = []
        precision = []
        recall = []
        f1_score = []
        
        for class_name in self.class_names:
            if class_name in report:
                classes.append(class_name)
                precision.append(report[class_name]['precision'])
                recall.append(report[class_name]['recall'])
                f1_score.append(report[class_name]['f1-score'])
        
        # 绘制柱状图
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title(f'{dataset_name} Dataset - Per-Class Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{dataset_name.lower()}_class_performance.png', dpi=300)
        plt.close()
        
        self.logger.info(f"类别性能图已保存: {dataset_name.lower()}_class_performance.png")
    
    def extract_features(self, dataloader, max_samples=1000):
        """提取特征用于可视化"""
        self.logger.info("提取特征用于t-SNE可视化...")
        
        features = []
        labels = []
        sample_count = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                if sample_count >= max_samples:
                    break
                
                data = data.to(self.device)
                
                # 获取特征（在最后一层全连接之前）
                if hasattr(self.model, 'features'):
                    feat = self.model.features(data)
                    feat = feat.view(feat.size(0), -1)
                else:
                    # 对于没有features属性的模型，使用完整前向传播
                    feat = self.model(data)
                
                features.extend(feat.cpu().numpy())
                labels.extend(target.numpy())
                sample_count += len(data)
        
        return np.array(features), np.array(labels)
    
    def plot_tsne(self, max_samples=1000):
        """绘制t-SNE可视化"""
        features, labels = self.extract_features(self.test_loader, max_samples)
        
        self.logger.info("计算t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)
        
        # 绘制t-SNE图
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab20', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of Learned Features')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tsne_visualization.png', dpi=300)
        plt.close()
        
        self.logger.info("t-SNE可视化已保存: tsne_visualization.png")
    
    def run_full_evaluation(self):
        """运行完整评估"""
        self.logger.info("开始完整评估...")
        
        # 评估测试集
        test_results = self.evaluate_dataset(self.test_loader, 'Test')
        
        # 评估验证集
        val_results = self.evaluate_dataset(self.val_loader, 'Validation')
        
        # 绘制可视化图表
        self.plot_confusion_matrix(test_results, 'Test')
        self.plot_class_performance(test_results, 'Test')
        
        # t-SNE可视化
        try:
            self.plot_tsne()
        except Exception as e:
            self.logger.warning(f"t-SNE可视化失败: {e}")
        
        # 保存综合报告
        summary = {
            'model_type': self.config['model']['type'],
            'test_metrics': test_results['metrics'],
            'validation_metrics': val_results['metrics']
        }
        
        with open(self.output_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info("评估完成！")
        return summary


def main():
    parser = argparse.ArgumentParser(description='评估时序模型')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation', help='输出目录')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # 尝试从检查点加载配置
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            raise ValueError("未找到配置文件，请提供 --config 参数")
    
    # 设置输出目录
    config['output_dir'] = args.output_dir
    
    # 创建评估器并运行评估
    evaluator = TemporalModelEvaluator(config, args.checkpoint)
    evaluator.run_full_evaluation()


if __name__ == '__main__':
    main()
