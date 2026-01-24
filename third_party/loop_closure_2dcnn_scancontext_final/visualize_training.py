#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练过程可视化脚本
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_training_history(model_name):
    """加载训练历史"""
    history_file = Path(f"outputs/{model_name}_training/training_history.json")
    if history_file.exists():
        with open(history_file, 'r') as f:
            return json.load(f)
    else:
        print(f"未找到训练历史文件: {history_file}")
        return None

def plot_training_curves():
    """绘制训练曲线"""
    # 加载训练历史
    history_3d = load_training_history("temporal_3d_cnn")
    history_2d = load_training_history("temporal_2d_cnn")
    
    if not history_3d or not history_2d:
        print("无法加载训练历史")
        return
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 提取数据
    epochs_3d = [item['epoch'] for item in history_3d['train']]
    train_loss_3d = [item['loss'] for item in history_3d['train']]
    train_acc_3d = [item['acc'] for item in history_3d['train']]
    val_loss_3d = [item['loss'] for item in history_3d['val']]
    val_acc_3d = [item['acc'] for item in history_3d['val']]
    
    epochs_2d = [item['epoch'] for item in history_2d['train']]
    train_loss_2d = [item['loss'] for item in history_2d['train']]
    train_acc_2d = [item['acc'] for item in history_2d['train']]
    val_loss_2d = [item['loss'] for item in history_2d['val']]
    val_acc_2d = [item['acc'] for item in history_2d['val']]
    
    # 绘制训练损失
    ax1.plot(epochs_3d, train_loss_3d, 'b-', label='3D CNN Train', linewidth=2)
    ax1.plot(epochs_2d, train_loss_2d, 'r-', label='2D CNN Train', linewidth=2)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制验证损失
    ax2.plot(epochs_3d, val_loss_3d, 'b--', label='3D CNN Val', linewidth=2)
    ax2.plot(epochs_2d, val_loss_2d, 'r--', label='2D CNN Val', linewidth=2)
    ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 绘制训练准确率
    ax3.plot(epochs_3d, train_acc_3d, 'b-', label='3D CNN Train', linewidth=2)
    ax3.plot(epochs_2d, train_acc_2d, 'r-', label='2D CNN Train', linewidth=2)
    ax3.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 绘制验证准确率
    ax4.plot(epochs_3d, val_acc_3d, 'b--', label='3D CNN Val', linewidth=2)
    ax4.plot(epochs_2d, val_acc_2d, 'r--', label='2D CNN Val', linewidth=2)
    ax4.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/training_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("训练曲线已保存到: outputs/training_curves_comparison.png")

def print_training_summary():
    """打印训练过程总结"""
    print("=" * 80)
    print("训练过程详细总结")
    print("=" * 80)
    
    # 加载训练历史
    history_3d = load_training_history("temporal_3d_cnn")
    history_2d = load_training_history("temporal_2d_cnn")
    
    if not history_3d or not history_2d:
        return
    
    print("\n🔥 3D CNN 训练过程:")
    print("-" * 50)
    print(f"总训练轮数: {len(history_3d['train'])} epochs")
    print(f"最佳验证准确率: {history_3d['best_val_acc']:.2f}%")
    
    # 显示关键训练节点
    key_epochs_3d = [0, 9, 19, 29, 39, 49]  # 每10轮显示一次
    print("\n关键训练节点:")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
    print("-" * 60)
    
    for epoch in key_epochs_3d:
        if epoch < len(history_3d['train']):
            train_data = history_3d['train'][epoch]
            val_data = history_3d['val'][epoch]
            print(f"{epoch:<8} {train_data['loss']:<12.4f} {train_data['acc']:<12.2f} "
                  f"{val_data['loss']:<12.4f} {val_data['acc']:<12.2f}")
    
    print("\n🔥 2D CNN 训练过程:")
    print("-" * 50)
    print(f"总训练轮数: {len(history_2d['train'])} epochs")
    print(f"最佳验证准确率: {history_2d['best_val_acc']:.2f}%")
    
    # 显示关键训练节点
    key_epochs_2d = [0, 9, 19, 29, 39, 49]
    print("\n关键训练节点:")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
    print("-" * 60)
    
    for epoch in key_epochs_2d:
        if epoch < len(history_2d['train']):
            train_data = history_2d['train'][epoch]
            val_data = history_2d['val'][epoch]
            print(f"{epoch:<8} {train_data['loss']:<12.4f} {train_data['acc']:<12.2f} "
                  f"{val_data['loss']:<12.4f} {val_data['acc']:<12.2f}")
    
    # 训练过程分析
    print("\n📊 训练过程分析:")
    print("-" * 50)
    
    # 3D CNN分析
    final_train_loss_3d = history_3d['train'][-1]['loss']
    final_val_loss_3d = history_3d['val'][-1]['loss']
    final_train_acc_3d = history_3d['train'][-1]['acc']
    final_val_acc_3d = history_3d['val'][-1]['acc']
    
    print(f"3D CNN 最终性能:")
    print(f"  训练损失: {final_train_loss_3d:.4f}")
    print(f"  验证损失: {final_val_loss_3d:.4f}")
    print(f"  训练准确率: {final_train_acc_3d:.2f}%")
    print(f"  验证准确率: {final_val_acc_3d:.2f}%")
    print(f"  过拟合程度: {abs(final_train_acc_3d - final_val_acc_3d):.2f}%")
    
    # 2D CNN分析
    final_train_loss_2d = history_2d['train'][-1]['loss']
    final_val_loss_2d = history_2d['val'][-1]['loss']
    final_train_acc_2d = history_2d['train'][-1]['acc']
    final_val_acc_2d = history_2d['val'][-1]['acc']
    
    print(f"\n2D CNN 最终性能:")
    print(f"  训练损失: {final_train_loss_2d:.4f}")
    print(f"  验证损失: {final_val_loss_2d:.4f}")
    print(f"  训练准确率: {final_train_acc_2d:.2f}%")
    print(f"  验证准确率: {final_val_acc_2d:.2f}%")
    print(f"  过拟合程度: {abs(final_train_acc_2d - final_val_acc_2d):.2f}%")
    
    # 收敛分析
    print(f"\n🎯 收敛分析:")
    print("-" * 50)
    
    # 找到最佳验证准确率的epoch
    best_epoch_3d = max(range(len(history_3d['val'])), key=lambda i: history_3d['val'][i]['acc'])
    best_epoch_2d = max(range(len(history_2d['val'])), key=lambda i: history_2d['val'][i]['acc'])
    
    print(f"3D CNN 最佳性能在第 {best_epoch_3d} 轮达到")
    print(f"2D CNN 最佳性能在第 {best_epoch_2d} 轮达到")
    
    # 学习曲线趋势
    if len(history_3d['val']) >= 10:
        early_val_acc_3d = np.mean([history_3d['val'][i]['acc'] for i in range(5)])
        late_val_acc_3d = np.mean([history_3d['val'][i]['acc'] for i in range(-5, 0)])
        improvement_3d = late_val_acc_3d - early_val_acc_3d
        print(f"3D CNN 验证准确率提升: {improvement_3d:.2f}% (前5轮 vs 后5轮)")
    
    if len(history_2d['val']) >= 10:
        early_val_acc_2d = np.mean([history_2d['val'][i]['acc'] for i in range(5)])
        late_val_acc_2d = np.mean([history_2d['val'][i]['acc'] for i in range(-5, 0)])
        improvement_2d = late_val_acc_2d - early_val_acc_2d
        print(f"2D CNN 验证准确率提升: {improvement_2d:.2f}% (前5轮 vs 后5轮)")

def show_training_logs():
    """显示训练日志"""
    print("\n📝 训练日志摘要:")
    print("=" * 80)
    
    # 3D CNN训练日志
    log_file_3d = Path("outputs/temporal_3d_cnn_training/training.log")
    if log_file_3d.exists():
        print("\n🔥 3D CNN 训练日志 (最后10行):")
        print("-" * 50)
        with open(log_file_3d, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(line.strip())
    
    # 2D CNN训练日志
    log_file_2d = Path("outputs/temporal_2d_cnn_training/training.log")
    if log_file_2d.exists():
        print("\n🔥 2D CNN 训练日志 (最后10行):")
        print("-" * 50)
        with open(log_file_2d, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(line.strip())

def main():
    """主函数"""
    print("🚀 生成训练过程可视化...")
    
    # 打印训练总结
    print_training_summary()
    
    # 显示训练日志
    show_training_logs()
    
    # 生成训练曲线图
    try:
        plot_training_curves()
        print("\n✅ 训练过程可视化完成！")
    except Exception as e:
        print(f"❌ 生成训练曲线失败: {e}")
        print("请确保安装了matplotlib: pip install matplotlib")

if __name__ == '__main__':
    main()
