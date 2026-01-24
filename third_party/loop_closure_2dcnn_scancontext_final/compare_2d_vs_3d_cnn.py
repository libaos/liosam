#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对比分析Temporal 2D CNN vs Temporal 3D CNN在果园数据上的表现
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_results():
    """加载两个模型的结果"""
    
    # 加载2D CNN结果
    results_2d_file = Path("results/simple_test_results.pkl")
    if results_2d_file.exists():
        with open(results_2d_file, 'rb') as f:
            results_2d = pickle.load(f)
    else:
        results_2d = None
    
    # 加载3D CNN结果
    results_3d_file = Path("results/realtime_3dcnn_results.pkl")
    if results_3d_file.exists():
        with open(results_3d_file, 'rb') as f:
            results_3d = pickle.load(f)
    else:
        results_3d = None
    
    return results_2d, results_3d

def compare_models():
    """对比两个模型的性能"""
    
    print("="*80)
    print("Temporal 2D CNN vs Temporal 3D CNN 对比分析")
    print("="*80)
    
    results_2d, results_3d = load_results()
    
    if results_2d is None or results_3d is None:
        print("❌ 无法加载结果文件")
        return
    
    # 提取数据
    predictions_2d = np.array(results_2d['predictions'])
    confidences_2d = np.array(results_2d['confidences'])
    
    predictions_3d = np.array(results_3d['predictions'])
    confidences_3d = np.array(results_3d['confidences'])
    
    print(f"\n📊 基本统计对比")
    print(f"{'='*50}")
    print(f"{'指标':<20} {'2D CNN':<15} {'3D CNN':<15}")
    print(f"{'-'*50}")
    print(f"{'总预测数':<20} {len(predictions_2d):<15} {len(predictions_3d):<15}")
    print(f"{'有效预测数':<20} {len(predictions_2d[predictions_2d >= 0]):<15} {len(predictions_3d[predictions_3d >= 0]):<15}")
    
    # 有效预测分析
    valid_2d = predictions_2d[predictions_2d >= 0]
    valid_conf_2d = confidences_2d[predictions_2d >= 0]
    
    valid_3d = predictions_3d[predictions_3d >= 0]
    valid_conf_3d = confidences_3d[predictions_3d >= 0]
    
    print(f"{'预测类别数':<20} {len(np.unique(valid_2d)):<15} {len(np.unique(valid_3d)):<15}")
    print(f"{'平均置信度':<20} {np.mean(valid_conf_2d):.4f}{'':>10} {np.mean(valid_conf_3d):.4f}{'':>10}")
    print(f"{'置信度标准差':<20} {np.std(valid_conf_2d):.4f}{'':>10} {np.std(valid_conf_3d):.4f}{'':>10}")
    print(f"{'最高置信度':<20} {np.max(valid_conf_2d):.4f}{'':>10} {np.max(valid_conf_3d):.4f}{'':>10}")
    
    # 预测多样性分析
    print(f"\n🎯 预测多样性分析")
    print(f"{'='*50}")
    
    # 计算熵
    def calculate_entropy(predictions):
        unique, counts = np.unique(predictions, return_counts=True)
        probs = counts / len(predictions)
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(unique))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        return entropy, normalized_entropy
    
    entropy_2d, norm_entropy_2d = calculate_entropy(valid_2d)
    entropy_3d, norm_entropy_3d = calculate_entropy(valid_3d)
    
    print(f"{'模型':<15} {'熵':<10} {'归一化熵':<10} {'多样性':<10}")
    print(f"{'-'*45}")
    print(f"{'2D CNN':<15} {entropy_2d:.3f}{'':>5} {norm_entropy_2d:.3f}{'':>6} {'高' if norm_entropy_2d > 0.8 else '中' if norm_entropy_2d > 0.5 else '低':<10}")
    print(f"{'3D CNN':<15} {entropy_3d:.3f}{'':>5} {norm_entropy_3d:.3f}{'':>6} {'高' if norm_entropy_3d > 0.8 else '中' if norm_entropy_3d > 0.5 else '低':<10}")
    
    # 预测分布对比
    print(f"\n📈 预测类别分布对比")
    print(f"{'='*50}")
    
    print("2D CNN预测分布:")
    unique_2d, counts_2d = np.unique(valid_2d, return_counts=True)
    for cls, count in zip(unique_2d, counts_2d):
        percentage = count / len(valid_2d) * 100
        print(f"  类别 {cls:2d}: {count:3d} 次 ({percentage:5.1f}%)")
    
    print("\n3D CNN预测分布:")
    unique_3d, counts_3d = np.unique(valid_3d, return_counts=True)
    for cls, count in zip(unique_3d, counts_3d):
        percentage = count / len(valid_3d) * 100
        print(f"  类别 {cls:2d}: {count:3d} 次 ({percentage:5.1f}%)")
    
    # 模型行为分析
    print(f"\n🤖 模型行为分析")
    print(f"{'='*50}")
    
    print("2D CNN特点:")
    if len(unique_2d) > 1:
        print("  ✅ 预测多样化，能区分不同场景")
        print("  ✅ 显示出一定的环境适应能力")
        if np.max(valid_conf_2d) > 0.2:
            print("  ✅ 部分预测具有较高置信度")
        else:
            print("  ⚠️  整体置信度偏低")
    else:
        print("  ❌ 预测单一，可能过拟合到特定模式")
    
    print("\n3D CNN特点:")
    if len(unique_3d) > 1:
        print("  ✅ 预测多样化，能区分不同场景")
        print("  ✅ 显示出一定的环境适应能力")
        if np.max(valid_conf_3d) > 0.2:
            print("  ✅ 部分预测具有较高置信度")
        else:
            print("  ⚠️  整体置信度偏低")
    else:
        print("  ❌ 预测单一，可能过拟合到特定模式")
        print("  ❌ 未充分利用3D时序信息")
    
    # 回环检测能力对比
    print(f"\n🔄 回环检测能力对比")
    print(f"{'='*50}")
    
    def analyze_loops(predictions, model_name):
        class_positions = {}
        for i, pred in enumerate(predictions):
            if pred not in class_positions:
                class_positions[pred] = []
            class_positions[pred].append(i)
        
        loops = []
        for cls, positions in class_positions.items():
            if len(positions) > 1:
                gaps = []
                for i in range(1, len(positions)):
                    gap = positions[i] - positions[i-1]
                    gaps.append(gap)
                if any(gap > 10 for gap in gaps):  # 间隔大于10帧
                    loops.append((cls, positions, gaps))
        
        print(f"{model_name}:")
        if loops:
            print(f"  发现 {len(loops)} 个潜在回环")
            for cls, positions, gaps in loops:
                print(f"    类别 {cls}: 出现 {len(positions)} 次，间隔 {gaps}")
        else:
            print("  未发现明显回环模式")
    
    analyze_loops(valid_2d, "2D CNN")
    analyze_loops(valid_3d, "3D CNN")
    
    # 综合评价
    print(f"\n🏆 综合评价")
    print(f"{'='*50}")
    
    # 计算综合得分
    def calculate_score(predictions, confidences):
        valid_preds = predictions[predictions >= 0]
        valid_confs = confidences[predictions >= 0]
        
        if len(valid_preds) == 0:
            return 0.0
        
        # 多样性得分 (0-40分)
        diversity_score = len(np.unique(valid_preds)) / 20 * 40
        
        # 置信度得分 (0-30分)
        confidence_score = np.mean(valid_confs) * 30 / 0.2  # 假设0.2是满分
        confidence_score = min(confidence_score, 30)
        
        # 稳定性得分 (0-30分) - 置信度标准差越小越好
        stability_score = max(0, 30 - np.std(valid_confs) * 1000)
        
        total_score = diversity_score + confidence_score + stability_score
        return min(total_score, 100)
    
    score_2d = calculate_score(predictions_2d, confidences_2d)
    score_3d = calculate_score(predictions_3d, confidences_3d)
    
    print(f"{'模型':<15} {'综合得分':<10} {'评级':<10}")
    print(f"{'-'*35}")
    
    def get_grade(score):
        if score >= 80: return "优秀"
        elif score >= 60: return "良好"
        elif score >= 40: return "一般"
        else: return "较差"
    
    print(f"{'2D CNN':<15} {score_2d:.1f}{'':>5} {get_grade(score_2d):<10}")
    print(f"{'3D CNN':<15} {score_3d:.1f}{'':>5} {get_grade(score_3d):<10}")
    
    # 结论和建议
    print(f"\n📝 结论和建议")
    print(f"{'='*50}")
    
    if score_2d > score_3d:
        winner = "2D CNN"
        print(f"🏆 {winner} 在当前测试中表现更好")
    elif score_3d > score_2d:
        winner = "3D CNN"
        print(f"🏆 {winner} 在当前测试中表现更好")
    else:
        winner = "平局"
        print(f"🤝 两个模型表现相当")
    
    print(f"\n主要发现:")
    print(f"1. 两个模型都面临域适应挑战（农田→果园）")
    print(f"2. 置信度普遍偏低，说明模型不确定性较高")
    print(f"3. 需要更充分的训练或域适应技术")
    
    print(f"\n改进建议:")
    print(f"1. 在果园数据上进行微调或域适应")
    print(f"2. 增加训练数据的多样性")
    print(f"3. 尝试无监督或半监督学习方法")
    print(f"4. 优化网络架构和超参数")
    
    # 生成对比图表
    create_comparison_plots(results_2d, results_3d)

def create_comparison_plots(results_2d, results_3d):
    """创建对比图表"""
    
    predictions_2d = np.array(results_2d['predictions'])
    confidences_2d = np.array(results_2d['confidences'])
    
    predictions_3d = np.array(results_3d['predictions'])
    confidences_3d = np.array(results_3d['confidences'])
    
    # 有效数据
    valid_2d = predictions_2d[predictions_2d >= 0]
    valid_conf_2d = confidences_2d[predictions_2d >= 0]
    
    valid_3d = predictions_3d[predictions_3d >= 0]
    valid_conf_3d = confidences_3d[predictions_3d >= 0]
    
    plt.figure(figsize=(15, 10))
    
    # 1. 置信度对比
    plt.subplot(2, 3, 1)
    plt.hist(valid_conf_2d, bins=20, alpha=0.7, label='2D CNN', color='blue')
    plt.hist(valid_conf_3d, bins=20, alpha=0.7, label='3D CNN', color='red')
    plt.xlabel('置信度')
    plt.ylabel('频次')
    plt.title('置信度分布对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 预测类别分布对比
    plt.subplot(2, 3, 2)
    unique_2d, counts_2d = np.unique(valid_2d, return_counts=True)
    unique_3d, counts_3d = np.unique(valid_3d, return_counts=True)
    
    x_2d = unique_2d
    x_3d = unique_3d + 0.4  # 偏移以避免重叠
    
    plt.bar(x_2d, counts_2d, width=0.4, alpha=0.7, label='2D CNN', color='blue')
    plt.bar(x_3d, counts_3d, width=0.4, alpha=0.7, label='3D CNN', color='red')
    plt.xlabel('预测类别')
    plt.ylabel('频次')
    plt.title('预测类别分布对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 时序预测对比（前50个）
    plt.subplot(2, 3, 3)
    n_show = min(50, len(valid_2d), len(valid_3d))
    plt.plot(range(n_show), valid_2d[:n_show], 'bo-', markersize=3, label='2D CNN', alpha=0.7)
    plt.plot(range(n_show), valid_3d[:n_show], 'ro-', markersize=3, label='3D CNN', alpha=0.7)
    plt.xlabel('时间步')
    plt.ylabel('预测类别')
    plt.title('时序预测对比（前50帧）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 置信度时序对比
    plt.subplot(2, 3, 4)
    plt.plot(range(n_show), valid_conf_2d[:n_show], 'b-', label='2D CNN', alpha=0.7)
    plt.plot(range(n_show), valid_conf_3d[:n_show], 'r-', label='3D CNN', alpha=0.7)
    plt.xlabel('时间步')
    plt.ylabel('置信度')
    plt.title('置信度时序对比（前50帧）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. 统计对比
    plt.subplot(2, 3, 5)
    metrics = ['平均置信度', '最高置信度', '预测类别数', '标准差×10']
    values_2d = [np.mean(valid_conf_2d), np.max(valid_conf_2d), 
                len(np.unique(valid_2d)), np.std(valid_conf_2d)*10]
    values_3d = [np.mean(valid_conf_3d), np.max(valid_conf_3d), 
                len(np.unique(valid_3d)), np.std(valid_conf_3d)*10]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, values_2d, width, label='2D CNN', alpha=0.7, color='blue')
    plt.bar(x + width/2, values_3d, width, label='3D CNN', alpha=0.7, color='red')
    plt.xlabel('指标')
    plt.ylabel('数值')
    plt.title('性能指标对比')
    plt.xticks(x, metrics, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. 预测一致性分析
    plt.subplot(2, 3, 6)
    # 计算滑动窗口内的预测一致性
    window_size = 10
    consistency_2d = []
    consistency_3d = []
    
    for i in range(window_size, min(len(valid_2d), len(valid_3d))):
        window_2d = valid_2d[i-window_size:i]
        window_3d = valid_3d[i-window_size:i]
        
        # 计算窗口内最频繁类别的占比
        unique_2d, counts_2d = np.unique(window_2d, return_counts=True)
        unique_3d, counts_3d = np.unique(window_3d, return_counts=True)
        
        consistency_2d.append(np.max(counts_2d) / window_size)
        consistency_3d.append(np.max(counts_3d) / window_size)
    
    plt.plot(consistency_2d, label='2D CNN', alpha=0.7, color='blue')
    plt.plot(consistency_3d, label='3D CNN', alpha=0.7, color='red')
    plt.xlabel('时间窗口')
    plt.ylabel('预测一致性')
    plt.title('预测一致性对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    save_dir = Path("results/comparison")
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "2d_vs_3d_cnn_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 对比图表已保存到 {save_dir}/2d_vs_3d_cnn_comparison.png")

if __name__ == '__main__':
    compare_models()
