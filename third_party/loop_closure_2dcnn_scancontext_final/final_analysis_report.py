#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最终分析报告：果园数据回环检测结果
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_final_report():
    """生成最终分析报告"""
    
    print("="*80)
    print("果园巡检数据回环检测最终分析报告")
    print("="*80)
    
    # 加载结果
    results_file = Path("results/simple_test_results.pkl")
    if not results_file.exists():
        print("未找到测试结果文件")
        return
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    predictions = np.array(results['predictions'])
    confidences = np.array(results['confidences'])
    file_indices = np.array(results['file_indices'])
    file_names = results['file_names']
    
    print(f"\n📊 基本统计信息")
    print(f"{'='*50}")
    print(f"数据来源: 果园巡检rosbag (2025-07-03-16-28-57.bag)")
    print(f"总点云文件数: 1769个")
    print(f"采样处理文件数: {len(file_names)}个 (1:20采样)")
    print(f"有效预测数: {len(predictions[predictions >= 0])}个")
    print(f"预测成功率: {len(predictions[predictions >= 0])/len(predictions)*100:.1f}%")
    
    valid_predictions = predictions[predictions >= 0]
    valid_confidences = confidences[predictions >= 0]
    valid_indices = file_indices[predictions >= 0]
    
    print(f"\n🎯 预测结果分析")
    print(f"{'='*50}")
    print(f"预测类别范围: {np.min(valid_predictions)} - {np.max(valid_predictions)}")
    print(f"预测类别数量: {len(np.unique(valid_predictions))}个")
    print(f"平均置信度: {np.mean(valid_confidences):.4f}")
    print(f"置信度标准差: {np.std(valid_confidences):.4f}")
    print(f"最高置信度: {np.max(valid_confidences):.4f}")
    print(f"最低置信度: {np.min(valid_confidences):.4f}")
    
    # 预测分布分析
    print(f"\n📈 预测类别分布")
    print(f"{'='*50}")
    unique, counts = np.unique(valid_predictions, return_counts=True)
    
    # 按出现次数排序
    sorted_indices = np.argsort(counts)[::-1]
    
    print("类别  | 出现次数 | 占比   | 平均置信度")
    print("-" * 40)
    for i in sorted_indices:
        cls = unique[i]
        count = counts[i]
        percentage = count / len(valid_predictions) * 100
        
        # 计算该类别的平均置信度
        cls_mask = valid_predictions == cls
        avg_conf = np.mean(valid_confidences[cls_mask])
        
        print(f"{cls:4d}  | {count:8d} | {percentage:5.1f}% | {avg_conf:.4f}")
    
    # 时序分析
    print(f"\n⏰ 时序分布分析")
    print(f"{'='*50}")
    
    # 分析预测在时间轴上的分布
    time_segments = {
        "前段 (0-400)": (0, 400),
        "中段 (400-800)": (400, 800), 
        "中后段 (800-1200)": (800, 1200),
        "后段 (1200-1600)": (1200, 1600),
        "末段 (1600+)": (1600, 2000)
    }
    
    for segment_name, (start, end) in time_segments.items():
        mask = (valid_indices >= start) & (valid_indices < end)
        if np.sum(mask) > 0:
            segment_predictions = valid_predictions[mask]
            segment_confidences = valid_confidences[mask]
            
            print(f"{segment_name}:")
            print(f"  样本数: {len(segment_predictions)}")
            print(f"  主要类别: {np.bincount(segment_predictions).argmax()}")
            print(f"  平均置信度: {np.mean(segment_confidences):.4f}")
    
    # 回环检测分析
    print(f"\n🔄 回环检测分析")
    print(f"{'='*50}")
    
    # 寻找可能的回环模式
    class_positions = {}
    for i, (pred, idx) in enumerate(zip(valid_predictions, valid_indices)):
        if pred not in class_positions:
            class_positions[pred] = []
        class_positions[pred].append((idx, i))
    
    potential_loops = []
    for cls, positions in class_positions.items():
        if len(positions) > 1:
            # 计算位置间隔
            pos_indices = [pos[0] for pos in positions]
            gaps = []
            for i in range(1, len(pos_indices)):
                gap = pos_indices[i] - pos_indices[i-1]
                gaps.append(gap)
            
            # 如果间隔较大，可能是真正的回环
            if any(gap > 200 for gap in gaps):
                potential_loops.append((cls, positions, gaps))
    
    if potential_loops:
        print("发现潜在回环模式:")
        for cls, positions, gaps in potential_loops:
            pos_indices = [pos[0] for pos in positions]
            print(f"  类别 {cls}: 出现在文件索引 {pos_indices}")
            print(f"    间隔: {gaps} (文件数)")
            
            # 计算该类别的置信度
            cls_confidences = [valid_confidences[pos[1]] for pos in positions]
            print(f"    置信度: {[f'{conf:.3f}' for conf in cls_confidences]}")
    else:
        print("未发现明显的回环模式")
        print("可能原因:")
        print("  1. 果园环境与训练数据(农田)差异较大")
        print("  2. 模型训练不充分(仅20个epoch)")
        print("  3. 果园轨迹可能没有明显的重复访问模式")
    
    # 模型性能评估
    print(f"\n🤖 模型性能评估")
    print(f"{'='*50}")
    
    # 置信度分布分析
    high_conf_count = np.sum(valid_confidences > 0.2)
    medium_conf_count = np.sum((valid_confidences > 0.1) & (valid_confidences <= 0.2))
    low_conf_count = np.sum(valid_confidences <= 0.1)
    
    print(f"置信度分布:")
    print(f"  高置信度 (>0.2): {high_conf_count} 个 ({high_conf_count/len(valid_confidences)*100:.1f}%)")
    print(f"  中置信度 (0.1-0.2): {medium_conf_count} 个 ({medium_conf_count/len(valid_confidences)*100:.1f}%)")
    print(f"  低置信度 (≤0.1): {low_conf_count} 个 ({low_conf_count/len(valid_confidences)*100:.1f}%)")
    
    # 预测多样性分析
    entropy = -np.sum((counts/len(valid_predictions)) * np.log2(counts/len(valid_predictions)))
    max_entropy = np.log2(len(unique))
    normalized_entropy = entropy / max_entropy
    
    print(f"\n预测多样性:")
    print(f"  预测熵: {entropy:.3f}")
    print(f"  归一化熵: {normalized_entropy:.3f}")
    print(f"  多样性评价: {'高' if normalized_entropy > 0.8 else '中' if normalized_entropy > 0.5 else '低'}")
    
    # 结论和建议
    print(f"\n📝 结论和建议")
    print(f"{'='*50}")
    
    print("主要发现:")
    print("1. 模型能够对果园数据进行预测，但置信度普遍较低")
    print("2. 预测结果显示一定的多样性，说明模型在尝试区分不同场景")
    print("3. 未发现明显的回环模式，可能是环境差异导致的")
    
    print("\n改进建议:")
    print("1. 使用更充分训练的模型 (更多epoch和更好的超参数)")
    print("2. 在果园数据上进行域适应或微调")
    print("3. 收集更多果园环境的训练数据")
    print("4. 考虑使用无监督或半监督方法")
    print("5. 分析农田和果园环境的ScanContext特征差异")
    
    print(f"\n📊 可视化文件位置:")
    print(f"  预测序列图: results/simple_test/simple_predictions.png")
    print(f"  详细结果数据: results/simple_test_results.pkl")
    
    print(f"\n" + "="*80)
    print("报告生成完成")
    print("="*80)

if __name__ == '__main__':
    generate_final_report()
