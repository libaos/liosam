#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
时序回环检测系统演示脚本

该脚本演示了完整的时序回环检测系统，包括：
1. 数据预处理
2. 模型训练
3. 模型评估
4. 结果可视化
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from scripts.tools.preprocess_temporal_data import preprocess_ply_files
from scripts.training.train_temporal_models import TemporalModelTrainer, create_default_config
from scripts.evaluation.evaluate_temporal_models import TemporalModelEvaluator

def run_preprocessing(data_dir, output_dir, visualize=True):
    """运行数据预处理"""
    print("=" * 60)
    print("步骤 1: 数据预处理")
    print("=" * 60)
    
    # 预处理数据
    preprocess_ply_files(
        data_dir=data_dir,
        output_dir=output_dir,
        sequence_length=5,
        num_classes=20
    )
    
    # 生成可视化
    if visualize:
        from scripts.tools.preprocess_temporal_data import visualize_scan_context, visualize_temporal_sequence
        
        vis_dir = Path(output_dir) / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # 可视化ScanContext
        sc_dir = Path(output_dir) / "scan_contexts"
        if sc_dir.exists():
            sc_files = list(sc_dir.glob("*.npy"))[:3]
            for i, sc_file in enumerate(sc_files):
                output_file = vis_dir / f"scan_context_sample_{i}.png"
                visualize_scan_context(sc_file, output_file)
        
        # 可视化时序序列
        sequence_file = Path(output_dir) / "temporal_sequences_len5.pkl"
        if sequence_file.exists():
            for i in range(2):
                visualize_temporal_sequence(
                    sequence_file, 
                    sequence_idx=i, 
                    output_dir=vis_dir / "temporal_sequences"
                )
    
    print("数据预处理完成！")


def run_training(data_dir, model_type='temporal_3d_cnn', epochs=50, batch_size=8):
    """运行模型训练"""
    print("=" * 60)
    print(f"步骤 2: 训练 {model_type} 模型")
    print("=" * 60)
    
    # 创建配置
    config = create_default_config(model_type, sequence_length=5)
    config['data']['data_dir'] = data_dir
    config['data']['batch_size'] = batch_size
    config['training']['epochs'] = epochs
    config['output_dir'] = f"outputs/{model_type}_demo"
    
    # 创建训练器
    trainer = TemporalModelTrainer(config)
    
    # 开始训练
    trainer.train()
    
    print(f"{model_type} 模型训练完成！")
    return config['output_dir']


def run_evaluation(config_or_output_dir, checkpoint_path):
    """运行模型评估"""
    print("=" * 60)
    print("步骤 3: 模型评估")
    print("=" * 60)
    
    # 加载配置
    if isinstance(config_or_output_dir, str):
        config_file = Path(config_or_output_dir) / "config.json"
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = config_or_output_dir
    
    # 设置评估输出目录
    eval_output_dir = Path(config['output_dir']) / "evaluation"
    config['output_dir'] = str(eval_output_dir)
    
    # 创建评估器
    evaluator = TemporalModelEvaluator(config, checkpoint_path)
    
    # 运行评估
    results = evaluator.run_full_evaluation()
    
    print("模型评估完成！")
    return results


def compare_models(data_dir, models_to_compare=None, epochs=30, batch_size=8):
    """比较不同模型的性能"""
    print("=" * 60)
    print("步骤 4: 模型性能比较")
    print("=" * 60)
    
    if models_to_compare is None:
        models_to_compare = [
            'temporal_2d_cnn',
            'temporal_3d_cnn',
            'temporal_2d_cnn_lite',
            'temporal_3d_cnn_lite'
        ]
    
    results = {}
    
    for model_type in models_to_compare:
        print(f"\n训练和评估 {model_type}...")
        
        try:
            # 训练模型
            output_dir = run_training(data_dir, model_type, epochs, batch_size)
            
            # 评估模型
            checkpoint_path = Path(output_dir) / "checkpoint_best.pth"
            if checkpoint_path.exists():
                eval_results = run_evaluation(output_dir, str(checkpoint_path))
                results[model_type] = eval_results['test_metrics']
            else:
                print(f"警告: 未找到 {model_type} 的最佳检查点")
                
        except Exception as e:
            print(f"错误: 训练/评估 {model_type} 时出错: {e}")
            continue
    
    # 生成比较报告
    generate_comparison_report(results)
    
    return results


def generate_comparison_report(results):
    """生成模型比较报告"""
    print("\n" + "=" * 60)
    print("模型性能比较报告")
    print("=" * 60)
    
    if not results:
        print("没有可比较的结果")
        return
    
    # 打印表格头
    print(f"{'模型类型':<25} {'准确率':<10} {'Top-5准确率':<12} {'F1分数':<10} {'推理速度(FPS)':<15}")
    print("-" * 80)
    
    # 打印每个模型的结果
    for model_type, metrics in results.items():
        accuracy = metrics.get('accuracy', 0) * 100
        top5_acc = metrics.get('top5_accuracy', 0) * 100
        f1_score = metrics.get('macro_f1', 0) * 100
        fps = metrics.get('fps', 0)
        
        print(f"{model_type:<25} {accuracy:<10.2f} {top5_acc:<12.2f} {f1_score:<10.2f} {fps:<15.2f}")
    
    # 找出最佳模型
    best_accuracy_model = max(results.keys(), key=lambda k: results[k].get('accuracy', 0))
    best_speed_model = max(results.keys(), key=lambda k: results[k].get('fps', 0))
    
    print("\n" + "=" * 60)
    print("总结:")
    print(f"最高准确率模型: {best_accuracy_model} ({results[best_accuracy_model]['accuracy']*100:.2f}%)")
    print(f"最快推理速度模型: {best_speed_model} ({results[best_speed_model]['fps']:.2f} FPS)")
    
    # 保存比较结果
    comparison_file = "outputs/model_comparison_results.json"
    os.makedirs("outputs", exist_ok=True)
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"详细比较结果已保存到: {comparison_file}")


def main():
    parser = argparse.ArgumentParser(description='时序回环检测系统演示')
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['preprocess', 'train', 'evaluate', 'compare', 'full'],
                       help='运行模式')
    parser.add_argument('--model', type=str, default='temporal_3d_cnn',
                       choices=['temporal_2d_cnn', 'temporal_2d_cnn_lite', 'temporal_2d_cnn_resnet',
                               'temporal_3d_cnn', 'temporal_3d_cnn_lite', 'temporal_3d_cnn_deep'],
                       help='模型类型')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--checkpoint', type=str, help='评估时使用的检查点路径')
    parser.add_argument('--no_visualize', action='store_true', help='不生成可视化')
    
    args = parser.parse_args()
    
    print("时序回环检测系统演示")
    print(f"数据目录: {args.data_dir}")
    print(f"运行模式: {args.mode}")
    print(f"使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    if args.mode == 'preprocess':
        # 只运行数据预处理
        processed_data_dir = Path(args.data_dir) / "processed"
        run_preprocessing(args.data_dir, processed_data_dir, not args.no_visualize)
        
    elif args.mode == 'train':
        # 只运行训练
        processed_data_dir = Path(args.data_dir) / "processed"
        run_training(processed_data_dir, args.model, args.epochs, args.batch_size)
        
    elif args.mode == 'evaluate':
        # 只运行评估
        if not args.checkpoint:
            print("错误: 评估模式需要提供 --checkpoint 参数")
            return
        
        processed_data_dir = Path(args.data_dir) / "processed"
        run_evaluation(processed_data_dir, args.checkpoint)
        
    elif args.mode == 'compare':
        # 比较多个模型
        processed_data_dir = Path(args.data_dir) / "processed"
        compare_models(processed_data_dir, epochs=args.epochs, batch_size=args.batch_size)
        
    elif args.mode == 'full':
        # 运行完整流程
        print("运行完整演示流程...")
        
        # 1. 数据预处理
        processed_data_dir = Path(args.data_dir) / "processed"
        run_preprocessing(args.data_dir, processed_data_dir, not args.no_visualize)
        
        # 2. 训练模型
        output_dir = run_training(processed_data_dir, args.model, args.epochs, args.batch_size)
        
        # 3. 评估模型
        checkpoint_path = Path(output_dir) / "checkpoint_best.pth"
        if checkpoint_path.exists():
            run_evaluation(output_dir, str(checkpoint_path))
        else:
            print("警告: 未找到最佳检查点，跳过评估")
        
        print("\n" + "=" * 60)
        print("完整演示流程结束！")
        print("=" * 60)
        print(f"训练输出目录: {output_dir}")
        print(f"可视化目录: {processed_data_dir}/visualizations")
        print("请查看输出目录中的结果文件和可视化图表。")


if __name__ == '__main__':
    main()
