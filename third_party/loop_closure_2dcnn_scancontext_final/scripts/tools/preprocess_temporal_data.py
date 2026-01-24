#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import glob
from pathlib import Path
import numpy as np
import pickle
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.scan_context import ScanContext
from utils.temporal_dataset import TemporalScanContextDataset

def preprocess_ply_files(data_dir, output_dir, sequence_length=5, num_classes=20):
    """
    预处理PLY文件，生成时序ScanContext数据
    
    参数:
        data_dir (str): 原始PLY文件目录
        output_dir (str): 输出目录
        sequence_length (int): 时序序列长度
        num_classes (int): 路径类别数量
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化ScanContext生成器
    sc_generator = ScanContext(
        num_sectors=60, 
        num_rings=20,
        min_range=0.1,
        max_range=80.0,
        height_lower_bound=-1.0,
        height_upper_bound=9.0
    )
    
    # 获取所有PLY文件
    ply_files = sorted(glob.glob(str(data_dir / "raw" / "ply_files" / "*.ply")))
    
    if not ply_files:
        print(f"错误: 在 {data_dir / 'raw' / 'ply_files'} 中未找到PLY文件")
        return
    
    print(f"找到 {len(ply_files)} 个PLY文件")
    
    # 生成ScanContext特征图
    print("生成ScanContext特征图...")
    scan_contexts = []
    valid_files = []
    
    for ply_file in tqdm(ply_files, desc="处理PLY文件"):
        try:
            # 加载点云并生成ScanContext
            point_cloud = sc_generator.load_point_cloud(ply_file)
            if point_cloud is None or len(point_cloud) == 0:
                print(f"跳过空点云文件: {ply_file}")
                continue
                
            sc = sc_generator.make_scan_context(point_cloud)
            scan_contexts.append(sc)
            valid_files.append(ply_file)
                
        except Exception as e:
            print(f"处理文件 {ply_file} 时出错: {e}")
            continue
    
    print(f"成功生成 {len(scan_contexts)} 个ScanContext特征图")
    
    # 保存单独的ScanContext特征图
    sc_output_dir = output_dir / "scan_contexts"
    sc_output_dir.mkdir(exist_ok=True)
    
    for i, (sc, file_path) in enumerate(zip(scan_contexts, valid_files)):
        filename = Path(file_path).stem + ".npy"
        np.save(sc_output_dir / filename, sc)
    
    print(f"ScanContext特征图已保存到: {sc_output_dir}")
    
    # 按顺序分成20段作为标签
    total_frames = len(scan_contexts)
    frames_per_segment = total_frames // num_classes
    
    print(f"将 {total_frames} 帧分成 {num_classes} 段，每段约 {frames_per_segment} 帧")
    
    # 生成时序序列数据
    sequences_data = []
    
    for sequence_len in [3, 5, 7, 10]:  # 生成不同长度的序列
        print(f"生成序列长度为 {sequence_len} 的时序数据...")
        
        sequences = []
        labels = []
        file_paths = []
        
        # 计算步长（50%重叠）
        step_size = max(1, sequence_len // 2)
        
        for segment_id in range(num_classes):
            start_idx = segment_id * frames_per_segment
            end_idx = min((segment_id + 1) * frames_per_segment, total_frames)
            
            # 在每个段内生成时序序列
            for i in range(start_idx, end_idx - sequence_len + 1, step_size):
                sequence = []
                sequence_files = []
                
                for j in range(sequence_len):
                    sequence.append(scan_contexts[i + j])
                    sequence_files.append(valid_files[i + j])
                
                # 将序列堆叠成 (N, H, W) 张量
                sequence_tensor = np.stack(sequence, axis=0)  # (N, H, W)
                
                sequences.append(sequence_tensor)
                labels.append(segment_id)
                file_paths.append(sequence_files)
        
        print(f"生成 {len(sequences)} 个长度为 {sequence_len} 的时序序列")
        
        # 保存时序数据
        temporal_data = {
            'sequences': sequences,
            'labels': labels,
            'file_paths': file_paths,
            'sequence_length': sequence_len,
            'num_classes': num_classes,
            'metadata': {
                'total_frames': total_frames,
                'frames_per_segment': frames_per_segment,
                'step_size': step_size,
                'scan_context_shape': scan_contexts[0].shape
            }
        }
        
        output_file = output_dir / f"temporal_sequences_len{sequence_len}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(temporal_data, f)
        
        print(f"时序数据已保存到: {output_file}")
        
        sequences_data.append({
            'sequence_length': sequence_len,
            'num_sequences': len(sequences),
            'output_file': str(output_file)
        })
    
    # 生成数据集统计信息
    stats = {
        'total_ply_files': len(ply_files),
        'valid_scan_contexts': len(scan_contexts),
        'num_classes': num_classes,
        'frames_per_segment': frames_per_segment,
        'scan_context_shape': scan_contexts[0].shape,
        'sequences_data': sequences_data
    }
    
    with open(output_dir / "dataset_stats.json", 'w') as f:
        import json
        json.dump(stats, f, indent=2)
    
    print(f"数据集统计信息已保存到: {output_dir / 'dataset_stats.json'}")
    print("数据预处理完成！")


def visualize_scan_context(sc_file, output_file=None):
    """可视化ScanContext特征图"""
    import matplotlib.pyplot as plt
    
    sc = np.load(sc_file)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(sc, cmap='viridis', aspect='auto')
    plt.colorbar(label='Height Value')
    plt.title(f'ScanContext Visualization - {Path(sc_file).stem}')
    plt.xlabel('Sectors')
    plt.ylabel('Rings')
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"可视化图已保存到: {output_file}")
    else:
        plt.show()
    
    plt.close()


def visualize_temporal_sequence(sequence_file, sequence_idx=0, output_dir=None):
    """可视化时序序列"""
    import matplotlib.pyplot as plt
    
    with open(sequence_file, 'rb') as f:
        data = pickle.load(f)
    
    sequences = data['sequences']
    labels = data['labels']
    sequence_length = data['sequence_length']
    
    if sequence_idx >= len(sequences):
        print(f"序列索引 {sequence_idx} 超出范围，最大索引为 {len(sequences) - 1}")
        return
    
    sequence = sequences[sequence_idx]  # (N, H, W)
    label = labels[sequence_idx]
    
    fig, axes = plt.subplots(1, sequence_length, figsize=(4 * sequence_length, 4))
    if sequence_length == 1:
        axes = [axes]
    
    for i in range(sequence_length):
        axes[i].imshow(sequence[i], cmap='viridis', aspect='auto')
        axes[i].set_title(f'Frame {i+1}')
        axes[i].set_xlabel('Sectors')
        axes[i].set_ylabel('Rings')
    
    plt.suptitle(f'Temporal Sequence (Label: Path_{label:02d})')
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"temporal_sequence_idx{sequence_idx}_label{label}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"时序序列可视化已保存到: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='预处理时序ScanContext数据')
    parser.add_argument('--data_dir', type=str, required=True, help='原始数据目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--sequence_length', type=int, default=5, help='时序序列长度')
    parser.add_argument('--num_classes', type=int, default=20, help='路径类别数量')
    parser.add_argument('--visualize', action='store_true', help='生成可视化图')
    parser.add_argument('--vis_output_dir', type=str, help='可视化输出目录')
    
    args = parser.parse_args()
    
    # 预处理数据
    preprocess_ply_files(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        num_classes=args.num_classes
    )
    
    # 生成可视化（如果需要）
    if args.visualize:
        output_dir = Path(args.output_dir)
        vis_output_dir = Path(args.vis_output_dir) if args.vis_output_dir else output_dir / "visualizations"
        vis_output_dir.mkdir(exist_ok=True)
        
        # 可视化一些ScanContext特征图
        sc_dir = output_dir / "scan_contexts"
        if sc_dir.exists():
            sc_files = list(sc_dir.glob("*.npy"))[:5]  # 只可视化前5个
            for i, sc_file in enumerate(sc_files):
                output_file = vis_output_dir / f"scan_context_{i:03d}.png"
                visualize_scan_context(sc_file, output_file)
        
        # 可视化一些时序序列
        for sequence_len in [3, 5, 7, 10]:
            sequence_file = output_dir / f"temporal_sequences_len{sequence_len}.pkl"
            if sequence_file.exists():
                for i in range(3):  # 每个长度可视化3个序列
                    visualize_temporal_sequence(
                        sequence_file, 
                        sequence_idx=i, 
                        output_dir=vis_output_dir / f"sequences_len{sequence_len}"
                    )
        
        print(f"可视化图已保存到: {vis_output_dir}")


if __name__ == '__main__':
    main()
