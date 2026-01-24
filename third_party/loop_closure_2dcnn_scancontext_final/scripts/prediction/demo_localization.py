#!/usr/bin/env python3
"""
位置定位演示脚本
展示如何使用训练好的模型进行位置定位
"""
import argparse
from pathlib import Path
import json
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from localize_position import PositionLocalizer

def demo_single_localization():
    """单次定位演示"""
    print("🎯 单次位置定位演示")
    print("=" * 50)
    
    # 使用最新的模型
    model_path = "outputs/models/best_sc_ring_cnn_20250802_165529.pth"
    map_database_path = "data/raw/ply_files"
    
    # 检查文件存在性
    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先训练模型或指定正确的模型路径")
        return
    
    if not Path(map_database_path).exists():
        print(f"❌ 地图数据库不存在: {map_database_path}")
        return
    
    try:
        # 初始化定位器
        print("🔄 初始化位置定位器...")
        localizer = PositionLocalizer(model_path, map_database_path, device='cpu')
        
        # 获取地图中的一个文件作为查询示例
        ply_files = list(Path(map_database_path).glob("*.ply"))
        if len(ply_files) == 0:
            print("❌ 地图数据库中没有PLY文件")
            return
        
        # 选择中间的一个文件作为查询
        query_file = ply_files[len(ply_files) // 2]
        print(f"🔍 使用查询文件: {query_file.name}")
        
        # 执行定位
        print("🚀 执行位置定位...")
        start_time = time.time()
        result = localizer.localize_from_ply(str(query_file), top_k=5)
        end_time = time.time()
        
        # 显示结果
        print("\n✅ 定位完成！")
        print(f"⏱️  处理时间: {end_time - start_time:.3f}秒")
        print(f"🎯 最佳匹配位置: {result['best_position_index']}")
        print(f"📄 对应文件: {result['best_map_file']}")
        print(f"🔗 相似度: {result['best_similarity']:.4f}")
        print(f"🎚️  置信度: {result['confidence']}")
        
        print(f"\n📋 前5个候选位置:")
        for i, candidate in enumerate(result['top_k_candidates']):
            print(f"  {i+1}. 位置 {candidate['position_index']:3d}: "
                  f"相似度={candidate['similarity']:.4f}, "
                  f"文件={candidate['map_file']}")
        
        # 分析结果
        print(f"\n📊 结果分析:")
        if result['best_similarity'] > 0.9:
            print("🟢 定位精度: 极高 - 几乎完美匹配")
        elif result['best_similarity'] > 0.8:
            print("🟡 定位精度: 高 - 可靠的匹配")
        elif result['best_similarity'] > 0.6:
            print("🟠 定位精度: 中等 - 可能的匹配")
        else:
            print("🔴 定位精度: 低 - 不确定的匹配")
        
        return result
        
    except Exception as e:
        print(f"❌ 定位失败: {e}")
        return None

def demo_batch_localization():
    """批量定位演示"""
    print("\n🎯 批量位置定位演示")
    print("=" * 50)
    
    model_path = "outputs/models/best_sc_ring_cnn_20250802_165529.pth"
    map_database_path = "data/raw/ply_files"
    
    if not Path(model_path).exists() or not Path(map_database_path).exists():
        print("❌ 文件不存在，跳过批量演示")
        return
    
    try:
        # 初始化定位器
        localizer = PositionLocalizer(model_path, map_database_path, device='cpu')
        
        # 获取测试文件
        ply_files = list(Path(map_database_path).glob("*.ply"))
        test_files = ply_files[::50]  # 每50个文件取一个进行测试
        
        print(f"📊 测试 {len(test_files)} 个文件的定位精度")
        
        correct_predictions = 0
        total_time = 0
        
        for i, test_file in enumerate(test_files):
            start_time = time.time()
            result = localizer.localize_from_ply(str(test_file), top_k=1)
            end_time = time.time()
            
            processing_time = end_time - start_time
            total_time += processing_time
            
            # 检查是否正确预测（文件名匹配）
            predicted_file = result['best_map_file']
            actual_file = test_file.name
            
            is_correct = predicted_file == actual_file
            if is_correct:
                correct_predictions += 1
            
            print(f"  测试 {i+1:2d}/{len(test_files)}: "
                  f"实际={actual_file[:15]:<15} "
                  f"预测={predicted_file[:15]:<15} "
                  f"相似度={result['best_similarity']:.3f} "
                  f"{'✅' if is_correct else '❌'}")
        
        # 统计结果
        accuracy = correct_predictions / len(test_files)
        avg_time = total_time / len(test_files)
        
        print(f"\n📈 批量测试结果:")
        print(f"  准确率: {accuracy:.2%} ({correct_predictions}/{len(test_files)})")
        print(f"  平均处理时间: {avg_time:.3f}秒")
        print(f"  总处理时间: {total_time:.2f}秒")
        
    except Exception as e:
        print(f"❌ 批量测试失败: {e}")

def demo_usage_guide():
    """使用指南"""
    print("\n📖 位置定位系统使用指南")
    print("=" * 50)
    
    print("1. 🏗️  准备工作:")
    print("   - 训练好的模型文件 (.pth)")
    print("   - 地图数据库目录 (包含所有参考位置的PLY文件)")
    print("   - 查询PLY文件 (当前位置的点云)")
    
    print("\n2. 🔧 命令行使用:")
    print("   python localize_position.py \\")
    print("     --model outputs/models/best_sc_ring_cnn_xxx.pth \\")
    print("     --map_database data/raw/ply_files \\")
    print("     --query /path/to/current_position.ply \\")
    print("     --top_k 5")
    
    print("\n3. 🤖 ROS节点使用:")
    print("   rosrun your_package ros_localization_node.py \\")
    print("     _model_path:=outputs/models/best_sc_ring_cnn_xxx.pth \\")
    print("     _map_database_path:=data/raw/ply_files \\")
    print("     _pointcloud_topic:=/velodyne_points")
    
    print("\n4. 🐍 Python API使用:")
    print("   from localize_position import PositionLocalizer")
    print("   localizer = PositionLocalizer(model_path, map_db_path)")
    print("   result = localizer.localize_from_ply(query_ply)")
    print("   position_index = result['best_position_index']")
    
    print("\n5. 📊 输出解释:")
    print("   - position_index: 在地图中的位置索引 (0, 1, 2, ...)")
    print("   - similarity: 相似度分数 (0-1, 越高越相似)")
    print("   - confidence: 置信度等级 (high/medium/low)")
    print("   - map_file: 对应的地图文件名")
    
    print("\n6. 🎯 后续路径规划:")
    print("   - 获得位置索引后，可以:")
    print("     * 查询预定义的路径规划表")
    print("     * 计算到目标位置的路径")
    print("     * 执行导航控制命令")

def main():
    parser = argparse.ArgumentParser(description='位置定位演示')
    parser.add_argument('--demo', type=str, choices=['single', 'batch', 'guide', 'all'], 
                       default='all', help='演示类型')
    
    args = parser.parse_args()
    
    print("🎯 位置定位系统演示")
    print("基于深度学习的点云位置定位")
    print("=" * 60)
    
    if args.demo in ['single', 'all']:
        demo_single_localization()
    
    if args.demo in ['batch', 'all']:
        demo_batch_localization()
    
    if args.demo in ['guide', 'all']:
        demo_usage_guide()
    
    print("\n🎉 演示完成！")
    print("现在您可以使用这个系统进行实时位置定位了。")

if __name__ == "__main__":
    main()
