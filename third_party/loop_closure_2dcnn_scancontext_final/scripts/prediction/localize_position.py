#!/usr/bin/env python3
"""
基于点云的位置定位脚本
用于确定当前位置在预建地图中的索引位置
"""
import argparse
import torch
import numpy as np
from pathlib import Path
import json
import time
import pickle
from typing import List, Dict, Tuple, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from models import SCRingCNN, SCStandardCNN, SCStandardCNNLite, SimpleCNN, SimpleCNNLite
from utils import ScanContext, PLYReader, setup_logger, get_timestamp

class PositionLocalizer:
    """位置定位器 - 用于在预建地图中定位当前位置"""
    
    def __init__(self, model_path: str, map_database_path: str, device: str = 'cpu'):
        """
        初始化位置定位器
        
        参数:
            model_path (str): 训练好的模型文件路径
            map_database_path (str): 地图数据库路径（包含所有参考位置的PLY文件）
            device (str): 设备类型
        """
        self.device = torch.device(device)
        self.model = None
        self.config = None
        self.sc_generator = None
        self.map_database = {}  # 存储地图数据库 {位置索引: 描述子}
        self.map_files = []     # 存储文件路径列表
        
        # 加载模型
        self._load_model(model_path)
        
        # 创建ScanContext生成器
        self.sc_generator = ScanContext()
        
        # 加载地图数据库
        self._load_map_database(map_database_path)
        
    def _load_model(self, model_path: str):
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取配置
        self.config = checkpoint.get('config', {})
        model_type = self.config.get('MODEL_TYPE', 'sc_ring_cnn')
        
        # 创建模型
        if model_type == "simple_cnn":
            self.model = SimpleCNN(
                num_rings=self.config.get('INPUT_HEIGHT', 20),
                num_sectors=self.config.get('INPUT_WIDTH', 60),
                descriptor_dim=self.config.get('DESCRIPTOR_DIM', 256)
            )
        elif model_type == "simple_cnn_lite":
            self.model = SimpleCNNLite(
                num_rings=self.config.get('INPUT_HEIGHT', 20),
                num_sectors=self.config.get('INPUT_WIDTH', 60),
                descriptor_dim=self.config.get('DESCRIPTOR_DIM', 128)
            )
        elif model_type == "sc_standard_cnn":
            self.model = SCStandardCNN(
                num_rings=self.config.get('INPUT_HEIGHT', 20),
                num_sectors=self.config.get('INPUT_WIDTH', 60),
                descriptor_dim=self.config.get('DESCRIPTOR_DIM', 256)
            )
        elif model_type == "sc_standard_cnn_lite":
            self.model = SCStandardCNNLite(
                num_rings=self.config.get('INPUT_HEIGHT', 20),
                num_sectors=self.config.get('INPUT_WIDTH', 60),
                descriptor_dim=self.config.get('DESCRIPTOR_DIM', 128)
            )
        else:  # 默认使用SCRingCNN
            self.model = SCRingCNN(
                num_rings=self.config.get('INPUT_HEIGHT', 20),
                num_sectors=self.config.get('INPUT_WIDTH', 60),
                descriptor_dim=self.config.get('DESCRIPTOR_DIM', 256)
            )
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 模型加载成功: {model_type}")
        print(f"📊 模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_map_database(self, map_database_path: str):
        """加载地图数据库"""
        map_path = Path(map_database_path)
        
        if not map_path.exists():
            raise FileNotFoundError(f"地图数据库路径不存在: {map_database_path}")
        
        # 检查是否有预计算的描述子文件
        descriptor_cache_path = map_path / "descriptors_cache.pkl"
        
        if descriptor_cache_path.exists():
            print("🔄 加载预计算的描述子缓存...")
            with open(descriptor_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.map_database = cache_data['descriptors']
                self.map_files = cache_data['files']
            print(f"✅ 加载了 {len(self.map_database)} 个位置的描述子")
        else:
            print("🔄 首次运行，计算地图数据库描述子...")
            self._compute_map_descriptors(map_path)
            
            # 保存缓存
            cache_data = {
                'descriptors': self.map_database,
                'files': self.map_files
            }
            with open(descriptor_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"💾 描述子缓存已保存到: {descriptor_cache_path}")
    
    def _compute_map_descriptors(self, map_path: Path):
        """计算地图数据库中所有位置的描述子"""
        ply_files = sorted(list(map_path.glob("*.ply")))
        
        if len(ply_files) == 0:
            raise ValueError(f"地图数据库中没有PLY文件: {map_path}")
        
        print(f"📍 找到 {len(ply_files)} 个地图位置文件")
        
        for i, ply_file in enumerate(ply_files):
            try:
                # 提取描述子
                descriptor, _ = self._extract_descriptor_from_ply(str(ply_file))
                
                # 存储到数据库
                self.map_database[i] = descriptor
                self.map_files.append(str(ply_file))
                
                if (i + 1) % 50 == 0:
                    print(f"  处理进度: {i + 1}/{len(ply_files)}")
                    
            except Exception as e:
                print(f"⚠️  处理文件失败 {ply_file.name}: {e}")
                continue
        
        print(f"✅ 地图数据库构建完成，共 {len(self.map_database)} 个有效位置")
    
    def _extract_descriptor_from_ply(self, ply_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """从PLY文件提取描述子"""
        # 读取点云
        points = PLYReader.read_ply_file(ply_file_path)
        
        # 生成ScanContext
        scan_context = self.sc_generator.make_scan_context(points)
        
        # 转换为tensor
        sc_tensor = torch.from_numpy(scan_context).unsqueeze(0).unsqueeze(0).float()
        sc_tensor = sc_tensor.to(self.device)
        
        # 提取描述子
        with torch.no_grad():
            descriptor = self.model(sc_tensor)
            descriptor = descriptor.cpu().numpy().flatten()
        
        return descriptor, scan_context
    
    def _calculate_similarity(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """计算两个描述子之间的余弦相似度"""
        dot_product = np.dot(desc1, desc2)
        norm1 = np.linalg.norm(desc1)
        norm2 = np.linalg.norm(desc2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return similarity
    
    def localize_from_ply(self, query_ply_path: str, top_k: int = 5) -> Dict:
        """
        从PLY文件进行位置定位
        
        参数:
            query_ply_path (str): 查询PLY文件路径
            top_k (int): 返回前k个最相似的位置
            
        返回:
            定位结果字典
        """
        # 提取查询描述子
        query_desc, query_sc = self._extract_descriptor_from_ply(query_ply_path)
        
        return self._localize_from_descriptor(query_desc, top_k, query_ply_path)
    
    def localize_from_points(self, points: np.ndarray, top_k: int = 5) -> Dict:
        """
        从点云数据进行位置定位
        
        参数:
            points (np.ndarray): 点云数据 (N, 3)
            top_k (int): 返回前k个最相似的位置
            
        返回:
            定位结果字典
        """
        # 生成ScanContext
        scan_context = self.sc_generator.make_scan_context(points)
        
        # 转换为tensor
        sc_tensor = torch.from_numpy(scan_context).unsqueeze(0).unsqueeze(0).float()
        sc_tensor = sc_tensor.to(self.device)
        
        # 提取描述子
        with torch.no_grad():
            query_desc = self.model(sc_tensor)
            query_desc = query_desc.cpu().numpy().flatten()
        
        return self._localize_from_descriptor(query_desc, top_k, "点云数据")
    
    def _localize_from_descriptor(self, query_desc: np.ndarray, top_k: int, source: str) -> Dict:
        """从描述子进行位置定位"""
        similarities = []
        
        # 计算与所有地图位置的相似度
        for position_idx, map_desc in self.map_database.items():
            similarity = self._calculate_similarity(query_desc, map_desc)
            similarities.append({
                'position_index': position_idx,
                'similarity': similarity,
                'map_file': Path(self.map_files[position_idx]).name
            })
        
        # 按相似度排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 获取最佳匹配
        best_match = similarities[0]
        
        # 构建结果
        result = {
            'query_source': source,
            'best_position_index': best_match['position_index'],
            'best_similarity': best_match['similarity'],
            'best_map_file': best_match['map_file'],
            'confidence': 'high' if best_match['similarity'] > 0.8 else 
                         'medium' if best_match['similarity'] > 0.6 else 'low',
            'top_k_candidates': similarities[:top_k],
            'total_map_positions': len(self.map_database)
        }
        
        return result
    
    def get_position_info(self, position_index: int) -> Dict:
        """获取指定位置的信息"""
        if position_index not in self.map_database:
            return None

        return {
            'position_index': position_index,
            'map_file': Path(self.map_files[position_index]).name,
            'map_file_path': self.map_files[position_index],
            'has_descriptor': True
        }

def main():
    parser = argparse.ArgumentParser(description='基于点云的位置定位')
    parser.add_argument('--model', type=str, required=True,
                       help='训练好的模型文件路径')
    parser.add_argument('--map_database', type=str, required=True,
                       help='地图数据库目录路径（包含所有参考位置的PLY文件）')
    parser.add_argument('--query', type=str, required=True,
                       help='查询PLY文件路径（当前位置的点云）')
    parser.add_argument('--top_k', type=int, default=5,
                       help='返回前k个最相似的位置')
    parser.add_argument('--output', type=str, default=None,
                       help='输出结果文件路径')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备类型 (cpu, 0, 1, ...)')

    args = parser.parse_args()

    # 设置日志
    timestamp = get_timestamp()
    logger = setup_logger('localize', f"localize_{timestamp}.log")

    logger.info("🚀 开始位置定位")
    logger.info(f"📍 模型: {args.model}")
    logger.info(f"🗺️  地图数据库: {args.map_database}")
    logger.info(f"🔍 查询文件: {args.query}")
    logger.info(f"📊 返回前 {args.top_k} 个候选位置")

    # 检查文件存在性
    if not Path(args.model).exists():
        logger.error(f"❌ 模型文件不存在: {args.model}")
        return

    if not Path(args.query).exists():
        logger.error(f"❌ 查询文件不存在: {args.query}")
        return

    if not Path(args.map_database).exists():
        logger.error(f"❌ 地图数据库目录不存在: {args.map_database}")
        return

    try:
        # 创建位置定位器
        logger.info("🔄 初始化位置定位器...")
        localizer = PositionLocalizer(args.model, args.map_database, args.device)

        # 执行位置定位
        logger.info("🔍 执行位置定位...")
        start_time = time.time()
        result = localizer.localize_from_ply(args.query, args.top_k)
        end_time = time.time()

        # 显示结果
        logger.info("✅ 位置定位完成！")
        logger.info(f"⏱️  处理时间: {end_time - start_time:.2f}秒")
        logger.info(f"🎯 最佳匹配位置: {result['best_position_index']}")
        logger.info(f"📄 对应文件: {result['best_map_file']}")
        logger.info(f"🔗 相似度: {result['best_similarity']:.4f}")
        logger.info(f"🎚️  置信度: {result['confidence']}")

        logger.info(f"\n📋 前 {args.top_k} 个候选位置:")
        for i, candidate in enumerate(result['top_k_candidates']):
            logger.info(f"  {i+1}. 位置 {candidate['position_index']}: "
                       f"相似度={candidate['similarity']:.4f}, "
                       f"文件={candidate['map_file']}")

        # 保存结果
        output_data = {
            'localization_result': result,
            'processing_time': end_time - start_time,
            'timestamp': timestamp,
            'parameters': {
                'model_path': args.model,
                'map_database_path': args.map_database,
                'query_file': args.query,
                'top_k': args.top_k,
                'device': args.device
            }
        }

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(f"localization_result_{timestamp}.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"💾 结果保存至: {output_path}")

        # 输出关键信息到控制台
        print(f"\n🎯 定位结果:")
        print(f"   当前位置索引: {result['best_position_index']}")
        print(f"   置信度: {result['confidence']}")
        print(f"   相似度: {result['best_similarity']:.4f}")
        print(f"   对应地图文件: {result['best_map_file']}")

    except Exception as e:
        logger.error(f"❌ 位置定位失败: {e}")
        raise

if __name__ == "__main__":
    main()
