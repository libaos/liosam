#!/usr/bin/env python3
"""
回环检测预测脚本
"""
import argparse
import torch
import numpy as np
from pathlib import Path
import json
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from models import SCRingCNN, SCStandardCNN, SCStandardCNNLite, SimpleCNN, SimpleCNNLite
from utils import ScanContext, PLYReader, setup_logger, get_timestamp

class LoopClosurePredictor:
    """回环检测预测器"""
    
    def __init__(self, model_path, device='cpu'):
        """
        初始化预测器
        
        参数:
            model_path (str): 模型文件路径
            device (str): 设备类型
        """
        self.device = torch.device(device)
        self.model = None
        self.config = None
        self.sc_generator = None
        
        # 加载模型
        self._load_model(model_path)
        
        # 创建ScanContext生成器
        self.sc_generator = ScanContext()
        
    def _load_model(self, model_path):
        """加载模型"""
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
        
        print(f"模型加载成功: {model_type}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def extract_descriptor_from_ply(self, ply_file_path):
        """
        从PLY文件提取描述子
        
        参数:
            ply_file_path (str): PLY文件路径
            
        返回:
            descriptor (numpy.ndarray): 描述子向量
            scan_context (numpy.ndarray): ScanContext特征图
        """
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
    
    def extract_descriptor_from_points(self, points):
        """
        从点云数据提取描述子
        
        参数:
            points (numpy.ndarray): 点云数据
            
        返回:
            descriptor (numpy.ndarray): 描述子向量
            scan_context (numpy.ndarray): ScanContext特征图
        """
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
    
    def calculate_similarity(self, desc1, desc2):
        """
        计算两个描述子之间的相似度
        
        参数:
            desc1 (numpy.ndarray): 第一个描述子
            desc2 (numpy.ndarray): 第二个描述子
            
        返回:
            similarity (float): 余弦相似度
        """
        # 计算余弦相似度
        dot_product = np.dot(desc1, desc2)
        norm1 = np.linalg.norm(desc1)
        norm2 = np.linalg.norm(desc2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return similarity
    
    def find_loop_closure(self, query_ply, database_plys, threshold=0.8):
        """
        在数据库中查找回环
        
        参数:
            query_ply (str): 查询PLY文件路径
            database_plys (list): 数据库PLY文件路径列表
            threshold (float): 相似度阈值
            
        返回:
            results (list): 回环检测结果
        """
        # 提取查询描述子
        query_desc, query_sc = self.extract_descriptor_from_ply(query_ply)
        
        results = []
        
        for db_ply in database_plys:
            try:
                # 提取数据库描述子
                db_desc, db_sc = self.extract_descriptor_from_ply(db_ply)
                
                # 计算相似度
                similarity = self.calculate_similarity(query_desc, db_desc)
                
                # 判断是否为回环
                is_loop = similarity > threshold
                
                results.append({
                    'database_file': db_ply,
                    'similarity': similarity,
                    'is_loop': is_loop
                })
                
            except Exception as e:
                print(f"处理文件失败 {db_ply}: {e}")
                continue
        
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='回环检测预测')
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--query', type=str, required=True,
                       help='查询PLY文件路径')
    parser.add_argument('--database', type=str, required=True,
                       help='数据库目录路径')
    parser.add_argument('--threshold', type=float, default=0.8,
                       help='相似度阈值')
    parser.add_argument('--output', type=str, default=None,
                       help='输出结果文件路径')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备类型')
    
    args = parser.parse_args()
    
    # 设置日志
    timestamp = get_timestamp()
    logger = setup_logger('predict', f"predict_{timestamp}.log")
    
    logger.info(f"开始回环检测预测")
    logger.info(f"模型: {args.model}")
    logger.info(f"查询文件: {args.query}")
    logger.info(f"数据库目录: {args.database}")
    logger.info(f"相似度阈值: {args.threshold}")
    
    # 检查文件存在性
    if not Path(args.model).exists():
        logger.error(f"模型文件不存在: {args.model}")
        return
    
    if not Path(args.query).exists():
        logger.error(f"查询文件不存在: {args.query}")
        return
    
    if not Path(args.database).exists():
        logger.error(f"数据库目录不存在: {args.database}")
        return
    
    # 创建预测器
    predictor = LoopClosurePredictor(args.model, args.device)
    
    # 获取数据库文件列表
    database_dir = Path(args.database)
    database_plys = list(database_dir.glob("*.ply"))
    
    if len(database_plys) == 0:
        logger.error(f"数据库目录中没有PLY文件: {args.database}")
        return
    
    logger.info(f"数据库中有 {len(database_plys)} 个PLY文件")
    
    # 执行回环检测
    start_time = time.time()
    results = predictor.find_loop_closure(args.query, database_plys, args.threshold)
    end_time = time.time()
    
    # 统计结果
    loop_count = sum(1 for r in results if r['is_loop'])
    
    logger.info(f"回环检测完成，耗时: {end_time - start_time:.2f}秒")
    logger.info(f"找到 {loop_count} 个回环候选")
    
    # 显示前10个结果
    logger.info("Top 10 相似结果:")
    for i, result in enumerate(results[:10]):
        logger.info(f"  {i+1}. {Path(result['database_file']).name}: "
                   f"相似度={result['similarity']:.4f}, "
                   f"回环={'是' if result['is_loop'] else '否'}")
    
    # 保存结果
    output_data = {
        'query_file': args.query,
        'database_dir': args.database,
        'threshold': args.threshold,
        'total_candidates': len(results),
        'loop_candidates': loop_count,
        'processing_time': end_time - start_time,
        'results': results,
        'timestamp': timestamp
    }
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"loop_closure_results_{timestamp}.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"结果保存至: {output_path}")

if __name__ == "__main__":
    main()
