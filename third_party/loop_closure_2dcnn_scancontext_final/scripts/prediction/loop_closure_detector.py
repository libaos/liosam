#!/usr/bin/env python3
"""
回环检测预测器
支持单个查询和批量查询的回环检测
"""
import argparse
import torch
import numpy as np
from pathlib import Path
import json
import time
import sys
from typing import List, Dict, Tuple

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import get_config
from models import SCRingCNN, SCStandardCNN, SCStandardCNNLite, SimpleCNN, SimpleCNNLite
from utils import ScanContext, PLYReader, setup_model_logger, get_timestamp

class LoopClosureDetector:
    """回环检测器"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        初始化回环检测器
        
        参数:
            model_path: 模型文件路径
            device: 设备类型
        """
        self.device = torch.device(device)
        self.model = None
        self.model_type = None
        self.config = None
        self.sc_generator = None
        
        # 加载模型
        self._load_model(model_path)
        
        # 创建ScanContext生成器
        self.sc_generator = ScanContext()
        
    def _load_model(self, model_path: str):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 获取配置
        self.config = checkpoint.get('config', {})
        self.model_type = checkpoint.get('model_type', 'SCRingCNN')
        
        # 创建模型
        if self.model_type == 'SCRingCNN':
            self.model = SCRingCNN(
                num_rings=self.config.get('num_rings', 20),
                num_sectors=self.config.get('num_sectors', 60),
                descriptor_dim=self.config.get('descriptor_dim', 256)
            )
        elif self.model_type == 'SCStandardCNN':
            self.model = SCStandardCNN(
                num_rings=self.config.get('num_rings', 20),
                num_sectors=self.config.get('num_sectors', 60),
                descriptor_dim=self.config.get('descriptor_dim', 256),
                use_residual=self.config.get('use_residual', True)
            )
        elif self.model_type == 'SCStandardCNNLite':
            self.model = SCStandardCNNLite(
                num_rings=self.config.get('num_rings', 20),
                num_sectors=self.config.get('num_sectors', 60),
                descriptor_dim=self.config.get('descriptor_dim', 128)
            )
        elif self.model_type == 'SimpleCNN':
            self.model = SimpleCNN(
                num_rings=self.config.get('num_rings', 20),
                num_sectors=self.config.get('num_sectors', 60),
                descriptor_dim=self.config.get('descriptor_dim', 256)
            )
        elif self.model_type == 'SimpleCNNLite':
            self.model = SimpleCNNLite(
                num_rings=self.config.get('num_rings', 20),
                num_sectors=self.config.get('num_sectors', 60),
                descriptor_dim=self.config.get('descriptor_dim', 128)
            )
        else:
            raise ValueError(f"未知的模型类型: {self.model_type}")
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 模型加载成功: {self.model_type}")
        print(f"📊 模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def extract_descriptor(self, ply_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        从PLY文件提取描述子
        
        参数:
            ply_file_path: PLY文件路径
            
        返回:
            descriptor: 描述子向量
            scan_context: ScanContext特征图
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
    
    def calculate_similarity(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """
        计算两个描述子之间的相似度
        
        参数:
            desc1: 第一个描述子
            desc2: 第二个描述子
            
        返回:
            similarity: 余弦相似度
        """
        # 计算余弦相似度
        dot_product = np.dot(desc1, desc2)
        norm1 = np.linalg.norm(desc1)
        norm2 = np.linalg.norm(desc2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def detect_loop_closure(self, query_ply: str, database_plys: List[str], 
                          threshold: float = 0.8, top_k: int = 10) -> Dict:
        """
        检测回环
        
        参数:
            query_ply: 查询PLY文件路径
            database_plys: 数据库PLY文件路径列表
            threshold: 相似度阈值
            top_k: 返回前k个结果
            
        返回:
            results: 回环检测结果
        """
        start_time = time.time()
        
        # 提取查询描述子
        print(f"🔍 处理查询文件: {Path(query_ply).name}")
        query_desc, query_sc = self.extract_descriptor(query_ply)
        
        results = []
        
        print(f"📊 处理数据库文件: {len(database_plys)} 个")
        for i, db_ply in enumerate(database_plys):
            try:
                # 显示进度
                if (i + 1) % 50 == 0 or i == len(database_plys) - 1:
                    print(f"  进度: {i + 1}/{len(database_plys)}")
                
                # 提取数据库描述子
                db_desc, db_sc = self.extract_descriptor(db_ply)
                
                # 计算相似度
                similarity = self.calculate_similarity(query_desc, db_desc)
                
                # 判断是否为回环
                is_loop = similarity > threshold
                
                results.append({
                    'database_file': str(db_ply),
                    'database_name': Path(db_ply).name,
                    'similarity': similarity,
                    'is_loop': is_loop
                })
                
            except Exception as e:
                print(f"⚠️  处理文件失败 {Path(db_ply).name}: {e}")
                continue
        
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 统计结果
        total_time = time.time() - start_time
        loop_count = sum(1 for r in results if r['is_loop'])
        
        # 返回结果
        detection_results = {
            'query_file': str(query_ply),
            'query_name': Path(query_ply).name,
            'model_type': self.model_type,
            'database_size': len(database_plys),
            'threshold': threshold,
            'processing_time': total_time,
            'loop_candidates': loop_count,
            'top_results': results[:top_k],
            'all_results': results,
            'statistics': {
                'max_similarity': max(r['similarity'] for r in results) if results else 0,
                'min_similarity': min(r['similarity'] for r in results) if results else 0,
                'avg_similarity': sum(r['similarity'] for r in results) / len(results) if results else 0,
                'processing_speed': len(database_plys) / total_time if total_time > 0 else 0
            }
        }
        
        return detection_results
    
    def batch_detect(self, query_dir: str, database_dir: str, 
                    threshold: float = 0.8, top_k: int = 5) -> Dict:
        """
        批量回环检测
        
        参数:
            query_dir: 查询文件目录
            database_dir: 数据库文件目录
            threshold: 相似度阈值
            top_k: 每个查询返回前k个结果
            
        返回:
            batch_results: 批量检测结果
        """
        query_dir = Path(query_dir)
        database_dir = Path(database_dir)
        
        # 获取文件列表
        query_plys = list(query_dir.glob("*.ply"))
        database_plys = list(database_dir.glob("*.ply"))
        
        print(f"🔍 批量回环检测")
        print(f"  查询文件: {len(query_plys)} 个")
        print(f"  数据库文件: {len(database_plys)} 个")
        
        batch_results = {
            'query_dir': str(query_dir),
            'database_dir': str(database_dir),
            'model_type': self.model_type,
            'threshold': threshold,
            'top_k': top_k,
            'query_count': len(query_plys),
            'database_count': len(database_plys),
            'results': []
        }
        
        start_time = time.time()
        
        for i, query_ply in enumerate(query_plys):
            print(f"\n📁 处理查询 {i + 1}/{len(query_plys)}: {query_ply.name}")
            
            try:
                result = self.detect_loop_closure(
                    str(query_ply), 
                    [str(p) for p in database_plys], 
                    threshold, 
                    top_k
                )
                batch_results['results'].append(result)
                
            except Exception as e:
                print(f"⚠️  查询失败 {query_ply.name}: {e}")
                continue
        
        batch_results['total_time'] = time.time() - start_time
        batch_results['avg_time_per_query'] = batch_results['total_time'] / len(query_plys) if query_plys else 0
        
        return batch_results

def main():
    parser = argparse.ArgumentParser(description='回环检测预测器')
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--query', type=str, required=True,
                       help='查询PLY文件路径或目录')
    parser.add_argument('--database', type=str, required=True,
                       help='数据库目录路径')
    parser.add_argument('--threshold', type=float, default=0.8,
                       help='相似度阈值')
    parser.add_argument('--top_k', type=int, default=10,
                       help='返回前k个结果')
    parser.add_argument('--output', type=str, default=None,
                       help='输出结果文件路径')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备类型')
    parser.add_argument('--batch', action='store_true',
                       help='批量模式（查询为目录）')
    
    args = parser.parse_args()
    
    # 设置日志
    timestamp = get_timestamp()
    project_root = Path(__file__).parent.parent.parent

    # 从模型文件推断模型类型
    model_type = 'general'
    try:
        if Path(args.model).exists():
            checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
            model_type_from_checkpoint = checkpoint.get('model_type', 'general')
            if model_type_from_checkpoint in ['SCRingCNN', 'SCStandardCNN', 'SimpleCNN', 'SimpleCNNLite']:
                model_type = model_type_from_checkpoint.lower().replace('cnn', '_cnn')
    except:
        pass

    logger, log_file = setup_model_logger(
        model_type=model_type,
        script_type='prediction',
        timestamp=timestamp,
        project_root=project_root
    )
    
    print("🚀 回环检测预测器")
    print("="*50)
    print(f"模型: {args.model}")
    print(f"查询: {args.query}")
    print(f"数据库: {args.database}")
    print(f"阈值: {args.threshold}")
    print(f"设备: {args.device}")
    print(f"批量模式: {args.batch}")
    
    # 检查文件存在性
    if not Path(args.model).exists():
        print(f"❌ 模型文件不存在: {args.model}")
        return
    
    if not Path(args.query).exists():
        print(f"❌ 查询文件/目录不存在: {args.query}")
        return
    
    if not Path(args.database).exists():
        print(f"❌ 数据库目录不存在: {args.database}")
        return
    
    # 创建检测器
    print("\n📥 加载模型...")
    detector = LoopClosureDetector(args.model, args.device)
    
    # 执行检测
    if args.batch:
        # 批量检测
        print("\n🔄 开始批量回环检测...")
        results = detector.batch_detect(
            args.query, args.database, args.threshold, args.top_k
        )
    else:
        # 单个检测
        database_plys = list(Path(args.database).glob("*.ply"))
        print(f"\n🔍 开始回环检测...")
        print(f"数据库中有 {len(database_plys)} 个PLY文件")
        
        results = detector.detect_loop_closure(
            args.query, [str(p) for p in database_plys], args.threshold, args.top_k
        )
    
    # 显示结果
    if args.batch:
        print(f"\n📊 批量检测完成")
        print(f"处理查询: {results['query_count']} 个")
        print(f"总耗时: {results['total_time']:.2f}s")
        print(f"平均每查询: {results['avg_time_per_query']:.2f}s")
    else:
        print(f"\n📊 检测完成")
        print(f"处理时间: {results['processing_time']:.2f}s")
        print(f"回环候选: {results['loop_candidates']} 个")
        print(f"处理速度: {results['statistics']['processing_speed']:.2f} files/s")
        
        print(f"\n🏆 Top {args.top_k} 相似结果:")
        for i, result in enumerate(results['top_results']):
            status = "✅ 回环" if result['is_loop'] else "❌ 非回环"
            print(f"  {i+1}. {result['database_name']}: "
                  f"相似度={result['similarity']:.4f} {status}")
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = project_root / "outputs" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        if args.batch:
            output_path = results_dir / f"batch_loop_closure_{timestamp}.json"
        else:
            query_name = Path(args.query).stem
            output_path = results_dir / f"loop_closure_{query_name}_{timestamp}.json"
    
    # 添加时间戳
    results['timestamp'] = timestamp
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 结果保存至: {output_path}")

if __name__ == "__main__":
    main()
