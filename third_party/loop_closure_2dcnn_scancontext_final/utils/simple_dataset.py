"""
简化的数据集类，用于回环检测训练
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import glob
import pickle
from .scan_context import ScanContext
from .ply_reader import PLYReader

class SimpleLoopClosureDataset(Dataset):
    """简化的回环检测数据集"""
    
    def __init__(self, data_dir, cache_dir=None, max_files=None):
        """
        初始化数据集
        
        参数:
            data_dir: PLY文件目录
            cache_dir: 缓存目录
            max_files: 最大文件数量
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # 查找PLY文件
        self.ply_files = sorted(glob.glob(str(self.data_dir / "*.ply")))
        
        if max_files:
            self.ply_files = self.ply_files[:max_files]
        
        if len(self.ply_files) == 0:
            raise ValueError(f"在 {self.data_dir} 中未找到PLY文件")
        
        print(f"找到 {len(self.ply_files)} 个PLY文件")
        
        # 创建ScanContext生成器
        self.sc_generator = ScanContext()
        
        # 生成标签（简单地使用文件索引作为类别）
        self.labels = self._generate_labels()
        
        # 缓存
        self.cache = {}
        
    def _generate_labels(self):
        """生成简单的标签"""
        # 每5个文件为一个类别
        labels = []
        for i, _ in enumerate(self.ply_files):
            label = i // 5  # 每5个文件一个类别
            labels.append(label)
        return labels
    
    def _load_point_cloud(self, file_path):
        """加载点云"""
        try:
            # 尝试读取真实的PLY文件
            points = PLYReader.read_ply_file(file_path)

            if len(points) == 0:
                raise ValueError(f"点云文件为空: {file_path}")

            return points.astype(np.float32)
        except Exception as e:
            print(f"加载点云失败 {file_path}: {e}")
            print("使用随机点云作为替代")
            # 返回一个基于文件路径的确定性随机点云
            np.random.seed(hash(file_path) % 2**32)
            points = np.random.rand(1000, 3) * 20 - 10  # 生成-10到10范围的点云
            return points.astype(np.float32)
    
    def _generate_scan_context(self, points):
        """生成ScanContext特征"""
        try:
            # 使用ScanContext生成器
            sc_matrix = self.sc_generator.generate_scan_context(points)
            
            # 确保输出形状正确
            if sc_matrix.shape != (20, 60):
                # 如果形状不对，调整大小
                from scipy.ndimage import zoom
                scale_h = 20 / sc_matrix.shape[0]
                scale_w = 60 / sc_matrix.shape[1]
                sc_matrix = zoom(sc_matrix, (scale_h, scale_w))
            
            return sc_matrix.astype(np.float32)
        except Exception as e:
            print(f"生成ScanContext失败: {e}")
            # 返回默认的ScanContext
            return np.random.rand(20, 60).astype(np.float32)
    
    def __len__(self):
        return len(self.ply_files)
    
    def __getitem__(self, idx):
        """获取数据项"""
        if idx in self.cache:
            return self.cache[idx]
        
        # 加载点云
        file_path = self.ply_files[idx]
        points = self._load_point_cloud(file_path)
        
        # 生成ScanContext
        sc_matrix = self._generate_scan_context(points)
        
        # 转换为tensor
        sc_tensor = torch.from_numpy(sc_matrix).unsqueeze(0)  # 添加通道维度
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # 缓存结果
        result = (sc_tensor, label)
        self.cache[idx] = result
        
        return result
