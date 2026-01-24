#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import time

class ScanContext:
    """
    ScanContext: 点云场景描述子生成器
    
    该类实现了ScanContext算法，用于从LiDAR点云数据生成场景描述子。
    ScanContext是一种基于高度信息的极坐标网格表示，可用于场景识别和回环检测。
    """
    
    def __init__(self, num_sectors=60, num_rings=20, 
                 min_range=0.1, max_range=80.0, 
                 height_lower_bound=-1.0, height_upper_bound=9.0,
                 use_intensity=False):
        """
        初始化ScanContext生成器
        
        参数:
            num_sectors (int): 极坐标中的扇区数量（角度维度）
            num_rings (int): 极坐标中的环数量（径向维度）
            min_range (float): 考虑点云的最小距离（米）
            max_range (float): 考虑点云的最大距离（米）
            height_lower_bound (float): 高度下限（米）
            height_upper_bound (float): 高度上限（米）
            use_intensity (bool): 是否使用点云强度信息
        """
        self.num_sectors = num_sectors
        self.num_rings = num_rings
        self.min_range = min_range
        self.max_range = max_range
        self.height_lower_bound = height_lower_bound
        self.height_upper_bound = height_upper_bound
        self.use_intensity = use_intensity
        
        # 计算每个环的宽度
        self.ring_width = (max_range - min_range) / num_rings
        # 计算每个扇区的角度（弧度）
        self.sector_angle = 2 * np.pi / num_sectors
        
    def _xy_to_polar(self, x, y):
        """
        将笛卡尔坐标转换为极坐标
        
        参数:
            x (float): x坐标
            y (float): y坐标
            
        返回:
            r (float): 径向距离
            theta (float): 角度（弧度）
        """
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # 确保theta在[0, 2*pi)范围内
        if theta < 0:
            theta += 2 * np.pi
            
        return r, theta
    
    def _get_ring_idx(self, r):
        """
        根据径向距离获取环索引
        
        参数:
            r (float): 径向距离
            
        返回:
            ring_idx (int): 环索引
        """
        ring_idx = int((r - self.min_range) / self.ring_width)
        # 确保索引在有效范围内
        if ring_idx < 0:
            ring_idx = 0
        if ring_idx >= self.num_rings:
            ring_idx = self.num_rings - 1
            
        return ring_idx
    
    def _get_sector_idx(self, theta):
        """
        根据角度获取扇区索引
        
        参数:
            theta (float): 角度（弧度）
            
        返回:
            sector_idx (int): 扇区索引
        """
        sector_idx = int(theta / self.sector_angle)
        # 确保索引在有效范围内
        if sector_idx < 0:
            sector_idx = 0
        if sector_idx >= self.num_sectors:
            sector_idx = self.num_sectors - 1
            
        return sector_idx
    
    def make_scan_context(self, point_cloud):
        """
        从点云生成ScanContext特征图
        
        参数:
            point_cloud (numpy.ndarray): 形状为(N, 3)或(N, 4)的点云数据，
                                        每行为[x, y, z]或[x, y, z, intensity]
                                        
        返回:
            scan_context (numpy.ndarray): 形状为(num_rings, num_sectors)的ScanContext特征图
        """
        # 初始化ScanContext矩阵，填充为高度下限值
        scan_context = np.full((self.num_rings, self.num_sectors), self.height_lower_bound)
        
        # 创建辅助矩阵用于记录每个单元格中的点数
        point_count = np.zeros((self.num_rings, self.num_sectors), dtype=np.int32)
        
        # 处理每个点
        for i in range(point_cloud.shape[0]):
            x, y, z = point_cloud[i, 0:3]
            
            # 过滤掉超出范围的点
            r, theta = self._xy_to_polar(x, y)
            if r < self.min_range or r > self.max_range:
                continue
                
            # 过滤掉超出高度范围的点
            if z < self.height_lower_bound or z > self.height_upper_bound:
                continue
                
            # 获取点所在的环和扇区索引
            ring_idx = self._get_ring_idx(r)
            sector_idx = self._get_sector_idx(theta)
            
            # 更新ScanContext矩阵
            # 如果当前点的高度大于已记录的高度，则更新
            if z > scan_context[ring_idx, sector_idx]:
                scan_context[ring_idx, sector_idx] = z
                
            # 增加点计数
            point_count[ring_idx, sector_idx] += 1
        
        # 对没有点的单元格进行特殊处理
        # 将没有点的单元格设置为0（或其他特殊值）
        scan_context[point_count == 0] = 0
        
        # 归一化ScanContext
        # 将高度值归一化到[0, 1]范围
        if np.max(scan_context) > np.min(scan_context):
            scan_context = (scan_context - np.min(scan_context)) / (np.max(scan_context) - np.min(scan_context))
        
        return scan_context
    
    def make_ring_key(self, scan_context):
        """
        从ScanContext特征图生成Ring Key（用于快速检索）
        
        参数:
            scan_context (numpy.ndarray): 形状为(num_rings, num_sectors)的ScanContext特征图
            
        返回:
            ring_key (numpy.ndarray): 形状为(num_rings,)的Ring Key
        """
        # 计算每个环的平均高度作为Ring Key
        ring_key = np.mean(scan_context, axis=1)
        return ring_key
    
    def make_sector_key(self, scan_context):
        """
        从ScanContext特征图生成Sector Key
        
        参数:
            scan_context (numpy.ndarray): 形状为(num_rings, num_sectors)的ScanContext特征图
            
        返回:
            sector_key (numpy.ndarray): 形状为(num_sectors,)的Sector Key
        """
        # 计算每个扇区的平均高度作为Sector Key
        sector_key = np.mean(scan_context, axis=0)
        return sector_key
    
    def calc_dist(self, sc1, sc2, dist_type="L2"):
        """
        计算两个ScanContext特征图之间的距离
        
        参数:
            sc1 (numpy.ndarray): 第一个ScanContext特征图
            sc2 (numpy.ndarray): 第二个ScanContext特征图
            dist_type (str): 距离类型，可选"L1", "L2", "cosine"
            
        返回:
            min_dist (float): 最小距离
            shift_idx (int): 最佳偏移量（扇区索引）
        """
        # 确保两个ScanContext具有相同的形状
        assert sc1.shape == sc2.shape, "两个ScanContext特征图的形状必须相同"
        
        # 计算距离矩阵
        num_sectors = sc1.shape[1]
        dist_mat = np.zeros(num_sectors)
        
        # 对每个可能的偏移量计算距离
        for i in range(num_sectors):
            # 循环移位sc1
            sc1_shifted = np.roll(sc1, i, axis=1)
            
            # 根据距离类型计算距离
            if dist_type == "L1":
                dist = np.mean(np.abs(sc1_shifted - sc2))
            elif dist_type == "L2":
                dist = np.sqrt(np.mean((sc1_shifted - sc2) ** 2))
            elif dist_type == "cosine":
                sc1_flat = sc1_shifted.flatten()
                sc2_flat = sc2.flatten()
                dist = spatial.distance.cosine(sc1_flat, sc2_flat)
            else:
                raise ValueError("不支持的距离类型: " + dist_type)
                
            dist_mat[i] = dist
            
        # 找到最小距离及其对应的偏移量
        min_idx = np.argmin(dist_mat)
        min_dist = dist_mat[min_idx]
        
        return min_dist, min_idx
    
    def load_point_cloud(self, file_path):
        """
        从文件加载点云数据
        
        参数:
            file_path (str): 点云文件路径
            
        返回:
            point_cloud (numpy.ndarray): 点云数据
        """
        # 根据文件扩展名选择加载方法
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.pcd':
            # 简化的PCD文件加载（不使用open3d）
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                # 找到数据开始位置
                data_start = 0
                for i, line in enumerate(lines):
                    if line.startswith('DATA'):
                        data_start = i + 1
                        break

                # 读取点云数据
                points = []
                for i in range(data_start, len(lines)):
                    parts = lines[i].strip().split()
                    if len(parts) >= 3:
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            points.append([x, y, z])
                        except ValueError:
                            continue

                point_cloud = np.array(points) if points else np.empty((0, 3))

            except Exception as e:
                print(f"加载PCD文件失败: {e}")
                return None
                
        elif ext.lower() == '.bin':
            # 加载KITTI格式的二进制点云文件
            point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            
        elif ext.lower() == '.txt' or ext.lower() == '.xyz':
            # 加载文本格式的点云文件
            point_cloud = np.loadtxt(file_path, delimiter=',')
            
        elif ext.lower() == '.npy':
            # 加载NumPy格式的点云文件
            point_cloud = np.load(file_path)

        elif ext.lower() == '.ply':
            # 加载PLY格式的点云文件
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(file_path)
                point_cloud = np.asarray(pcd.points)
                # 如果只有xyz坐标，添加一个强度列（全为1）
                if point_cloud.shape[1] == 3:
                    point_cloud = np.hstack([point_cloud, np.ones((point_cloud.shape[0], 1))])
            except ImportError:
                print("警告: 未安装open3d，尝试使用plyfile库")
                try:
                    from plyfile import PlyData
                    plydata = PlyData.read(file_path)
                    vertex = plydata['vertex']
                    point_cloud = np.array([vertex['x'], vertex['y'], vertex['z']]).T
                    # 添加强度列
                    point_cloud = np.hstack([point_cloud, np.ones((point_cloud.shape[0], 1))])
                except ImportError:
                    print("错误: 需要安装open3d或plyfile库来读取PLY文件")
                    return None
            except Exception as e:
                print(f"加载PLY文件失败: {e}")
                return None

        else:
            raise ValueError("不支持的点云文件格式: " + ext)
            
        return point_cloud
    
    def process_point_cloud_file(self, file_path):
        """
        处理点云文件并生成ScanContext特征图
        
        参数:
            file_path (str): 点云文件路径
            
        返回:
            scan_context (numpy.ndarray): ScanContext特征图
            ring_key (numpy.ndarray): Ring Key
        """
        # 加载点云
        point_cloud = self.load_point_cloud(file_path)
        
        # 生成ScanContext特征图
        scan_context = self.make_scan_context(point_cloud)
        
        # 生成Ring Key
        ring_key = self.make_ring_key(scan_context)
        
        return scan_context, ring_key

    def generate_scan_context(self, point_cloud):
        """
        生成ScanContext特征图（兼容性方法）

        参数:
            point_cloud (numpy.ndarray): 点云数据

        返回:
            scan_context (numpy.ndarray): ScanContext特征图
        """
        return self.make_scan_context(point_cloud)