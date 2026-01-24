#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import struct
import os

# 尝试导入PLY读取库
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False

class PLYReader:
    """
    PLY文件读取器
    
    支持读取ASCII和二进制格式的PLY点云文件
    """
    
    @staticmethod
    def read_ply_file(file_path):
        """
        读取PLY文件

        参数:
            file_path (str): PLY文件路径

        返回:
            points (numpy.ndarray): 点云数据，形状为(N, 3)或(N, 4)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PLY文件不存在: {file_path}")

        # 优先使用open3d
        if HAS_OPEN3D:
            try:
                pcd = o3d.io.read_point_cloud(file_path)
                points = np.asarray(pcd.points, dtype=np.float32)
                if len(points) > 0:
                    return points
            except Exception as e:
                print(f"警告: open3d读取失败: {e}")

        # 其次使用plyfile
        if HAS_PLYFILE:
            try:
                plydata = PlyData.read(file_path)
                vertex = plydata['vertex']

                # 提取x, y, z坐标
                x = vertex['x']
                y = vertex['y']
                z = vertex['z']

                points = np.column_stack((x, y, z)).astype(np.float32)
                if len(points) > 0:
                    return points
            except Exception as e:
                print(f"警告: plyfile读取失败: {e}")

        # 最后使用自定义读取器
        print("警告: 未安装open3d或plyfile，使用自定义读取器")
        try:
            with open(file_path, 'rb') as f:
                # 读取头部信息
                header_info = PLYReader._parse_header(f)

                # 根据格式读取数据
                if header_info['format'] == 'ascii':
                    points = PLYReader._read_ascii_data(f, header_info)
                else:
                    points = PLYReader._read_binary_data(f, header_info)

            return points
        except Exception as e:
            print(f"错误: 自定义读取器也失败: {e}")
            return None
    
    @staticmethod
    def _parse_header(file_handle):
        """
        解析PLY文件头部
        
        参数:
            file_handle: 文件句柄
            
        返回:
            header_info (dict): 头部信息
        """
        header_info = {
            'format': 'ascii',
            'vertex_count': 0,
            'properties': [],
            'data_start_pos': 0
        }
        
        line = file_handle.readline().decode('utf-8').strip()
        if line != 'ply':
            raise ValueError("不是有效的PLY文件")
            
        while True:
            line = file_handle.readline().decode('utf-8').strip()
            
            if line.startswith('format'):
                parts = line.split()
                header_info['format'] = parts[1]
                
            elif line.startswith('element vertex'):
                parts = line.split()
                header_info['vertex_count'] = int(parts[2])
                
            elif line.startswith('property'):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                header_info['properties'].append((prop_type, prop_name))
                
            elif line == 'end_header':
                header_info['data_start_pos'] = file_handle.tell()
                break
                
        return header_info
    
    @staticmethod
    def _read_ascii_data(file_handle, header_info):
        """
        读取ASCII格式的PLY数据
        
        参数:
            file_handle: 文件句柄
            header_info (dict): 头部信息
            
        返回:
            points (numpy.ndarray): 点云数据
        """
        vertex_count = header_info['vertex_count']
        properties = header_info['properties']
        
        # 找到x, y, z坐标的索引
        coord_indices = {}
        for i, (prop_type, prop_name) in enumerate(properties):
            if prop_name in ['x', 'y', 'z']:
                coord_indices[prop_name] = i
                
        if len(coord_indices) < 3:
            raise ValueError("PLY文件缺少x, y, z坐标信息")
            
        # 读取数据
        points = []
        for _ in range(vertex_count):
            line = file_handle.readline().decode('utf-8').strip()
            if not line:
                break
                
            values = line.split()
            if len(values) >= len(properties):
                try:
                    x = float(values[coord_indices['x']])
                    y = float(values[coord_indices['y']])
                    z = float(values[coord_indices['z']])
                    points.append([x, y, z])
                except (ValueError, IndexError):
                    continue
                    
        return np.array(points, dtype=np.float32)
    
    @staticmethod
    def _read_binary_data(file_handle, header_info):
        """
        读取二进制格式的PLY数据
        
        参数:
            file_handle: 文件句柄
            header_info (dict): 头部信息
            
        返回:
            points (numpy.ndarray): 点云数据
        """
        vertex_count = header_info['vertex_count']
        properties = header_info['properties']
        
        # 构建数据类型格式字符串
        format_str = '<' if header_info['format'] == 'binary_little_endian' else '>'
        
        # 找到x, y, z坐标的索引和类型
        coord_info = {}
        struct_format = ''
        
        for i, (prop_type, prop_name) in enumerate(properties):
            if prop_type == 'float':
                struct_format += 'f'
            elif prop_type == 'double':
                struct_format += 'd'
            elif prop_type in ['int', 'int32']:
                struct_format += 'i'
            elif prop_type in ['uint', 'uint32']:
                struct_format += 'I'
            elif prop_type in ['short', 'int16']:
                struct_format += 'h'
            elif prop_type in ['ushort', 'uint16']:
                struct_format += 'H'
            elif prop_type in ['char', 'int8']:
                struct_format += 'b'
            elif prop_type in ['uchar', 'uint8']:
                struct_format += 'B'
            else:
                struct_format += 'f'  # 默认为float
                
            if prop_name in ['x', 'y', 'z']:
                coord_info[prop_name] = i
                
        if len(coord_info) < 3:
            raise ValueError("PLY文件缺少x, y, z坐标信息")
            
        format_str += struct_format
        struct_size = struct.calcsize(format_str)
        
        # 读取数据
        points = []
        for _ in range(vertex_count):
            data = file_handle.read(struct_size)
            if len(data) < struct_size:
                break
                
            try:
                values = struct.unpack(format_str, data)
                x = float(values[coord_info['x']])
                y = float(values[coord_info['y']])
                z = float(values[coord_info['z']])
                points.append([x, y, z])
            except (struct.error, IndexError):
                continue
                
        return np.array(points, dtype=np.float32)
    
    @staticmethod
    def get_ply_info(file_path):
        """
        获取PLY文件信息
        
        参数:
            file_path (str): PLY文件路径
            
        返回:
            info (dict): 文件信息
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PLY文件不存在: {file_path}")
            
        with open(file_path, 'rb') as f:
            header_info = PLYReader._parse_header(f)
            
        info = {
            'format': header_info['format'],
            'vertex_count': header_info['vertex_count'],
            'properties': [prop_name for _, prop_name in header_info['properties']],
            'file_size': os.path.getsize(file_path)
        }
        
        return info
    
    @staticmethod
    def validate_ply_file(file_path):
        """
        验证PLY文件是否有效
        
        参数:
            file_path (str): PLY文件路径
            
        返回:
            is_valid (bool): 是否有效
            error_msg (str): 错误信息
        """
        try:
            info = PLYReader.get_ply_info(file_path)
            
            if info['vertex_count'] == 0:
                return False, "PLY文件中没有顶点数据"
                
            if 'x' not in info['properties'] or 'y' not in info['properties'] or 'z' not in info['properties']:
                return False, "PLY文件缺少x, y, z坐标信息"
                
            # 尝试读取少量数据进行验证
            points = PLYReader.read_ply_file(file_path)
            if len(points) == 0:
                return False, "无法读取PLY文件数据"
                
            return True, "PLY文件有效"
            
        except Exception as e:
            return False, f"PLY文件验证失败: {str(e)}"

def test_ply_reader():
    """
    测试PLY读取器
    """
    # 这里可以添加测试代码
    print("PLY读取器测试")
    
    # 示例：创建一个简单的测试PLY文件
    test_ply_content = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
end_header
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
"""
    
    # 写入测试文件
    test_file = "test_points.ply"
    with open(test_file, 'w') as f:
        f.write(test_ply_content)
    
    try:
        # 测试读取
        points = PLYReader.read_ply_file(test_file)
        print(f"读取到 {len(points)} 个点:")
        print(points)
        
        # 测试文件信息
        info = PLYReader.get_ply_info(test_file)
        print(f"文件信息: {info}")
        
        # 测试验证
        is_valid, msg = PLYReader.validate_ply_file(test_file)
        print(f"文件验证: {is_valid}, {msg}")
        
    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    test_ply_reader()
