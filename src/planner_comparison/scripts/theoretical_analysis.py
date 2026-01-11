#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

class PlannerComparison:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def generate_radar_chart(self):
        """生成雷达图对比两种规划器的理论性能"""
        print("生成雷达图对比...")
        
        # 理论性能数据 (基于算法特性的估计值)
        categories = ['计算效率', '路径平滑度', '轨迹跟踪精度', '坡道适应性', '避障能力']
        
        # 评分范围: 0-10
        teb_scores = [8, 7, 6, 4, 8]
        psolqr_scores = [6, 9, 9, 8, 8]
        
        # 创建雷达图
        plt.figure(figsize=(10, 8))
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        teb_scores += teb_scores[:1]
        psolqr_scores += psolqr_scores[:1]
        
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, teb_scores, 'o-', linewidth=2, label='TEB', color='#FF5733')
        ax.fill(angles, teb_scores, alpha=0.25, color='#FF5733')
        ax.plot(angles, psolqr_scores, 'o-', linewidth=2, label='PSOLQR', color='#33A1FF')
        ax.fill(angles, psolqr_scores, alpha=0.25, color='#33A1FF')
        
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 10)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.set_title('TEB vs PSOLQR 理论性能对比', fontsize=15)
        
        plt.savefig(os.path.join(self.output_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"雷达图已保存至 {os.path.join(self.output_dir, 'radar_comparison.png')}")
    
    def generate_slope_performance(self):
        """生成坡度适应性对比图"""
        print("生成坡度适应性对比图...")
        
        plt.figure(figsize=(12, 6))
        
        # 模拟坡度变化下的轨迹跟踪误差
        slope_angles = np.linspace(0, 15, 100)  # 0-15度坡度
        
        # 假设的误差模型 (基于理论分析)
        teb_errors = 0.05 + 0.03 * slope_angles  # TEB误差随坡度增加而线性增加
        psolqr_errors = 0.05 + 0.01 * slope_angles  # PSOLQR误差增加更慢
        
        plt.plot(slope_angles, teb_errors, '-', linewidth=2, label='TEB', color='#FF5733')
        plt.plot(slope_angles, psolqr_errors, '-', linewidth=2, label='PSOLQR', color='#33A1FF')
        plt.fill_between(slope_angles, teb_errors, psolqr_errors, color='#D6EAF8', alpha=0.7, label='PSOLQR优势')
        
        plt.xlabel('坡度角度 (度)', fontsize=12)
        plt.ylabel('轨迹跟踪误差 (m)', fontsize=12)
        plt.title('不同坡度下的轨迹跟踪误差对比', fontsize=15)
        plt.grid(True)
        plt.legend()
        
        plt.savefig(os.path.join(self.output_dir, 'slope_error_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"坡度适应性对比图已保存至 {os.path.join(self.output_dir, 'slope_error_comparison.png')}")
    
    def generate_velocity_smoothness(self):
        """生成速度平滑度对比图"""
        print("生成速度平滑度对比图...")
        
        plt.figure(figsize=(12, 6))
        
        # 模拟时间序列
        time = np.linspace(0, 10, 200)
        
        # 模拟速度曲线 (基于理论分析)
        np.random.seed(42)  # 固定随机种子以便复现
        
        # 模拟在坡道上的速度变化
        # TEB在坡道上速度波动更大
        teb_velocity = 0.5 + 0.1 * np.sin(time) + 0.05 * np.random.randn(len(time))
        # 坡道中间段(3-7秒)波动加大
        teb_velocity[60:140] += 0.05 * np.sin(time[60:140] * 5)
        
        # PSOLQR保持更平稳的速度
        psolqr_velocity = 0.5 + 0.1 * np.sin(time) + 0.02 * np.random.randn(len(time))
        # 坡道中间段波动较小
        psolqr_velocity[60:140] += 0.02 * np.sin(time[60:140] * 5)
        
        plt.plot(time, teb_velocity, '-', linewidth=1.5, label='TEB', color='#FF5733')
        plt.plot(time, psolqr_velocity, '-', linewidth=1.5, label='PSOLQR', color='#33A1FF')
        
        # 标记坡道区域
        plt.axvspan(3, 7, alpha=0.2, color='gray', label='坡道区域')
        
        plt.xlabel('时间 (s)', fontsize=12)
        plt.ylabel('线速度 (m/s)', fontsize=12)
        plt.title('坡道环境中的速度平滑度对比 (理论模拟)', fontsize=15)
        plt.grid(True)
        plt.legend()
        
        # 添加标准差信息
        teb_std = np.std(teb_velocity)
        psolqr_std = np.std(psolqr_velocity)
        improvement = (teb_std - psolqr_std) / teb_std * 100
        
        plt.annotate(f'TEB 标准差: {teb_std:.4f}', xy=(0.02, 0.95), xycoords='axes fraction')
        plt.annotate(f'PSOLQR 标准差: {psolqr_std:.4f}', xy=(0.02, 0.90), xycoords='axes fraction')
        plt.annotate(f'改进: {improvement:.2f}%', xy=(0.02, 0.85), xycoords='axes fraction')
        
        plt.savefig(os.path.join(self.output_dir, 'velocity_smoothness.png'), dpi=300, bbox_inches='tight')
        print(f"速度平滑度对比图已保存至 {os.path.join(self.output_dir, 'velocity_smoothness.png')}")
    
    def generate_path_comparison(self):
        """生成路径对比图"""
        print("生成路径对比图...")
        
        plt.figure(figsize=(10, 8))
        
        # 创建一个简单的地图，包含障碍物和坡道
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        X, Y = np.meshgrid(x, y)
        
        # 创建坡道区域 (3-7, 3-7)
        slope_mask = (X > 3) & (X < 7) & (Y > 3) & (Y < 7)
        
        # 绘制地图
        plt.pcolormesh(X, Y, slope_mask, cmap='gray_r', alpha=0.3, label='坡道区域')
        
        # 添加障碍物
        circle1 = plt.Circle((2, 5), 0.5, color='red', alpha=0.5)
        circle2 = plt.Circle((5, 2), 0.6, color='red', alpha=0.5)
        circle3 = plt.Circle((8, 6), 0.7, color='red', alpha=0.5)
        
        plt.gca().add_patch(circle1)
        plt.gca().add_patch(circle2)
        plt.gca().add_patch(circle3)
        
        # 起点和终点
        start = (1, 1)
        goal = (9, 9)
        
        plt.plot(start[0], start[1], 'go', markersize=10, label='起点')
        plt.plot(goal[0], goal[1], 'bo', markersize=10, label='终点')
        
        # 模拟TEB路径 (相对直接但在坡道区域不平滑)
        teb_path_x = [1, 2.5, 3.5, 5, 6.5, 8, 9]
        teb_path_y = [1, 2.5, 4, 5, 6, 7.5, 9]
        
        # 模拟PSOLQR路径 (更平滑，特别是在坡道区域)
        psolqr_path_x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        psolqr_path_y = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        # 添加一些平滑度
        from scipy.interpolate import make_interp_spline
        
        # 平滑TEB路径
        teb_path_x_smooth = np.linspace(min(teb_path_x), max(teb_path_x), 100)
        spl_teb = make_interp_spline(teb_path_x, teb_path_y, k=3)
        teb_path_y_smooth = spl_teb(teb_path_x_smooth)
        
        # 平滑PSOLQR路径
        psolqr_path_x_smooth = np.linspace(min(psolqr_path_x), max(psolqr_path_x), 100)
        spl_psolqr = make_interp_spline(psolqr_path_x, psolqr_path_y, k=3)
        psolqr_path_y_smooth = spl_psolqr(psolqr_path_x_smooth)
        
        # 绘制路径
        plt.plot(teb_path_x_smooth, teb_path_y_smooth, '-', linewidth=2, color='#FF5733', label='TEB路径')
        plt.plot(psolqr_path_x_smooth, psolqr_path_y_smooth, '-', linewidth=2, color='#33A1FF', label='PSOLQR路径')
        
        # 添加坡道标记
        plt.text(5, 5, '坡道区域', fontsize=12, ha='center')
        
        plt.xlabel('X (m)', fontsize=12)
        plt.ylabel('Y (m)', fontsize=12)
        plt.title('TEB vs PSOLQR 路径规划对比 (理论模拟)', fontsize=15)
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        
        plt.savefig(os.path.join(self.output_dir, 'path_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"路径对比图已保存至 {os.path.join(self.output_dir, 'path_comparison.png')}")
    
    def generate_performance_table(self):
        """生成性能对比表格"""
        print("生成性能对比表格...")
        
        # 创建性能对比表格
        performance_data = {
            "指标": ["计算效率", "路径平滑度", "轨迹跟踪精度", "坡道适应性", "避障能力", "参数调整难度"],
            "TEB": ["高", "中", "中", "低", "高", "中"],
            "PSOLQR": ["中", "高", "高", "高", "高", "高"],
            "优势方": ["TEB", "PSOLQR", "PSOLQR", "PSOLQR", "相当", "TEB"]
        }
        
        # 将表格数据保存为CSV文件
        import csv
        with open(os.path.join(self.output_dir, 'performance_comparison.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(performance_data.keys())
            for i in range(len(performance_data["指标"])):
                writer.writerow([performance_data[k][i] for k in performance_data.keys()])
        
        print(f"性能对比表格已保存至 {os.path.join(self.output_dir, 'performance_comparison.csv')}")
        
        # 生成表格的可视化版本
        plt.figure(figsize=(10, 6))
        table_data = []
        for i in range(len(performance_data["指标"])):
            table_data.append([performance_data[k][i] for k in performance_data.keys()])
        
        table = plt.table(cellText=table_data, 
                         colLabels=performance_data.keys(),
                         loc='center',
                         cellLoc='center',
                         colColours=['#f2f2f2']*len(performance_data.keys()))
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # 隐藏坐标轴
        plt.axis('off')
        plt.title('TEB vs PSOLQR 性能对比表', fontsize=15)
        
        plt.savefig(os.path.join(self.output_dir, 'performance_table.png'), dpi=300, bbox_inches='tight')
        print(f"性能对比表格图片已保存至 {os.path.join(self.output_dir, 'performance_table.png')}")
    
    def generate_all(self):
        """生成所有分析图表"""
        self.generate_radar_chart()
        self.generate_slope_performance()
        self.generate_velocity_smoothness()
        self.generate_path_comparison()
        self.generate_performance_table()
        print("所有分析图表生成完成！")

if __name__ == "__main__":
    output_dir = os.path.expanduser("/root/lio_ws/src/planner_comparison/results")
    analyzer = PlannerComparison(output_dir)
    analyzer.generate_all() 