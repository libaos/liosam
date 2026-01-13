#!/usr/bin/env python3

import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from datetime import datetime

class PresentationGenerator:
    def __init__(self, results_dir, output_dir):
        self.results_dir = results_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def create_title_slide(self, pdf):
        """创建标题页"""
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis('off')
        
        # 标题
        ax.text(0.5, 0.7, 'PSOLQR vs TEB 局部规划器对比', 
                fontsize=24, ha='center', weight='bold')
        
        # 副标题
        ax.text(0.5, 0.6, '在坡道环境中的性能分析', 
                fontsize=18, ha='center')
        
        # 日期
        current_date = datetime.now().strftime("%Y-%m-%d")
        ax.text(0.5, 0.4, f'报告日期: {current_date}', 
                fontsize=14, ha='center')
        
        # 添加页脚
        ax.text(0.5, 0.1, '机器人导航团队', 
                fontsize=12, ha='center')
        
        pdf.savefig(fig)
        plt.close()
    
    def create_algorithm_intro_slide(self, pdf):
        """创建算法介绍页"""
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis('off')
        
        # 标题
        ax.text(0.5, 0.9, '算法介绍', 
                fontsize=20, ha='center', weight='bold')
        
        # TEB介绍
        ax.text(0.05, 0.8, 'TEB (Timed Elastic Band):', 
                fontsize=16, weight='bold')
        
        teb_points = [
            '基于优化的轨迹生成方法',
            '将路径规划问题转化为优化问题',
            '优化目标包括路径长度、平滑度、避障等',
            '使用g2o框架进行非线性优化',
            '广泛用于移动机器人导航'
        ]
        
        for i, point in enumerate(teb_points):
            ax.text(0.1, 0.75 - i*0.05, f'• {point}', fontsize=14)
        
        # PSOLQR介绍
        ax.text(0.05, 0.5, 'PSOLQR (Particle Swarm Optimization + Linear Quadratic Regulator):', 
                fontsize=16, weight='bold')
        
        psolqr_points = [
            'PSO: 粒子群优化算法，用于路径生成',
            'LQR: 线性二次调节器，用于轨迹跟踪控制',
            '结合了全局搜索和精确控制的优点',
            '在复杂环境(如坡道)中表现更优',
            '适应性强，可处理动态变化的环境约束'
        ]
        
        for i, point in enumerate(psolqr_points):
            ax.text(0.1, 0.45 - i*0.05, f'• {point}', fontsize=14)
        
        pdf.savefig(fig)
        plt.close()
    
    def create_slope_challenge_slide(self, pdf):
        """创建坡道挑战页"""
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis('off')
        
        # 标题
        ax.text(0.5, 0.9, '坡道环境的挑战', 
                fontsize=20, ha='center', weight='bold')
        
        # 坡道挑战点
        challenges = [
            '重力影响导致动力学特性变化',
            '轮胎与地面接触特性改变',
            '可能出现滑动或打滑现象',
            '需要更精确的轨迹跟踪控制',
            '传统规划器(如TEB)主要针对平面环境优化',
            '坡度变化会增加路径规划和跟踪的难度'
        ]
        
        for i, challenge in enumerate(challenges):
            ax.text(0.1, 0.8 - i*0.07, f'{i+1}. {challenge}', fontsize=16)
        
        # 添加坡道示意图
        x = np.linspace(0, 10, 100)
        y = 0.2 * x
        
        small_ax = fig.add_axes([0.6, 0.3, 0.35, 0.2])
        small_ax.plot(x, y, 'k-', linewidth=2)
        small_ax.set_xlabel('距离')
        small_ax.set_ylabel('高度')
        small_ax.set_title('坡道示意图')
        small_ax.grid(True)
        
        pdf.savefig(fig)
        plt.close()
    
    def create_performance_comparison_slide(self, pdf):
        """创建性能对比页"""
        # 加载性能对比图表
        radar_path = os.path.join(self.results_dir, 'radar_comparison.png')
        
        if os.path.exists(radar_path):
            # 创建带有雷达图的页面
            fig, ax = plt.subplots(figsize=(11, 8))
            ax.axis('off')
            
            # 标题
            ax.text(0.5, 0.95, 'PSOLQR vs TEB 性能对比', 
                    fontsize=20, ha='center', weight='bold')
            
            # 加载雷达图
            img = plt.imread(radar_path)
            img_ax = fig.add_axes([0.15, 0.2, 0.7, 0.7])
            img_ax.imshow(img)
            img_ax.axis('off')
            
            pdf.savefig(fig)
            plt.close()
        
        # 加载表格图
        table_path = os.path.join(self.results_dir, 'performance_table.png')
        
        if os.path.exists(table_path):
            # 创建带有表格的页面
            fig, ax = plt.subplots(figsize=(11, 8))
            ax.axis('off')
            
            # 标题
            ax.text(0.5, 0.95, '详细性能指标对比', 
                    fontsize=20, ha='center', weight='bold')
            
            # 加载表格图
            img = plt.imread(table_path)
            img_ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])
            img_ax.imshow(img)
            img_ax.axis('off')
            
            pdf.savefig(fig)
            plt.close()
    
    def create_slope_performance_slide(self, pdf):
        """创建坡度性能页"""
        # 加载坡度性能图
        slope_path = os.path.join(self.results_dir, 'slope_error_comparison.png')
        
        if os.path.exists(slope_path):
            # 创建带有坡度性能图的页面
            fig, ax = plt.subplots(figsize=(11, 8))
            ax.axis('off')
            
            # 标题
            ax.text(0.5, 0.95, '坡道环境中的轨迹跟踪精度', 
                    fontsize=20, ha='center', weight='bold')
            
            # 说明文字
            ax.text(0.1, 0.85, '随着坡度增加，PSOLQR保持更低的轨迹跟踪误差', fontsize=16)
            
            # 加载坡度性能图
            img = plt.imread(slope_path)
            img_ax = fig.add_axes([0.1, 0.2, 0.8, 0.6])
            img_ax.imshow(img)
            img_ax.axis('off')
            
            pdf.savefig(fig)
            plt.close()
    
    def create_velocity_smoothness_slide(self, pdf):
        """创建速度平滑度页"""
        # 加载速度平滑度图
        velocity_path = os.path.join(self.results_dir, 'velocity_smoothness.png')
        
        if os.path.exists(velocity_path):
            # 创建带有速度平滑度图的页面
            fig, ax = plt.subplots(figsize=(11, 8))
            ax.axis('off')
            
            # 标题
            ax.text(0.5, 0.95, '坡道环境中的速度平滑度', 
                    fontsize=20, ha='center', weight='bold')
            
            # 说明文字
            ax.text(0.1, 0.85, 'PSOLQR在坡道上保持更平滑的速度曲线，减少机械应力和能量消耗', fontsize=16)
            
            # 加载速度平滑度图
            img = plt.imread(velocity_path)
            img_ax = fig.add_axes([0.1, 0.2, 0.8, 0.6])
            img_ax.imshow(img)
            img_ax.axis('off')
            
            pdf.savefig(fig)
            plt.close()
    
    def create_path_comparison_slide(self, pdf):
        """创建路径对比页"""
        # 加载路径对比图
        path_comp_path = os.path.join(self.results_dir, 'path_comparison.png')
        
        if os.path.exists(path_comp_path):
            # 创建带有路径对比图的页面
            fig, ax = plt.subplots(figsize=(11, 8))
            ax.axis('off')
            
            # 标题
            ax.text(0.5, 0.95, '路径规划对比', 
                    fontsize=20, ha='center', weight='bold')
            
            # 说明文字
            ax.text(0.1, 0.85, 'PSOLQR生成更平滑的路径，特别是在坡道区域', fontsize=16)
            
            # 加载路径对比图
            img = plt.imread(path_comp_path)
            img_ax = fig.add_axes([0.1, 0.2, 0.8, 0.6])
            img_ax.imshow(img)
            img_ax.axis('off')
            
            pdf.savefig(fig)
            plt.close()
    
    def create_conclusion_slide(self, pdf):
        """创建结论页"""
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis('off')
        
        # 标题
        ax.text(0.5, 0.9, '结论：为什么选择PSOLQR？', 
                fontsize=20, ha='center', weight='bold')
        
        # 结论要点
        conclusions = [
            '在坡道环境中表现更优',
            '提供更平滑的导航体验',
            '更精确的轨迹跟踪',
            '适应复杂地形的能力更强',
            '特别适合山地或多层建筑等非平坦环境'
        ]
        
        for i, conclusion in enumerate(conclusions):
            ax.text(0.1, 0.8 - i*0.08, f'✓ {conclusion}', fontsize=18)
        
        # 添加总结
        ax.text(0.1, 0.3, '总结：PSOLQR结合了PSO的全局搜索能力和LQR的精确控制，', fontsize=16)
        ax.text(0.1, 0.25, '在坡道等复杂环境中能提供更稳定、更平滑的导航性能。', fontsize=16)
        
        pdf.savefig(fig)
        plt.close()
    
    def generate_presentation(self):
        """生成完整的演示文档"""
        pdf_path = os.path.join(self.output_dir, 'psolqr_presentation.pdf')
        
        with PdfPages(pdf_path) as pdf:
            self.create_title_slide(pdf)
            self.create_algorithm_intro_slide(pdf)
            self.create_slope_challenge_slide(pdf)
            self.create_performance_comparison_slide(pdf)
            self.create_slope_performance_slide(pdf)
            self.create_velocity_smoothness_slide(pdf)
            self.create_path_comparison_slide(pdf)
            self.create_conclusion_slide(pdf)
        
        print(f"演示文档已生成：{pdf_path}")

if __name__ == "__main__":
    from pathlib import Path

    tool_root = Path(__file__).resolve().parents[1]
    results_dir = tool_root / "results"
    output_dir = tool_root / "presentation"

    generator = PresentationGenerator(str(results_dir), str(output_dir))
    generator.generate_presentation()
