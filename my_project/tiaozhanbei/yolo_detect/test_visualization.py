#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D点云可视化测试脚本
用于测试matplotlib 3D可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test_3d_visualization():
    """测试3D可视化功能"""
    print("开始测试3D可视化功能...")
    
    # 生成测试点云数据
    np.random.seed(42)
    n_points = 1000
    
    # 生成在指定高度范围内的点云
    x = np.random.uniform(-0.1, 0.1, n_points)
    y = np.random.uniform(-0.1, 0.1, n_points)
    z = np.random.uniform(0.074, 0.076, n_points)  # 7.4-7.6cm高度范围
    
    # 添加一些噪声
    x += np.random.normal(0, 0.01, n_points)
    y += np.random.normal(0, 0.01, n_points)
    z += np.random.normal(0, 0.001, n_points)
    
    # 创建点云数组
    pointcloud = np.column_stack([x, y, z])
    
    print(f"生成了 {len(pointcloud)} 个测试点云点")
    print(f"高度范围: {z.min()*100:.2f}cm - {z.max()*100:.2f}cm")
    
    # 创建3D图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 根据高度着色
    colors = z
    
    # 绘制3D散点图
    scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=1, alpha=0.6)
    
    # 设置坐标轴标签
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # 设置标题
    ax.set_title(f'3D点云可视化测试\n点云数量: {len(pointcloud)}\n高度范围: 7.4-7.6cm')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('高度 (m)')
    
    # 设置相等的坐标轴比例
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 显示图形
    plt.tight_layout()
    plt.show()
    
    print("3D可视化测试完成！")

if __name__ == "__main__":
    test_3d_visualization()




