#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云裁剪功能测试脚本
测试crop_pointcloud_world函数的功能
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test_pointcloud_crop():
    """测试点云裁剪功能"""
    print("开始测试点云裁剪功能...")
    
    # 生成测试点云数据
    np.random.seed(42)
    n_points = 10000
    
    # 生成在较大范围内的随机点云
    x = np.random.uniform(-0.5, 1.0, n_points)  # X: -0.5 到 1.0
    y = np.random.uniform(-1.0, 0.5, n_points)  # Y: -1.0 到 0.5  
    z = np.random.uniform(0.05, 0.15, n_points) # Z: 0.05 到 0.15
    
    # 创建点云数组
    pointcloud = np.column_stack([x, y, z])
    
    print(f"生成了 {len(pointcloud)} 个测试点云点")
    print(f"原始点云范围:")
    print(f"  X: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  Y: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Z: [{z.min():.3f}, {z.max():.3f}]")
    
    # 模拟crop_pointcloud_world函数
    def crop_pointcloud_world(pcd_world, x_range=(0, 0.6), y_range=(0, -0.6), z_range=(0.07, 0.08)):
        """模拟点云裁剪函数"""
        if pcd_world is None or len(pcd_world) == 0:
            print("输入点云为空")
            return None, 0, 0
            
        # 记录原始点云数量
        original_count = len(pcd_world)
        
        # 提取坐标
        x = pcd_world[:, 0]
        y = pcd_world[:, 1] 
        z = pcd_world[:, 2]
        
        # 创建裁剪掩码
        x_mask = (x >= x_range[0]) & (x <= x_range[1])
        y_mask = (y >= y_range[1]) & (y <= y_range[0])  # 注意Y轴范围是(0, -0.6)
        z_mask = (z >= z_range[0]) & (z <= z_range[1])
        
        # 组合所有掩码
        combined_mask = x_mask & y_mask & z_mask
        
        # 应用掩码裁剪点云
        cropped_pcd = pcd_world[combined_mask]
        cropped_count = len(cropped_pcd)
        
        # 打印统计信息
        print(f"点云裁剪统计:")
        print(f"  原始点云数量: {original_count}")
        print(f"  裁剪后点云数量: {cropped_count}")
        print(f"  裁剪范围: X[{x_range[0]}, {x_range[1]}], Y[{y_range[1]}, {y_range[0]}], Z[{z_range[0]}, {z_range[1]}]")
        print(f"  保留比例: {cropped_count/original_count*100:.2f}%")
        
        if cropped_count > 0:
            # 打印裁剪后点云的坐标范围
            print(f"  裁剪后点云范围:")
            print(f"    X: [{cropped_pcd[:, 0].min():.4f}, {cropped_pcd[:, 0].max():.4f}]")
            print(f"    Y: [{cropped_pcd[:, 1].min():.4f}, {cropped_pcd[:, 1].max():.4f}]")
            print(f"    Z: [{cropped_pcd[:, 2].min():.4f}, {cropped_pcd[:, 2].max():.4f}]")
        else:
            print("  警告: 裁剪后没有剩余点云")
            
        return cropped_pcd, original_count, cropped_count
    
    # 测试裁剪功能
    cropped_pcd, original_count, cropped_count = crop_pointcloud_world(
        pointcloud,
        x_range=(0, 0.6),
        y_range=(0, -0.6), 
        z_range=(0.07, 0.08)
    )
    
    # 可视化结果
    if cropped_pcd is not None and len(cropped_pcd) > 0:
        # 创建3D图形
        fig = plt.figure(figsize=(15, 5))
        
        # 原始点云
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], 
                   c=pointcloud[:, 2], cmap='viridis', s=1, alpha=0.6)
        ax1.set_title(f'原始点云\n{len(pointcloud)} 个点')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # 裁剪后点云
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(cropped_pcd[:, 0], cropped_pcd[:, 1], cropped_pcd[:, 2], 
                   c=cropped_pcd[:, 2], cmap='viridis', s=1, alpha=0.6)
        ax2.set_title(f'裁剪后点云\n{len(cropped_pcd)} 个点')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        
        # 裁剪范围可视化
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(cropped_pcd[:, 0], cropped_pcd[:, 1], cropped_pcd[:, 2], 
                   c=cropped_pcd[:, 2], cmap='viridis', s=1, alpha=0.6)
        
        # 绘制裁剪边界框
        x_min, x_max = 0, 0.6
        y_min, y_max = -0.6, 0
        z_min, z_max = 0.07, 0.08
        
        # 绘制边界框的8个顶点
        vertices = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max]
        ])
        
        # 绘制边界框的边
        edges = [
            [0,1], [1,2], [2,3], [3,0],  # 底面
            [4,5], [5,6], [6,7], [7,4],  # 顶面
            [0,4], [1,5], [2,6], [3,7]   # 垂直边
        ]
        
        for edge in edges:
            points = vertices[edge]
            ax3.plot3D(points[:, 0], points[:, 1], points[:, 2], 'r-', linewidth=2)
        
        ax3.set_title(f'裁剪范围可视化\nX:[{x_min},{x_max}] Y:[{y_min},{y_max}] Z:[{z_min},{z_max}]')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n测试完成！成功裁剪出 {cropped_count} 个点云点")
    else:
        print("测试失败：裁剪后没有剩余点云")
    
    return cropped_pcd, original_count, cropped_count

if __name__ == "__main__":
    test_pointcloud_crop()




